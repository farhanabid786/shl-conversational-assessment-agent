"""
app/pipeline.py

Phase 6: Pipeline Orchestrator
SHL Conversational Assessment Recommendation System

Coordinates the already-implemented Phase 4, Phase 5, and Phase 6 modules
into a single conversational turn. This module contains NO retrieval logic,
NO fusion logic, NO metadata filtering logic, NO recommendation decision
logic, NO comparison logic, NO refusal logic, NO prompt-assembly logic, and
NO Gemini invocation logic. Every one of those responsibilities is
delegated to the existing module that already owns it; this file only
sequences calls and adapts data between module boundaries.

Execution order
----------------
Intent Detector
    -> Conversation State
    -> Clarification Engine
    -> IF clarification required:
           Prompt Builder -> Gemini -> Return
       ELSE:
           Hybrid Retriever -> Fusion -> Metadata Filter
               -> Recommendation Engine -> Comparison Engine
               -> Refusal Engine -> Prompt Builder -> Gemini -> Return

Python 3.10.11
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from scripts.intent_detector import (
    ConversationIntent,
    IntentDetectionError,
    IntentResult,
    ValidationError as IntentValidationError,
    detect_intent,
)
from scripts.conversation_state import (
    ConversationState,
    ConversationStateError,
    ValidationError as ConversationStateValidationError,
    parse_conversation,
)
from scripts.clarification_engine import (
    ClarificationDecision,
    ClarificationEngineError,
    ValidationError as ClarificationValidationError,
    evaluate_clarification,
)
from scripts.hybrid_retriever import (
    EncodingError as HybridRetrieverEncodingError,
    HybridCandidates as RetrieverHybridCandidates,
    HybridRetrieverError,
    ModelNotLoadedError as HybridRetrieverModelNotLoadedError,
    RetrievalError as HybridRetrieverRetrievalError,
    ValidationError as HybridRetrieverValidationError,
    retrieve as hybrid_retrieve,
)
from scripts.fusion import (
    FusionError,
    FusionResult,
    HybridCandidates as FusionHybridCandidates,
    LexicalCandidate as FusionLexicalCandidate,
    SemanticCandidate as FusionSemanticCandidate,
    ValidationError as FusionValidationError,
    fuse,
)
from scripts.metadata_filter import (
    FilterCriteria,
    FilterResult,
    MetadataFilterError,
    ValidationError as MetadataFilterValidationError,
    filter_candidates,
)
from scripts.recommendation_engine import (
    RecommendationDecision,
    RecommendationEngineError,
    ValidationError as RecommendationValidationError,
    decide_recommendation,
)
from scripts.comparison_engine import (
    ComparisonDecision,
    ComparisonEngineError,
    ValidationError as ComparisonValidationError,
    build_comparison,
)
from scripts.refusal_engine import (
    RefusalDecision,
    RefusalEngineError,
    ValidationError as RefusalValidationError,
    evaluate_refusal,
)
from scripts.prompt_builder import (
    PromptBuilderError,
    PromptPayload,
    ValidationError as PromptBuilderValidationError,
    build_prompt,
)
from scripts.retriever_loader import RetrieverResources

from app.gemini_client import (
    GeminiClient,
    GeminiClientError,
    GeminiConfigurationError,
    GeminiGenerationError,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #


class PipelineError(Exception):
    """Base exception for all pipeline orchestration failures."""


class ValidationError(PipelineError):
    """Raised when pipeline input or dependency validation fails."""


class ServiceUnavailableError(PipelineError):
    """Raised when a required dependency is temporarily unusable.

    Covers cases such as the SentenceTransformer model failing to load
    (e.g. no network access to download it) or Gemini being unreachable.
    Distinct from ValidationError (bad caller input) and the generic
    PipelineError (unexpected internal failure) so the HTTP layer can map
    it to 503 Service Unavailable rather than 400/500.
    """


# --------------------------------------------------------------------------- #
# Output Dataclass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PipelineResult:
    """Immutable result of a single pipeline run.

    Field names deliberately mirror the assignment's required API
    contract (POST /chat -> {"reply", "recommendations", "end_of_conversation"})
    so app.routes can build the ChatResponse with a straight field copy —
    no renaming or reshaping left to the HTTP layer.

    Attributes
    ----------
    reply:
        The natural-language reply text returned by Gemini.
    recommendations:
        The list of resolved assessment objects backing the reply
        (FilteredCandidate objects for a recommendation path, ComparisonItem
        objects for a comparison path, or an empty list for any other
        path). Never mutated or reranked by this module. Each object
        exposes `.canonical_name` and `.metadata` (a dict expected to
        carry 'url' and 'test_type' once catalog_metadata.json has been
        regenerated with those fields — see scripts.metadata_generator).
    end_of_conversation:
        True only when the agent considers the task complete: a
        recommendation or comparison shortlist was actually delivered, or
        the conversation has reached the evaluator's turn cap. False
        while still clarifying, refusing a single off-topic turn, or
        falling back to ask for more context — the user may reasonably
        continue the conversation in those cases.
    metadata:
        The metadata dictionary produced by the Prompt Builder, describing
        intent, routing path, and counts for the turn.
    """

    reply: str
    recommendations: list
    end_of_conversation: bool
    metadata: dict


# --------------------------------------------------------------------------- #
# Exception groups used for wrapping (no new logic — pass-through mapping)
# --------------------------------------------------------------------------- #

_UPSTREAM_ERRORS: tuple[type[Exception], ...] = (
    IntentDetectionError,
    IntentValidationError,
    ConversationStateError,
    ConversationStateValidationError,
    ClarificationEngineError,
    ClarificationValidationError,
    HybridRetrieverError,
    HybridRetrieverValidationError,
    HybridRetrieverEncodingError,
    HybridRetrieverRetrievalError,
    FusionError,
    FusionValidationError,
    MetadataFilterError,
    MetadataFilterValidationError,
    RecommendationEngineError,
    RecommendationValidationError,
    ComparisonEngineError,
    ComparisonValidationError,
    RefusalEngineError,
    RefusalValidationError,
    PromptBuilderError,
    PromptBuilderValidationError,
    # NOTE: GeminiClientError/GeminiConfigurationError/GeminiGenerationError
    # and HybridRetrieverModelNotLoadedError are intentionally NOT included
    # here. They represent dependency-availability failures (Gemini
    # unreachable, embedding model missing at query time — which, since
    # loading is eager at startup, indicates a startup/wiring bug) rather
    # than upstream logic errors, so Pipeline.run() catches them separately
    # and raises ServiceUnavailableError for the HTTP layer to map to 503
    # instead of 500.
)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class Pipeline:
    """Orchestrates one conversational turn across the existing modules.

    This class holds no retrieval, fusion, filtering, recommendation,
    comparison, refusal, prompt-generation, or Gemini logic itself. It only
    holds the shared, process-wide dependencies (retriever resources and
    the Gemini client) and sequences calls into the modules that already
    implement each responsibility.

    Note on Hybrid Retriever instantiation
    ---------------------------------------
    ``scripts.hybrid_retriever`` exposes a stateless, module-level
    ``retrieve()`` function — it defines no ``HybridRetriever`` class to
    instantiate. There is therefore nothing to construct once and reuse for
    that stage; ``retrieve()`` is called directly with the shared
    ``RetrieverResources`` on each turn that requires retrieval.
    """

    def __init__(
        self,
        resources: RetrieverResources,
        gemini_client: GeminiClient,
        max_conversation_turns: int = 8,
    ) -> None:
        """Store shared dependencies required for every pipeline run.

        Args:
            resources: Loaded retriever artifacts (metadata, mapping,
                FAISS index, BM25 index, entity_ids), with the embedding
                model already loaded eagerly at startup.
            gemini_client: Shared GeminiClient used to generate responses.
            max_conversation_turns: Mirrors the evaluator's hard turn cap
                (user + assistant messages combined). Once the incoming
                history reaches this length, the pipeline reports
                end_of_conversation=True regardless of routing outcome,
                since the evaluator will not extend the conversation
                further. Defaults to 8 to match the evaluator; pass
                settings.MAX_CONVERSATION_TURNS from app.config for a
                single source of truth.

        Raises:
            ValidationError: If ``resources`` or ``gemini_client`` is
                missing or of an invalid type.
        """
        if resources is None or not isinstance(resources, RetrieverResources):
            raise ValidationError(
                f"resources must be a RetrieverResources instance, got "
                f"{type(resources).__name__}."
            )
        if gemini_client is None or not isinstance(gemini_client, GeminiClient):
            raise ValidationError(
                f"gemini_client must be a GeminiClient instance, got "
                f"{type(gemini_client).__name__}."
            )
        if not isinstance(max_conversation_turns, int) or max_conversation_turns < 1:
            raise ValidationError(
                "max_conversation_turns must be a positive integer, got "
                f"{max_conversation_turns!r}."
            )

        self._resources = resources
        self._gemini_client = gemini_client
        self._max_conversation_turns = max_conversation_turns

        # Pipeline is constructed fresh on every HTTP request (see
        # app.routes._build_pipeline), so entity_lookup must NOT be
        # rebuilt here — resources.entity_lookup is already computed
        # exactly once, at startup, by retriever_loader.load_retriever_resources.
        # Rebuilding an identical dict from resources.metadata on every
        # request would be a pure repeated computation with no benefit.
        # The local build is retained only as a defensive fallback for
        # lightweight/stub resources (e.g. in unit tests) that don't carry
        # a pre-built entity_lookup.
        existing_lookup = getattr(resources, "entity_lookup", None)
        if isinstance(existing_lookup, dict) and existing_lookup:
            self._entity_lookup: dict[str, dict[str, Any]] = existing_lookup
        else:
            self._entity_lookup = self._build_entity_lookup(resources.metadata)

        logger.debug(
            "Pipeline initialized: %d entity_lookup records.",
            len(self._entity_lookup),
        )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_entity_lookup(
        metadata: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Build the entity_id -> metadata dict expected by downstream
        modules (metadata_filter.filter_candidates, comparison_engine.
        build_comparison) from RetrieverResources.metadata.
        """
        lookup: dict[str, dict[str, Any]] = {}
        for record in metadata:
            entity_id = str(record.get("entity_id", "")).strip()
            if entity_id:
                lookup[entity_id] = record
        return lookup

    @staticmethod
    def _to_fusion_candidates(
        hybrid: RetrieverHybridCandidates,
    ) -> FusionHybridCandidates:
        """Adapt scripts.hybrid_retriever.HybridCandidates into the
        structurally-identical but distinct scripts.fusion.HybridCandidates
        type expected by scripts.fusion.fuse(). Field values are copied
        verbatim; no scores, ranks, or ids are computed or altered here.
        """
        semantic = [
            FusionSemanticCandidate(
                entity_id=c.entity_id,
                canonical_name=c.canonical_name,
                score=c.score,
                rank=c.rank,
            )
            for c in hybrid.semantic
        ]
        lexical = [
            FusionLexicalCandidate(
                entity_id=c.entity_id,
                canonical_name=c.canonical_name,
                score=c.score,
                rank=c.rank,
            )
            for c in hybrid.lexical
        ]
        return FusionHybridCandidates(
            semantic=semantic,
            lexical=lexical,
            merged_ids=list(hybrid.merged_ids),
        )

    @staticmethod
    def _build_filter_criteria(state: ConversationState) -> FilterCriteria:
        """Map already-extracted ConversationState fields onto
        metadata_filter.FilterCriteria. No new extraction or decisioning
        is performed here — values are copied verbatim from state.
        """
        return FilterCriteria(
            adaptive=state.adaptive,
            remote=state.remote,
            max_duration=state.max_duration,
            languages=state.languages,
            job_levels=state.job_levels,
            assessment_family=state.assessment_family,
            keywords=state.skills,
        )

    @staticmethod
    def _select_recommendations(
        path: str,
        recommendation: RecommendationDecision,
        comparison: ComparisonDecision,
    ) -> list:
        """Select which already-produced decision list backs the response,
        based on the path chosen by prompt_builder. Performs no reranking,
        filtering, or mutation.
        """
        if path == "recommendation" and recommendation.ready:
            return recommendation.recommendations
        if path == "comparison" and comparison.ready:
            return comparison.comparisons
        return []

    def _compute_end_of_conversation(
        self,
        path: str,
        recommendation: RecommendationDecision,
        comparison: ComparisonDecision,
        validated_messages: list[dict[str, str]],
    ) -> bool:
        """Decide whether the agent considers this turn's task complete.

        True when:
          * a recommendation shortlist was actually delivered (path ==
            "recommendation" and recommendation.ready), or
          * a comparison was actually delivered (path == "comparison" and
            comparison.ready), or
          * the conversation has reached the evaluator's turn cap, so
            there is no further turn in which to continue regardless.

        False while clarifying, refusing a single off-topic ask, or
        falling back to request more context — the user may reasonably
        continue the conversation in those cases, so the task is not yet
        "done".
        """
        delivered_shortlist = (
            path == "recommendation" and recommendation.ready
        ) or (
            path == "comparison" and comparison.ready
        )
        turn_cap_reached = len(validated_messages) >= self._max_conversation_turns
        return bool(delivered_shortlist or turn_cap_reached)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_messages(messages: Any) -> list[dict[str, str]]:
        if not isinstance(messages, list) or not messages:
            raise ValidationError("messages must be a non-empty list.")
        return messages

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, messages: list[dict[str, str]]) -> PipelineResult:
        """Run one full conversational turn through the pipeline.

        Args:
            messages: Full conversation history. Each item must be a dict
                with 'role' and 'content' string keys; the final message
                must be from the 'user' role (enforced by the Intent
                Detector and Conversation State parser).

        Returns:
            PipelineResult containing the generated reply text, the
            backing recommendation/comparison objects (if any), the
            end_of_conversation flag, and the routing metadata produced
            by the Prompt Builder.

        Raises:
            ValidationError: If ``messages`` is empty or malformed at the
                pipeline boundary.
            PipelineError: If any downstream module raises during
                orchestration.
        """
        validated_messages = self._validate_messages(messages)

        try:
            # ---------------------------------------------------------- #
            # Intent Detector
            # ---------------------------------------------------------- #
            intent_result: IntentResult = detect_intent(validated_messages)
            intent: ConversationIntent = intent_result.intent

            # ---------------------------------------------------------- #
            # Conversation State
            # ---------------------------------------------------------- #
            state: ConversationState = parse_conversation(validated_messages)

            # ---------------------------------------------------------- #
            # Clarification Engine
            # ---------------------------------------------------------- #
            clarification: ClarificationDecision = evaluate_clarification(
                intent, state
            )

            if clarification.needs_clarification:
                logger.debug("Pipeline routing: clarification required.")

                # No retrieval has run yet, so recommendation_engine is
                # given an empty FilterResult. decide_recommendation()
                # itself short-circuits on clarification.needs_clarification
                # before inspecting filter_result.candidates.
                empty_filter_result = FilterResult(candidates=[])
                recommendation: RecommendationDecision = decide_recommendation(
                    intent, state, clarification, empty_filter_result
                )
                comparison: ComparisonDecision = build_comparison(
                    intent, state, self._entity_lookup
                )
                refusal: RefusalDecision = evaluate_refusal(
                    intent, validated_messages
                )
            else:
                logger.debug("Pipeline routing: proceeding to retrieval.")

                # ------------------------------------------------------ #
                # Hybrid Retriever
                # ------------------------------------------------------ #
                query = str(validated_messages[-1]["content"])
                hybrid_candidates: RetrieverHybridCandidates = hybrid_retrieve(
                    query, self._resources
                )

                # ------------------------------------------------------ #
                # Fusion
                # ------------------------------------------------------ #
                fusion_input = self._to_fusion_candidates(hybrid_candidates)
                fusion_result: FusionResult = fuse(fusion_input)

                # ------------------------------------------------------ #
                # Metadata Filter
                # ------------------------------------------------------ #
                criteria = self._build_filter_criteria(state)
                filter_result: FilterResult = filter_candidates(
                    fusion_result, self._entity_lookup, criteria
                )

                # ------------------------------------------------------ #
                # Recommendation Engine
                # ------------------------------------------------------ #
                recommendation = decide_recommendation(
                    intent, state, clarification, filter_result
                )

                # ------------------------------------------------------ #
                # Comparison Engine
                # ------------------------------------------------------ #
                comparison = build_comparison(intent, state, self._entity_lookup)

                # ------------------------------------------------------ #
                # Refusal Engine
                # ------------------------------------------------------ #
                refusal = evaluate_refusal(intent, validated_messages)

            # ---------------------------------------------------------- #
            # Prompt Builder
            # ---------------------------------------------------------- #
            prompt_payload: PromptPayload = build_prompt(
                intent,
                state,
                clarification,
                recommendation,
                comparison,
                refusal,
                validated_messages,
            )

            # ---------------------------------------------------------- #
            # Gemini
            # ---------------------------------------------------------- #
            response_text: str = self._gemini_client.generate(prompt_payload)

            recommendations_out = self._select_recommendations(
                str(prompt_payload.metadata.get("path", "")),
                recommendation,
                comparison,
            )

            end_of_conversation = self._compute_end_of_conversation(
                str(prompt_payload.metadata.get("path", "")),
                recommendation,
                comparison,
                validated_messages,
            )

            logger.info(
                "Pipeline run complete: intent=%s path=%s recommendation_count=%d "
                "end_of_conversation=%s",
                intent.value,
                prompt_payload.metadata.get("path"),
                len(recommendations_out),
                end_of_conversation,
            )

            return PipelineResult(
                reply=response_text,
                recommendations=recommendations_out,
                end_of_conversation=end_of_conversation,
                metadata=dict(prompt_payload.metadata),
            )

        except ValidationError:
            raise
        except HybridRetrieverModelNotLoadedError as exc:
            # Should be unreachable in normal operation — the embedding
            # model is loaded eagerly at startup, before the process is
            # ever considered ready. If this fires, it is a startup/wiring
            # bug, not a transient condition, but it is still surfaced as
            # a 503 rather than a 500 since the fix ("restart with a
            # correct startup sequence") is operational, not a code defect
            # exercised by user input. `from exc` preserves the traceback.
            logger.exception(
                "Embedding model unexpectedly absent at query time "
                "(expected eager load at startup)."
            )
            raise ServiceUnavailableError(
                f"Embedding model is unavailable: {exc}"
            ) from exc
        except GeminiClientError as exc:
            # Covers Gemini configuration failures, generation failures,
            # and timeouts — all surfaced as 503 by the HTTP layer.
            logger.exception("Gemini service unavailable.")
            raise ServiceUnavailableError(
                f"Gemini service is temporarily unavailable: {exc}"
            ) from exc
        except _UPSTREAM_ERRORS as exc:
            logger.exception("Pipeline run failed in an upstream module.")
            raise PipelineError(f"Pipeline run failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            # Unexpected, unclassified failure. logger.exception() records
            # the full traceback in the server logs; `from exc` preserves
            # __cause__/__traceback__ on the raised PipelineError so the
            # original stack remains inspectable by anything upstream that
            # catches it before it reaches the HTTP layer.
            logger.exception("Pipeline run failed unexpectedly.")
            raise PipelineError(f"Pipeline run failed unexpectedly: {exc}") from exc
