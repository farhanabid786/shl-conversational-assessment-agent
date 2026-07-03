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


# --------------------------------------------------------------------------- #
# Output Dataclass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PipelineResult:
    """Immutable result of a single pipeline run.

    Attributes
    ----------
    response:
        The natural-language response text returned by Gemini.
    recommendations:
        The list of resolved assessment objects backing the response
        (FilteredCandidate objects for a recommendation path, ComparisonItem
        objects for a comparison path, or an empty list for any other
        path). Never mutated or reranked by this module.
    metadata:
        The metadata dictionary produced by the Prompt Builder, describing
        intent, routing path, and counts for the turn.
    """

    response: str
    recommendations: list
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
    GeminiClientError,
    GeminiConfigurationError,
    GeminiGenerationError,
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
    ) -> None:
        """Store shared dependencies required for every pipeline run.

        Args:
            resources: Loaded retriever artifacts (metadata, mapping,
                FAISS index, BM25 index, entity_ids).
            gemini_client: Shared GeminiClient used to generate responses.

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

        self._resources = resources
        self._gemini_client = gemini_client
        self._entity_lookup: dict[str, dict[str, Any]] = self._build_entity_lookup(
            resources.metadata
        )

        logger.info(
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
            PipelineResult containing the generated response text, the
            backing recommendation/comparison objects (if any), and the
            routing metadata produced by the Prompt Builder.

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
                logger.info("Pipeline routing: clarification required.")

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
                logger.info("Pipeline routing: proceeding to retrieval.")

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

            logger.info(
                "Pipeline run complete: intent=%s path=%s recommendation_count=%d",
                intent.value,
                prompt_payload.metadata.get("path"),
                len(recommendations_out),
            )

            return PipelineResult(
                response=response_text,
                recommendations=recommendations_out,
                metadata=dict(prompt_payload.metadata),
            )

        except ValidationError:
            raise
        except _UPSTREAM_ERRORS as exc:
            logger.exception("Pipeline run failed in an upstream module.")
            raise PipelineError(f"Pipeline run failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline run failed unexpectedly.")
            raise PipelineError(f"Pipeline run failed unexpectedly: {exc}") from exc
