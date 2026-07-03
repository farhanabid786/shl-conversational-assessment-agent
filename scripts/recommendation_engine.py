"""
Recommendation Engine
======================

Phase 5 module for the SHL Conversational Assessment Recommendation System.

This module is the orchestration layer between the retrieval pipeline and
the response generation layer. It decides whether recommendations can be
returned. It performs ONLY deterministic decision logic over already-produced
inputs. It NEVER calls Gemini, performs SentenceTransformer encoding, FAISS
retrieval, BM25 retrieval, fusion, metadata filtering, or prompt generation.

Python: 3.10.11
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from scripts.clarification_engine import ClarificationDecision
from scripts.conversation_state import ConversationState
from scripts.intent_detector import ConversationIntent
from scripts.metadata_filter import FilteredCandidate, FilterResult

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS
# ==============================================================================


class RecommendationEngineError(Exception):
    """Raised when recommendation decisioning fails due to an internal error."""


class ValidationError(RecommendationEngineError):
    """Raised when an input to the recommendation engine is invalid."""


# ==============================================================================
# OUTPUT DATA STRUCTURE
# ==============================================================================


@dataclass(frozen=True)
class RecommendationDecision:
    """Result of the recommendation decisioning process."""

    ready: bool
    recommendations: list[FilteredCandidate]
    reason: str
    confidence: float


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Intents for which recommendation decisioning is applicable.
_SUPPORTED_INTENTS: tuple[ConversationIntent, ...] = (
    ConversationIntent.RECOMMEND,
    ConversationIntent.REFINE,
    ConversationIntent.CLARIFY,
)

# Reasons.
_REASON_INTENT_NOT_APPLICABLE: str = "Recommendation not applicable."
_REASON_CLARIFICATION_REQUIRED: str = "Additional clarification required."
_REASON_NO_CANDIDATES: str = "No matching SHL assessments found."
_REASON_RECOMMENDATIONS_READY: str = "Recommendations ready."

# Confidence values.
_CONFIDENCE_INTENT_NOT_APPLICABLE: float = 0.99
_CONFIDENCE_CLARIFICATION_REQUIRED: float = 0.99
_CONFIDENCE_NO_CANDIDATES: float = 0.95
_CONFIDENCE_RECOMMENDATIONS_READY: float = 1.00


# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================


def _validate_intent(intent: ConversationIntent) -> None:
    if intent is None:
        raise ValidationError("intent must not be None.")
    if not isinstance(intent, ConversationIntent):
        raise ValidationError(f"intent must be a ConversationIntent, got {type(intent)!r}.")


def _validate_state(state: ConversationState) -> None:
    if state is None:
        raise ValidationError("state must not be None.")
    if not isinstance(state, ConversationState):
        raise ValidationError(f"state must be a ConversationState, got {type(state)!r}.")


def _validate_clarification(clarification: ClarificationDecision) -> None:
    if clarification is None:
        raise ValidationError("clarification must not be None.")
    if not isinstance(clarification, ClarificationDecision):
        raise ValidationError(
            f"clarification must be a ClarificationDecision, got {type(clarification)!r}."
        )


def _validate_filter_result(filter_result: FilterResult) -> None:
    if filter_result is None:
        raise ValidationError("filter_result must not be None.")
    if not isinstance(filter_result, FilterResult):
        raise ValidationError(
            f"filter_result must be a FilterResult, got {type(filter_result)!r}."
        )
    if not isinstance(filter_result.candidates, list):
        raise ValidationError("filter_result.candidates must be a list.")
    for idx, candidate in enumerate(filter_result.candidates):
        if not isinstance(candidate, FilteredCandidate):
            raise ValidationError(
                f"Candidate at index {idx} is not a FilteredCandidate, got {type(candidate)!r}."
            )


def _validate_inputs(
    intent: ConversationIntent,
    state: ConversationState,
    clarification: ClarificationDecision,
    filter_result: FilterResult,
) -> None:
    _validate_intent(intent)
    _validate_state(state)
    _validate_clarification(clarification)
    _validate_filter_result(filter_result)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _is_supported_intent(intent: ConversationIntent) -> bool:
    return intent in _SUPPORTED_INTENTS


def _requires_clarification(clarification: ClarificationDecision) -> bool:
    return clarification.needs_clarification


def _has_candidates(filter_result: FilterResult) -> bool:
    return len(filter_result.candidates) > 0


# ==============================================================================
# DECISION HELPERS (single responsibility, one per outcome)
# ==============================================================================


def _intent_not_applicable_decision() -> RecommendationDecision:
    return RecommendationDecision(
        ready=False,
        recommendations=[],
        reason=_REASON_INTENT_NOT_APPLICABLE,
        confidence=_CONFIDENCE_INTENT_NOT_APPLICABLE,
    )


def _clarification_required_decision() -> RecommendationDecision:
    return RecommendationDecision(
        ready=False,
        recommendations=[],
        reason=_REASON_CLARIFICATION_REQUIRED,
        confidence=_CONFIDENCE_CLARIFICATION_REQUIRED,
    )


def _no_candidates_decision() -> RecommendationDecision:
    return RecommendationDecision(
        ready=False,
        recommendations=[],
        reason=_REASON_NO_CANDIDATES,
        confidence=_CONFIDENCE_NO_CANDIDATES,
    )


def _recommendations_ready_decision(filter_result: FilterResult) -> RecommendationDecision:
    return RecommendationDecision(
        ready=True,
        recommendations=filter_result.candidates,
        reason=_REASON_RECOMMENDATIONS_READY,
        confidence=_CONFIDENCE_RECOMMENDATIONS_READY,
    )


# ==============================================================================
# DECISION LOGIC
# ==============================================================================


def _decide(
    intent: ConversationIntent,
    clarification: ClarificationDecision,
    filter_result: FilterResult,
) -> RecommendationDecision:
    if not _is_supported_intent(intent):
        logger.debug("Intent %s is not supported for recommendation.", intent)
        return _intent_not_applicable_decision()

    if _requires_clarification(clarification):
        logger.debug("Clarification required; skipping filter result inspection.")
        return _clarification_required_decision()

    if not _has_candidates(filter_result):
        logger.debug("FilterResult contains zero candidates.")
        return _no_candidates_decision()

    logger.debug(
        "Recommendations ready: %d candidate(s), original order preserved.",
        len(filter_result.candidates),
    )
    return _recommendations_ready_decision(filter_result)


# ==============================================================================
# PUBLIC API
# ==============================================================================


def decide_recommendation(
    intent: ConversationIntent,
    state: ConversationState,
    clarification: ClarificationDecision,
    filter_result: FilterResult,
) -> RecommendationDecision:
    """Determine whether recommendations can be returned.

    This function never generates recommendations. It only decides, based on
    deterministic rules, whether the already-filtered SHL assessment
    candidates should be surfaced to the user. Candidate order, scores, and
    metadata are never modified, reranked, sorted, or deduplicated; they are
    assumed to already be correctly ordered by the retrieval pipeline.

    Args:
        intent: The classified conversation intent.
        state: The parsed structured conversation state.
        clarification: The clarification engine's decision for this turn.
        filter_result: The already metadata-filtered candidate set.

    Returns:
        RecommendationDecision describing whether recommendations are ready,
        the candidates (if any), a human-readable reason, and a confidence
        score.

    Raises:
        ValidationError: if any input is None or of an invalid type.
        RecommendationEngineError: if decisioning fails unexpectedly.
    """
    try:
        _validate_inputs(intent, state, clarification, filter_result)
    except ValidationError:
        raise
    except Exception as exc:  # defensive: unexpected validation failure
        raise ValidationError(f"Failed to validate inputs: {exc}") from exc

    try:
        decision = _decide(intent, clarification, filter_result)
        logger.info(
            "Recommendation decision: ready=%s reason=%r confidence=%.2f candidate_count=%d",
            decision.ready,
            decision.reason,
            decision.confidence,
            len(decision.recommendations),
        )
        return decision
    except (ValidationError, RecommendationEngineError):
        raise
    except Exception as exc:  # defensive: unexpected decisioning failure
        raise RecommendationEngineError(f"Recommendation decisioning failed: {exc}") from exc
