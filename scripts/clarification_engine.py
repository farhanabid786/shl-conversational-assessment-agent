"""
Clarification Engine
=====================

Phase 5 module for the SHL Conversational Assessment Recommendation System.

This module decides whether enough structured hiring information exists to
proceed with recommendation. It performs ONLY deterministic decision logic
over already-extracted conversation state. It NEVER calls Gemini, performs
retrieval, FAISS, BM25, fusion, metadata filtering, recommendation
generation, comparison generation, or prompt generation.

Python: 3.10.11
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

# from conversation_state import ConversationState
from scripts.conversation_state import ConversationState
from scripts.intent_detector import ConversationIntent

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS
# ==============================================================================


class ClarificationEngineError(Exception):
    """Raised when clarification decisioning fails due to an internal error."""


class ValidationError(ClarificationEngineError):
    """Raised when the input intent or state is invalid."""


# ==============================================================================
# OUTPUT DATA STRUCTURE
# ==============================================================================


@dataclass(frozen=True)
class ClarificationDecision:
    """Result of the clarification decisioning process."""

    needs_clarification: bool
    question: str | None
    reason: str
    missing_fields: list[str]
    confidence: float


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Intents that are evaluated for clarification.
_EVALUATED_INTENTS: tuple[ConversationIntent, ...] = (
    ConversationIntent.RECOMMEND,
    ConversationIntent.CLARIFY,
    ConversationIntent.REFINE,
)

# Intents for which clarification is always skipped.
_SKIPPED_INTENTS: tuple[ConversationIntent, ...] = (
    ConversationIntent.COMPARE,
    ConversationIntent.REFUSE,
    ConversationIntent.UNKNOWN,
)

# Missing field names.
_FIELD_ROLE: str = "role"
_FIELD_SKILLS: str = "skills"

# Priority 1: Role missing.
_QUESTION_MISSING_ROLE: str = "What role are you hiring for?"
_REASON_MISSING_ROLE: str = "Target role missing."
_CONFIDENCE_MISSING_ROLE: float = 0.99

# Priority 2: Role exists, both skills and assessment family missing.
_QUESTION_MISSING_SKILLS_OR_FAMILY: str = (
    "What skills or competencies would you like to assess?"
)
_REASON_MISSING_SKILLS_OR_FAMILY: str = "Skills or assessment type missing."
_CONFIDENCE_MISSING_SKILLS_OR_FAMILY: float = 0.95

# No clarification needed.
_REASON_SUFFICIENT_INFO: str = "Enough hiring information available."
_CONFIDENCE_SUFFICIENT_INFO: float = 1.00

# Reason used when the intent is skipped entirely.
_REASON_INTENT_SKIPPED: str = "Clarification is not applicable for this intent."
_CONFIDENCE_INTENT_SKIPPED: float = 1.00


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


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _is_evaluated_intent(intent: ConversationIntent) -> bool:
    return intent in _EVALUATED_INTENTS


def _has_role(state: ConversationState) -> bool:
    return bool(state.target_roles)


def _has_skills(state: ConversationState) -> bool:
    return bool(state.skills)


def _has_assessment_family(state: ConversationState) -> bool:
    return bool(state.assessment_family)


def _no_clarification_decision() -> ClarificationDecision:
    return ClarificationDecision(
        needs_clarification=False,
        question=None,
        reason=_REASON_SUFFICIENT_INFO,
        missing_fields=[],
        confidence=_CONFIDENCE_SUFFICIENT_INFO,
    )


def _skipped_intent_decision() -> ClarificationDecision:
    return ClarificationDecision(
        needs_clarification=False,
        question=None,
        reason=_REASON_INTENT_SKIPPED,
        missing_fields=[],
        confidence=_CONFIDENCE_INTENT_SKIPPED,
    )


def _missing_role_decision() -> ClarificationDecision:
    return ClarificationDecision(
        needs_clarification=True,
        question=_QUESTION_MISSING_ROLE,
        reason=_REASON_MISSING_ROLE,
        missing_fields=[_FIELD_ROLE],
        confidence=_CONFIDENCE_MISSING_ROLE,
    )


def _missing_skills_or_family_decision() -> ClarificationDecision:
    return ClarificationDecision(
        needs_clarification=True,
        question=_QUESTION_MISSING_SKILLS_OR_FAMILY,
        reason=_REASON_MISSING_SKILLS_OR_FAMILY,
        missing_fields=[_FIELD_SKILLS],
        confidence=_CONFIDENCE_MISSING_SKILLS_OR_FAMILY,
    )


# ==============================================================================
# DECISION LOGIC
# ==============================================================================


def _decide(intent: ConversationIntent, state: ConversationState) -> ClarificationDecision:
    if not _is_evaluated_intent(intent):
        logger.debug("Intent %s is not evaluated for clarification; skipping.", intent)
        return _skipped_intent_decision()

    if not _has_role(state):
        logger.debug("Target role missing; requesting role clarification.")
        return _missing_role_decision()

    if not _has_skills(state) and not _has_assessment_family(state):
        logger.debug(
            "Role present but both skills and assessment family missing; "
            "requesting skills/assessment-type clarification."
        )
        return _missing_skills_or_family_decision()

    logger.debug("Sufficient hiring information available; no clarification needed.")
    return _no_clarification_decision()


# ==============================================================================
# PUBLIC API
# ==============================================================================


def evaluate_clarification(
    intent: ConversationIntent,
    state: ConversationState,
) -> ClarificationDecision:
    """Determine whether clarification is needed before recommendation can
    proceed, and if so, produce a single, deterministic clarification
    question.

    Args:
        intent: The classified conversation intent.
        state: The parsed structured conversation state.

    Returns:
        ClarificationDecision describing whether clarification is needed,
        the single question to ask (if any), the reason, missing fields,
        and a confidence score.

    Raises:
        ValidationError: if intent or state is None or of an invalid type.
        ClarificationEngineError: if decisioning fails unexpectedly.
    """
    try:
        _validate_intent(intent)
        _validate_state(state)
    except ValidationError:
        raise
    except Exception as exc:  # defensive: unexpected validation failure
        raise ValidationError(f"Failed to validate inputs: {exc}") from exc

    try:
        decision = _decide(intent, state)
        logger.info(
            "Clarification decision: needs_clarification=%s reason=%r confidence=%.2f",
            decision.needs_clarification,
            decision.reason,
            decision.confidence,
        )
        return decision
    except (ValidationError, ClarificationEngineError):
        raise
    except Exception as exc:  # defensive: unexpected decisioning failure
        raise ClarificationEngineError(f"Clarification decisioning failed: {exc}") from exc
