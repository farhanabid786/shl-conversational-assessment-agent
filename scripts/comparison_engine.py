"""
Comparison Engine
==================

Phase 5 module for the SHL Conversational Assessment Recommendation System.

This module compares SHL assessments that already exist in the catalog. It
performs ONLY deterministic dictionary lookups against a pre-built
entity_lookup. It never calls Gemini, never runs SentenceTransformer,
FAISS, BM25, fusion, or metadata filtering, and never generates
recommendations or natural-language prompts of any kind.

Its sole responsibility is to resolve user-supplied comparison targets
against the catalog and package the corresponding metadata into
ComparisonItem objects for downstream consumption (e.g. a future Prompt
Builder module). It never sorts, reranks, mutates metadata, or produces
explanations.

Python: 3.10.11
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from scripts.intent_detector import ConversationIntent
from scripts.conversation_state import ConversationState

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS
# ==============================================================================


class ComparisonEngineError(Exception):
    """Base exception for all comparison engine failures."""


class ValidationError(ComparisonEngineError):
    """Raised when an input to the comparison engine is invalid."""


# ==============================================================================
# CONSTANTS
# ==============================================================================

REASON_NOT_APPLICABLE: str = "Comparison not applicable."
REASON_NO_MATCHES: str = "No matching SHL assessments found."
REASON_SINGLE_MATCH: str = "At least two assessments are required for comparison."
REASON_READY: str = "Comparison targets resolved successfully."

CONFIDENCE_NOT_APPLICABLE: float = 0.99
CONFIDENCE_NO_MATCHES: float = 0.95
CONFIDENCE_SINGLE_MATCH: float = 0.95
CONFIDENCE_READY: float = 1.00

MIN_REQUIRED_MATCHES: int = 2


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================


@dataclass(frozen=True)
class ComparisonItem:
    """A single resolved comparison target.

    Attributes
    ----------
    entity_id:
        The catalog entity_id the target resolved to.
    canonical_name:
        The canonical display name of the assessment.
    metadata:
        The full metadata record for this entity, as stored in
        entity_lookup. Never mutated.
    """

    entity_id: str
    canonical_name: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ComparisonDecision:
    """Result of a comparison request.

    Attributes
    ----------
    ready:
        True only when two or more comparison targets were successfully
        resolved against the catalog.
    comparisons:
        Ordered list of ComparisonItem objects, preserving the order the
        user supplied the targets in. Empty unless ready is True.
    reason:
        Human-readable explanation of the decision.
    confidence:
        Deterministic confidence score associated with the decision.
    """

    ready: bool
    comparisons: list[ComparisonItem] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0


# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================


def _validate_intent(intent: Any) -> ConversationIntent:
    if intent is None:
        raise ValidationError("intent must not be None.")
    if not isinstance(intent, ConversationIntent):
        raise ValidationError(
            f"intent must be a ConversationIntent, got {type(intent).__name__}."
        )
    return intent


def _validate_state(state: Any) -> ConversationState:
    if state is None:
        raise ValidationError("state must not be None.")
    if not isinstance(state, ConversationState):
        raise ValidationError(
            f"state must be a ConversationState, got {type(state).__name__}."
        )
    if not isinstance(state.comparison_targets, list):
        raise ValidationError("state.comparison_targets must be a list.")
    for idx, target in enumerate(state.comparison_targets):
        if not isinstance(target, str):
            raise ValidationError(
                f"state.comparison_targets[{idx}] must be a string, got "
                f"{type(target).__name__}."
            )
    return state


def _validate_entity_lookup(entity_lookup: Any) -> dict[str, dict[str, Any]]:
    if entity_lookup is None:
        raise ValidationError("entity_lookup must not be None.")
    if not isinstance(entity_lookup, dict):
        raise ValidationError(
            f"entity_lookup must be a dict, got {type(entity_lookup).__name__}."
        )
    for entity_id, record in entity_lookup.items():
        if not isinstance(entity_id, str):
            raise ValidationError("entity_lookup keys must be strings.")
        if not isinstance(record, dict):
            raise ValidationError(
                f"entity_lookup[{entity_id!r}] must be a dict, got "
                f"{type(record).__name__}."
            )
    return entity_lookup


# ==============================================================================
# MATCHING
# ==============================================================================


def resolve_target(
    target: str, entity_lookup: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """Resolve a single user-supplied comparison target against the catalog.

    Matching order:
        1. Exact match against canonical_name.
        2. Case-insensitive exact match against canonical_name.

    No fuzzy matching is performed. Returns the metadata record on success,
    or None if no match was found.

    Parameters
    ----------
    target:
        The raw comparison target string supplied by the user.
    entity_lookup:
        dict[entity_id, metadata] built by retriever_loader.py.

    Returns
    -------
    dict[str, Any] | None
        The matching metadata record, or None if unresolved.
    """
    # Stage 1: exact match
    for record in entity_lookup.values():
        canonical_name = record.get("canonical_name")
        if isinstance(canonical_name, str) and canonical_name == target:
            return record

    # Stage 2: case-insensitive exact match
    target_lower = target.lower()
    for record in entity_lookup.values():
        canonical_name = record.get("canonical_name")
        if isinstance(canonical_name, str) and canonical_name.lower() == target_lower:
            return record

    return None


def _resolve_all_targets(
    targets: list[str], entity_lookup: dict[str, dict[str, Any]]
) -> list[ComparisonItem]:
    """Resolve each target in order, silently ignoring unresolved targets.

    Preserves user-supplied order. Does not deduplicate, sort, or rerank.
    """
    resolved: list[ComparisonItem] = []

    for target in targets:
        record = resolve_target(target, entity_lookup)

        if record is None:
            logger.info("Comparison target not found in catalog: %r", target)
            continue

        entity_id = str(record.get("entity_id", ""))
        canonical_name = str(record.get("canonical_name", ""))

        resolved.append(
            ComparisonItem(
                entity_id=entity_id,
                canonical_name=canonical_name,
                metadata=record,
            )
        )

    return resolved


# ==============================================================================
# DECISION BUILDERS
# ==============================================================================


def _not_applicable_decision() -> ComparisonDecision:
    return ComparisonDecision(
        ready=False,
        comparisons=[],
        reason=REASON_NOT_APPLICABLE,
        confidence=CONFIDENCE_NOT_APPLICABLE,
    )


def _no_matches_decision() -> ComparisonDecision:
    return ComparisonDecision(
        ready=False,
        comparisons=[],
        reason=REASON_NO_MATCHES,
        confidence=CONFIDENCE_NO_MATCHES,
    )


def _single_match_decision() -> ComparisonDecision:
    return ComparisonDecision(
        ready=False,
        comparisons=[],
        reason=REASON_SINGLE_MATCH,
        confidence=CONFIDENCE_SINGLE_MATCH,
    )


def _ready_decision(comparisons: list[ComparisonItem]) -> ComparisonDecision:
    return ComparisonDecision(
        ready=True,
        comparisons=comparisons,
        reason=REASON_READY,
        confidence=CONFIDENCE_READY,
    )


# ==============================================================================
# PUBLIC API
# ==============================================================================


def build_comparison(
    intent: ConversationIntent,
    state: ConversationState,
    entity_lookup: dict[str, dict[str, Any]],
) -> ComparisonDecision:
    """Build a ComparisonDecision for the current conversational turn.

    This function performs deterministic dictionary lookups only. It never
    calls Gemini, never runs SentenceTransformer/FAISS/BM25/fusion/metadata
    filtering, and never generates natural language or recommendations.

    Parameters
    ----------
    intent:
        The classified ConversationIntent for the current turn. Only
        ConversationIntent.COMPARE is supported; all other intents result
        in a not-applicable decision.
    state:
        The current ConversationState. Only state.comparison_targets is
        used.
    entity_lookup:
        dict[entity_id, metadata] built by retriever_loader.py.

    Returns
    -------
    ComparisonDecision
        The resolved comparison decision.

    Raises
    ------
    ValidationError
        If any input is missing or structurally invalid.
    ComparisonEngineError
        If resolution fails unexpectedly.
    """
    try:
        intent = _validate_intent(intent)
        state = _validate_state(state)
        entity_lookup = _validate_entity_lookup(entity_lookup)
    except ValidationError:
        raise
    except Exception as exc:  # defensive: unexpected validation failure
        raise ValidationError(f"Failed to validate comparison inputs: {exc}") from exc

    try:
        if intent != ConversationIntent.COMPARE:
            logger.debug("Intent %s is not COMPARE; comparison not applicable.", intent)
            return _not_applicable_decision()

        targets = state.comparison_targets

        resolved = _resolve_all_targets(targets, entity_lookup)

        if len(resolved) == 0:
            logger.info("No comparison targets resolved out of %d supplied.", len(targets))
            return _no_matches_decision()

        if len(resolved) == 1:
            logger.info("Only one comparison target resolved; two are required.")
            return _single_match_decision()

        logger.info(
            "Comparison ready: %d targets resolved out of %d supplied.",
            len(resolved),
            len(targets),
        )
        return _ready_decision(resolved)

    except (ValidationError, ComparisonEngineError):
        raise
    except Exception as exc:  # defensive: unexpected resolution failure
        raise ComparisonEngineError(f"Comparison resolution failed: {exc}") from exc
