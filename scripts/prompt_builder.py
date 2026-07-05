"""
Prompt Builder
==============

Phase 5 module for the SHL Conversational Assessment Recommendation System.

This module assembles ONE deterministic Gemini prompt from the outputs of
all upstream Phase 5 modules (Intent Detector, Conversation State, Clarification
Engine, Recommendation Engine, Comparison Engine, Refusal Engine). It performs
ONLY deterministic string assembly and dictionary construction. It NEVER calls
Gemini, performs retrieval, FAISS, BM25, fusion, metadata filtering, makes
recommendations, makes comparisons, or changes conversation state.

Python: 3.10.11
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from scripts.intent_detector import ConversationIntent
from scripts.conversation_state import ConversationState
from scripts.clarification_engine import ClarificationDecision
from scripts.recommendation_engine import RecommendationDecision
from scripts.comparison_engine import ComparisonDecision
from scripts.refusal_engine import RefusalDecision

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS
# ==============================================================================


class PromptBuilderError(Exception):
    """Raised when prompt building fails due to an internal error."""


class ValidationError(PromptBuilderError):
    """Raised when an input to the prompt builder is invalid."""


# ==============================================================================
# OUTPUT DATA STRUCTURE
# ==============================================================================


@dataclass(frozen=True)
class PromptPayload:
    """Result of the prompt building process."""

    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# CONSTANTS
# ==============================================================================

_SYSTEM_PROMPT: str = (
    "You are an assistant that recommends SHL assessments ONLY. "
    "You must never hallucinate or invent assessments that were not "
    "explicitly supplied to you. Use ONLY the metadata provided in this "
    "prompt as your source of truth. Do not rely on prior knowledge of SHL's "
    "catalog. If the supplied information is insufficient to answer, say so "
    "rather than guessing. Be concise, factual, and professional in your "
    "response.\n\n"
    "Formatting rules — follow these exactly, they are not optional:\n"
    "- Write plain conversational sentences only. This text is returned as a "
    "single JSON string field, not rendered as markdown, so any markdown "
    "syntax you use (asterisks, bullet dashes, bold, headers) will appear "
    "to the reader as literal stray characters.\n"
    "- Never use *, **, #, or - as formatting/bullet characters. Never start "
    "a line with a number followed by a period to fake a list. If you are "
    "listing multiple assessments, do it inside one flowing sentence or "
    "short paragraph, separated by commas or 'and' — not as a vertical list.\n"
    "- Never copy the internal field names you are given (e.g. "
    "'adaptive', 'duration_minutes', 'job_levels', 'remote', 'entity_id', "
    "'rrf_score') into your answer verbatim. Paraphrase each fact naturally "
    "instead — e.g. turn duration_minutes=11 and adaptive=False into "
    "'an 11-minute, non-adaptive test', not 'Duration: 11 / Adaptive: False'.\n"
    "- Keep the whole reply to a short paragraph or two. Do not restate "
    "every supplied field for every assessment — mention the details that "
    "actually matter to the user's request."
)

_NO_SUMMARY_TEXT: str = "No structured hiring context has been captured yet."

# Only these catalog_metadata.json fields are useful to a human-readable
# prompt or response. Fields such as searchable_text, keywords,
# filter_tokens, ranking_tokens, and metadata_version exist purely to
# support retrieval/matching and are large, redundant, and irrelevant to
# Gemini's answer or the API response — including them would needlessly
# inflate prompt tokens and serialize the full catalog record outward.
# Public so app.routes can reuse the exact same whitelist when building the
# HTTP response, without duplicating the field list.
DISPLAY_METADATA_FIELDS: tuple[str, ...] = (
    "assessment_family",
    "duration_minutes",
    "adaptive",
    "remote",
    "languages",
    "job_levels",
)


def display_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return only the whitelisted, human-relevant subset of a metadata dict.

    Never serializes the full catalog record (e.g. searchable_text,
    keywords, filter_tokens, ranking_tokens) into the prompt or response.
    """
    if not metadata:
        return {}
    return {
        key: metadata[key]
        for key in DISPLAY_METADATA_FIELDS
        if key in metadata
    }


# Backwards-compatible private alias (internal call sites in this module).
_display_metadata = display_metadata


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
    return state


def _validate_clarification(clarification: Any) -> ClarificationDecision:
    if clarification is None:
        raise ValidationError("clarification must not be None.")
    if not isinstance(clarification, ClarificationDecision):
        raise ValidationError(
            f"clarification must be a ClarificationDecision, got "
            f"{type(clarification).__name__}."
        )
    return clarification


def _validate_recommendation(recommendation: Any) -> RecommendationDecision:
    if recommendation is None:
        raise ValidationError("recommendation must not be None.")
    if not isinstance(recommendation, RecommendationDecision):
        raise ValidationError(
            f"recommendation must be a RecommendationDecision, got "
            f"{type(recommendation).__name__}."
        )
    return recommendation


def _validate_comparison(comparison: Any) -> ComparisonDecision:
    if comparison is None:
        raise ValidationError("comparison must not be None.")
    if not isinstance(comparison, ComparisonDecision):
        raise ValidationError(
            f"comparison must be a ComparisonDecision, got "
            f"{type(comparison).__name__}."
        )
    return comparison


def _validate_refusal(refusal: Any) -> RefusalDecision:
    if refusal is None:
        raise ValidationError("refusal must not be None.")
    if not isinstance(refusal, RefusalDecision):
        raise ValidationError(
            f"refusal must be a RefusalDecision, got {type(refusal).__name__}."
        )
    return refusal


def _validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValidationError("messages must be a non-empty list.")

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValidationError(f"Message at index {idx} must be a dictionary.")
        if "role" not in msg or "content" not in msg:
            raise ValidationError(
                f"Message at index {idx} is missing 'role' or 'content'."
            )
        if not isinstance(msg["role"], str) or not msg["role"].strip():
            raise ValidationError(f"Message at index {idx} has an invalid role.")
        if not isinstance(msg["content"], str) or not msg["content"].strip():
            raise ValidationError(f"Message at index {idx} has invalid content.")

    return messages


def _validate_inputs(
    intent: Any,
    state: Any,
    clarification: Any,
    recommendation: Any,
    comparison: Any,
    refusal: Any,
    messages: Any,
) -> tuple[
    ConversationIntent,
    ConversationState,
    ClarificationDecision,
    RecommendationDecision,
    ComparisonDecision,
    RefusalDecision,
    list[dict[str, str]],
]:
    validated_intent = _validate_intent(intent)
    validated_state = _validate_state(state)
    validated_clarification = _validate_clarification(clarification)
    validated_recommendation = _validate_recommendation(recommendation)
    validated_comparison = _validate_comparison(comparison)
    validated_refusal = _validate_refusal(refusal)
    validated_messages = _validate_messages(messages)

    return (
        validated_intent,
        validated_state,
        validated_clarification,
        validated_recommendation,
        validated_comparison,
        validated_refusal,
        validated_messages,
    )


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _conversation_summary(state: ConversationState) -> str:
    return state.conversation_summary or _NO_SUMMARY_TEXT


def _format_recommendation_metadata(recommendation: RecommendationDecision) -> str:
    lines: list[str] = []
    for idx, candidate in enumerate(recommendation.recommendations, start=1):
        lines.append(
            f"{idx}. {candidate.canonical_name} "
            f"(entity_id={candidate.entity_id}, rrf_score={candidate.rrf_score:.4f})"
        )
        display_meta = _display_metadata(candidate.metadata)
        if display_meta:
            for key in sorted(display_meta.keys()):
                lines.append(f"    - {key}: {display_meta[key]}")
    return "\n".join(lines)


def _format_comparison_data(comparison: ComparisonDecision) -> str:
    lines: list[str] = []
    for idx, item in enumerate(comparison.comparisons, start=1):
        lines.append(f"{idx}. {item.canonical_name} (entity_id={item.entity_id})")
        display_meta = _display_metadata(item.metadata)
        if display_meta:
            for key in sorted(display_meta.keys()):
                lines.append(f"    - {key}: {display_meta[key]}")
    return "\n".join(lines)


# ==============================================================================
# BUILDER FUNCTIONS (one per response type)
# ==============================================================================


def _build_refusal(
    state: ConversationState,
    refusal: RefusalDecision,
) -> str:
    category = refusal.category.value if refusal.category else "UNKNOWN"
    parts = [
        "Response type: REFUSAL",
        f"Refusal category: {category}",
        f"Refusal reason: {refusal.reason}",
        f"Refusal message to relay to the user: {refusal.message}",
        f"Conversation summary: {_conversation_summary(state)}",
        "",
        "Instruction: Politely relay the refusal message above to the user, "
        "in plain sentences with no markdown formatting. Do not attempt to "
        "fulfill the original request. Do not recommend, compare, or invent "
        "any SHL assessments in this response.",
    ]
    return "\n".join(parts)


def _build_clarification(
    state: ConversationState,
    clarification: ClarificationDecision,
) -> str:
    parts = [
        "Response type: CLARIFICATION",
        f"Clarification reason: {clarification.reason}",
        f"Missing fields: {', '.join(clarification.missing_fields) if clarification.missing_fields else 'none'}",
        f"Clarification question to ask the user: {clarification.question or ''}",
        f"Conversation summary: {_conversation_summary(state)}",
        "",
        "Instruction: Ask the user the clarification question above in a "
        "concise, friendly way, in one or two plain sentences with no "
        "markdown formatting. Do not recommend or invent any SHL "
        "assessments until the missing information is provided.",
    ]
    return "\n".join(parts)


def _build_comparison(
    state: ConversationState,
    comparison: ComparisonDecision,
) -> str:
    parts = [
        "Response type: COMPARISON",
        f"Comparison reason: {comparison.reason}",
        f"Conversation summary: {_conversation_summary(state)}",
    ]

    if comparison.ready:
        parts.append(f"Number of assessments to compare: {len(comparison.comparisons)}")
        parts.append(
            "Comparison reference data below (internal field names — use "
            "ONLY these facts, do not invent details, but NEVER copy this "
            "data's structure, field names, or formatting into your answer):"
        )
        parts.append(_format_comparison_data(comparison))
        parts.append("")
        parts.append(
            "Instruction: Using ONLY the facts in the reference data above, "
            "write a short, natural-sounding paragraph comparing the listed "
            "SHL assessments, highlighting the differences that matter — in "
            "plain sentences, not a bulleted or numbered list, not a field-"
            "by-field dump, and no markdown formatting of any kind."
        )
    else:
        parts.append(
            "Comparison data: none available. Fewer than two assessments "
            "could be resolved against the catalog."
        )
        parts.append("")
        parts.append(
            "Instruction: Explain to the user, in plain sentences with no "
            "markdown formatting, that a comparison could not be completed "
            "for the reason above, and ask them to specify valid SHL "
            "assessment names to compare."
        )

    return "\n".join(parts)


def _build_recommendation(
    state: ConversationState,
    recommendation: RecommendationDecision,
) -> str:
    parts = [
        "Response type: RECOMMENDATION",
        f"Recommendation reason: {recommendation.reason}",
        f"Conversation summary: {_conversation_summary(state)}",
    ]

    if recommendation.ready:
        parts.append(f"Number of candidate assessments: {len(recommendation.recommendations)}")
        parts.append(
            "Recommendation reference data below (internal field names — "
            "use ONLY these facts, do not invent details, but NEVER copy "
            "this data's structure, field names, or formatting into your "
            "answer):"
        )
        parts.append(_format_recommendation_metadata(recommendation))
        parts.append("")
        parts.append(
            "Instruction: Using ONLY the facts in the reference data above, "
            "write a short, natural-sounding paragraph recommending these "
            "SHL assessments to the user. Name each assessment and weave in "
            "only the details relevant to what the user asked for, in plain "
            "sentences — not a bulleted or numbered list, not a field-by-"
            "field dump, and no markdown formatting of any kind."
        )
    else:
        parts.append(
            "Recommendation metadata: none available. No matching SHL "
            "assessments were found or recommendations are not yet ready."
        )
        parts.append("")
        parts.append(
            "Instruction: Explain to the user, in plain sentences with no "
            "markdown formatting, that no matching SHL assessments could be "
            "found for the reason above, and invite them to refine their "
            "request."
        )

    return "\n".join(parts)


def _build_fallback(state: ConversationState) -> str:
    parts = [
        "Response type: FALLBACK",
        f"Conversation summary: {_conversation_summary(state)}",
        "",
        "Instruction: The current turn does not map to a refusal, "
        "clarification, comparison, or recommendation path. Ask the user, "
        "in a concise and friendly way, with no markdown formatting, what "
        "role or skills they are hiring for so that SHL assessments can be "
        "recommended.",
    ]
    return "\n".join(parts)


# ==============================================================================
# ROUTING LOGIC
# ==============================================================================


def _route(
    intent: ConversationIntent,
    state: ConversationState,
    clarification: ClarificationDecision,
    recommendation: RecommendationDecision,
    comparison: ComparisonDecision,
    refusal: RefusalDecision,
) -> tuple[str, str]:
    """Determine the active path and build the corresponding user prompt.

    Returns:
        A tuple of (path_name, user_prompt).
    """
    if refusal.should_refuse:
        logger.debug("Routing to refusal prompt.")
        return "refusal", _build_refusal(state, refusal)

    if clarification.needs_clarification:
        logger.debug("Routing to clarification prompt.")
        return "clarification", _build_clarification(state, clarification)

    if intent == ConversationIntent.COMPARE:
        logger.debug("Routing to comparison prompt.")
        return "comparison", _build_comparison(state, comparison)

    if recommendation.ready:
        logger.debug("Routing to recommendation prompt.")
        return "recommendation", _build_recommendation(state, recommendation)

    logger.debug("Routing to fallback prompt.")
    return "fallback", _build_fallback(state)


def _build_metadata(
    intent: ConversationIntent,
    path: str,
    clarification: ClarificationDecision,
    recommendation: RecommendationDecision,
    comparison: ComparisonDecision,
    refusal: RefusalDecision,
) -> dict[str, Any]:
    return {
        "intent": intent.value,
        "path": path,
        "ready": bool(recommendation.ready or comparison.ready),
        "clarification_required": bool(clarification.needs_clarification),
        "recommendation_count": len(recommendation.recommendations),
        "comparison_count": len(comparison.comparisons),
        "refusal": bool(refusal.should_refuse),
    }


# ==============================================================================
# PUBLIC API
# ==============================================================================


def build_prompt(
    intent: ConversationIntent,
    state: ConversationState,
    clarification: ClarificationDecision,
    recommendation: RecommendationDecision,
    comparison: ComparisonDecision,
    refusal: RefusalDecision,
    messages: list[dict[str, str]],
) -> PromptPayload:
    """Assemble a single deterministic Gemini prompt for the current turn.

    This function never calls Gemini, never performs retrieval, FAISS,
    BM25, fusion, or metadata filtering, and never makes recommendation or
    comparison decisions itself. It only assembles a prompt from the
    already-computed decisions produced by upstream Phase 5 modules.

    Routing priority (first match wins):
        1. RefusalDecision.should_refuse       -> refusal prompt
        2. ClarificationDecision.needs_clarification -> clarification prompt
        3. ConversationIntent == COMPARE       -> comparison prompt
        4. RecommendationDecision.ready         -> recommendation prompt
        5. otherwise                            -> generic fallback prompt

    Args:
        intent: The classified conversation intent.
        state: The parsed structured conversation state.
        clarification: The clarification engine's decision for this turn.
        recommendation: The recommendation engine's decision for this turn.
        comparison: The comparison engine's decision for this turn.
        refusal: The refusal engine's decision for this turn.
        messages: Full conversation history (validated for structure only).

    Returns:
        PromptPayload containing the constant system prompt, the
        deterministically assembled user prompt, and a metadata dictionary.

    Raises:
        ValidationError: if any input is None or of an invalid type/shape.
        PromptBuilderError: if prompt assembly fails unexpectedly.
    """
    try:
        (
            validated_intent,
            validated_state,
            validated_clarification,
            validated_recommendation,
            validated_comparison,
            validated_refusal,
            validated_messages,
        ) = _validate_inputs(
            intent, state, clarification, recommendation, comparison, refusal, messages
        )
    except ValidationError:
        raise
    except Exception as exc:  # defensive: unexpected validation failure
        raise ValidationError(f"Failed to validate inputs: {exc}") from exc

    try:
        path, user_prompt = _route(
            validated_intent,
            validated_state,
            validated_clarification,
            validated_recommendation,
            validated_comparison,
            validated_refusal,
        )

        metadata = _build_metadata(
            validated_intent,
            path,
            validated_clarification,
            validated_recommendation,
            validated_comparison,
            validated_refusal,
        )

        logger.info(
            "Prompt built: path=%s intent=%s clarification_required=%s "
            "recommendation_count=%d comparison_count=%d refusal=%s",
            path,
            metadata["intent"],
            metadata["clarification_required"],
            metadata["recommendation_count"],
            metadata["comparison_count"],
            metadata["refusal"],
        )

        return PromptPayload(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            metadata=metadata,
        )
    except (ValidationError, PromptBuilderError):
        raise
    except Exception as exc:  # defensive: unexpected assembly failure
        raise PromptBuilderError(f"Prompt assembly failed: {exc}") from exc
