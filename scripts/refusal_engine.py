"""
Refusal Engine
==============

Phase 5 module for the SHL Conversational Assessment Recommendation System.

This module decides HOW to refuse a request once the Intent Detector has
already classified the current turn as ConversationIntent.REFUSE. It
performs ONLY deterministic, rule-based categorization and message
selection. It NEVER calls Gemini, performs retrieval, FAISS, BM25, fusion,
metadata filtering, recommendation generation, or comparison generation,
and it NEVER re-decides intent — that is the Intent Detector's exclusive
responsibility.

Its sole responsibility is: given an already-classified REFUSE intent and
the raw conversation, determine the most specific refusal category (e.g.
prompt injection, out-of-scope content generation, medical/legal/financial
advice, unsafe/security requests, general chit-chat, etc.), and produce a
single deterministic, catalog-safe refusal message that redirects the user
back toward SHL assessment recommendation, comparison, or refinement.

Python: 3.10.11
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Pattern
from scripts.intent_detector import ConversationIntent

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS
# ==============================================================================


class RefusalEngineError(Exception):
    """Raised when refusal decisioning fails due to an internal error."""


class ValidationError(RefusalEngineError):
    """Raised when the input intent or messages are invalid."""


# ==============================================================================
# ENUMS / DATA STRUCTURES
# ==============================================================================


class RefusalCategory(str, Enum):
    """Fine-grained reasons a request is being refused."""

    PROMPT_INJECTION = "PROMPT_INJECTION"
    OUT_OF_SCOPE_CONTENT = "OUT_OF_SCOPE_CONTENT"
    CAREER_ADVICE = "CAREER_ADVICE"
    MEDICAL_ADVICE = "MEDICAL_ADVICE"
    LEGAL_ADVICE = "LEGAL_ADVICE"
    FINANCIAL_ADVICE = "FINANCIAL_ADVICE"
    SECURITY_UNSAFE = "SECURITY_UNSAFE"
    POLITICAL_OPINION = "POLITICAL_OPINION"
    GENERAL_CHITCHAT = "GENERAL_CHITCHAT"
    OFF_TOPIC = "OFF_TOPIC"


@dataclass(frozen=True)
class RefusalDecision:
    """Result of the refusal decisioning process."""

    should_refuse: bool
    category: Optional[RefusalCategory]
    message: str
    reason: str
    confidence: float


@dataclass(frozen=True)
class _Message:
    """Internal normalized representation of a single conversation message."""

    role: str
    content: str


# ==============================================================================
# CONSTANTS
# ==============================================================================

_VALID_ROLES = frozenset({"user", "assistant", "system"})

# Reason / message used when the intent is not REFUSE at all.
_REASON_NOT_APPLICABLE: str = "Refusal is not applicable; intent is not REFUSE."
_CONFIDENCE_NOT_APPLICABLE: float = 0.99

# Fallback reason/confidence when REFUSE was set but no specific category matched.
_REASON_OFF_TOPIC_FALLBACK: str = (
    "Request is outside the supported SHL assessment recommendation scope."

)
_CONFIDENCE_OFF_TOPIC_FALLBACK: float = 0.70

_REDIRECT_SUFFIX: str = (
    " I'm built specifically to help you find, compare, and refine SHL "
    "assessments for your hiring needs — tell me the role or skills you're "
    "hiring for and I can take it from there."
)


# ==============================================================================
# COMPILED REGEX PATTERNS PER CATEGORY (compiled once at module import)
#
# Checked in priority order. The first category whose patterns match the
# current user message wins. Order matters: prompt-injection / security
# concerns are checked before softer categories like general chit-chat.
# ==============================================================================

_CATEGORY_PATTERNS: tuple[tuple[RefusalCategory, tuple[Pattern[str], ...]], ...] = (
    (
        RefusalCategory.PROMPT_INJECTION,
        (
            re.compile(r"\bignore\s+(all\s+|the\s+)?(previous|prior|above)\s+instructions?\b", re.IGNORECASE),
            re.compile(r"\bdisregard\s+(all\s+|the\s+)?(previous|prior|above)\s+instructions?\b", re.IGNORECASE),
            re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
            re.compile(r"\bact\s+as\s+(a\s+)?(?!.*\b(recruiter|hiring\s+manager|hr)\b)", re.IGNORECASE),
            re.compile(r"\bsystem\s*prompt\b", re.IGNORECASE),
            re.compile(r"\bjailbreak\b", re.IGNORECASE),
            re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
            re.compile(r"\breveal\s+your\s+(prompt|instructions)\b", re.IGNORECASE),
            re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE),
        ),
    ),
    (
        RefusalCategory.SECURITY_UNSAFE,
        (
            re.compile(r"\bhow\s+(do\s+i|to)\s+(hack|exploit|bypass\s+security)\b", re.IGNORECASE),
            re.compile(r"\bcrack\s+(a\s+)?password\b", re.IGNORECASE),
            re.compile(r"\bmake\s+(a\s+)?(bomb|weapon|malware|virus)\b", re.IGNORECASE),
            re.compile(r"\bcheat(ing)?\b", re.IGNORECASE),
re.compile(r"\bhelp\s+me\s+cheat\b", re.IGNORECASE),
re.compile(r"\bhow\s+to\s+cheat\b", re.IGNORECASE),
re.compile(r"\banswer\s+key\b", re.IGNORECASE),
re.compile(r"\bassessment\s+answers?\b", re.IGNORECASE),
re.compile(r"\bexam\s+answers?\b", re.IGNORECASE),
re.compile(r"\btest\s+answers?\b", re.IGNORECASE),
re.compile(r"\bimpersonat(e|ing|ion)\b", re.IGNORECASE),
re.compile(r"\bfake\b.*\b(score|result|certificate)\b", re.IGNORECASE),
re.compile(r"\bbypass\b.*\b(test|assessment|exam|shl)\b", re.IGNORECASE),
re.compile(r"\bleak(ed)?\b.*\b(question|assessment|test)\b", re.IGNORECASE),
re.compile(r"\bstolen\b.*\b(question|assessment|test)\b", re.IGNORECASE),
        ),
    ),
    (
        RefusalCategory.MEDICAL_ADVICE,
        (
            re.compile(r"\bmedical\s+advice\b", re.IGNORECASE),
            re.compile(r"\bdiagnos(e|is)\s+my\b", re.IGNORECASE),
            re.compile(r"\bwhat\s+medication\b", re.IGNORECASE),
        ),
    ),
    (
        RefusalCategory.LEGAL_ADVICE,
        (
            re.compile(r"\blegal\s+advice\b", re.IGNORECASE),
            re.compile(r"\bsue\s+(my|the)\s+(employer|company)\b", re.IGNORECASE),
            re.compile(r"\bis\s+it\s+legal\s+to\s+fire\b", re.IGNORECASE),
        ),
    ),
    (
        RefusalCategory.FINANCIAL_ADVICE,
        (
            re.compile(r"\bstock\s+(tips|advice|picks)\b", re.IGNORECASE),
            re.compile(r"\bshould\s+i\s+invest\b", re.IGNORECASE),
            re.compile(r"\bcrypto(currency)?\s+advice\b", re.IGNORECASE),
        ),
    ),
    (
        RefusalCategory.POLITICAL_OPINION,
        (
            re.compile(r"\bwho\s+(will|should)\s+i\s+vote\s+for\b", re.IGNORECASE),
            re.compile(r"\bwhich\s+political\s+party\b", re.IGNORECASE),
        ),
    ),
    (
        RefusalCategory.CAREER_ADVICE,
        (
            re.compile(r"\bresume\s+writing\b", re.IGNORECASE),
            re.compile(r"\bsalary\s+negotiat", re.IGNORECASE),
            re.compile(r"\bwrite\s+(me\s+)?(a\s+)?(resume|cover\s+letter)\b", re.IGNORECASE),
            re.compile(r"\bshould\s+i\s+accept\s+this\s+job\s+offer\b", re.IGNORECASE),
        ),
    ),
    (
        RefusalCategory.OUT_OF_SCOPE_CONTENT,
        (
            re.compile(r"\bwrite\s+(me\s+)?(a\s+)?(poem|essay|story|song|code|joke|screenplay)\b", re.IGNORECASE),
            re.compile(r"\btranslate\s+this\b", re.IGNORECASE),
            re.compile(r"\bsummarize\s+this\s+(article|document|pdf)\b", re.IGNORECASE),
        ),
    ),
    (
        RefusalCategory.GENERAL_CHITCHAT,
        (
            re.compile(r"\b(what'?s|what\s+is)\s+the\s+weather\b", re.IGNORECASE),
            re.compile(r"\bweather\s+(today|tomorrow|forecast|like)\b", re.IGNORECASE),
            re.compile(r"\btell\s+me\s+a\s+joke\b", re.IGNORECASE),
            re.compile(r"\bmake\s+me\s+laugh\b", re.IGNORECASE),
            re.compile(r"\bplay\s+(a\s+)?game\b", re.IGNORECASE),
            re.compile(r"\bhow\s+are\s+you\s+(feeling|doing)\b", re.IGNORECASE),
            
        ),
        
    ),
)


# ==============================================================================
# REFUSAL MESSAGES PER CATEGORY
# ==============================================================================

_BASE_MESSAGES: dict[RefusalCategory, str] = {
    RefusalCategory.PROMPT_INJECTION: (
        "I can't follow instructions that try to change my role or override "
        "how I operate."
    ),
    RefusalCategory.SECURITY_UNSAFE: (
    "I can't help with cheating, bypassing, impersonating, hacking, or obtaining answers for SHL or other assessments."
),

    RefusalCategory.MEDICAL_ADVICE: (
        "I'm not able to provide medical advice or diagnoses."
    ),
    RefusalCategory.LEGAL_ADVICE: (
        "I'm not able to provide legal advice."
    ),
    RefusalCategory.FINANCIAL_ADVICE: (
        "I'm not able to provide financial or investment advice."
    ),
    RefusalCategory.POLITICAL_OPINION: (
        "I don't provide political opinions or voting recommendations."
    ),
    RefusalCategory.CAREER_ADVICE: (
        "I'm not able to help with resume writing, salary negotiation, or "
        "job-offer decisions."
    ),
    RefusalCategory.OUT_OF_SCOPE_CONTENT: (
        "I'm not able to write creative or general-purpose content like "
        "that."
    ),
    RefusalCategory.GENERAL_CHITCHAT: (
        "That's outside what I can help with here."
    ),
    RefusalCategory.OFF_TOPIC: (
        "That request is outside what I'm able to help with."
    ),
}


# ==============================================================================
# VALIDATION
# ==============================================================================

def _validate_intent(intent: Any) -> None:
    if intent is None:
        raise ValidationError("intent must not be None.")
    if not isinstance(intent, ConversationIntent):
        raise ValidationError(f"intent must be a ConversationIntent, got {type(intent)!r}.")
    try:
        ConversationIntent(intent.value)
    except Exception as exc:
        raise ValidationError("Invalid ConversationIntent value.") from exc



def _validate_messages(messages: list[dict[str, str]]) -> list[_Message]:
    """Validate raw input messages and normalize them.

    Raises:
        ValidationError: if the message list or any message is malformed.
    """
    if not isinstance(messages, list) or not messages:
        raise ValidationError("messages must be a non-empty list.")

    normalized: list[_Message] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValidationError(f"Message at index {idx} is not a dict.")

        if "role" not in msg or "content" not in msg:
            raise ValidationError(f"Message at index {idx} is missing 'role' or 'content'.")

        role = msg["role"]
        content = msg["content"]

        if not isinstance(role, str) or not role.strip():
            raise ValidationError(f"Message at index {idx} has an invalid role.")

        if not isinstance(content, str) or not content.strip():
            raise ValidationError(f"Message at index {idx} has empty or invalid content.")

        role_normalized = role.strip().lower()
        if role_normalized not in _VALID_ROLES:
            raise ValidationError(f"Message at index {idx} has an unrecognized role: '{role}'.")

        normalized.append(_Message(role=role_normalized, content=content.strip()))

        if len(normalized) > 100:
            raise ValidationError(
        "Conversation exceeds maximum supported length." )

    if normalized[-1].role != "user":
        raise ValidationError("The final message must be from the 'user' role.")

    return normalized



# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _matches_any(patterns: tuple[Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _classify_category(text: str) -> tuple[RefusalCategory, float]:
    """Determine the most specific refusal category for the given text.

    Categories are checked in priority order; the first match wins. If no
    category-specific pattern matches, falls back to a generic OFF_TOPIC
    category with reduced confidence.
    """
    for category, patterns in _CATEGORY_PATTERNS:
        if _matches_any(patterns, text):
            return category, 0.95

    return RefusalCategory.OFF_TOPIC, _CONFIDENCE_OFF_TOPIC_FALLBACK


def _build_message(category: RefusalCategory) -> str:
    base = _BASE_MESSAGES.get(category, _BASE_MESSAGES[RefusalCategory.OFF_TOPIC])
    return base + _REDIRECT_SUFFIX


def _not_applicable_decision() -> RefusalDecision:
    return RefusalDecision(
        should_refuse=False,
        category=None,
        message="",
        reason=_REASON_NOT_APPLICABLE,
        confidence=_CONFIDENCE_NOT_APPLICABLE,
    )


# ==============================================================================
# DECISION LOGIC
# ==============================================================================


def _decide(intent: ConversationIntent, messages: list[dict[str, str]]) -> RefusalDecision:
    if intent != ConversationIntent.REFUSE:
        logger.debug("Intent %s is not REFUSE; refusal engine is not applicable.", intent)
        return _not_applicable_decision()

    normalized = _validate_messages(messages)
    text = normalized[-1].content

    category, confidence = _classify_category(text)
    message = _build_message(category)

    if category is RefusalCategory.OFF_TOPIC:
        reason = _REASON_OFF_TOPIC_FALLBACK
    else:
        reason = f"Message matched the {category.value} refusal category."

    logger.debug(
        "Refusal decision: category=%s confidence=%.2f",
        category.value,
        confidence,
    )

    return RefusalDecision(
        should_refuse=True,
        category=category,
        message=message,
        reason=reason,
        confidence=confidence,
    )


# ==============================================================================
# PUBLIC API
# ==============================================================================


def evaluate_refusal(
    intent: ConversationIntent,
    messages: list[dict[str, str]],
) -> RefusalDecision:
    """Determine whether and how to refuse the current turn.

    This function assumes intent classification has already happened
    upstream (via ``intent_detector.detect_intent``). It does not
    re-classify intent; it only decides on the refusal category and
    produces a single deterministic, catalog-safe refusal message when the
    intent is ConversationIntent.REFUSE.

    Args:
        intent: The classified conversation intent.
        messages: Full conversation history. Each item must be a dict with
            'role' and 'content' string keys. The last message must be from
            the 'user' role. Only required/validated when intent is REFUSE.

    Returns:
        RefusalDecision describing whether to refuse, the refusal category
        (if applicable), the user-facing message, a reason, and a
        confidence score.

    Raises:
        ValidationError: if intent is invalid, or if intent is REFUSE and
            the message structure is invalid.
        RefusalEngineError: if decisioning fails unexpectedly.
    """
    try:
        _validate_intent(intent)
    except ValidationError:
        raise
    except Exception as exc:  # defensive: unexpected validation failure
        raise ValidationError(f"Failed to validate intent: {exc}") from exc

    try:
        decision = _decide(intent, messages)
        logger.info(
            "Refusal decision: should_refuse=%s category=%s confidence=%.2f",
            decision.should_refuse,
            decision.category.value if decision.category else None,
            decision.confidence,
        )
        return decision
    except (ValidationError, RefusalEngineError):
        raise
    except Exception as exc:  # defensive: unexpected decisioning failure
        raise RefusalEngineError(f"Refusal decisioning failed: {exc}") from exc
