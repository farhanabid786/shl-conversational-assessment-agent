"""
Intent Detector
===============

Phase 5 module for the SHL Conversational Assessment Recommendation System.

This module performs ONLY deterministic, rule-based classification of the
current user intent based on the conversation history. It does not call any
LLM, perform retrieval, fusion, metadata filtering, or generate
recommendations. Its sole responsibility is to determine which branch of the
downstream workflow (clarify / recommend / refine / compare / refuse) should
run next.

Python: 3.10.11
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Pattern

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS
# ==============================================================================


class IntentDetectionError(Exception):
    """Raised when intent detection cannot be completed due to an internal error."""


class ValidationError(IntentDetectionError):
    """Raised when the input message structure is invalid."""


# ==============================================================================
# ENUMS / DATA STRUCTURES
# ==============================================================================


class ConversationIntent(str, Enum):
    """Supported conversation intents."""

    RECOMMEND = "RECOMMEND"
    CLARIFY = "CLARIFY"
    REFINE = "REFINE"
    COMPARE = "COMPARE"
    REFUSE = "REFUSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class IntentResult:
    """Result of intent classification."""

    intent: ConversationIntent
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class _Message:
    """Internal normalized representation of a single conversation message."""

    role: str
    content: str


# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

_MIN_INFORMATIVE_WORDS: int = 4
_VAGUE_MAX_WORDS: int = 6

_VALID_ROLES = frozenset({"user", "assistant", "system"})


# ==============================================================================
# COMPILED REGEX PATTERNS (compiled once at module import)
# ==============================================================================

# --- REFUSE ---------------------------------------------------------------

_REFUSE_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"\bignore\s+(all\s+|the\s+)?(previous|prior|above)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+|the\s+)?(previous|prior|above)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(a\s+)?(?!.*\b(recruiter|hiring\s+manager|hr)\b)", re.IGNORECASE),
    re.compile(r"\bsystem\s*prompt\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
    re.compile(r"\bresume\s+writing\b", re.IGNORECASE),
    re.compile(r"\bwrite\s+(me\s+)?(a\s+)?(resume|cover\s+letter|poem|essay|story|song|code)\b", re.IGNORECASE),
    re.compile(r"\bsalary\s+negotiat", re.IGNORECASE),
    re.compile(r"\b(what'?s|what\s+is)\s+the\s+weather\b", re.IGNORECASE),
    re.compile(r"\bweather\s+(today|tomorrow|forecast|like)\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+a\s+joke\b", re.IGNORECASE),
    re.compile(r"\bmake\s+me\s+laugh\b", re.IGNORECASE),
    re.compile(r"\bmedical\s+advice\b", re.IGNORECASE),
    re.compile(r"\bdiagnos(e|is)\s+my\b", re.IGNORECASE),
    re.compile(r"\blegal\s+advice\b", re.IGNORECASE),
    re.compile(r"\bsue\s+(my|the)\s+(employer|company)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(do\s+i|to)\s+(hack|exploit|bypass\s+security)\b", re.IGNORECASE),
    re.compile(r"\bstock\s+(tips|advice|picks)\b", re.IGNORECASE),
    re.compile(r"\bwho\s+(will|should)\s+i\s+vote\s+for\b", re.IGNORECASE),
    re.compile(r"\bplay\s+(a\s+)?game\b", re.IGNORECASE),
)

# --- COMPARE ----------------------------------------------------------------

_COMPARE_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"\bdifference\s+between\b", re.IGNORECASE),
    re.compile(r"\bcompar(e|ing|ison)\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(one|is\s+better|is\s+best)\b", re.IGNORECASE),
    re.compile(r"\bbetter\s+than\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(do|does)\s+.+\s+differ\b", re.IGNORECASE),
)

# --- REFINE -------------------------------------------------------------

_REFINE_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"^\s*actually\b", re.IGNORECASE),
    re.compile(r"\binstead\b", re.IGNORECASE),
    re.compile(r"\brather\s+than\b", re.IGNORECASE),
    re.compile(r"\b(include|add)\s+(personality|cognitive|coding|situational|remote)\b", re.IGNORECASE),
    re.compile(r"\bexclude\s+", re.IGNORECASE),
    re.compile(r"\bremove\s+", re.IGNORECASE),
    re.compile(r"\bwithout\s+", re.IGNORECASE),
    re.compile(r"\bremote\s+only\b", re.IGNORECASE),
    re.compile(r"\bonsite\s+only\b", re.IGNORECASE),
    re.compile(r"\breduce\s+(the\s+)?(duration|time|length)\b", re.IGNORECASE),
    re.compile(r"\bshorter\b", re.IGNORECASE),
    re.compile(r"\blonger\b", re.IGNORECASE),
    re.compile(r"\bunder\s+\d+\s*(min|minutes|mins)?\b", re.IGNORECASE),
    re.compile(r"\bless\s+than\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bmore\s+(options|results|assessments)\b", re.IGNORECASE),
    re.compile(r"\bfewer\s+(options|results|assessments)\b", re.IGNORECASE),
    re.compile(r"\bnarrow\s+(it\s+)?down\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+about\b", re.IGNORECASE),
    re.compile(r"\bcan\s+you\s+(also|instead)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+adaptive\b", re.IGNORECASE),
    re.compile(r"\bnon[\s-]?adaptive\b", re.IGNORECASE),
)

# --- RECOMMEND ----------------------------------------------------------

_RECOMMEND_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"\bi\s+need\s+(a\s+|an\s+|some\s+)?tests?\b", re.IGNORECASE),
    re.compile(r"\bi\s+need\s+(an\s+)?assessments?\b", re.IGNORECASE),
    re.compile(r"\brecommend\b", re.IGNORECASE),
    re.compile(r"\bsuggest\s+(an?\s+)?(test|assessment)\b", re.IGNORECASE),
    re.compile(r"\bhiring\s+(for\s+)?[a-z]+", re.IGNORECASE),
    re.compile(r"\blooking\s+for\s+(a\s+|an\s+|some\s+)?(test|assessment|evaluation)\b", re.IGNORECASE),
    re.compile(r"\bwe\s+(are|'re)\s+hiring\b", re.IGNORECASE),
    re.compile(r"\btests?\s+for\s+[a-z]+", re.IGNORECASE),
    re.compile(r"\bassessments?\s+for\s+[a-z]+", re.IGNORECASE),
    re.compile(r"\bcandidate(s)?\s+for\b", re.IGNORECASE),
    re.compile(r"\bevaluat(e|ing)\s+(candidates|applicants|skills)\b", re.IGNORECASE),
    re.compile(r"\bscreen(ing)?\s+(candidates|applicants)\b", re.IGNORECASE),
    re.compile(r"\bposition\s+for\b", re.IGNORECASE),
    re.compile(r"\brole\s+(of|for)\b", re.IGNORECASE),
)

# --- CLARIFY (vague requests) --------------------------------------------

_CLARIFY_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"^\s*i\s+need\s+a\s+test\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*hiring\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*assessment\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*test\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(i\s+need\s+)?(help|assessments?|tests?)\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(hi|hello|hey)\.?\s*$", re.IGNORECASE),
)

# Minimal domain keywords used to decide whether a short/vague message is at
# least on-topic (assessment/hiring related) versus pure noise/unrelated text.
_DOMAIN_HINT_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"\btest(s)?\b", re.IGNORECASE),
    re.compile(r"\bassessment(s)?\b", re.IGNORECASE),
    re.compile(r"\bhir(e|ing)\b", re.IGNORECASE),
    re.compile(r"\bcandidate(s)?\b", re.IGNORECASE),
    re.compile(r"\bevaluat", re.IGNORECASE),
    re.compile(r"\brecruit", re.IGNORECASE),
    re.compile(r"\bscreen(ing)?\b", re.IGNORECASE),
    re.compile(r"\brole(s)?\b", re.IGNORECASE),
    re.compile(r"\bposition(s)?\b", re.IGNORECASE),
    re.compile(r"\bskill(s)?\b", re.IGNORECASE),
    re.compile(r"\bhi\b", re.IGNORECASE),
    re.compile(r"\bhello\b", re.IGNORECASE),
    re.compile(r"\bhey\b", re.IGNORECASE),
    re.compile(r"\bhelp\b", re.IGNORECASE),
)

# Words that suggest the message is describing a hiring / skills / role /
# evaluation context, used as a fallback signal for RECOMMEND.
_CONTEXT_KEYWORDS: tuple[Pattern[str], ...] = (
    re.compile(r"\bdeveloper(s)?\b", re.IGNORECASE),
    re.compile(r"\bengineer(s)?\b", re.IGNORECASE),
    re.compile(r"\bmanager(s)?\b", re.IGNORECASE),
    re.compile(r"\bsales\b", re.IGNORECASE),
    re.compile(r"\banalyst(s)?\b", re.IGNORECASE),
    re.compile(r"\bcustomer\s+service\b", re.IGNORECASE),
    re.compile(r"\bleadership\b", re.IGNORECASE),
    re.compile(r"\bskill(s)?\b", re.IGNORECASE),
    re.compile(r"\brole(s)?\b", re.IGNORECASE),
    re.compile(r"\bposition(s)?\b", re.IGNORECASE),
    re.compile(r"\bteam\b", re.IGNORECASE),
    re.compile(r"\bemployee(s)?\b", re.IGNORECASE),
    re.compile(r"\bapplicant(s)?\b", re.IGNORECASE),
    re.compile(r"\bcandidate(s)?\b", re.IGNORECASE),
    re.compile(r"\bjava\b", re.IGNORECASE),
    re.compile(r"\bpython\b", re.IGNORECASE),
    re.compile(r"\bcoding\b", re.IGNORECASE),
    re.compile(r"\bcognitive\b", re.IGNORECASE),
    re.compile(r"\bpersonality\b", re.IGNORECASE),
    re.compile(r"\bexperience\b", re.IGNORECASE),
    re.compile(r"\bjunior\b", re.IGNORECASE),
    re.compile(r"\bsenior\b", re.IGNORECASE),
    re.compile(r"\bgraduate(s)?\b", re.IGNORECASE),
    re.compile(r"\bentry[\s-]?level\b", re.IGNORECASE),
    re.compile(r"\bhire\b", re.IGNORECASE),
    re.compile(r"\bhiring\b", re.IGNORECASE),
    re.compile(r"\bevaluat", re.IGNORECASE),
)


# ==============================================================================
# VALIDATION
# ==============================================================================


def _validate_messages(messages: list[dict[str, str]]) -> list[_Message]:
    """Validate raw input messages and normalize them.

    Raises:
        ValidationError: if the message list or any message is malformed.
    """
    if not messages:
        raise ValidationError("messages list is empty.")

    if not isinstance(messages, list):
        raise ValidationError("messages must be a list.")

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

    if normalized[-1].role != "user":
        raise ValidationError("The final message must be from the 'user' role.")

    return normalized


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _matches_any(patterns: tuple[Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _count_words(text: str) -> int:
    return len(text.split())


def _has_prior_recommendations(history: list[_Message]) -> bool:
    """Determine whether an assistant message earlier in the conversation
    appears to contain recommendations.
    """
    recommendation_markers = (
        "recommend",
        "assessment",
        "here are",
        "suggested",
        "top pick",
        "based on",
    )
    for msg in history:
        if msg.role != "assistant":
            continue
        lowered = msg.content.lower()
        if any(marker in lowered for marker in recommendation_markers):
            return True
    return False


def _looks_informative_context(text: str) -> bool:
    """Fallback heuristic: does the message look like it's describing a
    hiring / skills / role / evaluation context, even without an explicit
    recommendation trigger phrase?
    """
    if _count_words(text) < _MIN_INFORMATIVE_WORDS:
        return False

    keyword_hits = sum(1 for pattern in _CONTEXT_KEYWORDS if pattern.search(text))
    return keyword_hits >= 1


def _looks_vague(text: str) -> bool:
    if _matches_any(_CLARIFY_PATTERNS, text):
        return True
    is_short = _count_words(text) <= _VAGUE_MAX_WORDS
    has_domain_hint = _matches_any(_DOMAIN_HINT_PATTERNS, text)
    return is_short and has_domain_hint and not _matches_any(_RECOMMEND_PATTERNS, text)


# ==============================================================================
# CLASSIFICATION STAGES (checked in order)
# ==============================================================================


def _check_refuse(text: str) -> Optional[IntentResult]:
    if _matches_any(_REFUSE_PATTERNS, text):
        return IntentResult(
            intent=ConversationIntent.REFUSE,
            confidence=0.95,
            reason="Message matched a refuse pattern (off-topic, unsafe, or prompt injection attempt).",
        )
    return None


def _check_compare(text: str) -> Optional[IntentResult]:
    if _matches_any(_COMPARE_PATTERNS, text):
        return IntentResult(
            intent=ConversationIntent.COMPARE,
            confidence=0.9,
            reason="Message matched a comparison pattern (e.g. 'vs', 'compare', 'difference between').",
        )
    return None


def _check_refine(text: str, history: list[_Message]) -> Optional[IntentResult]:
    if not _has_prior_recommendations(history):
        return None
    if _matches_any(_REFINE_PATTERNS, text):
        return IntentResult(
            intent=ConversationIntent.REFINE,
            confidence=0.85,
            reason="Prior recommendations exist and the message modifies them.",
        )
    return None


def _check_clarify(text: str) -> Optional[IntentResult]:
    if _looks_vague(text):
        return IntentResult(
            intent=ConversationIntent.CLARIFY,
            confidence=0.8,
            reason="Message is too vague to determine assessment requirements.",
        )
    return None


def _check_recommend(text: str) -> Optional[IntentResult]:
    if _matches_any(_RECOMMEND_PATTERNS, text):
        return IntentResult(
            intent=ConversationIntent.RECOMMEND,
            confidence=0.9,
            reason="Message matched an explicit recommendation-request pattern.",
        )

    if _looks_informative_context(text):
        return IntentResult(
            intent=ConversationIntent.RECOMMEND,
            confidence=0.60,
            reason="No explicit recommendation pattern matched, but message appears to describe "
            "a hiring/skills/role/evaluation context. Classified as RECOMMEND with reduced "
            "confidence to minimize false negatives.",
        )
    return None


# ==============================================================================
# PUBLIC API
# ==============================================================================


def detect_intent(messages: list[dict[str, str]]) -> IntentResult:
    """Classify the current conversational intent using deterministic,
    rule-based logic only.

    Args:
        messages: Full conversation history. Each item must be a dict with
            'role' and 'content' string keys. The last message must be from
            the 'user' role and determines the current intent; earlier
            messages provide context only.

    Returns:
        IntentResult describing the classified intent, a confidence score,
        and a human-readable reason.

    Raises:
        ValidationError: if the input message structure is invalid.
        IntentDetectionError: if classification fails unexpectedly.
    """
    try:
        normalized = _validate_messages(messages)
    except ValidationError:
        raise
    except Exception as exc:  # defensive: unexpected validation failure
        raise ValidationError(f"Failed to validate messages: {exc}") from exc

    current = normalized[-1]
    history = normalized[:-1]
    text = current.content

    try:
        for check in (
            lambda: _check_refuse(text),
            lambda: _check_compare(text),
            lambda: _check_refine(text, history),
            lambda: _check_clarify(text),
            lambda: _check_recommend(text),
        ):
            result = check()
            if result is not None:
                logger.debug("Intent classified as %s (confidence=%.2f)", result.intent, result.confidence)
                return result

        logger.debug("No pattern matched; defaulting to UNKNOWN.")
        return IntentResult(
            intent=ConversationIntent.UNKNOWN,
            confidence=0.5,
            reason="No deterministic pattern matched the message; intent could not be determined.",
        )
    except (ValidationError, IntentDetectionError):
        raise
    except Exception as exc:  # defensive: unexpected classification failure
        raise IntentDetectionError(f"Intent classification failed: {exc}") from exc
