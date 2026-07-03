from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class ConversationStateError(Exception):
    """Base conversation state exception."""


class ValidationError(ConversationStateError):
    """Raised when conversation state validation fails."""


VALID_ROLES = {"user", "assistant", "system"}

MAX_ROLE_WORDS = 4
MAX_ROLE_CHARS = 40

ROLE_LEXICON: tuple[str, ...] = (
    "software engineer",
    "software developer",
    "backend developer",
    "frontend developer",
    "full stack developer",
    "data scientist",
    "data analyst",
    "data engineer",
    "business analyst",
    "project manager",
    "product manager",
    "program manager",
    "sales representative",
    "sales manager",
    "sales executive",
    "customer service representative",
    "customer support representative",
    "administrative assistant",
    "executive assistant",
    "hr manager",
    "human resources manager",
    "recruiter",
    "financial analyst",
    "accountant",
    "marketing manager",
    "marketing specialist",
    "operations manager",
    "quality analyst",
    "quality assurance engineer",
    "network engineer",
    "systems engineer",
    "systems administrator",
    "database administrator",
    "cybersecurity analyst",
    "cyber security analyst",
    "graphic designer",
    "ux designer",
    "ui designer",
    "content writer",
    "technical writer",
    "team lead",
    "team leader",
    "supervisor",
    "store manager",
    "branch manager",
    "call center agent",
    "customer success manager",
    "solutions architect",
    "devops engineer",
    "machine learning engineer",
    "java developer",
    "python developer",
    "web developer",
)

_ROLE_LEXICON_SORTED = sorted(ROLE_LEXICON, key=len, reverse=True)

ROLE_TRIGGER_PATTERN = re.compile(
    r"\b(?:hiring for|hiring|looking for)\b\s+(?:a|an)?\s*",
    re.IGNORECASE,
)

_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-]*")

SKILL_LEXICON: dict[str, str] = {
    "python": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "sql": "SQL",
    "excel": "Excel",
    "communication": "Communication",
    "leadership": "Leadership",
    "teamwork": "Teamwork",
    "problem solving": "Problem Solving",
    "critical thinking": "Critical Thinking",
    "sales": "Sales",
    "negotiation": "Negotiation",
    "customer service": "Customer Service",
    "project management": "Project Management",
    "data analysis": "Data Analysis",
    "coding": "Coding",
    "programming": "Programming",
    "numerical reasoning": "Numerical Reasoning",
    "verbal reasoning": "Verbal Reasoning",
    "logical reasoning": "Logical Reasoning",
    "attention to detail": "Attention to Detail",
    "time management": "Time Management",
    "collaboration": "Collaboration",
    "decision making": "Decision Making",
    "c++": "C++",
    "c#": "C#",
    "html": "HTML",
    "css": "CSS",
    "cloud": "Cloud",
    "aws": "AWS",
    "typing": "Typing",
    "data entry": "Data Entry",
    "financial acumen": "Financial Acumen",
    "accounting": "Accounting",
}

ASSESSMENT_FAMILY_LEXICON: dict[str, str] = {
    "coding": "Coding",
    "programming": "Coding",
    "cognitive": "Cognitive",
    "aptitude": "Cognitive",
    "personality": "Personality",
    "behavioral": "Behavioral",
    "behaviour": "Behavioral",
    "situational judgment": "Situational Judgment",
    "situational judgement": "Situational Judgment",
    "sjt": "Situational Judgment",
    "skills": "Skills",
    "skill": "Skills",
}

JOB_LEVEL_LEXICON: dict[str, str] = {
    "intern": "Intern",
    "internship": "Intern",
    "graduate": "Graduate",
    "entry level": "Graduate",
    "entry-level": "Graduate",
    "junior": "Junior",
    "mid level": "Mid",
    "mid-level": "Mid",
    "mid": "Mid",
    "senior": "Senior",
    "manager": "Manager",
    "managerial": "Manager",
    "executive": "Executive",
    "director": "Director",
}

LANGUAGE_LEXICON: dict[str, str] = {
    "english": "English",
    "french": "French",
    "german": "German",
    "spanish": "Spanish",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "dutch": "Dutch",
    "chinese": "Chinese",
    "japanese": "Japanese",
    "korean": "Korean",
    "russian": "Russian",
    "arabic": "Arabic",
    "polish": "Polish",
    "turkish": "Turkish",
    "vietnamese": "Vietnamese",
    "thai": "Thai",
    "swedish": "Swedish",
    "danish": "Danish",
    "norwegian": "Norwegian",
    "finnish": "Finnish",
    "greek": "Greek",
    "czech": "Czech",
    "romanian": "Romanian",
    "hungarian": "Hungarian",
    "malay": "Malay",
    "latvian": "Latvian",
    "lithuanian": "Lithuanian",
    "estonian": "Estonian",
    "slovak": "Slovak",
    "serbian": "Serbian",
    "icelandic": "Icelandic",
}

DURATION_PATTERN = re.compile(
    r"\b(?:under|within|less than|no more than|up to)?\s*"
    r"(\d{1,3})\s*(?:min(?:ute)?s?|mins?)\b",
    re.IGNORECASE,
)

REMOTE_TRUE_PATTERN = re.compile(r"\b(remote|online|virtual)\b", re.IGNORECASE)
REMOTE_FALSE_PATTERN = re.compile(r"\b(in-person|in person|onsite|on-site)\b", re.IGNORECASE)

ADAPTIVE_TRUE_PATTERN = re.compile(r"\bnon[- ]?adaptive\b", re.IGNORECASE)
ADAPTIVE_SIMPLE_PATTERN = re.compile(r"\badaptive\b", re.IGNORECASE)

COMPARISON_TRIGGER_PATTERN = re.compile(
    r"\b(?:compare|comparison|vs\.?|versus)\b", re.IGNORECASE
)

PREV_REC_LINE_PATTERN = re.compile(
    r"^\s*(?:\d+[.)]|[-*])\s*(.+?)(?:\s*[-:–]\s+.*)?$"
)

QUOTED_NAME_PATTERN = re.compile(r'"([^"]{2,80})"')


@dataclass(frozen=True)
class ConversationState:
    target_roles: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    assessment_family: str | None = None
    job_levels: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    max_duration: int | None = None
    adaptive: bool | None = None
    remote: bool | None = None
    comparison_targets: list[str] = field(default_factory=list)
    previous_recommendations: list[str] = field(default_factory=list)
    clarification_required: bool = False
    missing_fields: list[str] = field(default_factory=list)
    conversation_summary: str = ""


def _validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValidationError("messages must be a non-empty list.")

    validated: list[dict[str, str]] = []

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValidationError(f"Message at index {idx} must be a dictionary.")

        if "role" not in msg:
            raise ValidationError(f"Message at index {idx} is missing 'role'.")

        if "content" not in msg:
            raise ValidationError(f"Message at index {idx} is missing 'content'.")

        role = msg["role"]
        content = msg["content"]

        if not isinstance(role, str) or role not in VALID_ROLES:
            raise ValidationError(f"Message at index {idx} has invalid role: {role!r}.")

        if not isinstance(content, str) or not content.strip():
            raise ValidationError(f"Message at index {idx} has empty or invalid content.")

        validated.append({"role": role, "content": content})

    return validated


def _user_texts(messages: list[dict[str, str]]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def _all_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages)


def _title_case_role(text: str) -> str:
    return " ".join(w.capitalize() for w in text.split())


def _extract_roles_from_lexicon(text_lower: str) -> list[str]:
    found: list[str] = []
    for phrase in _ROLE_LEXICON_SORTED:
        if phrase in text_lower:
            found.append(_title_case_role(phrase))
    return found


_ROLE_STOPWORDS = {
    "a",
    "an",
    "the",
    "role",
    "position",
    "candidate",
    "who",
    "with",
    "that",
    "for",
    "our",
    "strong",
    "great",
    "good",
    "experienced",
}


def _extract_roles_from_triggers(text: str) -> list[str]:
    found: list[str] = []

    for match in ROLE_TRIGGER_PATTERN.finditer(text):
        window = text[match.end(): match.end() + 60]
        words = _WORD_PATTERN.findall(window)

        candidate_words: list[str] = []
        for word in words:
            if word.lower() in _ROLE_STOPWORDS:
                if candidate_words:
                    break
                continue
            candidate_words.append(word)
            if len(candidate_words) >= MAX_ROLE_WORDS:
                break

        if not candidate_words:
            continue

        candidate = " ".join(candidate_words)
        candidate = candidate[:MAX_ROLE_CHARS].strip()

        if candidate:
            found.append(_title_case_role(candidate))

    return found


def _extract_target_roles(text: str) -> list[str]:
    text_lower = text.lower()

    roles = _extract_roles_from_lexicon(text_lower)
    roles.extend(_extract_roles_from_triggers(text))

    deduped: list[str] = []
    seen: set[str] = set()

    for role in roles:
        key = role.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(role)

    return deduped


def _extract_skills(text_lower: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    for keyword, normalized in SKILL_LEXICON.items():
        if keyword in text_lower and normalized not in seen:
            seen.add(normalized)
            found.append(normalized)

    return found


def _extract_assessment_family(text_lower: str) -> str | None:
    for keyword, normalized in ASSESSMENT_FAMILY_LEXICON.items():
        if keyword in text_lower:
            return normalized
    return None


def _extract_job_levels(text_lower: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    for keyword, normalized in JOB_LEVEL_LEXICON.items():
        if keyword in text_lower and normalized not in seen:
            seen.add(normalized)
            found.append(normalized)

    return found


def _extract_languages(text_lower: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    for keyword, normalized in LANGUAGE_LEXICON.items():
        if keyword in text_lower and normalized not in seen:
            seen.add(normalized)
            found.append(normalized)

    return found


def _extract_max_duration(text: str) -> int | None:
    best: int | None = None

    for match in DURATION_PATTERN.finditer(text):
        raw_value = match.group(1)

        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            logger.warning("Malformed duration value encountered: %r", raw_value)
            continue

        if value <= 0 or value > 480:
            logger.warning("Out-of-range duration value ignored: %d", value)
            continue

        if best is None or value < best:
            best = value

    return best


def _extract_remote(text: str) -> bool | None:
    has_false = bool(REMOTE_FALSE_PATTERN.search(text))
    has_true = bool(REMOTE_TRUE_PATTERN.search(text))

    if has_false and not has_true:
        return False
    if has_true and not has_false:
        return True
    if has_true and has_false:
        return None
    return None


def _extract_adaptive(text: str) -> bool | None:
    if ADAPTIVE_TRUE_PATTERN.search(text):
        return False
    if ADAPTIVE_SIMPLE_PATTERN.search(text):
        return True
    return None


def _extract_comparison_targets(text: str) -> list[str]:
    if not COMPARISON_TRIGGER_PATTERN.search(text):
        return []

    found: list[str] = []
    seen: set[str] = set()

    for match in QUOTED_NAME_PATTERN.finditer(text):
        name = match.group(1).strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            found.append(name)

    return found


def _extract_previous_recommendations(messages: list[dict[str, str]]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    for msg in messages:
        if msg["role"] != "assistant":
            continue

        for line in msg["content"].splitlines():
            match = PREV_REC_LINE_PATTERN.match(line)

            if not match:
                continue

            name = match.group(1).strip()

            if not name:
                continue

            key = name.lower()
            if key not in seen:
                seen.add(key)
                found.append(name)

    return found


def _determine_clarification(
    target_roles: list[str],
    skills: list[str],
    assessment_family: str | None,
) -> tuple[bool, list[str]]:
    missing_fields: list[str] = []

    has_role = bool(target_roles)
    has_skills = bool(skills)
    has_family = bool(assessment_family)

    if not has_role:
        missing_fields.append("role")

    if not has_skills and not has_family:
        missing_fields.append("skills")

    clarification_required = not has_role or (not has_skills and not has_family)

    return clarification_required, missing_fields


def _build_summary(
    target_roles: list[str],
    skills: list[str],
    assessment_family: str | None,
    job_levels: list[str],
    languages: list[str],
    max_duration: int | None,
    adaptive: bool | None,
    remote: bool | None,
) -> str:
    parts: list[str] = []

    if target_roles:
        parts.append(f"Role(s): {', '.join(target_roles)}")

    if skills:
        parts.append(f"Skills: {', '.join(skills)}")

    if assessment_family:
        parts.append(f"Assessment family: {assessment_family}")

    if job_levels:
        parts.append(f"Job level(s): {', '.join(job_levels)}")

    if languages:
        parts.append(f"Language(s): {', '.join(languages)}")

    if max_duration is not None:
        parts.append(f"Max duration: {max_duration} minutes")

    if adaptive is not None:
        parts.append(f"Adaptive: {'yes' if adaptive else 'no'}")

    if remote is not None:
        parts.append(f"Remote: {'yes' if remote else 'no'}")

    if not parts:
        return "No structured hiring context has been captured yet."

    return "; ".join(parts) + "."


def parse_conversation(messages: list[dict[str, str]]) -> ConversationState:
    validated = _validate_messages(messages)

    user_only_text = "\n".join(_user_texts(validated))
    full_text = _all_text(validated)
    full_text_lower = full_text.lower()

    target_roles = _extract_target_roles(user_only_text)
    skills = _extract_skills(full_text_lower)
    assessment_family = _extract_assessment_family(full_text_lower)
    job_levels = _extract_job_levels(full_text_lower)
    languages = _extract_languages(full_text_lower)
    max_duration = _extract_max_duration(full_text)
    adaptive = _extract_adaptive(full_text)
    remote = _extract_remote(full_text)
    comparison_targets = _extract_comparison_targets(full_text)
    previous_recommendations = _extract_previous_recommendations(validated)

    clarification_required, missing_fields = _determine_clarification(
        target_roles, skills, assessment_family
    )

    summary = _build_summary(
        target_roles,
        skills,
        assessment_family,
        job_levels,
        languages,
        max_duration,
        adaptive,
        remote,
    )

    state = ConversationState(
        target_roles=target_roles,
        skills=skills,
        assessment_family=assessment_family,
        job_levels=job_levels,
        languages=languages,
        max_duration=max_duration,
        adaptive=adaptive,
        remote=remote,
        comparison_targets=comparison_targets,
        previous_recommendations=previous_recommendations,
        clarification_required=clarification_required,
        missing_fields=missing_fields,
        conversation_summary=summary,
    )

    logger.info(
        "Conversation parsed: roles=%d skills=%d family=%s clarification_required=%s",
        len(target_roles),
        len(skills),
        assessment_family,
        clarification_required,
    )

    return state
