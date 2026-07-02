
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class MetadataFilterError(Exception):
    """Base metadata filtering exception."""


class ValidationError(MetadataFilterError):
    """Raised when metadata filtering validation fails."""


@dataclass(frozen=True)
class FilterCriteria:
    adaptive: bool | None = None
    remote: bool | None = None
    max_duration: int | None = None
    languages: list[str] | None = None
    job_levels: list[str] | None = None
    assessment_family: str | None = None
    keywords: list[str] | None = None


@dataclass(frozen=True)
class FilteredCandidate:
    entity_id: str
    canonical_name: str
    rrf_score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FilterResult:
    candidates: list[FilteredCandidate]

    @property
    def entity_ids(self) -> list[str]:
        return [c.entity_id for c in self.candidates]


def _norm(values: list[str] | None) -> set[str]:
    return {
        str(v).strip().lower()
        for v in (values or [])
        if isinstance(v, str) and v.strip()
    }


def _matches(
    meta: dict[str, Any],
    criteria: FilterCriteria,
    languages: set[str],
    job_levels: set[str],
    keywords: set[str],
) -> bool:

    if criteria.adaptive is not None and bool(meta.get("adaptive")) != criteria.adaptive:
        return False

    if criteria.remote is not None and bool(meta.get("remote")) != criteria.remote:
        return False

    if criteria.max_duration is not None:
        raw_duration = meta.get("duration_minutes")

        if raw_duration is None:
            return False

        try:
            duration = int(raw_duration)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid duration_minutes for entity %s",
                meta.get("entity_id", "<unknown>"),
            )
            return False

        if duration > criteria.max_duration:
            return False

    if criteria.assessment_family:
        if (
            str(meta.get("assessment_family", "")).lower()
            != criteria.assessment_family.lower()
        ):
            return False

    if languages:
        if not (_norm(meta.get("languages")) & languages):
            return False

    if job_levels:
        if not (_norm(meta.get("job_levels")) & job_levels):
            return False

    if keywords:
        if not (_norm(meta.get("keywords")) & keywords):
            return False

    return True


def filter_candidates(
    fusion_result: Any,
    entity_lookup: dict[str, dict[str, Any]],
    criteria: FilterCriteria | None = None,
) -> FilterResult:

    if criteria is None:
        criteria = FilterCriteria()

    if fusion_result is None or not hasattr(fusion_result, "candidates"):
        raise ValidationError("Invalid FusionResult.")

    if not isinstance(entity_lookup, dict) or not entity_lookup:
        raise ValidationError("entity_lookup must be a non-empty dictionary.")

    norm_languages = _norm(criteria.languages)
    norm_job_levels = _norm(criteria.job_levels)
    norm_keywords = _norm(criteria.keywords)

    seen: set[str] = set()
    filtered: list[FilteredCandidate] = []

    for candidate in fusion_result.candidates:
        entity_id = candidate.entity_id

        if entity_id in seen:
            raise ValidationError(f"Duplicate entity_id: {entity_id}")

        seen.add(entity_id)

        metadata = entity_lookup.get(entity_id)

        if metadata is None:
            logger.warning("Metadata missing for %s", entity_id)
            continue

        if not isinstance(metadata, dict):
            raise ValidationError(
                f"Metadata for entity_id '{entity_id}' must be a dictionary."
            )

        if _matches(
            metadata,
            criteria,
            norm_languages,
            norm_job_levels,
            norm_keywords,
        ):
            filtered.append(
                FilteredCandidate(
                    entity_id=entity_id,
                    canonical_name=candidate.canonical_name,
                    rrf_score=float(candidate.rrf_score),
                    metadata=metadata,
                )
            )

    logger.info(
        "Metadata filter: %d -> %d candidates",
        len(fusion_result.candidates),
        len(filtered),
    )

    return FilterResult(filtered)
