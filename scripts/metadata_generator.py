"""
scripts/metadata_generator.py

Phase 2: Metadata Generation
SHL Conversational Assessment Recommendation System

Reads data/processed/catalog_clean.json (read-only) and generates a
retrieval-optimized metadata layer for BM25 / FAISS at:
    data/processed/catalog_metadata.json

Output contract fields (added)
-------------------------------
Every generated record now also carries:

  * url        — copied verbatim from catalog_clean.json's ``link`` field.
                 This is the exact value app.routes serializes back to the
                 client as RecommendationItem.url — it is never invented,
                 rewritten, or reconstructed anywhere downstream.
  * test_type  — derived from catalog_clean.json's ``keys`` field (the
                 SHL category list, e.g. "Knowledge & Skills") by mapping
                 each category to its single-letter SHL test-type code
                 (see CATEGORY_TO_TEST_TYPE_CODE) and joining all codes
                 that apply, in SHL's canonical badge order
                 (A, B, C, D, E, K, P, S). A record tagged both
                 "Ability & Aptitude" and "Personality & Behavior"
                 produces test_type "AP".

Both fields are required (non-empty) for every record; generation fails
loudly via MetadataValidationError if either is missing, rather than
silently emitting a record the API layer would have to paper over with
an empty string at serve time.

Python 3.10.11
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

CLEAN_CATALOG_PATH = Path("data/processed/catalog_clean.json")
METADATA_OUTPUT_PATH = Path("data/processed/catalog_metadata.json")

METADATA_VERSION = "1.0"

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("metadata_generator")


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #

class MetadataGeneratorError(Exception):
    """Base exception for metadata generation failures."""


class CatalogLoadError(MetadataGeneratorError):
    """Raised when the cleaned catalog file cannot be read or parsed."""


class MetadataValidationError(MetadataGeneratorError):
    """Raised when a generated metadata record fails validation."""


class DuplicateEntityIdError(MetadataValidationError):
    """Raised when a duplicate entity_id is detected."""


class DuplicateCanonicalNameError(MetadataValidationError):
    """Raised when a duplicate canonical_name is detected."""


class EmptySearchableTextError(MetadataValidationError):
    """Raised when a record has empty searchable_text."""


class InvalidBooleanValueError(MetadataValidationError):
    """Raised when adaptive/remote fields are not valid booleans."""


class InvalidDurationError(MetadataValidationError):
    """Raised when duration_minutes is neither an integer nor None."""


class MissingUrlError(MetadataValidationError):
    """Raised when a record has no catalog URL ('link' in the source data)."""


class MissingTestTypeError(MetadataValidationError):
    """Raised when a record's 'keys' do not map to any known test-type code."""


# --------------------------------------------------------------------------- #
# Stop words
# --------------------------------------------------------------------------- #

STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "of",
        "to", "in", "on", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after", "above",
        "below", "from", "up", "down", "out", "off", "over", "under",
        "again", "further", "once", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "this", "that", "these", "those", "it", "its",
        "as", "such", "can", "will", "would", "should", "may", "might",
        "must", "shall", "not", "no", "nor", "so", "than", "too", "very",
        "s", "t", "just", "don", "now", "their", "they", "them", "he",
        "she", "his", "her", "you", "your", "we", "our", "i", "me", "my",
        "who", "whom", "which", "what", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "own", "same", "also", "each","minutes","assessments",
    }
)

# --------------------------------------------------------------------------- #
# SHL test-type taxonomy
# --------------------------------------------------------------------------- #

# Maps each SHL catalog category (as it appears verbatim in a cleaned
# record's 'keys' list) to its single-letter SHL test-type badge code.
# This is SHL's own published taxonomy — every value observed across the
# full 377-record catalog maps to exactly one of these eight categories;
# there is no "unknown category" fallback by design, so a category added
# to the source site that isn't in this table fails loudly (see
# derive_test_type) rather than silently producing an incomplete code.
CATEGORY_TO_TEST_TYPE_CODE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Canonical badge order used when a record maps to multiple categories,
# matching the order SHL itself displays multi-letter badges in.
_TEST_TYPE_CODE_ORDER: str = "ABCDEKPS"

# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class MetadataRecord:
    entity_id: str
    canonical_name: str
    normalized_name: str
    assessment_family: str
    url: str = ""
    test_type: str = ""
    keywords: list[str] = field(default_factory=list)
    job_levels: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    duration_minutes: Optional[int] = None
    adaptive: bool = False
    remote: bool = False
    searchable_text: str = ""
    filter_tokens: list[str] = field(default_factory=list)
    ranking_tokens: list[str] = field(default_factory=list)
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_clean_catalog(path: Path) -> list[dict[str, Any]]:
    """Load the cleaned catalog JSON file (read-only)."""
    logger.info("Loading cleaned catalog from %s", path)
    if not path.exists():
        raise CatalogLoadError(f"Cleaned catalog file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise CatalogLoadError(f"Failed to parse cleaned catalog JSON: {exc}") from exc
    except OSError as exc:
        raise CatalogLoadError(f"Failed to read cleaned catalog file: {exc}") from exc

    if not isinstance(data, list):
        raise CatalogLoadError("Cleaned catalog root must be a JSON array of records.")

    if not data:
        raise CatalogLoadError("Cleaned catalog is empty.")

    logger.info("Loaded %d cleaned catalog records", len(data))
    return data


# --------------------------------------------------------------------------- #
# Field-level helpers
# --------------------------------------------------------------------------- #

def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace in a name."""
    lowered = name.lower()
    no_punct = re.sub(r"[^\w\s]", " ", lowered)
    collapsed = re.sub(r"\s+", " ", no_punct).strip()
    return collapsed


def derive_assessment_family(keys: list[str]) -> str:
    """
    Derive the assessment family from the existing 'keys' field.
    Uses the primary (first) category listed; does not invent new categories.
    """
    if not keys:
        return ""
    return str(keys[0]).strip()


def derive_test_type(keys: list[str], entity_id: str = "") -> str:
    """Derive the SHL test-type code string from a record's category keys.

    Maps every category in *keys* to its single-letter code via
    CATEGORY_TO_TEST_TYPE_CODE, deduplicates, and joins the results in
    SHL's canonical badge order (A, B, C, D, E, K, P, S) — e.g. a record
    tagged ["Personality & Behavior", "Ability & Aptitude"] produces "AP",
    not "PA", regardless of the input order.

    Parameters
    ----------
    keys:
        The record's category list (catalog_clean.json's 'keys' field).
    entity_id:
        Used only to produce a more useful error message.

    Returns
    -------
    str
        Concatenated single-letter test-type codes, e.g. "K" or "AP".

    Raises
    ------
    MissingTestTypeError
        If *keys* is empty, or contains a category not present in
        CATEGORY_TO_TEST_TYPE_CODE (an unrecognized/new SHL category —
        fails loudly rather than silently dropping it).
    """
    if not keys:
        raise MissingTestTypeError(
            f"Cannot derive test_type for entity_id {entity_id!r}: "
            "'keys' is empty."
        )

    codes: set[str] = set()
    for key in keys:
        normalized_key = str(key).strip()
        code = CATEGORY_TO_TEST_TYPE_CODE.get(normalized_key)
        if code is None:
            raise MissingTestTypeError(
                f"Cannot derive test_type for entity_id {entity_id!r}: "
                f"unrecognized category {normalized_key!r} is not in "
                "CATEGORY_TO_TEST_TYPE_CODE. Add it to the taxonomy table "
                "if SHL has introduced a new category."
            )
        codes.add(code)

    ordered_codes = [c for c in _TEST_TYPE_CODE_ORDER if c in codes]
    return "".join(ordered_codes)


def extract_keywords(searchable_text: str) -> list[str]:
    """Extract unique, lowercase, stop-word-filtered keywords from text."""
    tokens = re.findall(r"[a-zA-Z0-9]+", searchable_text.lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token in STOP_WORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def build_filter_tokens(
    assessment_family: str,
    job_levels: list[str],
    languages: list[str],
    adaptive: bool,
    remote: bool,
) -> list[str]:
    """Combine family, job levels, languages, adaptive and remote into filter tokens."""
    raw_tokens: list[str] = []

    if assessment_family:
        raw_tokens.append(assessment_family.lower())

    raw_tokens.extend(level.lower() for level in job_levels if level)
    raw_tokens.extend(lang.lower() for lang in languages if lang)
    raw_tokens.append(f"adaptive:{str(adaptive).lower()}")
    raw_tokens.append(f"remote:{str(remote).lower()}")

    seen: set[str] = set()
    filter_tokens: list[str] = []
    for token in raw_tokens:
        if token not in seen:
            seen.add(token)
            filter_tokens.append(token)
    return filter_tokens


def build_ranking_tokens(normalized_name: str, keywords: list[str]) -> list[str]:
    """Combine normalized name tokens and keywords into ranking tokens."""
    name_tokens = normalized_name.split()
    raw_tokens = name_tokens + keywords

    seen: set[str] = set()
    ranking_tokens: list[str] = []
    for token in raw_tokens:
        if token not in seen:
            seen.add(token)
            ranking_tokens.append(token)
    return ranking_tokens


# --------------------------------------------------------------------------- #
# Record construction
# --------------------------------------------------------------------------- #

def build_metadata_record(record: dict[str, Any]) -> MetadataRecord:
    """Build a single MetadataRecord from a cleaned catalog record."""
    entity_id = str(record.get("entity_id", "")).strip()
    canonical_name = str(record.get("name", "")).strip()
    searchable_text = str(record.get("search_text", "")).strip()
    url = str(record.get("link", "")).strip()

    job_levels = list(record.get("job_levels") or [])
    languages = list(record.get("languages") or [])
    keys = list(record.get("keys") or [])

    adaptive = record.get("adaptive", False)
    remote = record.get("remote", False)
    duration_minutes = record.get("duration_minutes", None)

    normalized_name = normalize_name(canonical_name)
    assessment_family = derive_assessment_family(keys)
    test_type = derive_test_type(keys, entity_id=entity_id)
    keywords = extract_keywords(searchable_text)

    filter_tokens = build_filter_tokens(
        assessment_family=assessment_family,
        job_levels=job_levels,
        languages=languages,
        adaptive=bool(adaptive) if isinstance(adaptive, bool) else adaptive,
        remote=bool(remote) if isinstance(remote, bool) else remote,
    )
    ranking_tokens = build_ranking_tokens(normalized_name, keywords)

    return MetadataRecord(
        entity_id=entity_id,
        canonical_name=canonical_name,
        normalized_name=normalized_name,
        assessment_family=assessment_family,
        url=url,
        test_type=test_type,
        keywords=keywords,
        job_levels=job_levels,
        languages=languages,
        duration_minutes=duration_minutes,
        adaptive=adaptive,
        remote=remote,
        searchable_text=searchable_text,
        filter_tokens=filter_tokens,
        ranking_tokens=ranking_tokens,
        metadata_version=METADATA_VERSION,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_record(
    metadata: MetadataRecord,
    seen_entity_ids: set[str],
    seen_canonical_names: set[str],
) -> None:
    """Validate a single metadata record, raising descriptive exceptions."""
    if metadata.entity_id in seen_entity_ids:
        raise DuplicateEntityIdError(
            f"Duplicate entity_id detected: {metadata.entity_id!r}"
        )

    if metadata.canonical_name in seen_canonical_names:
        raise DuplicateCanonicalNameError(
            f"Duplicate canonical_name detected: {metadata.canonical_name!r}"
        )

    if not metadata.searchable_text:
        raise EmptySearchableTextError(
            f"Empty searchable_text for entity_id: {metadata.entity_id!r}"
        )

    if not metadata.url:
        raise MissingUrlError(
            f"Empty url (catalog 'link') for entity_id: {metadata.entity_id!r}. "
            "Every recommendation must carry a real, scraped catalog URL — "
            "never fabricated at serve time."
        )

    if not metadata.test_type:
        # Should be unreachable: derive_test_type() already raises
        # MissingTestTypeError before a record with no valid test_type
        # ever reaches this point. Kept as a defense-in-depth check in
        # case build_metadata_record is ever called with a pre-built
        # MetadataRecord that bypassed derive_test_type.
        raise MissingTestTypeError(
            f"Empty test_type for entity_id: {metadata.entity_id!r}."
        )

    if not isinstance(metadata.adaptive, bool):
        raise InvalidBooleanValueError(
            f"Invalid 'adaptive' value for entity_id {metadata.entity_id!r}: "
            f"{metadata.adaptive!r} (expected bool)"
        )

    if not isinstance(metadata.remote, bool):
        raise InvalidBooleanValueError(
            f"Invalid 'remote' value for entity_id {metadata.entity_id!r}: "
            f"{metadata.remote!r} (expected bool)"
        )

    duration = metadata.duration_minutes
    if duration is not None and not isinstance(duration, int):
        raise InvalidDurationError(
            f"Invalid duration_minutes for entity_id {metadata.entity_id!r}: "
            f"{duration!r} (expected int or null)"
        )
    if isinstance(duration, bool):
        # bool is a subclass of int; explicitly reject it.
        raise InvalidDurationError(
            f"Invalid duration_minutes for entity_id {metadata.entity_id!r}: "
            f"boolean value not allowed"
        )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def generate_metadata(records: list[dict[str, Any]]) -> list[MetadataRecord]:
    """Generate and validate metadata records for the full catalog."""
    logger.info("Starting metadata generation for %d records", len(records))

    metadata_records: list[MetadataRecord] = []
    seen_entity_ids: set[str] = set()
    seen_canonical_names: set[str] = set()

    for raw_record in records:
        metadata = build_metadata_record(raw_record)
        validate_record(metadata, seen_entity_ids, seen_canonical_names)

        seen_entity_ids.add(metadata.entity_id)
        seen_canonical_names.add(metadata.canonical_name)
        metadata_records.append(metadata)

    logger.info("Metadata generation complete: %d records", len(metadata_records))
    return metadata_records


# --------------------------------------------------------------------------- #
# Output writer
# --------------------------------------------------------------------------- #

def write_metadata_json(
    metadata_records: list[MetadataRecord], path: Path
) -> None:
    """Write the metadata records to a pretty-printed UTF-8 JSON file."""
    logger.info("Writing metadata JSON to %s", path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [record.to_dict() for record in metadata_records]
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise MetadataGeneratorError(f"Failed to write metadata JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run(
    clean_catalog_path: Path = CLEAN_CATALOG_PATH,
    metadata_output_path: Path = METADATA_OUTPUT_PATH,
) -> list[MetadataRecord]:
    """Execute the full metadata generation pipeline."""
    try:
        records = load_clean_catalog(clean_catalog_path)
        metadata_records = generate_metadata(records)
        write_metadata_json(metadata_records, metadata_output_path)
        logger.info("Metadata generation pipeline completed successfully")
        return metadata_records
    except MetadataGeneratorError:
        logger.exception("Metadata generation failed")
        raise
    except Exception:
        logger.exception("Unexpected error during metadata generation")
        raise


def main() -> None:
    try:
        run()
    except MetadataGeneratorError as exc:
        logger.error("Metadata generator terminated with error: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()