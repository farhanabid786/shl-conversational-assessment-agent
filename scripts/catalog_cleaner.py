"""
scripts/catalog_cleaner.py

Phase 1: Catalog Cleaning
SHL Conversational Assessment Recommendation System

Reads data/raw/shl_catalog.json (read-only), cleans and normalizes it, and
writes the result to data/processed/catalog_clean.json.

Python 3.10.11
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

RAW_CATALOG_PATH = Path("data/raw/shl_catalog.json")
CLEAN_OUTPUT_PATH = Path("data/processed/catalog_clean.json")

# Fields to retain in the cleaned record (order preserved in output)
KEEP_FIELDS: tuple[str, ...] = (
    "entity_id",
    "name",
    "description",
    "link",
    "job_levels",
    "languages",
    "duration",
    "adaptive",
    "remote",
    "keys",
)

# Fields explicitly dropped from the raw record
DROP_FIELDS: tuple[str, ...] = (
    "scraped_at",
    "status",
    "duration_raw",
    "languages_raw",
    "job_levels_raw",
)

# Duration values that should map to null / not-a-number rather than 0
_NON_NUMERIC_DURATION_TOKENS: tuple[str, ...] = (
    "variable",
    "untimed",
    "tbc",
    "n/a",
    "-",
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("catalog_cleaner")


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #

class CatalogCleanerError(Exception):
    """Base exception for catalog cleaner failures."""


class CatalogLoadError(CatalogCleanerError):
    """Raised when the catalog file cannot be read or parsed."""


class CatalogValidationError(CatalogCleanerError):
    """Raised when the catalog does not have the expected structure."""


class CatalogWriteError(CatalogCleanerError):
    """Raised when the cleaned catalog cannot be written to disk."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_catalog(path: Path) -> list[dict[str, Any]]:
    """Load and validate the raw SHL catalog JSON file.

    The original file is opened in read-only mode and is never modified.
    """
    logger.info("Loading raw catalog from %s", path)
    if not path.exists():
        raise CatalogLoadError(f"Catalog file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise CatalogLoadError(f"Failed to parse catalog JSON: {exc}") from exc
    except OSError as exc:
        raise CatalogLoadError(f"Failed to read catalog file: {exc}") from exc

    if not isinstance(data, list):
        raise CatalogValidationError("Catalog root must be a JSON array of records.")

    if not data:
        raise CatalogValidationError("Catalog is empty.")

    logger.info("Loaded %d raw catalog records", len(data))
    return data


# --------------------------------------------------------------------------- #
# Field-level cleaning helpers
# --------------------------------------------------------------------------- #

def is_blank(value: Any) -> bool:
    """Return True if a value should be treated as missing/empty."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def to_bool_flag(value: Any) -> bool:
    """Convert a yes/no style string field into a boolean.

    Any value that is not case-insensitively equal to "yes" is treated as
    False (this includes missing/blank values).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "yes"
    return False


def parse_duration_minutes(duration: Any) -> Optional[int]:
    """Parse a numeric duration (in minutes) out of a free-form duration string.

    Examples:
        "30 minutes" -> 30
        "Approximate Completion Time in minutes = 30" -> 30
        "max 60" -> 60
        "Variable" -> None
        "" / None -> None
    """
    if is_blank(duration):
        return None

    text = str(duration).strip()

    if text.lower() in _NON_NUMERIC_DURATION_TOKENS:
        return None

    match = re.search(r"\d+(\.\d+)?", text)
    if not match:
        return None

    try:
        return int(float(match.group()))
    except ValueError:
        return None


def normalize_list_field(value: Any) -> list[str]:
    """Normalize a list-valued field (job_levels, languages, keys) to a
    clean list of stripped strings, dropping blank entries."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if not is_blank(item)]


def build_search_text(record: dict[str, Any], duration_raw_value: Any) -> str:
    """Concatenate the searchable fields of a cleaned record into one string.

    Concatenates: name, description, keys, job_levels, languages, duration.
    """
    parts: list[str] = [
        str(record.get("name", "")),
        str(record.get("description", "")),
        " ".join(record.get("keys", [])),
        " ".join(record.get("job_levels", [])),
        " ".join(record.get("languages", [])),
        str(duration_raw_value) if not is_blank(duration_raw_value) else "",
    ]
    search_text = " ".join(part.strip() for part in parts if part.strip())
    return re.sub(r"\s+", " ", search_text).strip()


# --------------------------------------------------------------------------- #
# Record-level cleaning
# --------------------------------------------------------------------------- #

def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Clean and normalize a single raw catalog record."""
    duration_value = record.get("duration")

    cleaned: dict[str, Any] = {
        "entity_id": str(record.get("entity_id", "")).strip(),
        "name": str(record.get("name", "")).strip(),
        "description": str(record.get("description", "")).strip(),
        "link": str(record.get("link", "")).strip(),
        "job_levels": normalize_list_field(record.get("job_levels")),
        "languages": normalize_list_field(record.get("languages")),
        "duration": duration_value if not is_blank(duration_value) else None,
        "duration_minutes": parse_duration_minutes(duration_value),
        "adaptive": to_bool_flag(record.get("adaptive")),
        "remote": to_bool_flag(record.get("remote")),
        "keys": normalize_list_field(record.get("keys")),
    }

    cleaned["search_text"] = build_search_text(cleaned, cleaned["duration"])

    return cleaned


def clean_catalog(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the cleaning pipeline over every record in the catalog."""
    logger.info("Starting catalog cleaning")
    cleaned_records: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        try:
            cleaned_records.append(clean_record(record))
        except Exception:
            entity_id = record.get("entity_id", f"index={index}")
            logger.exception("Failed to clean record %s; skipping", entity_id)
            continue

    logger.info(
        "Catalog cleaning complete: %d/%d records cleaned successfully",
        len(cleaned_records),
        len(records),
    )
    return cleaned_records


# --------------------------------------------------------------------------- #
# Output writer
# --------------------------------------------------------------------------- #

def write_clean_catalog(records: list[dict[str, Any]], path: Path) -> None:
    """Write the cleaned catalog to disk as JSON."""
    logger.info("Writing cleaned catalog to %s", path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise CatalogWriteError(f"Failed to write cleaned catalog: {exc}") from exc


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run(
    raw_path: Path = RAW_CATALOG_PATH,
    clean_path: Path = CLEAN_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    """Execute the full catalog cleaning pipeline."""
    try:
        records = load_catalog(raw_path)
        cleaned_records = clean_catalog(records)
        write_clean_catalog(cleaned_records, clean_path)
        logger.info("Catalog cleaning pipeline completed successfully")
        return cleaned_records
    except CatalogCleanerError:
        logger.exception("Catalog cleaning failed")
        raise
    except Exception:
        logger.exception("Unexpected error during catalog cleaning")
        raise


def main() -> None:
    try:
        run()
    except CatalogCleanerError as exc:
        logger.error("Catalog cleaner terminated with error: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()