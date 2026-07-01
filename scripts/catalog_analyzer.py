"""
scripts/catalog_analyzer.py

Phase 1: Catalog Analysis
SHL Conversational Assessment Recommendation System

Reads data/raw/shl_catalog.json (read-only), analyzes it, and generates:
  1. docs/CATALOG_ANALYSIS.md
  2. data/processed/catalog_statistics.json

Python 3.10.11
"""

from __future__ import annotations

import json
import logging
import statistics
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

RAW_CATALOG_PATH = Path("data/raw/shl_catalog.json")
DOCS_OUTPUT_PATH = Path("docs/CATALOG_ANALYSIS.md")
STATS_OUTPUT_PATH = Path("data/processed/catalog_statistics.json")

EXPECTED_FIELDS: tuple[str, ...] = (
    "entity_id",
    "name",
    "link",
    "scraped_at",
    "job_levels",
    "job_levels_raw",
    "languages",
    "languages_raw",
    "duration",
    "duration_raw",
    "status",
    "remote",
    "adaptive",
    "description",
    "keys",
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("catalog_analyzer")


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #

class CatalogAnalyzerError(Exception):
    """Base exception for catalog analyzer failures."""


class CatalogLoadError(CatalogAnalyzerError):
    """Raised when the catalog file cannot be read or parsed."""


class CatalogValidationError(CatalogAnalyzerError):
    """Raised when the catalog does not have the expected structure."""


# --------------------------------------------------------------------------- #
# Data container
# --------------------------------------------------------------------------- #

@dataclass
class CatalogStatistics:
    total_assessments: int = 0
    available_fields: list[str] = field(default_factory=list)
    missing_fields_by_record: dict[str, list[str]] = field(default_factory=dict)
    field_presence_counts: dict[str, int] = field(default_factory=dict)

    duplicate_names: dict[str, int] = field(default_factory=dict)
    duplicate_urls: dict[str, int] = field(default_factory=dict)
    duplicate_ids: dict[str, int] = field(default_factory=dict)

    unique_languages: list[str] = field(default_factory=list)
    unique_job_levels: list[str] = field(default_factory=list)
    unique_categories: list[str] = field(default_factory=list)

    missing_descriptions_count: int = 0
    missing_urls_count: int = 0
    missing_languages_count: int = 0
    missing_durations_count: int = 0

    average_duration_minutes: Optional[float] = None
    duration_sample_size: int = 0

    adaptive_count: int = 0
    remote_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_assessments": self.total_assessments,
            "available_fields": self.available_fields,
            "field_presence_counts": self.field_presence_counts,
            "records_with_missing_fields": len(self.missing_fields_by_record),
            "missing_fields_by_record": self.missing_fields_by_record,
            "duplicates": {
                "duplicate_names": self.duplicate_names,
                "duplicate_urls": self.duplicate_urls,
                "duplicate_ids": self.duplicate_ids,
            },
            "unique_languages": {
                "count": len(self.unique_languages),
                "values": self.unique_languages,
            },
            "unique_job_levels": {
                "count": len(self.unique_job_levels),
                "values": self.unique_job_levels,
            },
            "unique_categories": {
                "count": len(self.unique_categories),
                "values": self.unique_categories,
            },
            "missing_data": {
                "missing_descriptions": self.missing_descriptions_count,
                "missing_urls": self.missing_urls_count,
                "missing_languages": self.missing_languages_count,
                "missing_durations": self.missing_durations_count,
            },
            "duration": {
                "average_duration_minutes": self.average_duration_minutes,
                "sample_size": self.duration_sample_size,
            },
            "adaptive_assessment_count": self.adaptive_count,
            "remote_assessment_count": self.remote_count,
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_catalog(path: Path) -> list[dict[str, Any]]:
    """Load and validate the raw SHL catalog JSON file."""
    logger.info("Loading catalog from %s", path)
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

    logger.info("Loaded %d catalog records", len(data))
    return data


# --------------------------------------------------------------------------- #
# Analysis helpers
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


def analyze_fields(
    records: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    """Determine available/missing fields across all records."""
    available_fields: set[str] = set()
    for record in records:
        available_fields.update(record.keys())

    field_presence_counts: dict[str, int] = {f: 0 for f in sorted(available_fields)}
    missing_fields_by_record: dict[str, list[str]] = {}

    for record in records:
        record_id = str(record.get("entity_id", "UNKNOWN"))
        missing_here: list[str] = []
        for expected_field in EXPECTED_FIELDS:
            value = record.get(expected_field, None)
            if expected_field not in record or is_blank(value):
                missing_here.append(expected_field)
            else:
                field_presence_counts[expected_field] = (
                    field_presence_counts.get(expected_field, 0) + 1
                )
        for other_field in record.keys():
            if other_field not in EXPECTED_FIELDS and not is_blank(record.get(other_field)):
                field_presence_counts[other_field] = (
                    field_presence_counts.get(other_field, 0) + 1
                )
        if missing_here:
            missing_fields_by_record[record_id] = missing_here

    return sorted(available_fields), missing_fields_by_record, field_presence_counts


def find_duplicates(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Return a mapping of duplicated values -> occurrence count for a given key."""
    counter: Counter[str] = Counter()
    for record in records:
        value = record.get(key)
        if is_blank(value):
            continue
        counter[str(value).strip()] += 1
    return {value: count for value, count in counter.items() if count > 1}


def collect_unique_values(records: list[dict[str, Any]], key: str) -> list[str]:
    """Collect unique values from a list-valued field across all records."""
    unique: set[str] = set()
    for record in records:
        values = record.get(key)
        if isinstance(values, list):
            for v in values:
                if not is_blank(v):
                    unique.add(str(v).strip())
    return sorted(unique)


def parse_duration_minutes(record: dict[str, Any]) -> Optional[float]:
    """
    Attempt to parse a numeric duration (in minutes) from a record's
    'duration' field. Returns None if not parseable (e.g. 'Untimed', 'Variable').
    """
    raw = record.get("duration")
    if is_blank(raw):
        return None
    match = re.search(r"\d+(\.\d+)?", str(raw))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def count_flag_yes(records: list[dict[str, Any]], key: str) -> int:
    """Count records where a yes/no field equals 'yes' (case-insensitive)."""
    count = 0
    for record in records:
        value = record.get(key)
        if isinstance(value, str) and value.strip().lower() == "yes":
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Main analysis orchestration
# --------------------------------------------------------------------------- #

def analyze_catalog(records: list[dict[str, Any]]) -> CatalogStatistics:
    """Run the full analysis pipeline and return a populated CatalogStatistics object."""
    logger.info("Starting catalog analysis")
    stats = CatalogStatistics()

    stats.total_assessments = len(records)

    (
        stats.available_fields,
        stats.missing_fields_by_record,
        stats.field_presence_counts,
    ) = analyze_fields(records)

    stats.duplicate_names = find_duplicates(records, "name")
    stats.duplicate_urls = find_duplicates(records, "link")
    stats.duplicate_ids = find_duplicates(records, "entity_id")

    stats.unique_languages = collect_unique_values(records, "languages")
    stats.unique_job_levels = collect_unique_values(records, "job_levels")
    stats.unique_categories = collect_unique_values(records, "keys")

    stats.missing_descriptions_count = sum(
        1 for r in records if is_blank(r.get("description"))
    )
    stats.missing_urls_count = sum(1 for r in records if is_blank(r.get("link")))
    stats.missing_languages_count = sum(
        1 for r in records if is_blank(r.get("languages"))
    )
    stats.missing_durations_count = sum(
        1 for r in records if is_blank(r.get("duration"))
    )

    durations = [
        d for d in (parse_duration_minutes(r) for r in records) if d is not None
    ]
    stats.duration_sample_size = len(durations)
    stats.average_duration_minutes = (
        round(statistics.mean(durations), 2) if durations else None
    )

    stats.adaptive_count = count_flag_yes(records, "adaptive")
    stats.remote_count = count_flag_yes(records, "remote")

    logger.info("Catalog analysis complete")
    return stats


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def write_statistics_json(stats: CatalogStatistics, path: Path) -> None:
    """Write the statistics dictionary to a JSON file."""
    logger.info("Writing statistics JSON to %s", path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(stats.to_dict(), fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise CatalogAnalyzerError(f"Failed to write statistics JSON: {exc}") from exc


def render_markdown_report(stats: CatalogStatistics) -> str:
    """Render the human-readable markdown analysis report."""
    lines: list[str] = []
    lines.append("# SHL Catalog Analysis")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Total assessments:** {stats.total_assessments}")
    lines.append(f"- **Available fields:** {', '.join(stats.available_fields)}")
    lines.append(
        f"- **Records with missing fields:** {len(stats.missing_fields_by_record)}"
    )
    lines.append("")

    lines.append("## Field Presence Counts")
    lines.append("")
    lines.append("| Field | Present In Records |")
    lines.append("|---|---|")
    for f, count in sorted(stats.field_presence_counts.items()):
        lines.append(f"| {f} | {count} |")
    lines.append("")

    lines.append("## Duplicates")
    lines.append("")
    lines.append(f"- **Duplicate names:** {len(stats.duplicate_names)}")
    lines.append(f"- **Duplicate URLs:** {len(stats.duplicate_urls)}")
    lines.append(f"- **Duplicate IDs:** {len(stats.duplicate_ids)}")
    if stats.duplicate_names:
        lines.append("")
        lines.append("### Duplicate Names")
        for name, count in sorted(
            stats.duplicate_names.items(), key=lambda x: -x[1]
        ):
            lines.append(f"- {name}: {count} occurrences")
    if stats.duplicate_urls:
        lines.append("")
        lines.append("### Duplicate URLs")
        for url, count in sorted(stats.duplicate_urls.items(), key=lambda x: -x[1]):
            lines.append(f"- {url}: {count} occurrences")
    if stats.duplicate_ids:
        lines.append("")
        lines.append("### Duplicate IDs")
        for rid, count in sorted(stats.duplicate_ids.items(), key=lambda x: -x[1]):
            lines.append(f"- {rid}: {count} occurrences")
    lines.append("")

    lines.append("## Unique Values")
    lines.append("")
    lines.append(f"- **Unique languages:** {len(stats.unique_languages)}")
    lines.append(f"- **Unique job levels:** {len(stats.unique_job_levels)}")
    lines.append(f"- **Unique assessment categories:** {len(stats.unique_categories)}")
    lines.append("")
    lines.append("### Job Levels")
    lines.append(", ".join(stats.unique_job_levels) or "_none found_")
    lines.append("")
    lines.append("### Categories")
    lines.append(", ".join(stats.unique_categories) or "_none found_")
    lines.append("")
    lines.append("### Languages")
    lines.append(", ".join(stats.unique_languages) or "_none found_")
    lines.append("")

    lines.append("## Missing Data")
    lines.append("")
    lines.append(f"- **Missing descriptions:** {stats.missing_descriptions_count}")
    lines.append(f"- **Missing URLs:** {stats.missing_urls_count}")
    lines.append(f"- **Missing languages:** {stats.missing_languages_count}")
    lines.append(f"- **Missing durations:** {stats.missing_durations_count}")
    lines.append("")

    lines.append("## Duration")
    lines.append("")
    if stats.average_duration_minutes is not None:
        lines.append(
            f"- **Average duration:** {stats.average_duration_minutes} minutes "
            f"(sample size: {stats.duration_sample_size})"
        )
    else:
        lines.append("- **Average duration:** not available (no parseable durations)")
    lines.append("")

    lines.append("## Assessment Flags")
    lines.append("")
    lines.append(f"- **Adaptive assessments:** {stats.adaptive_count}")
    lines.append(f"- **Remote assessments:** {stats.remote_count}")
    lines.append("")

    return "\n".join(lines)


def write_markdown_report(stats: CatalogStatistics, path: Path) -> None:
    """Write the rendered markdown report to disk."""
    logger.info("Writing markdown report to %s", path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        report = render_markdown_report(stats)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(report)
    except OSError as exc:
        raise CatalogAnalyzerError(f"Failed to write markdown report: {exc}") from exc


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run(
    raw_path: Path = RAW_CATALOG_PATH,
    docs_path: Path = DOCS_OUTPUT_PATH,
    stats_path: Path = STATS_OUTPUT_PATH,
) -> CatalogStatistics:
    """Execute the full catalog analysis pipeline."""
    try:
        records = load_catalog(raw_path)
        stats = analyze_catalog(records)
        write_statistics_json(stats, stats_path)
        write_markdown_report(stats, docs_path)
        logger.info("Catalog analysis pipeline completed successfully")
        return stats
    except CatalogAnalyzerError:
        logger.exception("Catalog analysis failed")
        raise
    except Exception:
        logger.exception("Unexpected error during catalog analysis")
        raise


def main() -> None:
    try:
        run()
    except CatalogAnalyzerError as exc:
        logger.error("Catalog analyzer terminated with error: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()