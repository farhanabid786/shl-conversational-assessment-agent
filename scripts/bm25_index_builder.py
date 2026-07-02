
"""
scripts/bm25_index_builder.py

Phase 3: BM25 Index Builder
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

METADATA_PATH = Path("data/processed/catalog_metadata.json")
OUTPUT_PATH = Path("data/cache/bm25_index.pkl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("bm25_index_builder")


class BM25BuilderError(Exception):
    pass


class MetadataLoadError(BM25BuilderError):
    pass


class ValidationError(BM25BuilderError):
    pass


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise MetadataLoadError(f"Metadata file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise MetadataLoadError(f"Unable to load metadata: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise ValidationError("Metadata must be a non-empty JSON array.")
    return data


def validate_records(records: list[dict[str, Any]]) -> tuple[list[list[str]], list[str]]:
    seen: set[str] = set()
    corpus: list[list[str]] = []
    entity_ids: list[str] = []

    for idx, rec in enumerate(records):
        entity = str(rec.get("entity_id", "")).strip()
        if not entity:
            raise ValidationError(f"Missing entity_id at record {idx}")
        if entity in seen:
            raise ValidationError(f"Duplicate entity_id: {entity}")
        seen.add(entity)

        tokens = rec.get("ranking_tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ValidationError(f"Invalid ranking_tokens for entity_id {entity}")

        clean_tokens: list[str] = []
        for token in tokens:
            if not isinstance(token, str):
                raise ValidationError(f"Non-string token in entity_id {entity}")
            token = token.strip()
            if token:
                clean_tokens.append(token)

        if not clean_tokens:
            raise ValidationError(f"Empty ranking_tokens for entity_id {entity}")

        corpus.append(clean_tokens)
        entity_ids.append(entity)

    return corpus, entity_ids


def build_bm25(corpus: list[list[str]]) -> BM25Okapi:
    logger.info("Building BM25 index for %d documents...", len(corpus))
    return BM25Okapi(corpus)


def save_index(index: BM25Okapi, entity_ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bm25": index,
        "entity_ids": entity_ids,
        "corpus_size": len(entity_ids),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def run() -> None:
    records = load_metadata(METADATA_PATH)
    corpus, entity_ids = validate_records(records)
    bm25 = build_bm25(corpus)
    save_index(bm25, entity_ids, OUTPUT_PATH)
    logger.info("BM25 index created successfully (%d documents).", len(entity_ids))


def main() -> None:
    try:
        run()
    except BM25BuilderError as exc:
        logger.exception("BM25 index build failed.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
