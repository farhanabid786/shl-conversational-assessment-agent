
"""
scripts/faiss_index_builder.py

Phase 3: FAISS Index Builder
SHL Conversational Assessment Recommendation System

Builds a FAISS IndexFlatIP from generated embeddings.

Inputs
------
data/embeddings/catalog_embeddings.npy
data/embeddings/embedding_mapping.json

Output
------
data/faiss/catalog.index
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

EMBEDDINGS_PATH = Path("data/embeddings/catalog_embeddings.npy")
MAPPING_PATH = Path("data/embeddings/embedding_mapping.json")
INDEX_OUTPUT_PATH = Path("data/faiss/catalog.index")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("faiss_index_builder")


class FaissIndexBuilderError(Exception):
    """Base exception."""


class EmbeddingLoadError(FaissIndexBuilderError):
    pass


class MappingLoadError(FaissIndexBuilderError):
    pass


class ValidationError(FaissIndexBuilderError):
    pass


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise EmbeddingLoadError(f"Embeddings not found: {path}")

    try:
        emb = np.load(path)
    except Exception as exc:
        raise EmbeddingLoadError(f"Unable to load embeddings: {exc}") from exc

    emb = np.ascontiguousarray(emb, dtype=np.float32)
    return emb


def load_mapping(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise MappingLoadError(f"Mapping not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception as exc:
        raise MappingLoadError(f"Unable to load mapping: {exc}") from exc

    if not isinstance(mapping, list):
        raise MappingLoadError("Mapping must be a JSON array.")

    return mapping


def validate_mapping(mapping: list[dict[str, Any]]) -> None:
    required = {"row", "entity_id", "canonical_name"}

    seen_rows = set()
    seen_entities = set()

    for item in mapping:
        if not isinstance(item, dict):
            raise ValidationError("Every mapping record must be an object.")

        missing = required - item.keys()
        if missing:
            raise ValidationError(f"Missing keys: {sorted(missing)}")

        row = item["row"]
        entity = str(item["entity_id"])

        if row in seen_rows:
            raise ValidationError(f"Duplicate row detected: {row}")
        seen_rows.add(row)

        if entity in seen_entities:
            raise ValidationError(f"Duplicate entity_id detected: {entity}")
        seen_entities.add(entity)

    expected = list(range(len(mapping)))
    actual = sorted(item["row"] for item in mapping)

    if actual != expected:
        raise ValidationError("Mapping rows are not sequential.")


def validate_embeddings(
    embeddings: np.ndarray,
    mapping: list[dict[str, Any]],
) -> None:
    if embeddings.ndim != 2:
        raise ValidationError("Embeddings must be a 2D matrix.")

    if embeddings.shape[0] == 0:
        raise ValidationError("Embedding matrix is empty.")

    if embeddings.shape[0] != len(mapping):
        raise ValidationError(
            f"Embedding count ({embeddings.shape[0]}) does not match "
            f"mapping count ({len(mapping)})."
        )

    if np.isnan(embeddings).any():
        raise ValidationError("Embedding matrix contains NaN values.")

    if np.isinf(embeddings).any():
        raise ValidationError("Embedding matrix contains Inf values.")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    logger.info("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def save_index(index: faiss.IndexFlatIP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def run() -> None:
    logger.info("Loading embeddings...")
    embeddings = load_embeddings(EMBEDDINGS_PATH)

    logger.info("Loading mapping...")
    mapping = load_mapping(MAPPING_PATH)

    logger.info("Validating mapping...")
    validate_mapping(mapping)

    logger.info("Validating embeddings...")
    validate_embeddings(embeddings, mapping)

    logger.info("Building FAISS index...")
    index = build_index(embeddings)

    if index.ntotal != len(mapping):
        raise ValidationError("FAISS index size does not match mapping.")

    logger.info("Saving index...")
    save_index(index, INDEX_OUTPUT_PATH)

    logger.info(
        "Successfully indexed %d vectors (dimension=%d).",
        index.ntotal,
        index.d,
    )


def main() -> None:
    try:
        run()
    except FaissIndexBuilderError as exc:
        logger.exception("FAISS index build failed.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
