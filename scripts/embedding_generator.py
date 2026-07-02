"""
scripts/embedding_generator.py

Phase 3: Embedding Generation
SHL Conversational Assessment Recommendation System

Reads data/processed/catalog_metadata.json (read-only) and generates
sentence embeddings from the 'searchable_text' field of each record using
sentence-transformers/all-MiniLM-L6-v2.

Outputs:
  1. data/embeddings/catalog_embeddings.npy   (float32 numpy array)
  2. data/embeddings/embedding_mapping.json   (row -> entity_id / canonical_name)

Do NOT modify catalog_clean.json or catalog_metadata.json. They are frozen.

Python 3.10.11
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

METADATA_INPUT_PATH = Path("data/processed/catalog_metadata.json")
EMBEDDINGS_OUTPUT_PATH = Path("data/embeddings/catalog_embeddings.npy")
MAPPING_OUTPUT_PATH = Path("data/embeddings/embedding_mapping.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("embedding_generator")


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #

class EmbeddingGeneratorError(Exception):
    """Base exception for embedding generator failures."""


class MetadataLoadError(EmbeddingGeneratorError):
    """Raised when the metadata file cannot be read or parsed."""


class MetadataValidationError(EmbeddingGeneratorError):
    """Raised when the metadata does not have the expected structure."""


class MissingSearchableTextError(MetadataValidationError):
    """Raised when a record is missing searchable_text."""


class DuplicateEntityIdError(MetadataValidationError):
    """Raised when a duplicate entity_id is detected."""


class ModelLoadError(EmbeddingGeneratorError):
    """Raised when the SentenceTransformer model fails to load."""


class EmbeddingDimensionError(EmbeddingGeneratorError):
    """Raised when generated embeddings do not match the expected dimension."""


class EmbeddingWriteError(EmbeddingGeneratorError):
    """Raised when embeddings or mapping cannot be written to disk."""


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MappingEntry:
    row: int
    entity_id: str
    canonical_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_metadata(path: Path) -> list[dict[str, Any]]:
    """Load the catalog metadata JSON file (read-only)."""
    logger.info("Loading catalog metadata from %s", path)
    if not path.exists():
        raise MetadataLoadError(f"Metadata file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise MetadataLoadError(f"Failed to parse metadata JSON: {exc}") from exc
    except OSError as exc:
        raise MetadataLoadError(f"Failed to read metadata file: {exc}") from exc

    if not isinstance(data, list):
        raise MetadataValidationError("Metadata root must be a JSON array of records.")

    if not data:
        raise MetadataValidationError("Metadata file is empty.")

    logger.info("Loaded %d metadata records", len(data))
    return data


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_metadata_records(records: list[dict[str, Any]]) -> None:
    """Validate that every record has a non-empty searchable_text and a
    unique entity_id, preserving deterministic ordering as given."""
    logger.info("Validating metadata records")
    seen_entity_ids: set[str] = set()

    for index, record in enumerate(records):
        entity_id = str(record.get("entity_id", "")).strip()
        if not entity_id:
            raise MetadataValidationError(
                f"Record at index {index} is missing a valid entity_id."
            )

        if entity_id in seen_entity_ids:
            raise DuplicateEntityIdError(
                f"Duplicate entity_id detected at index {index}: {entity_id!r}"
            )
        seen_entity_ids.add(entity_id)

        searchable_text = record.get("searchable_text")
        if not isinstance(searchable_text, str) or not searchable_text.strip():
            raise MissingSearchableTextError(
                f"Record at index {index} (entity_id={entity_id!r}) is missing "
                f"a valid 'searchable_text' field."
            )

    logger.info("Validation passed for %d records", len(records))


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #

def extract_searchable_texts(records: list[dict[str, Any]]) -> list[str]:
    """Extract the searchable_text field from each record, in order.

    Only 'searchable_text' is used. Raw catalog fields are never touched.
    """
    return [str(record["searchable_text"]).strip() for record in records]


def build_mapping(records: list[dict[str, Any]]) -> list[MappingEntry]:
    """Build the row -> entity_id / canonical_name mapping, preserving order."""
    mapping: list[MappingEntry] = []
    for row, record in enumerate(records):
        mapping.append(
            MappingEntry(
                row=row,
                entity_id=str(record.get("entity_id", "")).strip(),
                canonical_name=str(record.get("canonical_name", "")).strip(),
            )
        )
    return mapping


# --------------------------------------------------------------------------- #
# Model loading and embedding generation
# --------------------------------------------------------------------------- #

def load_model(model_name: str = MODEL_NAME) -> Any:
    """Load the SentenceTransformer model."""
    logger.info("Loading SentenceTransformer model: %s", model_name)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ModelLoadError(
            "sentence-transformers is not installed. "
            "Install it with `pip install sentence-transformers`."
        ) from exc

    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Failed to load model {model_name!r}: {exc}") from exc

    logger.info("Model loaded successfully")
    return model


def generate_embeddings(
    model: Any,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Generate embeddings for a list of texts, preserving input order.

    Uses tqdm for progress reporting across batches. The final array is
    cast to float32.
    """
    logger.info(
        "Generating embeddings for %d texts (batch_size=%d)", len(texts), batch_size
    )

    all_embeddings: list[np.ndarray] = []
    try:
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="Generating embeddings",
            unit="batch",
        ):
            batch = texts[start : start + batch_size]
            batch_embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            all_embeddings.append(np.asarray(batch_embeddings, dtype=np.float32))
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingGeneratorError(f"Embedding generation failed: {exc}") from exc

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    logger.info("Generated embeddings with shape %s", embeddings.shape)
    return embeddings


# --------------------------------------------------------------------------- #
# Validation of output
# --------------------------------------------------------------------------- #

def validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: int,
    expected_dim: int = EXPECTED_EMBEDDING_DIM,
) -> None:
    """Validate the shape and dtype of the generated embedding matrix."""
    logger.info("Validating generated embeddings")

    if embeddings.ndim != 2:
        raise EmbeddingDimensionError(
            f"Expected a 2D embedding array, got shape {embeddings.shape}"
        )

    rows, dim = embeddings.shape

    if rows != expected_rows:
        raise EmbeddingDimensionError(
            f"Embedding row count ({rows}) does not match metadata record "
            f"count ({expected_rows})."
        )

    if dim != expected_dim:
        raise EmbeddingDimensionError(
            f"Embedding dimension ({dim}) does not match expected dimension "
            f"({expected_dim}) for model {MODEL_NAME!r}."
        )

    if embeddings.dtype != np.float32:
        raise EmbeddingDimensionError(
            f"Embeddings must be float32, got dtype {embeddings.dtype}."
        )

    if not np.isfinite(embeddings).all():
        raise EmbeddingDimensionError(
            "Embeddings contain non-finite values (NaN or Inf)."
        )

    logger.info("Embedding validation passed: shape=%s dtype=%s", embeddings.shape, embeddings.dtype)


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def write_embeddings(embeddings: np.ndarray, path: Path) -> None:
    """Save the embedding matrix as a float32 .npy file."""
    logger.info("Writing embeddings to %s", path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings.astype(np.float32))
    except OSError as exc:
        raise EmbeddingWriteError(f"Failed to write embeddings file: {exc}") from exc


def write_mapping(mapping: list[MappingEntry], path: Path) -> None:
    """Save the row -> entity_id / canonical_name mapping as JSON."""
    logger.info("Writing embedding mapping to %s", path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [entry.to_dict() for entry in mapping]
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise EmbeddingWriteError(f"Failed to write mapping JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run(
    metadata_path: Path = METADATA_INPUT_PATH,
    embeddings_output_path: Path = EMBEDDINGS_OUTPUT_PATH,
    mapping_output_path: Path = MAPPING_OUTPUT_PATH,
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
) -> np.ndarray:
    """Execute the full embedding generation pipeline."""
    try:
        records = load_metadata(metadata_path)
        validate_metadata_records(records)

        texts = extract_searchable_texts(records)
        mapping = build_mapping(records)

        model = load_model(model_name)
        embeddings = generate_embeddings(model, texts, batch_size=batch_size)

        validate_embeddings(embeddings, expected_rows=len(records))

        write_embeddings(embeddings, embeddings_output_path)
        write_mapping(mapping, mapping_output_path)

        logger.info("Embedding generation pipeline completed successfully")
        return embeddings
    except EmbeddingGeneratorError:
        logger.exception("Embedding generation failed")
        raise
    except Exception:
        logger.exception("Unexpected error during embedding generation")
        raise


def main() -> None:
    try:
        run()
    except EmbeddingGeneratorError as exc:
        logger.error("Embedding generator terminated with error: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()