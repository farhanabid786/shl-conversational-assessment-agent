"""
scripts/retriever_loader.py

Phase 4: Hybrid Retrieval — Resource Loader
SHL Conversational Assessment Recommendation System

Loads and validates every retrieval artifact produced by Phases 1–3, then
returns a single frozen RetrieverResources dataclass for consumption by the
HybridRetriever.  Nothing is retrieved, reranked, or filtered here.

Artifact inputs (all read-only):
    data/processed/catalog_metadata.json
    data/embeddings/embedding_mapping.json
    data/faiss/catalog.index
    data/cache/bm25_index.pkl

Python 3.10.11
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Default artifact paths
# --------------------------------------------------------------------------- #

METADATA_PATH: Path = Path("data/processed/catalog_metadata.json")
MAPPING_PATH: Path = Path("data/embeddings/embedding_mapping.json")
FAISS_INDEX_PATH: Path = Path("data/faiss/catalog.index")
BM25_INDEX_PATH: Path = Path("data/cache/bm25_index.pkl")

SENTENCE_TRANSFORMER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("retriever_loader")

# --------------------------------------------------------------------------- #
# Custom exception hierarchy
# --------------------------------------------------------------------------- #


class RetrieverLoaderError(Exception):
    """Base exception for all retriever loader failures."""


class MissingArtifactError(RetrieverLoaderError):
    """Raised when a required artifact file does not exist on disk."""


class ArtifactLoadError(RetrieverLoaderError):
    """Raised when an artifact file exists but cannot be read or parsed."""


class ArtifactValidationError(RetrieverLoaderError):
    """Raised when a loaded artifact fails structural or content validation."""


class ConsistencyError(RetrieverLoaderError):
    """Raised when artifact sizes or identifiers are mutually inconsistent."""


class ModelLoadError(RetrieverLoaderError):
    """Raised when the SentenceTransformer model cannot be loaded."""


# --------------------------------------------------------------------------- #
# Frozen resource container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrieverResources:
    """Immutable snapshot of every artifact needed by the HybridRetriever.

    Attributes
    ----------
    metadata:
        Non-empty list of catalog record dicts from catalog_metadata.json.
        Each record must contain at least ``entity_id`` and ``canonical_name``.
    mapping:
        Non-empty list of ``{"row": int, "entity_id": str, "canonical_name": str}``
        dicts from embedding_mapping.json.  Row values are sequential and
        unique; entity_ids are unique.
    entity_lookup:
        Dict keyed by ``entity_id`` (str) → metadata record (dict).
        Built from *metadata* at load time; avoids linear scans at query time.
    faiss_index:
        Loaded ``faiss.Index`` with ``ntotal == len(metadata)``.
    bm25:
        BM25Okapi (or compatible) object unpickled from bm25_index.pkl.
    entity_ids:
        Ordered list[str] aligned with the BM25 corpus rows, sourced from
        the pickle payload.
    model:
        SentenceTransformer instance for ``sentence-transformers/all-MiniLM-L6-v2``,
        loaded exactly once at startup and reused for every query.
    """

    metadata: list[dict[str, Any]]
    mapping: list[dict[str, Any]]
    entity_lookup: dict[str, dict[str, Any]]
    faiss_index: Any  # faiss.Index — typed as Any to avoid a hard module-level import
    bm25: Any         # rank_bm25.BM25Okapi — typed as Any for the same reason
    entity_ids: list[str]
    model: Any        # sentence_transformers.SentenceTransformer


# --------------------------------------------------------------------------- #
# Step 0 — file-existence guard
# --------------------------------------------------------------------------- #


def _assert_file_exists(path: Path) -> None:
    """Raise MissingArtifactError if *path* is absent from the filesystem."""
    if not path.exists():
        raise MissingArtifactError(
            f"Required artifact not found on disk: {path}"
        )
    logger.debug("Artifact present: %s", path)


def _check_all_files_exist(
    metadata_path: Path,
    mapping_path: Path,
    faiss_path: Path,
    bm25_path: Path,
) -> None:
    """Verify every artifact path exists before attempting any I/O."""
    logger.info("Verifying artifact file existence")
    for path in (metadata_path, mapping_path, faiss_path, bm25_path):
        _assert_file_exists(path)
    logger.info("All four artifact files are present on disk")


# --------------------------------------------------------------------------- #
# Step 1 — load catalog_metadata.json
# --------------------------------------------------------------------------- #


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    """Deserialize catalog_metadata.json and return its top-level list.

    Raises
    ------
    ArtifactLoadError
        If the file cannot be read or contains invalid JSON.
    ArtifactValidationError
        If the parsed value is not a non-empty list.
    """
    logger.info("Loading catalog metadata: %s", path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data: Any = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ArtifactLoadError(
            f"catalog_metadata.json is not valid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise ArtifactLoadError(
            f"Cannot read catalog_metadata.json: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise ArtifactValidationError(
            "catalog_metadata.json must contain a JSON array at the root level; "
            f"got {type(data).__name__!r}."
        )
    if not data:
        raise ArtifactValidationError(
            "catalog_metadata.json is empty — expected at least one record."
        )

    logger.info("Loaded %d metadata records", len(data))
    return data


def _validate_metadata(metadata: list[dict[str, Any]]) -> None:
    """Check each record for a non-empty entity_id and detect duplicates.

    Raises
    ------
    ArtifactValidationError
        On missing/empty entity_id or on the first duplicate entity_id found.
    """
    logger.info("Validating metadata records")
    seen: set[str] = set()
    for idx, record in enumerate(metadata):
        entity_id = str(record.get("entity_id", "")).strip()
        if not entity_id:
            raise ArtifactValidationError(
                f"catalog_metadata.json: record at index {idx} is missing a "
                "non-empty 'entity_id'."
            )
        if entity_id in seen:
            raise ArtifactValidationError(
                f"catalog_metadata.json: duplicate entity_id {entity_id!r} "
                f"detected at index {idx}."
            )
        seen.add(entity_id)
    logger.info("Metadata validation passed: %d unique entity_ids", len(seen))


# --------------------------------------------------------------------------- #
# Step 2 — load embedding_mapping.json
# --------------------------------------------------------------------------- #


def _load_mapping(path: Path) -> list[dict[str, Any]]:
    """Deserialize embedding_mapping.json and return its top-level list.

    Raises
    ------
    ArtifactLoadError
        If the file cannot be read or contains invalid JSON.
    ArtifactValidationError
        If the parsed value is not a non-empty list.
    """
    logger.info("Loading embedding mapping: %s", path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data: Any = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ArtifactLoadError(
            f"embedding_mapping.json is not valid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise ArtifactLoadError(
            f"Cannot read embedding_mapping.json: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise ArtifactValidationError(
            "embedding_mapping.json must contain a JSON array at the root level; "
            f"got {type(data).__name__!r}."
        )
    if not data:
        raise ArtifactValidationError(
            "embedding_mapping.json is empty — expected at least one entry."
        )

    logger.info("Loaded %d mapping entries", len(data))
    return data


def _validate_mapping(mapping: list[dict[str, Any]]) -> None:
    """Verify required keys, uniqueness of rows and entity_ids, and sequential rows.

    Required keys per record: ``row``, ``entity_id``, ``canonical_name``.

    Raises
    ------
    ArtifactValidationError
        On missing keys, non-sequential rows, duplicate rows, or duplicate
        entity_ids.
    """
    logger.info("Validating mapping entries")
    required: frozenset[str] = frozenset({"row", "entity_id", "canonical_name"})
    seen_rows: set[int] = set()
    seen_entity_ids: set[str] = set()

    for idx, entry in enumerate(mapping):
        if not isinstance(entry, dict):
            raise ArtifactValidationError(
                f"embedding_mapping.json: entry at index {idx} is not an object; "
                f"got {type(entry).__name__!r}."
            )

        missing = required - entry.keys()
        if missing:
            raise ArtifactValidationError(
                f"embedding_mapping.json: entry at index {idx} is missing "
                f"required keys: {sorted(missing)}."
            )

        row = entry["row"]
        entity_id = str(entry["entity_id"]).strip()

        if row in seen_rows:
            raise ArtifactValidationError(
                f"embedding_mapping.json: duplicate 'row' value {row!r} "
                f"at index {idx}."
            )
        seen_rows.add(row)

        if entity_id in seen_entity_ids:
            raise ArtifactValidationError(
                f"embedding_mapping.json: duplicate 'entity_id' {entity_id!r} "
                f"at index {idx}."
            )
        seen_entity_ids.add(entity_id)

    # Rows must form the sequence 0, 1, …, N-1
    expected_rows = set(range(len(mapping)))
    if seen_rows != expected_rows:
        raise ArtifactValidationError(
            "embedding_mapping.json: 'row' values are not a contiguous "
            f"sequence from 0 to {len(mapping) - 1}."
        )

    logger.info(
        "Mapping validation passed: %d unique rows, %d unique entity_ids",
        len(seen_rows),
        len(seen_entity_ids),
    )


# --------------------------------------------------------------------------- #
# Step 3 — load FAISS index
# --------------------------------------------------------------------------- #


def _load_faiss_index(path: Path) -> Any:
    """Read the FAISS binary index from disk.

    Raises
    ------
    ArtifactLoadError
        If ``faiss`` is not installed or the index file cannot be read.
    ArtifactValidationError
        If the loaded index is empty (ntotal == 0).
    """
    logger.info("Loading FAISS index: %s", path)
    try:
        import faiss  # type: ignore[import]
    except ImportError as exc:
        raise ArtifactLoadError(
            "faiss-cpu (or faiss-gpu) is not installed.  "
            "Install it with: pip install faiss-cpu"
        ) from exc

    try:
        index = faiss.read_index(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ArtifactLoadError(
            f"Failed to read FAISS index from {path}: {exc}"
        ) from exc

    if index.ntotal == 0:
        raise ArtifactValidationError(
            f"FAISS index at {path} is empty (ntotal=0)."
        )

    logger.info("FAISS index loaded: ntotal=%d, d=%d", index.ntotal, index.d)
    return index


# --------------------------------------------------------------------------- #
# Step 4 — load BM25 pickle
# --------------------------------------------------------------------------- #


def _load_bm25_pickle(
    path: Path,
) -> tuple[Any, list[str], int]:
    """Unpickle bm25_index.pkl and validate its payload structure.

    Expected payload keys:
        bm25          — BM25Okapi or compatible object
        entity_ids    — list[str], one per corpus document
        corpus_size   — int, must equal len(entity_ids)

    Returns
    -------
    tuple of (bm25, entity_ids, corpus_size)

    Raises
    ------
    ArtifactLoadError
        If the file cannot be unpickled.
    ArtifactValidationError
        If required keys are absent, types are wrong, or corpus_size is
        inconsistent with entity_ids.
    """
    logger.info("Loading BM25 index: %s", path)
    try:
        with path.open("rb") as fh:
            payload: Any = pickle.load(fh)  # noqa: S301 — trusted internal artifact
    except pickle.UnpicklingError as exc:
        raise ArtifactLoadError(
            f"bm25_index.pkl is not a valid pickle file: {exc}"
        ) from exc
    except OSError as exc:
        raise ArtifactLoadError(
            f"Cannot read bm25_index.pkl: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ArtifactValidationError(
            "bm25_index.pkl must unpickle to a dict; "
            f"got {type(payload).__name__!r}."
        )

    missing_keys = {"bm25", "entity_ids", "corpus_size"} - payload.keys()
    if missing_keys:
        raise ArtifactValidationError(
            f"bm25_index.pkl is missing required keys: {sorted(missing_keys)}."
        )

    bm25 = payload["bm25"]
    entity_ids: Any = payload["entity_ids"]
    corpus_size: Any = payload["corpus_size"]

    if bm25 is None:
        raise ArtifactValidationError(
            "bm25_index.pkl: 'bm25' value must not be None."
        )

    if not isinstance(entity_ids, list) or not entity_ids:
        raise ArtifactValidationError(
            "bm25_index.pkl: 'entity_ids' must be a non-empty list; "
            f"got {type(entity_ids).__name__!r}."
        )

    if not isinstance(corpus_size, int) or corpus_size <= 0:
        raise ArtifactValidationError(
            "bm25_index.pkl: 'corpus_size' must be a positive integer; "
            f"got {corpus_size!r}."
        )

    if corpus_size != len(entity_ids):
        raise ArtifactValidationError(
            f"bm25_index.pkl: 'corpus_size' ({corpus_size}) does not match "
            f"len(entity_ids) ({len(entity_ids)})."
        )

    logger.info("BM25 index loaded: corpus_size=%d", corpus_size)
    return bm25, list(entity_ids), corpus_size


# --------------------------------------------------------------------------- #
# Step 5 — load SentenceTransformer model
# --------------------------------------------------------------------------- #


def _load_sentence_transformer(model_name: str) -> Any:
    """Instantiate and return the SentenceTransformer model.

    The model is loaded once here; callers must cache the result.

    Raises
    ------
    ArtifactLoadError
        If ``sentence_transformers`` is not installed.
    ModelLoadError
        If the model cannot be initialised (network error, bad name, etc.).
    """
    logger.info("Loading SentenceTransformer model: %s", model_name)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError as exc:
        raise ArtifactLoadError(
            "sentence-transformers is not installed.  "
            "Install it with: pip install sentence-transformers"
        ) from exc

    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"Failed to load SentenceTransformer model {model_name!r}: {exc}"
        ) from exc

    logger.info("SentenceTransformer model loaded successfully: %s", model_name)
    return model


# --------------------------------------------------------------------------- #
# Step 6 — build entity_lookup
# --------------------------------------------------------------------------- #


def _build_entity_lookup(
    metadata: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Create an O(1) entity_id → metadata record mapping.

    Parameters
    ----------
    metadata:
        Already-validated list of catalog record dicts.

    Returns
    -------
    dict[str, dict[str, Any]]
        Keys are ``entity_id`` strings; values are the corresponding records.
    """
    logger.info("Building entity_lookup from metadata")
    lookup: dict[str, dict[str, Any]] = {
        str(record["entity_id"]): record for record in metadata
    }
    logger.info("entity_lookup built: %d entries", len(lookup))
    return lookup


# --------------------------------------------------------------------------- #
# Step 7 — cross-artifact consistency checks
# --------------------------------------------------------------------------- #


def _validate_consistency(
    metadata: list[dict[str, Any]],
    mapping: list[dict[str, Any]],
    faiss_index: Any,
    corpus_size: int,
) -> None:
    """Assert that all four artifact cardinalities agree.

    Raises
    ------
    ConsistencyError
        If any pair of artifact sizes diverges.
    """
    logger.info("Running cross-artifact consistency checks")

    n_meta: int = len(metadata)
    n_map: int = len(mapping)
    n_faiss: int = faiss_index.ntotal

    mismatches: list[str] = []

    if n_map != n_meta:
        mismatches.append(
            f"mapping length ({n_map}) != metadata length ({n_meta})"
        )
    if n_faiss != n_meta:
        mismatches.append(
            f"FAISS ntotal ({n_faiss}) != metadata length ({n_meta})"
        )
    if corpus_size != n_meta:
        mismatches.append(
            f"BM25 corpus_size ({corpus_size}) != metadata length ({n_meta})"
        )

    if mismatches:
        detail = "; ".join(mismatches)
        raise ConsistencyError(
            f"Artifact size mismatch — {detail}.  "
            "Re-run the relevant generation pipeline to produce consistent artifacts."
        )

    logger.info(
        "Consistency check passed: %d records across all artifacts", n_meta
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def load_retriever_resources(
    metadata_path: Path = METADATA_PATH,
    mapping_path: Path = MAPPING_PATH,
    faiss_index_path: Path = FAISS_INDEX_PATH,
    bm25_index_path: Path = BM25_INDEX_PATH,
    model_name: str = SENTENCE_TRANSFORMER_MODEL,
) -> RetrieverResources:
    """Load, validate, and assemble all retrieval artifacts into a RetrieverResources.

    This function is the single startup call.  It should be invoked once at
    application initialisation; the returned RetrieverResources is then passed
    into HybridRetriever (and all downstream modules) for the lifetime of the
    process.

    Parameters
    ----------
    metadata_path:
        Path to ``catalog_metadata.json``.
    mapping_path:
        Path to ``embedding_mapping.json``.
    faiss_index_path:
        Path to the FAISS binary index ``catalog.index``.
    bm25_index_path:
        Path to ``bm25_index.pkl``.
    model_name:
        HuggingFace model identifier for the SentenceTransformer.

    Returns
    -------
    RetrieverResources
        Frozen dataclass containing all validated artifacts and the
        pre-loaded embedding model.

    Raises
    ------
    MissingArtifactError
        If any required file is absent from the filesystem.
    ArtifactLoadError
        If any file exists but cannot be read or deserialised.
    ArtifactValidationError
        If any artifact fails its structural or content validation.
    ConsistencyError
        If artifact cardinalities are mutually inconsistent.
    ModelLoadError
        If the SentenceTransformer model cannot be initialised.
    """
    logger.info("=== RetrieverLoader startup ===")

    # 0. Guard — all files must be present before any I/O is attempted
    _check_all_files_exist(
        metadata_path, mapping_path, faiss_index_path, bm25_index_path
    )

    # 1. Load catalog metadata
    metadata = _load_metadata(metadata_path)
    _validate_metadata(metadata)

    # 2. Load embedding mapping
    mapping = _load_mapping(mapping_path)
    _validate_mapping(mapping)

    # 3. Load FAISS index
    faiss_index = _load_faiss_index(faiss_index_path)

    # 4. Load BM25 pickle
    bm25, entity_ids, corpus_size = _load_bm25_pickle(bm25_index_path)

    # 5. Cross-artifact size consistency
    _validate_consistency(metadata, mapping, faiss_index, corpus_size)

    # 6. Build O(1) entity lookup
    entity_lookup = _build_entity_lookup(metadata)

    # 7. Load SentenceTransformer (once, after all artifact checks succeed)
    model = _load_sentence_transformer(model_name)

    resources = RetrieverResources(
        metadata=metadata,
        mapping=mapping,
        entity_lookup=entity_lookup,
        faiss_index=faiss_index,
        bm25=bm25,
        entity_ids=entity_ids,
        model=model,
    )

    logger.info(
        "=== RetrieverResources ready: %d records | FAISS ntotal=%d | "
        "BM25 corpus=%d | model=%s ===",
        len(metadata),
        faiss_index.ntotal,
        corpus_size,
        model_name,
    )
    return resources


# --------------------------------------------------------------------------- #
# CLI smoke-test entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Standalone smoke-test: python -m scripts.retriever_loader"""
    try:
        resources = load_retriever_resources()
        logger.info(
            "Smoke-test passed — RetrieverResources ready with %d records.",
            len(resources.metadata),
        )
    except RetrieverLoaderError as exc:
        logger.error("RetrieverLoader failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
