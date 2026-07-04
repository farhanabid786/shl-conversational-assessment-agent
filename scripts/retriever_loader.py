"""
scripts/retriever_loader.py

Phase 4: Hybrid Retrieval — Resource Loader
SHL Conversational Assessment Recommendation System

Loads and validates every retrieval artifact produced by Phases 1-3,
INCLUDING the SentenceTransformer embedding model, then returns a single
frozen RetrieverResources dataclass for consumption by the HybridRetriever.
Nothing is retrieved, reranked, or filtered here.

Eager loading
-------------
Every artifact — FAISS index, BM25 index, catalog metadata, mapping, AND
the embedding model — is loaded synchronously, once, during this call.
There is no lazy/first-request loading and therefore no need for
runtime locking: by the time `load_retriever_resources()` returns, the
process is either fully ready to serve traffic or it has raised and the
process should not start accepting requests at all. This trades a
longer, one-time startup cost (the evaluator explicitly allows up to 2
minutes for a cold-started service to become healthy) for uniform,
predictable per-request latency and the elimination of a whole class of
"first request pays the download tax" / concurrent-download bugs.

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

# 3-layer MiniLM: chosen over the 6-layer default because the catalog is
# well under 1,000 rows, so the larger model's extra capacity buys
# negligible recall while costing more download size, RAM, and encode
# latency. Both variants share a 384-dim output, so this is a drop-in
# swap for every downstream consumer (FAISS index dimension unchanged).
SENTENCE_TRANSFORMER_MODEL: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
EXPECTED_EMBEDDING_DIMENSION: int = 384

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
    """Raised when the SentenceTransformer model cannot be loaded.

    Because loading is now eager (at startup, inside this module), this
    exception surfaces during application boot rather than mid-request.
    Callers (app.lifespan) should treat it as fatal to startup: a service
    that cannot embed queries should not come up as "ready".
    """


# --------------------------------------------------------------------------- #
# Frozen resource container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrieverResources:
    """Immutable snapshot of every artifact needed by the HybridRetriever.

    Every field is populated by the time `load_retriever_resources()`
    returns — including `model`, which is loaded eagerly (no lazy
    first-request loading, no runtime locking required).

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
        SentenceTransformer instance, loaded eagerly and exactly once at
        startup, reused for every query for the lifetime of the process.
        Never None once RetrieverResources has been constructed.
    row_map:
        Dict keyed by FAISS row (int) → ``{"entity_id": str, "canonical_name": str}``.
        Built once from *mapping* at load time and reused for every query.
    canonical_lookup:
        Dict keyed by ``entity_id`` (str) → ``canonical_name`` (str), built
        once from *mapping* at load time.
    """

    metadata: list[dict[str, Any]]
    mapping: list[dict[str, Any]]
    entity_lookup: dict[str, dict[str, Any]]
    faiss_index: Any  # faiss.Index — typed as Any to avoid a hard module-level import
    bm25: Any         # rank_bm25.BM25Okapi — typed as Any for the same reason
    entity_ids: list[str]
    model: Any        # sentence_transformers.SentenceTransformer — never None
    row_map: dict[int, dict[str, str]]
    canonical_lookup: dict[str, str]


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
    logger.debug("Verifying artifact file existence")
    for path in (metadata_path, mapping_path, faiss_path, bm25_path):
        _assert_file_exists(path)
    logger.debug("All four artifact files are present on disk")


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
    logger.debug("Loading catalog metadata: %s", path)
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

    logger.debug("Loaded %d metadata records", len(data))
    return data


def _validate_metadata(metadata: list[dict[str, Any]]) -> None:
    """Check each record for a non-empty entity_id and detect duplicates.

    Raises
    ------
    ArtifactValidationError
        On missing/empty entity_id or on the first duplicate entity_id found.
    """
    logger.debug("Validating metadata records")
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
    logger.debug("Metadata validation passed: %d unique entity_ids", len(seen))


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
    logger.debug("Loading embedding mapping: %s", path)
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
            "embedding_mapping.json must contain a JSON array at the root "
            f"level; got {type(data).__name__!r}."
        )
    if not data:
        raise ArtifactValidationError(
            "embedding_mapping.json is empty — expected at least one entry."
        )

    logger.debug("Loaded %d mapping entries", len(data))
    return data


def _validate_mapping(mapping: list[dict[str, Any]]) -> None:
    """Check each mapping entry for required keys and detect duplicates.

    Raises
    ------
    ArtifactValidationError
        On a missing key, a non-sequential/duplicate row, or a duplicate
        entity_id.
    """
    logger.debug("Validating mapping entries")
    seen_rows: set[int] = set()
    seen_ids: set[str] = set()

    for idx, entry in enumerate(mapping):
        if "row" not in entry or "entity_id" not in entry:
            raise ArtifactValidationError(
                f"embedding_mapping.json: entry at index {idx} is missing "
                "'row' or 'entity_id'."
            )

        row = entry["row"]
        if not isinstance(row, int):
            raise ArtifactValidationError(
                f"embedding_mapping.json: entry at index {idx} has a "
                f"non-integer 'row': {row!r}."
            )
        if row in seen_rows:
            raise ArtifactValidationError(
                f"embedding_mapping.json: duplicate row {row} detected at "
                f"index {idx}."
            )
        seen_rows.add(row)

        entity_id = str(entry.get("entity_id", "")).strip()
        if not entity_id:
            raise ArtifactValidationError(
                f"embedding_mapping.json: entry at index {idx} has an "
                "empty 'entity_id'."
            )
        if entity_id in seen_ids:
            raise ArtifactValidationError(
                f"embedding_mapping.json: duplicate entity_id {entity_id!r} "
                f"detected at index {idx}."
            )
        seen_ids.add(entity_id)

    expected_rows = set(range(len(mapping)))
    if seen_rows != expected_rows:
        raise ArtifactValidationError(
            "embedding_mapping.json: row values are not a contiguous "
            f"0..N-1 sequence. Expected {len(mapping)} unique rows "
            f"0..{len(mapping) - 1}."
        )

    logger.debug("Mapping validation passed: %d unique rows/entity_ids", len(mapping))


# --------------------------------------------------------------------------- #
# Step 3 — load FAISS index
# --------------------------------------------------------------------------- #


def _load_faiss_index(path: Path) -> Any:
    """Read the FAISS binary index from disk.

    Raises
    ------
    ArtifactLoadError
        If ``faiss`` is not installed or the file cannot be deserialized.
    """
    logger.debug("Loading FAISS index: %s", path)
    try:
        import faiss  # type: ignore[import]
    except ImportError as exc:
        raise ArtifactLoadError(
            "faiss-cpu is not installed. Install it with `pip install faiss-cpu`."
        ) from exc

    try:
        index = faiss.read_index(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ArtifactLoadError(f"Failed to read FAISS index: {exc}") from exc

    logger.debug("FAISS index loaded: ntotal=%d, d=%d", index.ntotal, index.d)
    return index


# --------------------------------------------------------------------------- #
# Step 4 — load BM25 pickle
# --------------------------------------------------------------------------- #


def _load_bm25_pickle(path: Path) -> tuple[Any, list[str], int]:
    """Unpickle bm25_index.pkl and return (bm25, entity_ids, corpus_size).

    Raises
    ------
    ArtifactLoadError
        If the file cannot be read or unpickled.
    ArtifactValidationError
        If the payload does not have the expected shape.
    """
    logger.debug("Loading BM25 index: %s", path)
    try:
        with path.open("rb") as fh:
            payload: Any = pickle.load(fh)
    except (OSError, pickle.UnpicklingError) as exc:
        raise ArtifactLoadError(f"Cannot read bm25_index.pkl: {exc}") from exc

    if not isinstance(payload, dict):
        raise ArtifactValidationError(
            "bm25_index.pkl must unpickle to a dict payload; got "
            f"{type(payload).__name__!r}."
        )

    bm25 = payload.get("bm25")
    entity_ids = payload.get("entity_ids")

    if bm25 is None:
        raise ArtifactValidationError("bm25_index.pkl payload is missing 'bm25'.")
    if not isinstance(entity_ids, list) or not entity_ids:
        raise ArtifactValidationError(
            "bm25_index.pkl payload 'entity_ids' must be a non-empty list."
        )

    corpus_size = len(entity_ids)
    logger.debug("BM25 index loaded: corpus_size=%d", corpus_size)
    return bm25, entity_ids, corpus_size


# --------------------------------------------------------------------------- #
# Step 5 — eagerly load the SentenceTransformer model
# --------------------------------------------------------------------------- #


def _load_sentence_transformer(
    model_name: str,
    expected_dimension: int | None = None,
) -> Any:
    """Instantiate and return the SentenceTransformer model, eagerly.

    Called exactly once, synchronously, during
    `load_retriever_resources()`. There is no lazy/first-request path and
    therefore no locking required — by construction, only one thread is
    ever inside application startup at a time.

    Raises
    ------
    ArtifactLoadError
        If ``sentence_transformers`` is not installed.
    ModelLoadError
        If the model cannot be initialised (network error, bad name, etc.)
        or its output dimension does not match ``expected_dimension``.
    """
    logger.info("Eagerly loading SentenceTransformer model: %s", model_name)
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

    if expected_dimension is not None:
        actual_dimension = int(model.get_sentence_embedding_dimension())
        if actual_dimension != expected_dimension:
            raise ModelLoadError(
                f"SentenceTransformer model {model_name!r} produces "
                f"{actual_dimension}-dim embeddings but the FAISS index "
                f"expects {expected_dimension}-dim vectors. The model and "
                "the persisted FAISS index were built with incompatible "
                "embedding models — regenerate the index for this model."
            )

    logger.info(
        "SentenceTransformer model loaded successfully: %s (dim=%d)",
        model_name,
        int(model.get_sentence_embedding_dimension()),
    )
    return model


# --------------------------------------------------------------------------- #
# Step 6 — build entity_lookup / row_map / canonical_lookup
# --------------------------------------------------------------------------- #


def _build_entity_lookup(
    metadata: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Create an O(1) entity_id → metadata record mapping."""
    logger.debug("Building entity_lookup from metadata")
    lookup: dict[str, dict[str, Any]] = {
        str(record["entity_id"]): record for record in metadata
    }
    logger.debug("entity_lookup built: %d entries", len(lookup))
    return lookup


def _build_row_map(mapping: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    """Build the FAISS row -> {entity_id, canonical_name} lookup once.

    Cached permanently on ``RetrieverResources.row_map``; the hybrid
    retriever reuses this object on every query rather than rebuilding it.
    """
    logger.debug("Building row_map from mapping")
    row_map: dict[int, dict[str, str]] = {}
    for entry in mapping:
        row = int(entry.get("row", -1))
        eid = str(entry.get("entity_id", "")).strip()
        cname = str(entry.get("canonical_name", "")).strip()
        if row >= 0 and eid:
            row_map[row] = {"entity_id": eid, "canonical_name": cname}
    logger.debug("row_map built: %d entries", len(row_map))
    return row_map


def _build_canonical_lookup(mapping: list[dict[str, Any]]) -> dict[str, str]:
    """Build the entity_id -> canonical_name lookup once.

    Cached permanently on ``RetrieverResources.canonical_lookup``.
    """
    logger.debug("Building canonical_lookup from mapping")
    lookup: dict[str, str] = {}
    for entry in mapping:
        eid = str(entry.get("entity_id", "")).strip()
        cname = str(entry.get("canonical_name", "")).strip()
        if eid:
            lookup[eid] = cname
    logger.debug("canonical_lookup built: %d entries", len(lookup))
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
    logger.debug("Running cross-artifact consistency checks")

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

    logger.debug(
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
    expected_embedding_dimension: int | None = EXPECTED_EMBEDDING_DIMENSION,
) -> RetrieverResources:
    """Load, validate, and assemble all retrieval artifacts, EAGERLY.

    This function is the single startup call. It should be invoked once
    at application initialisation (see app.lifespan); the returned
    RetrieverResources — including its loaded embedding model — is then
    passed into HybridRetriever (and all downstream modules) for the
    lifetime of the process. There is no further loading to do after
    this call returns: the process is fully ready or this call raised.

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
    expected_embedding_dimension:
        If given, the loaded model's output dimension is validated against
        this value (and, implicitly, against the FAISS index's own `d`
        during consistency checks) before startup is allowed to succeed.
        Pass None to skip this check.

    Returns
    -------
    RetrieverResources
        Frozen dataclass containing all validated artifacts and the
        eagerly-loaded embedding model. ``resources.model`` is guaranteed
        to be non-None.

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
        If the SentenceTransformer model cannot be initialised or its
        dimension does not match the FAISS index.
    """
    logger.info("=== RetrieverLoader startup (eager) ===")

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

    # 5b. FAISS index dimension vs. expected embedding dimension. Catches a
    # stale index (built with a different embedding model) at startup
    # instead of failing confusingly on the first real query.
    if expected_embedding_dimension is not None:
        faiss_dim = int(faiss_index.d)
        if faiss_dim != expected_embedding_dimension:
            raise ConsistencyError(
                f"FAISS index dimension ({faiss_dim}) does not match the "
                f"configured embedding dimension ({expected_embedding_dimension}). "
                "The persisted FAISS index appears to have been built with a "
                "different embedding model — regenerate the index for the "
                f"current model ({model_name!r})."
            )

    # 6. Build O(1) lookups — computed exactly once here and cached
    # permanently on RetrieverResources.
    entity_lookup = _build_entity_lookup(metadata)
    row_map = _build_row_map(mapping)
    canonical_lookup = _build_canonical_lookup(mapping)

    # 7. Eagerly load the embedding model. No lazy loading, no locking:
    # this is the only place the model is ever constructed, and it
    # happens once, synchronously, before the process is considered ready.
    model = _load_sentence_transformer(model_name, expected_embedding_dimension)

    resources = RetrieverResources(
        metadata=metadata,
        mapping=mapping,
        entity_lookup=entity_lookup,
        faiss_index=faiss_index,
        bm25=bm25,
        entity_ids=entity_ids,
        model=model,
        row_map=row_map,
        canonical_lookup=canonical_lookup,
    )

    logger.info(
        "RetrieverResources ready: %d records | FAISS ntotal=%d | "
        "BM25 corpus=%d | model=%s (loaded eagerly)",
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
            "Smoke-test passed — RetrieverResources ready with %d records "
            "and a loaded embedding model.",
            len(resources.metadata),
        )
    except RetrieverLoaderError as exc:
        logger.error("RetrieverLoader failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
