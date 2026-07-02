"""
scripts/hybrid_retriever.py

Phase 4: Hybrid Retriever
SHL Conversational Assessment Recommendation System

Performs semantic retrieval (FAISS / SentenceTransformer) and lexical
retrieval (BM25Okapi) against the artefacts loaded by RetrieverResources.

This module is RETRIEVAL ONLY.  It does NOT perform:
  - Fusion scoring
  - Metadata filtering
  - Reranking
  - Recommendation generation
  - Clarification logic
  - Conversation management

Outputs:
  HybridCandidates containing SemanticCandidate and LexicalCandidate lists
  together with an insertion-ordered, deduplicated merged_ids list.

Python 3.10.11
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
# )
# logger = logging.getLogger("hybrid_retriever")

logger = logging.getLogger(__name__)
# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
MAX_QUERY_LENGTH: int = 5_000

# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #


class HybridRetrieverError(Exception):
    """Base exception for all hybrid retriever failures."""


class ValidationError(HybridRetrieverError):
    """Raised when input validation fails (resources, query, parameters)."""


class EncodingError(HybridRetrieverError):
    """Raised when query encoding fails."""


class RetrievalError(HybridRetrieverError):
    """Raised when FAISS or BM25 retrieval fails."""


# --------------------------------------------------------------------------- #
# Output Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SemanticCandidate:
    """A single candidate returned by semantic (FAISS) retrieval.

    Attributes
    ----------
    entity_id:
        Unique identifier for the SHL assessment.
    canonical_name:
        Human-readable name of the assessment.
    score:
        Inner-product similarity score after L2 normalisation (range [0, 1]).
    rank:
        1-based rank within the semantic result list (1 = most similar).
    """

    entity_id: str
    canonical_name: str
    score: float
    rank: int


@dataclass(frozen=True)
class LexicalCandidate:
    """A single candidate returned by lexical (BM25) retrieval.

    Attributes
    ----------
    entity_id:
        Unique identifier for the SHL assessment.
    canonical_name:
        Human-readable name of the assessment.
    score:
        BM25 relevance score (positive float).
    rank:
        1-based rank within the lexical result list (1 = most relevant).
    """

    entity_id: str
    canonical_name: str
    score: float
    rank: int


@dataclass(frozen=True)
class HybridCandidates:
    """Container holding the outputs of one hybrid retrieval call.

    Attributes
    ----------
    semantic:
        Ordered list of SemanticCandidate objects (highest score first).
    lexical:
        Ordered list of LexicalCandidate objects (highest score first).
    merged_ids:
        Insertion-ordered, deduplicated sequence of entity_ids.
        Semantic results are listed first, then any additional lexical ids.
    """

    semantic: list[SemanticCandidate]
    lexical: list[LexicalCandidate]
    merged_ids: list[str]

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def semantic_ids(self) -> list[str]:
        """Return entity_ids from the semantic result list, in rank order."""
        return [c.entity_id for c in self.semantic]

    @property
    def lexical_ids(self) -> list[str]:
        """Return entity_ids from the lexical result list, in rank order."""
        return [c.entity_id for c in self.lexical]


# --------------------------------------------------------------------------- #
# Query Preprocessing
# --------------------------------------------------------------------------- #


def clean_query(raw: str) -> str:
    """Normalise a raw query string.

    Steps
    -----
    1. Strip leading/trailing whitespace.
    2. Collapse internal whitespace runs to a single space.
    3. Lowercase.

    Parameters
    ----------
    raw:
        The raw query string supplied by the caller.

    Returns
    -------
    str
        Cleaned, lowercased query string.

    Raises
    ------
    ValidationError
        If the query is empty, whitespace-only, or exceeds MAX_QUERY_LENGTH
        characters (measured after stripping).
    """
    if not isinstance(raw, str):
        raise ValidationError(
            f"Query must be a string, got {type(raw).__name__!r}."
        )

    stripped: str = raw.strip()

    if not stripped:
        raise ValidationError(
            "Query must not be empty or consist solely of whitespace."
        )

    if len(stripped) > MAX_QUERY_LENGTH:
        raise ValidationError(
            f"Query length ({len(stripped)} chars) exceeds the maximum "
            f"allowed length of {MAX_QUERY_LENGTH} characters."
        )

    collapsed: str = re.sub(r"\s+", " ", stripped)
    return collapsed.lower()


def tokenize_query(query: str) -> list[str]:
    """Split a cleaned query into tokens on non-alphanumeric boundaries.

    Parameters
    ----------
    query:
        A cleaned query string (output of clean_query()).

    Returns
    -------
    list[str]
        Non-empty tokens in the order they appear in the query string.
        The output is deterministic for identical inputs.
    """
    raw_tokens: list[str] = re.split(r"[^a-z0-9]+", query)
    return [tok for tok in raw_tokens if tok]


# --------------------------------------------------------------------------- #
# Internal Helpers
# --------------------------------------------------------------------------- #


def _build_entity_lookup(mapping: list[dict[str, Any]]) -> dict[str, str]:
    """Build a dict mapping entity_id -> canonical_name from the mapping list.

    Parameters
    ----------
    mapping:
        The ``resources.mapping`` list from RetrieverResources.

    Returns
    -------
    dict[str, str]
        Maps each entity_id to its canonical_name.
    """
    lookup: dict[str, str] = {}
    for entry in mapping:
        eid = str(entry.get("entity_id", "")).strip()
        cname = str(entry.get("canonical_name", "")).strip()
        if eid:
            lookup[eid] = cname
    return lookup


def _build_row_to_entity(mapping: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    """Build a dict mapping FAISS row index -> {entity_id, canonical_name}.

    Parameters
    ----------
    mapping:
        The ``resources.mapping`` list from RetrieverResources.

    Returns
    -------
    dict[int, dict[str, str]]
        Keyed by integer row, value carries entity_id and canonical_name.
    """
    row_map: dict[int, dict[str, str]] = {}
    for entry in mapping:
        row = int(entry.get("row", -1))
        eid = str(entry.get("entity_id", "")).strip()
        cname = str(entry.get("canonical_name", "")).strip()
        if row >= 0 and eid:
            row_map[row] = {"entity_id": eid, "canonical_name": cname}
    return row_map


def _get_model(resources: Any) -> Any:
    """Return a SentenceTransformer model from resources or load it fresh.

    Checks for ``resources.model`` first (forward-compatible with a future
    RetrieverResources that exposes the model).  Falls back to loading
    MODEL_NAME from sentence-transformers if the attribute is absent or None.

    Parameters
    ----------
    resources:
        A RetrieverResources (or compatible) instance.

    Returns
    -------
    SentenceTransformer
        A ready-to-encode model instance.

    Raises
    ------
    ValidationError
        If the model cannot be loaded.
    """
    model = getattr(resources, "model", None)
    if model is not None:
        logger.debug("Using model from RetrieverResources.")
        return model

    logger.info(
        "resources.model not available — loading %s directly.", MODEL_NAME
    )
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        return SentenceTransformer(MODEL_NAME)
    except ImportError as exc:
        raise ValidationError(
            "sentence-transformers is not installed. "
            "Install it with `pip install sentence-transformers`."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"Failed to load SentenceTransformer model {MODEL_NAME!r}: {exc}"
        ) from exc


def _get_entity_lookup(resources: Any) -> dict[str, str]:
    """Return an entity_id -> canonical_name lookup from resources or derived.

    Checks for ``resources.entity_lookup`` first (forward-compatible).
    Falls back to deriving the lookup from ``resources.mapping``.

    Parameters
    ----------
    resources:
        A RetrieverResources (or compatible) instance.

    Returns
    -------
    dict[str, str]
        Maps entity_id -> canonical_name.
    """
    lookup = getattr(resources, "entity_lookup", None)
    if isinstance(lookup, dict) and lookup:
        logger.debug("Using entity_lookup from RetrieverResources.")
        return lookup

    logger.debug("Deriving entity_lookup from resources.mapping.")
    return _build_entity_lookup(resources.mapping)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate_resources(resources: Any) -> None:
    """Assert that all required retriever artefacts are present and usable.

    Parameters
    ----------
    resources:
        Expected to be a RetrieverResources instance.

    Raises
    ------
    ValidationError
        If any required attribute is missing or None.
    """
    if resources is None:
        raise ValidationError("resources must not be None.")

    required_attrs = ("faiss_index", "bm25", "mapping", "entity_ids")
    for attr in required_attrs:
        val = getattr(resources, attr, None)
        if val is None:
            raise ValidationError(
                f"RetrieverResources is missing a required attribute: "
                f"'{attr}' is None or absent."
            )

    if not resources.mapping:
        raise ValidationError(
            "resources.mapping is empty — cannot resolve row -> entity_id."
        )

    if not resources.entity_ids:
        raise ValidationError(
            "resources.entity_ids is empty — BM25 corpus has no documents."
        )

    logger.debug("Resource validation passed.")


def _validate_top_k(semantic_top_k: int, lexical_top_k: int) -> None:
    """Assert that both top-k values are positive integers.

    Parameters
    ----------
    semantic_top_k:
        Number of semantic candidates to return.
    lexical_top_k:
        Number of lexical candidates to return.

    Raises
    ------
    ValidationError
        If either value is not a positive integer.
    """
    if not isinstance(semantic_top_k, int) or semantic_top_k < 1:
        raise ValidationError(
            f"semantic_top_k must be a positive integer, got {semantic_top_k!r}."
        )
    if not isinstance(lexical_top_k, int) or lexical_top_k < 1:
        raise ValidationError(
            f"lexical_top_k must be a positive integer, got {lexical_top_k!r}."
        )


def _validate_embedding_dim(embedding: np.ndarray, faiss_index: Any) -> None:
    """Assert that the query embedding dimension matches the FAISS index.

    Parameters
    ----------
    embedding:
        The L2-normalised query embedding with shape (1, d).
    faiss_index:
        The loaded FAISS IndexFlatIP object.

    Raises
    ------
    ValidationError
        If dimensions do not agree.
    """
    query_dim: int = embedding.shape[1]
    index_dim: int = faiss_index.d
    if query_dim != index_dim:
        raise ValidationError(
            f"Query embedding dimension ({query_dim}) does not match "
            f"FAISS index dimension ({index_dim}). "
            "Ensure the same SentenceTransformer model is used for both "
            "index construction and retrieval."
        )


# --------------------------------------------------------------------------- #
# Semantic Retrieval
# --------------------------------------------------------------------------- #


def _encode_query(model: Any, query: str) -> np.ndarray:
    """Encode a cleaned query string into an L2-normalised float32 embedding.

    Steps
    -----
    1. Encode via model.encode() → numpy array.
    2. Cast to float32.
    3. Reshape to (1, d).
    4. L2 normalise in-place via faiss.normalize_L2().

    Parameters
    ----------
    model:
        A loaded SentenceTransformer (or compatible) model.
    query:
        A cleaned query string.

    Returns
    -------
    np.ndarray
        Shape (1, d), dtype float32, L2-normalised.

    Raises
    ------
    EncodingError
        If encoding or normalisation fails.
    """
    try:
        import faiss  # type: ignore[import]

        raw: np.ndarray = model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vec: np.ndarray = np.asarray(raw, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec
    except ImportError as exc:
        raise EncodingError(
            "faiss-cpu is not installed. Install it with `pip install faiss-cpu`."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise EncodingError(
            f"Failed to encode query {query!r}: {exc}"
        ) from exc


def _run_semantic_retrieval(
    faiss_index: Any,
    query_vec: np.ndarray,
    row_map: dict[int, dict[str, str]],
    top_k: int,
) -> list[SemanticCandidate]:
    """Search the FAISS IndexFlatIP and return ranked SemanticCandidates.

    Row-score alignment is guaranteed by iterating zip(indices[0],
    distances[0]) rather than processing the two arrays independently.

    Parameters
    ----------
    faiss_index:
        A loaded FAISS index (IndexFlatIP or wrapping index).
    query_vec:
        L2-normalised query embedding with shape (1, d), dtype float32.
    row_map:
        Maps FAISS row integer -> {entity_id, canonical_name}.
    top_k:
        Maximum number of candidates to return.

    Returns
    -------
    list[SemanticCandidate]
        Ranked from highest to lowest similarity score.

    Raises
    ------
    RetrievalError
        If the FAISS search call raises an exception.
    """
    try:
        distances: np.ndarray
        indices: np.ndarray
        distances, indices = faiss_index.search(query_vec, top_k)
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(f"FAISS search failed: {exc}") from exc

    candidates: list[SemanticCandidate] = []
    rank: int = 1

    # Guarantee row-score alignment by zipping the paired arrays.
    for row, score in zip(indices[0], distances[0]):
        row_int: int = int(row)
        if row_int == -1:
            # FAISS sentinel for an unfilled slot — skip.
            continue

        entry = row_map.get(row_int)
        if entry is None:
            logger.warning(
                "FAISS returned row %d with no mapping entry — skipping.",
                row_int,
            )
            continue

        candidates.append(
            SemanticCandidate(
                entity_id=entry["entity_id"],
                canonical_name=entry["canonical_name"],
                score=float(score),
                rank=rank,
            )
        )
        rank += 1

    logger.debug(
        "Semantic retrieval returned %d candidates (top_k=%d).",
        len(candidates),
        top_k,
    )
    return candidates


# --------------------------------------------------------------------------- #
# Lexical Retrieval
# --------------------------------------------------------------------------- #


def _run_lexical_retrieval(
    bm25: Any,
    tokens: list[str],
    entity_ids: list[str],
    entity_lookup: dict[str, str],
    top_k: int,
) -> list[LexicalCandidate]:
    """Score all BM25 corpus documents and return the top-k positive results.

    Steps
    -----
    1. Obtain per-document BM25 scores via bm25.get_scores(tokens).
    2. Pair each score with its aligned entity_id from entity_ids.
    3. Sort descending by score.
    4. Keep only results with a positive score.
    5. Truncate to top_k.
    6. Resolve canonical_name via entity_lookup.

    Parameters
    ----------
    bm25:
        A BM25Okapi (or compatible) object.
    tokens:
        Tokenised query (output of tokenize_query()).
    entity_ids:
        Ordered list of entity_id strings aligned to BM25 corpus rows
        (resources.entity_ids).
    entity_lookup:
        Maps entity_id -> canonical_name.
    top_k:
        Maximum number of candidates to return.

    Returns
    -------
    list[LexicalCandidate]
        Ranked from highest to lowest BM25 score.  Only positive-score
        results are included.

    Raises
    ------
    RetrievalError
        If BM25 scoring raises an exception.
    """
    try:
        scores: np.ndarray = bm25.get_scores(tokens)
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(f"BM25 scoring failed: {exc}") from exc

    # Pair each score with its entity_id, preserving the alignment.
    scored: list[tuple[float, str]] = [
        (float(s), eid)
        for s, eid in zip(scores, entity_ids)
    ]

    # Sort descending by score.
    scored.sort(key=lambda x: x[0], reverse=True)

    candidates: list[LexicalCandidate] = []
    rank: int = 1

    for score, entity_id in scored:
        if score <= 0.0:
            # BM25 returns 0 for documents with no token overlap — discard.
            break
        if rank > top_k:
            break

        canonical_name: str = entity_lookup.get(entity_id, "")
        if not canonical_name:
            logger.warning(
                "entity_lookup has no entry for entity_id %r — "
                "canonical_name will be empty.",
                entity_id,
            )

        candidates.append(
            LexicalCandidate(
                entity_id=entity_id,
                canonical_name=canonical_name,
                score=score,
                rank=rank,
            )
        )
        rank += 1

    logger.debug(
        "Lexical retrieval returned %d candidates (top_k=%d).",
        len(candidates),
        top_k,
    )
    return candidates


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #


def _merge_ids(
    semantic: list[SemanticCandidate],
    lexical: list[LexicalCandidate],
) -> list[str]:
    """Produce an insertion-ordered, deduplicated list of entity_ids.

    Semantic ids are added first, then any lexical ids not already present.

    Parameters
    ----------
    semantic:
        Semantic candidates in rank order.
    lexical:
        Lexical candidates in rank order.

    Returns
    -------
    list[str]
        Unique entity_ids in insertion order (semantic first).
    """
    seen: set[str] = set()
    merged: list[str] = []

    for candidate in semantic:
        if candidate.entity_id not in seen:
            seen.add(candidate.entity_id)
            merged.append(candidate.entity_id)

    for candidate in lexical:
        if candidate.entity_id not in seen:
            seen.add(candidate.entity_id)
            merged.append(candidate.entity_id)

    return merged


# --------------------------------------------------------------------------- #
# Public Entry Point
# --------------------------------------------------------------------------- #


def retrieve(
    query: str,
    resources: Any,
    semantic_top_k: int = 10,
    lexical_top_k: int = 10,
) -> HybridCandidates:
    """Run hybrid retrieval for a natural-language query.

    Performs:
      1. Query cleaning and tokenisation.
      2. Resource and parameter validation.
      3. Semantic retrieval via FAISS IndexFlatIP.
      4. Lexical retrieval via BM25Okapi.
      5. Insertion-ordered, deduplicated merge of result ids.

    Does NOT perform fusion scoring, reranking, metadata filtering,
    recommendation generation, or any conversation logic.

    Parameters
    ----------
    query:
        Raw natural-language query string supplied by the caller.
    resources:
        A loaded RetrieverResources instance (from retriever_loader.py).
    semantic_top_k:
        Number of candidates to retrieve from FAISS.  Must be > 0.
    lexical_top_k:
        Number of candidates to retrieve from BM25.  Must be > 0.

    Returns
    -------
    HybridCandidates
        Contains semantic candidates, lexical candidates, and merged_ids.

    Raises
    ------
    ValidationError
        If the query, resources, or parameters fail validation.
    EncodingError
        If query encoding fails.
    RetrievalError
        If FAISS or BM25 retrieval raises an exception.
    """
    # ------------------------------------------------------------------ #
    # 1.  Query preprocessing
    # ------------------------------------------------------------------ #
    cleaned: str = clean_query(query)
    tokens: list[str] = tokenize_query(cleaned)

    logger.info(
        "retrieve() called | query=%r | tokens=%d | "
        "semantic_top_k=%d | lexical_top_k=%d",
        cleaned,
        len(tokens),
        semantic_top_k,
        lexical_top_k,
    )

    # ------------------------------------------------------------------ #
    # 2.  Validation
    # ------------------------------------------------------------------ #
    _validate_resources(resources)
    _validate_top_k(semantic_top_k, lexical_top_k)

    # ------------------------------------------------------------------ #
    # 3.  Resolve model and entity_lookup
    # ------------------------------------------------------------------ #
    model: Any = _get_model(resources)
    entity_lookup: dict[str, str] = _get_entity_lookup(resources)
    row_map: dict[int, dict[str, str]] = _build_row_to_entity(resources.mapping)

    # ------------------------------------------------------------------ #
    # 4.  Encode query
    # ------------------------------------------------------------------ #
    query_vec: np.ndarray = _encode_query(model, cleaned)

    # ------------------------------------------------------------------ #
    # 5.  Validate embedding dimension against FAISS index
    # ------------------------------------------------------------------ #
    _validate_embedding_dim(query_vec, resources.faiss_index)

    # ------------------------------------------------------------------ #
    # 6.  Semantic retrieval
    # ------------------------------------------------------------------ #
    semantic_candidates: list[SemanticCandidate] = _run_semantic_retrieval(
        faiss_index=resources.faiss_index,
        query_vec=query_vec,
        row_map=row_map,
        top_k=semantic_top_k,
    )

    # ------------------------------------------------------------------ #
    # 7.  Lexical retrieval
    # ------------------------------------------------------------------ #
    lexical_candidates: list[LexicalCandidate] = _run_lexical_retrieval(
        bm25=resources.bm25,
        tokens=tokens,
        entity_ids=resources.entity_ids,
        entity_lookup=entity_lookup,
        top_k=lexical_top_k,
    )

    # ------------------------------------------------------------------ #
    # 8.  Merge
    # ------------------------------------------------------------------ #
    merged: list[str] = _merge_ids(semantic_candidates, lexical_candidates)

    logger.info(
        "Retrieval complete | semantic=%d | lexical=%d | merged=%d",
        len(semantic_candidates),
        len(lexical_candidates),
        len(merged),
    )

    return HybridCandidates(
        semantic=semantic_candidates,
        lexical=lexical_candidates,
        merged_ids=merged,
    )
