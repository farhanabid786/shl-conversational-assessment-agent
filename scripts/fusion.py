"""
scripts/fusion.py

Phase 4 – Hybrid Retrieval: Fusion Layer
SHL Conversational Assessment Recommendation System

Combines semantic and lexical retrieval results using Reciprocal Rank Fusion
(RRF).  This module is strictly scoped to fusion; it performs no metadata
filtering, LLM reranking, Gemini calls, recommendation generation,
clarification, or conversation logic.

RRF Formula
-----------
    RRF score = Σ  1 / (k + rank)

where k defaults to 60 and rank is the 1-based position within each retrieval
list.  Original similarity scores are deliberately ignored.

Python 3.10.11
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
# )
# logger = logging.getLogger("fusion")
logger = logging.getLogger(__name__)
# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_K: int = 60

# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #


class FusionError(Exception):
    """Base exception for all fusion failures."""


class ValidationError(FusionError):
    """Raised when input validation fails before or after fusion."""


# --------------------------------------------------------------------------- #
# Input Candidate Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SemanticCandidate:
    """A single result from semantic (FAISS) retrieval.

    Attributes
    ----------
    entity_id:
        Unique identifier for the assessment.
    canonical_name:
        Human-readable assessment name.
    score:
        Cosine similarity or inner-product score from FAISS (not used by RRF).
    rank:
        1-based position within the semantic result list.
    """

    entity_id: str
    canonical_name: str
    score: float
    rank: int


@dataclass(frozen=True)
class LexicalCandidate:
    """A single result from lexical (BM25) retrieval.

    Attributes
    ----------
    entity_id:
        Unique identifier for the assessment.
    canonical_name:
        Human-readable assessment name.
    score:
        BM25 score (not used by RRF).
    rank:
        1-based position within the lexical result list.
    """

    entity_id: str
    canonical_name: str
    score: float
    rank: int


@dataclass(frozen=True)
class HybridCandidates:
    """Container produced by the upstream Hybrid Retriever.

    Attributes
    ----------
    semantic:
        Ordered list of SemanticCandidate objects (rank-ascending).
    lexical:
        Ordered list of LexicalCandidate objects (rank-ascending).
    merged_ids:
        Union of entity_ids seen across both lists (order not guaranteed).
    """

    semantic: list[SemanticCandidate]
    lexical: list[LexicalCandidate]
    merged_ids: list[str]


# --------------------------------------------------------------------------- #
# Output Candidate Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FusionCandidate:
    """A single fused result with its RRF score and source ranks.

    Attributes
    ----------
    entity_id:
        Unique identifier for the assessment.
    canonical_name:
        Human-readable assessment name.
    rrf_score:
        Accumulated Reciprocal Rank Fusion score across all retrieval lists.
    semantic_rank:
        Rank from semantic retrieval, or None if absent from that list.
    lexical_rank:
        Rank from lexical retrieval, or None if absent from that list.
    """

    entity_id: str
    canonical_name: str
    rrf_score: float
    semantic_rank: Optional[int] = field(default=None)
    lexical_rank: Optional[int] = field(default=None)


@dataclass(frozen=True)
class FusionResult:
    """Immutable container for the fully fused and ranked candidates.

    Attributes
    ----------
    candidates:
        List of FusionCandidate objects sorted by RRF score (descending),
        with deterministic tie-breaking on semantic_rank, lexical_rank, and
        entity_id.
    """

    candidates: list[FusionCandidate]

    @property
    def entity_ids(self) -> list[str]:
        """Return ordered list of entity_ids from the fused ranking."""
        return [c.entity_id for c in self.candidates]


# --------------------------------------------------------------------------- #
# Validation Helpers
# --------------------------------------------------------------------------- #


def _validate_hybrid_candidates(candidates: HybridCandidates) -> None:
    """Raise ValidationError if *candidates* is structurally invalid.

    Checks performed
    ----------------
    - ``candidates`` is a HybridCandidates instance.
    - ``candidates.semantic`` is a list (may be empty).
    - ``candidates.lexical`` is a list (may be empty).
    - ``candidates.merged_ids`` is a non-empty list.
    - Ranks within each list are positive integers.
    - No duplicate entity_ids within the semantic list.
    - No duplicate entity_ids within the lexical list.
    """
    if not isinstance(candidates, HybridCandidates):
        raise ValidationError(
            f"Expected HybridCandidates, got {type(candidates).__name__!r}."
        )

    if not isinstance(candidates.semantic, list):
        raise ValidationError(
            "HybridCandidates.semantic must be a list, "
            f"got {type(candidates.semantic).__name__!r}."
        )

    if not isinstance(candidates.lexical, list):
        raise ValidationError(
            "HybridCandidates.lexical must be a list, "
            f"got {type(candidates.lexical).__name__!r}."
        )

    if not isinstance(candidates.merged_ids, list) or not candidates.merged_ids:
        raise ValidationError(
            "HybridCandidates.merged_ids must be a non-empty list."
        )

    # Validate semantic candidates
    seen_semantic: set[str] = set()
    for idx, sc in enumerate(candidates.semantic):
        if not isinstance(sc, SemanticCandidate):
            raise ValidationError(
                f"candidates.semantic[{idx}] is not a SemanticCandidate "
                f"(got {type(sc).__name__!r})."
            )
        if sc.rank < 1:
            raise ValidationError(
                f"candidates.semantic[{idx}] has invalid rank {sc.rank!r}; "
                "ranks must be >= 1."
            )
        if sc.entity_id in seen_semantic:
            raise ValidationError(
                f"Duplicate entity_id {sc.entity_id!r} in semantic candidate list."
            )
        seen_semantic.add(sc.entity_id)

    # Validate lexical candidates
    seen_lexical: set[str] = set()
    for idx, lc in enumerate(candidates.lexical):
        if not isinstance(lc, LexicalCandidate):
            raise ValidationError(
                f"candidates.lexical[{idx}] is not a LexicalCandidate "
                f"(got {type(lc).__name__!r})."
            )
        if lc.rank < 1:
            raise ValidationError(
                f"candidates.lexical[{idx}] has invalid rank {lc.rank!r}; "
                "ranks must be >= 1."
            )
        if lc.entity_id in seen_lexical:
            raise ValidationError(
                f"Duplicate entity_id {lc.entity_id!r} in lexical candidate list."
            )
        seen_lexical.add(lc.entity_id)

    logger.debug(
        "HybridCandidates validated: %d semantic, %d lexical, %d merged",
        len(candidates.semantic),
        len(candidates.lexical),
        len(candidates.merged_ids),
    )


def _validate_k(k: int) -> None:
    """Raise ValidationError if *k* is not a positive integer."""
    if not isinstance(k, int) or k <= 0:
        raise ValidationError(
            f"RRF constant k must be a positive integer, got {k!r}."
        )


def _validate_no_duplicate_output(candidates: list[FusionCandidate]) -> None:
    """Raise ValidationError if the output list contains duplicate entity_ids."""
    seen: set[str] = set()
    for fc in candidates:
        if fc.entity_id in seen:
            raise ValidationError(
                f"Duplicate entity_id {fc.entity_id!r} detected in fusion output."
            )
        seen.add(fc.entity_id)


# --------------------------------------------------------------------------- #
# RRF Core Logic
# --------------------------------------------------------------------------- #


def _build_name_lookup(candidates: HybridCandidates) -> dict[str, str]:
    """Return a mapping from entity_id → canonical_name.

    Semantic names take precedence over lexical names when both are present,
    since semantic candidates are the primary retrieval source.
    """
    name_lookup: dict[str, str] = {}

    # Lexical first (lower precedence)
    for lc in candidates.lexical:
        name_lookup[lc.entity_id] = lc.canonical_name

    # Semantic second (higher precedence — overwrites if conflict)
    for sc in candidates.semantic:
        name_lookup[sc.entity_id] = sc.canonical_name

    return name_lookup


def _accumulate_rrf_scores(
    candidates: HybridCandidates,
    k: int,
) -> dict[str, dict[str, float | int | None]]:
    """Accumulate RRF contributions into a per-entity-id accumulator dict.

    Returns
    -------
    dict mapping entity_id to::

        {
            "rrf_score":     float,
            "semantic_rank": int | None,
            "lexical_rank":  int | None,
        }
    """
    acc: dict[str, dict[str, float | int | None]] = {}

    def _get_or_create(eid: str) -> dict[str, float | int | None]:
        if eid not in acc:
            acc[eid] = {"rrf_score": 0.0, "semantic_rank": None, "lexical_rank": None}
        return acc[eid]

    # Semantic contributions
    for sc in candidates.semantic:
        entry = _get_or_create(sc.entity_id)
        entry["rrf_score"] = float(entry["rrf_score"]) + 1.0 / (k + sc.rank)
        entry["semantic_rank"] = sc.rank
        logger.debug(
            "Semantic  entity=%-40s rank=%3d  contribution=%.6f",
            sc.entity_id,
            sc.rank,
            1.0 / (k + sc.rank),
        )

    # Lexical contributions
    for lc in candidates.lexical:
        entry = _get_or_create(lc.entity_id)
        entry["rrf_score"] = float(entry["rrf_score"]) + 1.0 / (k + lc.rank)
        entry["lexical_rank"] = lc.rank
        logger.debug(
            "Lexical   entity=%-40s rank=%3d  contribution=%.6f",
            lc.entity_id,
            lc.rank,
            1.0 / (k + lc.rank),
        )

    return acc


def _sort_key(fc: FusionCandidate) -> tuple[float, int, int, str]:
    """Return a deterministic sort key for a FusionCandidate.

    Ordering criteria (applied left-to-right)
    ------------------------------------------
    1. Highest rrf_score  (negated for ascending sort)
    2. Lowest semantic_rank  (None → treated as worst possible rank)
    3. Lowest lexical_rank   (None → treated as worst possible rank)
    4. entity_id ascending   (final deterministic tie-breaker)
    """
    _MAX_RANK = 10_000_000

    s_rank = fc.semantic_rank if fc.semantic_rank is not None else _MAX_RANK
    l_rank = fc.lexical_rank if fc.lexical_rank is not None else _MAX_RANK

    return (-fc.rrf_score, s_rank, l_rank, fc.entity_id)


def _build_fusion_candidates(
    acc: dict[str, dict[str, float | int | None]],
    name_lookup: dict[str, str],
) -> list[FusionCandidate]:
    """Construct and sort FusionCandidate objects from the accumulator."""
    candidates: list[FusionCandidate] = []

    for entity_id, entry in acc.items():
        canonical_name = name_lookup.get(entity_id, entity_id)
        fc = FusionCandidate(
            entity_id=entity_id,
            canonical_name=canonical_name,
            rrf_score=float(entry["rrf_score"]),
            semantic_rank=entry["semantic_rank"],   # type: ignore[arg-type]
            lexical_rank=entry["lexical_rank"],      # type: ignore[arg-type]
        )
        candidates.append(fc)

    candidates.sort(key=_sort_key)
    return candidates


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def fuse(
    candidates: HybridCandidates,
    k: int = DEFAULT_K,
) -> FusionResult:
    """Apply Reciprocal Rank Fusion to *candidates* and return a FusionResult.

    Parameters
    ----------
    candidates:
        HybridCandidates produced by the upstream Hybrid Retriever.
    k:
        RRF smoothing constant.  Must be a positive integer.  The default of
        60 follows the original Cormack et al. (2009) recommendation and is
        well-suited for SHL catalog retrieval.

    Returns
    -------
    FusionResult
        Immutable result containing fused candidates sorted by RRF score,
        with deterministic tie-breaking.

    Raises
    ------
    ValidationError
        If *candidates* or *k* fails validation, or if the output contains
        duplicate entity_ids (should never occur under correct input).
    FusionError
        For any unexpected failure during fusion.
    """
    logger.info(
        "Starting RRF fusion — semantic=%d, lexical=%d, k=%d",
        len(candidates.semantic),
        len(candidates.lexical),
        k,
    )

    # --- Input validation -------------------------------------------------- #
    _validate_hybrid_candidates(candidates)
    _validate_k(k)

    # --- Accumulate RRF scores --------------------------------------------- #
    acc = _accumulate_rrf_scores(candidates, k)

    # --- Build canonical name lookup --------------------------------------- #
    name_lookup = _build_name_lookup(candidates)

    # --- Construct and sort output ----------------------------------------- #
    fused: list[FusionCandidate] = _build_fusion_candidates(acc, name_lookup)

    # --- Output validation ------------------------------------------------- #
    _validate_no_duplicate_output(fused)

    result = FusionResult(candidates=fused)

    logger.info(
        "RRF fusion complete — %d unique candidates in output",
        len(result.candidates),
    )
    if result.candidates:
        top = result.candidates[0]
        logger.info(
            "Top candidate: entity_id=%s  rrf_score=%.6f  "
            "semantic_rank=%s  lexical_rank=%s",
            top.entity_id,
            top.rrf_score,
            top.semantic_rank,
            top.lexical_rank,
        )

    return result
