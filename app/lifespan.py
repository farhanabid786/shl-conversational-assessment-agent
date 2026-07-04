"""
app/lifespan.py

Phase 6: Application Lifespan
SHL Conversational Assessment Recommendation System

Implements the FastAPI application lifespan context manager. All shared,
process-wide resources — settings, retrieval artifacts (FAISS, BM25,
metadata), the embedding model, and the Gemini client — are loaded
exactly once during startup and released during shutdown. This module
contains no API routes, no business logic, and no retrieval or
generation execution.

Eager loading, by design
-------------------------
`_initialize_resources()` calls `load_retriever_resources()`, which loads
EVERY artifact — including the SentenceTransformer embedding model —
synchronously, before this context manager yields. There is no lazy,
first-request loading anywhere in this service: the process is either
fully ready when `/health` starts reporting "ok", or startup fails loudly
and the process should not be considered up. The evaluator explicitly
allows up to 2 minutes for a cold-started service to become healthy,
which comfortably covers this one-time cost.

Python 3.10.11
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from app.config import get_settings
from app.gemini_client import GeminiClient
from scripts.retriever_loader import (
    ModelLoadError,
    RetrieverLoaderError,
    RetrieverResources,
    load_retriever_resources,
)

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _cap_torch_thread_usage(settings: "Settings") -> None:
    """Cap PyTorch's intra-op thread pool before the model is constructed.

    Left unset, PyTorch sizes its thread pool to the host's core count,
    which inflates resident memory on multi-core containers with no
    throughput benefit at this service's request volume. Must run before
    the SentenceTransformer (and therefore torch) is initialised inside
    `load_retriever_resources()`, so this is called first, here, rather
    than left to library defaults.

    This is best-effort: if torch is not yet importable for any reason,
    we log and continue — `load_retriever_resources()` will surface a
    clear ModelLoadError shortly afterward if torch is genuinely missing.
    """
    try:
        import torch  # type: ignore[import]

        torch.set_num_threads(settings.TORCH_NUM_THREADS)
        logger.debug(
            "torch.set_num_threads(%d) applied before model load.",
            settings.TORCH_NUM_THREADS,
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "Could not cap torch thread usage ahead of model load; "
            "continuing with library defaults.",
            exc_info=True,
        )


def _initialize_resources(settings: "Settings") -> RetrieverResources:
    """Load and validate all retrieval artifacts required at query time.

    This includes the embedding model — loaded eagerly, once, here —
    not lazily on first request. Every field on the returned
    RetrieverResources is guaranteed populated by the time this function
    returns successfully.

    Parameters
    ----------
    settings:
        Application settings providing artifact paths, the embedding
        model identifier, and the expected embedding dimension.

    Returns
    -------
    RetrieverResources
        Frozen dataclass containing the catalog metadata, embedding
        mapping, FAISS index, BM25 index, and the eagerly-loaded
        embedding model.

    Raises
    ------
    RuntimeError
        If any retrieval artifact is missing, malformed, mutually
        inconsistent, or if the embedding model fails to load.
    """
    logger.info("Initializing retriever resources (eager, including model)")

    _cap_torch_thread_usage(settings)

    try:
        resources = load_retriever_resources(
            metadata_path=settings.CATALOG_METADATA_PATH,
            mapping_path=settings.EMBEDDING_MAPPING_PATH,
            faiss_index_path=settings.FAISS_INDEX_PATH,
            bm25_index_path=settings.BM25_INDEX_PATH,
            model_name=settings.EMBEDDING_MODEL,
            expected_embedding_dimension=settings.EMBEDDING_DIMENSION,
        )
    except ModelLoadError as exc:
        # Distinguished from other loader failures for operator clarity:
        # artifacts on disk were fine, but the embedding model itself
        # (download, dimension mismatch, etc.) could not be brought up.
        logger.exception(
            "Embedding model failed to load during eager startup."
        )
        raise RuntimeError(
            "Embedding model initialization failed — the service cannot "
            "start without it (no lazy fallback is used)."
        ) from exc
    except RetrieverLoaderError as exc:
        logger.exception("Failed to initialize retriever resources")
        raise RuntimeError("Retriever resource initialization failed") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Unexpected failure initializing retriever resources"
        )
        raise RuntimeError("Retriever resource initialization failed") from exc

    logger.info(
        "Retriever resources initialized: %d catalog records, "
        "embedding model loaded and ready",
        len(resources.metadata),
    )
    return resources


def _initialize_gemini(settings: "Settings") -> GeminiClient:
    """Instantiate the shared GeminiClient for the lifetime of the process.

    Parameters
    ----------
    settings:
        Application settings, used to configure the Gemini client
        (e.g. API key, model name, timeout budget).

    Returns
    -------
    GeminiClient
        A single, reusable Gemini client instance.

    Raises
    ------
    RuntimeError
        If the Gemini client cannot be initialized.
    """
    logger.info("Initializing Gemini client")
    try:
        gemini = GeminiClient(settings=settings)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize Gemini client")
        raise RuntimeError("Gemini client initialization failed") from exc

    logger.info("Gemini client initialized")
    return gemini


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan context manager.

    Loads all shared, process-wide resources once — synchronously,
    eagerly, including the embedding model — before the application
    starts serving requests, stores them on ``app.state``, and releases
    them on shutdown.

    Parameters
    ----------
    app:
        The FastAPI application instance.

    Yields
    ------
    None

    Raises
    ------
    RuntimeError
        If settings, retriever resources (including the embedding
        model), or the Gemini client fail to initialize. This aborts
        application startup — by design, a service that cannot embed
        queries or reach Gemini should never report itself as ready.
    """
    logger.info("=== Application startup: begin (eager resource load) ===")

    try:
        settings = get_settings()
        app.state.settings = settings

        resources = _initialize_resources(settings)
        app.state.resources = resources

        gemini = _initialize_gemini(settings)
        app.state.gemini = gemini
    except Exception as exc:  # noqa: BLE001
        logger.exception("Application startup failed")
        raise RuntimeError("Application startup failed") from exc

    logger.info("=== Application startup: complete — ready to serve ===")

    try:
        yield
    finally:
        logger.info("=== Application shutdown: begin ===")
        app.state.resources = None
        app.state.gemini = None
        logger.info("=== Application shutdown: complete ===")
