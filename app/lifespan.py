"""
app/lifespan.py

Phase 6: Application Lifespan
SHL Conversational Assessment Recommendation System

Implements the FastAPI application lifespan context manager.  All shared,
process-wide resources (settings, retrieval artifacts, and the Gemini
client) are loaded exactly once during startup and released during
shutdown.  This module contains no API routes, no business logic, and no
retrieval or generation execution.

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
from scripts.retriever_loader import RetrieverResources, load_retriever_resources

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _initialize_resources() -> RetrieverResources:
    """Load and validate all retrieval artifacts required at query time.

    Returns
    -------
    RetrieverResources
        Frozen dataclass containing the catalog metadata, embedding
        mapping, FAISS index, BM25 index, and the pre-loaded embedding
        model.

    Raises
    ------
    RuntimeError
        If any retrieval artifact is missing, malformed, or mutually
        inconsistent.
    """
    logger.info("Initializing retriever resources")
    try:
        resources = load_retriever_resources()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize retriever resources")
        raise RuntimeError("Retriever resource initialization failed") from exc

    logger.info(
        "Retriever resources initialized: %d catalog records",
        len(resources.metadata),
    )
    return resources


def _initialize_gemini(settings: "Settings") -> GeminiClient:
    """Instantiate the shared GeminiClient for the lifetime of the process.

    Parameters
    ----------
    settings:
        Application settings, used to configure the Gemini client
        (e.g. API key, model name, timeouts).

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

    Loads all shared, process-wide resources once before the application
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
        If settings, retriever resources, or the Gemini client fail to
        initialize. This aborts application startup.
    """
    logger.info("=== Application startup: begin ===")

    try:
        settings = get_settings()
        app.state.settings = settings

        resources = _initialize_resources()
        app.state.resources = resources

        gemini = _initialize_gemini(settings)
        app.state.gemini = gemini
    except Exception as exc:  # noqa: BLE001
        logger.exception("Application startup failed")
        raise RuntimeError("Application startup failed") from exc

    logger.info("=== Application startup: complete ===")

    try:
        yield
    finally:
        logger.info("=== Application shutdown: begin ===")
        app.state.resources = None
        app.state.gemini = None
        logger.info("=== Application shutdown: complete ===")
