"""
app/config.py

Phase 6: Configuration
SHL Conversational Assessment Recommendation System

Centralizes all runtime configuration for the application. This module is
strictly configuration-only: it does not perform retrieval, call Gemini,
define FastAPI routes, or contain any business logic.

Values are loaded from environment variables / a `.env` file at the project
root via pydantic-settings. All filesystem paths are resolved relative to
the project root using pathlib.Path so the application behaves consistently
regardless of the current working directory it is launched from.

Deployment footprint
---------------------
The service targets a ~512MB deployed image/runtime budget. Two settings
below exist specifically to support that:

* EMBEDDING_MODEL defaults to a 3-layer MiniLM (~60MB on disk) rather than
  the 6-layer variant (~90MB). With well under 1,000 catalog rows, the
  extra capacity of the 6-layer model buys negligible recall — the smaller
  model is the better trade-off for this catalog size.
* TORCH_NUM_THREADS caps PyTorch's internal thread pool. Left unset,
  PyTorch sizes its thread pool to the host's core count, which inflates
  resident memory on multi-core containers for no throughput benefit at
  this request volume.

Python 3.10.11
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Project root
# --------------------------------------------------------------------------- #

# app/config.py -> app/ -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide runtime configuration.

    All values may be overridden via environment variables or a `.env` file
    located at the project root. Filesystem fields are stored as absolute,
    resolved `Path` objects relative to `PROJECT_ROOT`.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Application metadata ------------------------------------------- #
    APP_NAME: str = Field(
        default="SHL Conversational Assessment Recommendation System",
        description="Human-readable application name.",
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version (logged, not part of the API schema).",
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode (verbose errors, reload, etc.).",
    )
    HOST: str = Field(
        default="0.0.0.0",
        description="Host/interface the API server binds to.",
    )
    PORT: int = Field(
        default=8000,
        description="Port the API server listens on.",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Root log level for the application (e.g. DEBUG, INFO, WARNING).",
    )

    # --- Gemini ------------------------------------------------------------ #
    GEMINI_API_KEY: str = Field(
        default=...,
        description="API key used to authenticate against the Gemini API.",
    )
    GEMINI_MODEL: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model identifier used for generation.",
    )
    GEMINI_TIMEOUT_SECONDS: float = Field(
        default=20.0,
        description=(
            "Per-request timeout budget for Gemini calls. Kept comfortably "
            "under the evaluator's 30s per-call timeout so a slow Gemini "
            "response surfaces as a clean 503 instead of the evaluator's "
            "own timeout killing the connection first."
        ),
    )

    # --- Embeddings ---------------------------------------------------------#
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/paraphrase-MiniLM-L3-v2",
        description=(
            "SentenceTransformer model identifier used for embeddings. "
            "3-layer MiniLM: smaller download/RAM footprint than the "
            "6-layer default, chosen because the catalog is well under "
            "1,000 rows so the larger model's extra capacity is wasted."
        ),
    )
    EMBEDDING_DIMENSION: int = Field(
        default=384,
        description=(
            "Expected output dimension of EMBEDDING_MODEL. Used to validate "
            "that the FAISS index on disk was built with a compatible "
            "model before serving any traffic — both paraphrase-MiniLM-L3-v2 "
            "and all-MiniLM-L6-v2 happen to share this dimension, but if the "
            "model is changed again this catches a stale/mismatched index "
            "at startup instead of failing confusingly mid-query."
        ),
    )
    TORCH_NUM_THREADS: int = Field(
        default=1,
        description=(
            "Caps PyTorch's intra-op thread pool. At this request volume, "
            "a small fixed pool avoids the memory overhead of PyTorch "
            "auto-sizing threads to the host's core count."
        ),
    )
    EAGER_LOAD_MODEL: bool = Field(
        default=True,
        description=(
            "Load the embedding model synchronously during application "
            "startup (lifespan), not lazily on first request. Trades a "
            "slightly longer cold start (covered by the evaluator's 2 "
            "minute startup allowance) for predictable, uniform per-request "
            "latency and no first-request thundering-herd download risk."
        ),
    )

    # --- Filesystem paths --------------------------------------------------#
    DATA_DIR: Path = Field(
        default=PROJECT_ROOT / "data",
        description="Root directory containing all data artifacts.",
    )
    CATALOG_METADATA_PATH: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "catalog_metadata.json",
        description="Path to the cleaned/enriched catalog metadata JSON file.",
    )
    FAISS_INDEX_PATH: Path = Field(
        default=PROJECT_ROOT / "data" / "faiss" / "catalog.index",
        description="Path to the persisted FAISS vector index.",
    )
    BM25_INDEX_PATH: Path = Field(
        default=PROJECT_ROOT / "data" / "cache" / "bm25_index.pkl",
        description="Path to the persisted BM25 lexical index.",
    )
    EMBEDDING_MAPPING_PATH: Path = Field(
        default=PROJECT_ROOT / "data" / "embeddings" / "embedding_mapping.json",
        description="Path to the mapping between embedding rows and catalog IDs.",
    )

    # --- Conversation limits ------------------------------------------------#
    MAX_CONVERSATION_TURNS: int = Field(
        default=8,
        description=(
            "Mirrors the evaluator's hard turn cap (user + assistant "
            "combined). Kept here (in addition to app.schemas.MAX_MESSAGES) "
            "so the pipeline can decide end_of_conversation without "
            "importing the schemas module."
        ),
    )

    # --- Validators ---------------------------------------------------------#
    @field_validator("GEMINI_API_KEY")
    @classmethod
    def _validate_gemini_api_key(cls, value: str) -> str:
        """Ensure the Gemini API key is present and non-empty."""
        if not value or not value.strip():
            raise ValueError("GEMINI_API_KEY must be a non-empty string.")
        return value

    @field_validator(
        "DATA_DIR",
        "CATALOG_METADATA_PATH",
        "FAISS_INDEX_PATH",
        "BM25_INDEX_PATH",
        "EMBEDDING_MAPPING_PATH",
        mode="before",
    )
    @classmethod
    def _validate_non_empty_path(cls, value: str | Path) -> Path:
        """Ensure path fields are non-empty and resolved to absolute paths."""
        if value is None or str(value).strip() == "":
            raise ValueError("Path fields must be non-empty.")
        path = Path(value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Ensure the configured log level is a recognized logging level name."""
        normalized = value.strip().upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in valid_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of {sorted(valid_levels)}, got {value!r}."
            )
        return normalized

    @field_validator("GEMINI_TIMEOUT_SECONDS")
    @classmethod
    def _validate_gemini_timeout(cls, value: float) -> float:
        """Ensure the Gemini timeout leaves headroom under the evaluator's cap."""
        if value <= 0:
            raise ValueError("GEMINI_TIMEOUT_SECONDS must be positive.")
        if value > 28:
            raise ValueError(
                "GEMINI_TIMEOUT_SECONDS must leave headroom under the "
                "evaluator's 30s per-call timeout (recommended <= 25)."
            )
        return value

    @field_validator("TORCH_NUM_THREADS")
    @classmethod
    def _validate_torch_num_threads(cls, value: int) -> int:
        """Ensure the configured thread count is a positive integer."""
        if value < 1:
            raise ValueError("TORCH_NUM_THREADS must be >= 1.")
        return value

    @field_validator("MAX_CONVERSATION_TURNS")
    @classmethod
    def _validate_max_turns(cls, value: int) -> int:
        """Ensure the turn cap is a positive integer."""
        if value < 1:
            raise ValueError("MAX_CONVERSATION_TURNS must be >= 1.")
        return value


@lru_cache()
def get_settings() -> Settings:
    """Return a cached, process-wide `Settings` instance.

    The settings are constructed once and memoized via `lru_cache`, avoiding
    repeated environment/`.env` parsing on every access while keeping a
    single source of truth for configuration throughout the application.

    Returns:
        Settings: The validated application settings.
    """
    logger.debug("Loading application settings from environment / .env file.")
    return Settings()
