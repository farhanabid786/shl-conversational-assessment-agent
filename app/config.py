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

    # --- Embeddings ---------------------------------------------------------#
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model identifier used for embeddings.",
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
