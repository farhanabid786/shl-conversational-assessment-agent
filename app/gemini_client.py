"""
app/gemini_client.py

Phase 6: Gemini Client
SHL Conversational Assessment Recommendation System

Thin wrapper around the Gemini SDK. This module contains no business
logic, no retrieval logic, no FastAPI, and no prompt generation. It is
responsible only for configuring the SDK, issuing generation requests
built from an already-assembled `PromptPayload`, reporting client
health, and releasing SDK resources.

Python 3.10.11
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from app.config import Settings
from scripts.prompt_builder import PromptPayload

if TYPE_CHECKING:
    from google.genai import Client

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class GeminiClientError(Exception):
    """Base exception for all GeminiClient errors."""


class GeminiConfigurationError(GeminiClientError):
    """Raised when the Gemini SDK cannot be configured or initialized."""


class GeminiGenerationError(GeminiClientError):
    """Raised when a Gemini generation request fails or returns no usable text."""


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class GeminiClient:
    """Thin, stateless-per-request wrapper around the Gemini SDK.

    Holds a single configured SDK `Client` instance for the lifetime of
    the process. All business logic (what to say, how to route, what
    data to include) lives upstream in the prompt builder; this class only
    knows how to send a `PromptPayload` to Gemini and return the resulting
    text.
    """

    def __init__(self, settings: Settings) -> None:
        """Configure the Gemini SDK and construct the shared client.

        Args:
            settings: Application settings providing the Gemini API key
                and model identifier.

        Raises:
            GeminiConfigurationError: If the SDK cannot be configured or
                the client cannot be constructed.
        """
        if settings is None or not isinstance(settings, Settings):
            raise GeminiConfigurationError(
                f"settings must be a Settings instance, got {type(settings).__name__}."
            )

        self._settings = settings
        self._api_key: str = settings.GEMINI_API_KEY
        self._model_name: str = settings.GEMINI_MODEL
        self._client: "Client"

        try:
            self._client = genai.Client(api_key=self._api_key)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to configure Gemini client")
            raise GeminiConfigurationError(
                "Failed to configure Gemini client"
            ) from exc

        logger.info("GeminiClient initialized (model=%s)", self._model_name)

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #

    def generate(self, prompt_payload: PromptPayload) -> str:
        """Send an assembled prompt to Gemini and return the generated text.

        Args:
            prompt_payload: The deterministic system/user prompt assembled
                by the prompt builder.

        Returns:
            The generated response text.

        Raises:
            GeminiGenerationError: If `prompt_payload` is invalid, the
                request fails, or Gemini returns no usable text.
        """
        if prompt_payload is None or not isinstance(prompt_payload, PromptPayload):
            raise GeminiGenerationError(
                f"prompt_payload must be a PromptPayload, got "
                f"{type(prompt_payload).__name__}."
            )

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt_payload.user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=prompt_payload.system_prompt,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini generation request failed")
            raise GeminiGenerationError("Gemini generation request failed") from exc

        text = getattr(response, "text", None)
        if not text or not text.strip():
            logger.error("Gemini returned an empty or missing response.")
            raise GeminiGenerationError("Gemini returned an empty response.")

        return text

    def health_check(self) -> bool:
        """Verify that the Gemini client can successfully reach the API.

        Returns:
            True if a minimal generation request succeeds and returns
            non-empty text, False otherwise. Never raises.
        """
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents="ping",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Gemini health check failed")
            return False

        text = getattr(response, "text", None)
        healthy = bool(text and text.strip())
        if not healthy:
            logger.warning("Gemini health check returned no usable text.")
        return healthy

    def close(self) -> None:
        """Release Gemini SDK resources.

        The underlying SDK does not expose async cleanup or persistent
        connections requiring teardown, so this is a no-op provided for
        interface symmetry with other process-wide resources.
        """
        logger.info("GeminiClient close() called (no-op).")
