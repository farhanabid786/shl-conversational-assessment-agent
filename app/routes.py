"""
app/routes.py

Phase 6
FastAPI Routes

SHL Conversational Assessment Recommendation System

Responsibilities
----------------
* Expose HTTP endpoints.
* Validate request payloads.
* Invoke the Pipeline.
* Convert domain exceptions into HTTP responses.

This module MUST NOT contain:
- Retrieval logic
- Recommendation logic
- Gemini logic
- Prompt generation
- Business rules

Output contract (non-negotiable, per assignment spec)
-------------------------------------------------------
POST /chat ->
    {
      "reply": "...",
      "recommendations": [{"name": "...", "url": "...", "test_type": "..."}],
      "end_of_conversation": false
    }

GET /health -> {"status": "ok"}  (HTTP 200 when ready)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas import (
    ChatRequest,
    ChatResponse,
    RecommendationItem,
    HealthResponse,
    ErrorResponse,
)
from app.pipeline import (
    Pipeline,
    PipelineError,
    ServiceUnavailableError,
    ValidationError,
)

from app.gemini_client import (
    GeminiClientError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["SHL Assessment Recommendation System"]
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_pipeline(request: Request) -> Pipeline:
    """
    Construct a Pipeline instance using application resources.

    Raises
    ------
    HTTPException
        If application resources are unavailable.
    """

    resources = getattr(request.app.state, "resources", None)
    gemini = getattr(request.app.state, "gemini", None)
    settings = getattr(request.app.state, "settings", None)

    if resources is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retriever resources are unavailable.",
        )

    if gemini is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini client is unavailable.",
        )

    max_turns = getattr(settings, "MAX_CONVERSATION_TURNS", 8) if settings else 8

    return Pipeline(
        resources=resources,
        gemini_client=gemini,
        max_conversation_turns=max_turns,
    )


def _convert_messages(
    request_model: ChatRequest,
) -> list[dict[str, str]]:
    """
    Convert validated Pydantic message models into the format
    expected by the pipeline.
    """

    return [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request_model.messages
    ]


def _to_recommendation_item(item: Any) -> RecommendationItem:
    """Convert one internal candidate object into the spec-exact shape.

    ``item`` is a FilteredCandidate (recommendation path) or ComparisonItem
    (comparison path) produced upstream — both expose ``.canonical_name``
    and ``.metadata`` (the full catalog record dict for that entity).

    Requires ``item.metadata`` to carry ``url`` and ``test_type``. These
    fields are populated by scripts.metadata_generator when building
    catalog_metadata.json — see that module's docstring for the exact
    source fields (``link`` -> ``url``) and the category -> test_type
    code mapping. Falls back to an empty string (never raises, never
    fabricates a URL) if either field is missing from an older/stale
    catalog_metadata.json, and logs a warning so the gap is visible in
    server logs without breaking the response schema.
    """
    metadata = item.metadata or {}

    url = str(metadata.get("url", "")).strip()
    test_type = str(metadata.get("test_type", "")).strip()

    if not url:
        logger.warning(
            "Recommendation %r is missing 'url' in catalog metadata — "
            "returning an empty string. Regenerate catalog_metadata.json "
            "via scripts.metadata_generator to fix this permanently.",
            getattr(item, "canonical_name", "<unknown>"),
        )
    if not test_type:
        logger.warning(
            "Recommendation %r is missing 'test_type' in catalog metadata — "
            "returning an empty string. Regenerate catalog_metadata.json "
            "via scripts.metadata_generator to fix this permanently.",
            getattr(item, "canonical_name", "<unknown>"),
        )

    return RecommendationItem(
        name=item.canonical_name,
        url=url,
        test_type=test_type,
    )


def _build_response(result: Any) -> ChatResponse:
    """
    Convert PipelineResult into the spec-exact ChatResponse.

    Only the three required fields are serialized — no internal routing
    metadata (intent, path, counts) is exposed to the client, since the
    schema is fixed and any extra top-level field would violate it
    (ChatResponse uses ConfigDict(extra="forbid")).
    """

    recommendations = [
        _to_recommendation_item(item) for item in result.recommendations
    ]

    return ChatResponse(
        reply=result.reply,
        recommendations=recommendations,
        end_of_conversation=result.end_of_conversation,
    )


# ---------------------------------------------------------------------
# Chat Endpoint
# ---------------------------------------------------------------------


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Generate SHL assessment recommendations",
)
def chat(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    """
    Generate SHL assessment recommendations for one conversational turn.

    Workflow

        Validate request
            ↓
        Build Pipeline
            ↓
        Execute Pipeline
            ↓
        Return ChatResponse {reply, recommendations, end_of_conversation}

    This endpoint contains NO business logic. The API is stateless: the
    full conversation history is supplied on every call and no
    per-conversation state is retained between requests.
    """

    logger.debug("Received /chat request.")

    pipeline = _build_pipeline(request)

    messages = _convert_messages(payload)

    try:
        result = pipeline.run(messages)

        logger.debug("/chat request completed successfully.")

        return _build_response(result)

    except ValidationError as exc:
        logger.warning("Validation error: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except GeminiClientError as exc:
        # Defensive: normally unreachable, since Pipeline.run() already
        # catches GeminiClientError and re-raises it as
        # ServiceUnavailableError below. Kept in case a GeminiClientError
        # is ever raised outside of pipeline.run() (e.g. future call sites).
        logger.error("Gemini client failure.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini service is currently unavailable.",
        ) from exc

    except ServiceUnavailableError as exc:
        # Dependency-availability failure (embedding model missing, or
        # Gemini unreachable/timed out) — the process itself is fine,
        # but a required dependency is temporarily unusable.
        logger.error("Required dependency unavailable: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except PipelineError as exc:
        logger.error("Pipeline execution failed.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected /chat failure.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal server error.",
        ) from exc


# ---------------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Health Check",
)
def health(request: Request) -> HealthResponse:
    """
    Service readiness endpoint. Returns {"status": "ok"} (HTTP 200) once
    startup has completed successfully.

    Deliberately minimal per spec: no internal detail (which artifacts
    are loaded, Gemini reachability, etc.) is exposed in the body — that
    detail is logged server-side instead, so it can never cause a schema
    mismatch with the evaluator.

    Readiness is a structural check only (are resources/gemini present on
    app.state, both of which are only ever set after
    app.lifespan.lifespan() completes its eager startup sequence — model
    included). It deliberately does NOT make a live Gemini API call on
    every health check: since loading is eager, "present at all" already
    means "constructed successfully at startup"; probing Gemini on every
    poll would add needless latency and API cost without materially
    improving the signal.
    """

    resources = getattr(request.app.state, "resources", None)
    gemini = getattr(request.app.state, "gemini", None)

    ready = (
        resources is not None
        and getattr(resources, "faiss_index", None) is not None
        and getattr(resources, "bm25", None) is not None
        and getattr(resources, "model", None) is not None
        and gemini is not None
    )

    if not ready:
        logger.warning("Health check: service not yet ready.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is starting up or a required dependency is unavailable.",
        )

    return HealthResponse(status="ok")


# ---------------------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------------------


@router.get(
    "/",
    summary="API Root",
)
def root() -> dict[str, str]:
    """
    Root endpoint.
    """

    return {
        "service": "SHL Conversational Assessment Recommendation System",
        "status": "running",
    }
