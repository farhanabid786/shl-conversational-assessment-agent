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
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    AssessmentCandidate,
    ResponseMetadata,
    HealthResponse,
    ErrorResponse,
)
from app.pipeline import (
    Pipeline,
    PipelineError,
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

    return Pipeline(
        resources=resources,
        gemini_client=gemini,
    )


def _convert_messages(
    request_model: RecommendationRequest,
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


def _build_response(
    result: Any,
) -> RecommendationResponse:
    """
    Convert PipelineResult into RecommendationResponse.
    """

    recommendations = [
        AssessmentCandidate(
            entity_id=item.entity_id,
            canonical_name=item.canonical_name,
            metadata=item.metadata,
        )
        for item in result.recommendations
    ]

    metadata = ResponseMetadata(
        intent=result.metadata["intent"],
        clarification_required=result.metadata["clarification_required"],
        recommendation_count=result.metadata["recommendation_count"],
        comparison_count=result.metadata["comparison_count"],
        refusal=result.metadata["refusal"],
    )

    return RecommendationResponse(
        response=result.response,
        recommendations=recommendations,
        metadata=metadata,
    )

# ---------------------------------------------------------------------
# Recommendation Endpoint
# ---------------------------------------------------------------------


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Generate SHL assessment recommendations",
)
def recommend(
    payload: RecommendationRequest,
    request: Request,
) -> RecommendationResponse:
    """
    Generate SHL assessment recommendations.

    Workflow

        Validate request
            ↓
        Build Pipeline
            ↓
        Execute Pipeline
            ↓
        Return RecommendationResponse

    This endpoint contains NO business logic.
    """

    logger.info("Received recommendation request.")

    pipeline = _build_pipeline(request)

    messages = _convert_messages(payload)

    try:
        result = pipeline.run(messages)

        logger.info(
            "Recommendation request completed successfully."
        )

        return _build_response(result)

    except ValidationError as exc:
        logger.warning("Validation error: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except GeminiClientError as exc:
        logger.error("Gemini client failure.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini service is currently unavailable.",
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
        logger.exception("Unexpected recommendation failure.")

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
    summary="Health Check",
)
def health(
    request: Request,
) -> HealthResponse:
    """
    Service health endpoint.

    Verifies that the major application resources have been
    initialized successfully.
    """

    resources = getattr(request.app.state, "resources", None)
    gemini = getattr(request.app.state, "gemini", None)
    # settings = getattr(request.app.state, "settings", None)

    faiss_loaded = False
    bm25_loaded = False
    embedding_model_loaded = False
    gemini_available = False

    if resources is not None:
        faiss_loaded = getattr(resources, "faiss_index", None) is not None
        bm25_loaded = getattr(resources, "bm25", None) is not None
        embedding_model_loaded = getattr(resources, "model", None) is not None

    if gemini is not None:
        try:
            gemini_available = gemini.health_check()
        except Exception:  # noqa: BLE001
            logger.warning("Gemini health check failed.")
            gemini_available = False

    overall_status = (
        "healthy"
        if (
            faiss_loaded
            and bm25_loaded
            and embedding_model_loaded
            and gemini_available
        )
        else "degraded"
    )

    version = "1.0.0"

    # if settings is not None:
    #     version = getattr(settings, "APP_VERSION", version)

    return HealthResponse(
        status=overall_status,
        version=version,
        faiss_loaded=faiss_loaded,
        bm25_loaded=bm25_loaded,
        embedding_model_loaded=embedding_model_loaded,
        gemini_available=gemini_available,
    )


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