"""
app/schemas.py

Phase 6 - Schemas Module

Pydantic request/response models only.
No retrieval logic, no Gemini calls, no FastAPI routes, no business logic.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Message(BaseModel):
    """A single conversational message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Role of the message author.",
    )
    content: str = Field(
        ...,
        description="Text content of the message.",
    )

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("content must not be empty")
        return stripped


class RecommendationRequest(BaseModel):
    """Request payload for the recommendation endpoint."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(
        ...,
        description="Conversation history. The final message must be from the user.",
    )

    @field_validator("messages")
    @classmethod
    def messages_must_not_be_empty(cls, value: list[Message]) -> list[Message]:
        if not value:
            raise ValueError("messages must not be empty")
        if len(value) > 50:
            raise ValueError("messages must not exceed 50 items")
        return value

    @model_validator(mode="after")
    def final_message_must_be_user(self) -> "RecommendationRequest":
        if self.messages[-1].role != "user":
            raise ValueError("final message must have role 'user'")
        return self


class AssessmentCandidate(BaseModel):
    """A recommended assessment entity."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(
        ...,
        description="Unique identifier of the assessment entity.",
    )
    canonical_name: str = Field(
        ...,
        description="Canonical display name of the assessment.",
    )
    metadata: dict[str, Any] = Field(
        ...,
        description="Additional metadata associated with the assessment.",
    )


class ResponseMetadata(BaseModel):
    """Metadata describing how a response was produced."""

    model_config = ConfigDict(extra="forbid")

    intent: str = Field(
        ...,
        description="Detected intent for the conversation turn.",
    )
    clarification_required: bool = Field(
        ...,
        description="Whether clarification is required before recommending.",
    )
    recommendation_count: int = Field(
        ...,
        description="Number of recommendations returned.",
    )
    comparison_count: int = Field(
        ...,
        description="Number of assessments included in a comparison.",
    )
    refusal: bool = Field(
        ...,
        description="Whether the request was refused.",
    )


class RecommendationResponse(BaseModel):
    """Response payload for the recommendation endpoint."""

    model_config = ConfigDict(extra="forbid")

    response: str = Field(
        ...,
        description="Natural language response to the user.",
    )
    recommendations: list[AssessmentCandidate] = Field(
        ...,
        description="List of recommended assessment candidates.",
    )
    metadata: ResponseMetadata = Field(
        ...,
        description="Metadata describing how the response was produced.",
    )


class HealthResponse(BaseModel):
    """Response payload for the health check endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Overall service status.",
    )
    version: str = Field(
        ...,
        description="Application version.",
    )
    faiss_loaded: bool = Field(
        ...,
        description="Whether the FAISS index is loaded.",
    )
    bm25_loaded: bool = Field(
        ...,
        description="Whether the BM25 index is loaded.",
    )
    embedding_model_loaded: bool = Field(
        ...,
        description="Whether the embedding model is loaded.",
    )
    gemini_available: bool = Field(
        ...,
        description="Whether the Gemini client is available.",
    )


class ErrorResponse(BaseModel):
    """Standard error response payload."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(
        ...,
        description="Short error identifier or message.",
    )
    detail: str = Field(
        ...,
        description="Detailed error description.",
    )
