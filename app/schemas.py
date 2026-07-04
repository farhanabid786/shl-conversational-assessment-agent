"""
app/schemas.py

Phase 6 - Schemas Module
SHL Conversational Assessment Recommendation System

Pydantic request/response models ONLY. No retrieval logic, no Gemini
calls, no FastAPI routes, no business logic.

IMPORTANT — Output contract
----------------------------
The response shapes below are dictated verbatim by the assignment spec
and are NON-NEGOTIABLE:

    POST /chat ->
        {
          "reply": "<natural language text>",
          "recommendations": [
              {"name": "...", "url": "...", "test_type": "..."}
          ],
          "end_of_conversation": false
        }

    GET /health ->
        {"status": "ok"}          (HTTP 200)

`recommendations` is an empty list while the agent is still gathering
context, clarifying, or refusing — and an array of 1 to 10 items once a
shortlist has been produced. `end_of_conversation` is true only when the
agent considers the task complete (a shortlist was delivered, or the
turn cap was reached). Any deviation from this shape breaks the
automated evaluator.

Python 3.10.11
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #

# The evaluator caps each conversation at 8 turns (user + assistant combined)
# and each call at a 30 second timeout. MAX_MESSAGES is deliberately a little
# above the hard cap so a slightly-over-cap request still gets a clean 422
# instead of silently truncating history the client actually sent.
MAX_MESSAGES: int = 8
MAX_RECOMMENDATIONS: int = 10


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


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


class ChatRequest(BaseModel):
    """Request payload for POST /chat.

    The API is stateless: every call carries the full conversation
    history and the service stores no per-conversation state.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(
        ...,
        description=(
            "Full, stateless conversation history. The final message "
            "must be from the 'user' role."
        ),
    )

    @field_validator("messages")
    @classmethod
    def messages_must_not_be_empty(cls, value: list[Message]) -> list[Message]:
        if not value:
            raise ValueError("messages must not be empty")
        if len(value) > MAX_MESSAGES:
            raise ValueError(
                f"messages must not exceed {MAX_MESSAGES} items "
                "(evaluator turn cap)"
            )
        return value

    @model_validator(mode="after")
    def final_message_must_be_user(self) -> "ChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("final message must have role 'user'")
        return self


# Backwards-compatible alias — some existing call sites may still import
# the old name. Prefer ChatRequest in new code.
RecommendationRequest = ChatRequest


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class RecommendationItem(BaseModel):
    """A single recommended SHL assessment, exactly as required by spec.

    Deliberately minimal: only the three fields the evaluator checks.
    Internal fields (entity_id, full catalog metadata, scores, ranks)
    are never serialized to the client — see scripts.prompt_builder
    .display_metadata for the analogous internal whitelist used when
    building the Gemini prompt.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Canonical display name of the assessment.",
    )
    url: str = Field(
        ...,
        description=(
            "Catalog URL for the assessment. Must be a URL that was "
            "actually scraped into the catalog — never fabricated."
        ),
    )
    test_type: str = Field(
        ...,
        description="SHL test-type code for the assessment (e.g. 'K', 'P').",
    )


class ChatResponse(BaseModel):
    """Response payload for POST /chat. Shape is fixed by the spec."""

    model_config = ConfigDict(extra="forbid")

    reply: str = Field(
        ...,
        description="Natural language reply to show the user.",
    )
    recommendations: list[RecommendationItem] = Field(
        default_factory=list,
        description=(
            "0 items while clarifying/refusing/gathering context; "
            "1-10 items once a shortlist has been produced."
        ),
    )
    end_of_conversation: bool = Field(
        ...,
        description="True only when the agent considers the task complete.",
    )

    @field_validator("recommendations")
    @classmethod
    def recommendations_must_not_exceed_ten(
        cls, value: list[RecommendationItem]
    ) -> list[RecommendationItem]:
        if len(value) > MAX_RECOMMENDATIONS:
            raise ValueError(
                f"recommendations must not exceed {MAX_RECOMMENDATIONS} items"
            )
        return value


# Backwards-compatible alias.
RecommendationResponse = ChatResponse


class HealthResponse(BaseModel):
    """Response payload for GET /health.

    Kept deliberately minimal — the spec requires exactly
    {"status": "ok"} on a healthy, ready service. Extra internal detail
    (which artifacts are loaded, Gemini reachability, etc.) is logged
    server-side rather than exposed here, so it can never cause a
    schema mismatch with the evaluator.
    """

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Service readiness status. 'ok' when ready to serve.",
    )


class ErrorResponse(BaseModel):
    """Standard error response payload (used for 4xx/5xx documentation)."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(
        ...,
        description="Short error identifier or message.",
    )
    detail: str = Field(
        ...,
        description="Detailed error description.",
    )
