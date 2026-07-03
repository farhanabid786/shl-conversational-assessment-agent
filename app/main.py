"""
app/main.py

Phase 6
FastAPI Application Entry Point

SHL Conversational Assessment Recommendation System

Responsibilities
----------------
* Create the FastAPI application.
* Configure application metadata.
* Register lifespan handlers.
* Register API routes.
* Configure logging.

This module MUST NOT contain:
- Retrieval logic
- Recommendation logic
- Gemini logic
- Business logic
- Prompt generation

Python 3.10.11
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.lifespan import lifespan
from app.routes import router

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Conversational Assessment Recommendation System "
        "for SHL assessment discovery."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production if required.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------

app.include_router(router)

# ---------------------------------------------------------------------
# Startup Log
# ---------------------------------------------------------------------

logger.info("%s initialized successfully.", settings.APP_NAME)

# ---------------------------------------------------------------------
# Local Development Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )