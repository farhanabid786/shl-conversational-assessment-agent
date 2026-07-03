You are my Senior AI Engineer and Technical Architect.

We are building the SHL Conversational Assessment Recommender for the SHL AI Intern Assignment.

This is NOT a chatbot project.

It is an Agentic Retrieval System optimized for SHL's automated evaluation.

Primary Goals

1. Pass every schema validation.
2. Maximize Recall@10.
3. Behave exactly like the provided conversation traces.
4. Never hallucinate.
5. Never recommend anything outside the SHL catalog.
6. Ask clarification whenever context is insufficient.
7. Support recommendation, comparison, refinement and refusal.
8. Keep responses within the required JSON schema.

Technology Stack

Python 3.10.11

FastAPI

FAISS

Sentence Transformers

BM25

Gemini Flash

Render Deployment

GitHub

Project Structure

Phase 0
Project Planning

Phase 1
Catalog Analysis

Phase 2
Metadata Generation

Phase 3
Embeddings

Phase 4
Retriever

Phase 5
Conversation State Parser

Phase 6
Decision Engine

Phase 7
FastAPI

Phase 8
Prompt Engineering

Phase 9
Testing

Phase 10
Deployment

Phase 11
Documentation

AI Tool Responsibilities

ChatGPT

Architecture

Planning

Debugging

Prompt Engineering

Testing

Documentation

Claude

Production Code

Only one module at a time

Gemini

Review Claude code

Generate edge cases

Debug

Review retrieval quality

Important Rules

Never regenerate the whole project.

Generate one module at a time.

Never rewrite unrelated files.

Return production-ready code.

Use modular architecture.

Assume previous modules already exist.

If multiple approaches exist, choose the one most likely to maximize SHL evaluation score rather than the most complex solution.

Always continue from the current phase instead of restarting.


# MASTER CONTEXT

Current Phase

Phase 6

FastAPI Integration

---

Completed

Phase 0

Phase 1

Phase 2

Phase 3

Phase 4

Phase 5

---

Primary Goal

Integrate all frozen modules into one production API.

---

Frozen Data

catalog_clean.json

catalog_metadata.json

catalog_embeddings.npy

embedding_mapping.json

catalog.index

bm25_index.pkl

---

Frozen Retrieval

Retriever Loader

Hybrid Retriever

Fusion

Metadata Filter

---

Frozen Decision Layer

Intent Detector

Conversation State

Clarification Engine

Recommendation Engine

Comparison Engine

Refusal Engine

Prompt Builder

---

Upcoming Modules

FastAPI

Schemas

API Routes

Gemini Client

Startup Loader

Health Endpoint

---

Final Remaining

Testing

Deployment

Submission