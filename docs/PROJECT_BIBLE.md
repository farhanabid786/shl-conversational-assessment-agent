# SHL Conversational Assessment Recommender

## Project Objective

Build a conversational AI agent that recommends only SHL Individual Test Solutions.

The agent must:

- Ask clarification questions when context is insufficient.
- Recommend between 1 and 10 assessments.
- Compare assessments.
- Refine recommendations.
- Refuse off-topic or unsafe requests.
- Never hallucinate.
- Never recommend outside the SHL catalog.

---

# Technology Stack

Backend

- Python 3.10.11
- FastAPI

Retrieval

- Sentence Transformers
- FAISS
- BM25

LLM

- Gemini 2.5 Flash

Deployment

- Render

---

# Evaluation Targets

Must Pass

- Schema Validation
- Behavior Probes
- Recall@10
- Hidden Test Conversations

---

# Architecture

Conversation

↓

Conversation Parser

↓

Decision Engine

↓

Retriever

↓

Gemini Flash

↓

JSON Response

---

# Core Principles

- Stateless API
- Retrieval Augmented
- Grounded Responses
- No Hallucinations
- Modular Code
- Production Ready

## Phase 1 Summary

The SHL Product Catalog has been analyzed and transformed into a clean knowledge base.

Outputs

- catalog_statistics.json
- catalog_clean.json
- CATALOG_ANALYSIS.md

The cleaned catalog contains

- normalized durations
- normalized boolean fields
- search_text
- retrieval-ready records

The raw catalog is never modified.

Knowledge Base Status

Frozen

# Phase 2 Summary

The retrieval metadata layer has been successfully implemented.

Outputs

- catalog_metadata.json
- metadata_generator.py

The metadata layer is designed specifically for downstream retrieval.

Each assessment now includes:

- canonical_name
- normalized_name
- assessment_family
- searchable_text
- keywords
- filter_tokens
- ranking_tokens
- duration_minutes
- adaptive
- remote
- metadata_version

The canonical knowledge base remains immutable.

Current Retrieval Pipeline

Raw Catalog

↓

Catalog Cleaner

↓

catalog_clean.json

↓

Metadata Generator

↓

catalog_metadata.json

↓

Embeddings

↓

FAISS

↓

BM25

Status

Frozen

# Phase 3 Summary

The retrieval indexing pipeline has been completed.

Outputs

catalog_embeddings.npy

embedding_mapping.json

catalog.index

bm25_index.pkl

------------------------------------------------

Pipeline

catalog_clean.json

↓

catalog_metadata.json

↓

Embedding Generator

↓

FAISS Index

↓

BM25 Index

------------------------------------------------

Embedding Model

sentence-transformers/all-MiniLM-L6-v2

Similarity

Cosine Similarity

FAISS Index

IndexFlatIP

------------------------------------------------

Current Retrieval Assets

Knowledge Base

Retrieval Metadata

Dense Embeddings

FAISS Index

BM25 Index

------------------------------------------------

Status

Frozen

These assets should not be modified unless a genuine bug is discovered.