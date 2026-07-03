# SHL Conversational Assessment Recommendation System

## Project Bible

Version: v0.5

---

# Vision

Build a production-quality conversational assessment recommendation system for the SHL AI Intern Assignment.

The system is designed as an Agentic Retrieval Pipeline rather than a traditional chatbot.

Primary objectives:

- Recommend only assessments from the official SHL catalog.
- Maximize Recall@10.
- Never hallucinate.
- Support recommendation, comparison, clarification, refinement, and refusal.
- Produce deterministic behavior before invoking Gemini.
- Keep latency low for automated evaluation.
- Maintain a modular, production-ready architecture.

---

# Core Design Principles

## Grounded Retrieval

Every recommendation originates from the SHL catalog.

Gemini never invents assessments.

---

## Separation of Responsibilities

Knowledge Base

↓

Retrieval

↓

Decision Logic

↓

Prompt Construction

↓

LLM

↓

API

Each layer has a single responsibility.

---

## Frozen Data Pipeline

catalog_clean.json

↓

catalog_metadata.json

↓

Embeddings

↓

FAISS

↓

BM25

↓

Hybrid Retrieval

↓

Decision Layer

Only downstream modules consume upstream outputs.

---

# Technology Stack

Python 3.10.11

FastAPI

Sentence Transformers

FAISS

BM25Okapi

Gemini Flash

Render

GitHub

---

# Completed Phases

## Phase 0

Architecture

Environment

Documentation

Planning

---

## Phase 1

Knowledge Base

catalog_clean.json

Status

Frozen

---

## Phase 2

Metadata Layer

catalog_metadata.json

Status

Frozen

---

## Phase 3

Embeddings

FAISS

BM25

Status

Frozen

---

## Phase 4

Hybrid Retrieval

Retriever Loader

Fusion

Metadata Filtering

Status

Frozen

---

## Phase 5

Decision Layer

Intent Detection

Conversation State

Clarification

Recommendation

Comparison

Refusal

Prompt Builder

Status

Frozen

---

# Current Architecture

User Query

↓

Intent Detector

↓

Conversation State

↓

Clarification Engine

↓

Hybrid Retrieval

↓

Fusion

↓

Metadata Filter

↓

Decision Engines

↓

Prompt Builder

↓

Gemini Flash

↓

Structured Response

---

# Phase 6 Goal

Integrate all modules into FastAPI.

---

# Coding Standards

• Type hints everywhere

• Modular architecture

• Logging

• Frozen dataclasses

• Custom exceptions

• Deterministic outputs

• Separation of concerns

---

# Rule

Frozen phases are never modified unless a genuine bug is discovered.