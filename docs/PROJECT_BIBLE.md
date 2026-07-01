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