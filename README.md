# SHL Conversational Assessment Recommendation System

An AI-powered Retrieval-Augmented Generation (RAG) conversational assistant that recommends SHL assessments using natural language conversations. The system combines hybrid retrieval (FAISS + BM25), metadata filtering, conversation understanding, and Google's Gemini model to generate grounded assessment recommendations exclusively from the official SHL assessment catalog.

---

## Features

- Conversational SHL assessment recommendations
- Hybrid Retrieval (Semantic + Lexical Search)
- FAISS Vector Search
- BM25 Keyword Search
- Reciprocal Rank Fusion (RRF)
- Metadata-based filtering
- Multi-turn conversation support
- Clarification for vague queries
- Recommendation refinement
- Assessment comparison
- Out-of-scope request refusal
- FastAPI REST API
- Gemini-powered grounded response generation

---

## System Architecture

```
User
   │
   ▼
FastAPI (/chat)
   │
   ▼
Pipeline
   │
   ├── Conversation State
   ├── Intent Detection
   ├── Clarification Engine
   ├── Hybrid Retriever
   │      ├── FAISS
   │      └── BM25
   ├── Metadata Filter
   ├── Recommendation Engine
   ├── Comparison Engine
   ├── Refusal Engine
   ├── Prompt Builder
   └── Gemini Client
```

---

## Project Structure

```
SHL-Conversational-Recommender/

├── app/
│   ├── main.py
│   ├── routes.py
│   ├── pipeline.py
│   ├── lifespan.py
│   ├── config.py
│   ├── gemini_client.py
│   └── schemas.py
│
├── scripts/
│   ├── conversation_state.py
│   ├── intent_detector.py
│   ├── clarification_engine.py
│   ├── recommendation_engine.py
│   ├── comparison_engine.py
│   ├── refusal_engine.py
│   ├── prompt_builder.py
│   ├── metadata_filter.py
│   ├── hybrid_retriever.py
│   ├── fusion.py
│   ├── retriever_loader.py
│   ├── embedding_generator.py
│   ├── faiss_index_builder.py
│   ├── bm25_index_builder.py
│   ├── catalog_cleaner.py
│   └── metadata_generator.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   ├── faiss/
│   └── cache/
│
├── docs/
├── tests/
├── requirements.txt
├── README.md
└── .env
```

---

## Technology Stack

### Backend

- Python 3.10
- FastAPI
- Pydantic
- Uvicorn

### AI / ML

- Google Gemini
- Sentence Transformers
- FAISS
- BM25
- NumPy
- Scikit-learn

### Retrieval

- Semantic Search
- Lexical Search
- Reciprocal Rank Fusion

---

## Dataset

The system uses the complete SHL assessment catalog.

Current indexed catalog:

- **377 assessments**

Artifacts generated:

- Catalog Metadata
- Embedding Mapping
- FAISS Index
- BM25 Index

---

## Retrieval Pipeline

```
User Query

      │

      ▼

Conversation State

      │

      ▼

Intent Detection

      │

      ▼

Hybrid Retrieval

 ┌─────────────┐
 │             │
 ▼             ▼
FAISS        BM25

 └─────┬───────┘
       ▼

Reciprocal Rank Fusion

       ▼

Metadata Filtering

       ▼

Recommendation Engine

       ▼

Prompt Builder

       ▼

Gemini

       ▼

Response
```

---

## API Endpoints

### Health Check

```
GET /health
```

Example Response

```json
{
    "status": "healthy",
    "version": "1.0.0"
}
```

---

### Chat Endpoint

```
POST /chat
```

Example Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Recommend assessments for a Python Backend Developer."
    }
  ]
}
```

Example Response

```json
{
  "reply": "...",
  "recommendations": [
    {
      "entity_id": "4123",
      "canonical_name": "Python (New)"
    }
  ]
}
```

---

## Supported Conversational Behaviors

### Clarification

```
User:
I need an assessment.
```

Assistant asks follow-up questions before recommending.

---

### Recommendation

```
Recommend assessments for a Python Backend Developer.
```

Returns relevant SHL assessments.

---

### Refinement

```
Actually add personality tests.
```

Updates the recommendation list using previous conversation context.

---

### Comparison

```
Compare OPQ and GSA.
```

Returns grounded comparison from catalog metadata.

---

### Refusal

Rejects:

- Prompt Injection
- Hacking
- Cheating Requests
- Legal Advice
- Medical Advice
- General Hiring Advice
- Non-SHL Topics

---

## Installation

Clone repository

```bash
git clone https://github.com/your-username/SHL-Conversational-Recommender.git

cd SHL-Conversational-Recommender
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env`

```
GEMINI_API_KEY=YOUR_API_KEY

GEMINI_MODEL=gemini-2.5-flash
```

---

## Run Locally

```bash
python -m uvicorn app.main:app --reload
```

Swagger

```
http://127.0.0.1:8000/docs
```

Health

```
http://127.0.0.1:8000/health
```

---

## Deployment

Recommended Platform

- Render

Start Command

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

## Evaluation

The project was evaluated using:

- Functional Testing
- Retrieval Validation
- FastAPI Endpoint Testing
- End-to-End Pipeline Testing
- Recommendation Quality
- Comparison Validation
- Clarification Flow
- Refusal Handling

---

## Future Improvements

- Streaming responses
- Redis conversation cache
- Better ranking optimization
- Multi-language support
- Feedback-based reranking
- Continuous catalog synchronization

---

## Author

**Farhan Abid**

B.Tech Computer Science & Engineering (Artificial Intelligence)

Babu Banarasi Das University

GitHub: https://github.com/farhanabid786

LinkedIn: https://linkedin.com/in/farhan-abid-8001a9253

---

## License

This project was developed as part of the SHL AI Assessment Recommendation Challenge for educational and evaluation purposes.