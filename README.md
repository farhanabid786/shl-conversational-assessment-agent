# SHL Conversational Assessment Recommender

A conversational agent that turns a vague hiring need ("I'm hiring a Java developer who works with stakeholders") into a grounded shortlist of SHL assessments — through dialogue, not keyword search. Built for the SHL AI Intern take-home assignment.

**Live API:** https://shl-conversational-assessment-agent-3e3v.onrender.com
**Repo:** https://github.com/farhanabid786/shl-conversational-assessment-agent

> Confirm this is the currently-deployed URL before sharing it anywhere else (repeated redeploys can change the Render subdomain) — see [Deployment](#deployment).

---

## What it does

The agent handles four conversational behaviors over a stateless, multi-turn API:

| Behavior | Example |
|---|---|
| **Clarify** | "I need an assessment" → asks a targeted follow-up instead of guessing |
| **Recommend** | Enough context gathered → returns 1–10 assessments with name, URL, and test type |
| **Refine** | "Actually, add personality tests" → updates the shortlist, doesn't restart it |
| **Compare** | "What's the difference between OPQ and GSA?" → answers from catalog data, not the model's prior knowledge |

It stays in scope: SHL assessments only. General hiring advice, legal questions, and prompt-injection attempts are refused. Every URL returned was scraped from the real SHL catalog — never fabricated.

---

## API

Stateless — every request carries the full conversation history; the service holds no per-conversation state.

### `POST /chat`

```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure — what seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/products/product-catalog/view/java-8-new/", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/opq32r/", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` while clarifying, refusing, or still gathering context; 1–10 items once a shortlist has actually been produced.
- `end_of_conversation` is `true` only when a shortlist/comparison was delivered, or the conversation has reached the turn cap (default 8, see [Configuration](#configuration)) — never on a mid-conversation clarify or a single-turn refusal.

### `GET /health`

```json
{"status": "ok"}
```

Returns `200` once the embedding model, FAISS index, BM25 index, and Gemini client have all finished loading at startup; `503` while still starting up or if a dependency is unavailable. No internal detail is exposed in the body — that's logged server-side instead, so it can never cause a schema mismatch with a client or evaluator parsing this response.

---

## Architecture

```
Catalog Cleaning → Metadata Generation → Sentence-Transformer Embeddings
   → FAISS + BM25 Hybrid Retrieval → Reciprocal Rank Fusion → Metadata Filtering
   → Intent Detection & Conversation State → Clarify / Recommend / Compare / Refuse
   → Prompt Builder → Gemini → FastAPI Response
```

Retrieval is deterministic and always runs before Gemini is invoked. Gemini never sees the raw catalog or free-form retrieval — only a small, whitelisted set of already-filtered candidates — so it can converse and justify, but it cannot recommend anything that wasn't actually retrieved.

**Key design decisions:**
- **Eager loading, not lazy.** The embedding model, FAISS index, and BM25 index all load once, synchronously, at process startup — before `/health` ever reports `ok`. No first-request download risk, no locking needed, uniform per-request latency.
- **Small embedding model.** `sentence-transformers/paraphrase-MiniLM-L3-v2` (3-layer, 384-dim, ~60MB) — the catalog is well under 1,000 rows, so a larger 6-layer model buys negligible recall at a real cost in RAM and latency.
- **Deployment-budget aware.** Targets Render's free tier (512MB RAM): CPU-only torch, model baked into the Docker image at build time (no runtime download), capped PyTorch thread pool, single uvicorn worker (multiple workers would each hold a full duplicate copy of the model/indices in memory).
- **Minimal, whitelisted output.** Only `name`, `url`, and `test_type` are ever serialized per recommendation — internal fields (embeddings, ranking tokens, fusion scores, full catalog records) never reach the client or bloat the Gemini prompt.

---

## Project structure

```
SHL-Conversational-Recommender/
├── app/                        # FastAPI application layer
│   ├── prompts/                # Prompt templates
│   ├── config.py                # Settings (env-driven, pydantic-settings)
│   ├── gemini_client.py         # Gemini API wrapper
│   ├── lifespan.py               # Eager startup: loads resources + Gemini client once
│   ├── main.py                    # FastAPI app entrypoint
│   ├── pipeline.py                 # Orchestrates intent → retrieval → decision → generation
│   ├── routes.py                    # /chat, /health, / endpoints
│   ├── schemas.py                    # Request/response models (the API contract)
│   └── test_model.py
│
├── scripts/                    # Retrieval + data-pipeline logic (offline + runtime)
│   ├── catalog_cleaner.py       # raw catalog -> catalog_clean.json
│   ├── catalog_analyzer.py       # catalog_clean.json -> catalog_statistics.json (informational)
│   ├── metadata_generator.py      # catalog_clean.json -> catalog_metadata.json (adds url, test_type)
│   ├── embedding_generator.py      # catalog_metadata.json -> embeddings + mapping
│   ├── faiss_index_builder.py       # embeddings + mapping -> catalog.index
│   ├── bm25_index_builder.py         # catalog_metadata.json -> bm25_index.pkl
│   ├── retriever_loader.py            # Loads + validates all artifacts at startup (eager)
│   ├── hybrid_retriever.py             # FAISS + BM25 retrieval
│   ├── fusion.py                        # Reciprocal Rank Fusion
│   ├── metadata_filter.py                # Constraint filtering (job level, duration, etc.)
│   ├── intent_detector.py                 # Classifies user intent per turn
│   ├── conversation_state.py               # Tracks constraints/history across turns
│   ├── clarification_engine.py              # Decides when/what to ask
│   ├── recommendation_engine.py              # Produces the final shortlist
│   ├── comparison_engine.py                   # Grounded assessment-vs-assessment answers
│   ├── refusal_engine.py                       # Out-of-scope / injection refusal logic
│   └── prompt_builder.py                        # Builds the constrained Gemini prompt
│
├── data/
│   ├── raw/                    # shl_catalog.json (scraped, untouched)
│   ├── processed/               # catalog_clean.json, catalog_metadata.json, catalog_statistics.json
│   ├── embeddings/               # catalog_embeddings.npy, embedding_mapping.json
│   ├── faiss/                     # catalog.index
│   └── cache/                      # bm25_index.pkl
│
├── docs/                        # Design notes, roadmap, and process docs
│   ├── architecture.md
│   ├── AI_WORKFLOW.md
│   ├── CATALOG_ANALYSIS.md
│   ├── claude_prompts.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── DEVELOPMENT_ROADMAP.md
│   ├── LESSONS_LEARNED.md
│   ├── MASTER_CONTEXT.md
│   ├── PROJECT_BIBLE.md
│   ├── submission_checklist.md
│   └── todo.md
│
├── tests/
├── Dockerfile                   # CPU-only torch, model baked in at build time, single worker
├── .dockerignore                 # ⚠️ verify this is named with the leading dot — see note below
├── render.yaml                     # Render Blueprint for one-click deploy
├── requirements.txt
├── runtime.txt                      # Pinned Python version for the build environment
├── .env                               # Local secrets (never committed)
└── .gitignore
```

> **Housekeeping note:** your Explorer shows a root file named `dockerignore` (no leading dot). Docker only auto-applies exclusions from a file named exactly `.dockerignore` — if the leading dot is actually missing on disk, rename it, or the build context will silently include `venv/`, `tests/`, `docs/`, and `data/raw/` instead of excluding them. Also worth relocating `data/embeddings/test.py` into `tests/` — a test file living inside a data-artifact folder is easy to lose track of and looks like it was dropped there by accident.

---

## Configuration

All settings are environment-driven (`app/config.py`, loaded via `pydantic-settings` from `.env` locally or the platform's env vars in deployment).

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Gemini API authentication |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Generation model |
| `GEMINI_TIMEOUT_SECONDS` | `20` | Per-call Gemini timeout (kept under the evaluator's 30s cap) |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-MiniLM-L3-v2` | Embedding model (must match the model used to build `catalog.index`) |
| `EMBEDDING_DIMENSION` | `384` | Validated against the FAISS index at startup |
| `TORCH_NUM_THREADS` | `1` | Caps PyTorch's thread pool (memory, not just CPU) |
| `EAGER_LOAD_MODEL` | `true` | No lazy-loading path exists; kept as a documented invariant |
| `MAX_CONVERSATION_TURNS` | `8` | Mirrors the evaluator's turn cap; drives `end_of_conversation` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address (Render overrides `PORT` at runtime) |
| `LOG_LEVEL` | `INFO` | Root log level |
| `DEBUG` | `false` | Enables reload/verbose errors locally |

---

## Running locally

```bash
git clone https://github.com/farhanabid786/shl-conversational-assessment-agent
cd SHL-Conversational-Recommender
python -m venv venv && source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY
```

### Rebuilding the data artifacts

Only needed if `data/raw/shl_catalog.json` changes, or the embedding model changes. Run in order — each step consumes the previous step's output:

```bash
python -m scripts.catalog_cleaner        # data/raw -> data/processed/catalog_clean.json
python -m scripts.catalog_analyzer       # optional, informational stats only
python -m scripts.metadata_generator     # -> data/processed/catalog_metadata.json
python -m scripts.embedding_generator    # -> data/embeddings/*.npy, *.json
python -m scripts.faiss_index_builder    # -> data/faiss/catalog.index
python -m scripts.bm25_index_builder     # -> data/cache/bm25_index.pkl
```

### Start the server

```bash
python -m app.main
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

---

## Deployment

Deployed on **Render** as a persistent Docker web service (not a serverless platform like Vercel — the eager-loading design needs a long-lived process holding the model/indices in memory, not per-invocation cold starts).

1. Push to GitHub with `Dockerfile`, `.dockerignore`, and `render.yaml` at the repo root.
2. Render dashboard → **New +** → **Blueprint** → connect the repo → Render reads `render.yaml` and configures the service automatically.
3. Enter `GEMINI_API_KEY` when prompted (kept out of the repo).
4. Health Check Path is set to `/health` — Render won't route traffic until it returns `200`.

Free tier: 512MB RAM, 0.1 CPU, spins down after 15 minutes of inactivity, ~30–60s cold start on the next request — within the evaluator's stated 2-minute allowance.

---

## Evaluation

Evaluated against the 10 provided public conversation traces plus additional custom multi-turn scenarios:

- **Schema compliance** — every response validates against the fixed `{reply, recommendations, end_of_conversation}` contract; 0–10 recommendation items; turn cap honored.
- **Recall@10** — retrieved shortlist vs. each trace's labeled expected assessments.
- **Groundedness** — every returned `url` is verified to exist in the scraped catalog; no fabricated names or links.
- **Behavior probes** — refuses off-topic/legal/injection asks, never recommends on turn 1 for a vague query, honors mid-conversation edits without discarding prior constraints.

## What didn't work

- Dense retrieval alone missed exact lexical matches (product codes); BM25 alone missed semantic paraphrases — neither sufficient alone, which is why both feed Reciprocal Rank Fusion.
- Letting Gemini recommend directly (pre-retrieval) increased hallucinated names/URLs — recommending is now retrieval-only; Gemini only converses around a fixed candidate set.
- Lazy, first-request model loading caused inconsistent latency under concurrent load — replaced with eager, startup-time loading.
- Serializing full catalog metadata into the prompt/response bloated token cost and payload size — trimmed to the minimal whitelisted fields.

---

## Tech stack

FastAPI · Pydantic v2 / pydantic-settings · sentence-transformers (`paraphrase-MiniLM-L3-v2`) · FAISS (`IndexFlatIP`, cosine similarity) · `rank-bm25` · Google Gemini (`google-genai`) · Docker · Render

## AI tools used

**ChatGPT** — architecture, planning, prompt engineering, debugging. **Claude** — modular production code generation, refactoring, schema-contract correction, deployment configuration. **Google Gemini** — code review, retrieval validation, edge-case analysis. All AI-assisted output was manually reviewed, tested end-to-end, and integrated before submission.

---

## License

Built for the SHL AI Intern take-home assignment. Not affiliated with or endorsed by SHL.
