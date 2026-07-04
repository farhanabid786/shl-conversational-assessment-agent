# Dockerfile
# SHL Conversational Assessment Recommendation System
#
# Built specifically for Render's free Web Service tier (512 MB RAM,
# 0.1 CPU). Two things matter most for staying inside that budget:
#
#   1. Install the CPU-ONLY torch wheel explicitly. Letting pip resolve
#      "torch" from a generic requirements.txt on some platforms can pull
#      in CUDA runtime libraries that are hundreds of MB larger than the
#      CPU-only build — and this service never uses a GPU.
#   2. Run uvicorn with exactly ONE worker. The embedding model, FAISS
#      index, and BM25 index are all loaded once into this process's
#      memory at startup (see app/lifespan.py) and never lazily reloaded.
#      Each additional worker would be a full separate process with its
#      own copy of everything in memory — on a 512MB box that's an
#      almost-guaranteed OOM kill with more than one worker.

FROM python:3.10-slim AS base

WORKDIR /app

# faiss-cpu and torch both ship prebuilt wheels for this platform, so no
# compiler toolchain should be required — but keep this minimal build-essential
# install as a fallback in case any transitive dependency needs to compile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- CPU-only torch, installed BEFORE the rest of requirements.txt so it
# never gets re-resolved as a CUDA build by a transitive dependency ---
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# --- Remaining Python dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Bake the embedding model into the image at BUILD time ---
# Downloading ~60MB from HuggingFace on every cold start (after Render's
# free-tier 15-minute sleep) would slow every wake-up and depend on
# runtime network reliability. Doing it once, here, at build time means
# cold starts only need to read from local disk.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')"

# --- Application code ---
COPY app/ app/
COPY scripts/ scripts/

# --- Data artifacts actually needed at runtime ---
# Deliberately NOT copying data/raw/ (only used by the offline
# catalog_cleaner.py step) or data/cache's intermediate files beyond
# bm25_index.pkl — keeps the image lean.
COPY data/processed/catalog_metadata.json data/processed/catalog_metadata.json
COPY data/embeddings/catalog_embeddings.npy data/embeddings/catalog_embeddings.npy
COPY data/embeddings/embedding_mapping.json data/embeddings/embedding_mapping.json
COPY data/faiss/catalog.index data/faiss/catalog.index
COPY data/cache/bm25_index.pkl data/cache/bm25_index.pkl

EXPOSE 8000

# Render injects $PORT at runtime — bind to it directly rather than relying
# on app.config.Settings.PORT, so the container always listens on the port
# Render's load balancer actually expects.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
