# ---------------------------------------------------------------------------
# Medical Diagnosis Assistant — Submission Dockerfile
# Serves on port 8080 as required by the challenge spec.
# ---------------------------------------------------------------------------

FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager used by this project)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install all dependencies from the lock file (no-sync keeps it reproducible)
RUN uv sync --frozen --no-dev

# Copy application source
COPY src/server.py retrieval_and_reranking.py chunking_and_embedding.py parse_icd_codes.py icd_descriptions.json embeddings.npz evaluate.py ./

# Copy and unpack pre-built index files
COPY large_files.zip ./
RUN unzip large_files.zip && rm large_files.zip

# BGE-M3 and reranker model weights (cache them at build time so inference is
# fully offline — no external network calls during the container run).
# The models are downloaded to the HuggingFace cache directory.
RUN uv run python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-m3")
snapshot_download("BAAI/bge-reranker-v2-m3")
EOF

# Runtime environment
ENV CORPUS_FILE=processed_corpus.json \
    EMBED_FILE=embeddings.npz \
    SPARSE_FILE=sparse_weights.json \
    BM25_FILE=bm25_index.pkl \
    ICD_DESC_FILE=icd_descriptions.json \
    HUB_URL="" \
    API_KEY=""

EXPOSE 8080

# Run with uvicorn on port 8080 (challenge requirement)
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
