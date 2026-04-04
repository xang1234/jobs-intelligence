# ── Stage 1: Build ────────────────────────────────────────────────────────────
# Install compilers + build deps, export requirements, pip install everything.
# This stage is discarded — only site-packages are carried forward.

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry just long enough to export requirements.txt
RUN pip install --no-cache-dir poetry==1.8.5

WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Export pinned requirements (without dev deps, without hashes for compat)
RUN poetry export -f requirements.txt --without-hashes --without dev --with ml -o requirements.txt

# Phase 2 production image installs only ONNX inference/runtime dependencies.
# Torch and sentence-transformers move to the optional ml_torch group for
# export/dev workflows outside the production container.
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 1b: Export bundled ONNX model ──────────────────────────────────────
# Build the default ONNX bundle once during image build, then copy only the
# exported artifact into the runtime image. The throwaway stage keeps torch and
# sentence-transformers out of production.

FROM builder AS onnx-export

RUN poetry export -f requirements.txt --without-hashes --without dev --with ml,ml_torch -o requirements-onnx-export.txt
RUN pip install --no-cache-dir -r requirements-onnx-export.txt

WORKDIR /app
COPY src/ ./src/

ENV HF_HOME=/tmp/huggingface

RUN python -m src.cli embed-export-onnx all-MiniLM-L6-v2 --output-dir /opt/mcf/models/all-MiniLM-L6-v2-onnx --overwrite

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# Clean slim image with only the installed packages + application source.

FROM python:3.11-slim AS runtime

# libgomp1: OpenMP runtime — FAISS uses it for multi-threaded search.
# curl: used by HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=onnx-export /opt/mcf/models/all-MiniLM-L6-v2-onnx /opt/mcf/models/all-MiniLM-L6-v2-onnx

WORKDIR /app

# Copy application source
COPY src/ ./src/

# Environment: the ONNX bundle lives outside /app/data so host volume mounts do
# not shadow it. Database and FAISS indexes still come from the mounted data/
# volume in docker-compose.yml.
ENV MCF_DB_PATH=/app/data/mcf_jobs.db \
    MCF_INDEX_DIR=/app/data/embeddings \
    MCF_EMBEDDING_BACKEND=onnx \
    MCF_ONNX_MODEL_DIR=/opt/mcf/models/all-MiniLM-L6-v2-onnx \
    MCF_RATE_LIMIT_RPM=100 \
    MCF_SQLITE_JOURNAL_MODE=delete \
    HF_HOME=/app/.cache/huggingface

EXPOSE 8000

# start-period=120s: FAISS/pgvector index loading takes 5-30s on startup;
# the ONNX session/tokenizer still initialize lazily on first search.
# 120s accommodates cold starts on Oracle A1 with pgvector connections.
# Failures during start-period don't count toward retries.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker: each worker duplicates FAISS indexes in RAM (~2GB).
# Concurrency comes from the async event loop + thread pool executor.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
