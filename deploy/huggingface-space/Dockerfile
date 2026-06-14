# syntax=docker/dockerfile:1.7

ARG SOURCE_REPO=https://github.com/xang1234/jobs-intelligence.git
ARG SOURCE_REF=master

FROM python:3.11-slim AS builder

ARG SOURCE_REPO
ARG SOURCE_REF

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.5

WORKDIR /build
RUN git clone --depth 1 --branch "$SOURCE_REF" "$SOURCE_REPO" app

WORKDIR /build/app
RUN poetry export -f requirements.txt --without-hashes --without dev --with ml -o requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM builder AS onnx-export

RUN poetry export -f requirements.txt --without-hashes --without dev --with ml,ml_torch -o requirements-onnx-export.txt
RUN pip install --no-cache-dir -r requirements-onnx-export.txt

ENV HF_HOME=/tmp/huggingface

RUN python -m src.cli embed-export-onnx all-MiniLM-L6-v2 --output-dir /opt/mcf/models/all-MiniLM-L6-v2-onnx --overwrite

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=onnx-export /opt/mcf/models/all-MiniLM-L6-v2-onnx /opt/mcf/models/all-MiniLM-L6-v2-onnx

WORKDIR /app
COPY --from=builder --chown=user:user /build/app/src ./src

ENV MCF_SEARCH_BACKEND=pgvector \
    MCF_LEAN_HOSTED=1 \
    MCF_EMBEDDING_BACKEND=onnx \
    MCF_ONNX_MODEL_DIR=/opt/mcf/models/all-MiniLM-L6-v2-onnx \
    MCF_RATE_LIMIT_RPM=100 \
    HF_HOME=/home/user/.cache/huggingface

USER user

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
