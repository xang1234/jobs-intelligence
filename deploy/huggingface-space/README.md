---
title: Jobs Intelligence API
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Jobs Intelligence API

FastAPI backend for the Singapore hiring-market intelligence app.

The container connects to a Neon Postgres hosted slice via the `DATABASE_URL`
Space secret. Runtime configuration is managed through Hugging Face Space
variables.

Required secret:

```text
DATABASE_URL=postgresql://...
```

Required variables:

```text
MCF_SEARCH_BACKEND=pgvector
MCF_LEAN_HOSTED=1
MCF_EMBEDDING_BACKEND=onnx
MCF_CORS_ORIGINS=https://jobs.deepgradient.uk,https://deepgradient.uk
MCF_RATE_LIMIT_RPM=100
SOURCE_REPO=https://github.com/xang1234/jobs-intelligence.git
SOURCE_REF=claude/enhance-ui-ux-design-CNnPV
```
