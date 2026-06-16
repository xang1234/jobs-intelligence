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
MCF_CORS_ORIGINS=https://jobs-intelligence.pages.dev,https://jobs.deepgradient.uk,https://deepgradient.uk,http://localhost:3000
MCF_RATE_LIMIT_RPM=100
SOURCE_REPO=https://github.com/xang1234/jobs-intelligence.git
SOURCE_REF=master
SOURCE_VERSION=<git-sha-to-check-out>
```

## Performance / latency checklist

The frontend lives on Cloudflare's edge but the API is US-hosted, so the
slow paths are (1) the trans-Pacific network leg on every call and (2)
scale-to-zero cold starts. Apply these:

### Hugging Face Space
- **Upgrade hardware** to a paid persistent tier and set **Sleep time = never**.
  Free Spaces sleep on inactivity; the first visitor afterward gets a failed
  request (self-signed cert) during the cold start.

### Neon database
- **Disable scale-to-zero** (or raise the suspend timeout) on the branch this
  Space uses. Free Neon auto-suspends after ~5 min idle, which is the main
  cause of multi-second `/api/overview` responses. Startup now pre-warms the
  expensive aggregations, but that only helps if the DB stays awake.
- **Keep Neon in the same region as the Space** (HF Spaces default to AWS
  `us-east-1`). The browser never talks to Neon directly — only the Space does —
  so co-locating them collapses the per-query HF↔Neon hop to ~1 ms. Never run
  the Space in the US with Neon in SG (or vice-versa); that pays the geography
  penalty twice. The nightly refresh job runs on GitHub's US runners, so
  US-Neon also speeds writes.

### Cloudflare Pages (frontend) — front the API through the edge
The browser calls the API **same-origin**, routed by Pages Functions that proxy
to this Space (`src/frontend/functions/api/[[path]].ts`, `functions/health.ts`).
TLS then terminates at the nearest edge and public GETs are edge-cached.
- Set **`VITE_API_BASE_URL=/`** in the Pages project (or remove the variable —
  `.env.production` already defaults it to `/`). If it's still set to the
  `*.hf.space` URL the browser bypasses the proxy and pays the US handshake.
- Optionally set **`API_ORIGIN`** (Pages env var) to this Space's URL; the
  functions default to it otherwise.
- Public read endpoints send `Cache-Control: public, s-maxage=600,
  stale-while-revalidate=86400`, so Cloudflare serves repeat hits from the edge
  with no origin round-trip. Build assets are `immutable` via
  `src/frontend/public/_headers`.
