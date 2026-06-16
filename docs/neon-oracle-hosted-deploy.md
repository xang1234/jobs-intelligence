# Neon + Oracle/Hugging Face Hosted Deployment

This guide deploys the hosted Jobs Intelligence stack as small, separate
services:

- `Neon` hosts the lean Postgres serving database with `pgvector`
- `GitHub Actions` refreshes the hosted Neon slice on a schedule
- `Hugging Face Spaces` or `Oracle Cloud Always Free` hosts the FastAPI API
- `Cloudflare Pages` hosts the static React frontend

Local Postgres remains the full archive and the source of truth. The hosted
database is intentionally bounded to roughly the most recent 90 days of current
year jobs:

```text
posted_date >= max(Jan 1 of current year, today - 90 days)
```

The hosted slice keeps only `job` embeddings. `skill` and `company` embeddings
remain local-only.

## Current Hosted Shape

The simplest working deployment is:

- API: `https://xang1234-jobs-intelligence-api.hf.space`
- frontend: `https://jobs-intelligence.pages.dev`
- custom domain: optional

`jobs.deepgradient.uk` is not required. Keeping the `.pages.dev` URL is fine as
long as the API CORS allow-list includes `https://jobs-intelligence.pages.dev`.

## 1. Prerequisites

You should have:

- a local Postgres archive
- Poetry dependencies installed locally
- a Neon project
- a GitHub repo with Actions enabled
- either a Hugging Face account or an Oracle Cloud account
- optionally, a Cloudflare account for the frontend

Useful local values:

```bash
export LOCAL_DATABASE_URL='postgresql://postgres@127.0.0.1:55432/mcf'
export NEON_DATABASE_URL='postgresql://<neon-dsn>'
```

If you use the helper deploy scripts, keep secrets in `/tmp/mcf-deploy.env`:

```bash
export NEON_DATABASE_URL='postgresql://<neon-dsn>'
export HF_TOKEN='hf_...'
export CLOUDFLARE_API_TOKEN='...'
export CLOUDFLARE_ACCOUNT_ID='...'
```

Do not commit this file.

## 2. Create Neon

Create a Neon project and copy the direct Postgres connection string. In the
Neon SQL editor, enable `pgvector`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname FROM pg_extension WHERE extname = 'vector';
```

Store the connection string as `NEON_DATABASE_URL`.

## 3. Seed Neon From Local Postgres

Run the initial seed from your local machine:

```bash
export LOCAL_DATABASE_URL='postgresql://postgres@127.0.0.1:55432/mcf'
export NEON_DATABASE_URL='postgresql://<neon-dsn>'

poetry run python -m src.cli pg-seed-hosted \
  --source "$LOCAL_DATABASE_URL" \
  --target "$NEON_DATABASE_URL" \
  --max-age-days 90
```

Equivalent script form:

```bash
PYTHONPATH=. poetry run python scripts/seed_hosted_slice.py \
  --source "$LOCAL_DATABASE_URL" \
  --target "$NEON_DATABASE_URL" \
  --max-age-days 90
```

Verify Neon:

```sql
SELECT COUNT(*) AS jobs FROM jobs;

SELECT entity_type, COUNT(*)
FROM embeddings
GROUP BY entity_type
ORDER BY entity_type;

SELECT MIN(posted_date) AS min_posted_date, MAX(posted_date) AS max_posted_date
FROM jobs;

SELECT pg_size_pretty(pg_database_size(current_database())) AS database_size;
```

Expected shape:

- only recent current-year jobs are present
- `embeddings` contains only `entity_type = 'job'`
- job embedding coverage is close to 100%
- database size stays below the selected Neon tier limit

## 4. Configure Hosted Refresh

Add this GitHub Actions secret:

```text
NEON_DATABASE_URL
```

The workflow is:

- [neon-hosted-refresh.yml](../.github/workflows/neon-hosted-refresh.yml)

Each scheduled run:

```bash
YEAR=$(date +%Y)

poetry run python -m src.cli scrape-historical \
  --year "$YEAR" \
  --resume \
  --db "$NEON_DATABASE_URL"

poetry run python -m src.cli embed-sync \
  --db "$NEON_DATABASE_URL" \
  --embedding-backend onnx \
  --onnx-model-dir data/models/all-MiniLM-L6-v2-onnx \
  --no-update-index

poetry run python -m src.cli pg-purge-hosted \
  --target "$NEON_DATABASE_URL" \
  --max-age-days 90
```

This is a scheduled batch refresh, not a daemon:

- Neon does not run background processes
- the hosted slice stays bounded by the purge step
- hosted scrape state can resume between workflow runs
- the current year and retention cutoff are computed dynamically

### Manual Hosted Deploy

Use [hosted-deploy.yml](../.github/workflows/hosted-deploy.yml) when you want
to deploy the repository default branch to both Neon and the Hugging Face Space
from the GitHub Actions UI. It is intentionally manual-only and does not run on
merge.

Required GitHub Actions secrets:

```text
NEON_DATABASE_URL
HF_TOKEN
```

Optional GitHub Actions variables:

```text
HF_SPACE_REPO_ID=xang1234/jobs-intelligence-api
HF_SOURCE_REPO=https://github.com/xang1234/jobs-intelligence.git
HF_CORS_ORIGINS=https://jobs-intelligence.pages.dev,https://jobs.deepgradient.uk,https://deepgradient.uk,http://localhost:3000
```

Run it from GitHub:

1. Open **Actions**.
2. Select **Manual Hosted Deploy**.
3. Choose the repository default branch in the workflow branch selector
   (`master` in this repo).
4. Click **Run workflow**.

The manual workflow can refresh/purge Neon first, then deploy the Hugging Face
Space with `SOURCE_REF=<selected default branch>` and
`SOURCE_VERSION=<workflow commit SHA>`. The Space build checks out
`SOURCE_VERSION`, so a long Neon refresh cannot accidentally publish a newer
branch head.

## 5. API Hosting Option A: Hugging Face Spaces

Use Hugging Face Spaces when Oracle Free Tier capacity is unavailable, when you
want the lowest-friction free API host, or when you do not want to manage a VM.

This repo includes a Docker Space payload:

- [deploy/huggingface-space/Dockerfile](../deploy/huggingface-space/Dockerfile)
- [deploy/huggingface-space/README.md](../deploy/huggingface-space/README.md)
- [scripts/deploy_huggingface_space.py](../scripts/deploy_huggingface_space.py)

The Space builds a Docker image, checks out this repo at `SOURCE_VERSION` when
it is set, exports the ONNX model bundle during build, and runs FastAPI on
`${PORT:-8000}`. `SOURCE_REF` remains the branch/ref context and is used as a
fallback when `SOURCE_VERSION` is unset or `dev`.

Required Hugging Face Space secret:

```text
DATABASE_URL=<Neon DSN>
```

Required Space variables:

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

`scripts/deploy_huggingface_space.py` sets `SOURCE_VERSION` from the local git
revision by default so Docker checks out the exact source revision during the
Space build.

Deploy with:

```bash
export HF_TOKEN='hf_...'
export NEON_DATABASE_URL='postgresql://<neon-dsn>'

poetry run python scripts/deploy_huggingface_space.py \
  --repo-id xang1234/jobs-intelligence-api \
  --cors-origins 'https://jobs-intelligence.pages.dev,https://jobs.deepgradient.uk,https://deepgradient.uk,http://localhost:3000'
```

The script:

1. Creates or reuses the Docker Space.
2. Stores `DATABASE_URL` as a Space secret.
3. Sets the runtime variables.
4. Uploads `deploy/huggingface-space/`.
5. Restarts the Space.

Smoke-test the Space:

```bash
curl https://xang1234-jobs-intelligence-api.hf.space/health
curl https://xang1234-jobs-intelligence-api.hf.space/docs
curl -X POST https://xang1234-jobs-intelligence-api.hf.space/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"data analyst","limit":5}'
```

Expected:

- `/health` returns `healthy` or `degraded`, not a server error
- `/docs` loads FastAPI OpenAPI docs
- `/api/search` returns jobs from the Neon slice

Notes:

- First requests can be slow while the Space warms up.
- Keep `MCF_SEARCH_BACKEND=pgvector`; do not upload FAISS indexes.
- No mounted bucket is required for this API-only Space.
- Use Hugging Face secrets for the Neon DSN. Do not commit it.

## 6. API Hosting Option B: Oracle Always Free

Use Oracle when you want a dedicated VM, a stable custom API domain, and more
control over CPU, memory, logs, and reverse proxying.

Create an Oracle Cloud Always Free VM:

- shape: `VM.Standard.A1.Flex`
- OS: Ubuntu 22.04 or 24.04
- recommended size: `2 OCPU / 12 GB RAM`
- public IP enabled

Avoid `VM.Standard.E2.1.Micro`; it is too small for this backend and ONNX
runtime.

Allow inbound traffic on:

- `22` for SSH
- `80` and `443` for HTTPS
- optionally `8000` for direct API testing

Install Docker:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
newgrp docker
docker --version
```

Clone, build, and run:

```bash
git clone https://github.com/xang1234/jobs-intelligence.git
cd jobs-intelligence
docker build -f docker/backend.Dockerfile -t mcf-backend .
```

Create `/opt/mcf/.env`:

```bash
sudo mkdir -p /opt/mcf
sudo tee /opt/mcf/.env > /dev/null <<'EOF'
DATABASE_URL=postgresql://<neon-dsn>
MCF_SEARCH_BACKEND=pgvector
MCF_LEAN_HOSTED=1
MCF_EMBEDDING_BACKEND=onnx
MCF_CORS_ORIGINS=https://jobs-intelligence.pages.dev
MCF_RATE_LIMIT_RPM=100
EOF
sudo chmod 600 /opt/mcf/.env
```

Run the API:

```bash
docker run -d \
  --name mcf-api \
  --restart unless-stopped \
  --env-file /opt/mcf/.env \
  --memory 3g \
  --log-opt max-size=10m \
  --log-opt max-file=5 \
  -p 8000:8000 \
  mcf-backend \
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1
```

Install Caddy for HTTPS:

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | \
  sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | \
  sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update
sudo apt-get install -y caddy
```

Create `/etc/caddy/Caddyfile`:

```text
api.yourdomain.com {
    reverse_proxy 127.0.0.1:8000
}
```

Point DNS for `api.yourdomain.com` at the Oracle public IP before relying on
Caddy certificate issuance, then reload Caddy:

```bash
sudo systemctl reload caddy
```

After HTTPS works, close direct inbound port `8000`.

Smoke-test:

```bash
curl https://api.yourdomain.com/health
curl https://api.yourdomain.com/docs
curl -X POST https://api.yourdomain.com/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"data analyst","limit":5}'
```

## 7. Frontend on Cloudflare Pages

The frontend is a static Vite/React SPA under `src/frontend`.

Current working setup:

- project: `jobs-intelligence`
- production URL: `https://jobs-intelligence.pages.dev`
- API URL: `https://xang1234-jobs-intelligence-api.hf.space`
- custom domain: optional

Build locally:

```bash
cd src/frontend
VITE_API_BASE_URL=https://xang1234-jobs-intelligence-api.hf.space npm run build
```

The repo includes a Pages SPA fallback:

```text
src/frontend/public/_redirects
```

with:

```text
/* /index.html 200
```

This keeps deep links such as `/trends` working on Cloudflare Pages.

Cloudflare Pages settings for a Git-connected deployment:

```text
Root directory: src/frontend
Build command: npm run build
Build output directory: dist
Production branch: master
Environment variable:
  VITE_API_BASE_URL=https://xang1234-jobs-intelligence-api.hf.space
```

Direct upload is also valid. Build locally with `VITE_API_BASE_URL` set, then
upload `src/frontend/dist`.

If you want a custom domain later, create a Pages custom domain and DNS record:

```text
Type: CNAME
Name: jobs
Target: jobs-intelligence.pages.dev
Proxy: On
```

If the `.pages.dev` URL is acceptable, skip the custom domain entirely.

## 8. CORS Checklist

The API must allow the frontend origin. For the current Cloudflare Pages URL:

```text
MCF_CORS_ORIGINS=https://jobs-intelligence.pages.dev,http://localhost:3000
```

If you add a custom domain later, include both:

```text
MCF_CORS_ORIGINS=https://jobs-intelligence.pages.dev,https://jobs.deepgradient.uk,http://localhost:3000
```

Check CORS with:

```bash
curl -i -X OPTIONS https://xang1234-jobs-intelligence-api.hf.space/api/search \
  -H 'Origin: https://jobs-intelligence.pages.dev' \
  -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: content-type'
```

Expected:

```text
access-control-allow-origin: https://jobs-intelligence.pages.dev
```

## 9. Ongoing Operations

- Continue scraping the full archive locally.
- Let GitHub Actions refresh Neon on schedule.
- Monitor Neon storage and reduce retention if needed.
- Keep hosted API search on `pgvector`.
- Regenerate hosted embeddings after each scrape refresh.
- If the local archive catches up beyond the hosted slice, reseed or let the
  scheduled workflow ingest into Neon.

Manual Neon refresh from local Postgres:

```bash
export LOCAL_DATABASE_URL='postgresql://postgres@127.0.0.1:55432/mcf'
export NEON_DATABASE_URL='postgresql://<neon-dsn>'

poetry run python -m src.cli pg-seed-hosted \
  --source "$LOCAL_DATABASE_URL" \
  --target "$NEON_DATABASE_URL" \
  --max-age-days 90
```

## 10. Rollback

If the hosted database is wrong:

```bash
poetry run python -m src.cli pg-seed-hosted \
  --source "$LOCAL_DATABASE_URL" \
  --target "$NEON_DATABASE_URL" \
  --max-age-days 90
```

If the Hugging Face API misbehaves:

1. Check `https://huggingface.co/spaces/xang1234/jobs-intelligence-api`.
2. Restart the Space.
3. Re-run `scripts/deploy_huggingface_space.py`.
4. Temporarily point `VITE_API_BASE_URL` at an Oracle or local API if needed.

If Oracle becomes unreliable:

1. Keep Neon unchanged.
2. Deploy the Hugging Face Space.
3. Update Cloudflare Pages `VITE_API_BASE_URL`.
4. Redeploy the frontend.

If the frontend misbehaves:

1. Roll back to the previous Cloudflare Pages deployment.
2. Verify `VITE_API_BASE_URL`.
3. Confirm the API CORS allow-list includes the active frontend origin.
