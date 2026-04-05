# Neon + Oracle Hosted Deployment

This guide deploys the hosted stack as three separate services:

- `Neon Free` hosts the lean Postgres database
- `GitHub Actions` runs scheduled hosted scraping, embedding sync, and purge jobs
- `Oracle Cloud Always Free` hosts the FastAPI container on an Ampere A1 VM

Local Postgres remains the full archive and the primary source of truth. The
hosted database is a lean serving slice that keeps only rows where:

```text
posted_date >= max(Jan 1 of current year, today - 90 days)
```

The hosted slice keeps only `job` embeddings. `skill` and `company` embeddings
stay local-only.

## 1. Prerequisites

You should already have:

- a full local Postgres archive populated from SQLite
- working local CLI access via `poetry run python -m src.cli`
- a GitHub repository with Actions enabled
- an Oracle Cloud Free Tier account
- a Neon account

This guide assumes the repo is public, so standard GitHub-hosted scheduled
Actions are free. If the repo becomes private, review current GitHub Actions
billing before keeping the refresh workflow on hosted runners.

## Provider caveats

- Neon Free storage limits can change. Check the current pricing page before
  relying on the same headroom, and monitor database size after each hosted
  refresh.
- GitHub Actions scheduled runs are treated as free here because the repository
  is public. Private repositories have different included-runner limits.
- Oracle Always Free capacity is not guaranteed in every region, and idle free
  accounts can be suspended. Keep the VM active and be ready to recreate it in
  another region if Oracle does not offer A1 capacity in your first choice.

## 2. Create the Neon database

1. Create a new Neon project.
2. Copy the direct Postgres connection string for the project.
3. In the Neon SQL editor, enable `pgvector`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname FROM pg_extension WHERE extname = 'vector';
```

4. Keep that DSN for the remaining steps as `NEON_DATABASE_URL`.

## 3. Seed the hosted slice from local Postgres

Run the initial hosted seed from your local machine.

```bash
export LOCAL_DATABASE_URL='postgresql://postgres@127.0.0.1:55432/mcf'
export NEON_DATABASE_URL='postgresql://<neon-dsn>'

poetry run python -m src.cli pg-seed-hosted \
  --source "$LOCAL_DATABASE_URL" \
  --target "$NEON_DATABASE_URL" \
  --max-age-days 90
```

The `--min-posted-date` defaults to January 1 of the current year. Override it
only if you need a different floor.

The hosted seed keeps the retained `jobs`, `job` embeddings, and one resumable
historical scrape session for the current year when one exists locally. It does
not copy `skill` or `company` embeddings.

Verify the hosted contents with SQL:

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

- only recent jobs from the current year are present
- only `job` rows exist in `embeddings`
- the database size stays comfortably below Neon Free storage limits

## 4. Add the hosted refresh GitHub Actions secret

In GitHub repository settings, add:

- `NEON_DATABASE_URL`

The included workflow file is:

- [neon-hosted-refresh.yml](../.github/workflows/neon-hosted-refresh.yml)

It runs the hosted refresh sequence on a schedule and also supports manual
dispatch.

## 5. What the hosted refresh workflow does

Each scheduled run performs:

```bash
YEAR=$(date +%Y)
poetry run python -m src.cli scrape-historical --year "$YEAR" --resume --db "$NEON_DATABASE_URL"
poetry run python -m src.cli embed-sync --db "$NEON_DATABASE_URL" --embedding-backend onnx --onnx-model-dir data/models/all-MiniLM-L6-v2-onnx --no-update-index
poetry run python -m src.cli pg-purge-hosted --target "$NEON_DATABASE_URL" --max-age-days 90
```

This is intentionally a scheduled batch job, not a daemon:

- Neon does not run background processes
- the hosted slice stays bounded because old rows are purged each run
- `scrape-historical --resume` can continue from the hosted session state
- the scrape year and purge cutoff are computed dynamically — no manual update
  needed on year rollover

## 6. Create the Oracle API VM

Create an Oracle Cloud Always Free VM with these settings:

- shape: `VM.Standard.A1.Flex`
- OS: Ubuntu 22.04 or 24.04
- recommended size: `2 OCPU / 12 GB RAM`
- public IP enabled

Do not use `VM.Standard.E2.1.Micro` for the API. The micro shape is too small
for this backend and ONNX runtime.

Allow inbound traffic on:

- `22` for SSH
- `80` and `443` for HTTPS (Caddy reverse proxy)
- optionally `8000` for direct API testing (can be closed after Caddy is set up)

## 7. Install Docker on Oracle

SSH into the Oracle VM and install Docker:

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

## 8. Build and run the API container on Oracle

Clone the repo on the Oracle VM and build the backend image:

```bash
git clone https://github.com/xang1234/jobs-intelligence.git
cd jobs-intelligence
docker build -f docker/backend.Dockerfile -t mcf-backend .
```

Create an env file for secrets (keeps credentials out of `docker inspect` and
process listings):

```bash
sudo mkdir -p /opt/mcf
sudo tee /opt/mcf/.env > /dev/null <<'EOF'
DATABASE_URL=postgresql://<neon-dsn>
MCF_SEARCH_BACKEND=pgvector
MCF_LEAN_HOSTED=1
MCF_EMBEDDING_BACKEND=onnx
MCF_CORS_ORIGINS=https://<your-frontend-origin>
EOF
sudo chmod 600 /opt/mcf/.env
```

Run the API against Neon:

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

Notes:

- `MCF_ONNX_MODEL_DIR` is not required when you use the bundled backend image
- keep `--workers 1` unless you have measured memory and CPU headroom
- do not mount FAISS indexes on Oracle; hosted search should use `pgvector`
- `--memory 3g` matches the resource limits in `docker-compose.prod.yml`
- `--log-opt` prevents unbounded log growth on the small VM
- the Dockerfile does not set a default `MCF_CORS_ORIGINS` — you must set it in
  the env file or the API will reject cross-origin requests

## 9. Set up HTTPS with Caddy

Install Caddy on the Oracle VM for automatic TLS via Let's Encrypt:

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | \
  sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | \
  sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update
sudo apt-get install -y caddy
```

Create a Caddyfile:

```bash
sudo tee /etc/caddy/Caddyfile > /dev/null <<'EOF'
api.yourdomain.com {
    reverse_proxy 127.0.0.1:8000
}
EOF
sudo systemctl reload caddy
```

Caddy automatically provisions and renews Let's Encrypt certificates. Point
your DNS A record to the Oracle VM's public IP before starting Caddy.

Once Caddy is active, close port `8000` in the Oracle security list — all
traffic should go through ports 80/443.

Update `MCF_CORS_ORIGINS` in `/opt/mcf/.env` to use your HTTPS domain, then
restart the API container:

```bash
docker restart mcf-api
```

## 10. Smoke-test the hosted API

From the Oracle VM or your workstation:

```bash
curl https://api.yourdomain.com/health
curl https://api.yourdomain.com/docs
curl -X POST https://api.yourdomain.com/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"data analyst","limit":5}'
```

Expected results:

- `/health` returns `healthy` or `degraded`, but not a server error
- `/docs` loads the FastAPI OpenAPI UI
- `/api/search` returns recent retained jobs from the hosted slice

## 11. Frontend deployment

The frontend is a static React SPA that can be deployed independently of the
API backend. Options:

- **Static hosting** (Vercel, Netlify, Cloudflare Pages): push the
  `src/frontend/` directory and set the build command to `npm run build`. Set
  the API base URL via environment variable to point at the Oracle API.
- **Oracle VM co-hosting**: build and run the frontend Docker image on the same
  VM using `docker/frontend.Dockerfile`. Add a second Caddy site block to serve
  the frontend on a separate subdomain.

Either way, the frontend makes API calls to `/api/*` which must resolve to the
backend. On static hosts, configure a rewrite/proxy rule. On Oracle co-hosting,
update `docker/nginx.conf` to proxy to the backend container name or IP.

## 12. Ongoing operations

- Continue scraping the full archive locally. Local Postgres remains the system
  of record.
- Let GitHub Actions refresh Neon on schedule.
- Watch Neon storage usage in the dashboard. If the hosted slice grows too
  close to the Free storage cap, reduce retention before the database fills up.
- Keep Oracle focused on the API only. Do not run the scraper daemon there.

## 13. Rollback

If the hosted deployment misbehaves:

1. Stop the Oracle API container.
2. Truncate and reseed Neon from local Postgres:

```bash
poetry run python -m src.cli pg-seed-hosted \
  --source "$LOCAL_DATABASE_URL" \
  --target "$NEON_DATABASE_URL" \
  --max-age-days 90
```

3. Start the Oracle API container again.

If Oracle becomes unreliable, keep Neon and move the API container to another
compute host without changing the hosted database workflow.
