# Neon + Oracle Hosted Deployment

This guide deploys the hosted stack as three separate services:

- `Neon Free` hosts the lean Postgres database
- `GitHub Actions` runs scheduled hosted scraping, embedding sync, and purge jobs
- `Oracle Cloud Always Free` hosts the FastAPI container on an Ampere A1 VM

Local Postgres remains the full archive and the primary source of truth. The
hosted database is a lean serving slice that keeps only rows where:

```text
posted_date >= max(2026-01-01, today - 90 days)
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
  --min-posted-date 2026-01-01 \
  --max-age-days 90
```

The hosted seed keeps the retained `jobs`, `job` embeddings, and one resumable
2026 historical scrape session when one exists locally. It does not copy
`skill` or `company` embeddings.

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

- only recent 2026-floor jobs are present
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
poetry run python -m src.cli scrape-historical --year 2026 --resume --db "$NEON_DATABASE_URL"
poetry run python -m src.cli embed-sync --db "$NEON_DATABASE_URL" --embedding-backend onnx --onnx-model-dir data/models/all-MiniLM-L6-v2-onnx --no-update-index
poetry run python -m src.cli pg-purge-hosted --target "$NEON_DATABASE_URL" --min-posted-date 2026-01-01 --max-age-days 90
```

This is intentionally a scheduled batch job, not a daemon:

- Neon does not run background processes
- the hosted slice stays bounded because old rows are purged each run
- `scrape-historical --resume` can continue from the hosted 2026 session state

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
- `8000` for direct API testing
- optionally `80` and `443` if you place Nginx or Caddy in front

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

Run the API against Neon:

```bash
docker run -d \
  --name mcf-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e DATABASE_URL='postgresql://<neon-dsn>' \
  -e MCF_SEARCH_BACKEND='pgvector' \
  -e MCF_LEAN_HOSTED='1' \
  -e MCF_EMBEDDING_BACKEND='onnx' \
  -e MCF_CORS_ORIGINS='https://<your-frontend-origin>' \
  mcf-backend \
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1
```

Notes:

- `MCF_ONNX_MODEL_DIR` is not required when you use the bundled backend image
- keep `--workers 1` unless you have measured memory and CPU headroom
- do not mount FAISS indexes on Oracle; hosted search should use `pgvector`

If you want HTTPS, put Nginx or Caddy in front of the container and proxy to
`127.0.0.1:8000`.

## 9. Smoke-test the hosted API

From the Oracle VM or your workstation:

```bash
curl http://<oracle-ip>:8000/health
curl http://<oracle-ip>:8000/docs
curl -X POST http://<oracle-ip>:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"data analyst","limit":5}'
```

Expected results:

- `/health` returns `healthy` or `degraded`, but not a server error
- `/docs` loads the FastAPI OpenAPI UI
- `/api/search` returns recent retained jobs from the hosted slice

## 10. Ongoing operations

- Continue scraping the full archive locally. Local Postgres remains the system
  of record.
- Let GitHub Actions refresh Neon on schedule.
- Watch Neon storage usage in the dashboard. If the hosted slice grows too
  close to the Free storage cap, reduce retention before the database fills up.
- Keep Oracle focused on the API only. Do not run the scraper daemon there.

## 11. Rollback

If the hosted deployment misbehaves:

1. Stop the Oracle API container.
2. Truncate and reseed Neon from local Postgres:

```bash
poetry run python -m src.cli pg-seed-hosted \
  --source "$LOCAL_DATABASE_URL" \
  --target "$NEON_DATABASE_URL" \
  --min-posted-date 2026-01-01 \
  --max-age-days 90
```

3. Start the Oracle API container again.

If Oracle becomes unreliable, keep Neon and move the API container to another
compute host without changing the hosted database workflow.
