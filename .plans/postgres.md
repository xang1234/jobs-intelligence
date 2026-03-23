# Plan: PostgreSQL + Pluggable Search Backend (FAISS / PGVector)

## Context

The MCF Intelligence Platform currently uses SQLite (6.5 GB, 1.3M jobs) + FAISS for data storage and semantic search. To enable flexible remote hosting — from free serverless (Neon + Vercel) to self-hosted VMs — we need:

1. **PostgreSQL as the database** (replacing SQLite for remote) — works with every hosting platform
2. **Pluggable vector search** — FAISS for high-performance local, PGVector for serverless
3. **Deployment-time configuration** — choose backend via environment variable
4. **Data retention policy** — full archive stays local, Neon deployment purges old data

**Constraints:**
- Active scraper daemon must not be interrupted (currently scraping year 2026, plus 5 other years in progress)
- SQLite remains the local master copy — migration is a one-way copy, not a move
- Neon free tier (512MB) requires retention policy; local/VM deployments keep everything

---

## Architecture

```
                     SemanticSearchEngine
  (unchanged public API — search, find_similar, match_profile)

  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
  │ EmbeddingGen│    │  VectorBackend   │    │    Database      │
  │ (unchanged) │    │  (new Protocol)  │    │ (SQLite or PG)   │
  │             │    │                  │    │                  │
  │ sentence-   │    │ ┌──────────────┐ │    │ Factory selects  │
  │ transformers│    │ │ FAISSBackend │ │    │ based on         │
  │ 384-dim     │    │ └──────────────┘ │    │ DATABASE_URL     │
  │             │    │ ┌──────────────┐ │    │                  │
  │             │    │ │PGVecBackend  │ │    │                  │
  │             │    │ └──────────────┘ │    │                  │
  └─────────────┘    └──────────────────┘    └─────────────────┘

MCF_SEARCH_BACKEND=faiss|pgvector    DATABASE_URL=postgresql://... or sqlite:///...
MCF_RETENTION_DAYS=0 (keep all) or 180 (purge old)
```

---

## Phase 1: Pluggable Vector Backend

**Goal**: Abstract FAISS behind a `VectorBackend` protocol. Lowest risk — no data changes.

### 1.1 Vector backend protocol
- **New** `src/mcf/embeddings/vector_backend.py` — `Protocol` class defining the interface

### 1.2 FAISS backend (refactor, not rewrite)
- **Refactor** `src/mcf/embeddings/index_manager.py` → `src/mcf/embeddings/faiss_backend.py`
- Rename `FAISSIndexManager` → `FAISSBackend` conforming to `VectorBackend`
- Zero logic changes — same code, new interface

### 1.3 PGVector backend
- **New** `src/mcf/embeddings/pgvector_backend.py`
- `job_embeddings` table with `vector(384)` column + HNSW index
- Search via `<=>` cosine distance operator
- Skills + companies: separate tables or shared `entity_embeddings` table

### 1.4 Backend factory
- **New** `src/mcf/embeddings/backend_factory.py`
- Reads `MCF_SEARCH_BACKEND` env var (default: `faiss`)

### 1.5 Search engine update
- `src/mcf/embeddings/search_engine.py` — Replace `self.index_manager` → `self.backend`
- No changes to scoring logic, caching, or public API

### Files
| Action | File |
|--------|------|
| New | `src/mcf/embeddings/vector_backend.py` |
| Refactor | `src/mcf/embeddings/index_manager.py` → `faiss_backend.py` |
| New | `src/mcf/embeddings/pgvector_backend.py` |
| New | `src/mcf/embeddings/backend_factory.py` |
| Modify | `src/mcf/embeddings/search_engine.py` |
| Modify | `src/mcf/embeddings/__init__.py` |

---

## Phase 2: PostgreSQL Database Layer

**Goal**: Add PostgreSQL as an alternative to SQLite. SQLite stays for local dev.

### 2.1 SQL translation
| SQLite | PostgreSQL |
|--------|-----------|
| `?` placeholders | `%s` placeholders |
| `INSERT OR REPLACE` | `INSERT ... ON CONFLICT DO UPDATE` |
| `AUTOINCREMENT` | `SERIAL` / `BIGSERIAL` |
| `PRAGMA` directives | Connection/pool config |
| FTS5 + triggers | `tsvector` column + GIN index + `tsvector_update_trigger` |
| `strftime(...)` | `to_char(...)` / `::date` |
| `GROUP_CONCAT()` | `string_agg()` |

### 2.2 New modules
- **New** `src/mcf/pg_database.py` — PostgreSQL impl using `psycopg` v3 (sync)
- **New** `src/mcf/db_factory.py` — Selects backend from `DATABASE_URL`
- **Modify** `src/mcf/database.py` — Extract `DatabaseProtocol` (shared interface)

### 2.3 Retention policy
- **New** config: `MCF_RETENTION_DAYS` env var (default: `0` = keep all)
- When > 0: after each scrape, delete jobs older than N days
- Cascading cleanup: `job_history`, `fetch_attempts`, embeddings for deleted jobs
- Only affects the deployment DB — local SQLite never purges

### Files
| Action | File |
|--------|------|
| Modify | `src/mcf/database.py` — Extract `DatabaseProtocol` |
| New | `src/mcf/pg_database.py` |
| New | `src/mcf/db_factory.py` |
| Modify | `src/api/app.py` — Use factory |
| Modify | `src/cli.py` — Accept `DATABASE_URL`, `--search-backend` |

---

## Phase 3: Migration, Backup & Deployment

### 3.1 SQLite backup strategy

| Backup | Schedule | Command |
|--------|----------|---------|
| Pre-migration snapshot | Once | `sqlite3 data/mcf_jobs.db ".backup data/backups/pre_migration.db"` |
| Daily automated | Cron `0 1 * * *` | `sqlite3 data/mcf_jobs.db ".backup data/backups/mcf_$(date +%Y%m%d).db"` |
| WAL checkpoint | Before each backup | `sqlite3 data/mcf_jobs.db "PRAGMA wal_checkpoint(TRUNCATE)"` |
| Retention | After backup | `find data/backups/ -name "mcf_*.db" -mtime +7 -delete` |

SQLite `.backup` is safe during writes — it creates a consistent page-level snapshot even while the daemon is scraping.

### 3.2 Migration procedure (scraper keeps running)

```
Step 1: Hot backup SQLite (~30 sec)
  sqlite3 data/mcf_jobs.db ".backup data/backups/migration_source.db"

Step 2: Run migration script against the copy (~8-15 min)
  python scripts/migrate_sqlite_to_pg.py \
    --source data/backups/migration_source.db \
    --target postgresql://mcf:pass@localhost:5432/mcf

Step 3: Verify integrity (see checks below)

Step 4: For Neon deployment, apply retention policy
  python scripts/migrate_sqlite_to_pg.py \
    --source data/backups/migration_source.db \
    --target $NEON_DATABASE_URL \
    --retention-days 180
```

**Scraper downtime: zero.** Migration reads from a backup copy, not the live DB.

### 3.3 Migration time estimate

| Step | Data | Time |
|------|------|------|
| SQLite `.backup` | 6.5 GB | ~30 sec |
| COPY `jobs` to Postgres | 1,287,527 rows | ~3-5 min |
| COPY `fetch_attempts` | 1,305,633 rows | ~2-3 min |
| COPY small tables | ~53 rows total | <1 sec |
| Build tsvector + GIN index | 1.3M rows | ~2-3 min |
| Build column indexes | 17 indexes | ~1-2 min |
| Integrity verification | checksums + samples | ~1 min |
| **Total** | | **~10-15 min** |

### 3.4 Data integrity checks

The migration script runs all checks automatically and prints a report:

| Check | Method | Pass Criteria |
|-------|--------|--------------|
| **Row counts** | `COUNT(*)` both DBs, all tables | Exact match |
| **Column checksums** | `SUM(LENGTH(title))`, `SUM(salary_min)`, etc. | Exact match |
| **UUID uniqueness** | `COUNT(DISTINCT uuid) = COUNT(*)` in Postgres | No duplicates |
| **Sample row diff** | Random 100 rows, column-by-column comparison | 0 mismatches |
| **Date range** | `MIN/MAX(posted_date)` both DBs | Exact match |
| **Foreign keys** | `job_history.job_uuid` references valid `jobs.uuid` | 0 orphans |
| **FTS consistency** | Search "data scientist" in both, compare top-20 UUIDs | Same results |
| **Null distribution** | `COUNT(*) WHERE col IS NULL` per column | Exact match |

Output:
```
Migration Report:
  jobs:              1,287,527 → 1,287,527 ✓
  fetch_attempts:    1,305,633 → 1,305,633 ✓
  job_history:               2 →         2 ✓
  salary checksum:   match ✓
  sample diff (100): 0 mismatches ✓
  FTS "data scientist": top-20 match ✓
  Foreign keys:      0 orphans ✓
```

If any check fails, migration aborts with details and the Postgres DB is rolled back (transaction-wrapped).

### 3.5 Neon data sizing (with retention)

| Retention | Est. Jobs | Est. Postgres Size | Fits Free Tier? |
|-----------|----------|-------------------|----------------|
| 90 days | ~30-50K | ~100-150 MB | Yes (512MB) |
| 180 days | ~60-100K | ~200-350 MB | Yes (512MB) |
| 365 days | ~120-200K | ~400-700 MB | Borderline |
| All (1.3M) | 1,287,527 | ~3.7 GB | No — need paid |

Recommended: `MCF_RETENTION_DAYS=180` for Neon free tier.

---

## Phase 4: Docker & CI/CD

### 4.1 Docker Compose — add Postgres service
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    volumes:
      - pg-data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: mcf
      POSTGRES_USER: mcf
      POSTGRES_PASSWORD: ${PG_PASSWORD:-mcf_dev}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mcf"]

  backend:
    environment:
      DATABASE_URL: postgresql://mcf:${PG_PASSWORD:-mcf_dev}@postgres:5432/mcf
      MCF_SEARCH_BACKEND: ${MCF_SEARCH_BACKEND:-faiss}
      MCF_RETENTION_DAYS: ${MCF_RETENTION_DAYS:-0}
```

### 4.2 GitHub Actions — daily scrape + purge (serverless)
```yaml
name: Daily Scrape
on:
  schedule:
    - cron: '0 2 * * *'
jobs:
  scrape:
    strategy:
      matrix:
        query: ["data scientist", "machine learning", "data engineer"]
      max-parallel: 4
    runs-on: ubuntu-latest
    env:
      DATABASE_URL: ${{ secrets.NEON_DATABASE_URL }}
      MCF_SEARCH_BACKEND: pgvector
      MCF_RETENTION_DAYS: 180
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11', cache: 'pip' }
      - run: pip install -r requirements.txt
      - run: python -m src.cli scrape "${{ matrix.query }}"

  embed-and-purge:
    needs: scrape
    runs-on: ubuntu-latest
    env:
      DATABASE_URL: ${{ secrets.NEON_DATABASE_URL }}
      MCF_SEARCH_BACKEND: pgvector
      MCF_RETENTION_DAYS: 180
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11', cache: 'pip' }
      - run: pip install -r requirements.txt
      - run: python -m src.cli embed-sync
      - run: python -m src.cli purge-old  # new command, deletes jobs older than RETENTION_DAYS
```

### 4.3 New files
| File | Purpose |
|------|---------|
| `scripts/migrate_sqlite_to_pg.py` | One-time migration with integrity checks |
| `scripts/backup_sqlite.sh` | Daily backup cron script |
| `.github/workflows/scrape.yml` | Daily scrape + embed for Neon |
| Updated `docker-compose.yml` | Postgres service added |
| Updated `docker-compose.prod.yml` | Prod overrides for Postgres |

---

## Deployment Matrix

| Deployment | Database | Search | Retention | Cost |
|-----------|----------|--------|-----------|------|
| **Local dev** | SQLite | FAISS | Keep all | $0 |
| **Serverless** | Neon Postgres | PGVector | 180 days | $0/mo |
| **Fly.io** | Fly Postgres / Neon | PGVector or FAISS | Configurable | ~$12/mo |
| **Oracle VM** | Local Postgres | FAISS | Keep all | $0/mo |
| **VPS** | Local Postgres | FAISS | Keep all | $8-48/mo |

---

## Concurrency Notes

- **Current**: Single `ThreadPoolExecutor(max_workers=1)` serializes all search — ~5-10 searches/sec
- **With Postgres**: Connection pool handles concurrent reads natively
- **With PGVector**: No executor needed for search — async Postgres queries handle concurrency
- **Level 1 scaling** (easy): Increase executor to 4 workers + lock around TTLCache
- **Multi-user safe**: WAL (SQLite) / MVCC (Postgres) — readers never block writers

---

## Implementation Order

1. **Phase 1** — Vector backend abstraction (lowest risk, testable in isolation)
2. **Phase 2** — PostgreSQL database layer (larger effort, needs testing)
3. **Phase 3** — Migration script + backup + integrity checks
4. **Phase 4** — Docker + GitHub Actions deployment configs

---

## Verification

### Unit tests
- `test_faiss_backend.py` — FAISS tests via VectorBackend interface
- `test_pgvector_backend.py` — PGVector against Docker Postgres
- `test_pg_database.py` — Postgres database operations
- `test_search_engine.py` — Parameterized: works with either backend
- `test_retention.py` — Purge logic deletes only old data

### Integration
```bash
# FAISS + SQLite (current behavior, must not regress)
MCF_SEARCH_BACKEND=faiss python -m src.cli search "data scientist"

# PGVector + Postgres
DATABASE_URL=postgresql://mcf:pass@localhost:5432/mcf MCF_SEARCH_BACKEND=pgvector \
  python -m src.cli search "data scientist"
```

### Migration verification
```bash
python scripts/migrate_sqlite_to_pg.py \
  --source data/mcf_jobs.db \
  --target postgresql://mcf:pass@localhost:5432/mcf \
  --verify-only  # just run integrity checks, no data transfer
```

### Docker Compose
```bash
MCF_SEARCH_BACKEND=pgvector docker compose up -d
curl http://localhost:8000/health  # {"status": "healthy"}
curl http://localhost:8000/api/v1/stats  # verify job counts
```

---

## Current Database State (as of 2026-03-20)

### Tables (8 real + 4 FTS internal)
| Table | Rows | Purpose |
|-------|------|---------|
| `jobs` | 1,287,527 | Main job listings (26 columns) |
| `fetch_attempts` | 1,305,633 | Per-ID scrape tracking |
| `jobs_fts` | 1,287,527 | FTS5 virtual table (→ tsvector in PG) |
| `job_history` | 2 | Change snapshots |
| `historical_scrape_progress` | 10 | Scrape checkpoints |
| `search_analytics` | 40 | Query logging |
| `daemon_state` | 1 | Background process state |
| `scrape_sessions` | 0 | Search scrape sessions |
| `embeddings` | 0 | Unused (FAISS files used instead) |

### Size
- DB file: 6.5 GB (main) + 9.6 MB (WAL)
- 65,486 unique companies
- Date range: 2019-07-16 to 2026-03-19
- Avg description: 1,618 chars, avg title: 34 chars, avg skills: 212 chars

### Active Scrapes
| Year | Progress | Jobs Found |
|------|----------|-----------|
| 2026 | 23K / 250K (9%) | 11,394 |
| 2022 | 454K / 1M (45%) | 451,888 |
| 2021 | 603K / 700K (86%) | 595,904 |
| 2020 | 188K / 350K (54%) | 186,548 |
| 2019 | 38K / 50K (75%) | 36,560 |

Ingestion rate: ~30K jobs/day, ~210K/week
