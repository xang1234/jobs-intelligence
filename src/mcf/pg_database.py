"""
PostgreSQL database backend for MCF job data.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Iterator, Optional

import numpy as np

from .database import MCFDatabase
from .models import Job

logger = logging.getLogger(__name__)


EMBEDDING_VECTOR_STORAGE = "vector"
EMBEDDING_BINARY_STORAGE = "bytea"


PG_SCHEMA_SQL = """
{vector_extension_sql}

CREATE TABLE IF NOT EXISTS jobs (
    id BIGSERIAL PRIMARY KEY,
    uuid TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    company_name TEXT,
    company_uen TEXT,
    description TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type TEXT,
    employment_type TEXT,
    seniority TEXT,
    min_experience_years INTEGER,
    skills TEXT,
    categories TEXT,
    location TEXT,
    district TEXT,
    region TEXT,
    posted_date DATE,
    expiry_date DATE,
    applications_count INTEGER,
    job_url TEXT,
    title_family TEXT,
    industry_bucket TEXT,
    salary_annual_min INTEGER,
    salary_annual_max INTEGER,
    first_seen_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    search_document tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(company_name, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(skills, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'C')
    ) STORED
);

CREATE TABLE IF NOT EXISTS job_history (
    id BIGSERIAL PRIMARY KEY,
    job_uuid TEXT NOT NULL REFERENCES jobs(uuid) ON DELETE CASCADE,
    title TEXT,
    company_name TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    applications_count INTEGER,
    description TEXT,
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scrape_sessions (
    id BIGSERIAL PRIMARY KEY,
    search_query TEXT NOT NULL,
    total_jobs INTEGER,
    fetched_count INTEGER DEFAULT 0,
    current_offset INTEGER DEFAULT 0,
    status TEXT DEFAULT 'in_progress',
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS historical_scrape_progress (
    id BIGSERIAL PRIMARY KEY,
    year INTEGER NOT NULL,
    start_seq INTEGER NOT NULL,
    current_seq INTEGER NOT NULL,
    end_seq INTEGER,
    jobs_found INTEGER DEFAULT 0,
    jobs_not_found INTEGER DEFAULT 0,
    consecutive_not_found INTEGER DEFAULT 0,
    status TEXT DEFAULT 'in_progress',
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS fetch_attempts (
    id BIGSERIAL PRIMARY KEY,
    year INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    result TEXT NOT NULL,
    error_message TEXT,
    attempted_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(year, sequence)
);

CREATE TABLE IF NOT EXISTS daemon_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    pid INTEGER,
    status TEXT DEFAULT 'stopped',
    last_heartbeat TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    current_year INTEGER,
    current_seq INTEGER
);

CREATE TABLE IF NOT EXISTS embeddings (
    id BIGSERIAL PRIMARY KEY,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    embedding {embedding_column_type} NOT NULL,
    model_version TEXT DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_id, entity_type)
);

CREATE TABLE IF NOT EXISTS search_analytics (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    query_type TEXT DEFAULT 'semantic',
    result_count INTEGER,
    latency_ms DOUBLE PRECISION,
    cache_hit BOOLEAN DEFAULT FALSE,
    degraded BOOLEAN DEFAULT FALSE,
    filters_used JSONB,
    searched_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO daemon_state (id, status)
VALUES (1, 'stopped')
ON CONFLICT (id) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_jobs_uuid ON jobs(uuid);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_name);
CREATE INDEX IF NOT EXISTS idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX IF NOT EXISTS idx_jobs_posted ON jobs(posted_date);
CREATE INDEX IF NOT EXISTS idx_jobs_employment ON jobs(employment_type);
CREATE INDEX IF NOT EXISTS idx_jobs_search_document ON jobs USING GIN (search_document);
CREATE INDEX IF NOT EXISTS idx_history_uuid ON job_history(job_uuid);
CREATE INDEX IF NOT EXISTS idx_sessions_query ON scrape_sessions(search_query);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON scrape_sessions(status);
CREATE INDEX IF NOT EXISTS idx_historical_year ON historical_scrape_progress(year);
CREATE INDEX IF NOT EXISTS idx_historical_status ON historical_scrape_progress(status);
CREATE INDEX IF NOT EXISTS idx_fetch_year_seq ON fetch_attempts(year, sequence);
CREATE INDEX IF NOT EXISTS idx_fetch_result ON fetch_attempts(result);
CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(entity_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_version);
CREATE INDEX IF NOT EXISTS idx_analytics_time ON search_analytics(searched_at);
CREATE INDEX IF NOT EXISTS idx_analytics_query ON search_analytics(query);
{embedding_vector_indexes}
"""


ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def build_pg_schema_sql(*, pgvector_enabled: bool) -> str:
    """Build the PostgreSQL schema for vector or bytea embedding storage."""
    if pgvector_enabled:
        vector_extension_sql = "CREATE EXTENSION IF NOT EXISTS vector;"
        embedding_column_type = "vector(384)"
        embedding_vector_indexes = """
CREATE INDEX IF NOT EXISTS idx_embeddings_job_vector
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WHERE entity_type = 'job';
CREATE INDEX IF NOT EXISTS idx_embeddings_skill_vector
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WHERE entity_type = 'skill';
CREATE INDEX IF NOT EXISTS idx_embeddings_company_vector
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WHERE entity_type = 'company';"""
    else:
        vector_extension_sql = "-- pgvector extension not available; embeddings will be stored as BYTEA."
        embedding_column_type = "BYTEA"
        embedding_vector_indexes = ""

    return PG_SCHEMA_SQL.format(
        vector_extension_sql=vector_extension_sql,
        embedding_column_type=embedding_column_type,
        embedding_vector_indexes=embedding_vector_indexes,
    )


class PostgresDatabase:
    """PostgreSQL implementation of the MCF storage interface."""

    def __init__(
        self,
        dsn: str,
        read_only: bool = False,
        ensure_schema: bool = True,
    ):
        self.dsn = dsn
        self.read_only = read_only
        self.ensure_schema = ensure_schema
        self.embedding_storage_kind: str | None = None
        if ensure_schema and not read_only:
            self._ensure_schema()

    def _connect(self, write_optimized: bool = False):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("psycopg is required for PostgreSQL support") from exc

        conn = psycopg.connect(self.dsn, row_factory=dict_row)
        if self.read_only:
            conn.execute("SET default_transaction_read_only = on")
        if write_optimized:
            conn.execute("SET synchronous_commit = off")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        conn = self._connect()
        try:
            yield conn
            if not self.read_only:
                conn.commit()
        except Exception:
            if not self.read_only:
                conn.rollback()
            raise
        finally:
            conn.close()

    def _pgvector_extension_available(self, conn: Any | None = None) -> bool:
        owns_connection = conn is None
        if conn is None:
            conn = self._connect()
        try:
            row = conn.execute("SELECT 1 AS available FROM pg_available_extensions WHERE name = 'vector'").fetchone()
            return bool(row)
        finally:
            if owns_connection:
                conn.close()

    def _refresh_embedding_storage_kind(self, conn: Any | None = None) -> str:
        owns_connection = conn is None
        if conn is None:
            conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT udt_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'embeddings'
                  AND column_name = 'embedding'
                """
            ).fetchone()
            if row and row["udt_name"] == "vector":
                self.embedding_storage_kind = EMBEDDING_VECTOR_STORAGE
            elif row:
                self.embedding_storage_kind = EMBEDDING_BINARY_STORAGE
            else:
                self.embedding_storage_kind = (
                    EMBEDDING_VECTOR_STORAGE if self._pgvector_extension_available(conn) else EMBEDDING_BINARY_STORAGE
                )
            return self.embedding_storage_kind
        finally:
            if owns_connection:
                conn.close()

    def _using_pgvector(self) -> bool:
        if self.embedding_storage_kind not in {EMBEDDING_VECTOR_STORAGE, EMBEDDING_BINARY_STORAGE}:
            self._refresh_embedding_storage_kind()
        return self.embedding_storage_kind == EMBEDDING_VECTOR_STORAGE

    def supports_vector_search(self) -> bool:
        """Return True when pgvector-backed ANN search is actually available."""
        return self._using_pgvector()

    def _embedding_select_clause(self, column: str = "embedding", alias: str = "embedding") -> str:
        expression = f"{column}::text" if self._using_pgvector() else column
        return f"{expression} AS {alias}"

    def _embedding_placeholder(self) -> str:
        return "%s::vector" if self._using_pgvector() else "%s"

    @staticmethod
    def _embedding_bytes(embedding: np.ndarray) -> bytes:
        return np.asarray(embedding, dtype=np.float32).tobytes()

    def _embedding_parameter(self, embedding: np.ndarray | bytes | bytearray | memoryview) -> Any:
        if self._using_pgvector():
            return self._vector_literal(self._vector_from_value(embedding))
        if isinstance(embedding, memoryview):
            return embedding.tobytes()
        if isinstance(embedding, bytearray):
            return bytes(embedding)
        if isinstance(embedding, bytes):
            return embedding
        return self._embedding_bytes(embedding)

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(left, right) / denominator)

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            pgvector_enabled = self._pgvector_extension_available(conn)
            if not pgvector_enabled:
                logger.warning("pgvector extension is not installed; falling back to BYTEA embedding storage")
            conn.execute(build_pg_schema_sql(pgvector_enabled=pgvector_enabled))
            self._refresh_embedding_storage_kind(conn)

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None
        if isinstance(value, date):
            return value
        raw = str(value)
        if not ISO_DATE_RE.match(raw):
            return None
        return date.fromisoformat(raw)

    @staticmethod
    def _vector_literal(embedding: np.ndarray) -> str:
        flat = np.asarray(embedding, dtype=np.float32).tolist()
        return "[" + ",".join(f"{float(value):.8f}" for value in flat) + "]"

    @staticmethod
    def _vector_from_value(value: Any) -> np.ndarray:
        if value is None:
            raise ValueError("cannot decode null vector")
        if isinstance(value, list):
            return np.asarray(value, dtype=np.float32)
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, (bytes, bytearray)):
            return np.frombuffer(value, dtype=np.float32).copy()
        raw = str(value).strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        if not raw:
            return np.array([], dtype=np.float32)
        return np.fromiter((float(part) for part in raw.split(",")), dtype=np.float32)

    @staticmethod
    def _fetch_id(row: dict[str, Any] | tuple[Any, ...] | None) -> int:
        if row is None:
            raise ValueError("expected row with id")
        if isinstance(row, dict):
            return int(row["id"])
        return int(row[0])

    @staticmethod
    def _placeholder_list(size: int) -> str:
        return ",".join(["%s"] * size)

    @staticmethod
    def _executemany(conn: Any, query: str, params_seq: list[Any] | tuple[Any, ...]) -> None:
        if not params_seq:
            return
        with conn.cursor() as cursor:
            cursor.executemany(query, params_seq)

    @staticmethod
    def can_acquire_write_lock(db_path: str, timeout_ms: int = 1000) -> bool:
        if db_path.strip().lower().startswith(("postgres://", "postgresql://")):
            # Postgres permits concurrent writers, so "acquiring a write lock" only
            # means verifying we can open a writable connection. We deliberately let
            # connection, authentication, and schema errors propagate rather than
            # squash them into False — callers need to distinguish "server is
            # unreachable" from "another process holds the lock" to give users a
            # useful error message.
            db = PostgresDatabase(db_path, read_only=False, ensure_schema=False)
            with db._connection():
                return True
        return MCFDatabase.can_acquire_write_lock(db_path, timeout_ms=timeout_ms)

    def _save_to_history(self, conn: Any, existing: dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO job_history (
                job_uuid, title, company_name, salary_min, salary_max,
                applications_count, description
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                existing["uuid"],
                existing["title"],
                existing["company_name"],
                existing["salary_min"],
                existing["salary_max"],
                existing["applications_count"],
                existing["description"],
            ),
        )

    def _job_data_for_write(self, job_data: dict[str, Any], *, timestamp: str, is_insert: bool) -> dict[str, Any]:
        annual_min, annual_max = MCFDatabase._calculate_annual_salary(
            job_data.get("salary_min"),
            job_data.get("salary_max"),
            job_data.get("salary_type"),
        )
        title_family, industry_bucket = MCFDatabase._derive_normalized_job_metadata(
            title=job_data.get("title"),
            categories=job_data.get("categories"),
            skills=job_data.get("skills"),
        )
        payload = {
            **job_data,
            "posted_date": self._parse_date(job_data.get("posted_date")),
            "expiry_date": self._parse_date(job_data.get("expiry_date")),
            "salary_annual_min": annual_min,
            "salary_annual_max": annual_max,
            "title_family": title_family,
            "industry_bucket": industry_bucket,
            "last_updated_at": timestamp,
        }
        if is_insert:
            payload["first_seen_at"] = job_data.get("first_seen_at") or timestamp
        return payload

    def upsert_job(self, job: Job, conn: Any | None = None) -> tuple[bool, bool]:
        job_data = job.to_flat_dict()
        now = datetime.now().isoformat()

        owns_connection = conn is None
        if conn is None:
            conn = self._connect(write_optimized=True)

        try:
            existing = conn.execute("SELECT * FROM jobs WHERE uuid = %s", (job.uuid,)).fetchone()
            if existing is None:
                self._insert_job(conn, job_data, now)
                if owns_connection:
                    conn.commit()
                return True, False

            changes = MCFDatabase._detect_changes(self, dict(existing), job_data)
            if changes:
                self._save_to_history(conn, existing)
                self._update_job(conn, job_data, now)
                if owns_connection:
                    conn.commit()
                return False, True

            if owns_connection:
                conn.commit()
            return False, False
        except Exception:
            if owns_connection:
                conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()

    def _insert_job(self, conn: Any, job_data: dict[str, Any], timestamp: str) -> None:
        data = self._job_data_for_write(job_data, timestamp=timestamp, is_insert=True)
        conn.execute(
            """
            INSERT INTO jobs (
                uuid, title, company_name, company_uen, description,
                salary_min, salary_max, salary_type, employment_type,
                seniority, min_experience_years, skills, categories,
                location, district, region, posted_date, expiry_date,
                applications_count, job_url, first_seen_at, last_updated_at,
                salary_annual_min, salary_annual_max, title_family, industry_bucket
            ) VALUES (
                %(uuid)s, %(title)s, %(company_name)s, %(company_uen)s, %(description)s,
                %(salary_min)s, %(salary_max)s, %(salary_type)s, %(employment_type)s,
                %(seniority)s, %(min_experience_years)s, %(skills)s, %(categories)s,
                %(location)s, %(district)s, %(region)s, %(posted_date)s, %(expiry_date)s,
                %(applications_count)s, %(job_url)s, %(first_seen_at)s, %(last_updated_at)s,
                %(salary_annual_min)s, %(salary_annual_max)s, %(title_family)s, %(industry_bucket)s
            )
            """,
            data,
        )

    def _update_job(self, conn: Any, job_data: dict[str, Any], timestamp: str) -> None:
        data = self._job_data_for_write(job_data, timestamp=timestamp, is_insert=False)
        conn.execute(
            """
            UPDATE jobs SET
                title = %(title)s,
                company_name = %(company_name)s,
                company_uen = %(company_uen)s,
                description = %(description)s,
                salary_min = %(salary_min)s,
                salary_max = %(salary_max)s,
                salary_type = %(salary_type)s,
                employment_type = %(employment_type)s,
                seniority = %(seniority)s,
                min_experience_years = %(min_experience_years)s,
                skills = %(skills)s,
                categories = %(categories)s,
                location = %(location)s,
                district = %(district)s,
                region = %(region)s,
                posted_date = %(posted_date)s,
                expiry_date = %(expiry_date)s,
                applications_count = %(applications_count)s,
                job_url = %(job_url)s,
                last_updated_at = %(last_updated_at)s,
                salary_annual_min = %(salary_annual_min)s,
                salary_annual_max = %(salary_annual_max)s,
                title_family = %(title_family)s,
                industry_bucket = %(industry_bucket)s
            WHERE uuid = %(uuid)s
            """,
            data,
        )

    def populate_normalized_job_metadata(
        self,
        uuids: Optional[list[str]] = None,
        *,
        only_missing: bool = False,
        chunk_size: int = 500,
    ) -> int:
        total_updated = 0
        uuid_chunks = [uuids[i : i + chunk_size] for i in range(0, len(uuids), chunk_size)] if uuids else [None]
        with self._connection() as conn:
            for chunk in uuid_chunks:
                conditions: list[str] = []
                params: list[Any] = []
                if chunk:
                    conditions.append("uuid = ANY(%s)")
                    params.append(chunk)
                if only_missing:
                    conditions.append("(title_family IS NULL OR industry_bucket IS NULL)")
                where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                rows = conn.execute(
                    f"SELECT uuid, title, categories, skills FROM jobs {where_clause}",
                    params,
                ).fetchall()
                updates = []
                for row in rows:
                    title_family, industry_bucket = MCFDatabase._derive_normalized_job_metadata(
                        title=row["title"],
                        categories=row["categories"],
                        skills=row["skills"],
                    )
                    updates.append((title_family, industry_bucket, row["uuid"]))
                if updates:
                    self._executemany(
                        conn,
                        "UPDATE jobs SET title_family = %s, industry_bucket = %s WHERE uuid = %s",
                        updates,
                    )
                    total_updated += len(updates)
        return total_updated

    def get_job(self, uuid: str) -> Optional[dict]:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE uuid = %s", (uuid,)).fetchone()
        return dict(row) if row else None

    def get_jobs_bulk(self, uuids: list[str]) -> dict[str, dict]:
        if not uuids:
            return {}
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM jobs WHERE uuid = ANY(%s)", (uuids,)).fetchall()
        return {row["uuid"]: dict(row) for row in rows}

    def get_job_history(self, uuid: str) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM job_history WHERE job_uuid = %s ORDER BY recorded_at DESC",
                (uuid,),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_jobs(
        self,
        keyword: Optional[str] = None,
        company_name: Optional[str] = None,
        salary_min: Optional[int] = None,
        salary_max: Optional[int] = None,
        employment_type: Optional[str] = None,
        region: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        conditions = []
        params: list[Any] = []
        if keyword:
            like_pattern = f"%{keyword}%"
            conditions.append("(title ILIKE %s OR description ILIKE %s OR skills ILIKE %s)")
            params.extend([like_pattern, like_pattern, like_pattern])
        if company_name:
            conditions.append("company_name ILIKE %s")
            params.append(f"%{company_name}%")
        if salary_min is not None:
            conditions.append("salary_min >= %s")
            params.append(salary_min)
        if salary_max is not None:
            conditions.append("salary_max <= %s")
            params.append(salary_max)
        if employment_type:
            conditions.append("employment_type = %s")
            params.append(employment_type)
        if region:
            conditions.append("region = %s")
            params.append(region)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE {where_clause}
                ORDER BY posted_date DESC NULLS LAST, last_updated_at DESC
                LIMIT %s OFFSET %s
                """,
                params + [limit, offset],
            ).fetchall()
        return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        with self._connection() as conn:
            stats: dict[str, Any] = {}
            stats["total_jobs"] = conn.execute("SELECT COUNT(*) AS count FROM jobs").fetchone()["count"]
            rows = conn.execute(
                "SELECT employment_type, COUNT(*) AS count FROM jobs GROUP BY employment_type ORDER BY count DESC"
            ).fetchall()
            stats["by_employment_type"] = {row["employment_type"]: row["count"] for row in rows}
            rows = conn.execute(
                """
                SELECT company_name, COUNT(*) AS count
                FROM jobs
                WHERE company_name IS NOT NULL AND company_name != ''
                GROUP BY company_name
                ORDER BY count DESC
                LIMIT 10
                """
            ).fetchall()
            stats["top_companies"] = {row["company_name"]: row["count"] for row in rows}
            row = conn.execute(
                """
                SELECT
                    MIN(salary_min) AS min_salary,
                    MAX(salary_max) AS max_salary,
                    AVG(salary_min) AS avg_min,
                    AVG(salary_max) AS avg_max
                FROM jobs
                WHERE salary_min IS NOT NULL AND salary_max IS NOT NULL
                """
            ).fetchone()
            stats["salary_stats"] = {
                "min": row["min_salary"],
                "max": row["max_salary"],
                "avg_min": int(row["avg_min"]) if row["avg_min"] else None,
                "avg_max": int(row["avg_max"]) if row["avg_max"] else None,
            }
            stats["history_records"] = conn.execute("SELECT COUNT(*) AS count FROM job_history").fetchone()["count"]
            stats["jobs_with_history"] = conn.execute(
                "SELECT COUNT(DISTINCT job_uuid) AS count FROM job_history"
            ).fetchone()["count"]
            stats["jobs_added_today"] = conn.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE first_seen_at::date = CURRENT_DATE"
            ).fetchone()["count"]
            stats["jobs_updated_today"] = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM jobs
                WHERE last_updated_at::date = CURRENT_DATE
                  AND first_seen_at != last_updated_at
                """
            ).fetchone()["count"]
        return stats

    def has_job(self, uuid: str) -> bool:
        with self._connection() as conn:
            row = conn.execute("SELECT 1 AS found FROM jobs WHERE uuid = %s LIMIT 1", (uuid,)).fetchone()
        return row is not None

    def get_all_uuids(self) -> set[str]:
        with self._connection() as conn:
            rows = conn.execute("SELECT uuid FROM jobs").fetchall()
        return {row["uuid"] for row in rows}

    def count_jobs(self) -> int:
        with self._connection() as conn:
            return conn.execute("SELECT COUNT(*) AS count FROM jobs").fetchone()["count"]

    def create_session(self, search_query: str, total_jobs: int, session_id: int | None = None) -> int:
        with self._connection() as conn:
            if session_id is None:
                row = conn.execute(
                    """
                    INSERT INTO scrape_sessions (search_query, total_jobs, status)
                    VALUES (%s, %s, 'in_progress')
                    RETURNING id
                    """,
                    (search_query, total_jobs),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    INSERT INTO scrape_sessions (id, search_query, total_jobs, status)
                    VALUES (%s, %s, %s, 'in_progress')
                    RETURNING id
                    """,
                    (session_id, search_query, total_jobs),
                ).fetchone()
        return self._fetch_id(row)

    def update_session(self, session_id: int, fetched_count: int, current_offset: int) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE scrape_sessions
                SET fetched_count = %s, current_offset = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (fetched_count, current_offset, session_id),
            )

    def complete_session(self, session_id: int) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE scrape_sessions
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (session_id,),
            )

    def get_incomplete_session(self, search_query: str) -> Optional[dict]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM scrape_sessions
                WHERE search_query = %s AND status = 'in_progress'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (search_query,),
            ).fetchone()
        return dict(row) if row else None

    def get_all_sessions(self, status: Optional[str] = None) -> list[dict]:
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM scrape_sessions WHERE status = %s ORDER BY started_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM scrape_sessions ORDER BY started_at DESC").fetchall()
        return [dict(row) for row in rows]

    def clear_incomplete_sessions(self) -> int:
        with self._connection() as conn:
            row = conn.execute(
                "UPDATE scrape_sessions SET status = 'interrupted' WHERE status = 'in_progress' RETURNING 1"
            ).fetchall()
        return len(row)

    def export_to_csv(self, output_path, **filters) -> int:
        import pandas as pd

        jobs = self.search_jobs(**filters, limit=1_000_000)
        if not jobs:
            return 0
        df = pd.DataFrame(jobs)
        columns_to_drop = ["id", "first_seen_at", "last_updated_at", "search_document"]
        df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
        df.to_csv(output_path, index=False)
        logger.info("Exported %s jobs to %s", len(df), output_path)
        return len(df)

    def create_historical_session(
        self,
        year: int,
        start_seq: int,
        end_seq: Optional[int] = None,
        session_id: int | None = None,
        conn: Any | None = None,
    ) -> int:
        owns = conn is None
        if conn is None:
            conn = self._connect(write_optimized=True)
        try:
            if session_id is None:
                row = conn.execute(
                    """
                    INSERT INTO historical_scrape_progress
                        (year, start_seq, current_seq, end_seq, status)
                    VALUES (%s, %s, %s, %s, 'in_progress')
                    RETURNING id
                    """,
                    (year, start_seq, start_seq, end_seq),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    INSERT INTO historical_scrape_progress
                        (id, year, start_seq, current_seq, end_seq, status)
                    VALUES (%s, %s, %s, %s, %s, 'in_progress')
                    RETURNING id
                    """,
                    (session_id, year, start_seq, start_seq, end_seq),
                ).fetchone()
            if owns:
                conn.commit()
            return self._fetch_id(row)
        except Exception:
            if owns:
                conn.rollback()
            raise
        finally:
            if owns:
                conn.close()

    def update_historical_progress(
        self,
        session_id: int,
        current_seq: int,
        jobs_found: int,
        jobs_not_found: int,
        consecutive_not_found: int = 0,
        end_seq: Optional[int] = None,
        conn: Any | None = None,
    ) -> None:
        owns = conn is None
        if conn is None:
            conn = self._connect(write_optimized=True)
        try:
            conn.execute(
                """
                UPDATE historical_scrape_progress
                SET current_seq = %s,
                    jobs_found = %s,
                    jobs_not_found = %s,
                    consecutive_not_found = %s,
                    end_seq = COALESCE(%s, end_seq),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (current_seq, jobs_found, jobs_not_found, consecutive_not_found, end_seq, session_id),
            )
            if owns:
                conn.commit()
        except Exception:
            if owns:
                conn.rollback()
            raise
        finally:
            if owns:
                conn.close()

    def complete_historical_session(self, session_id: int, conn: Any | None = None) -> None:
        owns = conn is None
        if conn is None:
            conn = self._connect(write_optimized=True)
        try:
            conn.execute(
                """
                UPDATE historical_scrape_progress
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (session_id,),
            )
            if owns:
                conn.commit()
        except Exception:
            if owns:
                conn.rollback()
            raise
        finally:
            if owns:
                conn.close()

    def get_incomplete_historical_session(self, year: int) -> Optional[dict]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM historical_scrape_progress
                WHERE year = %s AND status = 'in_progress'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (year,),
            ).fetchone()
        return dict(row) if row else None

    def get_all_historical_sessions(self, status: Optional[str] = None) -> list[dict]:
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM historical_scrape_progress
                    WHERE status = %s
                    ORDER BY year DESC, started_at DESC
                    """,
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM historical_scrape_progress ORDER BY year DESC, started_at DESC"
                ).fetchall()
        return [dict(row) for row in rows]

    def clear_incomplete_historical_sessions(self) -> int:
        with self._connection() as conn:
            rows = conn.execute(
                """
                UPDATE historical_scrape_progress
                SET status = 'interrupted'
                WHERE status = 'in_progress'
                RETURNING 1
                """
            ).fetchall()
        return len(rows)

    def get_historical_stats(self) -> dict:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT EXTRACT(YEAR FROM posted_date)::text AS year, COUNT(*) AS count
                FROM jobs
                WHERE posted_date IS NOT NULL
                GROUP BY year
                ORDER BY year DESC
                """
            ).fetchall()
            jobs_by_year = {row["year"]: row["count"] for row in rows if row["year"]}
            rows = conn.execute(
                """
                SELECT year,
                       SUM(jobs_found) AS total_found,
                       SUM(jobs_not_found) AS total_not_found,
                       MAX(current_seq) AS max_seq_reached
                FROM historical_scrape_progress
                GROUP BY year
                ORDER BY year DESC
                """
            ).fetchall()
            scrape_progress = {
                row["year"]: {
                    "jobs_found": row["total_found"],
                    "jobs_not_found": row["total_not_found"],
                    "max_seq_reached": row["max_seq_reached"],
                }
                for row in rows
            }
        return {"jobs_by_year": jobs_by_year, "scrape_progress": scrape_progress}

    def batch_insert_attempts(self, attempts: list[dict], conn: Any | None = None) -> int:
        if not attempts:
            return 0
        owns = conn is None
        if conn is None:
            conn = self._connect(write_optimized=True)
        try:
            self._executemany(
                conn,
                """
                INSERT INTO fetch_attempts (year, sequence, result, error_message, attempted_at)
                VALUES (%(year)s, %(sequence)s, %(result)s, %(error_message)s, CURRENT_TIMESTAMP)
                ON CONFLICT (year, sequence) DO UPDATE SET
                    result = EXCLUDED.result,
                    error_message = EXCLUDED.error_message,
                    attempted_at = CURRENT_TIMESTAMP
                """,
                attempts,
            )
            if owns:
                conn.commit()
            return len(attempts)
        except Exception:
            if owns:
                conn.rollback()
            raise
        finally:
            if owns:
                conn.close()

    def get_missing_sequences(self, year: int) -> list[tuple[int, int]]:
        with self._connection() as conn:
            bounds = conn.execute(
                "SELECT MIN(sequence) AS min_seq, MAX(sequence) AS max_seq FROM fetch_attempts WHERE year = %s",
                (year,),
            ).fetchone()
            if not bounds or bounds["min_seq"] is None:
                return []
            rows = conn.execute(
                """
                SELECT sequence FROM fetch_attempts
                WHERE year = %s AND sequence BETWEEN %s AND %s
                ORDER BY sequence
                """,
                (year, bounds["min_seq"], bounds["max_seq"]),
            ).fetchall()
        attempted = {row["sequence"] for row in rows}
        gaps = []
        gap_start = None
        for seq in range(bounds["min_seq"], bounds["max_seq"] + 1):
            if seq not in attempted:
                if gap_start is None:
                    gap_start = seq
            elif gap_start is not None:
                gaps.append((gap_start, seq - 1))
                gap_start = None
        if gap_start is not None:
            gaps.append((gap_start, bounds["max_seq"]))
        return gaps

    def get_failed_attempts(self, year: int, limit: int = 10000) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT year, sequence, result, error_message, attempted_at
                FROM fetch_attempts
                WHERE year = %s AND result IN ('error', 'rate_limited')
                ORDER BY sequence
                LIMIT %s
                """,
                (year, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_attempt_stats(self, year: int) -> dict:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT result, COUNT(*) AS count FROM fetch_attempts WHERE year = %s GROUP BY result",
                (year,),
            ).fetchall()
            stats = {row["result"]: row["count"] for row in rows}
            stats["total"] = sum(stats.values())
            bounds = conn.execute(
                "SELECT MIN(sequence) AS min_seq, MAX(sequence) AS max_seq FROM fetch_attempts WHERE year = %s",
                (year,),
            ).fetchone()
        if bounds and bounds["min_seq"] is not None:
            stats["min_sequence"] = bounds["min_seq"]
            stats["max_sequence"] = bounds["max_seq"]
            stats["sequence_range"] = bounds["max_seq"] - bounds["min_seq"] + 1
        return stats

    def get_all_attempt_stats(self) -> dict[int, dict]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT year, result, COUNT(*) AS count
                FROM fetch_attempts
                GROUP BY year, result
                ORDER BY year
                """
            ).fetchall()
        stats: dict[int, dict] = {}
        for row in rows:
            stats.setdefault(row["year"], {})[row["result"]] = row["count"]
        for year in stats:
            stats[year]["total"] = sum(stats[year].values())
        return stats

    def update_daemon_state(
        self,
        pid: int,
        status: str,
        current_year: int | None = None,
        current_seq: int | None = None,
    ) -> None:
        with self._connection() as conn:
            if status == "running":
                conn.execute(
                    """
                    UPDATE daemon_state
                    SET pid = %s, status = %s, started_at = CURRENT_TIMESTAMP,
                        last_heartbeat = CURRENT_TIMESTAMP, current_year = %s, current_seq = %s
                    WHERE id = 1
                    """,
                    (pid, status, current_year, current_seq),
                )
            else:
                conn.execute(
                    """
                    UPDATE daemon_state
                    SET pid = %s, status = %s, current_year = %s, current_seq = %s
                    WHERE id = 1
                    """,
                    (pid, status, current_year, current_seq),
                )

    def update_daemon_heartbeat(
        self,
        current_year: int | None = None,
        current_seq: int | None = None,
    ) -> None:
        with self._connection() as conn:
            if current_year is not None:
                conn.execute(
                    """
                    UPDATE daemon_state
                    SET last_heartbeat = CURRENT_TIMESTAMP, current_year = %s, current_seq = %s
                    WHERE id = 1
                    """,
                    (current_year, current_seq),
                )
            else:
                conn.execute("UPDATE daemon_state SET last_heartbeat = CURRENT_TIMESTAMP WHERE id = 1")

    def get_daemon_state(self) -> dict:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM daemon_state WHERE id = 1").fetchone()
        if row:
            return dict(row)
        return {
            "pid": None,
            "status": "stopped",
            "last_heartbeat": None,
            "started_at": None,
            "current_year": None,
            "current_seq": None,
        }

    def clear_daemon_state(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE daemon_state
                SET pid = NULL, status = 'stopped', last_heartbeat = NULL, started_at = NULL,
                    current_year = NULL, current_seq = NULL
                WHERE id = 1
                """
            )

    def upsert_embedding(
        self,
        entity_id: str,
        entity_type: str,
        embedding: np.ndarray,
        model_version: str | None = None,
    ) -> None:
        model = model_version or "all-MiniLM-L6-v2"
        placeholder = self._embedding_placeholder()
        value = self._embedding_parameter(embedding)
        with self._connection() as conn:
            conn.execute(
                f"""
                INSERT INTO embeddings (entity_id, entity_type, embedding, model_version, updated_at)
                VALUES (%s, %s, {placeholder}, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (entity_id, entity_type) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    model_version = EXCLUDED.model_version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (entity_id, entity_type, value, model),
            )

    def get_embedding(self, entity_id: str, entity_type: str) -> Optional[np.ndarray]:
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT {self._embedding_select_clause()} FROM embeddings WHERE entity_id = %s AND entity_type = %s",
                (entity_id, entity_type),
            ).fetchone()
        return self._vector_from_value(row["embedding"]) if row else None

    def get_all_embeddings(
        self,
        entity_type: str,
        model_version: str | None = None,
    ) -> tuple[list[str], np.ndarray]:
        with self._connection() as conn:
            if model_version is None:
                rows = conn.execute(
                    f"""
                    SELECT entity_id, {self._embedding_select_clause()}
                    FROM embeddings
                    WHERE entity_type = %s
                    ORDER BY id
                    """,
                    (entity_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT entity_id, {self._embedding_select_clause()}
                    FROM embeddings
                    WHERE entity_type = %s AND model_version = %s
                    ORDER BY id
                    """,
                    (entity_type, model_version),
                ).fetchall()
        if not rows:
            return [], np.array([])
        ids = [row["entity_id"] for row in rows]
        embeddings = np.array([self._vector_from_value(row["embedding"]) for row in rows], dtype=np.float32)
        return ids, embeddings

    def get_embeddings_for_uuids(
        self,
        uuids: list[str],
        model_version: str | None = None,
    ) -> dict[str, np.ndarray]:
        if not uuids:
            return {}
        with self._connection() as conn:
            if model_version is None:
                rows = conn.execute(
                    f"""
                    SELECT entity_id, {self._embedding_select_clause()}
                    FROM embeddings
                    WHERE entity_type = 'job' AND entity_id = ANY(%s)
                    """,
                    (uuids,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT entity_id, {self._embedding_select_clause()}
                    FROM embeddings
                    WHERE entity_type = 'job' AND entity_id = ANY(%s) AND model_version = %s
                    """,
                    (uuids, model_version),
                ).fetchall()
        return {row["entity_id"]: self._vector_from_value(row["embedding"]) for row in rows}

    def get_embedding_stats(self) -> dict:
        with self._connection() as conn:
            rows = conn.execute("SELECT entity_type, COUNT(*) AS count FROM embeddings GROUP BY entity_type").fetchall()
            type_counts = {row["entity_type"]: row["count"] for row in rows}
            total_jobs = conn.execute("SELECT COUNT(*) AS count FROM jobs").fetchone()["count"]
            model = conn.execute("SELECT model_version FROM embeddings LIMIT 1").fetchone()
        jobs_with_embeddings = type_counts.get("job", 0)
        return {
            "job_embeddings": jobs_with_embeddings,
            "skill_embeddings": type_counts.get("skill", 0),
            "company_embeddings": type_counts.get("company", 0),
            "total_jobs": total_jobs,
            "coverage_pct": (jobs_with_embeddings / total_jobs * 100) if total_jobs > 0 else 0,
            "model_version": model["model_version"] if model else None,
        }

    def delete_embeddings_for_model(self, model_version: str) -> int:
        with self._connection() as conn:
            rows = conn.execute(
                "DELETE FROM embeddings WHERE model_version = %s RETURNING 1",
                (model_version,),
            ).fetchall()
        return len(rows)

    def batch_upsert_embeddings(
        self,
        entity_ids: list[str],
        entity_type: str,
        embeddings: np.ndarray,
        model_version: str | None = None,
    ) -> int:
        if len(entity_ids) != len(embeddings):
            raise ValueError("entity_ids and embeddings must have same length")
        model = model_version or "all-MiniLM-L6-v2"
        placeholder = self._embedding_placeholder()
        with self._connection() as conn:
            for entity_id, embedding in zip(entity_ids, embeddings):
                conn.execute(
                    f"""
                    INSERT INTO embeddings (entity_id, entity_type, embedding, model_version, updated_at)
                    VALUES (%s, %s, {placeholder}, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (entity_id, entity_type) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        model_version = EXCLUDED.model_version,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (entity_id, entity_type, self._embedding_parameter(embedding), model),
                )
        return len(entity_ids)

    def bm25_search(self, query: str, limit: int = 100) -> list[tuple[str, float]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT uuid, -ts_rank_cd(search_document, websearch_to_tsquery('english', %s)) AS score
                FROM jobs
                WHERE search_document @@ websearch_to_tsquery('english', %s)
                ORDER BY score
                LIMIT %s
                """,
                (query, query, limit),
            ).fetchall()
        return [(row["uuid"], row["score"]) for row in rows]

    def bm25_search_filtered(self, query: str, candidate_uuids: set[str]) -> list[tuple[str, float]]:
        if not candidate_uuids:
            return []
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT uuid, -ts_rank_cd(search_document, websearch_to_tsquery('english', %s)) AS score
                FROM jobs
                WHERE uuid = ANY(%s) AND search_document @@ websearch_to_tsquery('english', %s)
                ORDER BY score
                """,
                (query, list(candidate_uuids), query),
            ).fetchall()
        return [(row["uuid"], row["score"]) for row in rows]

    def rebuild_fts_index(self) -> None:
        with self._connection() as conn:
            conn.execute("REINDEX INDEX idx_jobs_search_document")

    def log_search(
        self,
        query: str,
        query_type: str,
        result_count: int,
        latency_ms: float,
        cache_hit: bool = False,
        degraded: bool = False,
        filters_used: dict | None = None,
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO search_analytics
                (query, query_type, result_count, latency_ms, cache_hit, degraded, filters_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    query,
                    query_type,
                    result_count,
                    latency_ms,
                    cache_hit,
                    degraded,
                    json.dumps(filters_used) if filters_used else None,
                ),
            )

    def get_popular_queries(self, days: int = 7, limit: int = 20) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT query, COUNT(*) AS count, AVG(latency_ms) AS avg_latency
                FROM search_analytics
                WHERE searched_at > CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')
                GROUP BY query
                ORDER BY count DESC
                LIMIT %s
                """,
                (days, limit),
            ).fetchall()
        return [{"query": row["query"], "count": row["count"], "avg_latency_ms": row["avg_latency"]} for row in rows]

    def get_search_latency_percentiles(self, days: int = 7) -> dict:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT latency_ms
                FROM search_analytics
                WHERE searched_at > CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')
                ORDER BY latency_ms
                """,
                (days,),
            ).fetchall()
        if not rows:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "count": 0}
        latencies = [row["latency_ms"] for row in rows]
        n = len(latencies)
        return {
            "p50": latencies[int(n * 0.5)],
            "p90": latencies[int(n * 0.9)],
            "p95": latencies[int(n * 0.95)],
            "p99": latencies[min(int(n * 0.99), n - 1)],
            "count": n,
        }

    def get_analytics_summary(self, days: int = 7) -> dict:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT query_type,
                       COUNT(*) AS count,
                       SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) AS cache_hits,
                       SUM(CASE WHEN degraded THEN 1 ELSE 0 END) AS degraded_count
                FROM search_analytics
                WHERE searched_at > CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')
                GROUP BY query_type
                """,
                (days,),
            ).fetchall()
        by_type = {}
        total = 0
        total_cache_hits = 0
        total_degraded = 0
        for row in rows:
            by_type[row["query_type"]] = {
                "count": row["count"],
                "cache_hits": row["cache_hits"],
                "degraded": row["degraded_count"],
            }
            total += row["count"]
            total_cache_hits += row["cache_hits"]
            total_degraded += row["degraded_count"]
        return {
            "total_searches": total,
            "cache_hit_rate": (total_cache_hits / total * 100) if total > 0 else 0,
            "degraded_rate": (total_degraded / total * 100) if total > 0 else 0,
            "by_type": by_type,
        }

    _subtract_months = staticmethod(MCFDatabase._subtract_months)
    _median = staticmethod(MCFDatabase._median)
    _salary_midpoint = staticmethod(MCFDatabase._salary_midpoint)
    _annotate_momentum = staticmethod(MCFDatabase._annotate_momentum)

    @classmethod
    def _month_labels(cls, months: int) -> list[str]:
        return MCFDatabase._month_labels(months)

    def _build_filter_conditions(
        self,
        company_name: str | None = None,
        employment_type: str | None = None,
        region: str | None = None,
        keyword: str | None = None,
        company_exact: bool = False,
    ) -> tuple[list[str], list[Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if company_name:
            if company_exact:
                conditions.append("LOWER(company_name) = LOWER(%s)")
                params.append(company_name)
            else:
                conditions.append("LOWER(company_name) LIKE LOWER(%s)")
                params.append(f"%{company_name}%")
        if employment_type:
            conditions.append("employment_type = %s")
            params.append(employment_type)
        if region:
            conditions.append("region = %s")
            params.append(region)
        if keyword:
            like_pattern = f"%{keyword}%"
            conditions.append(
                "(LOWER(title) LIKE LOWER(%s) OR LOWER(description) LIKE LOWER(%s) OR LOWER(skills) LIKE LOWER(%s))"
            )
            params.extend([like_pattern, like_pattern, like_pattern])
        return conditions, params

    def _fetch_trend_rows(
        self,
        months: int,
        company_name: str | None = None,
        employment_type: str | None = None,
        region: str | None = None,
        keyword: str | None = None,
        company_exact: bool = False,
        skill: str | None = None,
    ) -> list[dict]:
        start_month = self._subtract_months(date.today().replace(day=1), months - 1)
        conditions = ["posted_date IS NOT NULL", "posted_date >= %s"]
        params: list[Any] = [start_month]
        extra_conditions, extra_params = self._build_filter_conditions(
            company_name=company_name,
            employment_type=employment_type,
            region=region,
            keyword=keyword,
            company_exact=company_exact,
        )
        conditions.extend(extra_conditions)
        params.extend(extra_params)
        if skill:
            conditions.append("LOWER(skills) LIKE LOWER(%s)")
            params.append(f"%{skill}%")
        where_clause = " AND ".join(conditions)
        with self._connection() as conn:
            return conn.execute(
                f"""
                SELECT posted_date, salary_annual_min, salary_annual_max, company_name, skills
                FROM jobs
                WHERE {where_clause}
                ORDER BY posted_date ASC
                """,
                params,
            ).fetchall()

    def _get_market_monthly_counts(
        self,
        months: int,
        company_name: str | None = None,
        employment_type: str | None = None,
        region: str | None = None,
    ) -> dict[str, int]:
        rows = self._fetch_trend_rows(
            months=months,
            company_name=company_name,
            employment_type=employment_type,
            region=region,
        )
        counts = {label: 0 for label in self._month_labels(months)}
        for row in rows:
            posted = row["posted_date"]
            month = posted.strftime("%Y-%m") if hasattr(posted, "strftime") else str(posted)[:7]
            if month in counts:
                counts[month] += 1
        return counts

    def _rows_to_series(
        self,
        rows: list[dict],
        months: int,
        market_counts: Optional[dict[str, int]] = None,
    ) -> list[dict]:
        labels = self._month_labels(months)
        points = {
            label: {
                "month": label,
                "job_count": 0,
                "market_share": 0.0,
                "median_salary_annual": None,
                "momentum": 0.0,
            }
            for label in labels
        }
        salary_buckets = {label: [] for label in labels}
        for row in rows:
            posted = row["posted_date"]
            month = posted.strftime("%Y-%m") if hasattr(posted, "strftime") else str(posted)[:7]
            if month not in points:
                continue
            points[month]["job_count"] += 1
            salary = self._salary_midpoint(row)
            if salary is not None:
                salary_buckets[month].append(salary)
        for label in labels:
            points[label]["median_salary_annual"] = self._median(salary_buckets[label])
            market_total = (market_counts or {}).get(label, 0)
            if market_total > 0:
                points[label]["market_share"] = round((points[label]["job_count"] / market_total) * 100, 2)
        series = [points[label] for label in labels]
        self._annotate_momentum(series)
        return series

    def _series_from_aggregates(
        self,
        month_counts: Counter | dict[str, int],
        salary_buckets: dict[str, list[int]],
        labels: list[str],
        market_counts: Optional[dict[str, int]] = None,
    ) -> list[dict]:
        series = []
        for label in labels:
            market_total = (market_counts or {}).get(label, 0)
            job_count = int(month_counts.get(label, 0))
            series.append(
                {
                    "month": label,
                    "job_count": job_count,
                    "market_share": round((job_count / market_total) * 100, 2) if market_total > 0 else 0.0,
                    "median_salary_annual": self._median(salary_buckets.get(label, [])),
                    "momentum": 0.0,
                }
            )
        self._annotate_momentum(series)
        return series

    def get_skill_trends(
        self,
        skills: list[str],
        months: int = 12,
        company_name: str | None = None,
        employment_type: str | None = None,
        region: str | None = None,
    ) -> list[dict]:
        market_counts = self._get_market_monthly_counts(months, company_name, employment_type, region)
        trends = []
        for skill in skills:
            rows = self._fetch_trend_rows(
                months=months,
                company_name=company_name,
                employment_type=employment_type,
                region=region,
                skill=skill,
            )
            series = self._rows_to_series(rows, months, market_counts)
            trends.append({"skill": skill, "series": series, "latest": series[-1] if series else None})
        return trends

    def get_role_trend(
        self,
        query: str,
        months: int = 12,
        company_name: str | None = None,
        employment_type: str | None = None,
        region: str | None = None,
    ) -> dict:
        rows = self._fetch_trend_rows(months, company_name, employment_type, region, keyword=query)
        market_counts = self._get_market_monthly_counts(months, company_name, employment_type, region)
        series = self._rows_to_series(rows, months, market_counts)
        return {"query": query, "series": series, "latest": series[-1] if series else None}

    def get_company_trend(self, company_name: str, months: int = 12) -> dict:
        rows = self._fetch_trend_rows(months=months, company_name=company_name, company_exact=True)
        market_counts = self._get_market_monthly_counts(months=months)
        series = self._rows_to_series(rows, months, market_counts)
        skills_by_month: dict[str, Counter] = {label: Counter() for label in self._month_labels(months)}
        for row in rows:
            posted = row["posted_date"]
            month = posted.strftime("%Y-%m") if hasattr(posted, "strftime") else str(posted)[:7]
            for skill in [item.strip() for item in (row["skills"] or "").split(",") if item.strip()]:
                skills_by_month[month][skill] += 1
        top_skills_by_month = [
            {
                "month": month,
                "skills": [
                    {"skill": skill, "job_count": count, "cluster_id": None} for skill, count in counter.most_common(5)
                ],
            }
            for month, counter in skills_by_month.items()
        ]
        return {"company_name": company_name, "series": series, "top_skills_by_month": top_skills_by_month}

    def get_overview(self, months: int = 12) -> dict:
        labels = self._month_labels(months)
        current_month = labels[-1]
        previous_month = labels[-2] if len(labels) > 1 else labels[-1]
        rows = self._fetch_trend_rows(months=months)
        market_counts: Counter = Counter()
        market_salarys: dict[str, list[int]] = {label: [] for label in labels}
        skill_counts: dict[str, Counter] = defaultdict(Counter)
        skill_salarys: dict[str, dict[str, list[int]]] = defaultdict(lambda: {label: [] for label in labels})
        company_counts: dict[str, Counter] = defaultdict(Counter)
        company_salarys: dict[str, dict[str, list[int]]] = defaultdict(lambda: {label: [] for label in labels})
        unique_skills: set[str] = set()
        unique_companies: set[str] = set()
        salary_midpoints: list[int] = []
        for row in rows:
            posted = row["posted_date"]
            month = posted.strftime("%Y-%m") if hasattr(posted, "strftime") else str(posted)[:7]
            if month not in market_salarys:
                continue
            salary = self._salary_midpoint(row)
            market_counts[month] += 1
            if salary is not None:
                market_salarys[month].append(salary)
                salary_midpoints.append(salary)
            company_name = (row["company_name"] or "").strip()
            if company_name:
                unique_companies.add(company_name)
                company_counts[company_name][month] += 1
                if salary is not None:
                    company_salarys[company_name][month].append(salary)
            for skill in [item.strip() for item in (row["skills"] or "").split(",") if item.strip()]:
                unique_skills.add(skill)
                skill_counts[skill][month] += 1
                if salary is not None:
                    skill_salarys[skill][month].append(salary)
        market_series = self._series_from_aggregates(
            month_counts=market_counts,
            salary_buckets=market_salarys,
            labels=labels,
            market_counts={label: market_counts.get(label, 0) for label in labels},
        )
        top_skills = sorted(
            (skill for skill, counts in skill_counts.items() if sum(counts.values()) >= 10),
            key=lambda skill: sum(skill_counts[skill].values()),
            reverse=True,
        )[:30]
        rising_skills = sorted(
            [
                {
                    "name": skill,
                    "job_count": series[-1]["job_count"] if series else 0,
                    "momentum": series[-1]["momentum"] if series else 0.0,
                    "median_salary_annual": series[-1]["median_salary_annual"] if series else None,
                }
                for skill in top_skills
                for series in [
                    self._series_from_aggregates(
                        month_counts=skill_counts[skill],
                        salary_buckets=skill_salarys[skill],
                        labels=labels,
                        market_counts={label: market_counts.get(label, 0) for label in labels},
                    )
                ]
            ],
            key=lambda item: (item["momentum"], item["job_count"]),
            reverse=True,
        )[:8]
        top_companies = sorted(
            company_counts.keys(), key=lambda company: sum(company_counts[company].values()), reverse=True
        )[:20]
        rising_companies = sorted(
            [
                {
                    "name": company,
                    "job_count": series[-1]["job_count"] if series else 0,
                    "momentum": series[-1]["momentum"] if series else 0.0,
                    "median_salary_annual": series[-1]["median_salary_annual"] if series else None,
                }
                for company in top_companies
                for series in [
                    self._series_from_aggregates(
                        month_counts=company_counts[company],
                        salary_buckets=company_salarys[company],
                        labels=labels,
                        market_counts={label: market_counts.get(label, 0) for label in labels},
                    )
                ]
            ],
            key=lambda item: (item["momentum"], item["job_count"]),
            reverse=True,
        )[:8]
        current_salary = market_series[-1]["median_salary_annual"] if market_series else None
        previous_salary = market_series[-2]["median_salary_annual"] if len(market_series) > 1 else None
        salary_change_pct = 0.0
        if current_salary and previous_salary:
            salary_change_pct = round(((current_salary - previous_salary) / previous_salary) * 100, 2)
        insights = [
            {
                "label": "Monthly hiring velocity",
                "value": market_counts.get(current_month, 0),
                "delta": round(
                    (
                        (market_counts.get(current_month, 0) - market_counts.get(previous_month, 0))
                        / market_counts.get(previous_month, 0)
                    )
                    * 100,
                    2,
                )
                if market_counts.get(previous_month, 0)
                else (100.0 if market_counts.get(current_month, 0) else 0.0),
            },
            {
                "label": "Average annual salary",
                "value": int(sum(salary_midpoints) / len(salary_midpoints)) if salary_midpoints else None,
                "delta": salary_change_pct,
            },
        ]
        return {
            "headline_metrics": {
                "total_jobs": len(rows),
                "current_month_jobs": market_counts.get(current_month, 0),
                "unique_companies": len(unique_companies),
                "unique_skills": len(unique_skills),
                "avg_salary_annual": int(sum(salary_midpoints) / len(salary_midpoints)) if salary_midpoints else None,
            },
            "rising_skills": rising_skills,
            "rising_companies": rising_companies,
            "salary_movement": {
                "current_median_salary_annual": current_salary,
                "previous_median_salary_annual": previous_salary,
                "change_pct": salary_change_pct,
            },
            "market_insights": insights,
        }

    def get_all_companies(self) -> list[str]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT company_name
                FROM jobs
                WHERE company_name IS NOT NULL AND company_name != ''
                ORDER BY company_name
                """
            ).fetchall()
        return [row["company_name"] for row in rows]

    def get_company_stats(self, company_name: str) -> dict:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS job_count,
                    AVG(salary_annual_min) AS avg_salary_min,
                    AVG(salary_annual_max) AS avg_salary_max
                FROM jobs
                WHERE company_name = %s
                """,
                (company_name,),
            ).fetchone()
            skills_rows = conn.execute(
                "SELECT skills FROM jobs WHERE company_name = %s AND skills IS NOT NULL",
                (company_name,),
            ).fetchall()
        skill_counts: Counter = Counter()
        for result in skills_rows:
            skill_counts.update([skill.strip() for skill in (result["skills"] or "").split(",") if skill.strip()])
        avg_salary = None
        if row["avg_salary_min"] and row["avg_salary_max"]:
            avg_salary = int((row["avg_salary_min"] + row["avg_salary_max"]) / 2)
        return {
            "job_count": row["job_count"],
            "avg_salary": avg_salary,
            "top_skills": [name for name, _ in skill_counts.most_common(10)],
        }

    def get_all_unique_skills(self) -> list[str]:
        with self._connection() as conn:
            rows = conn.execute("SELECT DISTINCT skills FROM jobs WHERE skills IS NOT NULL AND skills != ''").fetchall()
        skills_set: set[str] = set()
        for row in rows:
            skills_set.update(skill.strip() for skill in row["skills"].split(",") if skill.strip())
        return sorted(skills_set)

    def get_skill_frequencies(self, min_jobs: int = 1, limit: int = 100) -> list[tuple[str, int]]:
        with self._connection() as conn:
            rows = conn.execute("SELECT skills FROM jobs WHERE skills IS NOT NULL AND skills != ''").fetchall()
        skill_counts: Counter = Counter()
        for row in rows:
            skill_counts.update(skill.strip() for skill in row["skills"].split(",") if skill.strip())
        filtered = [(skill, count) for skill, count in skill_counts.items() if count >= min_jobs]
        filtered.sort(key=lambda item: item[1], reverse=True)
        return filtered[:limit]

    def get_all_unique_companies(self) -> list[str]:
        return self.get_all_companies()

    def get_company_job_embeddings_bulk(self) -> dict[str, list[np.ndarray]]:
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT j.company_name, {self._embedding_select_clause(column="e.embedding")}
                FROM jobs j
                JOIN embeddings e ON e.entity_id = j.uuid AND e.entity_type = 'job'
                WHERE j.company_name IS NOT NULL AND j.company_name != ''
                ORDER BY j.company_name
                """
            ).fetchall()
        result: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in rows:
            result[row["company_name"]].append(self._vector_from_value(row["embedding"]))
        return dict(result)

    def get_jobs_without_embeddings(
        self,
        limit: int = 1000,
        since: date | None = None,
        model_version: str | None = None,
    ) -> list[dict]:
        params: list[Any] = []
        query = """
            SELECT j.uuid, j.title, j.description, j.skills, j.company_name
            FROM jobs j
            LEFT JOIN embeddings e
              ON j.uuid = e.entity_id
             AND e.entity_type = 'job'
        """
        if model_version is not None:
            query = """
                SELECT j.uuid, j.title, j.description, j.skills, j.company_name
                FROM jobs j
                LEFT JOIN embeddings e
                  ON j.uuid = e.entity_id
                 AND e.entity_type = 'job'
                 AND e.model_version = %s
            """
            params.append(model_version)
        query += " WHERE e.id IS NULL"
        if since is not None:
            query += " AND j.posted_date >= %s"
            params.append(since)
        query += " LIMIT %s"
        params.append(limit)
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_all_uuids_since(self, since: date) -> set[str]:
        with self._connection() as conn:
            rows = conn.execute("SELECT uuid FROM jobs WHERE posted_date >= %s", (since,)).fetchall()
        return {row["uuid"] for row in rows}

    def count_jobs_since(self, since: date) -> int:
        with self._connection() as conn:
            return conn.execute("SELECT COUNT(*) AS count FROM jobs WHERE posted_date >= %s", (since,)).fetchone()[
                "count"
            ]

    def vector_search(
        self,
        *,
        entity_type: str,
        query_embedding: np.ndarray,
        limit: int,
        model_version: str | None = None,
        entity_ids: list[str] | None = None,
        prefix: str | None = None,
    ) -> list[tuple[str, float]]:
        if not self._using_pgvector():
            raise RuntimeError("pgvector backend requested but pgvector extension is unavailable")

        literal = self._vector_literal(query_embedding)
        params: list[Any] = [entity_type]
        conditions = ["entity_type = %s"]
        if model_version is not None:
            conditions.append("model_version = %s")
            params.append(model_version)
        if entity_ids is not None:
            conditions.append("entity_id = ANY(%s)")
            params.append(entity_ids)
        if prefix is not None:
            conditions.append("entity_id LIKE %s")
            params.append(f"{prefix}%")
        with self._connection() as conn:
            if self._using_pgvector():
                rows = conn.execute(
                    f"""
                    SELECT entity_id, 1 - (embedding <=> %s::vector) AS score
                    FROM embeddings
                    WHERE {" AND ".join(conditions)}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [literal, *params, literal, limit],
                ).fetchall()
                return [(row["entity_id"], float(row["score"])) for row in rows]
        return []
