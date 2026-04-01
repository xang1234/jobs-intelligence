"""
SQLite database manager for MCF job data.

Provides persistent storage with:
- Automatic deduplication by job UUID
- History tracking when jobs are updated
- Scrape session tracking (replaces JSON checkpoints)
- Query and export capabilities
- Embedding storage for semantic search
- FTS5 full-text search indexing
- Search analytics tracking
"""

import json
import logging
import os
import sqlite3
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from .industry_taxonomy import classify_industry, normalize_title_family
from .models import Job

logger = logging.getLogger(__name__)

# SQL schema definitions
SCHEMA_SQL = """
-- Main jobs table: current state of each job
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- History table: previous versions of jobs (when updated)
CREATE TABLE IF NOT EXISTS job_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_uuid TEXT NOT NULL,
    title TEXT,
    company_name TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    applications_count INTEGER,
    description TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_uuid) REFERENCES jobs(uuid)
);

-- Scrape sessions table: replaces JSON checkpoints
CREATE TABLE IF NOT EXISTS scrape_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_query TEXT NOT NULL,
    total_jobs INTEGER,
    fetched_count INTEGER DEFAULT 0,
    current_offset INTEGER DEFAULT 0,
    status TEXT DEFAULT 'in_progress',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_jobs_uuid ON jobs(uuid);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_name);
CREATE INDEX IF NOT EXISTS idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX IF NOT EXISTS idx_jobs_posted ON jobs(posted_date);
CREATE INDEX IF NOT EXISTS idx_jobs_employment ON jobs(employment_type);
CREATE INDEX IF NOT EXISTS idx_history_uuid ON job_history(job_uuid);
CREATE INDEX IF NOT EXISTS idx_sessions_query ON scrape_sessions(search_query);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON scrape_sessions(status);

-- Historical scrape progress table: tracks enumeration of job IDs by year
CREATE TABLE IF NOT EXISTS historical_scrape_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    start_seq INTEGER NOT NULL,
    current_seq INTEGER NOT NULL,
    end_seq INTEGER,
    jobs_found INTEGER DEFAULT 0,
    jobs_not_found INTEGER DEFAULT 0,
    consecutive_not_found INTEGER DEFAULT 0,
    status TEXT DEFAULT 'in_progress',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_historical_year ON historical_scrape_progress(year);
CREATE INDEX IF NOT EXISTS idx_historical_status ON historical_scrape_progress(status);

-- Fetch attempts table: tracks every job ID fetch for completeness verification
CREATE TABLE IF NOT EXISTS fetch_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    result TEXT NOT NULL,
    error_message TEXT,
    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(year, sequence)
);

CREATE INDEX IF NOT EXISTS idx_fetch_year_seq ON fetch_attempts(year, sequence);
CREATE INDEX IF NOT EXISTS idx_fetch_result ON fetch_attempts(result);

-- Daemon state table: tracks background process for wake detection
CREATE TABLE IF NOT EXISTS daemon_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    pid INTEGER,
    status TEXT DEFAULT 'stopped',
    last_heartbeat TIMESTAMP,
    started_at TIMESTAMP,
    current_year INTEGER,
    current_seq INTEGER
);

-- Initialize daemon state row if not exists
INSERT OR IGNORE INTO daemon_state (id, status) VALUES (1, 'stopped');
"""

# Schema for embeddings storage (semantic search)
EMBEDDINGS_SCHEMA = """
-- Embeddings table: stores vector embeddings for jobs, skills, companies
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL,           -- UUID for jobs, name for skills/companies
    entity_type TEXT NOT NULL,         -- 'job', 'skill', 'company'
    embedding_blob BLOB NOT NULL,      -- Serialized numpy array (384 × 4 = 1536 bytes)
    model_version TEXT DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_id, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(entity_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_version);
"""

# Schema for FTS5 full-text search (external content table)
FTS5_SCHEMA = """
-- FTS5 virtual table for full-text search on jobs
-- Uses external content from jobs table (no data duplication)
CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts USING fts5(
    uuid,
    title,
    description,
    skills,
    company_name,
    content='jobs',
    content_rowid='id'
);

-- Triggers to keep FTS in sync with jobs table
CREATE TRIGGER IF NOT EXISTS jobs_ai AFTER INSERT ON jobs BEGIN
    INSERT INTO jobs_fts(rowid, uuid, title, description, skills, company_name)
    VALUES (new.id, new.uuid, new.title, new.description, new.skills, new.company_name);
END;

CREATE TRIGGER IF NOT EXISTS jobs_ad AFTER DELETE ON jobs BEGIN
    INSERT INTO jobs_fts(jobs_fts, rowid, uuid, title, description, skills, company_name)
    VALUES ('delete', old.id, old.uuid, old.title, old.description, old.skills, old.company_name);
END;

CREATE TRIGGER IF NOT EXISTS jobs_au AFTER UPDATE ON jobs BEGIN
    INSERT INTO jobs_fts(jobs_fts, rowid, uuid, title, description, skills, company_name)
    VALUES ('delete', old.id, old.uuid, old.title, old.description, old.skills, old.company_name);
    INSERT INTO jobs_fts(rowid, uuid, title, description, skills, company_name)
    VALUES (new.id, new.uuid, new.title, new.description, new.skills, new.company_name);
END;
"""

# Schema for search analytics tracking
ANALYTICS_SCHEMA = """
-- Search analytics: tracks queries for monitoring and optimization
CREATE TABLE IF NOT EXISTS search_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    query_type TEXT DEFAULT 'semantic',  -- 'semantic', 'keyword', 'hybrid'
    result_count INTEGER,
    latency_ms REAL,
    cache_hit BOOLEAN DEFAULT FALSE,
    degraded BOOLEAN DEFAULT FALSE,      -- True if fallback was used
    filters_used TEXT,                   -- JSON of applied filters
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_analytics_time ON search_analytics(searched_at);
CREATE INDEX IF NOT EXISTS idx_analytics_query ON search_analytics(query);
"""


class MCFDatabase:
    """
    SQLite database manager for job data.

    Handles all database operations including schema creation,
    job upserts with history tracking, and query operations.

    Example:
        db = MCFDatabase("data/mcf_jobs.db")
        is_new, was_updated = db.upsert_job(job)
        jobs = db.search_jobs(company_name="Google")
    """

    _VALID_JOURNAL_MODES = frozenset({"delete", "truncate", "persist", "memory", "wal", "off"})

    def __init__(
        self,
        db_path: str = "data/mcf_jobs.db",
        read_only: bool = False,
        ensure_schema: bool = True,
        journal_mode: str | None = None,
    ):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            read_only: Open the database without schema writes
            ensure_schema: Create tables and run migrations on open
            journal_mode: Override journal mode (e.g. 'delete' for Docker).
                Falls back to MCF_SQLITE_JOURNAL_MODE env var.
        """
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.ensure_schema = ensure_schema
        self._journal_mode = journal_mode
        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            if self.ensure_schema:
                self._ensure_schema()

    def _connect(self, write_optimized: bool = False) -> sqlite3.Connection:
        """Create a configured SQLite connection."""
        if self.read_only:
            conn = sqlite3.connect(f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        # Disable memory-mapped I/O for database reads.  Docker Desktop's
        # VirtioFS / gRPC-FUSE does not implement mmap correctly, causing
        # SIGBUS crashes inside the container.  (Note: WAL's -shm file
        # uses a separate mmap that this pragma does NOT control —
        # switching away from WAL mode via _ensure_journal_mode handles that.)
        conn.execute("PRAGMA mmap_size = 0")
        if self.read_only:
            conn.execute("PRAGMA query_only = ON")

        if write_optimized:
            # Only enable WAL if no explicit journal mode override is active.
            if not self._journal_mode and not os.environ.get("MCF_SQLITE_JOURNAL_MODE"):
                conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA temp_store = MEMORY")

        return conn

    @staticmethod
    def can_acquire_write_lock(db_path: str, timeout_ms: int = 1000) -> bool:
        """Check whether a new writer can acquire the SQLite write lock."""
        conn = sqlite3.connect(str(Path(db_path)))
        try:
            conn.execute(f"PRAGMA busy_timeout = {timeout_ms}")
            conn.execute("BEGIN IMMEDIATE")
            conn.rollback()
            return True
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower():
                return False
            raise
        finally:
            conn.close()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
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

    def _ensure_journal_mode(self) -> None:
        """
        Switch journal mode when requested via constructor arg or env var.

        WAL mode uses a memory-mapped -shm file that crashes with SIGBUS
        on Docker Desktop's VirtioFS filesystem.  Setting the mode to
        'delete' checkpoints the WAL and removes the -shm file so the
        database can be safely bind-mounted into a container.

        When a journal mode override is active, write_optimized=True
        connections skip the WAL pragma to avoid reverting the override.
        """
        target = (self._journal_mode or os.environ.get("MCF_SQLITE_JOURNAL_MODE", "")).strip().lower()
        if not target:
            return

        if target not in self._VALID_JOURNAL_MODES:
            raise ValueError(f"Invalid journal mode: {target!r}")

        with self._connection() as conn:
            current = conn.execute("PRAGMA journal_mode").fetchone()[0]
            if current.lower() == target:
                return

            if current.lower() == "wal":
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute(f"PRAGMA journal_mode = {target}")
            logger.info("Switched journal mode from %s to %s", current, target)

    def _ensure_schema(self) -> None:
        """Create tables and run migrations."""
        self._ensure_journal_mode()

        with self._connection() as conn:
            # Core schema (jobs, history, sessions, etc.)
            conn.executescript(SCHEMA_SQL)

            # Embeddings table
            conn.executescript(EMBEDDINGS_SCHEMA)

            # Search analytics table
            conn.executescript(ANALYTICS_SCHEMA)

        # FTS5 must be set up before migrations, because migrations UPDATE
        # the jobs table and will fire FTS5 triggers if they already exist.
        self._ensure_fts5()

        # Run migrations for schema changes to existing tables
        self._migrate_salary_annual()
        self._migrate_normalized_job_metadata()

        logger.debug(f"Database schema ensured at {self.db_path}")

    def _migrate_salary_annual(self) -> None:
        """
        Add salary_annual columns if they don't exist.

        This handles the ALTER TABLE gracefully for existing databases.
        Normalizes all salaries to annual for consistent comparisons.
        """
        with self._connection() as conn:
            # Check if columns exist
            cursor = conn.execute("PRAGMA table_info(jobs)")
            columns = {row[1] for row in cursor.fetchall()}

            if "salary_annual_min" not in columns:
                logger.info("Adding salary_annual columns to jobs table...")
                conn.execute("ALTER TABLE jobs ADD COLUMN salary_annual_min INTEGER")
                conn.execute("ALTER TABLE jobs ADD COLUMN salary_annual_max INTEGER")

                # Disable FTS trigger during bulk UPDATE (non-FTS columns only)
                conn.execute("DROP TRIGGER IF EXISTS jobs_au")
                try:
                    conn.execute("""
                        UPDATE jobs SET
                            salary_annual_min = CASE salary_type
                                WHEN 'Monthly' THEN salary_min * 12
                                WHEN 'Yearly' THEN salary_min
                                WHEN 'Hourly' THEN salary_min * 2080
                                WHEN 'Daily' THEN salary_min * 260
                                ELSE salary_min * 12  -- Assume monthly as default
                            END,
                            salary_annual_max = CASE salary_type
                                WHEN 'Monthly' THEN salary_max * 12
                                WHEN 'Yearly' THEN salary_max
                                WHEN 'Hourly' THEN salary_max * 2080
                                WHEN 'Daily' THEN salary_max * 260
                                ELSE salary_max * 12
                            END
                        WHERE salary_min IS NOT NULL OR salary_max IS NOT NULL
                    """)
                finally:
                    conn.execute("""
                        CREATE TRIGGER IF NOT EXISTS jobs_au AFTER UPDATE ON jobs BEGIN
                            INSERT INTO jobs_fts(
                                jobs_fts, rowid, uuid, title, description, skills, company_name
                            )
                            VALUES (
                                'delete', old.id, old.uuid, old.title, old.description, old.skills, old.company_name
                            );
                            INSERT INTO jobs_fts(rowid, uuid, title, description, skills, company_name)
                            VALUES (new.id, new.uuid, new.title, new.description, new.skills, new.company_name);
                        END
                    """)
                conn.commit()
                logger.info("Salary annual migration complete")

    def _migrate_normalized_job_metadata(self) -> None:
        """
        Add normalized taxonomy columns if they don't exist and backfill safely.

        The migration is additive and idempotent. Unknown values are acceptable
        when a row cannot be classified conservatively.
        """
        with self._connection() as conn:
            cursor = conn.execute("PRAGMA table_info(jobs)")
            columns = {row[1] for row in cursor.fetchall()}

            added_columns = False
            if "title_family" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN title_family TEXT")
                added_columns = True
            if "industry_bucket" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN industry_bucket TEXT")
                added_columns = True

            if added_columns:
                logger.info("Added normalized taxonomy columns to jobs table")

        updated = self.populate_normalized_job_metadata(only_missing=True)
        if updated:
            logger.info("Normalized job metadata migration complete for %s rows", updated)

    def _ensure_fts5(self) -> None:
        """
        Set up FTS5 virtual table and triggers.

        FTS5 tables need special handling because:
        1. They can't be created with IF NOT EXISTS in all cases
        2. Triggers need to be created separately
        3. Existing data needs to be indexed on first setup
        4. External-content FTS indexes can get out of sync after DB
           copy/mount, causing IntegrityError in triggers on UPDATE
        """
        conn = self._connect()
        try:
            # Check if FTS table exists
            fts_exists = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='jobs_fts'
            """).fetchone()

            if not fts_exists:
                logger.info("Creating FTS5 index for jobs...")
                conn.executescript(FTS5_SCHEMA)
                # Populate FTS for existing jobs (first-time setup)
                jobs_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                if jobs_count > 0:
                    conn.execute("INSERT INTO jobs_fts(jobs_fts) VALUES('rebuild')")
                conn.commit()
                logger.info("FTS5 index created")
            else:
                # FTS table exists — run FTS5's built-in integrity check.
                # A simple SELECT COUNT(*) doesn't detect page-level corruption
                # in the shadow tables; integrity-check walks the B-tree pages.
                try:
                    conn.execute("INSERT INTO jobs_fts(jobs_fts) VALUES('integrity-check')")
                    logger.debug("FTS5 index verified")
                except Exception:
                    logger.warning("FTS5 index corrupt, rebuilding...")
                    conn.close()
                    conn = self._repair_fts5()
                    conn.commit()
                    logger.info("FTS5 index rebuilt after recovery")
        finally:
            conn.close()

    def _repair_fts5(self) -> sqlite3.Connection:
        """Drop malformed FTS5 tables and recreate from scratch.

        Returns an open connection that the caller must commit and close.
        On failure, the connection is closed before re-raising.
        """
        conn = self._connect()
        try:
            # Try normal DROP first for each shadow table individually.
            suffixes = ("", "_data", "_idx", "_content", "_docsize", "_config")
            all_dropped = True
            for suffix in suffixes:
                table = f"jobs_fts{suffix}"
                try:
                    conn.executescript(f"DROP TABLE IF EXISTS {table};")
                except Exception:
                    logger.warning("Could not DROP %s normally", table)
                    all_dropped = False

            if not all_dropped:
                # Corruption prevents normal DROP — forcibly remove table
                # entries from sqlite_master and reclaim space with VACUUM.
                logger.warning("Using writable_schema to remove corrupted FTS tables")
                conn.execute("PRAGMA writable_schema = ON")
                conn.execute("DELETE FROM sqlite_master WHERE name LIKE 'jobs_fts%'")
                conn.execute("PRAGMA writable_schema = OFF")
                conn.commit()
                # VACUUM reclaims orphaned pages from the corrupted tables.
                conn.execute("VACUUM")

            conn.executescript(FTS5_SCHEMA)
            conn.execute("INSERT INTO jobs_fts(jobs_fts) VALUES('rebuild')")
            return conn
        except Exception:
            conn.close()
            raise

    def upsert_job(
        self,
        job: Job,
        conn: sqlite3.Connection | None = None,
    ) -> tuple[bool, bool]:
        """
        Insert or update a job record.

        If the job exists and has changes, the old version is saved to history.

        Args:
            job: Job to insert or update

        Returns:
            Tuple of (is_new, was_updated)
            - is_new: True if this was a new job
            - was_updated: True if existing job was updated (changes detected)
        """
        job_data = job.to_flat_dict()
        now = datetime.now().isoformat()

        owns_connection = conn is None
        if conn is None:
            conn = self._connect(write_optimized=True)

        try:
            # Check if job exists
            existing = conn.execute("SELECT * FROM jobs WHERE uuid = ?", (job.uuid,)).fetchone()

            if existing is None:
                # New job - insert
                self._insert_job(conn, job_data, now)
                logger.debug(f"Inserted new job: {job.uuid}")
                if owns_connection:
                    conn.commit()
                return (True, False)

            # Existing job - check for changes
            changes = self._detect_changes(dict(existing), job_data)

            if changes:
                # Save current state to history
                self._save_to_history(conn, existing)

                # Update job
                self._update_job(conn, job_data, now)
                logger.debug(f"Updated job {job.uuid}: {', '.join(changes)}")
                if owns_connection:
                    conn.commit()
                return (False, True)

            # No changes
            if owns_connection:
                conn.commit()
            return (False, False)
        except Exception:
            if owns_connection:
                conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()

    def _insert_job(self, conn: sqlite3.Connection, job_data: dict, timestamp: str) -> None:
        """Insert a new job record."""
        # Calculate annual salary for consistent comparisons
        data = {**job_data, "first_seen_at": timestamp, "last_updated_at": timestamp}
        data["salary_annual_min"], data["salary_annual_max"] = self._calculate_annual_salary(
            job_data.get("salary_min"),
            job_data.get("salary_max"),
            job_data.get("salary_type"),
        )
        data["title_family"], data["industry_bucket"] = self._derive_normalized_job_metadata(
            title=job_data.get("title"),
            categories=job_data.get("categories"),
            skills=job_data.get("skills"),
        )

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
                :uuid, :title, :company_name, :company_uen, :description,
                :salary_min, :salary_max, :salary_type, :employment_type,
                :seniority, :min_experience_years, :skills, :categories,
                :location, :district, :region, :posted_date, :expiry_date,
                :applications_count, :job_url, :first_seen_at, :last_updated_at,
                :salary_annual_min, :salary_annual_max, :title_family, :industry_bucket
            )
            """,
            data,
        )

    def _update_job(self, conn: sqlite3.Connection, job_data: dict, timestamp: str) -> None:
        """Update an existing job record."""
        # Calculate annual salary for consistent comparisons
        data = {**job_data, "last_updated_at": timestamp}
        data["salary_annual_min"], data["salary_annual_max"] = self._calculate_annual_salary(
            job_data.get("salary_min"),
            job_data.get("salary_max"),
            job_data.get("salary_type"),
        )
        data["title_family"], data["industry_bucket"] = self._derive_normalized_job_metadata(
            title=job_data.get("title"),
            categories=job_data.get("categories"),
            skills=job_data.get("skills"),
        )

        conn.execute(
            """
            UPDATE jobs SET
                title = :title,
                company_name = :company_name,
                company_uen = :company_uen,
                description = :description,
                salary_min = :salary_min,
                salary_max = :salary_max,
                salary_type = :salary_type,
                employment_type = :employment_type,
                seniority = :seniority,
                min_experience_years = :min_experience_years,
                skills = :skills,
                categories = :categories,
                location = :location,
                district = :district,
                region = :region,
                posted_date = :posted_date,
                expiry_date = :expiry_date,
                applications_count = :applications_count,
                job_url = :job_url,
                last_updated_at = :last_updated_at,
                salary_annual_min = :salary_annual_min,
                salary_annual_max = :salary_annual_max,
                title_family = :title_family,
                industry_bucket = :industry_bucket
            WHERE uuid = :uuid
            """,
            data,
        )

    @staticmethod
    def _split_metadata_values(raw_value: str | None) -> list[str]:
        """Split comma-separated metadata fields into normalized parts."""
        if not raw_value:
            return []
        return [part.strip() for part in raw_value.split(",") if part.strip()]

    @classmethod
    def _derive_normalized_job_metadata(
        cls,
        *,
        title: str | None,
        categories: str | None,
        skills: str | None,
    ) -> tuple[str, str]:
        """Derive stable title-family and industry-bucket fields for storage."""
        title_family = normalize_title_family(title or "").canonical
        classification = classify_industry(
            cls._split_metadata_values(categories),
            skills=cls._split_metadata_values(skills),
        )
        industry_bucket = f"{classification.sector}/{classification.subsector}"
        return title_family, industry_bucket

    def populate_normalized_job_metadata(
        self,
        uuids: Optional[list[str]] = None,
        *,
        only_missing: bool = False,
        chunk_size: int = 500,
    ) -> int:
        """
        Populate persisted normalized metadata for all or a subset of jobs.

        Args:
            uuids: Optional list of job UUIDs to restrict updates to
            only_missing: Only fill rows with missing normalized columns
            chunk_size: Chunk size for UUID-restricted queries

        Returns:
            Number of job rows updated
        """
        total_updated = 0
        uuid_chunks = [uuids[i : i + chunk_size] for i in range(0, len(uuids), chunk_size)] if uuids else [None]

        with self._connection() as conn:
            # Temporarily disable FTS triggers during bulk UPDATE.
            # This migration only touches title_family/industry_bucket
            # (not FTS-indexed columns), so firing the FTS update trigger
            # is unnecessary and can fail with IntegrityError when the
            # FTS index is out of sync (e.g. after Docker volume mount).
            conn.execute("DROP TRIGGER IF EXISTS jobs_au")
            try:
                for chunk in uuid_chunks:
                    rows, params = self._select_jobs_for_normalized_metadata(
                        conn,
                        uuids=chunk,
                        only_missing=only_missing,
                    )
                    if not rows:
                        continue

                    updates = []
                    for row in rows:
                        title_family, industry_bucket = self._derive_normalized_job_metadata(
                            title=row["title"],
                            categories=row["categories"],
                            skills=row["skills"],
                        )
                        updates.append(
                            {
                                "uuid": row["uuid"],
                                "title_family": title_family,
                                "industry_bucket": industry_bucket,
                            }
                        )

                    conn.executemany(
                        """
                        UPDATE jobs
                        SET title_family = :title_family,
                            industry_bucket = :industry_bucket
                        WHERE uuid = :uuid
                        """,
                        updates,
                    )
                    total_updated += len(updates)
            finally:
                # Restore the FTS update trigger
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS jobs_au AFTER UPDATE ON jobs BEGIN
                        INSERT INTO jobs_fts(jobs_fts, rowid, uuid, title, description, skills, company_name)
                        VALUES ('delete', old.id, old.uuid, old.title, old.description, old.skills, old.company_name);
                        INSERT INTO jobs_fts(rowid, uuid, title, description, skills, company_name)
                        VALUES (new.id, new.uuid, new.title, new.description, new.skills, new.company_name);
                    END
                """)

        return total_updated

    @staticmethod
    def _select_jobs_for_normalized_metadata(
        conn: sqlite3.Connection,
        *,
        uuids: Optional[list[str]],
        only_missing: bool,
    ) -> tuple[list[sqlite3.Row], list[str]]:
        """Select jobs that should receive normalized metadata updates."""
        conditions: list[str] = []
        params: list[str] = []

        if uuids:
            placeholders = ",".join("?" for _ in uuids)
            conditions.append(f"uuid IN ({placeholders})")
            params.extend(uuids)

        if only_missing:
            conditions.append("(title_family IS NULL OR industry_bucket IS NULL)")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = conn.execute(
            f"""
            SELECT uuid, title, categories, skills
            FROM jobs
            {where_clause}
            """,
            params,
        ).fetchall()
        return rows, params

    @staticmethod
    def _calculate_annual_salary(
        salary_min: int | None,
        salary_max: int | None,
        salary_type: str | None,
    ) -> tuple[int | None, int | None]:
        """
        Convert salary to annual equivalent.

        Args:
            salary_min: Minimum salary in original units
            salary_max: Maximum salary in original units
            salary_type: 'Monthly', 'Yearly', 'Hourly', or 'Daily'

        Returns:
            Tuple of (annual_min, annual_max)
        """
        if salary_min is None and salary_max is None:
            return None, None

        # Conversion factors to annual
        multipliers = {
            "Monthly": 12,
            "Yearly": 1,
            "Hourly": 2080,  # 40 hours × 52 weeks
            "Daily": 260,  # 5 days × 52 weeks
        }
        multiplier = multipliers.get(salary_type, 12)  # Default to monthly

        annual_min = int(salary_min * multiplier) if salary_min else None
        annual_max = int(salary_max * multiplier) if salary_max else None

        return annual_min, annual_max

    def _save_to_history(self, conn: sqlite3.Connection, existing: sqlite3.Row) -> None:
        """Save current job state to history table."""
        conn.execute(
            """
            INSERT INTO job_history (
                job_uuid, title, company_name, salary_min, salary_max,
                applications_count, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
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

    def _detect_changes(self, existing: dict, new_data: dict) -> list[str]:
        """
        Detect which fields have changed between existing and new data.

        Returns list of changed field names.
        """
        # Fields we track for changes
        tracked_fields = [
            "title",
            "company_name",
            "salary_min",
            "salary_max",
            "applications_count",
            "description",
            "employment_type",
            "seniority",
        ]

        changes = []
        for field in tracked_fields:
            old_val = existing.get(field)
            new_val = new_data.get(field)

            # Normalize None comparisons
            if old_val != new_val:
                changes.append(field)

        return changes

    def get_job(self, uuid: str) -> Optional[dict]:
        """
        Get a job by UUID.

        Args:
            uuid: Job UUID

        Returns:
            Job data as dict, or None if not found
        """
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE uuid = ?", (uuid,)).fetchone()
            return dict(row) if row else None

    def get_jobs_bulk(self, uuids: list[str]) -> dict[str, dict]:
        """
        Get multiple jobs by UUID in a single query.

        Uses chunked IN clauses to stay within SQLite's variable limit.

        Args:
            uuids: List of job UUIDs

        Returns:
            Dict mapping uuid -> job data dict (missing UUIDs omitted)
        """
        if not uuids:
            return {}

        results: dict[str, dict] = {}
        chunk_size = 500  # Well under SQLite's 999 variable limit

        with self._connection() as conn:
            for i in range(0, len(uuids), chunk_size):
                chunk = uuids[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT * FROM jobs WHERE uuid IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    d = dict(row)
                    results[d["uuid"]] = d

        return results

    def get_job_history(self, uuid: str) -> list[dict]:
        """
        Get history records for a job.

        Args:
            uuid: Job UUID

        Returns:
            List of historical records, newest first
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_history
                WHERE job_uuid = ?
                ORDER BY recorded_at DESC
                """,
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
        """
        Search jobs with filters.

        Args:
            keyword: Search in title, description, skills
            company_name: Filter by company (partial match)
            salary_min: Minimum salary filter
            salary_max: Maximum salary filter
            employment_type: Filter by employment type
            region: Filter by region (exact match)
            limit: Max results to return
            offset: Results offset for pagination

        Returns:
            List of matching jobs
        """
        conditions = []
        params: list[Any] = []

        if keyword:
            conditions.append("(title LIKE ? OR description LIKE ? OR skills LIKE ?)")
            like_pattern = f"%{keyword}%"
            params.extend([like_pattern, like_pattern, like_pattern])

        if company_name:
            conditions.append("company_name LIKE ?")
            params.append(f"%{company_name}%")

        if salary_min is not None:
            conditions.append("salary_min >= ?")
            params.append(salary_min)

        if salary_max is not None:
            conditions.append("salary_max <= ?")
            params.append(salary_max)

        if employment_type:
            conditions.append("employment_type = ?")
            params.append(employment_type)

        if region:
            conditions.append("region = ?")
            params.append(region)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE {where_clause}
                ORDER BY posted_date DESC, last_updated_at DESC
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            ).fetchall()
            return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dict with various statistics
        """
        with self._connection() as conn:
            stats = {}

            # Total jobs
            stats["total_jobs"] = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

            # Jobs by employment type
            rows = conn.execute(
                """
                SELECT employment_type, COUNT(*) as count
                FROM jobs
                GROUP BY employment_type
                ORDER BY count DESC
                """
            ).fetchall()
            stats["by_employment_type"] = {row[0]: row[1] for row in rows}

            # Top companies
            rows = conn.execute(
                """
                SELECT company_name, COUNT(*) as count
                FROM jobs
                WHERE company_name IS NOT NULL AND company_name != ''
                GROUP BY company_name
                ORDER BY count DESC
                LIMIT 10
                """
            ).fetchall()
            stats["top_companies"] = {row[0]: row[1] for row in rows}

            # Salary ranges
            row = conn.execute(
                """
                SELECT
                    MIN(salary_min) as min_salary,
                    MAX(salary_max) as max_salary,
                    AVG(salary_min) as avg_min,
                    AVG(salary_max) as avg_max
                FROM jobs
                WHERE salary_min IS NOT NULL AND salary_max IS NOT NULL
                """
            ).fetchone()
            stats["salary_stats"] = {
                "min": row[0],
                "max": row[1],
                "avg_min": int(row[2]) if row[2] else None,
                "avg_max": int(row[3]) if row[3] else None,
            }

            # History count
            stats["history_records"] = conn.execute("SELECT COUNT(*) FROM job_history").fetchone()[0]

            # Jobs with history
            stats["jobs_with_history"] = conn.execute("SELECT COUNT(DISTINCT job_uuid) FROM job_history").fetchone()[0]

            # Recent activity
            stats["jobs_added_today"] = conn.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE DATE(first_seen_at) = DATE('now')
                """
            ).fetchone()[0]

            stats["jobs_updated_today"] = conn.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE DATE(last_updated_at) = DATE('now')
                AND first_seen_at != last_updated_at
                """
            ).fetchone()[0]

            return stats

    def has_job(self, uuid: str) -> bool:
        """Check if a job with the given UUID exists."""
        with self._connection() as conn:
            row = conn.execute("SELECT 1 FROM jobs WHERE uuid = ? LIMIT 1", (uuid,)).fetchone()
            return row is not None

    def get_all_uuids(self) -> set[str]:
        """Get set of all job UUIDs in database."""
        with self._connection() as conn:
            rows = conn.execute("SELECT uuid FROM jobs").fetchall()
            return {row[0] for row in rows}

    def count_jobs(self) -> int:
        """Get total number of jobs in database."""
        with self._connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    # Scrape session methods (replaces checkpoint functionality)

    def create_session(self, search_query: str, total_jobs: int, session_id: int | None = None) -> int:
        """
        Create a new scrape session.

        Args:
            search_query: The search query being scraped
            total_jobs: Total jobs available for this query

        Returns:
            Session ID
        """
        with self._connection() as conn:
            if session_id is None:
                cursor = conn.execute(
                    """
                    INSERT INTO scrape_sessions (search_query, total_jobs, status)
                    VALUES (?, ?, 'in_progress')
                    """,
                    (search_query, total_jobs),
                )
                return cursor.lastrowid

            conn.execute(
                """
                INSERT INTO scrape_sessions (id, search_query, total_jobs, status)
                VALUES (?, ?, ?, 'in_progress')
                """,
                (session_id, search_query, total_jobs),
            )
            return session_id

    def update_session(
        self,
        session_id: int,
        fetched_count: int,
        current_offset: int,
    ) -> None:
        """Update session progress."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE scrape_sessions
                SET fetched_count = ?, current_offset = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (fetched_count, current_offset, session_id),
            )

    def complete_session(self, session_id: int) -> None:
        """Mark session as completed."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE scrape_sessions
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (session_id,),
            )

    def get_incomplete_session(self, search_query: str) -> Optional[dict]:
        """
        Get the most recent incomplete session for a query.

        Args:
            search_query: Search query to look up

        Returns:
            Session data if found, None otherwise
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM scrape_sessions
                WHERE search_query = ? AND status = 'in_progress'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (search_query,),
            ).fetchone()
            return dict(row) if row else None

    def get_all_sessions(self, status: Optional[str] = None) -> list[dict]:
        """
        Get all scrape sessions.

        Args:
            status: Filter by status (in_progress, completed, interrupted)

        Returns:
            List of session records
        """
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM scrape_sessions
                    WHERE status = ?
                    ORDER BY started_at DESC
                    """,
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM scrape_sessions
                    ORDER BY started_at DESC
                    """
                ).fetchall()
            return [dict(row) for row in rows]

    def clear_incomplete_sessions(self) -> int:
        """
        Mark all incomplete sessions as interrupted.

        Returns:
            Number of sessions updated
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE scrape_sessions
                SET status = 'interrupted'
                WHERE status = 'in_progress'
                """
            )
            return cursor.rowcount

    def export_to_csv(self, output_path: Path, **filters) -> int:
        """
        Export jobs to CSV file.

        Args:
            output_path: Path for output CSV
            **filters: Filters to apply (same as search_jobs)

        Returns:
            Number of jobs exported
        """
        import pandas as pd

        # Use search with high limit to get all matching jobs
        jobs = self.search_jobs(**filters, limit=1000000)

        if not jobs:
            return 0

        df = pd.DataFrame(jobs)

        # Remove internal columns
        columns_to_drop = ["id", "first_seen_at", "last_updated_at"]
        df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} jobs to {output_path}")
        return len(df)

    # Historical scrape progress methods

    def create_historical_session(
        self,
        year: int,
        start_seq: int,
        end_seq: Optional[int] = None,
        session_id: int | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """
        Create a new historical scrape session.

        Args:
            year: Year being scraped (e.g., 2023)
            start_seq: Starting sequence number
            end_seq: Ending sequence number (None if unknown)

        Returns:
            Session ID
        """
        owns_connection = conn is None
        if conn is None:
            conn = self._connect(write_optimized=True)

        try:
            if session_id is None:
                cursor = conn.execute(
                    """
                    INSERT INTO historical_scrape_progress
                        (year, start_seq, current_seq, end_seq, status)
                    VALUES (?, ?, ?, ?, 'in_progress')
                    """,
                    (year, start_seq, start_seq, end_seq),
                )
                created_id = cursor.lastrowid
            else:
                conn.execute(
                    """
                    INSERT INTO historical_scrape_progress
                        (id, year, start_seq, current_seq, end_seq, status)
                    VALUES (?, ?, ?, ?, ?, 'in_progress')
                    """,
                    (session_id, year, start_seq, start_seq, end_seq),
                )
                created_id = session_id
            if owns_connection:
                conn.commit()
            return created_id
        except Exception:
            if owns_connection:
                conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()

    def update_historical_progress(
        self,
        session_id: int,
        current_seq: int,
        jobs_found: int,
        jobs_not_found: int,
        consecutive_not_found: int = 0,
        end_seq: Optional[int] = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Update historical scrape progress."""
        owns_connection = conn is None
        if conn is None:
            conn = self._connect(write_optimized=True)

        try:
            conn.execute(
                """
                UPDATE historical_scrape_progress
                SET current_seq = ?,
                    jobs_found = ?,
                    jobs_not_found = ?,
                    consecutive_not_found = ?,
                    end_seq = COALESCE(?, end_seq),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    current_seq,
                    jobs_found,
                    jobs_not_found,
                    consecutive_not_found,
                    end_seq,
                    session_id,
                ),
            )
            if owns_connection:
                conn.commit()
        except Exception:
            if owns_connection:
                conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()

    def complete_historical_session(
        self,
        session_id: int,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Mark historical session as completed."""
        owns_connection = conn is None
        if conn is None:
            conn = self._connect(write_optimized=True)

        try:
            conn.execute(
                """
                UPDATE historical_scrape_progress
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (session_id,),
            )
            if owns_connection:
                conn.commit()
        except Exception:
            if owns_connection:
                conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()

    def get_incomplete_historical_session(self, year: int) -> Optional[dict]:
        """
        Get the most recent incomplete historical session for a year.

        Args:
            year: Year to look up

        Returns:
            Session data if found, None otherwise
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM historical_scrape_progress
                WHERE year = ? AND status = 'in_progress'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (year,),
            ).fetchone()
            return dict(row) if row else None

    def get_all_historical_sessions(self, status: Optional[str] = None) -> list[dict]:
        """
        Get all historical scrape sessions.

        Args:
            status: Filter by status (in_progress, completed, interrupted)

        Returns:
            List of session records
        """
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM historical_scrape_progress
                    WHERE status = ?
                    ORDER BY year DESC, started_at DESC
                    """,
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM historical_scrape_progress
                    ORDER BY year DESC, started_at DESC
                    """
                ).fetchall()
            return [dict(row) for row in rows]

    def clear_incomplete_historical_sessions(self) -> int:
        """
        Mark all incomplete historical sessions as interrupted.

        Returns:
            Number of sessions updated
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE historical_scrape_progress
                SET status = 'interrupted'
                WHERE status = 'in_progress'
                """
            )
            return cursor.rowcount

    def get_historical_stats(self) -> dict:
        """
        Get statistics about historical scraping.

        Returns:
            Dict with statistics by year
        """
        with self._connection() as conn:
            # Jobs by year (based on posted_date)
            rows = conn.execute(
                """
                SELECT strftime('%Y', posted_date) as year, COUNT(*) as count
                FROM jobs
                WHERE posted_date IS NOT NULL
                GROUP BY year
                ORDER BY year DESC
                """
            ).fetchall()
            jobs_by_year = {row[0]: row[1] for row in rows if row[0]}

            # Session progress by year
            rows = conn.execute(
                """
                SELECT year,
                       SUM(jobs_found) as total_found,
                       SUM(jobs_not_found) as total_not_found,
                       MAX(current_seq) as max_seq_reached
                FROM historical_scrape_progress
                GROUP BY year
                ORDER BY year DESC
                """
            ).fetchall()
            scrape_progress = {
                row[0]: {
                    "jobs_found": row[1],
                    "jobs_not_found": row[2],
                    "max_seq_reached": row[3],
                }
                for row in rows
            }

            return {
                "jobs_by_year": jobs_by_year,
                "scrape_progress": scrape_progress,
            }

    # Fetch attempt logging methods

    def batch_insert_attempts(
        self,
        attempts: list[dict],
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """
        Insert or replace batch of fetch attempts.

        Args:
            attempts: List of attempt dicts with keys:
                - year: int
                - sequence: int
                - result: str ('found', 'not_found', 'error', 'skipped')
                - error_message: str or None

        Returns:
            Number of attempts inserted
        """
        if not attempts:
            return 0

        owns_connection = conn is None
        if conn is None:
            conn = self._connect(write_optimized=True)

        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO fetch_attempts
                    (year, sequence, result, error_message, attempted_at)
                VALUES (:year, :sequence, :result, :error_message, CURRENT_TIMESTAMP)
                """,
                attempts,
            )
            if owns_connection:
                conn.commit()
            return len(attempts)
        except Exception:
            if owns_connection:
                conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()

    def get_missing_sequences(self, year: int) -> list[tuple[int, int]]:
        """
        Find gaps in attempted sequences for a year.

        Uses window functions to detect ranges of missing sequence numbers
        between the first and last attempted sequence.

        Args:
            year: Year to check for gaps

        Returns:
            List of (start_seq, end_seq) tuples representing gaps
        """
        with self._connection() as conn:
            # Get the bounds
            bounds = conn.execute(
                """
                SELECT MIN(sequence) as min_seq, MAX(sequence) as max_seq
                FROM fetch_attempts
                WHERE year = ?
                """,
                (year,),
            ).fetchone()

            if not bounds or bounds["min_seq"] is None:
                return []

            min_seq, max_seq = bounds["min_seq"], bounds["max_seq"]

            # Get all attempted sequences as a set
            rows = conn.execute(
                """
                SELECT sequence FROM fetch_attempts
                WHERE year = ? AND sequence BETWEEN ? AND ?
                ORDER BY sequence
                """,
                (year, min_seq, max_seq),
            ).fetchall()

            attempted = {row["sequence"] for row in rows}

            # Find gaps
            gaps = []
            gap_start = None

            for seq in range(min_seq, max_seq + 1):
                if seq not in attempted:
                    if gap_start is None:
                        gap_start = seq
                else:
                    if gap_start is not None:
                        gaps.append((gap_start, seq - 1))
                        gap_start = None

            # Handle trailing gap
            if gap_start is not None:
                gaps.append((gap_start, max_seq))

            return gaps

    def get_failed_attempts(self, year: int, limit: int = 10000) -> list[dict]:
        """
        Get all retryable attempts for a year.

        Args:
            year: Year to query
            limit: Maximum number of results

        Returns:
            List of attempt dicts with error details
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT year, sequence, result, error_message, attempted_at
                FROM fetch_attempts
                WHERE year = ? AND result IN ('error', 'rate_limited')
                ORDER BY sequence
                LIMIT ?
                """,
                (year, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_attempt_stats(self, year: int) -> dict:
        """
        Get counts by result type for a year.

        Args:
            year: Year to get statistics for

        Returns:
            Dict with counts: found, not_found, error, skipped, total
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT result, COUNT(*) as count
                FROM fetch_attempts
                WHERE year = ?
                GROUP BY result
                """,
                (year,),
            ).fetchall()

            stats = {row["result"]: row["count"] for row in rows}
            stats["total"] = sum(stats.values())

            # Get sequence bounds
            bounds = conn.execute(
                """
                SELECT MIN(sequence) as min_seq, MAX(sequence) as max_seq
                FROM fetch_attempts
                WHERE year = ?
                """,
                (year,),
            ).fetchone()

            if bounds and bounds["min_seq"] is not None:
                stats["min_sequence"] = bounds["min_seq"]
                stats["max_sequence"] = bounds["max_seq"]
                stats["sequence_range"] = bounds["max_seq"] - bounds["min_seq"] + 1

            return stats

    def get_all_attempt_stats(self) -> dict[int, dict]:
        """
        Get attempt statistics for all years.

        Returns:
            Dict mapping year to stats dict
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT year, result, COUNT(*) as count
                FROM fetch_attempts
                GROUP BY year, result
                ORDER BY year
                """
            ).fetchall()

            stats: dict[int, dict] = {}
            for row in rows:
                year = row["year"]
                if year not in stats:
                    stats[year] = {}
                stats[year][row["result"]] = row["count"]

            # Add totals
            for year in stats:
                stats[year]["total"] = sum(stats[year].values())

            return stats

    # Daemon state methods

    def update_daemon_state(
        self,
        pid: int,
        status: str,
        current_year: int | None = None,
        current_seq: int | None = None,
    ) -> None:
        """
        Update daemon state in database.

        Args:
            pid: Process ID
            status: 'running', 'stopped', or 'sleeping'
            current_year: Current year being scraped
            current_seq: Current sequence being scraped
        """
        with self._connection() as conn:
            if status == "running":
                conn.execute(
                    """
                    UPDATE daemon_state
                    SET pid = ?, status = ?, started_at = CURRENT_TIMESTAMP,
                        last_heartbeat = CURRENT_TIMESTAMP,
                        current_year = ?, current_seq = ?
                    WHERE id = 1
                    """,
                    (pid, status, current_year, current_seq),
                )
            else:
                conn.execute(
                    """
                    UPDATE daemon_state
                    SET pid = ?, status = ?, current_year = ?, current_seq = ?
                    WHERE id = 1
                    """,
                    (pid, status, current_year, current_seq),
                )

    def update_daemon_heartbeat(
        self,
        current_year: int | None = None,
        current_seq: int | None = None,
    ) -> None:
        """
        Update daemon heartbeat timestamp.

        Args:
            current_year: Current year being scraped
            current_seq: Current sequence being scraped
        """
        with self._connection() as conn:
            if current_year is not None:
                conn.execute(
                    """
                    UPDATE daemon_state
                    SET last_heartbeat = CURRENT_TIMESTAMP,
                        current_year = ?, current_seq = ?
                    WHERE id = 1
                    """,
                    (current_year, current_seq),
                )
            else:
                conn.execute(
                    """
                    UPDATE daemon_state
                    SET last_heartbeat = CURRENT_TIMESTAMP
                    WHERE id = 1
                    """
                )

    def get_daemon_state(self) -> dict:
        """
        Get current daemon state.

        Returns:
            Dict with daemon state including:
            - pid, status, last_heartbeat, started_at
            - current_year, current_seq
        """
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
        """Reset daemon state to stopped."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE daemon_state
                SET pid = NULL, status = 'stopped',
                    last_heartbeat = NULL, started_at = NULL,
                    current_year = NULL, current_seq = NULL
                WHERE id = 1
                """
            )

    # =========================================================================
    # Embedding Methods (Semantic Search)
    # =========================================================================

    def upsert_embedding(
        self,
        entity_id: str,
        entity_type: str,
        embedding: np.ndarray,
        model_version: str | None = None,
    ) -> None:
        """
        Insert or update an embedding.

        Args:
            entity_id: UUID for jobs, name for skills/companies
            entity_type: 'job', 'skill', or 'company'
            embedding: numpy array of shape (384,) for MiniLM
            model_version: Model used to generate embedding
        """
        blob = embedding.astype(np.float32).tobytes()
        model = model_version or "all-MiniLM-L6-v2"

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO embeddings (entity_id, entity_type, embedding_blob, model_version, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(entity_id, entity_type) DO UPDATE SET
                    embedding_blob = excluded.embedding_blob,
                    model_version = excluded.model_version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (entity_id, entity_type, blob, model),
            )

    def get_embedding(self, entity_id: str, entity_type: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding as numpy array.

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity ('job', 'skill', 'company')

        Returns:
            Embedding array or None if not found
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT embedding_blob FROM embeddings WHERE entity_id = ? AND entity_type = ?",
                (entity_id, entity_type),
            ).fetchone()

        if row:
            return np.frombuffer(row[0], dtype=np.float32)
        return None

    def get_all_embeddings(
        self,
        entity_type: str,
        model_version: str | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """
        Get all embeddings of a type as (IDs, stacked array).

        Useful for batch similarity calculations.

        Args:
            entity_type: 'job', 'skill', or 'company'
            model_version: Optional model version filter

        Returns:
            Tuple of (entity_ids, embeddings_matrix)
            embeddings_matrix has shape (n_entities, embedding_dim)
        """
        with self._connection() as conn:
            if model_version is None:
                rows = conn.execute(
                    "SELECT entity_id, embedding_blob FROM embeddings WHERE entity_type = ? ORDER BY id",
                    (entity_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT entity_id, embedding_blob
                    FROM embeddings
                    WHERE entity_type = ? AND model_version = ?
                    ORDER BY id
                    """,
                    (entity_type, model_version),
                ).fetchall()

        if not rows:
            return [], np.array([])

        ids = [row[0] for row in rows]
        embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        return ids, embeddings

    def get_embeddings_for_uuids(
        self,
        uuids: list[str],
        model_version: str | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Get embeddings for specific job UUIDs.

        Args:
            uuids: List of job UUIDs
            model_version: Optional model version filter

        Returns:
            Dict mapping uuid -> embedding array
        """
        if not uuids:
            return {}

        placeholders = ",".join("?" * len(uuids))
        with self._connection() as conn:
            query = (
                f"SELECT entity_id, embedding_blob FROM embeddings "
                f"WHERE entity_type = 'job' AND entity_id IN ({placeholders})"
            )
            params: list[str] = list(uuids)
            if model_version is not None:
                query += " AND model_version = ?"
                params.append(model_version)
            rows = conn.execute(query, params).fetchall()

        return {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in rows}

    def get_embedding_stats(self) -> dict:
        """
        Get embedding statistics including coverage.

        Returns:
            Dict with counts by type, coverage percentage, model version
        """
        with self._connection() as conn:
            # Count by type
            type_counts = {}
            for row in conn.execute("SELECT entity_type, COUNT(*) FROM embeddings GROUP BY entity_type"):
                type_counts[row[0]] = row[1]

            # Job coverage
            total_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            jobs_with_embeddings = type_counts.get("job", 0)

            # Model version
            model = conn.execute("SELECT model_version FROM embeddings LIMIT 1").fetchone()

        return {
            "job_embeddings": type_counts.get("job", 0),
            "skill_embeddings": type_counts.get("skill", 0),
            "company_embeddings": type_counts.get("company", 0),
            "total_jobs": total_jobs,
            "coverage_pct": (jobs_with_embeddings / total_jobs * 100) if total_jobs > 0 else 0,
            "model_version": model[0] if model else None,
        }

    def delete_embeddings_for_model(self, model_version: str) -> int:
        """
        Delete embeddings for a specific model version.

        Useful when upgrading to a new embedding model.

        Args:
            model_version: Model version to delete

        Returns:
            Number of embeddings deleted
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM embeddings WHERE model_version = ?",
                (model_version,),
            )
            return cursor.rowcount

    def batch_upsert_embeddings(
        self,
        entity_ids: list[str],
        entity_type: str,
        embeddings: np.ndarray,
        model_version: str | None = None,
    ) -> int:
        """
        Batch insert or update embeddings efficiently.

        Args:
            entity_ids: List of entity identifiers
            entity_type: Type of entities
            embeddings: Matrix of shape (n_entities, embedding_dim)
            model_version: Model used

        Returns:
            Number of embeddings upserted
        """
        if len(entity_ids) != len(embeddings):
            raise ValueError("entity_ids and embeddings must have same length")

        model = model_version or "all-MiniLM-L6-v2"
        data = [(eid, entity_type, emb.astype(np.float32).tobytes(), model) for eid, emb in zip(entity_ids, embeddings)]

        with self._connection() as conn:
            conn.executemany(
                """
                INSERT INTO embeddings (entity_id, entity_type, embedding_blob, model_version, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(entity_id, entity_type) DO UPDATE SET
                    embedding_blob = excluded.embedding_blob,
                    model_version = excluded.model_version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                data,
            )
            return len(data)

    # =========================================================================
    # FTS5 Full-Text Search Methods
    # =========================================================================

    def bm25_search(self, query: str, limit: int = 100) -> list[tuple[str, float]]:
        """
        Full-text search using BM25 ranking.

        Args:
            query: Search query (supports FTS5 query syntax)
            limit: Maximum results to return

        Returns:
            List of (uuid, bm25_score) tuples, sorted by relevance.
            Lower scores = more relevant (BM25 returns negative scores).
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT uuid, bm25(jobs_fts) as score
                FROM jobs_fts
                WHERE jobs_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()

        return [(row[0], row[1]) for row in rows]

    def bm25_search_filtered(self, query: str, candidate_uuids: set[str]) -> list[tuple[str, float]]:
        """
        Full-text search restricted to a set of candidate UUIDs.

        Unlike bm25_search which scores globally then filters, this method
        only scores and returns results for the specified candidates. This
        ensures no relevant candidate is missed due to global ranking cutoffs.

        Args:
            query: Search query (supports FTS5 query syntax)
            candidate_uuids: Set of UUIDs to restrict scoring to

        Returns:
            List of (uuid, bm25_score) tuples for matching candidates.
            Lower scores = more relevant (BM25 returns negative scores).
        """
        if not candidate_uuids:
            return []

        with self._connection() as conn:
            # Use a temp table for efficient JOIN with the FTS index.
            # This lets SQLite compute BM25 only for matching candidates
            # rather than ranking the entire corpus first.
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS _bm25_candidates (uuid TEXT PRIMARY KEY)")
            conn.execute("DELETE FROM _bm25_candidates")
            conn.executemany(
                "INSERT INTO _bm25_candidates (uuid) VALUES (?)",
                [(u,) for u in candidate_uuids],
            )

            rows = conn.execute(
                """
                SELECT f.uuid, bm25(jobs_fts) as score
                FROM jobs_fts f
                INNER JOIN _bm25_candidates c ON c.uuid = f.uuid
                WHERE jobs_fts MATCH ?
                ORDER BY score
                """,
                (query,),
            ).fetchall()

        return [(row[0], row[1]) for row in rows]

    def rebuild_fts_index(self) -> None:
        """
        Rebuild FTS index from jobs table.

        Use this to recover from corruption or after bulk data changes.
        """
        with self._connection() as conn:
            conn.execute("INSERT INTO jobs_fts(jobs_fts) VALUES('rebuild')")
            conn.commit()
        logger.info("FTS5 index rebuilt")

    # =========================================================================
    # Search Analytics Methods
    # =========================================================================

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
        """
        Log a search query for analytics.

        Args:
            query: Search query string
            query_type: 'semantic', 'keyword', or 'hybrid'
            result_count: Number of results returned
            latency_ms: Query execution time in milliseconds
            cache_hit: Whether result came from cache
            degraded: Whether fallback was used
            filters_used: Dict of applied filters
        """
        filters_json = json.dumps(filters_used) if filters_used else None

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO search_analytics
                (query, query_type, result_count, latency_ms, cache_hit, degraded, filters_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (query, query_type, result_count, latency_ms, cache_hit, degraded, filters_json),
            )

    def get_popular_queries(self, days: int = 7, limit: int = 20) -> list[dict]:
        """
        Get most popular search queries in the last N days.

        Args:
            days: Number of days to look back
            limit: Maximum queries to return

        Returns:
            List of dicts with query, count, avg_latency_ms
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT query, COUNT(*) as count, AVG(latency_ms) as avg_latency
                FROM search_analytics
                WHERE searched_at > datetime('now', ?)
                GROUP BY query
                ORDER BY count DESC
                LIMIT ?
                """,
                (f"-{days} days", limit),
            ).fetchall()

        return [{"query": r[0], "count": r[1], "avg_latency_ms": r[2]} for r in rows]

    def get_search_latency_percentiles(self, days: int = 7) -> dict:
        """
        Get p50, p90, p95, p99 latency statistics.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with percentile values and total count
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT latency_ms FROM search_analytics
                WHERE searched_at > datetime('now', ?)
                ORDER BY latency_ms
                """,
                (f"-{days} days",),
            ).fetchall()

        if not rows:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "count": 0}

        latencies = [r[0] for r in rows]
        n = len(latencies)

        return {
            "p50": latencies[int(n * 0.5)],
            "p90": latencies[int(n * 0.9)],
            "p95": latencies[int(n * 0.95)],
            "p99": latencies[min(int(n * 0.99), n - 1)],
            "count": n,
        }

    def get_analytics_summary(self, days: int = 7) -> dict:
        """
        Get summary of search analytics.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with total searches, cache hit rate, degraded rate, by type
        """
        with self._connection() as conn:
            # Total and by type
            rows = conn.execute(
                """
                SELECT query_type, COUNT(*) as count,
                       SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
                       SUM(CASE WHEN degraded THEN 1 ELSE 0 END) as degraded_count
                FROM search_analytics
                WHERE searched_at > datetime('now', ?)
                GROUP BY query_type
                """,
                (f"-{days} days",),
            ).fetchall()

        by_type = {}
        total = 0
        total_cache_hits = 0
        total_degraded = 0

        for r in rows:
            by_type[r[0]] = {"count": r[1], "cache_hits": r[2], "degraded": r[3]}
            total += r[1]
            total_cache_hits += r[2]
            total_degraded += r[3]

        return {
            "total_searches": total,
            "cache_hit_rate": (total_cache_hits / total * 100) if total > 0 else 0,
            "degraded_rate": (total_degraded / total * 100) if total > 0 else 0,
            "by_type": by_type,
        }

    # =========================================================================
    # Market Intelligence Methods
    # =========================================================================

    @staticmethod
    def _subtract_months(base: date, months_back: int) -> date:
        """Return the first day of the month `months_back` months before `base`."""
        year = base.year
        month = base.month - months_back
        while month <= 0:
            month += 12
            year -= 1
        return date(year, month, 1)

    @classmethod
    def _month_labels(cls, months: int) -> list[str]:
        """Generate YYYY-MM labels from oldest to newest."""
        start = date.today().replace(day=1)
        labels = []
        for offset in range(months - 1, -1, -1):
            labels.append(cls._subtract_months(start, offset).strftime("%Y-%m"))
        return labels

    @staticmethod
    def _median(values: list[int]) -> Optional[int]:
        """Compute an integer median for a small salary sample."""
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return int((ordered[mid - 1] + ordered[mid]) / 2)

    @staticmethod
    def _salary_midpoint(row: sqlite3.Row | dict) -> Optional[int]:
        """Convert annual salary bounds into a single comparable salary figure."""
        annual_min = row["salary_annual_min"]
        annual_max = row["salary_annual_max"]
        if annual_min is not None and annual_max is not None:
            return int((annual_min + annual_max) / 2)
        if annual_min is not None:
            return int(annual_min)
        if annual_max is not None:
            return int(annual_max)
        return None

    @staticmethod
    def _annotate_momentum(series: list[dict]) -> None:
        """
        Compute month-over-trailing-3-month-average momentum in percent.

        When there is no previous volume baseline, treat a non-zero current value
        as a new emergence with +100 momentum.
        """
        for idx, point in enumerate(series):
            history = [series[j]["job_count"] for j in range(max(0, idx - 3), idx)]
            if not history:
                point["momentum"] = 0.0
                continue

            baseline = sum(history) / len(history)
            current = point["job_count"]

            if baseline == 0:
                point["momentum"] = 100.0 if current > 0 else 0.0
            else:
                point["momentum"] = round(((current - baseline) / baseline) * 100, 2)

    @staticmethod
    def _build_filter_conditions(
        company_name: str | None = None,
        employment_type: str | None = None,
        region: str | None = None,
        keyword: str | None = None,
        company_exact: bool = False,
    ) -> tuple[list[str], list[Any]]:
        """Build reusable WHERE conditions for trend/overview queries."""
        conditions: list[str] = []
        params: list[Any] = []

        if company_name:
            if company_exact:
                conditions.append("LOWER(company_name) = LOWER(?)")
                params.append(company_name)
            else:
                conditions.append("LOWER(company_name) LIKE LOWER(?)")
                params.append(f"%{company_name}%")

        if employment_type:
            conditions.append("employment_type = ?")
            params.append(employment_type)

        if region:
            conditions.append("region = ?")
            params.append(region)

        if keyword:
            conditions.append(
                "(LOWER(title) LIKE LOWER(?) OR LOWER(description) LIKE LOWER(?) OR LOWER(skills) LIKE LOWER(?))"
            )
            like_pattern = f"%{keyword}%"
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
    ) -> list[sqlite3.Row]:
        """Fetch minimally shaped rows for time-series aggregation."""
        start_month = self._subtract_months(date.today().replace(day=1), months - 1)
        conditions = ["posted_date IS NOT NULL", "posted_date >= ?"]
        params: list[Any] = [start_month.isoformat()]

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
            conditions.append("LOWER(skills) LIKE LOWER(?)")
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
        """Get baseline monthly job counts for market share calculations."""
        rows = self._fetch_trend_rows(
            months=months,
            company_name=company_name,
            employment_type=employment_type,
            region=region,
        )
        counts = {label: 0 for label in self._month_labels(months)}
        for row in rows:
            month = row["posted_date"][:7]
            if month in counts:
                counts[month] += 1
        return counts

    def _rows_to_series(
        self,
        rows: list[sqlite3.Row],
        months: int,
        market_counts: Optional[dict[str, int]] = None,
    ) -> list[dict]:
        """Convert raw rows into a dense monthly trend series."""
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
        salary_buckets: dict[str, list[int]] = {label: [] for label in labels}

        for row in rows:
            month = row["posted_date"][:7]
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
        """Build a dense monthly series from pre-aggregated counts and salaries."""
        series = []
        for label in labels:
            market_total = (market_counts or {}).get(label, 0)
            job_count = int(month_counts.get(label, 0))
            point = {
                "month": label,
                "job_count": job_count,
                "market_share": round((job_count / market_total) * 100, 2) if market_total > 0 else 0.0,
                "median_salary_annual": self._median(salary_buckets.get(label, [])),
                "momentum": 0.0,
            }
            series.append(point)

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
        """Return monthly trend series for each requested skill."""
        market_counts = self._get_market_monthly_counts(
            months=months,
            company_name=company_name,
            employment_type=employment_type,
            region=region,
        )
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
            trends.append(
                {
                    "skill": skill,
                    "series": series,
                    "latest": series[-1] if series else None,
                }
            )
        return trends

    def get_role_trend(
        self,
        query: str,
        months: int = 12,
        company_name: str | None = None,
        employment_type: str | None = None,
        region: str | None = None,
    ) -> dict:
        """Return monthly trend data for a role/query string."""
        rows = self._fetch_trend_rows(
            months=months,
            company_name=company_name,
            employment_type=employment_type,
            region=region,
            keyword=query,
        )
        market_counts = self._get_market_monthly_counts(
            months=months,
            company_name=company_name,
            employment_type=employment_type,
            region=region,
        )
        series = self._rows_to_series(rows, months, market_counts)
        return {
            "query": query,
            "series": series,
            "latest": series[-1] if series else None,
        }

    def get_company_trend(self, company_name: str, months: int = 12) -> dict:
        """Return hiring trend and skill mix for a single company."""
        rows = self._fetch_trend_rows(
            months=months,
            company_name=company_name,
            company_exact=True,
        )
        market_counts = self._get_market_monthly_counts(months=months)
        series = self._rows_to_series(rows, months, market_counts)

        skills_by_month: dict[str, Counter] = {label: Counter() for label in self._month_labels(months)}
        for row in rows:
            month = row["posted_date"][:7]
            raw_skills = row["skills"] or ""
            for skill in [item.strip() for item in raw_skills.split(",") if item.strip()]:
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

        return {
            "company_name": company_name,
            "series": series,
            "top_skills_by_month": top_skills_by_month,
        }

    def get_overview(self, months: int = 12) -> dict:
        """Return summary cards and top movers for the homepage overview."""
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
            month = row["posted_date"][:7]
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

            raw_skills = row["skills"] or ""
            for skill in [item.strip() for item in raw_skills.split(",") if item.strip()]:
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
            company_counts.keys(),
            key=lambda company: sum(company_counts[company].values()),
            reverse=True,
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

    # =========================================================================
    # Company and Skills Methods (for semantic search features)
    # =========================================================================

    def get_all_companies(self) -> list[str]:
        """
        Get list of all distinct company names.

        Used by embedding generator to enumerate companies for centroid generation.

        Returns:
            Sorted list of company names
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT company_name FROM jobs "
                "WHERE company_name IS NOT NULL AND company_name != '' "
                "ORDER BY company_name"
            ).fetchall()
        return [row[0] for row in rows]

    def get_company_stats(self, company_name: str) -> dict:
        """
        Get statistics for a company.

        Used by similar companies endpoint.

        Args:
            company_name: Company name to look up

        Returns:
            Dict with job_count, avg_salary, top_skills
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as job_count,
                    AVG(salary_annual_min) as avg_salary_min,
                    AVG(salary_annual_max) as avg_salary_max
                FROM jobs
                WHERE company_name = ?
                """,
                (company_name,),
            ).fetchone()

            # Get skills for this company
            skills_rows = conn.execute(
                "SELECT skills FROM jobs WHERE company_name = ? AND skills IS NOT NULL",
                (company_name,),
            ).fetchall()

        # Parse and count skills
        skill_counts: Counter = Counter()
        for r in skills_rows:
            skills = [s.strip() for s in r[0].split(",") if s.strip()]
            skill_counts.update(skills)

        top_skills = [s for s, _ in skill_counts.most_common(10)]

        avg_salary = None
        if row[1] and row[2]:
            avg_salary = int((row[1] + row[2]) / 2)

        return {
            "job_count": row[0],
            "avg_salary": avg_salary,
            "top_skills": top_skills,
        }

    def get_all_unique_skills(self) -> list[str]:
        """
        Extract all unique skills from job postings.

        Returns:
            Sorted list of unique skill names
        """
        with self._connection() as conn:
            rows = conn.execute("SELECT DISTINCT skills FROM jobs WHERE skills IS NOT NULL AND skills != ''").fetchall()

        skills_set: set[str] = set()
        for row in rows:
            skills = [s.strip() for s in row[0].split(",")]
            skills_set.update(s for s in skills if s)

        return sorted(list(skills_set))

    def get_skill_frequencies(self, min_jobs: int = 1, limit: int = 100) -> list[tuple[str, int]]:
        """
        Get skill frequencies for visualization.

        Args:
            min_jobs: Minimum jobs a skill must appear in
            limit: Maximum skills to return

        Returns:
            List of (skill_name, count) tuples, sorted by frequency descending
        """
        with self._connection() as conn:
            rows = conn.execute("SELECT skills FROM jobs WHERE skills IS NOT NULL AND skills != ''").fetchall()

        skill_counts: Counter = Counter()
        for row in rows:
            skills = [s.strip() for s in row[0].split(",")]
            skill_counts.update(s for s in skills if s)

        filtered = [(skill, count) for skill, count in skill_counts.items() if count >= min_jobs]
        filtered.sort(key=lambda x: x[1], reverse=True)

        return filtered[:limit]

    def get_all_unique_companies(self) -> list[str]:
        """
        Get all unique company names.

        Returns:
            Sorted list of company names
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT company_name FROM jobs
                WHERE company_name IS NOT NULL AND company_name != ''
                """
            ).fetchall()

        return sorted([row[0] for row in rows])

    def get_company_job_embeddings_bulk(self) -> dict[str, list[np.ndarray]]:
        """
        Fetch all job embeddings grouped by company in a single query.

        Replaces the N+1 pattern of querying per-company with one JOIN.
        Used by company centroid generation.

        Returns:
            Dict mapping company_name -> list of embedding arrays
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT j.company_name, e.embedding_blob
                FROM jobs j
                JOIN embeddings e ON e.entity_id = j.uuid AND e.entity_type = 'job'
                WHERE j.company_name IS NOT NULL AND j.company_name != ''
                ORDER BY j.company_name
                """
            ).fetchall()

        result: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in rows:
            result[row["company_name"]].append(np.frombuffer(row["embedding_blob"], dtype=np.float32))
        return dict(result)

    def get_jobs_without_embeddings(
        self,
        limit: int = 1000,
        since: "date | None" = None,
        model_version: str | None = None,
    ) -> list[dict]:
        """
        Get jobs that don't have embeddings yet.

        Useful for batch embedding generation.

        Args:
            limit: Maximum jobs to return
            since: Only include jobs posted on or after this date
            model_version: Optional model version to treat as current

        Returns:
            List of job dicts with uuid, title, description, skills
        """
        with self._connection() as conn:
            params: list = []
            query = """
                SELECT j.uuid, j.title, j.description, j.skills, j.company_name
                FROM jobs j
                LEFT JOIN embeddings e ON j.uuid = e.entity_id AND e.entity_type = 'job'
                """
            if model_version is not None:
                query = """
                    SELECT j.uuid, j.title, j.description, j.skills, j.company_name
                    FROM jobs j
                    LEFT JOIN embeddings e
                      ON j.uuid = e.entity_id
                     AND e.entity_type = 'job'
                     AND e.model_version = ?
                    """
                params.append(model_version)
            query += """
                WHERE e.id IS NULL
                """
            if since is not None:
                query += " AND j.posted_date >= ?"
                params.append(since.isoformat())
            query += " LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_all_uuids_since(self, since: "date") -> set[str]:
        """Get set of all job UUIDs posted on or after a date."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT uuid FROM jobs WHERE posted_date >= ?",
                (since.isoformat(),),
            ).fetchall()
            return {row[0] for row in rows}

    def count_jobs_since(self, since: "date") -> int:
        """Count jobs posted on or after a date."""
        with self._connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE posted_date >= ?",
                (since.isoformat(),),
            ).fetchone()[0]
