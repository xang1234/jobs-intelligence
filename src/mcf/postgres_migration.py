"""
Migration and hosted-slice tooling for PostgreSQL deployments.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from .hosted_slice import HostedSlicePolicy
from .pg_database import PostgresDatabase


TABLE_COPY_ORDER = [
    "jobs",
    "job_history",
    "scrape_sessions",
    "historical_scrape_progress",
    "fetch_attempts",
    "daemon_state",
    "search_analytics",
]

SEQUENCE_RESET_TABLES = (
    "scrape_sessions",
    "historical_scrape_progress",
    "fetch_attempts",
    "search_analytics",
)

TIMESTAMP_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "jobs": ("first_seen_at", "last_updated_at"),
    "job_history": ("recorded_at",),
    "scrape_sessions": ("started_at", "updated_at", "completed_at"),
    "historical_scrape_progress": ("started_at", "updated_at", "completed_at"),
    "fetch_attempts": ("attempted_at",),
    "daemon_state": ("last_heartbeat", "started_at"),
    "search_analytics": ("searched_at",),
}

BOOLEAN_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "search_analytics": ("cache_hit", "degraded"),
}


@dataclass
class MigrationAnomaly:
    table: str
    row_id: str
    column: str
    raw_value: str
    issue: str


@dataclass
class MigrationReport:
    source: str
    target: str
    started_at: str
    finished_at: str | None = None
    copied_rows: dict[str, int] = field(default_factory=dict)
    anomalies: list[MigrationAnomaly] = field(default_factory=list)


def _open_sqlite_source(path: str | Path) -> sqlite3.Connection:
    source = Path(path)
    conn = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.text_factory = lambda raw: raw.decode("utf-8", "replace")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _is_iso_date(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, date):
        return True
    raw = str(value)
    try:
        date.fromisoformat(raw)
    except ValueError:
        return False
    return len(raw) == 10


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def audit_sqlite_source(sqlite_path: str | Path) -> list[MigrationAnomaly]:
    """Find malformed date values before load."""
    conn = _open_sqlite_source(sqlite_path)
    anomalies: list[MigrationAnomaly] = []
    try:
        for batch in _stream_sqlite_rows(
            conn,
            """
            SELECT uuid, posted_date, expiry_date
            FROM jobs
            """,
        ):
            for row in batch:
                for column in ("posted_date", "expiry_date"):
                    raw_value = row[column]
                    if not _is_iso_date(raw_value):
                        anomalies.append(
                            MigrationAnomaly(
                                table="jobs",
                                row_id=row["uuid"],
                                column=column,
                                raw_value=str(raw_value),
                                issue="invalid_iso_date",
                            )
                        )
    finally:
        conn.close()
    return anomalies


def _truncate_postgres_target(db: PostgresDatabase) -> None:
    with db._connection() as conn:
        conn.execute(
            """
            TRUNCATE TABLE
                search_analytics,
                embeddings,
                fetch_attempts,
                historical_scrape_progress,
                scrape_sessions,
                job_history,
                jobs,
                daemon_state
            RESTART IDENTITY CASCADE
            """
        )
        conn.execute("INSERT INTO daemon_state (id, status) VALUES (1, 'stopped') ON CONFLICT (id) DO NOTHING")


def _coerce_job_row(row: sqlite3.Row, report: MigrationReport) -> dict[str, Any]:
    coerced = dict(row)
    for column in ("posted_date", "expiry_date"):
        raw = coerced.get(column)
        if not _is_iso_date(raw):
            report.anomalies.append(
                MigrationAnomaly(
                    table="jobs",
                    row_id=str(coerced["uuid"]),
                    column=column,
                    raw_value=str(raw),
                    issue="coerced_to_null",
                )
            )
            coerced[column] = None
    for column in TIMESTAMP_COLUMNS_BY_TABLE["jobs"]:
        raw = coerced.get(column)
        parsed = _parse_timestamp(raw)
        if raw not in (None, "") and parsed is None:
            report.anomalies.append(
                MigrationAnomaly(
                    table="jobs",
                    row_id=str(coerced["uuid"]),
                    column=column,
                    raw_value=str(raw),
                    issue="coerced_invalid_timestamp",
                )
            )
        coerced[column] = parsed
    return coerced


def _coerce_timestamp_fields(
    table: str,
    rows: Iterable[sqlite3.Row],
    report: MigrationReport,
    *,
    row_identifier_column: str = "id",
) -> list[dict[str, Any]]:
    timestamp_columns = TIMESTAMP_COLUMNS_BY_TABLE.get(table, ())
    if not timestamp_columns:
        return [dict(row) for row in rows]

    coerced_rows: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        row_id = str(payload.get(row_identifier_column, payload.get("uuid", "unknown")))
        for column in timestamp_columns:
            raw = payload.get(column)
            parsed = _parse_timestamp(raw)
            if raw not in (None, "") and parsed is None:
                report.anomalies.append(
                    MigrationAnomaly(
                        table=table,
                        row_id=row_id,
                        column=column,
                        raw_value=str(raw),
                        issue="coerced_invalid_timestamp",
                    )
                )
            payload[column] = parsed
        for column in BOOLEAN_COLUMNS_BY_TABLE.get(table, ()):
            raw = payload.get(column)
            if raw is None or isinstance(raw, bool):
                continue
            if isinstance(raw, (int, float)):
                payload[column] = bool(raw)
                continue
            lowered = str(raw).strip().lower()
            if lowered in {"1", "true", "t", "yes", "y"}:
                payload[column] = True
            elif lowered in {"0", "false", "f", "no", "n", ""}:
                payload[column] = False
            else:
                report.anomalies.append(
                    MigrationAnomaly(
                        table=table,
                        row_id=row_id,
                        column=column,
                        raw_value=str(raw),
                        issue="coerced_invalid_boolean",
                    )
                )
                payload[column] = None
        coerced_rows.append(payload)
    return coerced_rows


def _chunked(iterable: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _stream_sqlite_rows(
    conn: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] = (),
    *,
    fetch_size: int = 5000,
) -> Iterable[list[sqlite3.Row]]:
    cursor = conn.execute(query, params)
    while True:
        rows = cursor.fetchmany(fetch_size)
        if not rows:
            break
        yield rows


def _reset_postgres_sequences(conn: Any, tables: Iterable[str] = SEQUENCE_RESET_TABLES) -> None:
    for table in tables:
        seq_row = conn.execute(
            "SELECT pg_get_serial_sequence(%s, 'id') AS sequence_name",
            (table,),
        ).fetchone()
        sequence_name = seq_row["sequence_name"] if seq_row else None
        if not sequence_name:
            continue

        max_id = conn.execute(f"SELECT MAX(id) AS max_id FROM {table}").fetchone()["max_id"]
        if max_id is None:
            conn.execute("SELECT setval(CAST(%s AS regclass), %s, false)", (sequence_name, 1))
        else:
            conn.execute("SELECT setval(CAST(%s AS regclass), %s, true)", (sequence_name, max_id))


def _executemany(conn: Any, query: str, params_seq: list[Any]) -> None:
    """psycopg bulk insert helper using a cursor-level executemany."""
    if not params_seq:
        return
    with conn.cursor() as cursor:
        cursor.executemany(query, params_seq)


def _sqlite_table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _postgres_table_count(conn: Any, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"])


def _truncate_target_table(conn: Any, table: str) -> None:
    conn.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")


def _should_skip_resume_copy(table: str, *, source_count: int, target_count: int) -> bool:
    """
    Decide whether a resumed migration can safely skip copying a table.

    ``daemon_state`` must always be recopied because a prior truncated target may
    contain only the synthetic seed row inserted during schema/bootstrap.
    """
    if table == "daemon_state":
        return False
    return target_count == source_count


def migrate_sqlite_backup_to_postgres(
    *,
    sqlite_path: str | Path,
    postgres_dsn: str,
    batch_size: int = 5000,
    truncate_first: bool = True,
) -> MigrationReport:
    """Copy a full SQLite backup into PostgreSQL."""
    report = MigrationReport(
        source=str(sqlite_path),
        target=postgres_dsn,
        started_at=datetime.now().isoformat(),
    )
    source = _open_sqlite_source(sqlite_path)
    target = PostgresDatabase(postgres_dsn, read_only=False, ensure_schema=True)
    source_counts = {table: _sqlite_table_count(source, table) for table in (*TABLE_COPY_ORDER, "embeddings")}
    target_counts: dict[str, int] = {}

    try:
        if truncate_first:
            _truncate_postgres_target(target)
        else:
            with target._connection() as conn:
                for table in source_counts:
                    target_counts[table] = _postgres_table_count(conn, table)

        if not truncate_first and _should_skip_resume_copy(
            "jobs",
            source_count=source_counts["jobs"],
            target_count=target_counts.get("jobs", 0),
        ):
            report.copied_rows["jobs"] = 0
        else:
            conn = target._connect(write_optimized=True)
            try:
                copied = 0
                for chunk in _stream_sqlite_rows(source, "SELECT * FROM jobs ORDER BY id", fetch_size=batch_size):
                    for row in chunk:
                        payload = _coerce_job_row(row, report)
                        target._insert_job(conn, payload, payload.get("last_updated_at") or datetime.now().isoformat())
                        copied += 1
                    conn.commit()
                report.copied_rows["jobs"] = copied
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        conn = target._connect(write_optimized=True)
        try:
            copied = 0
            if not truncate_first and _should_skip_resume_copy(
                "job_history",
                source_count=source_counts["job_history"],
                target_count=target_counts.get("job_history", 0),
            ):
                report.copied_rows["job_history"] = 0
            else:
                if not truncate_first and target_counts.get("job_history", 0) > 0:
                    _truncate_target_table(conn, "job_history")
                    conn.commit()
                for chunk in _stream_sqlite_rows(
                    source,
                    """
                    SELECT job_uuid, title, company_name, salary_min, salary_max,
                           applications_count, description, recorded_at
                    FROM job_history ORDER BY id
                    """,
                    fetch_size=batch_size,
                ):
                    _executemany(
                        conn,
                        """
                        INSERT INTO job_history (
                            job_uuid, title, company_name, salary_min, salary_max,
                            applications_count, description, recorded_at
                        ) VALUES (
                            %(job_uuid)s, %(title)s, %(company_name)s, %(salary_min)s, %(salary_max)s,
                            %(applications_count)s, %(description)s, %(recorded_at)s
                        )
                        """,
                        _coerce_timestamp_fields("job_history", chunk, report, row_identifier_column="job_uuid"),
                    )
                    copied += len(chunk)
                    conn.commit()
                report.copied_rows["job_history"] = copied

            for table in ("scrape_sessions", "historical_scrape_progress", "fetch_attempts", "search_analytics"):
                if not truncate_first and _should_skip_resume_copy(
                    table,
                    source_count=source_counts[table],
                    target_count=target_counts.get(table, 0),
                ):
                    report.copied_rows[table] = 0
                    continue
                if not truncate_first and target_counts.get(table, 0) > 0:
                    _truncate_target_table(conn, table)
                    conn.commit()
                copied = 0
                column_list: str | None = None
                value_list: str | None = None
                for chunk in _stream_sqlite_rows(source, f"SELECT * FROM {table} ORDER BY id", fetch_size=batch_size):
                    if column_list is None or value_list is None:
                        columns = chunk[0].keys()
                        column_list = ", ".join(columns)
                        value_list = ", ".join(f"%({column})s" for column in columns)
                    payloads = _coerce_timestamp_fields(table, chunk, report)
                    _executemany(
                        conn,
                        f"INSERT INTO {table} ({column_list}) VALUES ({value_list})",
                        payloads,
                    )
                    copied += len(chunk)
                    conn.commit()
                report.copied_rows[table] = copied

            daemon_rows: list[dict[str, Any]] = []
            for chunk in _stream_sqlite_rows(source, "SELECT * FROM daemon_state", fetch_size=batch_size):
                daemon_rows.extend(_coerce_timestamp_fields("daemon_state", chunk, report))
            conn.execute("DELETE FROM daemon_state")
            if daemon_rows:
                _executemany(
                    conn,
                    """
                    INSERT INTO daemon_state (id, pid, status, last_heartbeat, started_at, current_year, current_seq)
                    VALUES (%(id)s, %(pid)s, %(status)s, %(last_heartbeat)s, %(started_at)s, %(current_year)s, %(current_seq)s)
                    """,
                    daemon_rows,
                )
            report.copied_rows["daemon_state"] = len(daemon_rows)
            _reset_postgres_sequences(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        if not truncate_first and _should_skip_resume_copy(
            "embeddings",
            source_count=source_counts["embeddings"],
            target_count=target_counts.get("embeddings", 0),
        ):
            report.copied_rows["embeddings"] = 0
        else:
            conn = target._connect(write_optimized=True)
            try:
                if not truncate_first and target_counts.get("embeddings", 0) > 0:
                    _truncate_target_table(conn, "embeddings")
                    conn.commit()
                copied = 0
                embedding_placeholder = target._embedding_placeholder()
                for chunk in _stream_sqlite_rows(
                    source,
                    "SELECT entity_id, entity_type, embedding_blob, model_version FROM embeddings ORDER BY id",
                    fetch_size=batch_size,
                ):
                    payloads = [
                        (
                            row["entity_id"],
                            row["entity_type"],
                            target._embedding_parameter(row["embedding_blob"]),
                            row["model_version"],
                        )
                        for row in chunk
                    ]
                    _executemany(
                        conn,
                        f"""
                        INSERT INTO embeddings (entity_id, entity_type, embedding, model_version, updated_at)
                        VALUES (%s, %s, {embedding_placeholder}, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (entity_id, entity_type) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            model_version = EXCLUDED.model_version,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        payloads,
                    )
                    copied += len(payloads)
                    conn.commit()
                report.copied_rows["embeddings"] = copied
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        report.finished_at = datetime.now().isoformat()
        return report
    finally:
        source.close()


def write_migration_report(report: MigrationReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["anomalies"] = [asdict(anomaly) for anomaly in report.anomalies]
    path.write_text(json.dumps(payload, indent=2))
    return path


def seed_hosted_slice_from_postgres(
    *,
    source_dsn: str,
    target_dsn: str,
    policy: HostedSlicePolicy,
) -> dict[str, int]:
    """
    Seed a lean hosted slice from a full local Postgres dataset.

    Only job rows and job embeddings are copied.
    """
    source = PostgresDatabase(source_dsn, read_only=True, ensure_schema=False)
    target = PostgresDatabase(target_dsn, read_only=False, ensure_schema=True)
    cutoff = policy.cutoff_date()
    _truncate_postgres_target(target)

    with source._connection() as source_conn, target._connection() as target_conn:
        jobs = source_conn.execute(
            """
            SELECT * FROM jobs
            WHERE posted_date IS NOT NULL
              AND posted_date >= %s
            ORDER BY posted_date DESC, id DESC
            """,
            (cutoff,),
        ).fetchall()
        for row in jobs:
            target._insert_job(target_conn, dict(row), row.get("last_updated_at") or datetime.now().isoformat())

    job_ids = []
    with target._connection() as conn:
        rows = conn.execute("SELECT uuid FROM jobs").fetchall()
        job_ids = [row["uuid"] for row in rows]

    with source._connection() as source_conn:
        embeddings = source_conn.execute(
            f"""
            SELECT entity_id, {source._embedding_select_clause()}, model_version
            FROM embeddings
            WHERE entity_type = 'job' AND entity_id = ANY(%s)
            """,
            (job_ids,),
        ).fetchall()
    with target._connection() as conn:
        embedding_placeholder = target._embedding_placeholder()
        _executemany(
            conn,
            f"""
            INSERT INTO embeddings (entity_id, entity_type, embedding, model_version, updated_at)
            VALUES (%s, 'job', {embedding_placeholder}, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (entity_id, entity_type) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                model_version = EXCLUDED.model_version,
                updated_at = CURRENT_TIMESTAMP
            """,
            [
                (
                    row["entity_id"],
                    target._embedding_parameter(PostgresDatabase._vector_from_value(row["embedding"])),
                    row["model_version"],
                )
                for row in embeddings
            ],
        )

    purge_counts = purge_hosted_slice(target_dsn=target_dsn, policy=policy)
    purge_counts["seeded_jobs"] = len(job_ids)
    purge_counts["seeded_job_embeddings"] = len(embeddings)
    return purge_counts


def purge_hosted_slice(*, target_dsn: str, policy: HostedSlicePolicy) -> dict[str, int]:
    """
    Enforce the lean hosted slice rules inside a Postgres deployment.
    """
    target = PostgresDatabase(target_dsn, read_only=False, ensure_schema=True)
    cutoff = policy.cutoff_date()
    with target._connection() as conn:
        company_deleted = conn.execute(
            "DELETE FROM embeddings WHERE entity_type IN ('skill', 'company') RETURNING 1"
        ).fetchall()
        orphan_job_embeddings = conn.execute(
            """
            DELETE FROM embeddings
            WHERE entity_type = 'job'
              AND entity_id NOT IN (
                  SELECT uuid FROM jobs WHERE posted_date IS NOT NULL AND posted_date >= %s
              )
            RETURNING 1
            """,
            (cutoff,),
        ).fetchall()
        deleted_jobs = conn.execute(
            """
            DELETE FROM jobs
            WHERE posted_date IS NULL OR posted_date < %s
            RETURNING 1
            """,
            (cutoff,),
        ).fetchall()
    return {
        "deleted_non_job_embeddings": len(company_deleted),
        "deleted_orphan_job_embeddings": len(orphan_job_embeddings),
        "deleted_jobs": len(deleted_jobs),
    }
