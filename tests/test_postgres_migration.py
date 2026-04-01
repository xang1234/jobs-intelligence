from __future__ import annotations

import sqlite3
from datetime import datetime

from src.mcf.postgres_migration import (
    MigrationReport,
    _coerce_timestamp_fields,
    _reset_postgres_sequences,
    _select_hosted_resume_progress_rows,
    _should_skip_resume_copy,
    _stream_sqlite_rows,
)


def test_stream_sqlite_rows_batches_without_fetchall():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, title TEXT)")
    conn.executemany("INSERT INTO jobs (title) VALUES (?)", [(f"job-{i}",) for i in range(5)])

    batches = list(_stream_sqlite_rows(conn, "SELECT * FROM jobs ORDER BY id", fetch_size=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [row["title"] for batch in batches for row in batch] == [f"job-{i}" for i in range(5)]


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeSequenceConn:
    def __init__(self):
        self.calls = []

    def execute(self, query, params=None):
        self.calls.append((query, params))
        if "pg_get_serial_sequence" in query:
            table = params[0]
            return _FakeResult({"sequence_name": f"public.{table}_id_seq"})
        if "SELECT MAX(id)" in query:
            if "scrape_sessions" in query:
                return _FakeResult({"max_id": 12})
            return _FakeResult({"max_id": None})
        return _FakeResult(None)


def test_reset_postgres_sequences_uses_max_id_and_empty_table_defaults():
    conn = _FakeSequenceConn()

    _reset_postgres_sequences(conn, tables=("scrape_sessions", "search_analytics"))

    setval_calls = [(query, params) for query, params in conn.calls if "SELECT setval" in query]
    assert setval_calls == [
        ("SELECT setval(CAST(%s AS regclass), %s, true)", ("public.scrape_sessions_id_seq", 12)),
        ("SELECT setval(CAST(%s AS regclass), %s, false)", ("public.search_analytics_id_seq", 1)),
    ]


def test_coerce_timestamp_fields_replaces_invalid_values_with_none():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE search_analytics (id INTEGER PRIMARY KEY, searched_at TEXT, cache_hit INTEGER, degraded TEXT)"
    )
    conn.execute(
        "INSERT INTO search_analytics (id, searched_at, cache_hit, degraded) "
        "VALUES (1, '2026-03-24T12:34:56', 1, 'false')"
    )
    conn.execute(
        "INSERT INTO search_analytics (id, searched_at, cache_hit, degraded) VALUES (2, 'bad-timestamp', 0, 'true')"
    )
    rows = conn.execute("SELECT * FROM search_analytics ORDER BY id").fetchall()

    report = MigrationReport(source="sqlite", target="postgres", started_at=datetime.now().isoformat())
    payloads = _coerce_timestamp_fields("search_analytics", rows, report)

    assert payloads[0]["searched_at"] == datetime.fromisoformat("2026-03-24T12:34:56")
    assert payloads[0]["cache_hit"] is True
    assert payloads[0]["degraded"] is False
    assert payloads[1]["searched_at"] is None
    assert payloads[1]["cache_hit"] is False
    assert payloads[1]["degraded"] is True
    assert report.anomalies[-1].issue == "coerced_invalid_timestamp"


def test_resume_never_skips_daemon_state_even_when_counts_match():
    assert _should_skip_resume_copy("jobs", source_count=10, target_count=10) is True
    assert _should_skip_resume_copy("daemon_state", source_count=1, target_count=1) is False


def test_select_hosted_resume_progress_rows_promotes_latest_completed_row_for_resume():
    rows = [
        {
            "id": 1,
            "year": 2026,
            "status": "completed",
            "current_seq": 100,
            "end_seq": 99,
            "completed_at": "2026-03-01T00:20:00",
            "started_at": "2026-03-01T00:00:00",
            "updated_at": "2026-03-01T00:10:00",
        },
        {
            "id": 2,
            "year": 2025,
            "status": "in_progress",
            "current_seq": 999,
            "started_at": "2026-03-02T00:00:00",
            "updated_at": "2026-03-02T00:10:00",
        },
        {
            "id": 3,
            "year": 2026,
            "status": "in_progress",
            "current_seq": 150,
            "started_at": "2026-03-03T00:00:00",
            "updated_at": "2026-03-03T00:10:00",
        },
        {
            "id": 4,
            "year": 2026,
            "status": "completed",
            "current_seq": 201,
            "end_seq": 200,
            "completed_at": "2026-03-04T00:20:00",
            "started_at": "2026-03-04T00:00:00",
            "updated_at": "2026-03-04T00:10:00",
        },
    ]

    selected = _select_hosted_resume_progress_rows(rows, year=2026)

    assert len(selected) == 1
    assert selected[0]["year"] == 2026
    assert selected[0]["status"] == "in_progress"
    assert selected[0]["current_seq"] == 200
    assert selected[0]["end_seq"] == 250_000
    assert selected[0]["completed_at"] is None
    assert "id" not in selected[0]


def test_select_hosted_resume_progress_rows_keeps_latest_existing_in_progress_row():
    rows = [
        {
            "id": 1,
            "year": 2026,
            "status": "completed",
            "current_seq": 100,
            "end_seq": 99,
            "started_at": "2026-03-01T00:00:00",
            "updated_at": "2026-03-01T00:10:00",
        },
        {
            "id": 2,
            "year": 2026,
            "status": "in_progress",
            "current_seq": 150,
            "end_seq": 250_000,
            "started_at": "2026-03-02T00:00:00",
            "updated_at": "2026-03-02T00:10:00",
        },
    ]

    selected = _select_hosted_resume_progress_rows(rows, year=2026)

    assert len(selected) == 1
    assert selected[0]["status"] == "in_progress"
    assert selected[0]["current_seq"] == 150
    assert selected[0]["end_seq"] == 250_000


def test_select_hosted_resume_progress_rows_returns_empty_when_hosted_year_missing():
    rows = [
        {
            "id": 1,
            "year": 2025,
            "status": "completed",
            "current_seq": 100,
            "end_seq": 99,
            "started_at": "2026-03-01T00:00:00",
            "updated_at": "2026-03-01T00:10:00",
        }
    ]

    assert _select_hosted_resume_progress_rows(rows, year=2026) == []
