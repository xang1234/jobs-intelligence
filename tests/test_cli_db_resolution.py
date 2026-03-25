"""CLI tests for shared database-target resolution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import src.cli as cli

runner = CliRunner()


def test_historical_status_uses_persisted_target_when_db_is_omitted(monkeypatch):
    persisted_dsn = "postgresql://postgres@127.0.0.1:55432/mcf"
    calls: dict[str, object] = {}

    class FakeDB:
        def get_all_historical_sessions(self):
            return []

    def fake_open_database(path: str | None, *, read_only: bool = False, ensure_schema: bool = True):
        calls["db_path"] = path
        calls["read_only"] = read_only
        calls["ensure_schema"] = ensure_schema
        return FakeDB()

    monkeypatch.setattr(
        cli,
        "resolve_preferred_database_value",
        lambda db_path, include_persisted=False: persisted_dsn,
    )
    monkeypatch.setattr(cli, "_open_database", fake_open_database)

    result = runner.invoke(cli.app, ["historical-status"])

    assert result.exit_code == 0
    assert calls == {
        "db_path": persisted_dsn,
        "read_only": True,
        "ensure_schema": True,
    }


def test_migrate_uses_persisted_target_when_db_is_omitted(monkeypatch):
    persisted_dsn = "postgresql://postgres@127.0.0.1:55432/mcf"
    calls: dict[str, object] = {}

    class FakeDB:
        def count_jobs(self):
            return 0

    class FakeMigrator:
        def __init__(self, db_path: str):
            calls["migrator_db_path"] = db_path

        def migrate_all(self, **kwargs):
            calls["migrate_kwargs"] = kwargs
            return SimpleNamespace(
                json_files_processed=0,
                csv_rows_processed=0,
                new_jobs=0,
                updated_jobs=0,
                skipped_duplicates=0,
                link_only_jobs=0,
                errors=[],
            )

    def fake_open_database(path: str | None, *, read_only: bool = False, ensure_schema: bool = True):
        calls["db_path"] = path
        calls["read_only"] = read_only
        return FakeDB()

    monkeypatch.setattr(
        cli,
        "resolve_preferred_database_value",
        lambda db_path, include_persisted=False: persisted_dsn,
    )
    monkeypatch.setattr(cli, "_open_database", fake_open_database)
    monkeypatch.setattr(cli, "MCFMigrator", FakeMigrator)

    result = runner.invoke(cli.app, ["migrate", "--dry-run"])

    assert result.exit_code == 0
    assert calls["db_path"] == persisted_dsn
    assert calls["read_only"] is True
    assert calls["migrator_db_path"] == persisted_dsn


def test_scrape_forwards_resolved_db_path_to_scraper(monkeypatch):
    persisted_dsn = "postgresql://postgres@127.0.0.1:55432/mcf"
    calls: dict[str, object] = {}

    class FakeScraper:
        def __init__(self, **kwargs):
            calls["scraper_kwargs"] = kwargs

        async def scrape(self, *args, **kwargs):
            calls["scrape_args"] = args
            calls["scrape_kwargs"] = kwargs
            return []

        def save(self, *args, **kwargs):
            return "unused.csv"

        def get_dataframe(self):
            raise AssertionError("get_dataframe should not be called when no jobs were scraped")

        @property
        def job_count(self):
            return 0

    monkeypatch.setattr(
        cli,
        "resolve_preferred_database_value",
        lambda db_path, include_persisted=False: persisted_dsn,
    )
    monkeypatch.setattr(cli, "MCFScraper", FakeScraper)

    result = runner.invoke(cli.app, ["scrape", "data scientist"])

    assert result.exit_code == 0
    assert calls["scraper_kwargs"]["db_path"] == persisted_dsn


def test_api_serve_uses_resolved_target_when_db_is_omitted(monkeypatch):
    persisted_dsn = "postgresql://postgres@127.0.0.1:55432/mcf"
    calls: dict[str, object] = {}
    previous_db_path = cli.os.environ.get("MCF_DB_PATH")
    previous_database_url = cli.os.environ.get("DATABASE_URL")
    previous_backend = cli.os.environ.get("MCF_EMBEDDING_BACKEND")
    previous_model_dir = cli.os.environ.get("MCF_ONNX_MODEL_DIR")
    previous_search_backend = cli.os.environ.get("MCF_SEARCH_BACKEND")
    previous_lean_hosted = cli.os.environ.get("MCF_LEAN_HOSTED")
    previous_cors = cli.os.environ.get("MCF_CORS_ORIGINS")
    previous_rate_limit = cli.os.environ.get("MCF_RATE_LIMIT_RPM")

    import uvicorn

    def fake_run(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs

    try:
        monkeypatch.setattr(
            cli,
            "resolve_preferred_database_value",
            lambda db_path, include_persisted=False: persisted_dsn,
        )
        monkeypatch.setattr(cli, "validate_embedding_backend_config", lambda **kwargs: None)
        monkeypatch.setattr(uvicorn, "run", fake_run)

        result = runner.invoke(cli.app, ["api-serve"])

        assert result.exit_code == 0
        assert calls["args"] == ("src.api.app:app",)
        assert cli.os.environ["MCF_DB_PATH"] == persisted_dsn
        assert cli.os.environ["DATABASE_URL"] == persisted_dsn
    finally:
        if previous_db_path is None:
            cli.os.environ.pop("MCF_DB_PATH", None)
        else:
            cli.os.environ["MCF_DB_PATH"] = previous_db_path
        if previous_database_url is None:
            cli.os.environ.pop("DATABASE_URL", None)
        else:
            cli.os.environ["DATABASE_URL"] = previous_database_url
        if previous_backend is None:
            cli.os.environ.pop("MCF_EMBEDDING_BACKEND", None)
        else:
            cli.os.environ["MCF_EMBEDDING_BACKEND"] = previous_backend
        if previous_model_dir is None:
            cli.os.environ.pop("MCF_ONNX_MODEL_DIR", None)
        else:
            cli.os.environ["MCF_ONNX_MODEL_DIR"] = previous_model_dir
        if previous_search_backend is None:
            cli.os.environ.pop("MCF_SEARCH_BACKEND", None)
        else:
            cli.os.environ["MCF_SEARCH_BACKEND"] = previous_search_backend
        if previous_lean_hosted is None:
            cli.os.environ.pop("MCF_LEAN_HOSTED", None)
        else:
            cli.os.environ["MCF_LEAN_HOSTED"] = previous_lean_hosted
        if previous_cors is None:
            cli.os.environ.pop("MCF_CORS_ORIGINS", None)
        else:
            cli.os.environ["MCF_CORS_ORIGINS"] = previous_cors
        if previous_rate_limit is None:
            cli.os.environ.pop("MCF_RATE_LIMIT_RPM", None)
        else:
            cli.os.environ["MCF_RATE_LIMIT_RPM"] = previous_rate_limit


def test_benchmark_uses_resolved_target_when_db_is_omitted(monkeypatch):
    persisted_dsn = "postgresql://postgres@127.0.0.1:55432/mcf"
    calls: dict[str, object] = {}

    def fake_run(cmd, *args, **kwargs):
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        cli,
        "resolve_preferred_database_value",
        lambda db_path, include_persisted=False: persisted_dsn,
    )
    monkeypatch.setattr(cli, "validate_embedding_backend_config", lambda **kwargs: None)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(cli.app, ["benchmark", "--queries", "1", "--warmup", "0"])

    assert result.exit_code == 0
    db_arg_index = calls["cmd"].index("--db")
    assert calls["cmd"][db_arg_index + 1] == persisted_dsn


def test_db_backup_uses_explicit_sqlite_source(monkeypatch, temp_dir: Path):
    calls: dict[str, object] = {}
    source = temp_dir / "live.db"
    backup = temp_dir / "backup.db"

    def fake_backup(source_path, backup_dir, *, prefix):
        calls["source"] = source_path
        calls["backup_dir"] = backup_dir
        calls["prefix"] = prefix
        return backup

    def fake_verify(path):
        calls["verify_path"] = path
        return SimpleNamespace(
            integrity_check="ok",
            jobs_count=123,
            size_bytes=456,
        )

    monkeypatch.setattr(cli, "create_sqlite_hot_backup", fake_backup)
    monkeypatch.setattr(cli, "verify_sqlite_backup", fake_verify)

    result = runner.invoke(cli.app, ["db-backup", "--source", str(source)])

    assert result.exit_code == 0
    assert calls["source"] == str(source)
    assert calls["verify_path"] == backup


def test_pg_migrate_uses_explicit_sqlite_source_and_postgres_target(monkeypatch, temp_dir: Path):
    calls: dict[str, object] = {}
    source = temp_dir / "backup.db"
    target = "postgresql://postgres@127.0.0.1:55432/mcf"

    monkeypatch.setattr(cli, "audit_sqlite_source", lambda path: [])

    def fake_migrate(*, sqlite_path, postgres_dsn, batch_size, truncate_first):
        calls["sqlite_path"] = sqlite_path
        calls["postgres_dsn"] = postgres_dsn
        calls["batch_size"] = batch_size
        calls["truncate_first"] = truncate_first
        return SimpleNamespace(copied_rows={"jobs": 1}, anomalies=[])

    monkeypatch.setattr(cli, "migrate_sqlite_backup_to_postgres", fake_migrate)
    monkeypatch.setattr(cli, "write_migration_report", lambda report, report_path: None)

    result = runner.invoke(
        cli.app,
        ["pg-migrate", "--source", str(source), "--target", target],
    )

    assert result.exit_code == 0
    assert calls["sqlite_path"] == str(source)
    assert calls["postgres_dsn"] == target
