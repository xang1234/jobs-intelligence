"""CLI tests for daemon lifecycle commands."""

import sqlite3
from pathlib import Path

from typer.testing import CliRunner

import src.cli as cli

runner = CliRunner()


def test_daemon_start_uses_read_only_database(monkeypatch, temp_dir: Path):
    """The parent launcher should not require a writable DB handle."""
    db_path = temp_dir / "test.db"
    calls: dict[str, object] = {}

    def fake_open_database(path: str | None, *, read_only: bool = False, ensure_schema: bool = True):
        calls["db_path"] = path
        calls["read_only"] = read_only
        calls["ensure_schema"] = ensure_schema
        return object()

    class FakeDaemon:
        logfile = temp_dir / "scraper_daemon.log"

        def __init__(self, db):
            calls["daemon_db"] = db

        def start(self, **kwargs):
            calls["start_kwargs"] = kwargs
            return 12345

    monkeypatch.setattr(cli, "_open_database", fake_open_database)
    monkeypatch.setattr(cli, "ScraperDaemon", FakeDaemon)

    result = runner.invoke(
        cli.app,
        ["daemon", "start", "--year", "2022", "--db", str(db_path)],
    )

    assert result.exit_code == 0
    assert calls["db_path"] == str(db_path)
    assert calls["read_only"] is True
    assert calls["ensure_schema"] is True
    assert calls["start_kwargs"] == {
        "year": 2022,
        "all_years": False,
        "rate_limit": 2.0,
        "db_path": str(db_path),
        "max_rate_limit_retries": 4,
        "cooldown_seconds": 30.0,
        "discover_bounds": True,
    }


def test_daemon_worker_skips_schema_on_locked_existing_database(monkeypatch, temp_dir: Path):
    """Worker should fall back when schema init hits a lock on an existing DB."""
    db_path = temp_dir / "test.db"
    db_path.touch()
    calls: list[tuple[str, bool, bool]] = []

    def fake_open_database(path: str | None, *, read_only: bool = False, ensure_schema: bool = True):
        calls.append((str(path), read_only, ensure_schema))
        if ensure_schema:
            raise sqlite3.OperationalError("database is locked")
        return object()

    class FakeDaemon:
        def __init__(self, db, **kwargs):
            self.db = db

        def run_worker(self, scraper_func):
            return None

    monkeypatch.setattr(cli, "_open_database", fake_open_database)
    monkeypatch.setattr(cli, "ScraperDaemon", FakeDaemon)

    result = runner.invoke(
        cli.app,
        ["_daemon-worker", "--year", "2022", "--db", str(db_path)],
    )

    assert result.exit_code == 0
    assert calls == [
        (str(db_path), False, True),
        (str(db_path), False, False),
    ]


def test_daemon_start_fails_cleanly_when_database_is_busy(monkeypatch, temp_dir: Path):
    """Start should surface daemon startup failures cleanly."""
    db_path = temp_dir / "test.db"

    def fake_open_database(path: str | None, *, read_only: bool = False, ensure_schema: bool = True):
        return object()

    class FakeDaemon:
        def __init__(self, db):
            self.db = db

        def start(self, **kwargs):
            raise cli.DaemonError("Database is busy: another process is writing to it")

    monkeypatch.setattr(cli, "_open_database", fake_open_database)
    monkeypatch.setattr(cli, "ScraperDaemon", FakeDaemon)

    result = runner.invoke(
        cli.app,
        ["daemon", "start", "--year", "2022", "--db", str(db_path)],
    )

    assert result.exit_code == 1
    assert "Database is busy" in result.stdout
    assert "Stop the other scrape process" in result.stdout


def test_daemon_start_surfaces_unreachable_database(monkeypatch, temp_dir: Path):
    """Start should distinguish 'server down' from 'another writer' so users
    don't chase phantom processes when the real problem is connectivity."""
    db_path = temp_dir / "test.db"

    def fake_open_database(path: str | None, *, read_only: bool = False, ensure_schema: bool = True):
        return object()

    class FakeDaemon:
        def __init__(self, db):
            self.db = db

        def start(self, **kwargs):
            raise cli.DaemonError(
                "Database unavailable: connection to server at '127.0.0.1', "
                "port 55432 failed: Connection refused"
            )

    monkeypatch.setattr(cli, "_open_database", fake_open_database)
    monkeypatch.setattr(cli, "ScraperDaemon", FakeDaemon)

    result = runner.invoke(
        cli.app,
        ["daemon", "start", "--year", "2022", "--db", str(db_path)],
    )

    assert result.exit_code == 1
    assert "Database unavailable" in result.stdout
    assert "Connection refused" in result.stdout
    # The "stop the other process" hint would be misleading here.
    assert "Stop the other scrape process" not in result.stdout
    assert "database server is running and reachable" in result.stdout


def test_daemon_start_uses_persisted_database_target_when_db_is_omitted(monkeypatch, temp_dir: Path):
    calls: dict[str, object] = {}
    persisted_dsn = "postgresql://postgres@127.0.0.1:55432/mcf"

    def fake_open_database(path: str | None, *, read_only: bool = False, ensure_schema: bool = True):
        calls["db_path"] = path
        calls["read_only"] = read_only
        return object()

    class FakeDaemon:
        logfile = temp_dir / "scraper_daemon.log"

        def __init__(self, db):
            self.db = db

        def start(self, **kwargs):
            calls["start_kwargs"] = kwargs
            return 9876

    monkeypatch.setattr(cli, "_open_database", fake_open_database)
    monkeypatch.setattr(cli, "ScraperDaemon", FakeDaemon)
    monkeypatch.setattr(
        cli,
        "resolve_preferred_database_value",
        lambda db_path, include_persisted=False: persisted_dsn,
    )

    result = runner.invoke(cli.app, ["daemon", "start", "--year", "2022"])

    assert result.exit_code == 0
    assert calls["db_path"] == persisted_dsn
    assert calls["read_only"] is True
    assert calls["start_kwargs"]["db_path"] == persisted_dsn
