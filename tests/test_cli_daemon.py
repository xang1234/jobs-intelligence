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

    class FakeDB:
        def __init__(self, path: str, read_only: bool = False):
            calls["db_path"] = path
            calls["read_only"] = read_only

    class FakeDaemon:
        logfile = temp_dir / "scraper_daemon.log"

        def __init__(self, db):
            calls["daemon_db"] = db

        def start(self, **kwargs):
            calls["start_kwargs"] = kwargs
            return 12345

    monkeypatch.setattr(cli, "MCFDatabase", FakeDB)
    monkeypatch.setattr(cli, "ScraperDaemon", FakeDaemon)

    result = runner.invoke(
        cli.app,
        ["daemon", "start", "--year", "2022", "--db", str(db_path)],
    )

    assert result.exit_code == 0
    assert calls["db_path"] == str(db_path)
    assert calls["read_only"] is True
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

    class FakeDB:
        def __init__(
            self,
            path: str,
            read_only: bool = False,
            ensure_schema: bool = True,
        ):
            calls.append((path, read_only, ensure_schema))
            if ensure_schema:
                raise sqlite3.OperationalError("database is locked")

    class FakeDaemon:
        def __init__(self, db, **kwargs):
            self.db = db

        def run_worker(self, scraper_func):
            return None

    monkeypatch.setattr(cli, "MCFDatabase", FakeDB)
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

    class FakeDB:
        def __init__(self, path: str, read_only: bool = False):
            self.path = path
            self.read_only = read_only

    class FakeDaemon:
        def __init__(self, db):
            self.db = db

        def start(self, **kwargs):
            raise cli.DaemonError("Database is busy: another process is writing to it")

    monkeypatch.setattr(cli, "MCFDatabase", FakeDB)
    monkeypatch.setattr(cli, "ScraperDaemon", FakeDaemon)

    result = runner.invoke(
        cli.app,
        ["daemon", "start", "--year", "2022", "--db", str(db_path)],
    )

    assert result.exit_code == 1
    assert "Database is busy" in result.stdout
