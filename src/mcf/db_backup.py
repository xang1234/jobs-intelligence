"""
SQLite hot backup helpers.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class BackupMetadata:
    path: Path
    created_at: str
    size_bytes: int
    integrity_check: str
    jobs_count: int


def create_sqlite_hot_backup(
    source_path: str | Path, backup_dir: str | Path, *, prefix: str = "mcf_pre_postgres"
) -> Path:
    """Create a consistent hot backup using SQLite's native backup API."""
    source = Path(source_path)
    destination_dir = Path(backup_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    destination = destination_dir / f"{prefix}_{timestamp}.db"

    source_conn = sqlite3.connect(str(source))
    try:
        dest_conn = sqlite3.connect(str(destination))
        try:
            source_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        source_conn.close()

    return destination


def verify_sqlite_backup(backup_path: str | Path) -> BackupMetadata:
    """Run integrity and basic shape checks on a SQLite backup."""
    path = Path(backup_path)
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        jobs_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    finally:
        conn.close()

    stat = path.stat()
    return BackupMetadata(
        path=path,
        created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        size_bytes=stat.st_size,
        integrity_check=integrity,
        jobs_count=jobs_count,
    )
