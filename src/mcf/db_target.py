"""
Database target parsing and compatibility helpers.

This keeps the existing SQLite path based CLI/API surface working while
allowing PostgreSQL DSNs and environment-driven selection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

POSTGRES_SCHEMES = ("postgres://", "postgresql://")
DEFAULT_DATABASE_TARGET_FILE = Path("data/default_db_target.txt")


@dataclass(frozen=True)
class DatabaseTarget:
    """Resolved database target for SQLite or PostgreSQL."""

    kind: str
    value: str

    @property
    def is_sqlite(self) -> bool:
        return self.kind == "sqlite"

    @property
    def is_postgres(self) -> bool:
        return self.kind == "postgres"

    @property
    def sqlite_path(self) -> Path:
        if not self.is_sqlite:
            raise ValueError("sqlite_path requested for non-SQLite target")
        return Path(self.value)

    @property
    def dsn(self) -> str:
        if not self.is_postgres:
            raise ValueError("dsn requested for non-Postgres target")
        return self.value


def is_postgres_dsn(value: str | None) -> bool:
    """Return True when a string looks like a PostgreSQL DSN."""
    if not value:
        return False
    lowered = value.strip().lower()
    return lowered.startswith(POSTGRES_SCHEMES)


def resolve_database_value_from_env() -> str | None:
    """Resolve a database target string from the supported environment variables."""
    for env_var in ("DATABASE_URL", "MCF_DATABASE_URL", "MCF_DB_PATH"):
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def read_persisted_database_target(path: str | Path = DEFAULT_DATABASE_TARGET_FILE) -> str | None:
    """Read a persisted default database target, if one has been configured."""
    target_path = Path(path)
    if not target_path.exists():
        return None
    value = target_path.read_text().strip()
    return value or None


def write_persisted_database_target(value: str, path: str | Path = DEFAULT_DATABASE_TARGET_FILE) -> Path:
    """Persist a default database target for commands that opt into it."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(value.strip() + "\n")
    return target_path


def resolve_database_value(explicit_value: str | None = None) -> str:
    """
    Resolve a database target string from explicit args or environment.

    Precedence:
    1. Explicit value
    2. DATABASE_URL
    3. MCF_DATABASE_URL
    4. MCF_DB_PATH
    5. data/mcf_jobs.db
    """
    return resolve_preferred_database_value(explicit_value)


def resolve_preferred_database_value(
    explicit_value: str | None = None,
    *,
    include_persisted: bool = False,
    persisted_path: str | Path = DEFAULT_DATABASE_TARGET_FILE,
) -> str:
    """
    Resolve a database target string from explicit args, env, and optional persisted state.

    Precedence:
    1. Explicit value
    2. DATABASE_URL
    3. MCF_DATABASE_URL
    4. MCF_DB_PATH
    5. Persisted target file when enabled
    6. data/mcf_jobs.db
    """
    if explicit_value:
        return explicit_value

    env_value = resolve_database_value_from_env()
    if env_value:
        return env_value

    if include_persisted:
        persisted_value = read_persisted_database_target(persisted_path)
        if persisted_value:
            return persisted_value

    return "data/mcf_jobs.db"


def resolve_database_target(explicit_value: str | None = None) -> DatabaseTarget:
    """Resolve the effective database target."""
    value = resolve_database_value(explicit_value).strip()
    if is_postgres_dsn(value):
        return DatabaseTarget(kind="postgres", value=value)
    return DatabaseTarget(kind="sqlite", value=value)


def resolve_dual_write_target() -> DatabaseTarget | None:
    """Resolve optional dual-write secondary target from environment."""
    for env_var in ("MCF_DUAL_WRITE_DATABASE_URL", "MCF_DUAL_WRITE_DB_PATH"):
        value = os.environ.get(env_var)
        if value:
            return resolve_database_target(value)
    return None
