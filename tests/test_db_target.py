from datetime import date

from src.mcf.db_target import resolve_database_target, resolve_preferred_database_value
from src.mcf.hosted_slice import HostedSlicePolicy


def test_resolve_database_target_detects_sqlite_path():
    target = resolve_database_target("data/mcf_jobs.db")
    assert target.is_sqlite is True
    assert target.is_postgres is False


def test_resolve_database_target_detects_postgres_dsn():
    target = resolve_database_target("postgresql://user:pass@localhost:5432/mcf")
    assert target.is_postgres is True
    assert target.is_sqlite is False


def test_resolve_preferred_database_value_precedence(monkeypatch, temp_dir):
    persisted_path = temp_dir / "default_db_target.txt"
    persisted_path.write_text("postgresql://persisted@localhost:5432/mcf\n")

    monkeypatch.setenv("DATABASE_URL", "postgresql://env@localhost:5432/mcf")
    monkeypatch.setenv("MCF_DATABASE_URL", "postgresql://second@localhost:5432/mcf")
    monkeypatch.setenv("MCF_DB_PATH", "data/from-env.db")

    assert (
        resolve_preferred_database_value(
            "postgresql://explicit@localhost:5432/mcf",
            include_persisted=True,
            persisted_path=persisted_path,
        )
        == "postgresql://explicit@localhost:5432/mcf"
    )
    assert (
        resolve_preferred_database_value(
            None,
            include_persisted=True,
            persisted_path=persisted_path,
        )
        == "postgresql://env@localhost:5432/mcf"
    )

    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.delenv("MCF_DATABASE_URL")
    monkeypatch.delenv("MCF_DB_PATH")

    assert (
        resolve_preferred_database_value(
            None,
            include_persisted=True,
            persisted_path=persisted_path,
        )
        == "postgresql://persisted@localhost:5432/mcf"
    )
    assert (
        resolve_preferred_database_value(
            None,
            include_persisted=False,
            persisted_path=persisted_path,
        )
        == "data/mcf_jobs.db"
    )


def test_hosted_slice_policy_uses_max_of_year_floor_and_rolling_cutoff():
    policy = HostedSlicePolicy(min_posted_date=date(2026, 1, 1), max_age_days=90)
    assert policy.cutoff_date(date(2026, 3, 24)) == date(2026, 1, 1)
    assert policy.cutoff_date(date(2026, 7, 1)) == date(2026, 4, 2)
