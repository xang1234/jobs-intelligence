from src.mcf.career_delta import CareerDeltaRequest
from src.mcf.market_stats import (
    MISSING_TREND_SERIES,
    NO_RECENT_JOBS,
    SPARSE_SALARY_DATA,
    MarketStatsCache,
)
from src.mcf.models import Category

from .factories import generate_metadata, generate_test_job, posted_days_ago_for_month_offset


def _insert_job(
    db,
    *,
    title: str,
    company_name: str,
    skills: list[str],
    categories: list[str],
    posted_days_ago: int,
    salary_min: int | None = 9000,
    salary_max: int | None = 12000,
) -> None:
    job = generate_test_job(
        title=title,
        company_name=company_name,
        skills=skills,
        salary_min=salary_min,
        salary_max=salary_max,
    )
    job.categories = [Category(category=category) for category in categories]
    job.metadata = generate_metadata(posted_days_ago=posted_days_ago)
    if salary_min is None and salary_max is None:
        job.salary = None
    db.upsert_job(job)


def test_skill_stats_align_with_existing_trend_semantics(empty_db):
    _insert_job(
        empty_db,
        title="Platform Engineer",
        company_name="Alpha",
        skills=["Python", "SQL"],
        categories=["Information Technology"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=20),
        salary_min=9000,
        salary_max=12000,
    )
    _insert_job(
        empty_db,
        title="Analytics Engineer",
        company_name="Beta",
        skills=["Python", "Airflow"],
        categories=["Information Technology"],
        posted_days_ago=posted_days_ago_for_month_offset(1, day=15),
        salary_min=8500,
        salary_max=11500,
    )

    cache = MarketStatsCache(empty_db, months=3)
    aggregate = cache.get_skill_stats("Python")
    latest = empty_db.get_skill_trends(["Python"], months=3)[0]["latest"]

    assert aggregate.job_count == latest["job_count"]
    assert aggregate.median_salary_annual == latest["median_salary_annual"]
    assert aggregate.momentum == latest["momentum"]


def test_title_family_stats_group_variants(empty_db):
    _insert_job(
        empty_db,
        title="Data Scientist",
        company_name="Alpha",
        skills=["Python"],
        categories=["Data Science"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=20),
        salary_min=10000,
        salary_max=14000,
    )
    _insert_job(
        empty_db,
        title="Senior Data Scientist",
        company_name="Beta",
        skills=["Python", "SQL"],
        categories=["Data Science"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=10),
        salary_min=12000,
        salary_max=16000,
    )

    cache = MarketStatsCache(empty_db, months=3)
    aggregate = cache.get_title_family_stats("Lead Data Scientist")

    assert aggregate.key == "data-scientist"
    assert aggregate.job_count == 2
    assert aggregate.median_salary_annual == 156000


def test_industry_stats_emit_sparse_salary_caveat(empty_db):
    _insert_job(
        empty_db,
        title="Platform Engineer",
        company_name="Alpha",
        skills=["AWS"],
        categories=["Information Technology"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=18),
        salary_min=None,
        salary_max=None,
    )

    cache = MarketStatsCache(empty_db, months=3)
    aggregate = cache.get_industry_stats("technology/software_and_platforms")

    assert aggregate.job_count == 1
    assert aggregate.median_salary_annual is None
    assert SPARSE_SALARY_DATA in aggregate.caveats


def test_missing_lookup_returns_explicit_caveats(empty_db):
    cache = MarketStatsCache(empty_db, months=3)
    aggregate = cache.get_skill_stats("Rust")

    assert aggregate.job_count == 0
    assert aggregate.momentum is None
    assert set(aggregate.caveats) == {NO_RECENT_JOBS, MISSING_TREND_SERIES, SPARSE_SALARY_DATA}


def test_cache_refreshes_after_ttl_and_invalidate(empty_db):
    fake_now = [1000.0]
    cache = MarketStatsCache(empty_db, months=3, ttl_seconds=10, clock=lambda: fake_now[0])

    _insert_job(
        empty_db,
        title="Platform Engineer",
        company_name="Alpha",
        skills=["Python"],
        categories=["Information Technology"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=20),
        salary_min=9000,
        salary_max=12000,
    )
    initial = cache.get_skill_stats("Python")
    assert initial.job_count == 1

    _insert_job(
        empty_db,
        title="Analytics Engineer",
        company_name="Beta",
        skills=["Python"],
        categories=["Information Technology"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=21),
        salary_min=10000,
        salary_max=13000,
    )

    before_expiry = cache.get_skill_stats("Python")
    assert before_expiry.job_count == 1

    fake_now[0] += 11
    after_expiry = cache.get_skill_stats("Python")
    assert after_expiry.job_count == 2

    cache.invalidate()
    refreshed = cache.get_skill_stats("Python")
    assert refreshed.job_count == 2


def test_market_snapshot_returns_request_relevant_aggregates(empty_db):
    _insert_job(
        empty_db,
        title="Product Manager",
        company_name="Alpha",
        skills=["SQL", "CRM"],
        categories=["Marketing"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=20),
        salary_min=10000,
        salary_max=13000,
    )

    cache = MarketStatsCache(empty_db, months=3)
    snapshot = cache.get_market_snapshot(
        CareerDeltaRequest(
            profile_text="Product manager with SQL and CRM experience",
            current_title="Lead Product Manager",
            current_categories=("Marketing",),
            current_skills=("SQL", "CRM"),
            target_titles=("Product Manager",),
        )
    )

    assert snapshot["current_title_family"].key == "product-manager"
    assert snapshot["skills"]["SQL"].job_count == 1
    assert snapshot["target_title_families"]["Product Manager"].key == "product-manager"


def test_market_stats_use_persisted_normalized_columns_for_aggregates(empty_db):
    _insert_job(
        empty_db,
        title="Lead Product Manager",
        company_name="Alpha",
        skills=["Python", "SQL"],
        categories=["Information Technology"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=20),
        salary_min=10000,
        salary_max=13000,
    )

    with empty_db._connection() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET title = '',
                categories = '',
                title_family = 'product-manager',
                industry_bucket = 'technology/software_and_platforms'
            """
        )

    cache = MarketStatsCache(empty_db, months=3)

    title_aggregate = cache.get_title_family_stats("Lead Product Manager")
    industry_aggregate = cache.get_industry_stats("technology/software_and_platforms")

    assert title_aggregate.job_count == 1
    assert industry_aggregate.job_count == 1


def test_market_snapshot_preserves_company_fallback_with_missing_persisted_bucket(empty_db):
    _insert_job(
        empty_db,
        title="Platform Engineer",
        company_name="Alpha",
        skills=["Python", "AWS"],
        categories=["Information Technology"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=20),
        salary_min=10000,
        salary_max=13000,
    )
    _insert_job(
        empty_db,
        title="Analytics Engineer",
        company_name="Alpha",
        skills=["SQL"],
        categories=["Information Technology"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=15),
        salary_min=9000,
        salary_max=12000,
    )
    _insert_job(
        empty_db,
        title="Site Reliability Engineer",
        company_name="Alpha",
        skills=["Kubernetes"],
        categories=["Information Technology"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=10),
        salary_min=11000,
        salary_max=14000,
    )

    with empty_db._connection() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET industry_bucket = NULL
            WHERE company_name = 'Alpha'
            """
        )

    cache = MarketStatsCache(empty_db, months=3)
    snapshot = cache.get_market_snapshot(
        CareerDeltaRequest(
            profile_text="Engineer at Alpha",
            current_company="Alpha",
            current_skills=(),
            target_titles=("Platform Engineer",),
        )
    )

    assert snapshot["current_industry"].key == "technology/software_and_platforms"
