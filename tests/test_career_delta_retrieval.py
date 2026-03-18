from src.mcf.career_delta import CareerDeltaDependencies, CareerDeltaEngine, CareerDeltaRequest, MarketPosition
from src.mcf.career_delta_retrieval import SearchEngineCareerDeltaProvider
from src.mcf.embeddings import SemanticSearchEngine
from src.mcf.industry_taxonomy import normalize_title_family
from src.mcf.market_stats import MarketStatsCache
from src.mcf.models import Category

from .factories import generate_metadata, generate_test_job


class _TaxonomyAdapter:
    def normalize_title_family(self, title: str):
        return normalize_title_family(title)


def _insert_job(
    db,
    *,
    title: str,
    company_name: str,
    skills: list[str],
    categories: list[str],
    posted_days_ago: int,
    salary_min: int,
    salary_max: int,
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
    db.upsert_job(job)


def test_provider_builds_reusable_candidate_pool(temp_dir, empty_db):
    _insert_job(
        empty_db,
        title="Senior Data Scientist",
        company_name="Insight Labs",
        skills=["Python", "SQL", "Machine Learning"],
        categories=["Data Science"],
        posted_days_ago=3,
        salary_min=12000,
        salary_max=16000,
    )
    _insert_job(
        empty_db,
        title="Analytics Engineer",
        company_name="Insight Labs",
        skills=["Python", "SQL", "Airflow"],
        categories=["Information Technology"],
        posted_days_ago=4,
        salary_min=10000,
        salary_max=14000,
    )

    engine = SemanticSearchEngine(
        db_path=str(empty_db.db_path),
        index_dir=temp_dir / "missing-indexes",
    )
    provider = SearchEngineCareerDeltaProvider(engine, minimum_pool_size=10)

    pool = provider.build_candidate_pool(
        CareerDeltaRequest(
            profile_text="Senior analytics leader with Python, SQL, machine learning, and experimentation experience.",
            target_titles=("Data Scientist",),
            target_salary_min=180000,
            limit=5,
        )
    )

    assert pool.degraded is True
    assert pool.total_candidates >= 1
    assert "Python" in pool.extracted_skills
    assert pool.candidates[0].title == "Senior Data Scientist"
    assert pool.candidates[0].industry_key.startswith("technology/")


def test_provider_keeps_pool_broad_even_with_target_titles(temp_dir, empty_db):
    _insert_job(
        empty_db,
        title="Senior Data Scientist",
        company_name="Insight Labs",
        skills=["Python", "SQL", "Machine Learning"],
        categories=["Data Science"],
        posted_days_ago=2,
        salary_min=12000,
        salary_max=16000,
    )
    _insert_job(
        empty_db,
        title="Analytics Engineer",
        company_name="Insight Labs",
        skills=["Python", "SQL", "Machine Learning"],
        categories=["Information Technology"],
        posted_days_ago=1,
        salary_min=11000,
        salary_max=15000,
    )

    engine = SemanticSearchEngine(
        db_path=str(empty_db.db_path),
        index_dir=temp_dir / "missing-indexes",
    )
    provider = SearchEngineCareerDeltaProvider(engine, minimum_pool_size=10)

    pool = provider.build_candidate_pool(
        CareerDeltaRequest(
            profile_text="Senior analytics leader with Python, SQL, machine learning experience.",
            target_titles=("Data Scientist",),
            limit=5,
        )
    )

    titles = {candidate.title: candidate.target_title_match for candidate in pool.candidates}
    assert "Senior Data Scientist" in titles
    assert "Analytics Engineer" in titles
    assert titles["Senior Data Scientist"] is True
    assert titles["Analytics Engineer"] is False


def test_engine_produces_baseline_from_candidate_pool(temp_dir, empty_db):
    for idx in range(6):
        _insert_job(
            empty_db,
            title="Senior Data Scientist" if idx < 3 else "Analytics Engineer",
            company_name="Insight Labs" if idx < 4 else "DataWorks",
            skills=["Python", "SQL", "Machine Learning"],
            categories=["Data Science" if idx < 3 else "Information Technology"],
            posted_days_ago=idx + 1,
            salary_min=11000 + idx * 300,
            salary_max=15000 + idx * 300,
        )

    engine = SemanticSearchEngine(
        db_path=str(empty_db.db_path),
        index_dir=temp_dir / "missing-indexes",
    )
    dependencies = CareerDeltaDependencies(
        taxonomy=_TaxonomyAdapter(),
        market_stats=MarketStatsCache(empty_db, months=3),
        search_scoring=SearchEngineCareerDeltaProvider(engine, minimum_pool_size=12),
    )

    response = CareerDeltaEngine(dependencies).analyze(
        CareerDeltaRequest(
            profile_text="Senior analytics leader with Python, SQL, machine learning, and experimentation experience.",
            current_company="Insight Labs",
            target_titles=("Data Scientist",),
            current_categories=("Unknown",),
            current_skills=("Python", "SQL"),
            target_salary_min=180000,
            limit=5,
        )
    )

    assert response.degraded is True
    assert response.candidate_pool is not None
    assert response.baseline is not None
    assert response.baseline.position in {MarketPosition.COMPETITIVE, MarketPosition.STRETCH, MarketPosition.THIN}
    assert response.baseline.total_candidates >= response.baseline.reachable_jobs
    assert response.baseline.extracted_skills
    assert response.baseline.top_companies
    assert response.baseline.top_industries
    assert response.baseline.salary_band.median_annual is not None


def test_market_snapshot_uses_current_company_for_industry_fallback(temp_dir, empty_db):
    for title in ("Platform Engineer", "Data Engineer", "Backend Engineer"):
        _insert_job(
            empty_db,
            title=title,
            company_name="Alpha",
            skills=["Python", "SQL"],
            categories=["Information Technology"],
            posted_days_ago=2,
            salary_min=9000,
            salary_max=12000,
        )

    cache = MarketStatsCache(empty_db, months=3)
    snapshot = cache.get_market_snapshot(
        CareerDeltaRequest(
            profile_text="Engineer profile",
            current_company="Alpha",
        )
    )

    assert snapshot["current_industry"].key == "technology/software_and_platforms"


def test_market_snapshot_prefers_explicit_categories_over_company_fallback(temp_dir, empty_db):
    for title in ("Platform Engineer", "Data Engineer", "Backend Engineer"):
        _insert_job(
            empty_db,
            title=title,
            company_name="Alpha",
            skills=["Python", "SQL"],
            categories=["Information Technology"],
            posted_days_ago=2,
            salary_min=9000,
            salary_max=12000,
        )

    cache = MarketStatsCache(empty_db, months=3)
    snapshot = cache.get_market_snapshot(
        CareerDeltaRequest(
            profile_text="Engineer profile",
            current_company="Alpha",
            current_categories=("Marketing",),
        )
    )

    assert snapshot["current_industry"].key == "commercial/marketing_and_brand"
