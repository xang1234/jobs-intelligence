from pathlib import Path
from unittest.mock import patch

from src.mcf.database import MCFDatabase
from src.mcf.embeddings import JobResult, SearchRequest, SearchResponse, SemanticSearchEngine

from .factories import generate_metadata, generate_test_job, posted_days_ago_for_month_offset


def _insert_job(
    db: MCFDatabase,
    *,
    title: str,
    company_name: str,
    skills: list[str],
    posted_days_ago: int,
    salary_min: int,
    salary_max: int,
    employment_type: str = "Full Time",
    region: str = "Central",
) -> None:
    job = generate_test_job(
        title=title,
        company_name=company_name,
        skills=skills,
        salary_min=salary_min,
        salary_max=salary_max,
        employment_type=employment_type,
    )
    job.metadata = generate_metadata(posted_days_ago=posted_days_ago)
    if job.address:
        job.address.region = region
    db.upsert_job(job)


def test_skill_trends_bucket_by_month_and_compute_momentum(empty_db: MCFDatabase):
    _insert_job(
        empty_db,
        title="Platform Engineer",
        company_name="Alpha",
        skills=["Python", "SQL"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=20),
        salary_min=9000,
        salary_max=12000,
    )
    _insert_job(
        empty_db,
        title="Analytics Engineer",
        company_name="Beta",
        skills=["Python", "Airflow"],
        posted_days_ago=posted_days_ago_for_month_offset(1, day=15),
        salary_min=8500,
        salary_max=11500,
    )
    _insert_job(
        empty_db,
        title="Database Engineer",
        company_name="Gamma",
        skills=["SQL", "PostgreSQL"],
        posted_days_ago=posted_days_ago_for_month_offset(3, day=15),
        salary_min=8000,
        salary_max=10000,
    )

    trends = empty_db.get_skill_trends(["Python"], months=3)
    series = trends[0]["series"]

    assert [point["job_count"] for point in series] == [0, 1, 1]
    assert series[-1]["median_salary_annual"] == 126000
    assert series[-1]["momentum"] == 100.0


def test_role_trend_uses_annual_salary_median(empty_db: MCFDatabase):
    _insert_job(
        empty_db,
        title="Data Scientist",
        company_name="Alpha",
        skills=["Python", "Machine Learning"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=20),
        salary_min=10000,
        salary_max=14000,
    )
    _insert_job(
        empty_db,
        title="Senior Data Scientist",
        company_name="Beta",
        skills=["Python", "SQL"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=10),
        salary_min=12000,
        salary_max=16000,
    )

    trend = empty_db.get_role_trend("data scientist", months=3)
    latest = trend["latest"]

    assert latest["job_count"] == 2
    assert latest["median_salary_annual"] == 156000
    assert latest["market_share"] == 100.0


def test_company_trend_top_skills_include_cluster_id(empty_db: MCFDatabase):
    _insert_job(
        empty_db,
        title="Data Scientist",
        company_name="Insight Labs",
        skills=["Python", "SQL"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=20),
        salary_min=10000,
        salary_max=14000,
    )

    trend = empty_db.get_company_trend("Insight Labs", months=3)

    assert trend["top_skills_by_month"]
    latest_skills = next(item for item in trend["top_skills_by_month"] if item["skills"])
    assert latest_skills["skills"][0]["cluster_id"] is None


def test_overview_avoids_nested_trend_queries(empty_db: MCFDatabase):
    for idx in range(12):
        _insert_job(
            empty_db,
            title=f"Engineer {idx}",
            company_name=f"Company {idx % 3}",
            skills=["Python", "SQL", "Machine Learning"],
            posted_days_ago=idx * 2,
            salary_min=9000 + idx * 100,
            salary_max=12000 + idx * 100,
        )

    with (
        patch.object(empty_db, "get_skill_trends", side_effect=AssertionError("should not be called")),
        patch.object(
            empty_db,
            "get_company_trend",
            side_effect=AssertionError("should not be called"),
        ),
    ):
        overview = empty_db.get_overview(months=3)

    assert overview["headline_metrics"]["total_jobs"] == 12
    assert overview["rising_skills"]
    assert overview["rising_companies"]


def test_search_results_include_explanations(temp_dir: Path, empty_db: MCFDatabase):
    _insert_job(
        empty_db,
        title="Machine Learning Engineer",
        company_name="Alpha",
        skills=["Python", "TensorFlow", "SQL"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=18),
        salary_min=11000,
        salary_max=15000,
    )

    engine = SemanticSearchEngine(
        db_path=str(empty_db.db_path),
        index_dir=temp_dir / "missing-indexes",
    )

    response = engine.search(SearchRequest(query="python machine learning", limit=5))

    assert response.results
    explanation = response.results[0].explanations
    assert explanation is not None
    assert "Python" in response.results[0].matched_skills
    assert explanation.bm25_score is not None
    assert explanation.freshness_score is not None


def test_profile_match_returns_fit_breakdown(temp_dir: Path, empty_db: MCFDatabase):
    _insert_job(
        empty_db,
        title="Senior Data Scientist",
        company_name="Insight Labs",
        skills=["Python", "SQL", "Machine Learning"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=20),
        salary_min=12000,
        salary_max=16000,
    )
    _insert_job(
        empty_db,
        title="Backend Engineer",
        company_name="Core Systems",
        skills=["Java", "Spring", "SQL"],
        posted_days_ago=posted_days_ago_for_month_offset(0, day=12),
        salary_min=9000,
        salary_max=12000,
    )

    engine = SemanticSearchEngine(
        db_path=str(empty_db.db_path),
        index_dir=temp_dir / "missing-indexes",
    )

    result = engine.match_profile(
        profile_text=("Senior analytics leader with Python, SQL, machine learning, and experimentation experience."),
        target_titles=["Data Scientist"],
        salary_expectation_annual=180000,
        limit=5,
    )

    assert {"Python", "SQL", "Machine Learning"}.issubset(set(result["extracted_skills"]))
    assert result["results"]
    top = result["results"][0]
    assert top.title == "Senior Data Scientist"
    assert top.explanations is not None
    assert top.explanations.skill_overlap_score is not None
    assert top.explanations.overall_fit is not None
    assert "Python" in top.matched_skills


def test_profile_match_degraded_mode_uses_fallback_relevance(temp_dir: Path, empty_db: MCFDatabase):
    higher_relevance = generate_test_job(
        title="Analytics Specialist",
        company_name="Signal Corp",
        skills=["Java"],
    )
    lower_relevance = generate_test_job(
        title="Analytics Specialist",
        company_name="Signal Corp",
        skills=["Java"],
    )
    empty_db.upsert_job(higher_relevance)
    empty_db.upsert_job(lower_relevance)

    engine = SemanticSearchEngine(
        db_path=str(empty_db.db_path),
        index_dir=temp_dir / "missing-indexes",
    )
    engine.load()

    fallback = SearchResponse(
        results=[
            JobResult(
                uuid=lower_relevance.uuid,
                title=lower_relevance.title,
                company_name=lower_relevance.company_name,
                description=lower_relevance.description_text,
                similarity_score=0.2,
            ),
            JobResult(
                uuid=higher_relevance.uuid,
                title=higher_relevance.title,
                company_name=higher_relevance.company_name,
                description=higher_relevance.description_text,
                similarity_score=0.9,
            ),
        ],
        total_candidates=2,
        degraded=True,
    )

    with patch.object(engine, "_keyword_fallback_search", return_value=fallback):
        result = engine.match_profile(
            profile_text="Generalist profile without explicit tracked skills.",
            limit=2,
        )

    assert result["results"][0].uuid == higher_relevance.uuid
    assert result["results"][0].bm25_score == 0.9
