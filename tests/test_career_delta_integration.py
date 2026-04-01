import random
from datetime import date

import pytest

import src.mcf.database as database_module
import src.mcf.embeddings.search_engine as search_engine_module
import src.mcf.market_stats as market_stats_module
from src.mcf.career_delta import (
    CareerDeltaDependencies,
    CareerDeltaEngine,
    CareerDeltaRequest,
    ComputeBudget,
    MarketPosition,
    ScenarioType,
    generate_pivot_scenarios,
    generate_skill_scenarios,
    rank_and_filter_scenarios,
    summarize_market_position,
)
from src.mcf.career_delta_retrieval import SearchEngineCareerDeltaProvider
from src.mcf.embeddings import SemanticSearchEngine
from src.mcf.industry_taxonomy import normalize_title_family
from src.mcf.market_stats import MarketStatsCache
from src.mcf.models import Category

from . import factories
from .factories import generate_metadata, generate_test_job


class _FrozenDate(date):
    @classmethod
    def today(cls) -> "_FrozenDate":
        return cls(2026, 3, 18)


@pytest.fixture(autouse=True)
def _freeze_integration_calendar(monkeypatch):
    monkeypatch.setattr(factories, "date", _FrozenDate)
    monkeypatch.setattr(database_module, "date", _FrozenDate)
    monkeypatch.setattr(market_stats_module, "date", _FrozenDate)
    monkeypatch.setattr(search_engine_module, "date", _FrozenDate)


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
    random_state = random.getstate()
    random.seed(f"{title}|{company_name}|{','.join(skills)}|{','.join(categories)}")
    try:
        job = generate_test_job(
            title=title,
            company_name=company_name,
            skills=skills,
            salary_min=salary_min,
            salary_max=salary_max,
        )
    finally:
        random.setstate(random_state)
    job.description = (
        f"{title} role at {company_name}. Core skills: {', '.join(skills)}. Categories: {', '.join(categories)}."
    )
    job.categories = [Category(category=category) for category in categories]
    job.metadata = generate_metadata(posted_days_ago=posted_days_ago)
    db.upsert_job(job)


def _build_engine(
    temp_dir,
    db,
    *,
    minimum_pool_size: int = 24,
) -> tuple[CareerDeltaEngine, SearchEngineCareerDeltaProvider, MarketStatsCache]:
    search_engine = SemanticSearchEngine(
        db_path=str(db.db_path),
        index_dir=temp_dir / "missing-indexes",
    )
    provider = SearchEngineCareerDeltaProvider(search_engine, minimum_pool_size=minimum_pool_size)
    market_stats = MarketStatsCache(db, months=3)
    engine = CareerDeltaEngine(
        CareerDeltaDependencies(
            taxonomy=_TaxonomyAdapter(),
            market_stats=market_stats,
            search_scoring=provider,
        ),
        budget=ComputeBudget(max_wall_time_ms=5000, max_scenarios_evaluated=20),
        clock=lambda: 0.0,
    )
    return engine, provider, market_stats


def _scenario_with_added_skill(response, skill: str):
    return next(
        summary
        for summary in response.summaries
        if summary.scenario_type == ScenarioType.SKILL_ADDITION
        and summary.change is not None
        and skill in summary.change.added_skills
    )


def _scenario_with_replacement(response, from_skill: str, to_skill: str):
    return next(
        summary
        for summary in response.summaries
        if summary.scenario_type == ScenarioType.SKILL_SUBSTITUTION
        and summary.change is not None
        and any(
            replacement.from_skill == from_skill and replacement.to_skill == to_skill
            for replacement in summary.change.replaced_skills
        )
    )


def test_engine_integration_generates_skill_addition_from_controlled_market(temp_dir, empty_db):
    for idx, title in enumerate(
        (
            "Platform Engineer",
            "Platform Engineer",
            "Data Engineer",
            "DevOps Engineer",
            "Analytics Engineer",
        )
    ):
        skills = ["Python", "SQL", "Docker"]
        if idx < 4:
            skills.append("Kubernetes")
        _insert_job(
            empty_db,
            title=title,
            company_name="Infra Labs",
            skills=skills,
            categories=["Information Technology"],
            posted_days_ago=idx + 1,
            salary_min=9000 + idx * 400,
            salary_max=12000 + idx * 400,
        )

    engine, _, _ = _build_engine(temp_dir, empty_db)
    response = engine.analyze(
        CareerDeltaRequest(
            profile_text="Senior data engineer with Python SQL and Docker experience building pipelines.",
            current_title="Data Engineer",
            current_skills=("Python", "SQL", "Docker"),
            target_salary_min=130000,
            limit=6,
        )
    )

    addition = _scenario_with_added_skill(response, "Kubernetes")

    assert response.degraded is True
    assert response.baseline is not None
    assert addition.change is not None
    assert addition.change.added_skills == ("Kubernetes",)
    assert addition.signals[0].supporting_jobs >= 2


def test_engine_integration_generates_skill_substitution_from_market_momentum(temp_dir, empty_db, monkeypatch):
    for posted_days_ago in (10, 35, 55):
        _insert_job(
            empty_db,
            title="Data Engineer",
            company_name="Legacy Systems",
            skills=["Python", "SQL", "Hadoop"],
            categories=["Information Technology"],
            posted_days_ago=posted_days_ago,
            salary_min=7000,
            salary_max=9000,
        )
    for idx in range(4):
        _insert_job(
            empty_db,
            title="Data Engineer",
            company_name="Modern Data",
            skills=["Python", "SQL", "Spark"],
            categories=["Data Science"],
            posted_days_ago=idx + 1,
            salary_min=11000 + idx * 500,
            salary_max=14000 + idx * 500,
        )

    engine, provider, _ = _build_engine(temp_dir, empty_db)
    monkeypatch.setattr(
        provider,
        "get_related_skills",
        lambda skill, k=10: (
            ({"skill": "Spark", "similarity": 0.91, "same_cluster": True},) if skill == "Hadoop" else ()
        ),
    )

    response = engine.analyze(
        CareerDeltaRequest(
            profile_text="Senior data engineer with Python SQL experience building ETL systems.",
            current_title="Data Engineer",
            current_skills=("Python", "SQL", "Hadoop"),
            target_salary_min=150000,
            limit=6,
        )
    )

    substitution = _scenario_with_replacement(response, "Hadoop", "Spark")

    assert substitution.change is not None
    assert substitution.change.removed_skills == ("Hadoop",)
    assert substitution.expected_salary_delta_pct is not None
    assert substitution.expected_salary_delta_pct > 0
    assert substitution.signals[0].market_job_count >= 4


def test_engine_integration_generates_same_role_industry_pivot(temp_dir, empty_db):
    for idx in range(2):
        _insert_job(
            empty_db,
            title="Data Scientist",
            company_name="Bank Alpha",
            skills=["Python", "SQL", "Machine Learning"],
            categories=["Banking"],
            posted_days_ago=35 + idx,
            salary_min=8500,
            salary_max=10500,
        )
    for idx in range(5):
        _insert_job(
            empty_db,
            title="Data Scientist",
            company_name="AI Labs",
            skills=["Python", "SQL", "Machine Learning"],
            categories=["Data Science"],
            posted_days_ago=idx + 1,
            salary_min=12000 + idx * 400,
            salary_max=15000 + idx * 400,
        )

    engine, _, _ = _build_engine(temp_dir, empty_db)
    response = engine.analyze(
        CareerDeltaRequest(
            profile_text="Senior data scientist with Python SQL and machine learning experience.",
            current_title="Data Scientist",
            current_categories=("Banking",),
            current_skills=("Python", "SQL", "Machine Learning"),
            target_salary_min=150000,
            limit=6,
        )
    )

    pivot = next(
        summary for summary in response.summaries if summary.scenario_type == ScenarioType.SAME_ROLE_INDUSTRY_PIVOT
    )

    assert pivot.change is not None
    assert pivot.change.source_title_family == "data-scientist"
    assert pivot.change.target_title_family == "data-scientist"
    assert pivot.change.source_industry == "financial_services/banking"
    assert pivot.change.target_industry == "technology/data_and_ai"


def test_engine_integration_generates_bounded_adjacent_role_industry_pivot(temp_dir, empty_db):
    for idx in range(2):
        _insert_job(
            empty_db,
            title="Data Scientist",
            company_name="Product Cloud",
            skills=["Python", "SQL", "Machine Learning"],
            categories=["Information Technology"],
            posted_days_ago=40 + idx,
            salary_min=9000,
            salary_max=11000,
        )
    for idx in range(4):
        _insert_job(
            empty_db,
            title="Data Analyst",
            company_name="Insight AI",
            skills=["Python", "SQL", "Machine Learning"],
            categories=["Data Science"],
            posted_days_ago=idx + 1,
            salary_min=11800 + idx * 300,
            salary_max=14200 + idx * 300,
        )

    engine, _, _ = _build_engine(temp_dir, empty_db)
    response = engine.analyze(
        CareerDeltaRequest(
            profile_text="Data scientist with Python SQL and machine learning experience.",
            current_title="Data Scientist",
            current_categories=("Information Technology",),
            current_skills=("Python", "SQL", "Machine Learning"),
            limit=6,
        )
    )

    pivot = next(
        summary for summary in response.summaries if summary.scenario_type == ScenarioType.ADJACENT_ROLE_INDUSTRY_PIVOT
    )

    assert pivot.change is not None
    assert pivot.change.source_title_family == "data-scientist"
    assert pivot.change.target_title_family == "data-analyst"
    assert pivot.change.source_industry == "technology/software_and_platforms"
    assert pivot.change.target_industry == "technology/data_and_ai"
    assert pivot.target_title == "Data Analyst"


def test_engine_integration_reuse_parity_matches_naive_reruns(temp_dir, empty_db):
    for idx, (title, company_name, categories, skills) in enumerate(
        (
            ("Data Engineer", "Data Infra", ["Information Technology"], ["Python", "SQL", "Docker"]),
            ("Data Engineer", "Data Infra", ["Information Technology"], ["Python", "SQL", "Docker"]),
            ("Platform Engineer", "Data Infra", ["Information Technology"], ["Python", "SQL", "Docker", "Kubernetes"]),
            ("Platform Engineer", "Data Infra", ["Information Technology"], ["Python", "SQL", "Docker", "Kubernetes"]),
            ("Data Scientist", "AI Labs", ["Data Science"], ["Python", "SQL", "Machine Learning"]),
            ("Data Scientist", "AI Labs", ["Data Science"], ["Python", "SQL", "Machine Learning"]),
        )
    ):
        _insert_job(
            empty_db,
            title=title,
            company_name=company_name,
            skills=skills,
            categories=categories,
            posted_days_ago=idx + 1,
            salary_min=9500 + idx * 350,
            salary_max=12500 + idx * 350,
        )

    request = CareerDeltaRequest(
        profile_text="Data engineer with Python SQL and Docker experience building pipelines.",
        current_title="Data Engineer",
        current_categories=("Information Technology",),
        current_skills=("Python", "SQL", "Docker"),
        limit=6,
    )
    engine, provider, market_stats = _build_engine(temp_dir, empty_db)

    response = engine.analyze(request)

    baseline_pool = provider.build_candidate_pool(request)
    market_snapshot = market_stats.get_market_snapshot(request)
    baseline = summarize_market_position(
        baseline_pool,
        market_snapshot=market_snapshot,
        target_salary_min=request.target_salary_min,
    )

    scenario_pool = provider.build_candidate_pool(request)
    skill_summaries = generate_skill_scenarios(
        request,
        scenario_pool,
        baseline,
        market_stats=market_stats,
        search_scoring=provider,
    )
    pivot_summaries, filtered_scenarios = generate_pivot_scenarios(
        request,
        scenario_pool,
        baseline,
        market_snapshot=market_snapshot,
        market_stats=market_stats,
    )
    ranked_summaries, ranking_filtered, budget_degraded = rank_and_filter_scenarios(
        skill_summaries + pivot_summaries,
        baseline=baseline,
        request=request,
        budget=engine.budget,
        started_at=0.0,
        clock=lambda: 0.0,
    )

    assert budget_degraded is False
    assert [summary.scenario_id for summary in response.summaries] == [
        summary.scenario_id for summary in ranked_summaries
    ]
    assert [(scenario.scenario_type, scenario.reason_code) for scenario in response.filtered_scenarios] == [
        (scenario.scenario_type, scenario.reason_code) for scenario in tuple(filtered_scenarios + ranking_filtered)[:3]
    ]


def test_engine_integration_reports_thin_market_on_sparse_degraded_market(temp_dir, empty_db):
    _insert_job(
        empty_db,
        title="Data Engineer",
        company_name="Solo Data",
        skills=["Python", "SQL"],
        categories=["Information Technology"],
        posted_days_ago=2,
        salary_min=8000,
        salary_max=10000,
    )

    engine, _, _ = _build_engine(temp_dir, empty_db)
    response = engine.analyze(
        CareerDeltaRequest(
            profile_text="Data engineer with Python SQL experience.",
            current_title="Data Engineer",
            current_skills=("Python", "SQL"),
            limit=6,
        )
    )

    assert response.degraded is True
    assert response.thin_market is True
    assert response.baseline is not None
    assert response.baseline.position == MarketPosition.THIN
    assert response.baseline.total_candidates == 1
