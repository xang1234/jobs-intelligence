"""Tests for API request/response models."""

from datetime import date

import pytest
from pydantic import ValidationError

from src.api.models import (
    CareerDeltaAnalysisRequest,
    CareerDeltaAnalysisResponse,
    CareerDeltaConfidence,
    CareerDeltaFilteredScenario,
    CareerDeltaScenarioDetail,
    CareerDeltaScenarioType,
    CompanySimilarity,
    CompanySimilarityRequest,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    JobResult,
    RelatedSkill,
    RelatedSkillsResponse,
    SearchRequest,
    SearchResponse,
    SimilarBatchRequest,
    SimilarBatchResponse,
    SimilarJobsRequest,
    SkillCloudItem,
    SkillCloudResponse,
    SkillSearchRequest,
    StatsResponse,
)
from src.mcf.career_delta import (
    BaselineMarketPosition,
    CareerDeltaResponse,
    FilteredScenario,
    MarketInsight,
    MarketPosition,
    PivotScenarioSignal,
    SalaryBand,
    ScenarioChange,
    ScenarioConfidence,
    ScenarioDetail,
    ScenarioScoreBreakdown,
    ScenarioSummary,
    ScenarioType,
    SkillReplacement,
    SkillScenarioSignal,
)
from src.mcf.embeddings.models import (
    CompanySimilarity as InternalCompanySimilarity,
)
from src.mcf.embeddings.models import (
    JobResult as InternalJobResult,
)
from src.mcf.embeddings.models import (
    SearchResponse as InternalSearchResponse,
)

# =============================================================================
# SearchRequest Tests
# =============================================================================


class TestSearchRequest:
    def test_minimal_valid(self):
        req = SearchRequest(query="test")
        assert req.query == "test"
        assert req.limit == 10
        assert req.alpha == 0.7
        assert req.freshness_weight == 0.1
        assert req.expand_query is True
        assert req.min_similarity == 0.3

    def test_full_request(self):
        req = SearchRequest(
            query="machine learning",
            limit=50,
            salary_min=8000,
            salary_max=15000,
            employment_type="Full Time",
            region="Central",
            company="Google",
            alpha=0.5,
            freshness_weight=0.2,
            expand_query=False,
            min_similarity=0.5,
        )
        assert req.salary_min == 8000
        assert req.salary_max == 15000
        assert req.company == "Google"

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_query_too_long(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="x" * 501)

    def test_limit_bounds(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=101)

    def test_salary_max_less_than_min(self):
        with pytest.raises(ValidationError, match="salary_max must be >= salary_min"):
            SearchRequest(query="test", salary_min=10000, salary_max=5000)

    def test_salary_max_equal_to_min(self):
        req = SearchRequest(query="test", salary_min=10000, salary_max=10000)
        assert req.salary_max == 10000

    def test_salary_max_without_min_ok(self):
        req = SearchRequest(query="test", salary_max=5000)
        assert req.salary_max == 5000

    def test_negative_salary_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", salary_min=-1)

    def test_alpha_out_of_range(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", alpha=1.5)
        with pytest.raises(ValidationError):
            SearchRequest(query="test", alpha=-0.1)

    def test_to_internal(self):
        req = SearchRequest(
            query="data engineer",
            limit=20,
            salary_min=8000,
            alpha=0.6,
        )
        internal = req.to_internal()
        assert internal.query == "data engineer"
        assert internal.limit == 20
        assert internal.salary_min == 8000
        assert internal.alpha == 0.6

    def test_json_schema_has_examples(self):
        schema = SearchRequest.model_json_schema()
        assert "examples" in schema


# =============================================================================
# SimilarJobsRequest Tests
# =============================================================================


class TestSimilarJobsRequest:
    def test_valid(self):
        req = SimilarJobsRequest(job_uuid="abc-123")
        assert req.limit == 10
        assert req.exclude_same_company is False

    def test_to_internal(self):
        req = SimilarJobsRequest(job_uuid="abc-123", limit=5, exclude_same_company=True)
        internal = req.to_internal()
        assert internal.job_uuid == "abc-123"
        assert internal.limit == 5
        assert internal.exclude_same_company is True


# =============================================================================
# SimilarBatchRequest Tests
# =============================================================================


class TestSimilarBatchRequest:
    def test_valid(self):
        req = SimilarBatchRequest(job_uuids=["a", "b", "c"])
        assert req.limit_per_job == 5
        assert req.exclude_same_company is False

    def test_empty_uuids_rejected(self):
        with pytest.raises(ValidationError):
            SimilarBatchRequest(job_uuids=[])

    def test_too_many_uuids_rejected(self):
        with pytest.raises(ValidationError):
            SimilarBatchRequest(job_uuids=["x"] * 51)

    def test_max_uuids_accepted(self):
        req = SimilarBatchRequest(job_uuids=["x"] * 50)
        assert len(req.job_uuids) == 50


# =============================================================================
# SkillSearchRequest Tests
# =============================================================================


class TestSkillSearchRequest:
    def test_valid(self):
        req = SkillSearchRequest(skill="Python")
        assert req.limit == 20

    def test_empty_skill_rejected(self):
        with pytest.raises(ValidationError):
            SkillSearchRequest(skill="")

    def test_salary_validation(self):
        with pytest.raises(ValidationError, match="salary_max must be >= salary_min"):
            SkillSearchRequest(skill="Python", salary_min=10000, salary_max=5000)

    def test_to_internal(self):
        req = SkillSearchRequest(skill="Python", limit=30)
        internal = req.to_internal()
        assert internal.skill == "Python"
        assert internal.limit == 30
        assert internal.salary_min is None
        assert internal.salary_max is None
        assert internal.employment_type is None

    def test_to_internal_with_filters(self):
        req = SkillSearchRequest(
            skill="Python",
            limit=30,
            salary_min=8000,
            salary_max=15000,
            employment_type="Full Time",
        )
        internal = req.to_internal()
        assert internal.skill == "Python"
        assert internal.limit == 30
        assert internal.salary_min == 8000
        assert internal.salary_max == 15000
        assert internal.employment_type == "Full Time"


# =============================================================================
# CompanySimilarityRequest Tests
# =============================================================================


class TestCompanySimilarityRequest:
    def test_valid(self):
        req = CompanySimilarityRequest(company_name="Google")
        assert req.limit == 10

    def test_empty_company_rejected(self):
        with pytest.raises(ValidationError):
            CompanySimilarityRequest(company_name="")

    def test_to_internal(self):
        req = CompanySimilarityRequest(company_name="Google", limit=5)
        internal = req.to_internal()
        assert internal.company_name == "Google"
        assert internal.limit == 5


class TestCareerDeltaAnalysisRequest:
    def test_to_internal_maps_public_request_shape(self):
        req = CareerDeltaAnalysisRequest(
            profile_text="  Senior analyst with Python, SQL, and experimentation experience.  ",
            current_title="  Senior Data Analyst ",
            target_titles=[" Analytics Engineer ", "Data Scientist", "Analytics Engineer"],
            current_categories=["Information Technology"],
            current_skills=[" Python ", "SQL", "Python"],
            current_company=" Example Corp ",
            location=" Singapore ",
            target_salary_min=90000,
            max_scenarios=6,
            include_filtered=False,
            delta_types=[
                CareerDeltaScenarioType.SKILL_ADDITION,
                CareerDeltaScenarioType.TITLE_PIVOT,
                CareerDeltaScenarioType.SKILL_ADDITION,
            ],
        )

        internal = req.to_internal()

        assert internal.profile_text == "Senior analyst with Python, SQL, and experimentation experience."
        assert internal.current_title == "Senior Data Analyst"
        assert internal.target_titles == ("Analytics Engineer", "Data Scientist")
        assert internal.current_categories == ("Information Technology",)
        assert internal.current_skills == ("Python", "SQL")
        assert internal.current_company == "Example Corp"
        assert internal.location == "Singapore"
        assert internal.target_salary_min == 90000
        assert internal.limit == 6
        assert internal.include_filtered is False
        assert req.delta_types == [
            CareerDeltaScenarioType.SKILL_ADDITION,
            CareerDeltaScenarioType.TITLE_PIVOT,
        ]
        assert req.selected_delta_types() == (
            CareerDeltaScenarioType.SKILL_ADDITION,
            CareerDeltaScenarioType.TITLE_PIVOT,
        )

    def test_profile_text_rejects_whitespace_only_content(self):
        with pytest.raises(ValidationError, match="at least 20 non-whitespace characters"):
            CareerDeltaAnalysisRequest(profile_text=" " * 25)

    def test_target_salary_min_must_be_positive_if_provided(self):
        with pytest.raises(ValidationError):
            CareerDeltaAnalysisRequest(
                profile_text="Experienced engineer with Python, SQL, APIs, and cloud operations.",
                target_salary_min=0,
            )

    def test_delta_types_reject_unknown_values(self):
        with pytest.raises(ValidationError):
            CareerDeltaAnalysisRequest(
                profile_text="Experienced engineer with Python, SQL, APIs, and cloud operations.",
                delta_types=["baseline"],
            )

    def test_target_titles_are_capped(self):
        with pytest.raises(ValidationError):
            CareerDeltaAnalysisRequest(
                profile_text="Experienced engineer with Python, SQL, APIs, and cloud operations.",
                target_titles=[f"Title {i}" for i in range(9)],
            )


# =============================================================================
# Response Model Tests
# =============================================================================


class TestJobResult:
    def test_from_internal(self):
        internal = InternalJobResult(
            uuid="abc-123",
            title="ML Engineer",
            company_name="Google",
            description="Great role",
            salary_min=12000,
            salary_max=18000,
            employment_type="Full Time",
            skills="Python, TensorFlow",
            location="Singapore",
            posted_date=date(2024, 1, 15),
            job_url="https://example.com/job",
            similarity_score=0.923,
        )
        result = JobResult.from_internal(internal)
        assert result.uuid == "abc-123"
        assert result.title == "ML Engineer"
        assert result.similarity_score == 0.923
        assert result.posted_date == date(2024, 1, 15)
        # Extra API fields not set by from_internal
        assert result.bm25_score is None
        assert result.freshness_score is None

    def test_optional_fields(self):
        result = JobResult(
            uuid="x",
            title="Job",
            description="Desc",
            similarity_score=0.5,
        )
        assert result.company_name is None
        assert result.salary_min is None


class TestSearchResponse:
    def test_from_internal(self):
        internal = InternalSearchResponse(
            results=[
                InternalJobResult(
                    uuid="abc",
                    title="Engineer",
                    company_name="Co",
                    description="Desc",
                    similarity_score=0.9,
                )
            ],
            total_candidates=100,
            search_time_ms=42.5,
            query_expansion=["ml", "machine learning"],
            degraded=False,
            cache_hit=True,
        )
        resp = SearchResponse.from_internal(internal)
        assert len(resp.results) == 1
        assert resp.results[0].uuid == "abc"
        assert resp.total_candidates == 100
        assert resp.search_time_ms == 42.5
        assert resp.query_expansion == ["ml", "machine learning"]
        assert resp.cache_hit is True

    def test_defaults(self):
        resp = SearchResponse(results=[], total_candidates=0, search_time_ms=1.0)
        assert resp.degraded is False
        assert resp.cache_hit is False
        assert resp.query_expansion is None


class TestSimilarBatchResponse:
    def test_structure(self):
        resp = SimilarBatchResponse(
            results={
                "uuid-1": [
                    JobResult(
                        uuid="x",
                        title="Job",
                        description="D",
                        similarity_score=0.8,
                    )
                ],
                "uuid-2": [],
            },
            search_time_ms=123.4,
        )
        assert len(resp.results["uuid-1"]) == 1
        assert resp.results["uuid-2"] == []


class TestCompanySimilarity:
    def test_from_internal(self):
        internal = InternalCompanySimilarity(
            company_name="Google",
            similarity_score=0.85,
            job_count=42,
            avg_salary=15000,
            top_skills=["Python", "Java"],
        )
        result = CompanySimilarity.from_internal(internal)
        assert result.company_name == "Google"
        assert result.job_count == 42
        assert result.top_skills == ["Python", "Java"]


class TestStatsResponse:
    def test_valid(self):
        resp = StatsResponse(
            total_jobs=50000,
            jobs_with_embeddings=48000,
            embedding_coverage_pct=96.0,
            unique_skills=1200,
            unique_companies=3000,
            index_size_mb=125.5,
            model_version="all-MiniLM-L6-v2",
        )
        assert resp.embedding_coverage_pct == 96.0


class TestHealthResponse:
    def test_healthy(self):
        resp = HealthResponse(status="healthy", index_loaded=True, degraded=False)
        assert resp.status == "healthy"

    def test_degraded(self):
        resp = HealthResponse(status="degraded", index_loaded=False, degraded=True)
        assert resp.degraded is True


# =============================================================================
# Skill Feature Model Tests
# =============================================================================


class TestSkillCloudItem:
    def test_valid(self):
        item = SkillCloudItem(skill="Python", job_count=5000, cluster_id=1)
        assert item.skill == "Python"
        assert item.job_count == 5000
        assert item.cluster_id == 1

    def test_cluster_id_optional(self):
        item = SkillCloudItem(skill="Python", job_count=5000)
        assert item.cluster_id is None


class TestSkillCloudResponse:
    def test_valid(self):
        resp = SkillCloudResponse(
            items=[
                SkillCloudItem(skill="Python", job_count=5000, cluster_id=1),
                SkillCloudItem(skill="Java", job_count=3000),
            ],
            total_unique_skills=1200,
        )
        assert len(resp.items) == 2
        assert resp.total_unique_skills == 1200

    def test_empty_items(self):
        resp = SkillCloudResponse(items=[], total_unique_skills=0)
        assert resp.items == []


class TestRelatedSkill:
    def test_valid(self):
        skill = RelatedSkill(skill="Pandas", similarity=0.85, same_cluster=True)
        assert skill.skill == "Pandas"
        assert skill.similarity == 0.85
        assert skill.same_cluster is True


class TestRelatedSkillsResponse:
    def test_valid(self):
        resp = RelatedSkillsResponse(
            skill="Python",
            related=[
                RelatedSkill(skill="Pandas", similarity=0.85, same_cluster=True),
                RelatedSkill(skill="Django", similarity=0.65, same_cluster=False),
            ],
        )
        assert resp.skill == "Python"
        assert len(resp.related) == 2

    def test_empty_related(self):
        resp = RelatedSkillsResponse(skill="ObscureSkill", related=[])
        assert resp.related == []


# =============================================================================
# Error Model Tests
# =============================================================================


class TestErrorResponse:
    def test_structure(self):
        resp = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="salary_min must be positive",
                details={"field": "salary_min"},
            )
        )
        assert resp.error.code == "VALIDATION_ERROR"
        assert resp.error.details["field"] == "salary_min"

    def test_without_details(self):
        resp = ErrorResponse(error=ErrorDetail(code="NOT_FOUND", message="Job not found"))
        assert resp.error.details is None


class TestCareerDeltaResponseModels:
    def test_analysis_response_from_internal_maps_baseline_and_summaries(self):
        internal = CareerDeltaResponse(
            request=CareerDeltaAnalysisRequest(
                profile_text="Experienced engineer with Python, SQL, APIs, and cloud operations."
            ).to_internal(),
            baseline=BaselineMarketPosition(
                position=MarketPosition.COMPETITIVE,
                reachable_jobs=18,
                total_candidates=34,
                fit_median=0.61,
                fit_p90=0.82,
                salary_band=SalaryBand(min_annual=70000, median_annual=90000, max_annual=120000),
                top_industries=(MarketInsight(name="technology/data_and_ai", job_count=12, share_pct=35.3),),
                top_companies=(MarketInsight(name="Example Corp", job_count=4, share_pct=11.8),),
                extracted_skills=("Python", "SQL"),
                skill_coverage=0.57,
                top_skill_gaps=(MarketInsight(name="Kubernetes", job_count=5, share_pct=27.8),),
                notes=("Candidate pool is concentrated in platform roles.",),
                thin_market=False,
                degraded=False,
            ),
            summaries=(
                ScenarioSummary(
                    scenario_id="skill_addition:abc",
                    scenario_type=ScenarioType.SKILL_ADDITION,
                    title="Add Kubernetes",
                    summary="Add Kubernetes to unlock more platform-engineering roles.",
                    market_position=MarketPosition.COMPETITIVE,
                    confidence=ScenarioConfidence(
                        score=0.82,
                        evidence_coverage=0.63,
                        market_sample_size=24,
                        reasons=("repeated_gap_signal",),
                    ),
                    score_breakdown=ScenarioScoreBreakdown(
                        opportunity=0.7,
                        quality=0.6,
                        salary=0.4,
                        momentum=0.3,
                        diversity=0.2,
                        raw_score=0.68,
                        pivot_cost=0.0,
                        final_score=0.68,
                    ),
                    change=ScenarioChange(
                        added_skills=("Kubernetes",),
                        source_title_family="data-engineer",
                        target_title_family="platform-engineer",
                    ),
                    signals=(
                        SkillScenarioSignal(
                            skill="Kubernetes",
                            supporting_jobs=4,
                            supporting_share_pct=33.3,
                            market_job_count=28,
                            market_salary_annual_median=110000,
                            market_momentum=0.14,
                            salary_lift_pct=0.09,
                            similarity=0.72,
                            same_cluster=False,
                        ),
                    ),
                    target_title="Platform Engineer",
                    target_sector="technology/platform",
                    expected_salary_delta_pct=0.09,
                ),
            ),
            filtered_scenarios=(
                FilteredScenario(
                    scenario_id="title_pivot:def",
                    scenario_type=ScenarioType.TITLE_PIVOT,
                    reason_code="overlapping_scenario",
                    explanation="A materially overlapping scenario with lower pivot cost was kept instead.",
                    confidence=ScenarioConfidence(score=0.66, evidence_coverage=0.45, market_sample_size=18),
                    market_position=MarketPosition.STRETCH,
                ),
            ),
            degraded=False,
            thin_market=False,
        )

        response = CareerDeltaAnalysisResponse.from_internal(internal, analysis_time_ms=42.5)

        assert response.analysis_time_ms == 42.5
        assert response.baseline is not None
        assert response.baseline.position.value == "competitive"
        assert response.baseline.salary_band.median_annual == 90000
        assert response.scenarios[0].scenario_type.value == "skill_addition"
        assert response.scenarios[0].change is not None
        assert response.scenarios[0].change.added_skills == ["Kubernetes"]
        assert response.scenarios[0].signals[0].signal_type == "skill"
        assert response.filtered_scenarios[0].reason_code == "overlapping_scenario"
        assert response.filtered_scenarios[0].market_position.value == "stretch"

    def test_analysis_response_can_filter_to_requested_delta_types(self):
        internal = CareerDeltaResponse(
            request=CareerDeltaAnalysisRequest(
                profile_text="Experienced engineer with Python, SQL, APIs, and cloud operations."
            ).to_internal(),
            summaries=(
                ScenarioSummary(
                    scenario_id="skill_addition:abc",
                    scenario_type=ScenarioType.SKILL_ADDITION,
                    title="Add Kubernetes",
                    summary="Add Kubernetes.",
                    market_position=MarketPosition.COMPETITIVE,
                    confidence=ScenarioConfidence(score=0.8, evidence_coverage=0.6, market_sample_size=20),
                ),
                ScenarioSummary(
                    scenario_id="title_pivot:abc",
                    scenario_type=ScenarioType.TITLE_PIVOT,
                    title="Pivot toward Platform Engineer",
                    summary="Shift toward platform roles.",
                    market_position=MarketPosition.STRETCH,
                    confidence=ScenarioConfidence(score=0.7, evidence_coverage=0.5, market_sample_size=18),
                ),
            ),
            filtered_scenarios=(
                FilteredScenario(
                    scenario_id="skill_substitution:abc",
                    scenario_type=ScenarioType.SKILL_SUBSTITUTION,
                    reason_code="low_signal",
                    explanation="Weak support.",
                    confidence=ScenarioConfidence(score=0.3, evidence_coverage=0.2, market_sample_size=3),
                ),
                FilteredScenario(
                    scenario_id="title_pivot:def",
                    scenario_type=ScenarioType.TITLE_PIVOT,
                    reason_code="overlapping_scenario",
                    explanation="Overlapped with a better pivot.",
                    confidence=ScenarioConfidence(score=0.6, evidence_coverage=0.4, market_sample_size=12),
                ),
            ),
        )

        response = CareerDeltaAnalysisResponse.from_internal(
            internal,
            allowed_delta_types=(CareerDeltaScenarioType.TITLE_PIVOT,),
        )

        assert [item.scenario_type for item in response.scenarios] == [CareerDeltaScenarioType.TITLE_PIVOT]
        assert [item.scenario_type for item in response.filtered_scenarios] == [CareerDeltaScenarioType.TITLE_PIVOT]

    def test_scenario_detail_from_internal_maps_nested_summary_and_pivot_signal(self):
        summary = ScenarioSummary(
            scenario_id="title_pivot:abc",
            scenario_type=ScenarioType.TITLE_PIVOT,
            title="Pivot toward Platform Engineer",
            summary="Shift toward platform engineering roles.",
            market_position=MarketPosition.STRETCH,
            confidence=ScenarioConfidence(score=0.74, evidence_coverage=0.52, market_sample_size=21),
        )
        internal = ScenarioDetail(
            scenario_id="title_pivot:abc",
            scenario_type=ScenarioType.TITLE_PIVOT,
            title="Pivot toward Platform Engineer",
            narrative="Platform roles show better demand and salary than the current lane.",
            market_position=MarketPosition.STRETCH,
            confidence=ScenarioConfidence(
                score=0.74,
                evidence_coverage=0.52,
                market_sample_size=21,
                reasons=("adjacent_role", "salary_lift"),
            ),
            score_breakdown=ScenarioScoreBreakdown(final_score=0.61, pivot_cost=0.18),
            summary=summary,
            change=ScenarioChange(
                source_title_family="data-engineer",
                target_title_family="platform-engineer",
                source_industry="technology/data_and_ai",
                target_industry="technology/platform",
                replaced_skills=(SkillReplacement(from_skill="Airflow", to_skill="Kubernetes"),),
            ),
            signals=(
                PivotScenarioSignal(
                    supporting_jobs=5,
                    supporting_share_pct=41.7,
                    target_title_family="platform-engineer",
                    target_industry="technology/platform",
                    title_distance="adjacent",
                    industry_distance=1,
                    fit_median=0.69,
                    market_job_count=26,
                    market_salary_annual_median=118000,
                    market_momentum=0.12,
                    salary_lift_pct=0.11,
                ),
            ),
            target_title="Platform Engineer",
            target_sector="technology/platform",
            evidence=("5 reachable jobs point to platform engineering.",),
            missing_skills=("Kubernetes",),
            search_queries=("platform engineer kubernetes",),
            thin_market=False,
            degraded=False,
        )

        detail = CareerDeltaScenarioDetail.from_internal(internal)

        assert detail.scenario_type is CareerDeltaScenarioType.TITLE_PIVOT
        assert detail.summary is not None
        assert detail.summary.scenario_id == "title_pivot:abc"
        assert detail.change is not None
        assert detail.change.replaced_skills[0].to_skill == "Kubernetes"
        assert detail.signals[0].signal_type == "pivot"
        assert detail.signals[0].target_title_family == "platform-engineer"
        assert detail.confidence.reasons == ["adjacent_role", "salary_lift"]

    def test_filtered_scenario_and_confidence_models_are_readable(self):
        filtered = CareerDeltaFilteredScenario.from_internal(
            FilteredScenario(
                scenario_id="skill_substitution:abc",
                scenario_type=ScenarioType.SKILL_SUBSTITUTION,
                reason_code="low_signal",
                explanation="Only one weak supporting job was available.",
                confidence=ScenarioConfidence(score=0.31, evidence_coverage=0.15, market_sample_size=3),
                market_position=MarketPosition.UNCLEAR,
            )
        )
        confidence = CareerDeltaConfidence.from_internal(
            ScenarioConfidence(score=0.91, evidence_coverage=0.8, market_sample_size=30, reasons=("broad_support",))
        )

        assert filtered.scenario_type is CareerDeltaScenarioType.SKILL_SUBSTITUTION
        assert filtered.reason_code == "low_signal"
        assert confidence.reasons == ["broad_support"]
