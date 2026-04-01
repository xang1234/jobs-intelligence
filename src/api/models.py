"""
API request/response models for the MCF semantic search API.

Pydantic models providing:
- Input validation with field constraints
- Cross-field validation (e.g., salary_max >= salary_min)
- OpenAPI schema generation with examples
- Conversion to/from internal search engine dataclasses

These models sit at the API boundary. The search engine uses plain
dataclasses internally (src/mcf/embeddings/models.py), and this module
translates between the HTTP layer and the engine layer.
"""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from ..mcf.career_delta import (
    BaselineMarketPosition as InternalBaselineMarketPosition,
)
from ..mcf.career_delta import (
    CareerDeltaRequest as InternalCareerDeltaRequest,
)
from ..mcf.career_delta import (
    CareerDeltaResponse as InternalCareerDeltaResponse,
)
from ..mcf.career_delta import (
    FilteredScenario as InternalFilteredScenario,
)
from ..mcf.career_delta import (
    MarketInsight as InternalMarketInsight,
)
from ..mcf.career_delta import (
    PivotScenarioSignal as InternalPivotScenarioSignal,
)
from ..mcf.career_delta import (
    SalaryBand as InternalSalaryBand,
)
from ..mcf.career_delta import (
    ScenarioChange as InternalScenarioChange,
)
from ..mcf.career_delta import (
    ScenarioConfidence as InternalScenarioConfidence,
)
from ..mcf.career_delta import (
    ScenarioDetail as InternalScenarioDetail,
)
from ..mcf.career_delta import (
    ScenarioScoreBreakdown as InternalScenarioScoreBreakdown,
)
from ..mcf.career_delta import (
    ScenarioSummary as InternalScenarioSummary,
)
from ..mcf.career_delta import (
    SkillReplacement as InternalSkillReplacement,
)
from ..mcf.career_delta import (
    SkillScenarioSignal as InternalSkillScenarioSignal,
)
from ..mcf.embeddings.models import (
    CompanySimilarityRequest as InternalCompanySimilarityRequest,
)
from ..mcf.embeddings.models import (
    SearchRequest as InternalSearchRequest,
)
from ..mcf.embeddings.models import (
    SimilarJobsRequest as InternalSimilarJobsRequest,
)
from ..mcf.embeddings.models import (
    SkillSearchRequest as InternalSkillSearchRequest,
)

# =============================================================================
# Request Models
# =============================================================================


class CareerDeltaScenarioType(str, Enum):
    """Public scenario families supported by the career-delta API."""

    SAME_ROLE = "same_role"
    ADJACENT_ROLE = "adjacent_role"
    INDUSTRY_PIVOT = "industry_pivot"
    TITLE_PIVOT = "title_pivot"
    SAME_ROLE_INDUSTRY_PIVOT = "same_role_industry_pivot"
    ADJACENT_ROLE_INDUSTRY_PIVOT = "adjacent_role_industry_pivot"
    SKILL_ADDITION = "skill_addition"
    SKILL_SUBSTITUTION = "skill_substitution"


class CareerDeltaMarketPosition(str, Enum):
    """Stable market-position vocabulary exposed over HTTP."""

    LEADING = "leading"
    COMPETITIVE = "competitive"
    STRETCH = "stretch"
    THIN = "thin"
    UNCLEAR = "unclear"


def _clean_scalar(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _clean_string_list(values: list[str], *, label: str, max_item_length: int) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError(f"{label} cannot contain blank values")
        if len(normalized) > max_item_length:
            raise ValueError(f"{label} entries must be <= {max_item_length} characters")
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned


class SearchRequest(BaseModel):
    """Request for semantic job search."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")

    # SQL Filters
    salary_min: Optional[int] = Field(None, ge=0, description="Minimum salary filter")
    salary_max: Optional[int] = Field(None, ge=0, description="Maximum salary filter")
    employment_type: Optional[str] = Field(None, description="Employment type: 'Full Time', 'Part Time', 'Contract'")
    region: Optional[str] = Field(None, description="Region filter")
    company: Optional[str] = Field(None, description="Company name filter (partial match)")

    # Hybrid Search Tuning
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="Weight for semantic vs BM25 (0.7 = 70% semantic)")
    freshness_weight: float = Field(0.1, ge=0.0, le=1.0, description="Weight for recency boost")
    expand_query: bool = Field(True, description="Enable query expansion with synonyms")
    min_similarity: float = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity score threshold")

    @field_validator("salary_max")
    @classmethod
    def salary_max_gte_min(cls, v: Optional[int], info) -> Optional[int]:
        if v is not None and info.data.get("salary_min") is not None:
            if v < info.data["salary_min"]:
                raise ValueError("salary_max must be >= salary_min")
        return v

    def to_internal(self) -> InternalSearchRequest:
        """Convert to the internal search engine dataclass."""
        return InternalSearchRequest(
            query=self.query,
            salary_min=self.salary_min,
            salary_max=self.salary_max,
            employment_type=self.employment_type,
            company=self.company,
            region=self.region,
            limit=self.limit,
            min_similarity=self.min_similarity,
            alpha=self.alpha,
            expand_query=self.expand_query,
            freshness_weight=self.freshness_weight,
        )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "machine learning engineer",
                    "limit": 20,
                    "salary_min": 10000,
                    "alpha": 0.7,
                }
            ]
        }
    }


class SimilarJobsRequest(BaseModel):
    """Request for finding similar jobs."""

    job_uuid: str = Field(..., description="Source job UUID")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    exclude_same_company: bool = Field(False, description="Exclude jobs from the same company")
    freshness_weight: float = Field(0.1, ge=0.0, le=1.0, description="Weight for recency boost")

    def to_internal(self) -> InternalSimilarJobsRequest:
        """Convert to the internal search engine dataclass."""
        return InternalSimilarJobsRequest(
            job_uuid=self.job_uuid,
            limit=self.limit,
            exclude_same_company=self.exclude_same_company,
            freshness_weight=self.freshness_weight,
        )


class SimilarBatchRequest(BaseModel):
    """Request for batch similar jobs lookup."""

    job_uuids: list[str] = Field(..., min_length=1, max_length=50, description="List of job UUIDs (max 50)")
    limit_per_job: int = Field(5, ge=1, le=20, description="Results per job")
    exclude_same_company: bool = Field(False, description="Exclude jobs from the same company")


class SkillSearchRequest(BaseModel):
    """Request for skill-based job search."""

    skill: str = Field(..., min_length=1, max_length=100, description="Skill to search for")
    limit: int = Field(20, ge=1, le=100, description="Maximum results to return")

    # Optional SQL filters
    salary_min: Optional[int] = Field(None, ge=0)
    salary_max: Optional[int] = Field(None, ge=0)
    employment_type: Optional[str] = None

    @field_validator("salary_max")
    @classmethod
    def salary_max_gte_min(cls, v: Optional[int], info) -> Optional[int]:
        if v is not None and info.data.get("salary_min") is not None:
            if v < info.data["salary_min"]:
                raise ValueError("salary_max must be >= salary_min")
        return v

    def to_internal(self) -> InternalSkillSearchRequest:
        """Convert to the internal search engine dataclass."""
        return InternalSkillSearchRequest(
            skill=self.skill,
            limit=self.limit,
            salary_min=self.salary_min,
            salary_max=self.salary_max,
            employment_type=self.employment_type,
        )


class CompanySimilarityRequest(BaseModel):
    """Request for finding similar companies."""

    company_name: str = Field(..., min_length=1, max_length=200, description="Source company name")
    limit: int = Field(10, ge=1, le=50, description="Maximum results to return")

    def to_internal(self) -> InternalCompanySimilarityRequest:
        """Convert to the internal search engine dataclass."""
        return InternalCompanySimilarityRequest(
            company_name=self.company_name,
            limit=self.limit,
        )


class TrendFilters(BaseModel):
    """Reusable filter set for market-intelligence endpoints."""

    company: Optional[str] = None
    employment_type: Optional[str] = None
    region: Optional[str] = None


class SkillTrendRequest(TrendFilters):
    """Request for one or more skill time series."""

    skills: list[str] = Field(..., min_length=1, max_length=3)
    months: int = Field(12, ge=3, le=24)


class RoleTrendRequest(TrendFilters):
    """Request for a role/query trend series."""

    query: str = Field(..., min_length=1, max_length=200)
    months: int = Field(12, ge=3, le=24)


class ProfileMatchRequest(BaseModel):
    """Request for profile-to-job matching."""

    profile_text: str = Field(..., min_length=20, max_length=20000)
    target_titles: list[str] = Field(default_factory=list, max_length=10)
    salary_expectation_annual: Optional[int] = Field(None, ge=0)
    employment_type: Optional[str] = None
    region: Optional[str] = None
    limit: int = Field(20, ge=1, le=100)


class CareerDeltaAnalysisRequest(BaseModel):
    """Request payload for career-delta analysis."""

    profile_text: str = Field(..., min_length=20, max_length=20000)
    current_title: Optional[str] = Field(None, max_length=160)
    target_titles: list[str] = Field(default_factory=list, max_length=8)
    current_categories: list[str] = Field(default_factory=list, max_length=12)
    current_skills: list[str] = Field(default_factory=list, max_length=50)
    current_company: Optional[str] = Field(None, max_length=200)
    location: Optional[str] = Field(None, max_length=120)
    target_salary_min: Optional[int] = Field(None, gt=0)
    max_scenarios: int = Field(12, ge=1, le=12)
    include_filtered: bool = Field(
        True,
        description="Include withheld scenarios and explanations in the response.",
    )
    delta_types: list[CareerDeltaScenarioType] = Field(
        default_factory=list,
        max_length=8,
        description="Optional scenario families to keep when rendering results.",
    )

    @field_validator("profile_text")
    @classmethod
    def profile_text_must_have_content(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 20:
            raise ValueError("profile_text must contain at least 20 non-whitespace characters")
        return cleaned

    @field_validator("current_title", "current_company", "location")
    @classmethod
    def normalize_optional_scalars(cls, value: Optional[str]) -> Optional[str]:
        return _clean_scalar(value)

    @field_validator("target_titles")
    @classmethod
    def normalize_target_titles(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value, label="target_titles", max_item_length=160)

    @field_validator("current_categories")
    @classmethod
    def normalize_current_categories(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value, label="current_categories", max_item_length=120)

    @field_validator("current_skills")
    @classmethod
    def normalize_current_skills(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value, label="current_skills", max_item_length=80)

    @field_validator("delta_types")
    @classmethod
    def dedupe_delta_types(cls, value: list[CareerDeltaScenarioType]) -> list[CareerDeltaScenarioType]:
        deduped: list[CareerDeltaScenarioType] = []
        seen: set[str] = set()
        for item in value:
            if item.value in seen:
                continue
            seen.add(item.value)
            deduped.append(item)
        return deduped

    def to_internal(self) -> InternalCareerDeltaRequest:
        """Convert to the internal career-delta engine request."""
        return InternalCareerDeltaRequest(
            profile_text=self.profile_text,
            current_title=self.current_title,
            target_titles=tuple(self.target_titles),
            current_categories=tuple(self.current_categories),
            current_skills=tuple(self.current_skills),
            current_company=self.current_company,
            location=self.location,
            target_salary_min=self.target_salary_min,
            limit=self.max_scenarios,
            include_filtered=self.include_filtered,
        )

    def selected_delta_types(self) -> tuple[CareerDeltaScenarioType, ...]:
        """Normalized public delta-type filter for API-layer response shaping."""
        return tuple(self.delta_types)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "profile_text": "Senior data analyst with Python, SQL, and dashboarding experience.",
                    "current_title": "Senior Data Analyst",
                    "target_titles": ["Analytics Engineer", "Data Scientist"],
                    "current_skills": ["Python", "SQL", "Tableau"],
                    "current_company": "Example Corp",
                    "location": "Singapore",
                    "target_salary_min": 90000,
                    "max_scenarios": 6,
                    "include_filtered": True,
                    "delta_types": ["skill_addition", "title_pivot"],
                }
            ]
        }
    }


# =============================================================================
# Response Models
# =============================================================================


class JobResult(BaseModel):
    """A job in search results."""

    uuid: str
    title: str
    company_name: Optional[str] = None
    description: str = Field(description="Truncated to 500 chars")
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    employment_type: Optional[str] = None
    seniority: Optional[str] = None
    skills: Optional[str] = None
    location: Optional[str] = None
    posted_date: Optional[date] = None
    job_url: Optional[str] = None
    similarity_score: float = Field(description="Combined hybrid score [0, 1]")
    semantic_score: Optional[float] = Field(None, description="Semantic similarity component")
    bm25_score: Optional[float] = Field(None, description="BM25 keyword score")
    freshness_score: Optional[float] = Field(None, description="Recency score")
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    explanations: Optional["MatchExplanation"] = None

    @classmethod
    def from_internal(cls, internal) -> "JobResult":
        """Create from internal search engine JobResult dataclass."""
        return cls(
            uuid=internal.uuid,
            title=internal.title,
            company_name=getattr(internal, "company_name", None),
            description=getattr(internal, "description", ""),
            salary_min=getattr(internal, "salary_min", None),
            salary_max=getattr(internal, "salary_max", None),
            employment_type=getattr(internal, "employment_type", None),
            seniority=getattr(internal, "seniority", None),
            skills=getattr(internal, "skills", None),
            location=getattr(internal, "location", None),
            posted_date=getattr(internal, "posted_date", None),
            job_url=getattr(internal, "job_url", None),
            similarity_score=getattr(internal, "similarity_score", 0.0),
            semantic_score=getattr(internal, "semantic_score", None),
            bm25_score=getattr(internal, "bm25_score", None),
            freshness_score=getattr(internal, "freshness_score", None),
            matched_skills=getattr(internal, "matched_skills", []),
            missing_skills=getattr(internal, "missing_skills", []),
            explanations=MatchExplanation.from_internal(getattr(internal, "explanations"))
            if getattr(internal, "explanations", None)
            else None,
        )


class MatchExplanation(BaseModel):
    """Structured evidence for search and profile matches."""

    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    freshness_score: Optional[float] = None
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    query_terms: list[str] = Field(default_factory=list)
    skill_overlap_score: Optional[float] = None
    seniority_fit: Optional[float] = None
    salary_fit: Optional[float] = None
    overall_fit: Optional[float] = None

    @classmethod
    def from_internal(cls, internal) -> "MatchExplanation":
        """Create from the internal explanation dataclass."""
        return cls(
            semantic_score=getattr(internal, "semantic_score", None),
            bm25_score=getattr(internal, "bm25_score", None),
            freshness_score=getattr(internal, "freshness_score", None),
            matched_skills=getattr(internal, "matched_skills", []),
            missing_skills=getattr(internal, "missing_skills", []),
            query_terms=getattr(internal, "query_terms", []),
            skill_overlap_score=getattr(internal, "skill_overlap_score", None),
            seniority_fit=getattr(internal, "seniority_fit", None),
            salary_fit=getattr(internal, "salary_fit", None),
            overall_fit=getattr(internal, "overall_fit", None),
        )


class SearchResponse(BaseModel):
    """Response from search endpoints."""

    results: list[JobResult]
    total_candidates: int = Field(description="Jobs matching filters before semantic ranking")
    search_time_ms: float
    query_expansion: Optional[list[str]] = Field(None, description="Expanded query terms if expansion was enabled")
    degraded: bool = Field(False, description="True if fell back to keyword-only search")
    cache_hit: bool = Field(False, description="True if result was served from cache")

    @classmethod
    def from_internal(cls, internal) -> "SearchResponse":
        """Create from internal search engine SearchResponse dataclass."""
        return cls(
            results=[JobResult.from_internal(r) for r in internal.results],
            total_candidates=internal.total_candidates,
            search_time_ms=internal.search_time_ms,
            query_expansion=internal.query_expansion,
            degraded=internal.degraded,
            cache_hit=internal.cache_hit,
        )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "results": [
                        {
                            "uuid": "abc-123",
                            "title": "ML Engineer",
                            "company_name": "Google",
                            "description": "We are looking for...",
                            "salary_min": 12000,
                            "salary_max": 18000,
                            "employment_type": "Full Time",
                            "skills": "Python, TensorFlow, PyTorch",
                            "location": "Singapore",
                            "posted_date": "2024-01-15",
                            "job_url": "https://www.mycareersfuture.gov.sg/job/ml-engineer-abc-123",
                            "similarity_score": 0.923,
                        }
                    ],
                    "total_candidates": 1234,
                    "search_time_ms": 47.2,
                    "degraded": False,
                    "cache_hit": False,
                }
            ]
        }
    }


class SimilarBatchResponse(BaseModel):
    """Response from batch similar jobs endpoint."""

    results: dict[str, list[JobResult]] = Field(description="Mapping of UUID -> similar jobs")
    search_time_ms: float


class CompanySimilarity(BaseModel):
    """A similar company result."""

    company_name: str
    similarity_score: float
    job_count: int = Field(description="Total jobs from this company")
    avg_salary: Optional[int] = Field(None, description="Average salary offered")
    top_skills: list[str] = Field(default_factory=list, description="Most common skills in job postings")

    @classmethod
    def from_internal(cls, internal) -> "CompanySimilarity":
        """Create from internal CompanySimilarity dataclass."""
        return cls(
            company_name=internal.company_name,
            similarity_score=internal.similarity_score,
            job_count=internal.job_count,
            avg_salary=internal.avg_salary,
            top_skills=internal.top_skills,
        )


class StatsResponse(BaseModel):
    """System statistics response."""

    total_jobs: int
    jobs_with_embeddings: int
    embedding_coverage_pct: float
    unique_skills: int
    unique_companies: int
    index_size_mb: float
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="'healthy' or 'degraded'")
    index_loaded: bool
    degraded: bool = Field(description="True if running without vector search")


# =============================================================================
# Skill Feature Models
# =============================================================================


class SkillCloudItem(BaseModel):
    """Item in skill cloud response."""

    skill: str
    job_count: int
    cluster_id: Optional[int] = Field(None, description="Cluster ID for color coding in visualizations")


class SkillCloudResponse(BaseModel):
    """Response for skill cloud endpoint."""

    items: list[SkillCloudItem]
    total_unique_skills: int = Field(description="Total unique skills across all jobs (before filtering)")


class RelatedSkill(BaseModel):
    """A related skill with similarity score."""

    skill: str
    similarity: float = Field(description="Cosine similarity [0, 1]")
    same_cluster: bool = Field(description="True if in same skill cluster (strong synonym)")


class RelatedSkillsResponse(BaseModel):
    """Response for related skills endpoint."""

    skill: str = Field(description="Input skill queried")
    related: list[RelatedSkill]


class TrendPoint(BaseModel):
    """Single monthly point in a market trend series."""

    month: str
    job_count: int
    market_share: float
    median_salary_annual: Optional[int] = None
    momentum: float


class SkillTrendSeries(BaseModel):
    """Trend series for one skill."""

    skill: str
    series: list[TrendPoint]
    latest: Optional[TrendPoint] = None


class RoleTrendResponse(BaseModel):
    """Trend series for a role/query."""

    query: str
    series: list[TrendPoint]
    latest: Optional[TrendPoint] = None


class MonthlySkillSnapshot(BaseModel):
    """Top skills snapshot for a single month."""

    month: str
    skills: list[SkillCloudItem]


class CompanyTrendResponse(BaseModel):
    """Company trend response."""

    company_name: str
    series: list[TrendPoint]
    top_skills_by_month: list[MonthlySkillSnapshot]
    similar_companies: list[CompanySimilarity] = Field(default_factory=list)


class OverviewMetric(BaseModel):
    """Headline overview metrics."""

    total_jobs: int
    current_month_jobs: int
    unique_companies: int
    unique_skills: int
    avg_salary_annual: Optional[int] = None


class MomentumCard(BaseModel):
    """Rising skill/company card."""

    name: str
    job_count: int
    momentum: float
    median_salary_annual: Optional[int] = None


class InsightCard(BaseModel):
    """Small overview insight card."""

    label: str
    value: Optional[int] = None
    delta: float


class SalaryMovement(BaseModel):
    """Salary movement summary."""

    current_median_salary_annual: Optional[int] = None
    previous_median_salary_annual: Optional[int] = None
    change_pct: float


class OverviewResponse(BaseModel):
    """Homepage overview payload."""

    headline_metrics: OverviewMetric
    rising_skills: list[MomentumCard]
    rising_companies: list[MomentumCard]
    salary_movement: SalaryMovement
    market_insights: list[InsightCard]


class ProfileMatchResponse(BaseModel):
    """Profile matching response."""

    results: list[JobResult]
    extracted_skills: list[str]
    total_candidates: int
    search_time_ms: float
    degraded: bool


class CareerDeltaConfidence(BaseModel):
    """Explainable confidence metadata for API consumers."""

    score: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(0.0, ge=0.0, le=1.0)
    market_sample_size: int = Field(0, ge=0)
    reasons: list[str] = Field(default_factory=list)

    @classmethod
    def from_internal(cls, internal: InternalScenarioConfidence) -> "CareerDeltaConfidence":
        return cls(
            score=internal.score,
            evidence_coverage=internal.evidence_coverage,
            market_sample_size=internal.market_sample_size,
            reasons=list(internal.reasons),
        )


class CareerDeltaScoreBreakdown(BaseModel):
    """Composite ranking components exposed for debugging and UI disclosure."""

    opportunity: float = 0.0
    quality: float = 0.0
    salary: float = 0.0
    momentum: float = 0.0
    diversity: float = 0.0
    raw_score: float = 0.0
    pivot_cost: float = 0.0
    final_score: float = 0.0

    @classmethod
    def from_internal(cls, internal: Optional[InternalScenarioScoreBreakdown]) -> Optional["CareerDeltaScoreBreakdown"]:
        if internal is None:
            return None
        return cls(
            opportunity=internal.opportunity,
            quality=internal.quality,
            salary=internal.salary,
            momentum=internal.momentum,
            diversity=internal.diversity,
            raw_score=internal.raw_score,
            pivot_cost=internal.pivot_cost,
            final_score=internal.final_score,
        )


class CareerDeltaMarketInsight(BaseModel):
    """Count/share metric used in baseline summaries."""

    name: str
    job_count: int = Field(ge=0)
    share_pct: float = Field(ge=0.0)

    @classmethod
    def from_internal(cls, internal: InternalMarketInsight) -> "CareerDeltaMarketInsight":
        return cls(
            name=internal.name,
            job_count=internal.job_count,
            share_pct=internal.share_pct,
        )


class CareerDeltaSalaryBand(BaseModel):
    """Salary range summary for a baseline or scenario market bucket."""

    min_annual: Optional[int] = Field(None, ge=0)
    median_annual: Optional[int] = Field(None, ge=0)
    max_annual: Optional[int] = Field(None, ge=0)

    @classmethod
    def from_internal(cls, internal: InternalSalaryBand) -> "CareerDeltaSalaryBand":
        return cls(
            min_annual=internal.min_annual,
            median_annual=internal.median_annual,
            max_annual=internal.max_annual,
        )


class CareerDeltaSkillReplacement(BaseModel):
    """One skill swap proposed by a scenario."""

    from_skill: str
    to_skill: str

    @classmethod
    def from_internal(cls, internal: InternalSkillReplacement) -> "CareerDeltaSkillReplacement":
        return cls(from_skill=internal.from_skill, to_skill=internal.to_skill)


class CareerDeltaScenarioChange(BaseModel):
    """Structured change payload for a scenario."""

    added_skills: list[str] = Field(default_factory=list)
    removed_skills: list[str] = Field(default_factory=list)
    replaced_skills: list[CareerDeltaSkillReplacement] = Field(default_factory=list)
    source_title_family: Optional[str] = None
    target_title_family: Optional[str] = None
    source_industry: Optional[str] = None
    target_industry: Optional[str] = None

    @classmethod
    def from_internal(cls, internal: Optional[InternalScenarioChange]) -> Optional["CareerDeltaScenarioChange"]:
        if internal is None:
            return None
        return cls(
            added_skills=list(internal.added_skills),
            removed_skills=list(internal.removed_skills),
            replaced_skills=[CareerDeltaSkillReplacement.from_internal(item) for item in internal.replaced_skills],
            source_title_family=internal.source_title_family,
            target_title_family=internal.target_title_family,
            source_industry=internal.source_industry,
            target_industry=internal.target_industry,
        )


class CareerDeltaScenarioSignal(BaseModel):
    """Flattened market evidence for either skill or pivot scenarios."""

    signal_type: str = Field(description="'skill' or 'pivot'")
    skill: Optional[str] = None
    supporting_jobs: int = Field(ge=0)
    supporting_share_pct: float = Field(ge=0.0)
    market_job_count: int = Field(ge=0)
    market_salary_annual_median: Optional[int] = Field(None, ge=0)
    market_momentum: Optional[float] = None
    salary_lift_pct: Optional[float] = None
    similarity: Optional[float] = None
    same_cluster: Optional[bool] = None
    target_title_family: Optional[str] = None
    target_industry: Optional[str] = None
    title_distance: Optional[str] = None
    industry_distance: Optional[int] = None
    fit_median: Optional[float] = None

    @classmethod
    def from_internal(
        cls,
        internal: InternalSkillScenarioSignal | InternalPivotScenarioSignal,
    ) -> "CareerDeltaScenarioSignal":
        if isinstance(internal, InternalSkillScenarioSignal):
            return cls(
                signal_type="skill",
                skill=internal.skill,
                supporting_jobs=internal.supporting_jobs,
                supporting_share_pct=internal.supporting_share_pct,
                market_job_count=internal.market_job_count,
                market_salary_annual_median=internal.market_salary_annual_median,
                market_momentum=internal.market_momentum,
                salary_lift_pct=internal.salary_lift_pct,
                similarity=internal.similarity,
                same_cluster=internal.same_cluster,
            )
        return cls(
            signal_type="pivot",
            supporting_jobs=internal.supporting_jobs,
            supporting_share_pct=internal.supporting_share_pct,
            market_job_count=internal.market_job_count,
            market_salary_annual_median=internal.market_salary_annual_median,
            market_momentum=internal.market_momentum,
            salary_lift_pct=internal.salary_lift_pct,
            target_title_family=internal.target_title_family,
            target_industry=internal.target_industry,
            title_distance=internal.title_distance,
            industry_distance=internal.industry_distance,
            fit_median=internal.fit_median,
        )


class CareerDeltaBaseline(BaseModel):
    """Baseline market-position payload."""

    position: CareerDeltaMarketPosition
    reachable_jobs: int = Field(ge=0)
    total_candidates: int = Field(ge=0)
    fit_median: float = Field(ge=0.0, le=1.0)
    fit_p90: float = Field(ge=0.0, le=1.0)
    salary_band: CareerDeltaSalaryBand
    top_industries: list[CareerDeltaMarketInsight] = Field(default_factory=list)
    top_companies: list[CareerDeltaMarketInsight] = Field(default_factory=list)
    extracted_skills: list[str] = Field(default_factory=list)
    skill_coverage: float = Field(0.0, ge=0.0, le=1.0)
    top_skill_gaps: list[CareerDeltaMarketInsight] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    thin_market: bool = False
    degraded: bool = False

    @classmethod
    def from_internal(cls, internal: InternalBaselineMarketPosition) -> "CareerDeltaBaseline":
        return cls(
            position=CareerDeltaMarketPosition(internal.position.value),
            reachable_jobs=internal.reachable_jobs,
            total_candidates=internal.total_candidates,
            fit_median=internal.fit_median,
            fit_p90=internal.fit_p90,
            salary_band=CareerDeltaSalaryBand.from_internal(internal.salary_band),
            top_industries=[CareerDeltaMarketInsight.from_internal(item) for item in internal.top_industries],
            top_companies=[CareerDeltaMarketInsight.from_internal(item) for item in internal.top_companies],
            extracted_skills=list(internal.extracted_skills),
            skill_coverage=internal.skill_coverage,
            top_skill_gaps=[CareerDeltaMarketInsight.from_internal(item) for item in internal.top_skill_gaps],
            notes=list(internal.notes),
            thin_market=internal.thin_market,
            degraded=internal.degraded,
        )


class CareerDeltaScenarioSummary(BaseModel):
    """Compact scenario row for list views."""

    scenario_id: str
    scenario_type: CareerDeltaScenarioType
    title: str
    summary: str
    market_position: CareerDeltaMarketPosition
    confidence: CareerDeltaConfidence
    score_breakdown: Optional[CareerDeltaScoreBreakdown] = None
    change: Optional[CareerDeltaScenarioChange] = None
    signals: list[CareerDeltaScenarioSignal] = Field(default_factory=list)
    target_title: Optional[str] = None
    target_sector: Optional[str] = None
    thin_market: bool = False
    degraded: bool = False
    expected_salary_delta_pct: Optional[float] = None

    @classmethod
    def from_internal(cls, internal: InternalScenarioSummary) -> "CareerDeltaScenarioSummary":
        return cls(
            scenario_id=internal.scenario_id,
            scenario_type=CareerDeltaScenarioType(internal.scenario_type.value),
            title=internal.title,
            summary=internal.summary,
            market_position=CareerDeltaMarketPosition(internal.market_position.value),
            confidence=CareerDeltaConfidence.from_internal(internal.confidence),
            score_breakdown=CareerDeltaScoreBreakdown.from_internal(internal.score_breakdown),
            change=CareerDeltaScenarioChange.from_internal(internal.change),
            signals=[CareerDeltaScenarioSignal.from_internal(signal) for signal in internal.signals],
            target_title=internal.target_title,
            target_sector=internal.target_sector,
            thin_market=internal.thin_market,
            degraded=internal.degraded,
            expected_salary_delta_pct=internal.expected_salary_delta_pct,
        )


class CareerDeltaScenarioDetail(BaseModel):
    """Expanded scenario payload for detail views."""

    scenario_id: str
    scenario_type: CareerDeltaScenarioType
    title: str
    narrative: str
    market_position: CareerDeltaMarketPosition
    confidence: CareerDeltaConfidence
    score_breakdown: Optional[CareerDeltaScoreBreakdown] = None
    summary: Optional[CareerDeltaScenarioSummary] = None
    change: Optional[CareerDeltaScenarioChange] = None
    signals: list[CareerDeltaScenarioSignal] = Field(default_factory=list)
    target_title: Optional[str] = None
    target_sector: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    thin_market: bool = False
    degraded: bool = False

    @classmethod
    def from_internal(cls, internal: InternalScenarioDetail) -> "CareerDeltaScenarioDetail":
        return cls(
            scenario_id=internal.scenario_id,
            scenario_type=CareerDeltaScenarioType(internal.scenario_type.value),
            title=internal.title,
            narrative=internal.narrative,
            market_position=CareerDeltaMarketPosition(internal.market_position.value),
            confidence=CareerDeltaConfidence.from_internal(internal.confidence),
            score_breakdown=CareerDeltaScoreBreakdown.from_internal(internal.score_breakdown),
            summary=CareerDeltaScenarioSummary.from_internal(internal.summary) if internal.summary else None,
            change=CareerDeltaScenarioChange.from_internal(internal.change),
            signals=[CareerDeltaScenarioSignal.from_internal(signal) for signal in internal.signals],
            target_title=internal.target_title,
            target_sector=internal.target_sector,
            evidence=list(internal.evidence),
            missing_skills=list(internal.missing_skills),
            search_queries=list(internal.search_queries),
            thin_market=internal.thin_market,
            degraded=internal.degraded,
        )


class CareerDeltaFilteredScenario(BaseModel):
    """Reason a plausible scenario was withheld from ranking."""

    scenario_id: str
    scenario_type: CareerDeltaScenarioType
    reason_code: str
    explanation: str
    confidence: CareerDeltaConfidence
    market_position: CareerDeltaMarketPosition = CareerDeltaMarketPosition.UNCLEAR

    @classmethod
    def from_internal(cls, internal: InternalFilteredScenario) -> "CareerDeltaFilteredScenario":
        return cls(
            scenario_id=internal.scenario_id,
            scenario_type=CareerDeltaScenarioType(internal.scenario_type.value),
            reason_code=internal.reason_code,
            explanation=internal.explanation,
            confidence=CareerDeltaConfidence.from_internal(internal.confidence),
            market_position=CareerDeltaMarketPosition(internal.market_position.value),
        )


class CareerDeltaAnalysisResponse(BaseModel):
    """Top-level response payload for career-delta analysis."""

    baseline: Optional[CareerDeltaBaseline] = None
    scenarios: list[CareerDeltaScenarioSummary] = Field(default_factory=list)
    filtered_scenarios: list[CareerDeltaFilteredScenario] = Field(default_factory=list)
    degraded: bool = False
    thin_market: bool = False
    analysis_time_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="End-to-end backend execution time in milliseconds when available.",
    )

    @classmethod
    def from_internal(
        cls,
        internal: InternalCareerDeltaResponse,
        *,
        allowed_delta_types: Optional[list[CareerDeltaScenarioType] | tuple[CareerDeltaScenarioType, ...]] = None,
        analysis_time_ms: Optional[float] = None,
    ) -> "CareerDeltaAnalysisResponse":
        allowed = {item.value for item in allowed_delta_types or ()}
        summaries = [item for item in internal.summaries if not allowed or item.scenario_type.value in allowed]
        filtered = [item for item in internal.filtered_scenarios if not allowed or item.scenario_type.value in allowed]
        return cls(
            baseline=CareerDeltaBaseline.from_internal(internal.baseline) if internal.baseline else None,
            scenarios=[CareerDeltaScenarioSummary.from_internal(item) for item in summaries],
            filtered_scenarios=[CareerDeltaFilteredScenario.from_internal(item) for item in filtered],
            degraded=internal.degraded,
            thin_market=internal.thin_market,
            analysis_time_ms=analysis_time_ms,
        )


JobResult.model_rebuild()


# =============================================================================
# Error Models
# =============================================================================


class ErrorDetail(BaseModel):
    """Error detail in response."""

    code: str = Field(description="Error code (e.g., VALIDATION_ERROR)")
    message: str
    details: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: ErrorDetail
