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
from typing import Optional

from pydantic import BaseModel, Field, field_validator

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
