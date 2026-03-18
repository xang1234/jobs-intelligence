"""
Internal career-delta engine contract and orchestration skeleton.

This module defines the internal semantics before API or UI-specific models
exist. Later tasks can build scenario generation, market stats, caching, and
detail retrieval on these dataclasses without inventing new shapes ad hoc.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

SCENARIO_ID_VERSION = "cd1"
_SCENARIO_PART_RE = re.compile(r"[^a-z0-9]+")


class MarketPosition(str, Enum):
    """
    Baseline market vocabulary used across summaries and details.

    These labels are intentionally coarse. They describe how favorable the
    external market looks for a scenario, not whether the user should take it.
    """

    LEADING = "leading"
    COMPETITIVE = "competitive"
    STRETCH = "stretch"
    THIN = "thin"
    UNCLEAR = "unclear"


class ScenarioType(str, Enum):
    """Stable scenario families for career delta analysis."""

    BASELINE = "baseline"
    SAME_ROLE = "same_role"
    ADJACENT_ROLE = "adjacent_role"
    INDUSTRY_PIVOT = "industry_pivot"
    FILTERED_OUT = "filtered_out"


@dataclass(frozen=True)
class CareerDeltaRequest:
    """
    Normalized engine request.

    This is the internal shape the orchestrator consumes after the API layer has
    handled validation and transport concerns.
    """

    profile_text: str
    current_title: Optional[str] = None
    target_titles: tuple[str, ...] = ()
    current_categories: tuple[str, ...] = ()
    current_skills: tuple[str, ...] = ()
    current_company: Optional[str] = None
    location: Optional[str] = None
    target_salary_min: Optional[int] = None
    limit: int = 12
    include_filtered: bool = True

    def normalized_target_titles(self) -> tuple[str, ...]:
        """Deduplicate titles while preserving input order."""
        deduped: list[str] = []
        for title in self.target_titles:
            cleaned = " ".join(title.split())
            if cleaned and cleaned not in deduped:
                deduped.append(cleaned)
        return tuple(deduped)


@dataclass(frozen=True)
class ScenarioConfidence:
    """
    Explainable confidence metadata for ranking and UI disclosure.
    """

    score: float
    evidence_coverage: float = 0.0
    market_sample_size: int = 0
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketInsight:
    """Lightweight count/share entry for companies, industries, and skill gaps."""

    name: str
    job_count: int
    share_pct: float


@dataclass(frozen=True)
class SalaryBand:
    """Baseline salary range summary derived from the candidate pool."""

    min_annual: Optional[int] = None
    median_annual: Optional[int] = None
    max_annual: Optional[int] = None


@dataclass(frozen=True)
class CareerDeltaCandidate:
    """Per-job evidence retained for later scenario rescoring and detail payloads."""

    uuid: str
    title: str
    company_name: str
    title_family: str
    industry_key: str
    industry_label: str
    overall_fit: float
    retrieval_score: float
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    skill_overlap_score: float = 0.0
    seniority_fit: Optional[float] = None
    salary_fit: Optional[float] = None
    matched_skills: tuple[str, ...] = ()
    missing_skills: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    salary_annual_min: Optional[int] = None
    salary_annual_max: Optional[int] = None
    employment_type: Optional[str] = None
    seniority: Optional[str] = None
    region: Optional[str] = None
    location: Optional[str] = None
    posted_date: Optional[str] = None
    job_url: Optional[str] = None


@dataclass(frozen=True)
class CareerDeltaCandidatePool:
    """Reusable wide-pool retrieval output for baseline and later scenarios."""

    candidates: tuple[CareerDeltaCandidate, ...] = ()
    extracted_skills: tuple[str, ...] = ()
    total_candidates: int = 0
    degraded: bool = False


@dataclass(frozen=True)
class BaselineMarketPosition:
    """Current-market baseline derived from the reusable candidate pool."""

    position: MarketPosition
    reachable_jobs: int
    total_candidates: int
    fit_median: float
    fit_p90: float
    salary_band: SalaryBand
    top_industries: tuple[MarketInsight, ...] = ()
    top_companies: tuple[MarketInsight, ...] = ()
    extracted_skills: tuple[str, ...] = ()
    skill_coverage: float = 0.0
    top_skill_gaps: tuple[MarketInsight, ...] = ()
    notes: tuple[str, ...] = ()
    thin_market: bool = False
    degraded: bool = False


@dataclass(frozen=True)
class ScenarioSummary:
    """
    Lightweight scenario row for list views.

    Keep this payload compact and stable. It is what callers can render or rank
    without fetching detail expansions.
    """

    scenario_id: str
    scenario_type: ScenarioType
    title: str
    summary: str
    market_position: MarketPosition
    confidence: ScenarioConfidence
    target_title: Optional[str] = None
    target_sector: Optional[str] = None
    thin_market: bool = False
    degraded: bool = False
    expected_salary_delta_pct: Optional[float] = None


@dataclass(frozen=True)
class ScenarioDetail:
    """
    Expanded scenario payload suitable for detail caching.

    The `scenario_id` must match the summary row and remain stable for the same
    logical scenario across list/detail requests.
    """

    scenario_id: str
    scenario_type: ScenarioType
    title: str
    narrative: str
    market_position: MarketPosition
    confidence: ScenarioConfidence
    summary: Optional[ScenarioSummary] = None
    target_title: Optional[str] = None
    target_sector: Optional[str] = None
    evidence: tuple[str, ...] = ()
    missing_skills: tuple[str, ...] = ()
    search_queries: tuple[str, ...] = ()
    thin_market: bool = False
    degraded: bool = False


@dataclass(frozen=True)
class FilteredScenario:
    """
    Records why a plausible scenario was considered but withheld from ranking.
    """

    scenario_id: str
    scenario_type: ScenarioType
    reason_code: str
    explanation: str
    confidence: ScenarioConfidence
    market_position: MarketPosition = MarketPosition.UNCLEAR


@dataclass(frozen=True)
class CareerDeltaResponse:
    """Orchestrator result contract for summaries, filtered rows, and flags."""

    request: CareerDeltaRequest
    baseline: Optional[BaselineMarketPosition] = None
    candidate_pool: Optional[CareerDeltaCandidatePool] = None
    summaries: tuple[ScenarioSummary, ...] = ()
    filtered_scenarios: tuple[FilteredScenario, ...] = ()
    degraded: bool = False
    thin_market: bool = False


class IndustryTaxonomyHelper(Protocol):
    """Small surface area needed from taxonomy logic."""

    def normalize_title_family(self, title: str): ...


class MarketStatsProvider(Protocol):
    """Provider for aggregated market statistics used by scenario generation."""

    def get_market_snapshot(self, request: CareerDeltaRequest) -> dict: ...


class SearchScoringProvider(Protocol):
    """Provider for wide-pool retrieval and scenario scoring helpers."""

    def build_candidate_pool(self, request: CareerDeltaRequest) -> CareerDeltaCandidatePool: ...


@dataclass
class CareerDeltaDependencies:
    """
    Swappable collaborators to avoid circular imports as the feature grows.
    """

    taxonomy: Optional[IndustryTaxonomyHelper] = None
    market_stats: Optional[MarketStatsProvider] = None
    search_scoring: Optional[SearchScoringProvider] = None


class CareerDeltaEngine:
    """
    Internal orchestrator skeleton.

    This intentionally does not generate scenarios yet. It centralizes request
    normalization, degraded/thin-market signaling, and the stable contract that
    downstream tasks will populate.
    """

    def __init__(self, dependencies: Optional[CareerDeltaDependencies] = None):
        self.dependencies = dependencies or CareerDeltaDependencies()

    def analyze(self, request: CareerDeltaRequest) -> CareerDeltaResponse:
        """
        Analyze a profile and return the internal career-delta response shape.

        The initial implementation returns an empty but fully-typed contract so
        later tasks can layer in taxonomy, market baselines, and scoring without
        revisiting request/response semantics.
        """
        normalized_request = CareerDeltaRequest(
            profile_text=request.profile_text.strip(),
            current_title=request.current_title,
            target_titles=request.normalized_target_titles(),
            current_categories=request.current_categories,
            current_skills=request.current_skills,
            current_company=request.current_company,
            location=request.location,
            target_salary_min=request.target_salary_min,
            limit=request.limit,
            include_filtered=request.include_filtered,
        )

        degraded = any(
            dependency is None
            for dependency in (
                self.dependencies.taxonomy,
                self.dependencies.market_stats,
                self.dependencies.search_scoring,
            )
        )

        if degraded:
            return CareerDeltaResponse(
                request=normalized_request,
                summaries=(),
                filtered_scenarios=(),
                degraded=True,
                thin_market=False,
            )

        candidate_pool = self.dependencies.search_scoring.build_candidate_pool(normalized_request)
        market_snapshot = self.dependencies.market_stats.get_market_snapshot(normalized_request)
        baseline = summarize_market_position(
            candidate_pool,
            market_snapshot=market_snapshot,
            target_salary_min=normalized_request.target_salary_min,
        )

        return CareerDeltaResponse(
            request=normalized_request,
            baseline=baseline,
            candidate_pool=candidate_pool,
            summaries=(),
            filtered_scenarios=(),
            degraded=bool(candidate_pool.degraded),
            thin_market=baseline.thin_market,
        )


def build_scenario_id(
    scenario_type: ScenarioType,
    *,
    source_title_family: Optional[str] = None,
    target_title_family: Optional[str] = None,
    target_sector: Optional[str] = None,
    market_position: Optional[MarketPosition] = None,
) -> str:
    """
    Build a stable, cache-safe identifier for a logical scenario.

    The human-readable prefix helps debugging. The digest keeps IDs short while
    staying stable for list/detail expansion within the same semantics version.
    """
    parts = [
        SCENARIO_ID_VERSION,
        scenario_type.value,
        _normalize_id_part(source_title_family),
        _normalize_id_part(target_title_family),
        _normalize_id_part(target_sector),
        market_position.value if market_position else "",
    ]
    fingerprint = "|".join(parts)
    digest = hashlib.blake2s(fingerprint.encode("utf-8"), digest_size=8).hexdigest()
    return f"{scenario_type.value}:{digest}"


def build_filtered_scenario(
    *,
    scenario_type: ScenarioType,
    reason_code: str,
    explanation: str,
    confidence: ScenarioConfidence,
    source_title_family: Optional[str] = None,
    target_title_family: Optional[str] = None,
    target_sector: Optional[str] = None,
    market_position: MarketPosition = MarketPosition.UNCLEAR,
) -> FilteredScenario:
    """Helper for producing stable filtered-scenario rows."""
    scenario_id = build_scenario_id(
        scenario_type,
        source_title_family=source_title_family,
        target_title_family=target_title_family,
        target_sector=target_sector,
        market_position=market_position,
    )
    return FilteredScenario(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        reason_code=reason_code,
        explanation=explanation,
        confidence=confidence,
        market_position=market_position,
    )


def _normalize_id_part(value: Optional[str]) -> str:
    if not value:
        return ""
    return _SCENARIO_PART_RE.sub("-", value.lower()).strip("-")


def summarize_market_position(
    candidate_pool: CareerDeltaCandidatePool,
    *,
    market_snapshot: Optional[dict] = None,
    target_salary_min: Optional[int] = None,
) -> BaselineMarketPosition:
    """Summarize the candidate's current market position from a reusable pool."""
    candidates = list(candidate_pool.candidates)
    fits = [candidate.overall_fit for candidate in candidates]
    reachable = [candidate for candidate in candidates if candidate.overall_fit >= 0.55]
    analysis_set = reachable or candidates

    reachable_jobs = len(reachable)
    total_candidates = candidate_pool.total_candidates or len(candidates)
    fit_median = _quantile(fits, 0.5)
    fit_p90 = _quantile(fits, 0.9)

    salary_values = [
        _salary_midpoint(candidate.salary_annual_min, candidate.salary_annual_max)
        for candidate in analysis_set
        if _salary_midpoint(candidate.salary_annual_min, candidate.salary_annual_max) is not None
    ]
    salary_band = SalaryBand(
        min_annual=min(salary_values) if salary_values else None,
        median_annual=_integer_median(salary_values),
        max_annual=max(salary_values) if salary_values else None,
    )

    top_industries = _top_insights([candidate.industry_label for candidate in analysis_set])
    top_companies = _top_insights([candidate.company_name for candidate in analysis_set])
    top_skill_gaps = _top_insights(
        gap
        for candidate in analysis_set[:20]
        for gap in candidate.missing_skills
        if gap not in candidate.matched_skills
    )

    skill_coverage_values = [
        len(candidate.matched_skills) / len(candidate_pool.extracted_skills)
        for candidate in analysis_set
        if candidate_pool.extracted_skills
    ]
    skill_coverage = round(sum(skill_coverage_values) / len(skill_coverage_values), 4) if skill_coverage_values else 0.0

    thin_market = total_candidates < 15 or reachable_jobs < 5
    notes: list[str] = []
    if thin_market:
        notes.append(
            "Baseline evidence is limited, so scenario confidence should be conservative."
        )
    if candidate_pool.degraded:
        notes.append(
            "Vector retrieval was unavailable, so baseline fit relies on keyword fallback relevance."
        )
    if (
        target_salary_min is not None
        and salary_band.median_annual is not None
        and salary_band.median_annual < target_salary_min
    ):
        notes.append("Typical reachable salary is below the requested target band.")
    if market_snapshot and market_snapshot.get("current_industry") is not None:
        current_industry = market_snapshot["current_industry"]
        if getattr(current_industry, "key", "").startswith("unknown/"):
            notes.append("Current industry baseline is uncertain because source evidence is incomplete.")

    if total_candidates == 0:
        position = MarketPosition.THIN
    elif thin_market:
        position = MarketPosition.THIN
    elif fit_median >= 0.72 and reachable_jobs >= 12:
        position = MarketPosition.LEADING
    elif fit_median >= 0.58 and reachable_jobs >= 6:
        position = MarketPosition.COMPETITIVE
    elif reachable_jobs >= 2 or fit_median >= 0.4:
        position = MarketPosition.STRETCH
    else:
        position = MarketPosition.UNCLEAR

    return BaselineMarketPosition(
        position=position,
        reachable_jobs=reachable_jobs,
        total_candidates=total_candidates,
        fit_median=fit_median,
        fit_p90=fit_p90,
        salary_band=salary_band,
        top_industries=top_industries,
        top_companies=top_companies,
        extracted_skills=candidate_pool.extracted_skills,
        skill_coverage=skill_coverage,
        top_skill_gaps=top_skill_gaps,
        notes=tuple(notes),
        thin_market=thin_market,
        degraded=candidate_pool.degraded,
    )


def _top_insights(names) -> tuple[MarketInsight, ...]:
    counts: dict[str, int] = {}
    total = 0
    for name in names:
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
        total += 1
    if total == 0:
        return ()
    ranked = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[:5]
    return tuple(
        MarketInsight(name=name, job_count=count, share_pct=round((count / total) * 100, 2))
        for name, count in ranked
    )


def _integer_median(values: list[int]) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return int((ordered[mid - 1] + ordered[mid]) / 2)


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return round(float(ordered[index]), 4)


def _salary_midpoint(low: Optional[int], high: Optional[int]) -> Optional[int]:
    if low is not None and high is not None:
        return int((low + high) / 2)
    if low is not None:
        return low
    return high
