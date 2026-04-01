"""
Internal career-delta engine contract and orchestration skeleton.

This module defines the internal semantics before API or UI-specific models
exist. Later tasks can build scenario generation, market stats, caching, and
detail retrieval on these dataclasses without inventing new shapes ad hoc.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Protocol

from .industry_taxonomy import IndustryClassification, industry_distance, is_adjacent_role, normalize_title_family

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
    TITLE_PIVOT = "title_pivot"
    SAME_ROLE_INDUSTRY_PIVOT = "same_role_industry_pivot"
    ADJACENT_ROLE_INDUSTRY_PIVOT = "adjacent_role_industry_pivot"
    SKILL_ADDITION = "skill_addition"
    SKILL_SUBSTITUTION = "skill_substitution"
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
class ScenarioScoreBreakdown:
    """Composite ranking components for a scenario."""

    opportunity: float = 0.0
    quality: float = 0.0
    salary: float = 0.0
    momentum: float = 0.0
    diversity: float = 0.0
    raw_score: float = 0.0
    pivot_cost: float = 0.0
    final_score: float = 0.0


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
class SkillReplacement:
    """Structured replacement from one current skill to a proposed alternative."""

    from_skill: str
    to_skill: str


@dataclass(frozen=True)
class ScenarioChange:
    """Structured change payload for generated scenarios."""

    added_skills: tuple[str, ...] = ()
    removed_skills: tuple[str, ...] = ()
    replaced_skills: tuple[SkillReplacement, ...] = ()
    source_title_family: Optional[str] = None
    target_title_family: Optional[str] = None
    source_industry: Optional[str] = None
    target_industry: Optional[str] = None


@dataclass(frozen=True)
class SkillScenarioSignal:
    """Structured market and adjacency evidence behind a skill scenario."""

    skill: str
    supporting_jobs: int
    supporting_share_pct: float
    market_job_count: int
    market_salary_annual_median: Optional[int] = None
    market_momentum: Optional[float] = None
    salary_lift_pct: Optional[float] = None
    similarity: Optional[float] = None
    same_cluster: Optional[bool] = None


@dataclass(frozen=True)
class PivotScenarioSignal:
    """Structured title/industry evidence behind pivot scenarios."""

    supporting_jobs: int
    supporting_share_pct: float
    target_title_family: str
    target_industry: str
    title_distance: str
    industry_distance: Optional[int]
    fit_median: float
    market_job_count: int
    market_salary_annual_median: Optional[int] = None
    market_momentum: Optional[float] = None
    salary_lift_pct: Optional[float] = None


ScenarioSignal = SkillScenarioSignal | PivotScenarioSignal


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
    target_title_match: bool = False
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    skill_overlap_score: float = 0.0
    seniority_fit: Optional[float] = None
    salary_fit: Optional[float] = None
    matched_skills: tuple[str, ...] = ()
    missing_skills: tuple[str, ...] = ()
    gap_skills: tuple[str, ...] = ()
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
    score_breakdown: Optional[ScenarioScoreBreakdown] = None
    change: Optional[ScenarioChange] = None
    signals: tuple[ScenarioSignal, ...] = ()
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
    score_breakdown: Optional[ScenarioScoreBreakdown] = None
    summary: Optional[ScenarioSummary] = None
    change: Optional[ScenarioChange] = None
    signals: tuple[ScenarioSignal, ...] = ()
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
    def get_skill_stats(self, skill: str): ...
    def get_title_family_stats(self, title_or_family: str): ...
    def get_industry_stats(self, industry: str): ...


class SearchScoringProvider(Protocol):
    """Provider for wide-pool retrieval and scenario scoring helpers."""

    def build_candidate_pool(self, request: CareerDeltaRequest) -> CareerDeltaCandidatePool: ...
    def get_related_skills(self, skill: str, k: int = 10): ...


@dataclass
class CareerDeltaDependencies:
    """
    Swappable collaborators to avoid circular imports as the feature grows.
    """

    taxonomy: Optional[IndustryTaxonomyHelper] = None
    market_stats: Optional[MarketStatsProvider] = None
    search_scoring: Optional[SearchScoringProvider] = None


@dataclass(frozen=True)
class ComputeBudget:
    """Soft compute limits for scenario scoring."""

    max_wall_time_ms: int = 5000
    max_scenarios_evaluated: int = 20


class CareerDeltaEngine:
    """
    Internal orchestrator skeleton.

    This intentionally does not generate scenarios yet. It centralizes request
    normalization, degraded/thin-market signaling, and the stable contract that
    downstream tasks will populate.
    """

    def __init__(
        self,
        dependencies: Optional[CareerDeltaDependencies] = None,
        *,
        budget: ComputeBudget = ComputeBudget(),
        clock: Callable[[], float] = time.monotonic,
    ):
        self.dependencies = dependencies or CareerDeltaDependencies()
        self.budget = budget
        self.clock = clock

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
        started_at = self.clock()
        skill_summaries = generate_skill_scenarios(
            normalized_request,
            candidate_pool,
            baseline,
            market_stats=self.dependencies.market_stats,
            search_scoring=self.dependencies.search_scoring,
        )
        pivot_summaries, filtered_scenarios = generate_pivot_scenarios(
            normalized_request,
            candidate_pool,
            baseline,
            market_snapshot=market_snapshot,
            market_stats=self.dependencies.market_stats,
        )
        raw_summaries = tuple(skill_summaries + pivot_summaries)
        ranked_summaries, ranking_filtered, budget_degraded = rank_and_filter_scenarios(
            raw_summaries,
            baseline=baseline,
            request=normalized_request,
            budget=self.budget,
            started_at=started_at,
            clock=self.clock,
        )
        filtered_union = tuple(filtered_scenarios + ranking_filtered)

        return CareerDeltaResponse(
            request=normalized_request,
            baseline=baseline,
            candidate_pool=candidate_pool,
            summaries=ranked_summaries,
            filtered_scenarios=filtered_union[:3] if normalized_request.include_filtered else (),
            degraded=bool(candidate_pool.degraded or budget_degraded),
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
    top_skill_gaps = _top_insights(gap for candidate in analysis_set[:20] for gap in candidate.gap_skills)

    skill_coverage_values = [
        len(candidate.matched_skills) / len(candidate_pool.extracted_skills)
        for candidate in analysis_set
        if candidate_pool.extracted_skills
    ]
    skill_coverage = round(sum(skill_coverage_values) / len(skill_coverage_values), 4) if skill_coverage_values else 0.0

    thin_market = total_candidates < 15 or reachable_jobs < 5
    notes: list[str] = []
    if thin_market:
        notes.append("Baseline evidence is limited, so scenario confidence should be conservative.")
    if candidate_pool.degraded:
        notes.append("Vector retrieval was unavailable, so baseline fit relies on keyword fallback relevance.")
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
        MarketInsight(name=name, job_count=count, share_pct=round((count / total) * 100, 2)) for name, count in ranked
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


MAX_SKILL_ADDITION_SCENARIOS = 3
MAX_SKILL_SUBSTITUTION_SCENARIOS = 3
MAX_TITLE_PIVOT_SCENARIOS = 1
MAX_SAME_ROLE_INDUSTRY_PIVOT_SCENARIOS = 2
MAX_ADJACENT_ROLE_INDUSTRY_PIVOT_SCENARIOS = 1
RELATED_SKILL_LIMIT = 6
REACHABLE_FIT_THRESHOLD = 0.55
MIN_SKILL_SUPPORTING_JOBS = 2
MIN_SKILL_SUPPORT_SHARE = 0.18
MIN_PIVOT_SUPPORTING_JOBS = 2
MIN_PIVOT_SUPPORT_SHARE = 0.18
MIN_SUBSTITUTION_SIMILARITY = 0.72
MIN_SUBSTITUTION_JOB_COUNT_RATIO = 1.25
MIN_SUBSTITUTION_MOMENTUM_DELTA = 0.05
MIN_SUBSTITUTION_SALARY_LIFT_PCT = 0.08
MIN_MATERIAL_JOB_COUNT_DELTA = 5
MIN_MATERIAL_JOB_COUNT_RATIO = 1.15
MIN_TITLE_PIVOT_FIT = 0.58
MIN_INDUSTRY_PIVOT_FIT = 0.55
MIN_TITLE_MARKET_IMPROVEMENT = 0.03
MIN_INDUSTRY_MARKET_IMPROVEMENT = 0.03
MAX_ADJACENT_INDUSTRY_DISTANCE = 1
MIN_SCENARIO_SIGNAL = 2
SEMANTIC_DEDUP_THRESHOLD = 0.7

# Rollout tuning note:
# The shortlist should default toward believable, lower-pivot moves rather than
# letting novelty or salary upside dominate. Evidence volume and fit quality do
# most of the ranking work; salary and momentum help break ties; diversity is a
# small nudge so the list is not redundant, not a reason to surface a bigger
# move by itself.
WEIGHT_OPPORTUNITY = 0.48
WEIGHT_QUALITY = 0.22
WEIGHT_SALARY = 0.13
WEIGHT_MOMENTUM = 0.12
WEIGHT_DIVERSITY = 0.05


def generate_skill_scenarios(
    request: CareerDeltaRequest,
    candidate_pool: CareerDeltaCandidatePool,
    baseline: BaselineMarketPosition,
    *,
    market_stats: MarketStatsProvider,
    search_scoring: SearchScoringProvider,
) -> tuple[ScenarioSummary, ...]:
    """Generate bounded low-pivot skill scenarios from the baseline evidence."""
    current_skills = _normalize_skill_inventory(tuple(request.current_skills) + tuple(candidate_pool.extracted_skills))
    if not current_skills:
        return ()
    analysis_set = _analysis_candidates(candidate_pool)
    if not analysis_set:
        return ()

    addition_summaries = _generate_skill_addition_scenarios(
        current_skills=current_skills,
        analysis_set=analysis_set,
        baseline=baseline,
        market_stats=market_stats,
    )
    substitution_summaries = _generate_skill_substitution_scenarios(
        current_skills=current_skills,
        analysis_set=analysis_set,
        baseline=baseline,
        market_stats=market_stats,
        search_scoring=search_scoring,
    )
    ranked = sorted(
        addition_summaries + substitution_summaries,
        key=lambda summary: (
            summary.confidence.score,
            summary.expected_salary_delta_pct or 0.0,
            summary.title,
        ),
        reverse=True,
    )
    return tuple(ranked)


def _generate_skill_addition_scenarios(
    *,
    current_skills: dict[str, str],
    analysis_set: list[CareerDeltaCandidate],
    baseline: BaselineMarketPosition,
    market_stats: MarketStatsProvider,
) -> list[ScenarioSummary]:
    support_counts: Counter[str] = Counter()
    display_names: dict[str, str] = {}
    total_jobs = len(analysis_set)
    for candidate in analysis_set:
        for skill in candidate.gap_skills:
            key = skill.strip().lower()
            if not key or key in current_skills:
                continue
            support_counts[key] += 1
            display_names.setdefault(key, skill)

    scored: list[tuple[float, ScenarioSummary]] = []
    for skill_key, supporting_jobs in support_counts.items():
        support_share = supporting_jobs / total_jobs if total_jobs else 0.0
        if supporting_jobs < MIN_SKILL_SUPPORTING_JOBS or support_share < MIN_SKILL_SUPPORT_SHARE:
            continue

        display_skill = display_names[skill_key]
        market = market_stats.get_skill_stats(display_skill)
        salary_lift_pct = _salary_lift_pct(
            market.median_salary_annual,
            baseline.salary_band.median_annual,
        )
        signal = SkillScenarioSignal(
            skill=display_skill,
            supporting_jobs=supporting_jobs,
            supporting_share_pct=round(support_share * 100, 2),
            market_job_count=market.job_count,
            market_salary_annual_median=market.median_salary_annual,
            market_momentum=market.momentum,
            salary_lift_pct=salary_lift_pct,
        )
        confidence = ScenarioConfidence(
            score=_bounded_score(
                0.35
                + (support_share * 0.35)
                + (_positive_ratio(salary_lift_pct, 0.25) * 0.15)
                + (_positive_ratio(market.momentum, 0.2) * 0.15)
            ),
            evidence_coverage=round(support_share, 4),
            market_sample_size=market.job_count,
            reasons=(
                f"{supporting_jobs} reachable jobs mention {display_skill}.",
                _format_market_reason(market, salary_lift_pct),
            ),
        )
        market_position = _market_position_from_signal(
            support_share=support_share,
            momentum=market.momentum,
            salary_lift_pct=salary_lift_pct,
        )
        scenario = ScenarioSummary(
            scenario_id=build_scenario_id(
                ScenarioType.SKILL_ADDITION,
                target_title_family=skill_key,
                market_position=market_position,
            ),
            scenario_type=ScenarioType.SKILL_ADDITION,
            title=f"Add {display_skill}",
            summary=(
                f"{display_skill} appears repeatedly in reachable jobs and can improve access "
                "without a large role change."
            ),
            market_position=market_position,
            confidence=confidence,
            change=ScenarioChange(added_skills=(display_skill,)),
            signals=(signal,),
            thin_market=baseline.thin_market,
            degraded=baseline.degraded,
            expected_salary_delta_pct=salary_lift_pct,
        )
        score = confidence.score + (salary_lift_pct or 0.0) + (market.momentum or 0.0)
        scored.append((score, scenario))

    scored.sort(key=lambda item: (item[0], item[1].title), reverse=True)
    return [summary for _, summary in scored[:MAX_SKILL_ADDITION_SCENARIOS]]


def _generate_skill_substitution_scenarios(
    *,
    current_skills: dict[str, str],
    analysis_set: list[CareerDeltaCandidate],
    baseline: BaselineMarketPosition,
    market_stats: MarketStatsProvider,
    search_scoring: SearchScoringProvider,
) -> list[ScenarioSummary]:
    if not current_skills:
        return []

    skill_presence: Counter[str] = Counter()
    display_names: dict[str, str] = {}
    for candidate in analysis_set:
        for skill in candidate.skills:
            key = skill.strip().lower()
            if not key:
                continue
            skill_presence[key] += 1
            display_names.setdefault(key, skill)

    scored: list[tuple[float, ScenarioSummary]] = []
    for source_key, source_skill in sorted(current_skills.items()):
        related_payload = search_scoring.get_related_skills(source_skill, k=RELATED_SKILL_LIMIT) or ()
        source_market = market_stats.get_skill_stats(source_skill)
        for related in related_payload:
            target_skill = str(related.get("skill", "")).strip()
            if not target_skill:
                continue
            target_key = target_skill.lower()
            if target_key in current_skills:
                continue

            similarity = float(related.get("similarity", 0.0) or 0.0)
            same_cluster = bool(related.get("same_cluster", False))
            if not same_cluster and similarity < MIN_SUBSTITUTION_SIMILARITY:
                continue

            target_market = market_stats.get_skill_stats(target_skill)
            if source_market.job_count <= 0 or target_market.job_count <= 0:
                continue

            job_count_ratio = target_market.job_count / max(source_market.job_count, 1)
            momentum_delta = (target_market.momentum or 0.0) - (source_market.momentum or 0.0)
            salary_lift_pct = _salary_lift_pct(
                target_market.median_salary_annual,
                baseline.salary_band.median_annual,
            )
            target_supporting_jobs = skill_presence.get(target_key, 0)
            supporting_share = target_supporting_jobs / len(analysis_set)
            if target_supporting_jobs < MIN_SKILL_SUPPORTING_JOBS:
                continue
            if job_count_ratio < MIN_SUBSTITUTION_JOB_COUNT_RATIO:
                continue
            if (
                momentum_delta < MIN_SUBSTITUTION_MOMENTUM_DELTA
                and (salary_lift_pct or 0.0) < MIN_SUBSTITUTION_SALARY_LIFT_PCT
            ):
                continue

            signal = SkillScenarioSignal(
                skill=target_skill,
                supporting_jobs=target_supporting_jobs,
                supporting_share_pct=round(supporting_share * 100, 2),
                market_job_count=target_market.job_count,
                market_salary_annual_median=target_market.median_salary_annual,
                market_momentum=target_market.momentum,
                salary_lift_pct=salary_lift_pct,
                similarity=similarity,
                same_cluster=same_cluster,
            )
            confidence = ScenarioConfidence(
                score=_bounded_score(
                    0.4
                    + (_positive_ratio(similarity, 1.0) * 0.2)
                    + (_positive_ratio(job_count_ratio - 1.0, 0.8) * 0.15)
                    + (_positive_ratio(momentum_delta, 0.2) * 0.15)
                    + (_positive_ratio(supporting_share, 0.5) * 0.1)
                ),
                evidence_coverage=round(supporting_share, 4),
                market_sample_size=target_market.job_count,
                reasons=(
                    f"{target_skill} is adjacent to {source_skill}.",
                    (
                        f"{target_skill} shows stronger demand than {source_skill} "
                        f"({target_market.job_count} vs {source_market.job_count} jobs)."
                    ),
                ),
            )
            market_position = _market_position_from_signal(
                support_share=supporting_share,
                momentum=target_market.momentum,
                salary_lift_pct=salary_lift_pct,
            )
            scenario = ScenarioSummary(
                scenario_id=build_scenario_id(
                    ScenarioType.SKILL_SUBSTITUTION,
                    source_title_family=source_key,
                    target_title_family=target_key,
                    market_position=market_position,
                ),
                scenario_type=ScenarioType.SKILL_SUBSTITUTION,
                title=f"Shift {source_skill} toward {target_skill}",
                summary=(
                    f"{target_skill} is a close neighbor to {source_skill} with better current "
                    "market support in the reachable pool."
                ),
                market_position=market_position,
                confidence=confidence,
                change=ScenarioChange(
                    added_skills=(target_skill,),
                    removed_skills=(source_skill,),
                    replaced_skills=(SkillReplacement(from_skill=source_skill, to_skill=target_skill),),
                ),
                signals=(signal,),
                thin_market=baseline.thin_market,
                degraded=baseline.degraded,
                expected_salary_delta_pct=salary_lift_pct,
            )
            score = confidence.score + (salary_lift_pct or 0.0) + momentum_delta
            scored.append((score, scenario))

    deduped = _dedupe_substitution_summaries(scored)
    deduped.sort(key=lambda item: (item[0], item[1].title), reverse=True)
    return [summary for _, summary in deduped[:MAX_SKILL_SUBSTITUTION_SCENARIOS]]


def generate_pivot_scenarios(
    request: CareerDeltaRequest,
    candidate_pool: CareerDeltaCandidatePool,
    baseline: BaselineMarketPosition,
    *,
    market_snapshot: dict,
    market_stats: MarketStatsProvider,
) -> tuple[tuple[ScenarioSummary, ...], tuple[FilteredScenario, ...]]:
    """Generate title and industry pivot scenarios from the shared candidate pool."""
    analysis_set = _analysis_candidates(candidate_pool)
    if not analysis_set:
        return (), ()

    source_title = _resolve_source_title(request, analysis_set)
    source_title_family = _resolve_source_title_family(request, analysis_set)
    current_industry_key = getattr(market_snapshot.get("current_industry"), "key", None)
    current_industry_key = (
        current_industry_key if current_industry_key and not current_industry_key.startswith("unknown/") else None
    )

    title_summaries = _generate_title_pivot_scenarios(
        source_title=source_title,
        source_title_family=source_title_family,
        current_industry_key=current_industry_key,
        analysis_set=analysis_set,
        baseline=baseline,
        market_snapshot=market_snapshot,
        market_stats=market_stats,
    )
    same_role_summaries = _generate_same_role_industry_pivots(
        source_title_family=source_title_family,
        current_industry_key=current_industry_key,
        analysis_set=analysis_set,
        baseline=baseline,
        market_stats=market_stats,
    )
    adjacent_summaries, filtered = _generate_adjacent_role_industry_pivots(
        source_title=source_title,
        source_title_family=source_title_family,
        current_industry_key=current_industry_key,
        analysis_set=analysis_set,
        baseline=baseline,
        market_stats=market_stats,
    )
    return (
        _rank_summaries(tuple(title_summaries + same_role_summaries + adjacent_summaries)),
        tuple(filtered),
    )


def _generate_title_pivot_scenarios(
    *,
    source_title: Optional[str],
    source_title_family: Optional[str],
    current_industry_key: Optional[str],
    analysis_set: list[CareerDeltaCandidate],
    baseline: BaselineMarketPosition,
    market_snapshot: dict,
    market_stats: MarketStatsProvider,
) -> list[ScenarioSummary]:
    if not source_title_family:
        return []

    grouped = _group_title_pivot_candidates(
        source_title_family=source_title_family,
        current_industry_key=current_industry_key,
        analysis_set=analysis_set,
    )
    current_title_market = market_snapshot.get("current_title_family") or market_stats.get_title_family_stats(
        source_title_family
    )
    scored: list[tuple[float, ScenarioSummary]] = []
    for target_title_family, group in grouped.items():
        dominant_title = _dominant_title(group)
        source_probe = source_title or source_title_family
        if not dominant_title or not is_adjacent_role(source_probe, dominant_title):
            continue

        supporting_jobs = len(group)
        support_share = supporting_jobs / len(analysis_set)
        if supporting_jobs < MIN_PIVOT_SUPPORTING_JOBS or support_share < MIN_PIVOT_SUPPORT_SHARE:
            continue

        fit_median = _quantile([candidate.overall_fit for candidate in group], 0.5)
        if fit_median < MIN_TITLE_PIVOT_FIT:
            continue

        target_market = market_stats.get_title_family_stats(target_title_family)
        salary_lift_pct = _salary_lift_pct(
            target_market.median_salary_annual,
            baseline.salary_band.median_annual,
        )
        if not _has_material_improvement(
            current_title_market,
            target_market,
            salary_lift_pct,
            threshold=MIN_TITLE_MARKET_IMPROVEMENT,
        ):
            continue

        dominant_industry_key = _dominant_industry_key(group)
        signal = PivotScenarioSignal(
            supporting_jobs=supporting_jobs,
            supporting_share_pct=round(support_share * 100, 2),
            target_title_family=target_title_family,
            target_industry=dominant_industry_key or "",
            title_distance="adjacent",
            industry_distance=0 if current_industry_key else None,
            fit_median=fit_median,
            market_job_count=target_market.job_count,
            market_salary_annual_median=target_market.median_salary_annual,
            market_momentum=target_market.momentum,
            salary_lift_pct=salary_lift_pct,
        )
        confidence = ScenarioConfidence(
            score=_bounded_score(
                0.35
                + (_positive_ratio(support_share, 0.5) * 0.2)
                + (_positive_ratio(fit_median - 0.5, 0.4) * 0.2)
                + (_positive_ratio(salary_lift_pct, 0.25) * 0.1)
                + (
                    _positive_ratio(
                        _market_delta(target_market.momentum, current_title_market.momentum),
                        0.2,
                    )
                    * 0.15
                )
            ),
            evidence_coverage=round(support_share, 4),
            market_sample_size=target_market.job_count,
            reasons=(
                f"{supporting_jobs} reachable jobs cluster around {dominant_title}.",
                _format_market_reason(target_market, salary_lift_pct),
            ),
        )
        market_position = _market_position_from_signal(
            support_share=support_share,
            momentum=target_market.momentum,
            salary_lift_pct=salary_lift_pct,
        )
        scenario = ScenarioSummary(
            scenario_id=build_scenario_id(
                ScenarioType.TITLE_PIVOT,
                source_title_family=source_title_family,
                target_title_family=target_title_family,
                target_sector=current_industry_key or dominant_industry_key,
                market_position=market_position,
            ),
            scenario_type=ScenarioType.TITLE_PIVOT,
            title=f"Pivot toward {dominant_title}",
            summary=(
                f"{dominant_title} appears as a nearby role family in the current market "
                "without requiring a sector change."
            ),
            market_position=market_position,
            confidence=confidence,
            change=ScenarioChange(
                source_title_family=source_title_family,
                target_title_family=target_title_family,
                source_industry=current_industry_key,
                target_industry=dominant_industry_key,
            ),
            signals=(signal,),
            target_title=dominant_title,
            target_sector=current_industry_key or dominant_industry_key,
            thin_market=baseline.thin_market,
            degraded=baseline.degraded,
            expected_salary_delta_pct=salary_lift_pct,
        )
        scored.append((confidence.score + fit_median + (salary_lift_pct or 0.0), scenario))

    scored.sort(key=lambda item: (item[0], item[1].title), reverse=True)
    return [summary for _, summary in scored[:MAX_TITLE_PIVOT_SCENARIOS]]


def _generate_same_role_industry_pivots(
    *,
    source_title_family: Optional[str],
    current_industry_key: Optional[str],
    analysis_set: list[CareerDeltaCandidate],
    baseline: BaselineMarketPosition,
    market_stats: MarketStatsProvider,
) -> list[ScenarioSummary]:
    if not source_title_family or not current_industry_key:
        return []

    current_industry_market = market_stats.get_industry_stats(current_industry_key)
    grouped = _group_candidates(
        candidate
        for candidate in analysis_set
        if candidate.title_family == source_title_family
        and candidate.industry_key != current_industry_key
        and not candidate.industry_key.startswith("unknown/")
    )
    scored: list[tuple[float, ScenarioSummary]] = []
    for (target_industry_key, _, target_label), group in grouped.items():
        supporting_jobs = len(group)
        support_share = supporting_jobs / len(analysis_set)
        fit_median = _quantile([candidate.overall_fit for candidate in group], 0.5)
        if (
            supporting_jobs < MIN_PIVOT_SUPPORTING_JOBS
            or support_share < MIN_PIVOT_SUPPORT_SHARE
            or fit_median < MIN_INDUSTRY_PIVOT_FIT
        ):
            continue

        target_market = market_stats.get_industry_stats(target_industry_key)
        salary_lift_pct = _salary_lift_pct(
            target_market.median_salary_annual,
            baseline.salary_band.median_annual,
        )
        if not _has_material_improvement(
            current_industry_market,
            target_market,
            salary_lift_pct,
            threshold=MIN_INDUSTRY_MARKET_IMPROVEMENT,
        ):
            continue

        distance = industry_distance(
            _industry_from_key(current_industry_key),
            _industry_from_key(target_industry_key),
        )
        signal = PivotScenarioSignal(
            supporting_jobs=supporting_jobs,
            supporting_share_pct=round(support_share * 100, 2),
            target_title_family=source_title_family,
            target_industry=target_industry_key,
            title_distance="same",
            industry_distance=distance,
            fit_median=fit_median,
            market_job_count=target_market.job_count,
            market_salary_annual_median=target_market.median_salary_annual,
            market_momentum=target_market.momentum,
            salary_lift_pct=salary_lift_pct,
        )
        confidence = ScenarioConfidence(
            score=_bounded_score(
                0.35
                + (_positive_ratio(support_share, 0.5) * 0.15)
                + (_positive_ratio(fit_median - 0.5, 0.4) * 0.2)
                + (_positive_ratio(salary_lift_pct, 0.25) * 0.15)
                + (
                    _positive_ratio(
                        _market_delta(target_market.momentum, current_industry_market.momentum),
                        0.2,
                    )
                    * 0.15
                )
            ),
            evidence_coverage=round(support_share, 4),
            market_sample_size=target_market.job_count,
            reasons=(
                f"{supporting_jobs} reachable jobs keep the same role family in {target_label}.",
                _format_market_reason(target_market, salary_lift_pct),
            ),
        )
        market_position = _market_position_from_signal(
            support_share=support_share,
            momentum=target_market.momentum,
            salary_lift_pct=salary_lift_pct,
        )
        scenario = ScenarioSummary(
            scenario_id=build_scenario_id(
                ScenarioType.SAME_ROLE_INDUSTRY_PIVOT,
                source_title_family=source_title_family,
                target_title_family=source_title_family,
                target_sector=target_industry_key,
                market_position=market_position,
            ),
            scenario_type=ScenarioType.SAME_ROLE_INDUSTRY_PIVOT,
            title=f"Keep the role, pivot into {target_label}",
            summary="The same role family appears in a stronger industry bucket with grounded reachable demand.",
            market_position=market_position,
            confidence=confidence,
            change=ScenarioChange(
                source_title_family=source_title_family,
                target_title_family=source_title_family,
                source_industry=current_industry_key,
                target_industry=target_industry_key,
            ),
            signals=(signal,),
            target_sector=target_industry_key,
            thin_market=baseline.thin_market,
            degraded=baseline.degraded,
            expected_salary_delta_pct=salary_lift_pct,
        )
        scored.append((confidence.score + fit_median + (salary_lift_pct or 0.0), scenario))

    scored.sort(key=lambda item: (item[0], item[1].title), reverse=True)
    return [summary for _, summary in scored[:MAX_SAME_ROLE_INDUSTRY_PIVOT_SCENARIOS]]


def _generate_adjacent_role_industry_pivots(
    *,
    source_title: Optional[str],
    source_title_family: Optional[str],
    current_industry_key: Optional[str],
    analysis_set: list[CareerDeltaCandidate],
    baseline: BaselineMarketPosition,
    market_stats: MarketStatsProvider,
) -> tuple[list[ScenarioSummary], list[FilteredScenario]]:
    if not source_title_family or not current_industry_key:
        return [], []

    grouped = _group_candidates(
        candidate
        for candidate in analysis_set
        if candidate.title_family != source_title_family
        and candidate.industry_key != current_industry_key
        and not candidate.industry_key.startswith("unknown/")
    )
    current_industry_market = market_stats.get_industry_stats(current_industry_key)
    scored: list[tuple[float, ScenarioSummary]] = []
    filtered: list[FilteredScenario] = []
    for (target_industry_key, target_title_family, target_label), group in grouped.items():
        supporting_jobs = len(group)
        support_share = supporting_jobs / len(analysis_set)
        if supporting_jobs < MIN_PIVOT_SUPPORTING_JOBS or support_share < MIN_PIVOT_SUPPORT_SHARE:
            continue

        dominant_title = _dominant_title(group)
        source_probe = source_title or source_title_family
        is_adjacent = bool(dominant_title and is_adjacent_role(source_probe, dominant_title))
        distance = industry_distance(
            _industry_from_key(current_industry_key),
            _industry_from_key(target_industry_key),
        )
        confidence = ScenarioConfidence(
            score=_bounded_score(0.25 + (_positive_ratio(support_share, 0.5) * 0.2)),
            evidence_coverage=round(support_share, 4),
            market_sample_size=supporting_jobs,
            reasons=(f"{supporting_jobs} reachable jobs support this pivot path.",),
        )
        if not is_adjacent:
            filtered.append(
                build_filtered_scenario(
                    scenario_type=ScenarioType.ADJACENT_ROLE_INDUSTRY_PIVOT,
                    reason_code="title_distance_too_high",
                    explanation=(
                        f"{dominant_title or target_title_family} is not close enough to the current role family."
                    ),
                    confidence=confidence,
                    source_title_family=source_title_family,
                    target_title_family=target_title_family,
                    target_sector=target_industry_key,
                )
            )
            continue
        if distance > MAX_ADJACENT_INDUSTRY_DISTANCE:
            filtered.append(
                build_filtered_scenario(
                    scenario_type=ScenarioType.ADJACENT_ROLE_INDUSTRY_PIVOT,
                    reason_code="industry_distance_too_high",
                    explanation=(f"{target_label} is too far from the current industry for a low-cost adjacent pivot."),
                    confidence=confidence,
                    source_title_family=source_title_family,
                    target_title_family=target_title_family,
                    target_sector=target_industry_key,
                )
            )
            continue

        fit_median = _quantile([candidate.overall_fit for candidate in group], 0.5)
        if fit_median < MIN_INDUSTRY_PIVOT_FIT:
            continue

        target_market = market_stats.get_industry_stats(target_industry_key)
        salary_lift_pct = _salary_lift_pct(
            target_market.median_salary_annual,
            baseline.salary_band.median_annual,
        )
        if not _has_material_improvement(
            current_industry_market,
            target_market,
            salary_lift_pct,
            threshold=MIN_INDUSTRY_MARKET_IMPROVEMENT,
        ):
            continue

        signal = PivotScenarioSignal(
            supporting_jobs=supporting_jobs,
            supporting_share_pct=round(support_share * 100, 2),
            target_title_family=target_title_family,
            target_industry=target_industry_key,
            title_distance="adjacent",
            industry_distance=distance,
            fit_median=fit_median,
            market_job_count=target_market.job_count,
            market_salary_annual_median=target_market.median_salary_annual,
            market_momentum=target_market.momentum,
            salary_lift_pct=salary_lift_pct,
        )
        scenario = ScenarioSummary(
            scenario_id=build_scenario_id(
                ScenarioType.ADJACENT_ROLE_INDUSTRY_PIVOT,
                source_title_family=source_title_family,
                target_title_family=target_title_family,
                target_sector=target_industry_key,
                market_position=_market_position_from_signal(
                    support_share=support_share,
                    momentum=target_market.momentum,
                    salary_lift_pct=salary_lift_pct,
                ),
            ),
            scenario_type=ScenarioType.ADJACENT_ROLE_INDUSTRY_PIVOT,
            title=f"Move toward {dominant_title} in {target_label}",
            summary=(f"{dominant_title} appears as a bounded adjacent-role move in a stronger nearby industry bucket."),
            market_position=_market_position_from_signal(
                support_share=support_share,
                momentum=target_market.momentum,
                salary_lift_pct=salary_lift_pct,
            ),
            confidence=ScenarioConfidence(
                score=_bounded_score(
                    0.32
                    + (_positive_ratio(support_share, 0.5) * 0.15)
                    + (_positive_ratio(fit_median - 0.5, 0.4) * 0.18)
                    + (_positive_ratio(salary_lift_pct, 0.25) * 0.12)
                    + (
                        _positive_ratio(
                            _market_delta(target_market.momentum, current_industry_market.momentum),
                            0.2,
                        )
                        * 0.13
                    )
                ),
                evidence_coverage=round(support_share, 4),
                market_sample_size=target_market.job_count,
                reasons=(
                    f"{supporting_jobs} reachable jobs support {dominant_title} in {target_label}.",
                    _format_market_reason(target_market, salary_lift_pct),
                ),
            ),
            change=ScenarioChange(
                source_title_family=source_title_family,
                target_title_family=target_title_family,
                source_industry=current_industry_key,
                target_industry=target_industry_key,
            ),
            signals=(signal,),
            target_title=dominant_title,
            target_sector=target_industry_key,
            thin_market=baseline.thin_market,
            degraded=baseline.degraded,
            expected_salary_delta_pct=salary_lift_pct,
        )
        scored.append((scenario.confidence.score + fit_median + (salary_lift_pct or 0.0), scenario))

    scored.sort(key=lambda item: (item[0], item[1].title), reverse=True)
    filtered.sort(key=lambda item: (item.confidence.score, item.scenario_id), reverse=True)
    return (
        [summary for _, summary in scored[:MAX_ADJACENT_ROLE_INDUSTRY_PIVOT_SCENARIOS]],
        filtered[:3],
    )


def _group_candidates(candidates) -> dict[tuple[str, str, str], list[CareerDeltaCandidate]]:
    grouped: dict[tuple[str, str, str], list[CareerDeltaCandidate]] = {}
    for candidate in candidates:
        key = (candidate.industry_key, candidate.title_family, candidate.industry_label)
        grouped.setdefault(key, []).append(candidate)
    return grouped


def _group_title_pivot_candidates(
    *,
    source_title_family: str,
    current_industry_key: Optional[str],
    analysis_set: list[CareerDeltaCandidate],
) -> dict[str, list[CareerDeltaCandidate]]:
    grouped: dict[str, list[CareerDeltaCandidate]] = {}
    for candidate in analysis_set:
        if candidate.title_family == source_title_family:
            continue
        if current_industry_key is not None and candidate.industry_key != current_industry_key:
            continue
        grouped.setdefault(candidate.title_family, []).append(candidate)
    return grouped


def _analysis_candidates(candidate_pool: CareerDeltaCandidatePool) -> list[CareerDeltaCandidate]:
    reachable = [
        candidate for candidate in candidate_pool.candidates if candidate.overall_fit >= REACHABLE_FIT_THRESHOLD
    ]
    return reachable or list(candidate_pool.candidates[:20])


def _normalize_skill_inventory(skills: tuple[str, ...]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for skill in skills:
        cleaned = skill.strip()
        if cleaned:
            normalized.setdefault(cleaned.lower(), cleaned)
    return normalized


def _dominant_industry_key(candidates: list[CareerDeltaCandidate]) -> Optional[str]:
    if not candidates:
        return None
    counts = Counter(candidate.industry_key for candidate in candidates if candidate.industry_key)
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]


def _resolve_source_title(request: CareerDeltaRequest, analysis_set: list[CareerDeltaCandidate]) -> Optional[str]:
    if request.current_title:
        return request.current_title
    if analysis_set:
        return analysis_set[0].title
    return request.target_titles[0] if request.target_titles else None


def _resolve_source_title_family(
    request: CareerDeltaRequest,
    analysis_set: list[CareerDeltaCandidate],
) -> Optional[str]:
    if request.current_title:
        return _normalize_title_family_key(request.current_title)
    if analysis_set:
        return analysis_set[0].title_family
    if request.target_titles:
        return _normalize_title_family_key(request.target_titles[0])
    return None


def _salary_lift_pct(
    target_salary_annual: Optional[int],
    baseline_salary_annual: Optional[int],
) -> Optional[float]:
    if not target_salary_annual or not baseline_salary_annual or baseline_salary_annual <= 0:
        return None
    return round((target_salary_annual - baseline_salary_annual) / baseline_salary_annual, 4)


def _market_delta(target_value: Optional[float], source_value: Optional[float]) -> float:
    return round((target_value or 0.0) - (source_value or 0.0), 4)


def _positive_ratio(value: Optional[float], cap: float) -> float:
    if value is None or value <= 0 or cap <= 0:
        return 0.0
    return min(value / cap, 1.0)


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(value, 0.99)), 4)


def _format_market_reason(market, salary_lift_pct: Optional[float]) -> str:
    details = [f"{market.job_count} recent postings"]
    if market.momentum is not None:
        details.append(f"momentum {market.momentum:+.2f}")
    if salary_lift_pct is not None:
        details.append(f"salary delta {salary_lift_pct:+.0%}")
    return ", ".join(details)


def _market_position_from_signal(
    *,
    support_share: float,
    momentum: Optional[float],
    salary_lift_pct: Optional[float],
) -> MarketPosition:
    if support_share >= 0.35 and (momentum or 0.0) >= 0.08:
        return MarketPosition.COMPETITIVE
    if support_share >= 0.2 or (salary_lift_pct or 0.0) >= 0.1:
        return MarketPosition.STRETCH
    return MarketPosition.UNCLEAR


def _rank_summaries(summaries: tuple[ScenarioSummary, ...]) -> tuple[ScenarioSummary, ...]:
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (
                summary.score_breakdown.final_score if summary.score_breakdown else summary.confidence.score,
                summary.expected_salary_delta_pct or 0.0,
                summary.title,
            ),
            reverse=True,
        )
    )


def _dominant_title(candidates: list[CareerDeltaCandidate]) -> Optional[str]:
    if not candidates:
        return None
    counts = Counter(candidate.title for candidate in candidates if candidate.title)
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]


def _normalize_title_family_key(title: str) -> str:
    return normalize_title_family(title).canonical


def _industry_from_key(key: Optional[str]) -> IndustryClassification:
    if not key or "/" not in key:
        return IndustryClassification()
    sector, subsector = key.split("/", 1)
    return IndustryClassification(sector=sector, subsector=subsector)


def _has_material_improvement(
    current_market,
    target_market,
    salary_lift_pct: Optional[float],
    *,
    threshold: float,
) -> bool:
    count_delta = target_market.job_count - current_market.job_count
    count_ratio = target_market.job_count / max(current_market.job_count, 1)
    if count_delta >= MIN_MATERIAL_JOB_COUNT_DELTA and count_ratio >= MIN_MATERIAL_JOB_COUNT_RATIO:
        return True
    if _market_delta(target_market.momentum, current_market.momentum) >= threshold:
        return True
    return (salary_lift_pct or 0.0) >= threshold


def rank_and_filter_scenarios(
    summaries: tuple[ScenarioSummary, ...],
    *,
    baseline: BaselineMarketPosition,
    request: CareerDeltaRequest,
    budget: ComputeBudget,
    started_at: float,
    clock: Callable[[], float],
) -> tuple[tuple[ScenarioSummary, ...], tuple[FilteredScenario, ...], bool]:
    """Rank raw scenarios, prune duplicates, enforce diversity, and honor budget."""
    filtered: list[FilteredScenario] = []
    deduped = _dedupe_scenarios(summaries, filtered)
    ranked_candidates, budget_degraded = _score_scenarios(
        deduped,
        baseline=baseline,
        budget=budget,
        started_at=started_at,
        clock=clock,
        filtered=filtered,
    )
    diversified = _enforce_diversity(ranked_candidates, limit=request.limit)
    final_ranked = _rank_summaries(tuple(diversified[: request.limit]))
    return final_ranked, tuple(filtered[:3]), budget_degraded


def _dedupe_scenarios(
    summaries: tuple[ScenarioSummary, ...],
    filtered: list[FilteredScenario],
) -> list[ScenarioSummary]:
    exact_kept: list[ScenarioSummary] = []
    exact_by_fingerprint: dict[tuple[str, ...], ScenarioSummary] = {}
    for summary in summaries:
        fingerprint = _scenario_fingerprint(summary)
        current = exact_by_fingerprint.get(fingerprint)
        if current is None:
            exact_by_fingerprint[fingerprint] = summary
            exact_kept.append(summary)
            continue
        preferred, rejected = _prefer_actionable(current, summary)
        exact_by_fingerprint[fingerprint] = preferred
        if rejected is current:
            exact_kept = [preferred if item is current else item for item in exact_kept]
        filtered.append(
            build_filtered_scenario(
                scenario_type=rejected.scenario_type,
                reason_code="duplicate_scenario",
                explanation="A lower-cost scenario with the same changes was kept instead.",
                confidence=rejected.confidence,
                source_title_family=rejected.change.source_title_family if rejected.change else None,
                target_title_family=rejected.change.target_title_family if rejected.change else None,
                target_sector=rejected.change.target_industry if rejected.change else rejected.target_sector,
                market_position=rejected.market_position,
            )
        )

    semantic_kept: list[ScenarioSummary] = []
    for summary in exact_kept:
        overlap_fingerprint = _scenario_overlap_fingerprint(summary)
        matched = next(
            (
                existing
                for existing in semantic_kept
                if _fingerprint_similarity(
                    overlap_fingerprint,
                    _scenario_overlap_fingerprint(existing),
                )
                >= SEMANTIC_DEDUP_THRESHOLD
            ),
            None,
        )
        if matched is None:
            semantic_kept.append(summary)
            continue
        preferred, rejected = _prefer_actionable(matched, summary)
        if rejected is matched:
            semantic_kept = [preferred if item is matched else item for item in semantic_kept]
        filtered.append(
            build_filtered_scenario(
                scenario_type=rejected.scenario_type,
                reason_code="overlapping_scenario",
                explanation="A materially overlapping scenario with lower pivot cost was kept instead.",
                confidence=rejected.confidence,
                source_title_family=rejected.change.source_title_family if rejected.change else None,
                target_title_family=rejected.change.target_title_family if rejected.change else None,
                target_sector=rejected.change.target_industry if rejected.change else rejected.target_sector,
                market_position=rejected.market_position,
            )
        )
    return semantic_kept


def _score_scenarios(
    summaries: list[ScenarioSummary],
    *,
    baseline: BaselineMarketPosition,
    budget: ComputeBudget,
    started_at: float,
    clock: Callable[[], float],
    filtered: list[FilteredScenario],
) -> tuple[list[ScenarioSummary], bool]:
    prioritized = sorted(
        summaries,
        key=lambda summary: (
            _scenario_type_priority(summary.scenario_type),
            -summary.confidence.score,
            summary.title,
        ),
    )
    evaluated: list[ScenarioSummary] = []
    metrics_by_id: dict[str, dict[str, float]] = {}
    budget_degraded = False
    for index, summary in enumerate(prioritized):
        if index >= budget.max_scenarios_evaluated or ((clock() - started_at) * 1000) >= budget.max_wall_time_ms:
            budget_degraded = True
            filtered.append(
                build_filtered_scenario(
                    scenario_type=summary.scenario_type,
                    reason_code="budget_exhausted",
                    explanation="Scenario evaluation stopped early to stay within the compute budget.",
                    confidence=summary.confidence,
                    source_title_family=summary.change.source_title_family if summary.change else None,
                    target_title_family=summary.change.target_title_family if summary.change else None,
                    target_sector=summary.change.target_industry if summary.change else summary.target_sector,
                    market_position=summary.market_position,
                )
            )
            continue
        signal_support = _scenario_supporting_jobs(summary)
        if signal_support < MIN_SCENARIO_SIGNAL:
            filtered.append(
                build_filtered_scenario(
                    scenario_type=summary.scenario_type,
                    reason_code="low_signal",
                    explanation="Scenario evidence is too sparse for a stable recommendation.",
                    confidence=summary.confidence,
                    source_title_family=summary.change.source_title_family if summary.change else None,
                    target_title_family=summary.change.target_title_family if summary.change else None,
                    target_sector=summary.change.target_industry if summary.change else summary.target_sector,
                    market_position=summary.market_position,
                )
            )
            continue
        evaluated.append(summary)
        metrics_by_id[summary.scenario_id] = _scenario_metrics(summary, baseline)

    if not evaluated:
        return [], budget_degraded

    normalized = _normalize_metrics(metrics_by_id)
    ranked: list[ScenarioSummary] = []
    for summary in evaluated:
        metrics = normalized[summary.scenario_id]
        pivot_cost = _estimate_pivot_cost(summary)
        raw_score = (
            WEIGHT_OPPORTUNITY * metrics["opportunity"]
            + WEIGHT_QUALITY * metrics["quality"]
            + WEIGHT_SALARY * metrics["salary"]
            + WEIGHT_MOMENTUM * metrics["momentum"]
            + WEIGHT_DIVERSITY * metrics["diversity"]
        )
        final_score = round(raw_score * (1.0 - pivot_cost), 4)
        score_breakdown = ScenarioScoreBreakdown(
            opportunity=metrics["opportunity"],
            quality=metrics["quality"],
            salary=metrics["salary"],
            momentum=metrics["momentum"],
            diversity=metrics["diversity"],
            raw_score=round(raw_score, 4),
            pivot_cost=pivot_cost,
            final_score=final_score,
        )
        confidence = ScenarioConfidence(
            score=final_score,
            evidence_coverage=summary.confidence.evidence_coverage,
            market_sample_size=summary.confidence.market_sample_size,
            reasons=summary.confidence.reasons + (f"pivot cost {pivot_cost:.2f}",),
        )
        ranked.append(
            ScenarioSummary(
                scenario_id=summary.scenario_id,
                scenario_type=summary.scenario_type,
                title=summary.title,
                summary=summary.summary,
                market_position=summary.market_position,
                confidence=confidence,
                score_breakdown=score_breakdown,
                change=summary.change,
                signals=summary.signals,
                target_title=summary.target_title,
                target_sector=summary.target_sector,
                thin_market=summary.thin_market,
                degraded=summary.degraded or budget_degraded,
                expected_salary_delta_pct=summary.expected_salary_delta_pct,
            )
        )
    return list(_rank_summaries(tuple(ranked))), budget_degraded


def _enforce_diversity(summaries: list[ScenarioSummary], *, limit: int) -> list[ScenarioSummary]:
    if limit <= 0:
        return []
    by_type: dict[ScenarioType, list[ScenarioSummary]] = {}
    for summary in summaries:
        by_type.setdefault(summary.scenario_type, []).append(summary)
    for bucket in by_type.values():
        bucket.sort(
            key=lambda item: (
                item.score_breakdown.final_score if item.score_breakdown else item.confidence.score,
                item.title,
            ),
            reverse=True,
        )
    selected: list[ScenarioSummary] = []
    for scenario_type in sorted(by_type.keys(), key=lambda item: _scenario_type_priority(item)):
        if len(selected) >= limit:
            break
        selected.append(by_type[scenario_type].pop(0))
    remaining = [
        item
        for bucket in by_type.values()
        for item in bucket
        if item.scenario_id not in {picked.scenario_id for picked in selected}
    ]
    remaining.sort(
        key=lambda item: (
            item.score_breakdown.final_score if item.score_breakdown else item.confidence.score,
            item.title,
        ),
        reverse=True,
    )
    for item in remaining:
        if len(selected) >= limit:
            break
        if item.scenario_id not in {picked.scenario_id for picked in selected}:
            selected.append(item)
    return selected


def _normalize_metrics(metrics_by_id: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = ("opportunity", "quality", "salary", "momentum", "diversity")
    normalized: dict[str, dict[str, float]] = {scenario_id: {} for scenario_id in metrics_by_id}
    for key in keys:
        ordered = sorted(
            ((scenario_id, metrics[key]) for scenario_id, metrics in metrics_by_id.items()),
            key=lambda item: (item[1], item[0]),
        )
        total = len(ordered)
        for index, (scenario_id, _) in enumerate(ordered):
            score = 1.0 if total == 1 else round(index / (total - 1), 4)
            normalized[scenario_id][key] = score
    return normalized


def _scenario_metrics(summary: ScenarioSummary, baseline: BaselineMarketPosition) -> dict[str, float]:
    supporting_jobs = float(_scenario_supporting_jobs(summary))
    fit_delta = max(_scenario_fit_median(summary) - baseline.fit_median, 0.0)
    salary = max(summary.expected_salary_delta_pct or 0.0, 0.0)
    momentum = max(_scenario_momentum(summary), 0.0)
    diversity = _scenario_diversity_gain(summary)
    return {
        "opportunity": supporting_jobs,
        "quality": fit_delta,
        "salary": salary,
        "momentum": momentum,
        "diversity": diversity,
    }


def _scenario_supporting_jobs(summary: ScenarioSummary) -> int:
    if not summary.signals:
        return 0
    return max(int(getattr(signal, "supporting_jobs", 0)) for signal in summary.signals)


def _scenario_fit_median(summary: ScenarioSummary) -> float:
    if not summary.signals:
        return summary.confidence.evidence_coverage
    return max(float(getattr(signal, "fit_median", summary.confidence.evidence_coverage)) for signal in summary.signals)


def _scenario_momentum(summary: ScenarioSummary) -> float:
    if not summary.signals:
        return 0.0
    return max(float(getattr(signal, "market_momentum", 0.0) or 0.0) for signal in summary.signals)


def _scenario_diversity_gain(summary: ScenarioSummary) -> float:
    if summary.change and summary.change.source_industry and summary.change.target_industry:
        return 1.0 if summary.change.source_industry != summary.change.target_industry else 0.4
    if summary.change and summary.change.target_title_family and summary.change.source_title_family:
        return 0.35 if summary.change.target_title_family != summary.change.source_title_family else 0.1
    return 0.1


def _scenario_type_priority(scenario_type: ScenarioType) -> int:
    priorities = {
        ScenarioType.SKILL_ADDITION: 0,
        ScenarioType.SKILL_SUBSTITUTION: 1,
        ScenarioType.TITLE_PIVOT: 2,
        ScenarioType.SAME_ROLE_INDUSTRY_PIVOT: 3,
        ScenarioType.ADJACENT_ROLE_INDUSTRY_PIVOT: 4,
    }
    return priorities.get(scenario_type, 9)


def _estimate_pivot_cost(summary: ScenarioSummary) -> float:
    if summary.scenario_type == ScenarioType.SKILL_ADDITION:
        skill_cost = 0.05 * len(summary.change.added_skills) if summary.change else 0.05
        return min(round(0.03 + skill_cost, 4), 0.95)
    if summary.scenario_type == ScenarioType.SKILL_SUBSTITUTION:
        similarity = (
            max(float(getattr(signal, "similarity", 0.0) or 0.0) for signal in summary.signals)
            if summary.signals
            else 0.0
        )
        return min(round(0.18 - (similarity * 0.08), 4), 0.95)
    if summary.scenario_type == ScenarioType.TITLE_PIVOT:
        return 0.46
    if summary.scenario_type == ScenarioType.SAME_ROLE_INDUSTRY_PIVOT:
        return min(round(0.14 + (_industry_distance_cost(summary) * 0.32), 4), 0.95)
    if summary.scenario_type == ScenarioType.ADJACENT_ROLE_INDUSTRY_PIVOT:
        return min(round(0.32 + (_industry_distance_cost(summary) * 0.38) + 0.14, 4), 0.95)
    return 0.4


def _industry_distance_cost(summary: ScenarioSummary) -> float:
    if not summary.signals:
        return 0.0
    distance = max(int(getattr(signal, "industry_distance", 0) or 0) for signal in summary.signals)
    return {0: 0.0, 1: 0.15, 2: 0.5, 3: 0.7}.get(distance, 0.7)


def _scenario_fingerprint(summary: ScenarioSummary) -> tuple[str, ...]:
    change = summary.change
    if change is None:
        return (summary.scenario_type.value, summary.scenario_id)
    parts = [summary.scenario_type.value]
    parts.extend(_scenario_overlap_fingerprint(summary))
    return tuple(sorted(parts))


def _scenario_overlap_fingerprint(summary: ScenarioSummary) -> tuple[str, ...]:
    change = summary.change
    if change is None:
        return (summary.scenario_id,)
    parts: list[str] = []
    parts.extend(f"add:{skill.lower()}" for skill in change.added_skills)
    parts.extend(f"remove:{skill.lower()}" for skill in change.removed_skills)
    parts.extend(
        f"replace:{replacement.from_skill.lower()}->{replacement.to_skill.lower()}"
        for replacement in change.replaced_skills
    )
    if change.source_title_family:
        parts.append(f"source_title:{change.source_title_family}")
    if change.target_title_family:
        parts.append(f"target_title:{change.target_title_family}")
    if change.source_industry:
        parts.append(f"source_industry:{change.source_industry}")
    if change.target_industry:
        parts.append(f"target_industry:{change.target_industry}")
    return tuple(sorted(parts))


def _fingerprint_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _prefer_actionable(left: ScenarioSummary, right: ScenarioSummary) -> tuple[ScenarioSummary, ScenarioSummary]:
    left_cost = _estimate_pivot_cost(left)
    right_cost = _estimate_pivot_cost(right)
    if left_cost != right_cost:
        return (left, right) if left_cost < right_cost else (right, left)
    if left.confidence.score != right.confidence.score:
        return (left, right) if left.confidence.score >= right.confidence.score else (right, left)
    return (left, right) if left.scenario_id <= right.scenario_id else (right, left)


def _dedupe_substitution_summaries(
    scored: list[tuple[float, ScenarioSummary]],
) -> list[tuple[float, ScenarioSummary]]:
    best_by_target: dict[str, tuple[float, ScenarioSummary]] = {}
    for item in scored:
        score, summary = item
        target = summary.change.added_skills[0].lower() if summary.change and summary.change.added_skills else ""
        current = best_by_target.get(target)
        if current is None or (score, summary.title) > (current[0], current[1].title):
            best_by_target[target] = item
    return list(best_by_target.values())
