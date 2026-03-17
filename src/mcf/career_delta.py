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

    def score_targets(self, request: CareerDeltaRequest) -> list[dict]: ...


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

        return CareerDeltaResponse(
            request=normalized_request,
            summaries=(),
            filtered_scenarios=(),
            degraded=degraded,
            thin_market=False,
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
