"""
TTL-backed market aggregate cache for career-delta scoring.

This module pre-aggregates the market signals that scenario generation needs
frequently: counts, salary medians, and momentum for skills, title families,
and industry buckets. The expensive work happens once per refresh so scoring
can use O(1) dictionary lookups after warmup.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

from .career_delta import CareerDeltaRequest
from .database import MCFDatabase
from .industry_taxonomy import (
    IndustryClassification,
    classify_industry,
    infer_company_dominant_industry,
    normalize_title_family,
)

NO_RECENT_JOBS = "no_recent_jobs"
SPARSE_SALARY_DATA = "sparse_salary_data"
MISSING_TREND_SERIES = "missing_trend_series"


@dataclass(frozen=True)
class MarketAggregate:
    """Latest aggregate for a single skill, title family, or industry bucket."""

    key: str
    label: str
    kind: str
    job_count: int = 0
    median_salary_annual: Optional[int] = None
    momentum: Optional[float] = None
    market_share: float = 0.0
    sample_months: int = 0
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketStatsSnapshot:
    """Materialized market aggregates for fast read paths."""

    months: int
    refreshed_at_epoch: float
    expires_at_epoch: float
    skills: dict[str, MarketAggregate]
    title_families: dict[str, MarketAggregate]
    industries: dict[str, MarketAggregate]
    company_industries: dict[str, IndustryClassification]


class MarketStatsCache:
    """
    Cache market aggregates with TTL-based refresh and explicit caveats.

    The cache keeps one in-memory snapshot. When stale, the next read refreshes
    it from recent job rows and reuses the database's existing series/momentum
    helpers to stay aligned with trend endpoints.
    """

    def __init__(
        self,
        db: MCFDatabase,
        *,
        ttl_seconds: int = 1800,
        months: int = 6,
        clock: Callable[[], float] = time.time,
    ):
        self.db = db
        self.ttl_seconds = ttl_seconds
        self.months = months
        self.clock = clock
        self._snapshot: Optional[MarketStatsSnapshot] = None

    def invalidate(self) -> None:
        """Drop the current snapshot so the next read forces a refresh."""
        self._snapshot = None

    def refresh(self, *, force: bool = False) -> MarketStatsSnapshot:
        """Refresh the snapshot when stale or when explicitly forced."""
        if not force and self._snapshot is not None and not self._is_stale(self._snapshot):
            return self._snapshot

        self._snapshot = self._build_snapshot()
        return self._snapshot

    def get_skill_stats(self, skill: str) -> MarketAggregate:
        """Get latest stats for a skill using case-insensitive lookup."""
        snapshot = self.refresh()
        return snapshot.skills.get(skill.strip().lower(), self._missing_aggregate(skill, "skill"))

    def get_title_family_stats(self, title_or_family: str) -> MarketAggregate:
        """Get latest stats for a normalized title family."""
        canonical = normalize_title_family(title_or_family).canonical
        snapshot = self.refresh()
        return snapshot.title_families.get(canonical, self._missing_aggregate(canonical, "title_family"))

    def get_industry_stats(self, industry: str | IndustryClassification) -> MarketAggregate:
        """Get latest stats for an industry classification or bucket key."""
        if isinstance(industry, IndustryClassification):
            key = self._industry_key(industry)
        else:
            key = industry.strip().lower()
        snapshot = self.refresh()
        return snapshot.industries.get(key, self._missing_aggregate(key, "industry"))

    def get_market_snapshot(self, request: CareerDeltaRequest) -> dict:
        """Return the request-relevant aggregates for the career-delta engine."""
        snapshot = self.refresh()

        current_title = request.current_title or ""
        target_titles = request.normalized_target_titles()
        company_industry = snapshot.company_industries.get((request.current_company or "").strip())
        direct_industry = classify_industry(request.current_categories)
        if not direct_industry.is_unknown:
            current_industry = direct_industry
        elif company_industry and not company_industry.is_unknown:
            current_industry = company_industry
        else:
            current_industry = classify_industry(
                skills=request.current_skills,
            )

        return {
            "refreshed_at_epoch": snapshot.refreshed_at_epoch,
            "expires_at_epoch": snapshot.expires_at_epoch,
            "skills": {skill: self.get_skill_stats(skill) for skill in request.current_skills},
            "current_title_family": self.get_title_family_stats(current_title) if current_title else None,
            "target_title_families": {title: self.get_title_family_stats(title) for title in target_titles},
            "current_industry": self.get_industry_stats(current_industry),
        }

    def _is_stale(self, snapshot: MarketStatsSnapshot) -> bool:
        return self.clock() >= snapshot.expires_at_epoch

    def _build_snapshot(self) -> MarketStatsSnapshot:
        now = self.clock()
        labels = self.db._month_labels(self.months)
        rows = self._fetch_recent_rows()

        market_counts: Counter[str] = Counter()
        skill_counts: dict[str, Counter[str]] = defaultdict(Counter)
        skill_salarys: dict[str, dict[str, list[int]]] = defaultdict(lambda: {label: [] for label in labels})
        title_counts: dict[str, Counter[str]] = defaultdict(Counter)
        title_salarys: dict[str, dict[str, list[int]]] = defaultdict(lambda: {label: [] for label in labels})
        industry_counts: dict[str, Counter[str]] = defaultdict(Counter)
        industry_salarys: dict[str, dict[str, list[int]]] = defaultdict(lambda: {label: [] for label in labels})
        industry_labels: dict[str, str] = {}
        company_direct_industries: dict[str, list[IndustryClassification]] = defaultdict(list)
        company_industries: dict[str, IndustryClassification] = {}

        for row in rows:
            company_name = (row["company_name"] or "").strip()
            direct = classify_industry(self._split_csv(row["categories"]))
            if company_name and not direct.is_unknown:
                company_direct_industries[company_name].append(direct)

        for company_name, classifications in company_direct_industries.items():
            inferred = infer_company_dominant_industry(classifications)
            if inferred is not None:
                company_industries[company_name] = inferred

        for row in rows:
            month = row["posted_date"][:7]
            if month not in labels:
                continue

            market_counts[month] += 1
            salary = self.db._salary_midpoint(row)
            skills = self._split_csv(row["skills"])

            for skill in skills:
                key = skill.lower()
                skill_counts[key][month] += 1
                if salary is not None:
                    skill_salarys[key][month].append(salary)

            title_family = normalize_title_family(row["title"] or "")
            title_counts[title_family.canonical][month] += 1
            if salary is not None:
                title_salarys[title_family.canonical][month].append(salary)

            company_name = (row["company_name"] or "").strip()
            classification = classify_industry(
                self._split_csv(row["categories"]),
                company_classifications=company_direct_industries.get(company_name, ()),
                skills=skills,
            )
            industry_key = self._industry_key(classification)
            industry_labels[industry_key] = self._industry_label(classification)
            industry_counts[industry_key][month] += 1
            if salary is not None:
                industry_salarys[industry_key][month].append(salary)

        market_counts_dict = {label: market_counts.get(label, 0) for label in labels}
        snapshot = MarketStatsSnapshot(
            months=self.months,
            refreshed_at_epoch=now,
            expires_at_epoch=now + self.ttl_seconds,
            skills={
                key: self._build_aggregate(key, key, "skill", counts, skill_salarys[key], labels, market_counts_dict)
                for key, counts in skill_counts.items()
            },
            title_families={
                key: self._build_aggregate(
                    key,
                    key,
                    "title_family",
                    counts,
                    title_salarys[key],
                    labels,
                    market_counts_dict,
                )
                for key, counts in title_counts.items()
            },
            industries={
                key: self._build_aggregate(
                    key,
                    industry_labels.get(key, key),
                    "industry",
                    counts,
                    industry_salarys[key],
                    labels,
                    market_counts_dict,
                )
                for key, counts in industry_counts.items()
            },
            company_industries=company_industries,
        )
        return snapshot

    def _fetch_recent_rows(self) -> list:
        start_month = self.db._subtract_months(date.today().replace(day=1), self.months - 1)
        with self.db._connection() as conn:
            return conn.execute(
                """
                SELECT posted_date, title, company_name, categories, skills,
                       salary_annual_min, salary_annual_max
                FROM jobs
                WHERE posted_date IS NOT NULL
                  AND posted_date >= ?
                ORDER BY posted_date ASC
                """,
                (start_month.isoformat(),),
            ).fetchall()

    def _build_aggregate(
        self,
        key: str,
        label: str,
        kind: str,
        month_counts: Counter[str],
        salary_buckets: dict[str, list[int]],
        labels: list[str],
        market_counts: dict[str, int],
    ) -> MarketAggregate:
        series = self.db._series_from_aggregates(month_counts, salary_buckets, labels, market_counts)
        latest = series[-1] if series else None

        caveats: list[str] = []
        if not series or all(point["job_count"] == 0 for point in series):
            caveats.append(NO_RECENT_JOBS)
            caveats.append(MISSING_TREND_SERIES)
        elif latest and latest["job_count"] == 0:
            caveats.append(NO_RECENT_JOBS)

        if latest and latest["median_salary_annual"] is None:
            caveats.append(SPARSE_SALARY_DATA)

        return MarketAggregate(
            key=key,
            label=label,
            kind=kind,
            job_count=latest["job_count"] if latest else 0,
            median_salary_annual=latest["median_salary_annual"] if latest else None,
            momentum=None if MISSING_TREND_SERIES in caveats else latest["momentum"],
            market_share=latest["market_share"] if latest else 0.0,
            sample_months=len(series),
            caveats=tuple(dict.fromkeys(caveats)),
        )

    @staticmethod
    def _split_csv(raw_value: Optional[str]) -> list[str]:
        if not raw_value:
            return []
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    @staticmethod
    def _industry_key(classification: IndustryClassification) -> str:
        return f"{classification.sector}/{classification.subsector}"

    @staticmethod
    def _industry_label(classification: IndustryClassification) -> str:
        return f"{classification.sector} / {classification.subsector}"

    def _missing_aggregate(self, key: str, kind: str) -> MarketAggregate:
        return MarketAggregate(
            key=key,
            label=key,
            kind=kind,
            sample_months=self.months,
            caveats=(NO_RECENT_JOBS, MISSING_TREND_SERIES, SPARSE_SALARY_DATA),
        )
