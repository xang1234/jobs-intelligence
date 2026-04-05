"""
Hosted-slice rules for lean Neon deployments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

HOSTED_MAX_AGE_DAYS = 90


@dataclass(frozen=True)
class HostedSlicePolicy:
    min_posted_date: date | None = None
    max_age_days: int = HOSTED_MAX_AGE_DAYS
    store_job_embeddings_only: bool = True

    def cutoff_date(self, today: date | None = None) -> date:
        current = today or date.today()
        year_floor = self.min_posted_date or date(current.year, 1, 1)
        rolling_cutoff = current - timedelta(days=self.max_age_days)
        return max(year_floor, rolling_cutoff)

    def include_posted_date(self, posted_date: date | None, today: date | None = None) -> bool:
        if posted_date is None:
            return False
        return posted_date >= self.cutoff_date(today)


DEFAULT_HOSTED_SLICE_POLICY = HostedSlicePolicy()
