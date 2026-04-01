"""
Vector backend protocol for search-time operations.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class VectorBackend(Protocol):
    def exists(self) -> bool: ...

    def load(self) -> bool: ...

    def search_jobs(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]: ...

    def search_jobs_filtered(
        self,
        query_vector: np.ndarray,
        candidate_uuids: list[str],
        k: int = 10,
    ) -> list[tuple[str, float]]: ...

    def total_jobs(self) -> int: ...

    def has_skill_index(self) -> bool: ...

    def get_skill_embedding(self, skill: str) -> Optional[np.ndarray]: ...

    def search_skills(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]: ...

    def has_company_index(self) -> bool: ...

    def get_company_centroids(self, company_name: str) -> Optional[np.ndarray]: ...

    def search_companies(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]: ...

    def get_stats(self) -> dict: ...
