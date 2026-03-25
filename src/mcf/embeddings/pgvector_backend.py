"""
pgvector-backed vector backend adapter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..hosted_slice import DEFAULT_HOSTED_SLICE_POLICY


class PGVectorBackend:
    def __init__(
        self,
        db,
        *,
        model_version: str,
        lean_hosted: bool = False,
    ):
        self.db = db
        self.model_version = model_version
        self.lean_hosted = lean_hosted

    def _vector_search_supported(self) -> bool:
        checker = getattr(self.db, "supports_vector_search", None)
        if checker is None:
            return True
        return bool(checker())

    def _job_embeddings_available(self) -> bool:
        stats = self.db.get_embedding_stats()
        return int(stats.get("job_embeddings", 0) or 0) > 0

    def exists(self) -> bool:
        return self._vector_search_supported() and self._job_embeddings_available()

    def load(self) -> bool:
        return self.exists()

    def _ensure_vector_search_supported(self) -> None:
        if not self._vector_search_supported():
            raise RuntimeError("pgvector backend requested but pgvector extension is unavailable")

    def search_jobs(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        self._ensure_vector_search_supported()
        return self.db.vector_search(
            entity_type="job",
            query_embedding=query_vector,
            limit=k,
            model_version=self.model_version,
        )

    def search_jobs_filtered(
        self,
        query_vector: np.ndarray,
        candidate_uuids: list[str],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        if not candidate_uuids:
            return []
        self._ensure_vector_search_supported()
        return self.db.vector_search(
            entity_type="job",
            query_embedding=query_vector,
            entity_ids=candidate_uuids,
            limit=k,
            model_version=self.model_version,
        )

    def total_jobs(self) -> int:
        return self.db.count_jobs()

    def has_skill_index(self) -> bool:
        return not self.lean_hosted

    def get_skill_embedding(self, skill: str) -> Optional[np.ndarray]:
        if not self.has_skill_index():
            return None
        return self.db.get_embedding(skill, "skill")

    def search_skills(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if not self.has_skill_index():
            return []
        self._ensure_vector_search_supported()
        return self.db.vector_search(
            entity_type="skill",
            query_embedding=query_vector,
            limit=k,
            model_version=self.model_version,
        )

    def has_company_index(self) -> bool:
        return not self.lean_hosted

    def get_company_centroids(self, company_name: str) -> Optional[np.ndarray]:
        if not self.has_company_index():
            return None
        self._ensure_vector_search_supported()
        rows = self.db.vector_search(
            entity_type="company",
            query_embedding=np.zeros(384, dtype=np.float32),
            limit=1000,
            model_version=self.model_version,
            prefix=f"{company_name}::centroid_",
        )
        if not rows:
            return None
        embeddings = []
        for entity_id, _ in rows:
            embedding = self.db.get_embedding(entity_id, "company")
            if embedding is not None:
                embeddings.append(embedding)
        if not embeddings:
            return None
        return np.asarray(embeddings, dtype=np.float32)

    def search_companies(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if not self.has_company_index():
            return []
        self._ensure_vector_search_supported()
        rows = self.db.vector_search(
            entity_type="company",
            query_embedding=query_vector,
            limit=max(k * 5, 25),
            model_version=self.model_version,
        )
        aggregated: dict[str, float] = {}
        for entity_id, score in rows:
            company_name = entity_id.split("::centroid_", 1)[0]
            if company_name not in aggregated or score > aggregated[company_name]:
                aggregated[company_name] = score
        return sorted(aggregated.items(), key=lambda item: item[1], reverse=True)[:k]

    def get_stats(self) -> dict:
        stats = self.db.get_embedding_stats()
        stats["hosted_cutoff"] = DEFAULT_HOSTED_SLICE_POLICY.cutoff_date().isoformat() if self.lean_hosted else None
        return stats
