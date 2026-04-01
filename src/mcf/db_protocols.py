"""
Capability-oriented database protocols.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import date
from typing import Any, Optional, Protocol

import numpy as np

from .models import Job


class ConnectionBackedDatabase(Protocol):
    def _connect(self, write_optimized: bool = False) -> Any: ...

    def _connection(self) -> AbstractContextManager[Any]: ...


class JobStore(Protocol):
    def upsert_job(self, job: Job, conn: Any | None = None) -> tuple[bool, bool]: ...

    def get_job(self, uuid: str) -> Optional[dict]: ...

    def get_jobs_bulk(self, uuids: list[str]) -> dict[str, dict]: ...

    def search_jobs(
        self,
        keyword: str | None = None,
        company_name: str | None = None,
        salary_min: int | None = None,
        salary_max: int | None = None,
        employment_type: str | None = None,
        region: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]: ...

    def count_jobs(self) -> int: ...

    def count_jobs_since(self, since: date) -> int: ...

    def get_all_uuids(self) -> set[str]: ...

    def get_all_uuids_since(self, since: date) -> set[str]: ...

    def has_job(self, uuid: str) -> bool: ...

    def get_stats(self) -> dict: ...


class EmbeddingStore(Protocol):
    def upsert_embedding(
        self,
        entity_id: str,
        entity_type: str,
        embedding: np.ndarray,
        model_version: str | None = None,
    ) -> None: ...

    def batch_upsert_embeddings(
        self,
        entity_ids: list[str],
        entity_type: str,
        embeddings: np.ndarray,
        model_version: str | None = None,
    ) -> int: ...

    def get_embedding(self, entity_id: str, entity_type: str) -> Optional[np.ndarray]: ...

    def get_all_embeddings(
        self,
        entity_type: str,
        model_version: str | None = None,
    ) -> tuple[list[str], np.ndarray]: ...

    def get_embeddings_for_uuids(
        self,
        uuids: list[str],
        model_version: str | None = None,
    ) -> dict[str, np.ndarray]: ...

    def get_embedding_stats(self) -> dict: ...

    def delete_embeddings_for_model(self, model_version: str) -> int: ...

    def get_jobs_without_embeddings(
        self,
        limit: int = 1000,
        since: date | None = None,
        model_version: str | None = None,
    ) -> list[dict]: ...

    def get_company_job_embeddings_bulk(self) -> dict[str, list[np.ndarray]]: ...


class AnalyticsStore(Protocol):
    def log_search(
        self,
        query: str,
        query_type: str,
        result_count: int,
        latency_ms: float,
        cache_hit: bool = False,
        degraded: bool = False,
        filters_used: dict | None = None,
    ) -> None: ...

    def get_popular_queries(self, days: int = 7, limit: int = 20) -> list[dict]: ...

    def get_search_latency_percentiles(self, days: int = 7) -> dict: ...

    def get_analytics_summary(self, days: int = 7) -> dict: ...


class ScrapeStateStore(Protocol):
    def create_session(self, search_query: str, total_jobs: int, session_id: int | None = None) -> int: ...

    def update_session(self, session_id: int, fetched_count: int, current_offset: int) -> None: ...

    def complete_session(self, session_id: int) -> None: ...

    def get_incomplete_session(self, search_query: str) -> Optional[dict]: ...

    def get_all_sessions(self, status: str | None = None) -> list[dict]: ...

    def clear_incomplete_sessions(self) -> int: ...

    def create_historical_session(
        self,
        year: int,
        start_seq: int,
        end_seq: int | None = None,
        session_id: int | None = None,
        conn: Any | None = None,
    ) -> int: ...

    def update_historical_progress(
        self,
        session_id: int,
        current_seq: int,
        jobs_found: int,
        jobs_not_found: int,
        consecutive_not_found: int = 0,
        end_seq: int | None = None,
        conn: Any | None = None,
    ) -> None: ...

    def complete_historical_session(self, session_id: int, conn: Any | None = None) -> None: ...

    def get_incomplete_historical_session(self, year: int) -> Optional[dict]: ...

    def get_all_historical_sessions(self, status: str | None = None) -> list[dict]: ...

    def clear_incomplete_historical_sessions(self) -> int: ...

    def get_historical_stats(self) -> dict: ...

    def batch_insert_attempts(self, attempts: list[dict], conn: Any | None = None) -> int: ...

    def get_missing_sequences(self, year: int) -> list[tuple[int, int]]: ...

    def get_failed_attempts(self, year: int, limit: int = 10000) -> list[dict]: ...

    def get_attempt_stats(self, year: int) -> dict: ...

    def get_all_attempt_stats(self) -> dict[int, dict]: ...

    def update_daemon_state(
        self,
        pid: int,
        status: str,
        current_year: int | None = None,
        current_seq: int | None = None,
    ) -> None: ...

    def update_daemon_heartbeat(
        self,
        current_year: int | None = None,
        current_seq: int | None = None,
    ) -> None: ...

    def get_daemon_state(self) -> dict: ...

    def clear_daemon_state(self) -> None: ...

    @staticmethod
    def can_acquire_write_lock(db_path: str, timeout_ms: int = 1000) -> bool: ...


class SearchSupportStore(Protocol):
    def bm25_search(self, query: str, limit: int = 100) -> list[tuple[str, float]]: ...

    def bm25_search_filtered(self, query: str, candidate_uuids: set[str]) -> list[tuple[str, float]]: ...

    def rebuild_fts_index(self) -> None: ...

    def get_all_companies(self) -> list[str]: ...

    def get_all_unique_companies(self) -> list[str]: ...

    def get_company_stats(self, company_name: str) -> dict: ...

    def get_all_unique_skills(self) -> list[str]: ...

    def get_skill_frequencies(self, min_jobs: int = 1, limit: int = 100) -> list[tuple[str, int]]: ...

    def get_skill_trends(
        self,
        skills: list[str],
        months: int = 12,
        company_name: str | None = None,
        employment_type: str | None = None,
        region: str | None = None,
    ) -> list[dict]: ...

    def get_role_trend(
        self,
        query: str,
        months: int = 12,
        company_name: str | None = None,
        employment_type: str | None = None,
        region: str | None = None,
    ) -> dict: ...

    def get_company_trend(self, company_name: str, months: int = 12) -> dict: ...

    def get_overview(self, months: int = 12) -> dict: ...


class DatabaseProtocol(
    ConnectionBackedDatabase,
    JobStore,
    EmbeddingStore,
    AnalyticsStore,
    ScrapeStateStore,
    SearchSupportStore,
    Protocol,
):
    """Aggregate runtime protocol for supported database backends."""
