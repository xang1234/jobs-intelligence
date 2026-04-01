"""
Dual-write adapter for migration burn-in.
"""

from __future__ import annotations

from typing import Any

WRITE_METHODS = {
    "upsert_job",
    "populate_normalized_job_metadata",
    "create_session",
    "update_session",
    "complete_session",
    "clear_incomplete_sessions",
    "create_historical_session",
    "update_historical_progress",
    "complete_historical_session",
    "clear_incomplete_historical_sessions",
    "batch_insert_attempts",
    "update_daemon_state",
    "update_daemon_heartbeat",
    "clear_daemon_state",
    "upsert_embedding",
    "batch_upsert_embeddings",
    "delete_embeddings_for_model",
    "rebuild_fts_index",
    "log_search",
}


class DualWriteDatabase:
    """Proxy reads to a primary DB while mirroring writes to a secondary DB."""

    def __init__(self, primary: Any, secondary: Any):
        self.primary = primary
        self.secondary = secondary

    @staticmethod
    def _secondary_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        secondary_kwargs = dict(kwargs)
        secondary_kwargs.pop("conn", None)
        return secondary_kwargs

    def __getattr__(self, name: str) -> Any:
        primary_attr = getattr(self.primary, name)
        if name not in WRITE_METHODS or not callable(primary_attr):
            return primary_attr

        secondary_attr = getattr(self.secondary, name)

        if name in {"create_session", "create_historical_session"}:

            def _wrapped_create(*args: Any, **kwargs: Any) -> Any:
                result = primary_attr(*args, **kwargs)
                secondary_kwargs = self._secondary_kwargs(kwargs)
                secondary_kwargs["session_id"] = int(result)
                secondary_attr(*args, **secondary_kwargs)
                return result

            return _wrapped_create

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            result = primary_attr(*args, **kwargs)
            secondary_kwargs = self._secondary_kwargs(kwargs)
            secondary_attr(*args, **secondary_kwargs)
            return result

        return _wrapped
