"""Short-lived response caching helpers for read-only API endpoints."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable
from typing import TypeVar

from cachetools import TTLCache
from pydantic import BaseModel

from ..mcf.embeddings import SemanticSearchEngine

PUBLIC_RESPONSE_CACHE_TTL_SECONDS = 300
PUBLIC_RESPONSE_CACHE_MAX_ENTRIES = 256

T = TypeVar("T")


def _get_public_response_cache(engine: SemanticSearchEngine) -> TTLCache:
    cached = getattr(engine, "_public_response_cache", None)
    if isinstance(cached, TTLCache):
        return cached

    cache = TTLCache(
        maxsize=PUBLIC_RESPONSE_CACHE_MAX_ENTRIES,
        ttl=PUBLIC_RESPONSE_CACHE_TTL_SECONDS,
    )
    setattr(engine, "_public_response_cache", cache)
    return cache


def _get_public_response_inflight(engine: SemanticSearchEngine) -> dict[str, asyncio.Task]:
    cached = getattr(engine, "_public_response_inflight", None)
    if isinstance(cached, dict):
        return cached

    inflight: dict[str, asyncio.Task] = {}
    setattr(engine, "_public_response_inflight", inflight)
    return inflight


def _clone_cached_response(value: T) -> T:
    """Return a defensive copy of a cached Pydantic/list/dict response."""
    if isinstance(value, BaseModel):
        return value.model_copy(deep=True)  # type: ignore[return-value]
    return copy.deepcopy(value)


async def cached_public_response(
    engine: SemanticSearchEngine,
    cache_key: str,
    producer: Callable[[], Awaitable[T]],
) -> T:
    cache = _get_public_response_cache(engine)
    cached = cache.get(cache_key)
    if cached is not None:
        return _clone_cached_response(cached)

    inflight = _get_public_response_inflight(engine)
    existing = inflight.get(cache_key)
    if existing is not None:
        return _clone_cached_response(await existing)

    async def produce_and_cache() -> T:
        response = await producer()
        cache[cache_key] = _clone_cached_response(response)
        return response

    task = asyncio.create_task(produce_and_cache())
    inflight[cache_key] = task
    try:
        return _clone_cached_response(await task)
    finally:
        if inflight.get(cache_key) is task:
            inflight.pop(cache_key, None)
