"""Short-lived response caching helpers for read-only API endpoints."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from cachetools import TTLCache
from fastapi import FastAPI, Request
from pydantic import BaseModel

from ..mcf.embeddings import SemanticSearchEngine

PUBLIC_RESPONSE_CACHE_TTL_SECONDS = 300
PUBLIC_RESPONSE_CACHE_MAX_ENTRIES = 256

# Read-only GET endpoints whose payloads are daily-stable. They get an HTTP
# Cache-Control header (browser/CDN reuse without a round trip). max-age aligns
# with the in-memory response cache above; stale-while-revalidate lets returning
# users repaint instantly while a fresh copy is fetched in the background.
CACHEABLE_GET_PREFIXES: tuple[str, ...] = (
    "/api/overview",
    "/api/stats",
    "/api/skills/cloud",
    "/api/skills/related",
    "/api/trends/companies",
    "/api/analytics/popular",
    "/api/analytics/performance",
)
PUBLIC_CACHE_CONTROL = "public, max-age=300, stale-while-revalidate=86400"


def is_cacheable_get(method: str, path: str, status_code: int) -> bool:
    """True for the read-only GET endpoints whose responses are safe to cache."""
    return method == "GET" and status_code == 200 and path.startswith(CACHEABLE_GET_PREFIXES)

T = TypeVar("T")
logger = logging.getLogger(__name__)


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


def _drop_inflight_when_done(
    inflight: dict[str, asyncio.Task],
    cache_key: str,
    task: asyncio.Task,
) -> None:
    if inflight.get(cache_key) is task:
        inflight.pop(cache_key, None)


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
        logger.debug("public response cache hit: %s", cache_key)
        return _clone_cached_response(cached)

    inflight = _get_public_response_inflight(engine)
    existing = inflight.get(cache_key)
    if existing is not None:
        logger.debug("public response cache coalesced with inflight fill: %s", cache_key)
        return _clone_cached_response(await asyncio.shield(existing))

    async def produce_and_cache() -> T:
        response = await producer()
        cache[cache_key] = _clone_cached_response(response)
        return response

    logger.debug("public response cache miss: %s", cache_key)
    task = asyncio.create_task(produce_and_cache())
    inflight[cache_key] = task
    task.add_done_callback(lambda done: _drop_inflight_when_done(inflight, cache_key, done))
    return _clone_cached_response(await asyncio.shield(task))


def install_cache_headers(app: FastAPI) -> None:
    """Register middleware that sets Cache-Control on cacheable read-only GETs."""

    @app.middleware("http")
    async def add_cache_headers(request: Request, call_next):
        response = await call_next(request)
        if is_cacheable_get(request.method, request.url.path, response.status_code):
            response.headers["Cache-Control"] = PUBLIC_CACHE_CONTROL
        return response
