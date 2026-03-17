"""
MCF Scraper - API-based job scraper for MyCareersFuture.sg

This package provides a fast, reliable way to scrape job listings
using the public MyCareersFuture API instead of browser automation.

Features:
- Fast async API-based scraping
- SQLite storage with automatic deduplication
- History tracking when jobs are updated
- Resume support for interrupted scrapes
- Per-ID attempt tracking for gap detection
- Adaptive rate limiting
- Daemon mode for long-running scrapes
- CSV/JSON export
"""

from .adaptive_rate import AdaptiveRateLimiter, RateState
from .api_client import MCFClient
from .batch_logger import BatchLogger
from .daemon import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_WAKE_THRESHOLD,
    DaemonAlreadyRunning,
    DaemonError,
    DaemonNotRunning,
    ScraperDaemon,
)
from .database import MCFDatabase
from .embeddings import EmbeddingGenerator, EmbeddingStats, SkillClusterResult
from .historical_scraper import YEAR_ESTIMATES, HistoricalScraper, ScrapeProgress
from .market_stats import MarketAggregate, MarketStatsCache, MarketStatsSnapshot
from .migration import LegacyJobParser, MCFMigrator, MigrationStats
from .models import Job, JobSearchResponse
from .scraper import MCFScraper
from .storage import JobStorage, SQLiteStorage

__all__ = [
    # Core models
    "Job",
    "JobSearchResponse",
    # API client
    "MCFClient",
    # Scrapers
    "MCFScraper",
    "HistoricalScraper",
    "ScrapeProgress",
    "YEAR_ESTIMATES",
    # Storage
    "JobStorage",
    "SQLiteStorage",
    "MCFDatabase",
    # Migration
    "MCFMigrator",
    "MigrationStats",
    "LegacyJobParser",
    "MarketStatsCache",
    "MarketStatsSnapshot",
    "MarketAggregate",
    # Robust pipeline components
    "BatchLogger",
    "AdaptiveRateLimiter",
    "RateState",
    "ScraperDaemon",
    "DaemonError",
    "DaemonAlreadyRunning",
    "DaemonNotRunning",
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DEFAULT_WAKE_THRESHOLD",
    # Embeddings
    "EmbeddingGenerator",
    "EmbeddingStats",
    "SkillClusterResult",
]
__version__ = "1.4.0"
