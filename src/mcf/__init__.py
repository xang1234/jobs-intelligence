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
from .db_backup import BackupMetadata, create_sqlite_hot_backup, verify_sqlite_backup
from .db_factory import open_database
from .embeddings import EmbeddingGenerator, EmbeddingStats, SkillClusterResult
from .historical_scraper import YEAR_ESTIMATES, HistoricalScraper, ScrapeProgress
from .hosted_slice import DEFAULT_HOSTED_SLICE_POLICY, HostedSlicePolicy
from .market_stats import MarketAggregate, MarketStatsCache, MarketStatsSnapshot
from .migration import LegacyJobParser, MCFMigrator, MigrationStats
from .models import Job, JobSearchResponse
from .pg_database import PostgresDatabase
from .postgres_migration import (
    MigrationAnomaly,
    MigrationReport,
    audit_sqlite_source,
    migrate_sqlite_backup_to_postgres,
    purge_hosted_slice,
    seed_hosted_slice_from_postgres,
    write_migration_report,
)
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
    "PostgresDatabase",
    "open_database",
    # Migration
    "MCFMigrator",
    "MigrationStats",
    "LegacyJobParser",
    "MigrationAnomaly",
    "MigrationReport",
    "audit_sqlite_source",
    "migrate_sqlite_backup_to_postgres",
    "write_migration_report",
    "seed_hosted_slice_from_postgres",
    "purge_hosted_slice",
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
    "BackupMetadata",
    "create_sqlite_hot_backup",
    "verify_sqlite_backup",
    "HostedSlicePolicy",
    "DEFAULT_HOSTED_SLICE_POLICY",
]
__version__ = "1.4.0"
