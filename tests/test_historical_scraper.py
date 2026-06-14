"""Tests for the historical scraper recovery path."""

import asyncio
import sqlite3

import pytest

from src.mcf import historical_scraper as historical_scraper_module
from src.mcf.api_client import MCFAPIError, MCFNotFoundError, MCFRateLimitError
from src.mcf.batch_logger import BatchLogger
from src.mcf.historical_scraper import HistoricalScraper

from .factories import generate_test_job


class FakeClient:
    """Minimal fake client for historical scraper tests."""

    def __init__(self, responses: dict[str, list[object]]):
        self.responses = {uuid: list(items) for uuid, items in responses.items()}
        self.requests_per_second = 2.0
        self.calls: list[str] = []

    async def get_job(self, uuid: str):
        self.calls.append(uuid)
        if uuid not in self.responses or not self.responses[uuid]:
            raise AssertionError(f"No fake response configured for {uuid}")

        outcome = self.responses[uuid].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def build_scraper(db_path, **kwargs) -> HistoricalScraper:
    """Create a scraper configured for local unit tests."""
    kwargs.setdefault("discover_bounds", False)
    kwargs.setdefault("cooldown_seconds", 0.0)
    return HistoricalScraper(
        str(db_path),
        **kwargs,
    )


async def attach_fake_client(scraper: HistoricalScraper, client: FakeClient) -> HistoricalScraper:
    """Attach a fake API client and long-lived write connection."""
    scraper._client = client
    scraper._write_conn = scraper.db._connect(write_optimized=True)
    scraper.batch_logger.conn = scraper._write_conn
    return scraper


async def close_scraper(scraper: HistoricalScraper) -> None:
    """Flush and close scraper resources created directly in tests."""
    scraper.batch_logger.flush()
    if scraper._write_conn:
        scraper._write_conn.commit()
        scraper._write_conn.close()
        scraper._write_conn = None
        scraper.batch_logger.conn = None


class TestHistoricalScraper:
    """Tests for rate-limit recovery and bound management."""

    def test_batch_logger_keeps_buffer_when_flush_fails(self):
        class FailingDB:
            def batch_insert_attempts(self, attempts, conn=None):
                raise sqlite3.OperationalError("connection is closed")

        logger = BatchLogger(FailingDB())
        logger.log(2026, 1, "found")

        with pytest.raises(sqlite3.OperationalError):
            logger.flush()

        assert logger.pending_count == 1
        logger._buffer.clear()

    def test_scrape_year_reopens_closed_write_connection(self, empty_db, monkeypatch):
        year = 2026
        scraper = build_scraper(empty_db.db_path, batch_size=10)

        seq1_uuid = scraper._job_uuid(year, 1)
        seq2_uuid = scraper._job_uuid(year, 2)
        job1 = generate_test_job(job_uuid=seq1_uuid)
        client = FakeClient(
            {
                seq1_uuid: [job1],
                seq2_uuid: [MCFNotFoundError("missing")],
            }
        )

        original_upsert_job = scraper.db.upsert_job
        attempts = 0

        def flaky_upsert_job(job, conn=None):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                conn.close()
                raise sqlite3.OperationalError("Cannot operate on a closed database.")
            return original_upsert_job(job, conn=conn)

        monkeypatch.setattr(scraper.db, "upsert_job", flaky_upsert_job)

        async def run():
            await attach_fake_client(scraper, client)
            try:
                return await scraper.scrape_year(year, start_seq=1, end_seq=2, resume=False)
            finally:
                await close_scraper(scraper)

        progress = asyncio.run(run())
        stats = empty_db.get_attempt_stats(year)

        assert attempts == 2
        assert progress.current_seq == 3
        assert progress.jobs_found == 1
        assert progress.jobs_not_found == 1
        assert stats["found"] == 1
        assert stats["not_found"] == 1
        assert empty_db.get_job(seq1_uuid) is not None
        assert client.calls == [seq1_uuid, seq2_uuid]

    def test_rate_limited_sequence_does_not_block_progress(self, empty_db):
        year = 2023
        scraper = build_scraper(empty_db.db_path, max_rate_limit_retries=2)

        seq1_uuid = scraper._job_uuid(year, 1)
        seq2_uuid = scraper._job_uuid(year, 2)
        seq3_uuid = scraper._job_uuid(year, 3)
        job2 = generate_test_job(job_uuid=seq2_uuid)

        client = FakeClient(
            {
                seq1_uuid: [
                    MCFRateLimitError("rate limited"),
                    MCFRateLimitError("rate limited again"),
                ],
                seq2_uuid: [job2],
                seq3_uuid: [MCFNotFoundError("missing")],
            }
        )

        async def run():
            await attach_fake_client(scraper, client)
            try:
                return await scraper.scrape_year(
                    year,
                    start_seq=1,
                    end_seq=3,
                    resume=False,
                )
            finally:
                await close_scraper(scraper)

        progress = asyncio.run(run())
        stats = empty_db.get_attempt_stats(year)
        failed = empty_db.get_failed_attempts(year)

        assert progress.current_seq == 4
        assert progress.jobs_found == 1
        assert progress.jobs_not_found == 1
        assert stats["rate_limited"] == 1
        assert stats["found"] == 1
        assert stats["not_found"] == 1
        assert failed[0]["sequence"] == 1
        assert empty_db.get_job(seq2_uuid) is not None
        assert client.calls == [seq1_uuid, seq1_uuid, seq2_uuid, seq3_uuid]

    def test_retry_gaps_recovers_rate_limited_sequences(self, empty_db):
        year = 2023
        scraper = build_scraper(empty_db.db_path, max_rate_limit_retries=2)

        seq1_uuid = scraper._job_uuid(year, 1)
        recovered_job = generate_test_job(job_uuid=seq1_uuid)

        empty_db.batch_insert_attempts(
            [
                {
                    "year": year,
                    "sequence": 1,
                    "result": "rate_limited",
                    "error_message": "rate_limited_after_2_retries",
                }
            ]
        )

        client = FakeClient({seq1_uuid: [recovered_job]})

        async def run():
            await attach_fake_client(scraper, client)
            try:
                return await scraper.retry_gaps(year)
            finally:
                await close_scraper(scraper)

        progress = asyncio.run(run())
        stats = empty_db.get_attempt_stats(year)

        assert progress.jobs_found == 1
        assert stats["found"] == 1
        assert stats.get("rate_limited", 0) == 0
        assert empty_db.get_failed_attempts(year) == []
        assert empty_db.get_job(seq1_uuid) is not None

    def test_discovered_bounds_limit_scan_and_persist(self, empty_db, monkeypatch):
        year = 2024
        scraper = build_scraper(empty_db.db_path, discover_bounds=True)

        seq1_uuid = scraper._job_uuid(year, 1)
        seq2_uuid = scraper._job_uuid(year, 2)
        seq3_uuid = scraper._job_uuid(year, 3)
        seq4_uuid = scraper._job_uuid(year, 4)
        seq5_uuid = scraper._job_uuid(year, 5)

        client = FakeClient(
            {
                seq1_uuid: [generate_test_job(job_uuid=seq1_uuid)],
                seq2_uuid: [MCFNotFoundError("missing")],
                seq3_uuid: [MCFNotFoundError("missing")],
                seq4_uuid: [MCFNotFoundError("missing")],
                seq5_uuid: [MCFNotFoundError("missing")],
            }
        )

        async def fake_find_year_bounds(requested_year: int) -> tuple[int, int]:
            assert requested_year == year
            return (1, 5)

        async def run():
            await attach_fake_client(scraper, client)
            monkeypatch.setattr(scraper, "find_year_bounds", fake_find_year_bounds)
            try:
                return await scraper.scrape_year(year, resume=False)
            finally:
                await close_scraper(scraper)

        progress = asyncio.run(run())
        sessions = empty_db.get_all_historical_sessions()

        assert progress.current_seq == 6
        assert sessions[0]["end_seq"] == 5
        assert client.calls == [seq1_uuid, seq2_uuid, seq3_uuid, seq4_uuid, seq5_uuid]

    def test_resume_uses_stored_bound_without_rediscovery(self, empty_db, monkeypatch):
        year = 2025
        scraper = build_scraper(empty_db.db_path, discover_bounds=True)
        session_id = empty_db.create_historical_session(year, 1, 5)
        empty_db.update_historical_progress(
            session_id,
            current_seq=2,
            jobs_found=1,
            jobs_not_found=1,
            consecutive_not_found=0,
            end_seq=5,
        )

        seq3_uuid = scraper._job_uuid(year, 3)
        seq4_uuid = scraper._job_uuid(year, 4)
        seq5_uuid = scraper._job_uuid(year, 5)
        client = FakeClient(
            {
                seq3_uuid: [generate_test_job(job_uuid=seq3_uuid)],
                seq4_uuid: [MCFNotFoundError("missing")],
                seq5_uuid: [MCFNotFoundError("missing")],
            }
        )

        async def fail_find_year_bounds(_: int) -> tuple[int, int]:
            raise AssertionError("find_year_bounds should not be called on resume")

        async def run():
            await attach_fake_client(scraper, client)
            monkeypatch.setattr(scraper, "find_year_bounds", fail_find_year_bounds)
            try:
                return await scraper.scrape_year(year, resume=True)
            finally:
                await close_scraper(scraper)

        progress = asyncio.run(run())
        sessions = empty_db.get_all_historical_sessions()

        assert progress.current_seq == 6
        assert sessions[0]["status"] == "completed"
        assert client.calls == [seq3_uuid, seq4_uuid, seq5_uuid]

    def test_bounds_discovery_falls_back_after_rate_limit_burst(self, empty_db, monkeypatch):
        year = 2026
        monkeypatch.setitem(historical_scraper_module.YEAR_ESTIMATES, year, 3)
        monkeypatch.setattr(historical_scraper_module, "BOUND_DISCOVERY_WINDOW", 1)
        monkeypatch.setattr(historical_scraper_module, "BOUND_DISCOVERY_STEP", 1)

        scraper = build_scraper(
            empty_db.db_path,
            discover_bounds=True,
            max_rate_limit_retries=2,
        )

        seq1_uuid = scraper._job_uuid(year, 1)
        seq2_uuid = scraper._job_uuid(year, 2)
        seq3_uuid = scraper._job_uuid(year, 3)
        client = FakeClient(
            {
                seq1_uuid: [generate_test_job(job_uuid=seq1_uuid)],
                seq2_uuid: [MCFNotFoundError("missing")],
                seq3_uuid: [
                    MCFRateLimitError("rate limited"),
                    MCFRateLimitError("rate limited again"),
                    MCFNotFoundError("missing"),
                ],
            }
        )

        async def run():
            await attach_fake_client(scraper, client)
            try:
                return await scraper.scrape_year(year, resume=False)
            finally:
                await close_scraper(scraper)

        progress = asyncio.run(run())
        sessions = empty_db.get_all_historical_sessions()

        assert progress.current_seq == 4
        assert sessions[0]["end_seq"] == 3
        assert client.calls == [seq3_uuid, seq3_uuid, seq1_uuid, seq2_uuid, seq3_uuid]

    def test_bounds_discovery_falls_back_after_api_error(self, empty_db, monkeypatch):
        year = 2027
        monkeypatch.setitem(historical_scraper_module.YEAR_ESTIMATES, year, 2)
        monkeypatch.setattr(historical_scraper_module, "BOUND_DISCOVERY_WINDOW", 1)
        monkeypatch.setattr(historical_scraper_module, "BOUND_DISCOVERY_STEP", 1)

        scraper = build_scraper(empty_db.db_path, discover_bounds=True)

        seq1_uuid = scraper._job_uuid(year, 1)
        seq2_uuid = scraper._job_uuid(year, 2)
        client = FakeClient(
            {
                seq1_uuid: [generate_test_job(job_uuid=seq1_uuid)],
                seq2_uuid: [
                    MCFAPIError("temporary 500"),
                    MCFNotFoundError("missing"),
                ],
            }
        )

        async def run():
            await attach_fake_client(scraper, client)
            try:
                return await scraper.scrape_year(year, resume=False)
            finally:
                await close_scraper(scraper)

        progress = asyncio.run(run())
        sessions = empty_db.get_all_historical_sessions()

        assert progress.current_seq == 3
        assert sessions[0]["end_seq"] == 2
        assert client.calls == [seq2_uuid, seq1_uuid, seq2_uuid]
