"""
Historical job scraper for MyCareersFuture.

Enumerates and fetches all historical MCF jobs by generating jobPostId values
and converting them to UUIDs via MD5 hash. This allows scraping the complete
job database from 2019 to present.

UUID Discovery:
- UUID = MD5(jobPostId) where jobPostId format is MCF-{YEAR}-{7-digit sequence}
- API endpoint /v2/jobs/{uuid} returns full job data even for closed/historical jobs
- Job IDs are sequential per year (starting at 0000001)

Enhanced Features:
- Per-ID attempt tracking via BatchLogger for gap detection
- Adaptive rate limiting that backs off on 429s and recovers slowly
- Daemon mode support for long-running background operation
"""

import asyncio
import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from tenacity import RetryError

from .adaptive_rate import AdaptiveRateLimiter
from .api_client import MCFAPIError, MCFClient, MCFNotFoundError, MCFRateLimitError
from .batch_logger import BatchLogger
from .db_factory import open_database
from .models import Job

logger = logging.getLogger(__name__)

# Estimated max sequences per year (conservative estimates)
YEAR_ESTIMATES = {
    2019: 50_000,
    2020: 350_000,
    2021: 700_000,
    2022: 1_000_000,
    2023: 1_000_000,
    2024: 1_450_000,
    2025: 1_500_000,
    2026: 250_000,  # Growing
}

# Stop scanning after this many consecutive not-found responses
DEFAULT_NOT_FOUND_THRESHOLD = 1000
DEFAULT_MAX_RATE_LIMIT_RETRIES = 4
DEFAULT_COOLDOWN_SECONDS = 30.0
DEFAULT_RATE_LIMIT_COOLDOWN_THRESHOLD = 3
DEFAULT_BATCH_SIZE = 250
BOUND_DISCOVERY_WINDOW = 125
BOUND_DISCOVERY_STEP = 25
MAX_SEQUENCE = 9_999_999


@dataclass
class ScrapeProgress:
    """Progress information for a scraping session."""

    year: int
    current_seq: int
    jobs_found: int
    jobs_not_found: int
    consecutive_not_found: int
    start_seq: int
    end_seq: Optional[int]

    @property
    def total_processed(self) -> int:
        return self.jobs_found + self.jobs_not_found

    @property
    def success_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.jobs_found / self.total_processed * 100


class HistoricalScraper:
    """
    Enumerate and fetch all historical MCF jobs by jobPostId.

    This scraper generates possible job IDs (MCF-YYYY-NNNNNNN format),
    converts them to UUIDs via MD5 hash, and fetches each job from the API.

    Features:
    - Resume support via SQLite session tracking
    - Adaptive rate limiting with backoff on 429 errors
    - Per-ID attempt tracking for gap detection and retry
    - Skips already-fetched jobs (deduplication)
    - Detects end-of-year via consecutive not-found threshold

    Example:
        async with HistoricalScraper("data/mcf_jobs.db") as scraper:
            await scraper.scrape_year(2023)
    """

    def __init__(
        self,
        db_path: str = "data/mcf_jobs.db",
        requests_per_second: float = 2.0,
        not_found_threshold: int = DEFAULT_NOT_FOUND_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
        min_rps: float = 0.5,
        max_rps: float = 5.0,
        max_rate_limit_retries: int = DEFAULT_MAX_RATE_LIMIT_RETRIES,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        discover_bounds: bool = True,
    ):
        """
        Initialize the historical scraper.

        Args:
            db_path: Path to SQLite database
            requests_per_second: Initial rate limit for API requests
            not_found_threshold: Stop after this many consecutive not-found responses
            batch_size: Number of attempts to buffer before flushing to DB
            min_rps: Minimum requests per second (during heavy rate limiting)
            max_rps: Maximum requests per second (after recovery)
            max_rate_limit_retries: Per-sequence retry cap for 429s
            cooldown_seconds: Global cooldown after repeated 429s
            discover_bounds: Discover a tighter end bound before scanning
        """
        self.db = open_database(db_path)
        self.initial_rps = requests_per_second
        self.not_found_threshold = not_found_threshold
        self.max_rate_limit_retries = max_rate_limit_retries
        self.cooldown_seconds = cooldown_seconds
        self.discover_bounds = discover_bounds
        self._client: Optional[MCFClient] = None
        self._write_conn: Optional[sqlite3.Connection] = None
        self._global_rate_limit_streak = 0

        # New components for robust operation
        self.batch_logger = BatchLogger(self.db, batch_size=batch_size)
        self.rate_limiter = AdaptiveRateLimiter(
            initial_rps=requests_per_second,
            min_rps=min_rps,
            max_rps=max_rps,
        )

    async def __aenter__(self) -> "HistoricalScraper":
        """Async context manager entry."""
        self._client = MCFClient(requests_per_second=self.initial_rps)
        await self._client.__aenter__()
        self._write_conn = self.db._connect(write_optimized=True)
        self.batch_logger.conn = self._write_conn
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # Flush any pending batch logger entries
        if self._write_conn:
            try:
                self.batch_logger.flush()
                self._write_conn.commit()
            finally:
                self._write_conn.close()
                self._write_conn = None
                self.batch_logger.conn = None

        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    @staticmethod
    def job_id_to_uuid(job_id: str) -> str:
        """
        Convert MCF-YYYY-NNNNNNN to UUID (MD5 hash).

        Args:
            job_id: Job ID in format MCF-2023-0000001

        Returns:
            32-character hex string (MD5 hash)
        """
        return hashlib.md5(job_id.encode()).hexdigest()

    @staticmethod
    def format_job_id(year: int, sequence: int) -> str:
        """
        Format a job ID from year and sequence number.

        Args:
            year: Year (e.g., 2023)
            sequence: Sequence number (1-9999999)

        Returns:
            Job ID in format MCF-2023-0000001
        """
        return f"MCF-{year}-{sequence:07d}"

    @staticmethod
    def parse_job_id(job_id: str) -> tuple[int, int]:
        """
        Parse a job ID into year and sequence.

        Args:
            job_id: Job ID in format MCF-2023-0000001

        Returns:
            Tuple of (year, sequence)
        """
        parts = job_id.split("-")
        if len(parts) != 3 or parts[0] != "MCF":
            raise ValueError(f"Invalid job ID format: {job_id}")
        return int(parts[1]), int(parts[2])

    def _job_uuid(self, year: int, sequence: int) -> str:
        """Build the deterministic UUID for a historical job sequence."""
        return self.job_id_to_uuid(self.format_job_id(year, sequence))

    def _update_client_rate(self, new_rps: float) -> None:
        """Keep the API client aligned with the adaptive limiter state."""
        if self._client:
            self._client.requests_per_second = new_rps

    def _mark_progress_committed(self) -> None:
        """Commit pending writes on the long-lived scraper connection."""
        if self._write_conn:
            self._write_conn.commit()

    def _reset_rate_limit_streak(self) -> None:
        """Clear the consecutive global rate-limit streak after progress."""
        self._global_rate_limit_streak = 0

    async def _handle_rate_limit(
        self,
        year: int,
        sequence: int,
        retry_count: int,
        *,
        context: str,
        log_failure: bool = True,
    ) -> bool:
        """
        Back off on 429s and decide whether to retry the same sequence.

        Returns:
            True if the caller should retry the same sequence, else False.
        """
        new_rps = self.rate_limiter.on_rate_limited()
        self._update_client_rate(new_rps)
        self._global_rate_limit_streak += 1

        if self._global_rate_limit_streak >= DEFAULT_RATE_LIMIT_COOLDOWN_THRESHOLD:
            logger.warning(
                f"{context} at seq {sequence} triggered cooldown after "
                f"{self._global_rate_limit_streak} consecutive 429s; "
                f"sleeping {self.cooldown_seconds:.1f}s"
            )
            await asyncio.sleep(self.cooldown_seconds)
            self._global_rate_limit_streak = 0
        else:
            backoff_delay = 1.0 / new_rps + 1.0
            logger.warning(
                f"{context} at seq {sequence}, backing off {backoff_delay:.1f}s, "
                f"new rate: {new_rps:.2f} req/sec "
                f"(retry {retry_count}/{self.max_rate_limit_retries})"
            )
            await asyncio.sleep(backoff_delay)

        if retry_count < self.max_rate_limit_retries:
            return True

        message = f"rate_limited_after_{retry_count}_retries at {new_rps:.2f} req/sec"
        logger.warning(f"Skipping seq {sequence} for year {year} after {retry_count} 429s; recording for retry-gaps")
        if log_failure:
            self.batch_logger.log(year, sequence, "rate_limited", message)
        return False

    async def _probe_job_exists(self, year: int, sequence: int) -> Optional[bool]:
        """Check whether a historical sequence resolves to a real job."""
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        uuid = self._job_uuid(year, sequence)
        retry_count = 0

        while True:
            try:
                await self._client.get_job(uuid)
                self.rate_limiter.on_success()
                self._reset_rate_limit_streak()
                self._update_client_rate(self.rate_limiter.current_rps)
                return True
            except MCFNotFoundError:
                self.rate_limiter.on_success()
                self._reset_rate_limit_streak()
                self._update_client_rate(self.rate_limiter.current_rps)
                return False
            except MCFRateLimitError:
                retry_count += 1
                should_retry = await self._handle_rate_limit(
                    year,
                    sequence,
                    retry_count=retry_count,
                    context="Rate limited during bounds discovery",
                    log_failure=False,
                )
                if not should_retry:
                    logger.warning(
                        f"Bounds discovery for year {year} became inconclusive at seq "
                        f"{sequence} after repeated 429s; falling back to estimate"
                    )
                    return None
            except (MCFAPIError, RetryError) as e:
                logger.warning(
                    f"Bounds discovery for year {year} became inconclusive at seq "
                    f"{sequence}: {e}. Falling back to estimate"
                )
                return None

    async def _window_has_job(
        self,
        year: int,
        start_seq: int,
    ) -> tuple[Optional[bool], int]:
        """Probe a sparse window to see whether a region still contains jobs."""
        highest_found = 0
        inconclusive = False
        for seq in range(
            start_seq,
            min(start_seq + BOUND_DISCOVERY_WINDOW, MAX_SEQUENCE + 1),
            BOUND_DISCOVERY_STEP,
        ):
            probe_result = await self._probe_job_exists(year, seq)
            if probe_result is True:
                highest_found = seq
            elif probe_result is None:
                inconclusive = True

        if highest_found > 0:
            return True, highest_found
        if inconclusive:
            return None, 0
        return False, 0

    async def fetch_job(self, year: int, sequence: int) -> Optional[Job]:
        """
        Fetch a single job by year and sequence.

        Args:
            year: Year of the job
            sequence: Sequence number

        Returns:
            Job if found, None if not found

        Raises:
            MCFRateLimitError: If rate limited (caller should back off)
            MCFAPIError: For other API errors
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        uuid = self._job_uuid(year, sequence)

        try:
            job = await self._client.get_job(uuid)
            return job
        except MCFNotFoundError:
            return None

    async def scrape_year(
        self,
        year: int,
        start_seq: int = 1,
        end_seq: Optional[int] = None,
        resume: bool = True,
        progress_callback: Optional[Callable[[ScrapeProgress], Awaitable[None]]] = None,
        dry_run: bool = False,
    ) -> ScrapeProgress:
        """
        Scrape all jobs for a given year.

        Args:
            year: Year to scrape (e.g., 2023)
            start_seq: Starting sequence number (default: 1)
            end_seq: Ending sequence number (default: estimated max for year)
            resume: Whether to resume from previous session
            progress_callback: Async callback for progress updates
            dry_run: If True, only preview without fetching

        Returns:
            Final scrape progress
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        explicit_end_seq = end_seq is not None
        estimated_end_seq = YEAR_ESTIMATES.get(year, 1_000_000)
        if end_seq is None:
            end_seq = estimated_end_seq

        # Check for existing session to resume
        session_id: Optional[int] = None
        jobs_found = 0
        jobs_not_found = 0
        consecutive_not_found = 0
        existing_session = None

        if resume:
            existing_session = self.db.get_incomplete_historical_session(year)
            if existing_session:
                session_id = existing_session["id"]
                start_seq = existing_session["current_seq"] + 1
                jobs_found = existing_session["jobs_found"]
                jobs_not_found = existing_session["jobs_not_found"]
                consecutive_not_found = existing_session["consecutive_not_found"]
                if existing_session.get("end_seq"):
                    end_seq = existing_session["end_seq"]
                logger.info(
                    f"Resuming year {year} from sequence {start_seq:,} "
                    f"({jobs_found:,} found, {jobs_not_found:,} not found)"
                )

        should_discover_bounds = (
            self.discover_bounds
            and not dry_run
            and not explicit_end_seq
            and end_seq is not None
            and (session_id is None or end_seq == estimated_end_seq)
        )

        if should_discover_bounds:
            _, discovered_end_seq = await self.find_year_bounds(year)
            if discovered_end_seq > 0:
                end_seq = min(end_seq, discovered_end_seq)
                logger.info(f"Using discovered end bound for {year}: {end_seq:,}")

        # Create new session if needed
        if session_id is None and not dry_run:
            session_id = self.db.create_historical_session(
                year,
                start_seq,
                end_seq,
                conn=self._write_conn,
            )
            self._mark_progress_committed()
            logger.info(f"Created new session {session_id} for year {year}")
        elif session_id and not dry_run and should_discover_bounds:
            self.db.update_historical_progress(
                session_id,
                start_seq - 1,
                jobs_found,
                jobs_not_found,
                consecutive_not_found,
                end_seq=end_seq,
                conn=self._write_conn,
            )
            self._mark_progress_committed()

        current_seq = start_seq
        checkpoint_interval = 100  # Save progress every N jobs

        logger.info(f"Scraping year {year}: sequences {start_seq:,} to {end_seq:,}{' (DRY RUN)' if dry_run else ''}")
        logger.info(f"Rate limiter: {self.rate_limiter.current_rps:.2f} req/sec")

        rate_limit_retries = 0
        try:
            while current_seq <= end_seq:
                # Check for early termination
                if consecutive_not_found >= self.not_found_threshold:
                    logger.info(f"Year {year}: {consecutive_not_found} consecutive not-found, assuming end of sequence")
                    break

                if dry_run:
                    # In dry run, just count without fetching
                    uuid = self._job_uuid(year, current_seq)
                    if self.db.has_job(uuid):
                        jobs_found += 1
                        self.batch_logger.log(year, current_seq, "skipped")
                    else:
                        jobs_not_found += 1
                        self.batch_logger.log(year, current_seq, "not_found")
                    current_seq += 1
                    rate_limit_retries = 0
                    continue

                try:
                    job = await self.fetch_job(year, current_seq)

                    if job:
                        # Save to database
                        is_new, _ = self.db.upsert_job(job, conn=self._write_conn)
                        jobs_found += 1
                        consecutive_not_found = 0

                        # Log successful fetch
                        self.batch_logger.log(year, current_seq, "found")
                        self.rate_limiter.on_success()
                        self._reset_rate_limit_streak()

                        # Update client rate if changed significantly
                        self._update_client_rate(self.rate_limiter.current_rps)

                        if is_new:
                            logger.debug(f"New job: {job.title[:50]} ({job.company_name})")
                    else:
                        jobs_not_found += 1
                        consecutive_not_found += 1
                        self.batch_logger.log(year, current_seq, "not_found")
                        self.rate_limiter.on_success()
                        self._reset_rate_limit_streak()
                        self._update_client_rate(self.rate_limiter.current_rps)

                except MCFRateLimitError:
                    rate_limit_retries += 1
                    should_retry = await self._handle_rate_limit(
                        year, current_seq, retry_count=rate_limit_retries, context="Rate limited"
                    )
                    if should_retry:
                        continue

                except MCFAPIError as e:
                    logger.error(f"API error at {year}-{current_seq}: {e}")
                    self.batch_logger.log(year, current_seq, "error", str(e))
                    self.rate_limiter.on_error()
                    jobs_not_found += 1
                    consecutive_not_found += 1

                except RetryError as e:
                    # Tenacity exhausted retries - check if underlying cause was rate limit
                    cause = e.last_attempt.exception() if e.last_attempt else None
                    if isinstance(cause, MCFRateLimitError):
                        rate_limit_retries += 1
                        should_retry = await self._handle_rate_limit(
                            year,
                            current_seq,
                            retry_count=rate_limit_retries,
                            context="Retries exhausted due to rate limiting",
                        )
                        if should_retry:
                            continue
                    else:
                        # Other retry error - log and continue
                        logger.error(f"Retry exhausted at {year}-{current_seq}: {e}")
                        self.batch_logger.log(year, current_seq, "error", str(e))
                        self.rate_limiter.on_error()
                        jobs_not_found += 1
                        consecutive_not_found += 1

                except Exception as e:
                    # Catch-all for unexpected errors - log and continue
                    logger.exception(f"Unexpected error at {year}-{current_seq}: {e}")
                    self.batch_logger.log(year, current_seq, "error", str(e))
                    self.rate_limiter.on_error()
                    jobs_not_found += 1
                    consecutive_not_found += 1

                # Update progress
                current_seq += 1
                rate_limit_retries = 0

                # Checkpoint periodically
                if session_id and (current_seq - start_seq) % checkpoint_interval == 0:
                    self.db.update_historical_progress(
                        session_id,
                        current_seq,
                        jobs_found,
                        jobs_not_found,
                        consecutive_not_found,
                        end_seq=end_seq,
                        conn=self._write_conn,
                    )
                    self.batch_logger.flush()
                    self._mark_progress_committed()

                    # Progress callback
                    if progress_callback:
                        progress = ScrapeProgress(
                            year=year,
                            current_seq=current_seq,
                            jobs_found=jobs_found,
                            jobs_not_found=jobs_not_found,
                            consecutive_not_found=consecutive_not_found,
                            start_seq=start_seq,
                            end_seq=end_seq,
                        )
                        await progress_callback(progress)

        finally:
            # Flush batch logger
            self.batch_logger.flush()
            self._mark_progress_committed()

            # Final progress update
            if session_id:
                self.db.update_historical_progress(
                    session_id,
                    current_seq,
                    jobs_found,
                    jobs_not_found,
                    consecutive_not_found,
                    end_seq=end_seq,
                    conn=self._write_conn,
                )
                self._mark_progress_committed()

        # Mark completed if we finished normally
        if session_id and current_seq > end_seq:
            self.db.complete_historical_session(session_id, conn=self._write_conn)
            self._mark_progress_committed()
            logger.info(f"Completed year {year}")

        return ScrapeProgress(
            year=year,
            current_seq=current_seq,
            jobs_found=jobs_found,
            jobs_not_found=jobs_not_found,
            consecutive_not_found=consecutive_not_found,
            start_seq=start_seq,
            end_seq=end_seq,
        )

    async def scrape_range(
        self,
        start_id: str,
        end_id: str,
        progress_callback: Optional[Callable[[ScrapeProgress], Awaitable[None]]] = None,
        dry_run: bool = False,
    ) -> ScrapeProgress:
        """
        Scrape a specific range of job IDs.

        Args:
            start_id: Starting job ID (e.g., MCF-2023-0500000)
            end_id: Ending job ID (e.g., MCF-2023-0600000)
            progress_callback: Async callback for progress updates
            dry_run: If True, only preview without fetching

        Returns:
            Scrape progress
        """
        start_year, start_seq = self.parse_job_id(start_id)
        end_year, end_seq = self.parse_job_id(end_id)

        if start_year != end_year:
            raise ValueError("Start and end IDs must be from the same year")

        return await self.scrape_year(
            start_year,
            start_seq=start_seq,
            end_seq=end_seq,
            resume=False,  # Don't resume for explicit range
            progress_callback=progress_callback,
            dry_run=dry_run,
        )

    async def scrape_all_years(
        self,
        years: Optional[list[int]] = None,
        resume: bool = True,
        progress_callback: Optional[Callable[[ScrapeProgress], Awaitable[None]]] = None,
        dry_run: bool = False,
    ) -> dict[int, ScrapeProgress]:
        """
        Scrape all years (2019-present).

        Args:
            years: List of years to scrape (default: all known years)
            resume: Whether to resume from previous sessions
            progress_callback: Async callback for progress updates
            dry_run: If True, only preview without fetching

        Returns:
            Dict mapping year to final progress
        """
        if years is None:
            years = sorted(YEAR_ESTIMATES.keys())

        results = {}
        for year in years:
            logger.info(f"Starting year {year}")
            results[year] = await self.scrape_year(
                year,
                resume=resume,
                progress_callback=progress_callback,
                dry_run=dry_run,
            )

        return results

    async def find_year_bounds(self, year: int) -> tuple[int, int]:
        """
        Discover a tighter upper sequence bound for a year.

        Historical IDs are dense but not perfectly contiguous, so the search
        probes sparse windows instead of assuming every missing ID means the
        year has ended.

        Args:
            year: Year to find bounds for

        Returns:
            Tuple of (min_seq, max_seq) that exist
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        min_seq = 1
        estimate = YEAR_ESTIMATES.get(year, 1_000_000)
        low = 1
        high = min(MAX_SEQUENCE - BOUND_DISCOVERY_WINDOW, max(estimate, 1))

        has_jobs, _ = await self._window_has_job(year, high)
        if has_jobs is None:
            return (min_seq, estimate)
        while has_jobs and high < MAX_SEQUENCE - BOUND_DISCOVERY_WINDOW:
            low = high
            high = min(high * 2, MAX_SEQUENCE - BOUND_DISCOVERY_WINDOW)
            has_jobs, _ = await self._window_has_job(year, high)
            if has_jobs is None:
                return (min_seq, estimate)

        while low + BOUND_DISCOVERY_WINDOW < high:
            mid = ((low + high) // 2 // BOUND_DISCOVERY_STEP) * BOUND_DISCOVERY_STEP
            if mid <= low:
                mid = low + BOUND_DISCOVERY_STEP

            has_jobs, _ = await self._window_has_job(year, mid)
            if has_jobs is None:
                return (min_seq, estimate)
            if has_jobs:
                low = mid
            else:
                high = mid - BOUND_DISCOVERY_STEP

        refine_start = max(min_seq, low - BOUND_DISCOVERY_WINDOW)
        refine_end = min(MAX_SEQUENCE, high + BOUND_DISCOVERY_WINDOW)

        for sequence in range(refine_end, refine_start - 1, -1):
            probe_result = await self._probe_job_exists(year, sequence)
            if probe_result is True:
                return (min_seq, sequence)
            if probe_result is None:
                return (min_seq, estimate)

        return (min_seq, estimate)

    async def retry_gaps(
        self,
        year: int,
        progress_callback: Optional[Callable[[ScrapeProgress], Awaitable[None]]] = None,
    ) -> ScrapeProgress:
        """
        Retry fetching jobs for missing/failed sequences in a year.

        Finds gaps in fetch_attempts and errors, then retries each.

        Args:
            year: Year to retry gaps for
            progress_callback: Async callback for progress updates

        Returns:
            Scrape progress with retry results
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        # Get gaps and failed attempts
        gaps = self.db.get_missing_sequences(year)
        failed = self.db.get_failed_attempts(year)

        # Build list of sequences to retry
        sequences_to_retry: list[int] = []

        for start, end in gaps:
            sequences_to_retry.extend(range(start, end + 1))

        for attempt in failed:
            if attempt["sequence"] not in sequences_to_retry:
                sequences_to_retry.append(attempt["sequence"])

        sequences_to_retry.sort()

        if not sequences_to_retry:
            logger.info(f"No gaps or failed attempts to retry for year {year}")
            return ScrapeProgress(
                year=year,
                current_seq=0,
                jobs_found=0,
                jobs_not_found=0,
                consecutive_not_found=0,
                start_seq=0,
                end_seq=0,
            )

        logger.info(
            f"Retrying {len(sequences_to_retry):,} sequences for year {year} (gaps: {len(gaps)}, failed: {len(failed)})"
        )

        jobs_found = 0
        jobs_not_found = 0

        for i, seq in enumerate(sequences_to_retry):
            rate_limit_retries = 0
            while True:
                try:
                    job = await self.fetch_job(year, seq)

                    if job:
                        is_new, _ = self.db.upsert_job(job, conn=self._write_conn)
                        jobs_found += 1
                        self.batch_logger.log(year, seq, "found")
                        self.rate_limiter.on_success()
                        self._reset_rate_limit_streak()
                        self._update_client_rate(self.rate_limiter.current_rps)

                        if is_new:
                            logger.debug(f"Recovered job: {job.title[:50]}")
                    else:
                        self.batch_logger.log(year, seq, "not_found")
                        jobs_not_found += 1
                        self.rate_limiter.on_success()
                        self._reset_rate_limit_streak()
                        self._update_client_rate(self.rate_limiter.current_rps)
                    break

                except MCFRateLimitError:
                    rate_limit_retries += 1
                    should_retry = await self._handle_rate_limit(
                        year,
                        seq,
                        retry_count=rate_limit_retries,
                        context="Gap retry rate limited",
                    )
                    if should_retry:
                        continue
                    break

                except MCFAPIError as e:
                    self.batch_logger.log(year, seq, "error", str(e))
                    self.rate_limiter.on_error()
                    jobs_not_found += 1
                    break

            # Progress callback every 100 sequences
            if (i + 1) % 100 == 0:
                self.batch_logger.flush()
                self._mark_progress_committed()

            if progress_callback and (i + 1) % 100 == 0:
                progress = ScrapeProgress(
                    year=year,
                    current_seq=seq,
                    jobs_found=jobs_found,
                    jobs_not_found=jobs_not_found,
                    consecutive_not_found=0,
                    start_seq=sequences_to_retry[0],
                    end_seq=sequences_to_retry[-1],
                )
                await progress_callback(progress)

        # Final flush
        self.batch_logger.flush()
        self._mark_progress_committed()

        logger.info(f"Gap retry complete for year {year}: {jobs_found:,} recovered, {jobs_not_found:,} still missing")

        return ScrapeProgress(
            year=year,
            current_seq=sequences_to_retry[-1] if sequences_to_retry else 0,
            jobs_found=jobs_found,
            jobs_not_found=jobs_not_found,
            consecutive_not_found=0,
            start_seq=sequences_to_retry[0] if sequences_to_retry else 0,
            end_seq=sequences_to_retry[-1] if sequences_to_retry else 0,
        )

    async def retry_all_gaps(
        self,
        years: Optional[list[int]] = None,
        progress_callback: Optional[Callable[[ScrapeProgress], Awaitable[None]]] = None,
    ) -> dict[int, ScrapeProgress]:
        """
        Retry gaps for all years.

        Args:
            years: List of years to retry (default: all years with attempts)
            progress_callback: Async callback for progress updates

        Returns:
            Dict mapping year to retry results
        """
        if years is None:
            # Get years from attempt stats
            stats = self.db.get_all_attempt_stats()
            years = sorted(stats.keys())

        results = {}
        for year in years:
            results[year] = await self.retry_gaps(year, progress_callback)

        return results
