"""
Scraper daemon for long-running background operation.

Provides a simple daemon process manager with:
- Background process execution via subprocess
- Heartbeat-based wake detection (detects sleep/resume)
- PID file management
- Status monitoring

Uses subprocess.Popen (fork+exec) instead of bare os.fork() to avoid
macOS ObjC runtime crashes in the child process.
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from .database import MCFDatabase

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_HEARTBEAT_INTERVAL = 10  # seconds
DEFAULT_WAKE_THRESHOLD = 300  # 5 minutes - gap indicates sleep/wake


class DaemonError(Exception):
    """Base exception for daemon errors."""

    pass


class DaemonAlreadyRunning(DaemonError):
    """Raised when trying to start a daemon that's already running."""

    pass


class DaemonNotRunning(DaemonError):
    """Raised when trying to stop a daemon that's not running."""

    pass


class ScraperDaemon:
    """
    Simple daemon process manager for long-running scrapes.

    Features:
    - Forks into background process
    - Maintains heartbeat for wake detection
    - PID file for process management
    - Database state tracking

    Example:
        daemon = ScraperDaemon(db)

        # Start in background
        daemon.start(scrape_year_func, year=2023)

        # Check status
        status = daemon.status()
        print(f"Running: {status['status'] == 'running'}")

        # Stop
        daemon.stop()
    """

    def __init__(
        self,
        db: "MCFDatabase",
        pidfile: str | Path = "data/.scraper.pid",
        logfile: str | Path = "data/scraper_daemon.log",
        heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        wake_threshold: int = DEFAULT_WAKE_THRESHOLD,
    ):
        """
        Initialize the daemon manager.

        Args:
            db: MCFDatabase for state tracking
            pidfile: Path to PID file
            logfile: Path to daemon log file
            heartbeat_interval: Seconds between heartbeats
            wake_threshold: Seconds gap that indicates sleep/wake
        """
        self.db = db
        self.pidfile = Path(pidfile)
        self.logfile = Path(logfile)
        self.heartbeat_interval = heartbeat_interval
        self.wake_threshold = wake_threshold

        # Ensure parent directories exist
        self.pidfile.parent.mkdir(parents=True, exist_ok=True)
        self.logfile.parent.mkdir(parents=True, exist_ok=True)

    def _pid_from_pidfile(self) -> int | None:
        """Read the PID file if present and parseable."""
        if not self.pidfile.exists():
            return None
        try:
            return int(self.pidfile.read_text().strip())
        except (ValueError, FileNotFoundError):
            return None

    def _process_exists(self, pid: int) -> bool:
        """Check whether a PID is still alive."""
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _find_worker_pids(self) -> list[int]:
        """Locate daemon workers that match this pidfile/logfile pair."""
        try:
            result = subprocess.run(
                ["ps", "-ax", "-o", "pid=,command="],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return []

        pidfile_arg = f"--pidfile {self.pidfile}"
        logfile_arg = f"--logfile {self.logfile}"
        matches: list[int] = []

        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            pid_str, _, command = stripped.partition(" ")
            if not pid_str.isdigit():
                continue
            if "src.cli _daemon-worker" not in command:
                continue
            if pidfile_arg not in command or logfile_arg not in command:
                continue
            matches.append(int(pid_str))

        return matches

    def is_running(self) -> bool:
        """Check if daemon is currently running."""
        pid = self._pid_from_pidfile()
        if pid is not None:
            if self._process_exists(pid):
                return True
            self.pidfile.unlink(missing_ok=True)

        return bool(self._find_worker_pids())

    def get_pid(self) -> int | None:
        """Get PID of running daemon, or None if not running."""
        pid = self._pid_from_pidfile()
        if pid is not None and self._process_exists(pid):
            return pid

        if pid is not None:
            self.pidfile.unlink(missing_ok=True)

        matches = self._find_worker_pids()
        if not matches:
            return None
        return matches[0]

    def start(
        self,
        year: int | None = None,
        all_years: bool = False,
        rate_limit: float = 2.0,
        db_path: str = "data/mcf_jobs.db",
        max_rate_limit_retries: int = 4,
        cooldown_seconds: float = 30.0,
        discover_bounds: bool = True,
    ) -> int:
        """
        Spawn a subprocess to run the scraper in the background.

        Uses subprocess.Popen instead of os.fork() to avoid macOS
        ObjC runtime crash (fork-without-exec is unsafe on macOS).

        Args:
            year: Specific year to scrape, or None if all_years
            all_years: If True, scrape all years (2019-2026)
            rate_limit: Requests per second
            db_path: Path to SQLite database
            max_rate_limit_retries: Per-sequence retry cap for 429s
            cooldown_seconds: Global cooldown after repeated 429s
            discover_bounds: Whether to discover tighter year bounds before scanning

        Returns:
            PID of the daemon process

        Raises:
            DaemonAlreadyRunning: If daemon is already running
        """
        if self.is_running():
            pid = self.get_pid()
            raise DaemonAlreadyRunning(f"Daemon already running with PID {pid}")
        try:
            writable = self.db.can_acquire_write_lock(db_path)
        except Exception as exc:
            # Connection refused, auth failure, missing database, etc. The
            # message from psycopg / sqlite3 already names the real problem
            # (e.g. "connection refused on port 55432"), so surface it as-is
            # instead of claiming another process is writing.
            raise DaemonError(f"Database unavailable: {exc}") from exc
        if not writable:
            raise DaemonError("Database is busy: another process is writing to it")

        # Build command for the worker subprocess
        cmd = [
            sys.executable,
            "-m",
            "src.cli",
            "_daemon-worker",
            "--db",
            db_path,
            "--rate-limit",
            str(rate_limit),
            "--max-rate-limit-retries",
            str(max_rate_limit_retries),
            "--cooldown-seconds",
            str(cooldown_seconds),
            "--pidfile",
            str(self.pidfile),
            "--logfile",
            str(self.logfile),
            "--heartbeat-interval",
            str(self.heartbeat_interval),
            "--wake-threshold",
            str(self.wake_threshold),
        ]
        cmd.append("--discover-bounds" if discover_bounds else "--no-discover-bounds")
        if year is not None:
            cmd.extend(["--year", str(year)])
        elif all_years:
            cmd.append("--all")

        # Open log file for subprocess stdout/stderr
        log_fd = open(self.logfile, "a")

        # Launch subprocess in new session (detached from terminal)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=log_fd,
            start_new_session=True,
        )
        log_fd.close()

        # Write PID file immediately from parent
        self.pidfile.write_text(str(proc.pid))

        return proc.pid

    def run_worker(
        self,
        scraper_func: Callable[..., Awaitable[Any]],
    ) -> None:
        """
        Run the scraper as a daemon worker (called from subprocess).

        This method is intended to be called from the _daemon-worker CLI
        command, which runs in a separate process spawned by start().
        """
        # Set up logging to file
        file_handler = logging.FileHandler(self.logfile)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.root.addHandler(file_handler)
        logging.root.setLevel(logging.INFO)

        logger.info(f"Daemon worker started with PID {os.getpid()}")

        # Update database state
        self.db.update_daemon_state(os.getpid(), "running")

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Run the scraper with heartbeat
        try:
            asyncio.run(self._run_with_heartbeat(scraper_func))
        except Exception as e:
            logger.exception(f"Daemon error: {e}")
        finally:
            self._cleanup()

    async def _run_with_heartbeat(
        self,
        scraper_func: Callable[..., Awaitable[Any]],
    ) -> Any:
        """Run scraper function with concurrent heartbeat task."""
        last_beat = time.time()
        should_stop = asyncio.Event()

        async def heartbeat_loop():
            nonlocal last_beat
            while not should_stop.is_set():
                now = time.time()
                gap = now - last_beat

                if gap > self.wake_threshold:
                    logger.warning(
                        f"Wake detected: {gap:.0f}s gap (threshold: {self.wake_threshold}s). Resuming scrape..."
                    )

                last_beat = now
                self.db.update_daemon_heartbeat()

                try:
                    await asyncio.wait_for(should_stop.wait(), timeout=self.heartbeat_interval)
                except asyncio.TimeoutError:
                    pass  # Expected - continue heartbeat loop

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat_loop())

        try:
            # Run the scraper
            result = await scraper_func()
            return result
        finally:
            # Stop heartbeat
            should_stop.set()
            await heartbeat_task

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        """Clean up daemon state."""
        try:
            self.db.update_daemon_state(os.getpid(), "stopped")
        except Exception as e:
            logger.error(f"Failed to update daemon state: {e}")

        try:
            self.pidfile.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Failed to remove PID file: {e}")

        logger.info("Daemon stopped")

    def stop(self, timeout: int = 10) -> bool:
        """
        Stop the running daemon.

        Args:
            timeout: Seconds to wait for graceful shutdown

        Returns:
            True if daemon was stopped, False if it wasn't running

        Raises:
            DaemonNotRunning: If no daemon is running
            DaemonError: If daemon couldn't be stopped
        """
        if not self.is_running():
            raise DaemonNotRunning("No daemon is running")

        pid_from_file = self._pid_from_pidfile()
        if pid_from_file is not None and self._process_exists(pid_from_file):
            target_pids = [pid_from_file]
        else:
            target_pids = self._find_worker_pids()

        if not target_pids:
            raise DaemonNotRunning("No daemon is running")

        for pid in target_pids:
            logger.info(f"Stopping daemon (PID {pid})...")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue

        for _ in range(timeout):
            time.sleep(1)
            remaining = [pid for pid in target_pids if self._process_exists(pid)]
            if not remaining:
                self.pidfile.unlink(missing_ok=True)
                return True
            target_pids = remaining

        for pid in target_pids:
            logger.warning(f"Daemon didn't exit gracefully, sending SIGKILL to PID {pid}...")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        time.sleep(0.5)

        self.pidfile.unlink(missing_ok=True)
        return True

    def status(self) -> dict:
        """
        Get daemon status.

        Returns:
            Dict with status information including:
            - running: bool
            - pid: int or None
            - database_state: dict from daemon_state table
        """
        running = self.is_running()
        pid = self.get_pid() if running else None
        db_state = self.db.get_daemon_state()

        return {
            "running": running,
            "pid": pid,
            "pidfile": str(self.pidfile),
            "logfile": str(self.logfile),
            **db_state,
        }
