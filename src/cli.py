#!/usr/bin/env python3
"""
CLI interface for the MyCareersFuture job scraper.

Usage:
    python -m src.cli scrape "data scientist"
    python -m src.cli scrape "machine learning" --max-jobs 500
    python -m src.cli status
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from src.mcf import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_WAKE_THRESHOLD,
    YEAR_ESTIMATES,
    DaemonAlreadyRunning,
    DaemonError,
    DaemonNotRunning,
    EmbeddingGenerator,
    EmbeddingStats,
    HistoricalScraper,
    MCFClient,
    MCFDatabase,
    MCFMigrator,
    MCFScraper,
    ScraperDaemon,
)
from src.mcf.db_backup import create_sqlite_hot_backup, verify_sqlite_backup
from src.mcf.db_factory import open_database
from src.mcf.db_target import (
    resolve_database_target,
    resolve_preferred_database_value,
)
from src.mcf.embeddings import (
    FAISSIndexManager,
    IndexCompatibilityError,
    SearchRequest,
    SearchResponse,
    SemanticSearchEngine,
    default_onnx_model_dir,
    export_sentence_transformer_to_onnx,
    validate_embedding_backend_config,
)
from src.mcf.hosted_slice import DEFAULT_HOSTED_SLICE_POLICY, HostedSlicePolicy
from src.mcf.postgres_migration import (
    audit_sqlite_source,
    migrate_sqlite_backup_to_postgres,
    purge_hosted_slice,
    seed_hosted_slice_from_postgres,
    write_migration_report,
)

app = typer.Typer(
    name="mcf",
    help="MyCareersFuture job scraper - fast API-based job data collection",
    add_completion=False,
)
console = Console()
CLI_DEFAULT_EMBEDDING_BACKEND = "onnx"


def _open_database(
    db_path: str | None,
    *,
    read_only: bool = False,
    ensure_schema: bool = True,
):
    """Open a database target while preserving the existing --db CLI surface."""
    return open_database(
        db_path,
        read_only=read_only,
        ensure_schema=ensure_schema,
    )


def _resolve_cli_db_path(db_path: str | None) -> str:
    """Resolve the preferred CLI database target, including the persisted local default."""
    return resolve_database_target(
        resolve_preferred_database_value(
            db_path,
            include_persisted=True,
        )
    ).value


def _resolve_daemon_db_path(db_path: str | None) -> str:
    """Backward-compatible wrapper for daemon-target resolution."""
    return _resolve_cli_db_path(db_path)


def _default_onnx_model_dir(model_name: str) -> Path:
    """Return the default export directory for an ONNX embedding model."""
    return default_onnx_model_dir(model_name)


def _resolve_cli_onnx_model_dir(
    *,
    backend: str,
    model_name: str | None = None,
    onnx_model_dir: str | Path | None = None,
) -> str | Path | None:
    """Resolve the effective ONNX model directory for CLI commands."""
    if onnx_model_dir is not None:
        return onnx_model_dir
    if backend.strip().lower() != "onnx":
        return None
    env_onnx_model_dir = os.environ.get("MCF_ONNX_MODEL_DIR")
    if env_onnx_model_dir:
        return env_onnx_model_dir
    return _default_onnx_model_dir(model_name or EmbeddingGenerator.MODEL_NAME)


def _validate_cli_backend_config_or_exit(
    *,
    backend: str,
    model_name: str | None = None,
    onnx_model_dir: str | Path | None = None,
) -> None:
    """Validate CLI backend configuration and exit with a helpful message."""
    try:
        validate_embedding_backend_config(
            backend=backend,
            model_name=model_name or EmbeddingGenerator.MODEL_NAME,
            dimension=EmbeddingGenerator.DIMENSION,
            onnx_model_dir=onnx_model_dir,
        )
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        console.print(f"[red]Invalid embedding backend configuration:[/red] {exc}")
        if backend.strip().lower() == "onnx":
            export_dir = onnx_model_dir or _default_onnx_model_dir(model_name or EmbeddingGenerator.MODEL_NAME)
            console.print("[yellow]Export the ONNX bundle first:[/yellow]")
            console.print(
                f"  python -m src.cli embed-export-onnx "
                f"{model_name or EmbeddingGenerator.MODEL_NAME} --output-dir {export_dir}"
            )
        raise typer.Exit(1)


def _create_embedding_generator(
    *,
    model_name: str | None = None,
    backend: str = CLI_DEFAULT_EMBEDDING_BACKEND,
    device: str | None = None,
    onnx_model_dir: str | Path | None = None,
) -> EmbeddingGenerator:
    """Create an embedding generator with the selected backend settings."""
    resolved_onnx_model_dir = _resolve_cli_onnx_model_dir(
        backend=backend,
        model_name=model_name,
        onnx_model_dir=onnx_model_dir,
    )
    validate_embedding_backend_config(
        backend=backend,
        model_name=model_name or EmbeddingGenerator.MODEL_NAME,
        dimension=EmbeddingGenerator.DIMENSION,
        onnx_model_dir=resolved_onnx_model_dir,
    )
    return EmbeddingGenerator(
        model_name=model_name,
        device=device,
        backend=backend,
        onnx_model_dir=resolved_onnx_model_dir,
    )


def _parse_since_date(since: Optional[str]) -> date | None:
    """Parse a YYYY-MM-DD string into a date, or exit on bad format."""
    if since is None:
        return None
    try:
        return date.fromisoformat(since)
    except ValueError:
        console.print(f"[red]Invalid date format: {since}. Use YYYY-MM-DD.[/red]")
        raise typer.Exit(1)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


@app.command()
def scrape(
    query: str = typer.Argument(..., help="Search query (e.g., 'data scientist')"),
    max_jobs: Optional[int] = typer.Option(None, "--max-jobs", "-n", help="Maximum number of jobs to scrape"),
    output_dir: str = typer.Option("data", "--output", "-o", help="Output directory for files"),
    format: str = typer.Option("csv", "--format", "-f", help="Output format: csv or json"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Don't resume from previous checkpoint"),
    rate_limit: float = typer.Option(2.0, "--rate-limit", "-r", help="Requests per second"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Scrape job listings from MyCareersFuture.

    Examples:
        mcf scrape "data scientist"
        mcf scrape "machine learning" --max-jobs 500
        mcf scrape "data engineer" -o ./jobs -f json
    """
    setup_logging(verbose)
    resolved_db_path = _resolve_cli_db_path(db_path)

    console.print("\n[bold blue]MCF Job Scraper[/bold blue]")
    console.print(f"Search query: [green]{query}[/green]")
    console.print(f"Database: [green]{resolved_db_path}[/green]")

    if max_jobs:
        console.print(f"Max jobs: {max_jobs}")

    console.print()

    async def run():
        scraper = MCFScraper(
            output_dir=output_dir,
            requests_per_second=rate_limit,
            db_path=resolved_db_path,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            await scraper.scrape(
                query,
                max_jobs=max_jobs,
                resume=not no_resume,
                progress=progress,
            )

        # Save results
        if scraper.job_count > 0:
            filepath = scraper.save(query, format=format)
            console.print(f"\n[green]Success![/green] Saved {scraper.job_count} jobs to {filepath}")

            # Show sample
            df = scraper.get_dataframe()
            if len(df) > 0:
                console.print("\n[bold]Sample jobs:[/bold]")
                table = Table(show_header=True)
                table.add_column("Title", style="cyan", max_width=40)
                table.add_column("Company", max_width=30)
                table.add_column("Salary", justify="right")

                for _, row in df.head(5).iterrows():
                    salary = ""
                    if row.get("salary_min") and row.get("salary_max"):
                        salary = f"${row['salary_min']:,} - ${row['salary_max']:,}"
                    table.add_row(
                        str(row.get("title", ""))[:40],
                        str(row.get("company_name", ""))[:30],
                        salary,
                    )

                console.print(table)
        else:
            console.print("[yellow]No jobs found[/yellow]")

    asyncio.run(run())


@app.command()
def scrape_multi(
    queries: list[str] = typer.Argument(..., help="Search queries to scrape"),
    max_jobs: Optional[int] = typer.Option(None, "--max-jobs", "-n", help="Maximum jobs per query"),
    output_dir: str = typer.Option("data", "--output", "-o", help="Output directory"),
    output_name: str = typer.Option("jobs", "--name", help="Base name for output file"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """
    Scrape multiple search queries with deduplication.

    Example:
        mcf scrape-multi "data scientist" "machine learning" "data engineer"
    """
    setup_logging(verbose)
    resolved_db_path = _resolve_cli_db_path(db_path)

    console.print("\n[bold blue]MCF Multi-Query Scraper[/bold blue]")
    console.print(f"Queries: {', '.join(queries)}")
    console.print(f"Database: [green]{resolved_db_path}[/green]")
    console.print()

    async def run():
        scraper = MCFScraper(output_dir=output_dir, db_path=resolved_db_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            await scraper.scrape_multiple(
                queries,
                max_jobs_per_query=max_jobs,
                progress=progress,
            )

        if scraper.job_count > 0:
            filepath = scraper.save(output_name)
            console.print(f"\n[green]Success![/green] Saved {scraper.job_count} unique jobs to {filepath}")

    asyncio.run(run())


@app.command()
def status() -> None:
    """
    Show status of pending checkpoints (incomplete scrapes).
    """
    checkpoint_dir = Path(".mcf_checkpoints")

    if not checkpoint_dir.exists():
        console.print("No checkpoints found")
        return

    checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))

    if not checkpoints:
        console.print("No incomplete scrapes found")
        return

    console.print("\n[bold]Incomplete Scrapes:[/bold]\n")

    table = Table(show_header=True)
    table.add_column("Query")
    table.add_column("Progress", justify="right")
    table.add_column("Last Updated")

    import json
    from datetime import datetime

    for cp_file in checkpoints:
        try:
            with open(cp_file) as f:
                data = json.load(f)

            query = data.get("search_query", "unknown")
            fetched = data.get("fetched_count", 0)
            total = data.get("total_jobs", 0)
            updated = data.get("updated_at", "")

            if updated:
                try:
                    dt = datetime.fromisoformat(updated)
                    updated = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    pass

            progress_pct = (fetched / total * 100) if total > 0 else 0
            progress_str = f"{fetched}/{total} ({progress_pct:.1f}%)"

            table.add_row(query, progress_str, updated)
        except Exception as e:
            console.print(f"Error reading {cp_file}: {e}")

    console.print(table)
    console.print("\nRun [bold]mcf scrape <query>[/bold] to resume")


@app.command()
def preview(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of jobs to show"),
) -> None:
    """
    Preview job search results without saving.
    """
    setup_logging(verbose=False)

    console.print(f"\n[bold blue]Preview: {query}[/bold blue]\n")

    async def run():
        async with MCFClient() as client:
            response = await client.search_jobs(query, limit=limit)

            console.print(f"Found {response.total} total jobs\n")

            if not response.results:
                console.print("[yellow]No results[/yellow]")
                return

            for job in response.results:
                salary = ""
                if job.salary_min and job.salary_max:
                    salary = f"${job.salary_min:,} - ${job.salary_max:,} {job.salary_type}"

                console.print(f"[bold cyan]{job.title}[/bold cyan]")
                console.print(f"  Company: {job.company_name}")
                if salary:
                    console.print(f"  Salary: {salary}")
                console.print(f"  Type: {job.employment_type} | Level: {job.seniority}")
                if job.skills_list:
                    console.print(f"  Skills: {job.skills_list[:80]}...")
                console.print(f"  URL: {job.job_url}")
                console.print()

    asyncio.run(run())


@app.command()
def clear_checkpoints() -> None:
    """
    Clear all saved checkpoints.
    """
    checkpoint_dir = Path(".mcf_checkpoints")

    if not checkpoint_dir.exists():
        console.print("No checkpoints to clear")
        return

    import shutil

    shutil.rmtree(checkpoint_dir)
    console.print("[green]All checkpoints cleared[/green]")


# Database query commands


@app.command(name="list")
def list_jobs(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of jobs to show"),
    company: Optional[str] = typer.Option(None, "--company", "-c", help="Filter by company name"),
    salary_min: Optional[int] = typer.Option(None, "--salary-min", help="Minimum salary"),
    salary_max: Optional[int] = typer.Option(None, "--salary-max", help="Maximum salary"),
    employment_type: Optional[str] = typer.Option(None, "--employment-type", "-e", help="Employment type"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
) -> None:
    """
    List jobs from the database with optional filters.

    Examples:
        mcf list --limit 20
        mcf list --company Google --salary-min 8000
        mcf list --employment-type "Full Time"
    """
    db = _open_database(_resolve_cli_db_path(db_path), read_only=True)

    jobs = db.search_jobs(
        company_name=company,
        salary_min=salary_min,
        salary_max=salary_max,
        employment_type=employment_type,
        limit=limit,
    )

    if not jobs:
        console.print("[yellow]No jobs found matching filters[/yellow]")
        return

    console.print(f"\n[bold blue]Jobs ({len(jobs)} shown)[/bold blue]\n")

    table = Table(show_header=True)
    table.add_column("Title", style="cyan", max_width=35)
    table.add_column("Company", max_width=25)
    table.add_column("Salary", justify="right")
    table.add_column("Type", max_width=12)
    table.add_column("Posted")

    for job in jobs:
        salary = ""
        if job.get("salary_min") and job.get("salary_max"):
            salary = f"${job['salary_min']:,} - ${job['salary_max']:,}"

        table.add_row(
            str(job.get("title", ""))[:35],
            str(job.get("company_name", ""))[:25],
            salary,
            str(job.get("employment_type", ""))[:12],
            str(job.get("posted_date", ""))[:10],
        )

    console.print(table)


@app.command()
def search(
    keyword: str = typer.Argument(..., help="Search keyword"),
    field: str = typer.Option("all", "--field", "-f", help="Field to search: all, title, skills"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
) -> None:
    """
    Search jobs by keyword.

    Examples:
        mcf search "machine learning"
        mcf search "Python" --field skills
        mcf search "Senior" --field title --limit 50
    """
    db = _open_database(_resolve_cli_db_path(db_path), read_only=True)

    jobs = db.search_jobs(keyword=keyword, limit=limit)

    if not jobs:
        console.print(f"[yellow]No jobs found matching '{keyword}'[/yellow]")
        return

    console.print(f"\n[bold blue]Search results for '{keyword}' ({len(jobs)} found)[/bold blue]\n")

    table = Table(show_header=True)
    table.add_column("Title", style="cyan", max_width=40)
    table.add_column("Company", max_width=25)
    table.add_column("Skills", max_width=40)

    for job in jobs:
        skills = str(job.get("skills", ""))
        if len(skills) > 40:
            skills = skills[:37] + "..."

        table.add_row(
            str(job.get("title", ""))[:40],
            str(job.get("company_name", ""))[:25],
            skills,
        )

    console.print(table)


@app.command()
def stats(
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
) -> None:
    """
    Show database statistics.
    """
    db = _open_database(_resolve_cli_db_path(db_path), read_only=True)
    stats_data = db.get_stats()

    console.print("\n[bold blue]Database Statistics[/bold blue]\n")

    # General stats
    console.print(f"[bold]Total jobs:[/bold] {stats_data['total_jobs']:,}")
    console.print(f"[bold]Jobs with history:[/bold] {stats_data['jobs_with_history']:,}")
    console.print(f"[bold]History records:[/bold] {stats_data['history_records']:,}")
    console.print(f"[bold]Added today:[/bold] {stats_data['jobs_added_today']:,}")
    console.print(f"[bold]Updated today:[/bold] {stats_data['jobs_updated_today']:,}")

    # Salary stats
    salary = stats_data.get("salary_stats", {})
    if salary.get("min"):
        console.print(f"\n[bold]Salary range:[/bold] ${salary['min']:,} - ${salary['max']:,}")
        console.print(f"[bold]Average range:[/bold] ${salary['avg_min']:,} - ${salary['avg_max']:,}")

    # Employment types
    if stats_data.get("by_employment_type"):
        console.print("\n[bold]By Employment Type:[/bold]")
        for emp_type, count in stats_data["by_employment_type"].items():
            console.print(f"  {emp_type}: {count:,}")

    # Top companies
    if stats_data.get("top_companies"):
        console.print("\n[bold]Top Companies:[/bold]")
        for company, count in list(stats_data["top_companies"].items())[:5]:
            console.print(f"  {company}: {count:,} jobs")


@app.command()
def export(
    output: Path = typer.Argument(..., help="Output CSV file path"),
    keyword: Optional[str] = typer.Option(None, "--keyword", "-k", help="Filter by keyword"),
    company: Optional[str] = typer.Option(None, "--company", "-c", help="Filter by company"),
    salary_min: Optional[int] = typer.Option(None, "--salary-min", help="Minimum salary"),
    salary_max: Optional[int] = typer.Option(None, "--salary-max", help="Maximum salary"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
) -> None:
    """
    Export jobs from database to CSV.

    Examples:
        mcf export jobs.csv
        mcf export high_salary.csv --salary-min 10000
        mcf export google_jobs.csv --company Google
    """
    db = _open_database(_resolve_cli_db_path(db_path), read_only=True)

    count = db.export_to_csv(
        output,
        keyword=keyword,
        company_name=company,
        salary_min=salary_min,
        salary_max=salary_max,
    )

    if count > 0:
        console.print(f"[green]Exported {count:,} jobs to {output}[/green]")
    else:
        console.print("[yellow]No jobs found matching filters[/yellow]")


@app.command()
def history(
    uuid: str = typer.Argument(..., help="Job UUID"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
) -> None:
    """
    Show history of changes for a job.

    Example:
        mcf history abc123-def456
    """
    db = _open_database(_resolve_cli_db_path(db_path), read_only=True)

    # Get current job
    job = db.get_job(uuid)
    if not job:
        console.print(f"[red]Job not found: {uuid}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Job: {job['title']}[/bold blue]")
    console.print(f"Company: {job['company_name']}")
    console.print(f"First seen: {job['first_seen_at']}")
    console.print(f"Last updated: {job['last_updated_at']}")

    # Get history
    history_records = db.get_job_history(uuid)

    if not history_records:
        console.print("\n[yellow]No history records (job hasn't been updated)[/yellow]")
        return

    console.print(f"\n[bold]History ({len(history_records)} updates):[/bold]\n")

    table = Table(show_header=True)
    table.add_column("Date", style="dim")
    table.add_column("Title")
    table.add_column("Company")
    table.add_column("Salary")
    table.add_column("Applications")

    for record in history_records:
        salary = ""
        if record.get("salary_min") and record.get("salary_max"):
            salary = f"${record['salary_min']:,} - ${record['salary_max']:,}"

        table.add_row(
            str(record.get("recorded_at", ""))[:19],
            str(record.get("title", "")),
            str(record.get("company_name", "")),
            salary,
            str(record.get("applications_count", "")),
        )

    console.print(table)


@app.command()
def db_status(
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
) -> None:
    """
    Show status of scrape sessions in database.
    """
    db = _open_database(_resolve_cli_db_path(db_path), read_only=True)
    sessions = db.get_all_sessions()

    if not sessions:
        console.print("No scrape sessions found")
        return

    console.print("\n[bold blue]Scrape Sessions[/bold blue]\n")

    table = Table(show_header=True)
    table.add_column("ID")
    table.add_column("Query")
    table.add_column("Progress", justify="right")
    table.add_column("Status")
    table.add_column("Started")

    for session in sessions[:20]:  # Show last 20
        progress_str = f"{session['fetched_count']}/{session['total_jobs']}"
        if session["total_jobs"] > 0:
            pct = session["fetched_count"] / session["total_jobs"] * 100
            progress_str += f" ({pct:.0f}%)"

        status_style = {
            "completed": "green",
            "in_progress": "yellow",
            "interrupted": "red",
        }.get(session["status"], "")

        table.add_row(
            str(session["id"]),
            session["search_query"],
            progress_str,
            f"[{status_style}]{session['status']}[/{status_style}]",
            str(session["started_at"])[:16],
        )

    console.print(table)


@app.command()
def migrate(
    data_dir: str = typer.Option("data", "--data-dir", "-d", help="Data directory"),
    json_only: bool = typer.Option(False, "--json-only", help="Only import JSON files"),
    csv_only: bool = typer.Option(False, "--csv-only", help="Only import CSV files"),
    skip_link_only: bool = typer.Option(False, "--skip-link-only", help="Skip link-only CSVs (minimal data)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without importing"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """
    Import legacy MCF data from JSON files and CSVs into the configured database.

    Processes data sources in order for best data quality:
    1. JSON files first (richest data, from data/scrape_jsons/)
    2. Full MCF CSVs second (16 columns with full job data)
    3. Link-only CSVs last (minimal data: Company, Title, Location, Link)

    Examples:
        mcf migrate                    # Import all legacy data
        mcf migrate --json-only        # Only JSON files
        mcf migrate --dry-run          # Preview without importing
        mcf migrate --skip-link-only   # Skip minimal data CSVs
    """
    setup_logging(verbose)
    resolved_db_path = _resolve_cli_db_path(db_path)

    if dry_run:
        console.print("[bold yellow]DRY RUN[/bold yellow] - No data will be imported\n")

    console.print("[bold blue]MCF Legacy Data Migration[/bold blue]")
    console.print(f"Data directory: [green]{data_dir}[/green]")
    console.print(f"Database: [green]{resolved_db_path}[/green]")

    if json_only:
        console.print("Mode: JSON files only")
    elif csv_only:
        console.print("Mode: CSV files only")
    else:
        console.print("Mode: All sources (JSON + CSV)")

    if skip_link_only:
        console.print("Skipping: Link-only CSVs")

    console.print()

    # Get initial stats
    db = _open_database(resolved_db_path, read_only=True)
    initial_count = db.count_jobs()
    console.print(f"Jobs in database before migration: [cyan]{initial_count:,}[/cyan]")
    console.print()

    # Run migration with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Migrating legacy data...", total=None)

        migrator = MCFMigrator(resolved_db_path)
        stats = migrator.migrate_all(
            data_dir=data_dir,
            json_only=json_only,
            csv_only=csv_only,
            skip_link_only=skip_link_only,
            dry_run=dry_run,
        )

        progress.update(task, completed=True)

    # Display results
    console.print("\n[bold green]Migration Complete![/bold green]\n")

    # Stats table
    table = Table(title="Migration Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")

    table.add_row("JSON files processed", f"{stats.json_files_processed:,}")
    table.add_row("CSV rows processed", f"{stats.csv_rows_processed:,}")
    table.add_row("New jobs imported", f"[green]{stats.new_jobs:,}[/green]")
    table.add_row("Jobs updated", f"[yellow]{stats.updated_jobs:,}[/yellow]")
    table.add_row("Skipped (duplicates)", f"{stats.skipped_duplicates:,}")
    table.add_row("Link-only records", f"{stats.link_only_jobs:,}")
    table.add_row("Errors", f"[red]{len(stats.errors):,}[/red]" if stats.errors else "0")

    console.print(table)

    # Final count
    if not dry_run:
        final_count = db.count_jobs()
        console.print(f"\nJobs in database after migration: [cyan]{final_count:,}[/cyan]")
        console.print(f"Net change: [green]+{final_count - initial_count:,}[/green]")

    # Show sample errors if any
    if stats.errors and verbose:
        console.print(f"\n[bold red]Sample Errors ({len(stats.errors)} total):[/bold red]")
        for error in stats.errors[:5]:
            console.print(f"  - {error.source}")
            if error.row is not None:
                console.print(f"    Row {error.row}: {error.error}")
            else:
                console.print(f"    {error.error}")


# Historical scraping commands


@app.command(name="scrape-historical")
def scrape_historical(
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Specific year to scrape (2019-2026)"),
    all_years: bool = typer.Option(False, "--all", help="Scrape all years (2019-2026)"),
    start: Optional[str] = typer.Option(None, "--start", help="Starting jobPostId (e.g., MCF-2023-0500000)"),
    end: Optional[str] = typer.Option(None, "--end", help="Ending jobPostId (e.g., MCF-2023-0600000)"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from previous checkpoint"),
    rate_limit: float = typer.Option(2.0, "--rate-limit", "-r", help="Requests per second"),
    max_rate_limit_retries: int = typer.Option(
        4,
        "--max-rate-limit-retries",
        help="Per-sequence retry cap for 429 responses",
    ),
    cooldown_seconds: float = typer.Option(
        30.0,
        "--cooldown-seconds",
        help="Global cooldown after repeated 429 responses",
    ),
    discover_bounds: bool = typer.Option(
        True,
        "--discover-bounds/--no-discover-bounds",
        help="Discover a tighter end bound before scanning",
    ),
    not_found_threshold: int = typer.Option(1000, "--not-found-threshold", help="Stop after N consecutive not-found"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without fetching"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """
    Scrape historical jobs by enumerating job IDs.

    This scrapes jobs from the MCF archive by generating all possible job IDs
    (MCF-YYYY-NNNNNNN format) and fetching each one. Jobs are stored in the
    configured database backend
    with automatic deduplication.

    Examples:
        mcf scrape-historical --year 2023
        mcf scrape-historical --all
        mcf scrape-historical --start MCF-2023-0500000 --end MCF-2023-0600000
        mcf scrape-historical --resume  # Resume any incomplete session
    """
    setup_logging(verbose)
    resolved_db_path = _resolve_cli_db_path(db_path)

    # Validate options
    option_count = sum([year is not None, all_years, start is not None])
    if option_count == 0:
        console.print("[red]Error: Must specify --year, --all, or --start/--end[/red]")
        raise typer.Exit(1)
    if option_count > 1 and not (start and end and not year and not all_years):
        console.print("[red]Error: Use only one of --year, --all, or --start/--end[/red]")
        raise typer.Exit(1)

    if start and not end:
        console.print("[red]Error: --start requires --end[/red]")
        raise typer.Exit(1)
    if end and not start:
        console.print("[red]Error: --end requires --start[/red]")
        raise typer.Exit(1)

    console.print("\n[bold blue]MCF Historical Scraper[/bold blue]")

    if dry_run:
        console.print("[yellow]DRY RUN[/yellow] - No data will be fetched")

    if year:
        console.print(f"Year: [green]{year}[/green]")
        estimated = YEAR_ESTIMATES.get(year, 1_000_000)
        console.print(f"Estimated jobs: ~{estimated:,}")
    elif all_years:
        console.print("Scraping: [green]All years (2019-2026)[/green]")
        total_estimated = sum(YEAR_ESTIMATES.values())
        console.print(f"Total estimated jobs: ~{total_estimated:,}")
    elif start and end:
        console.print(f"Range: [green]{start}[/green] to [green]{end}[/green]")

    console.print(f"Rate limit: {rate_limit} req/sec")
    console.print(f"Max 429 retries: {max_rate_limit_retries}")
    console.print(f"429 cooldown: {cooldown_seconds:.1f}s")
    console.print(f"Discover bounds: {'yes' if discover_bounds else 'no'}")
    console.print(f"Database: {resolved_db_path}")
    console.print()

    async def run():
        async with HistoricalScraper(
            db_path=resolved_db_path,
            requests_per_second=rate_limit,
            not_found_threshold=not_found_threshold,
            max_rate_limit_retries=max_rate_limit_retries,
            cooldown_seconds=cooldown_seconds,
            discover_bounds=discover_bounds,
        ) as scraper:
            # Set up progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.fields[stats]}"),
                console=console,
                transient=True,
            ) as progress:
                task_id = None

                async def update_progress(p):
                    nonlocal task_id
                    if task_id is None:
                        total = p.end_seq - p.start_seq + 1 if p.end_seq else None
                        task_id = progress.add_task(
                            f"Year {p.year}",
                            total=total,
                            stats="",
                        )
                    completed = p.current_seq - p.start_seq
                    stats = f"Found: {p.jobs_found:,} | Not found: {p.jobs_not_found:,}"
                    progress.update(task_id, completed=completed, stats=stats)

                if year:
                    result = await scraper.scrape_year(
                        year,
                        resume=resume,
                        progress_callback=update_progress,
                        dry_run=dry_run,
                    )
                    results = {year: result}

                elif all_years:
                    results = {}
                    for y in sorted(YEAR_ESTIMATES.keys()):
                        task_id = None  # Reset for new year
                        results[y] = await scraper.scrape_year(
                            y,
                            resume=resume,
                            progress_callback=update_progress,
                            dry_run=dry_run,
                        )

                elif start and end:
                    result = await scraper.scrape_range(
                        start,
                        end,
                        progress_callback=update_progress,
                        dry_run=dry_run,
                    )
                    results = {result.year: result}

        # Display results
        console.print("\n[bold green]Scrape Complete![/bold green]\n")

        table = Table(title="Results by Year", show_header=True)
        table.add_column("Year", style="cyan")
        table.add_column("Jobs Found", justify="right", style="green")
        table.add_column("Not Found", justify="right", style="dim")
        table.add_column("Success Rate", justify="right")
        table.add_column("Last Seq", justify="right")

        total_found = 0
        total_not_found = 0

        for y, p in sorted(results.items()):
            total_found += p.jobs_found
            total_not_found += p.jobs_not_found

            table.add_row(
                str(y),
                f"{p.jobs_found:,}",
                f"{p.jobs_not_found:,}",
                f"{p.success_rate:.1f}%",
                f"{p.current_seq:,}",
            )

        console.print(table)

        console.print(f"\n[bold]Total jobs found:[/bold] [green]{total_found:,}[/green]")
        console.print(f"[bold]Total not found:[/bold] {total_not_found:,}")

        if not dry_run:
            # Show current database stats
            db = _open_database(resolved_db_path)
            total_jobs = db.count_jobs()
            console.print(f"\n[bold]Total jobs in database:[/bold] [cyan]{total_jobs:,}[/cyan]")

    asyncio.run(run())


@app.command(name="historical-status")
def historical_status(
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
) -> None:
    """
    Show status of historical scrape sessions.

    Displays progress for each year being scraped, including jobs found,
    not found, and current sequence position.
    """
    db = _open_database(_resolve_cli_db_path(db_path), read_only=True)

    # Get sessions
    sessions = db.get_all_historical_sessions()

    if not sessions:
        console.print("No historical scrape sessions found")
        console.print("\nRun [bold]mcf scrape-historical --year 2023[/bold] to start scraping")
        return

    console.print("\n[bold blue]Historical Scrape Status[/bold blue]\n")

    # Active sessions
    active = [s for s in sessions if s["status"] == "in_progress"]
    if active:
        console.print("[bold]Active Sessions:[/bold]\n")

        table = Table(show_header=True)
        table.add_column("ID")
        table.add_column("Year", style="cyan")
        table.add_column("Progress", justify="right")
        table.add_column("Found", justify="right", style="green")
        table.add_column("Not Found", justify="right")
        table.add_column("Consecutive NF", justify="right", style="dim")
        table.add_column("Started")

        for s in active:
            end_seq = s["end_seq"] or YEAR_ESTIMATES.get(s["year"], 1_000_000)
            progress_pct = (s["current_seq"] - s["start_seq"]) / (end_seq - s["start_seq"]) * 100
            progress_str = f"{s['current_seq']:,}/{end_seq:,} ({progress_pct:.1f}%)"

            table.add_row(
                str(s["id"]),
                str(s["year"]),
                progress_str,
                f"{s['jobs_found']:,}",
                f"{s['jobs_not_found']:,}",
                str(s["consecutive_not_found"]),
                str(s["started_at"])[:16] if s["started_at"] else "",
            )

        console.print(table)
        console.print()

    # Completed sessions
    completed = [s for s in sessions if s["status"] == "completed"]
    if completed:
        console.print("[bold]Completed Sessions:[/bold]\n")

        table = Table(show_header=True)
        table.add_column("Year", style="cyan")
        table.add_column("Jobs Found", justify="right", style="green")
        table.add_column("Max Seq", justify="right")
        table.add_column("Completed")

        for s in completed:
            table.add_row(
                str(s["year"]),
                f"{s['jobs_found']:,}",
                f"{s['current_seq']:,}",
                str(s["completed_at"])[:16] if s["completed_at"] else "",
            )

        console.print(table)
        console.print()

    # Summary stats
    stats = db.get_historical_stats()

    if stats.get("jobs_by_year"):
        console.print("[bold]Jobs in Database by Year:[/bold]\n")

        table = Table(show_header=True)
        table.add_column("Year", style="cyan")
        table.add_column("Jobs", justify="right", style="green")
        table.add_column("Est. Total", justify="right", style="dim")
        table.add_column("Coverage", justify="right")

        for year, count in sorted(stats["jobs_by_year"].items(), reverse=True):
            estimated = YEAR_ESTIMATES.get(int(year), 0)
            coverage = (count / estimated * 100) if estimated else 0
            table.add_row(
                year,
                f"{count:,}",
                f"~{estimated:,}" if estimated else "?",
                f"{coverage:.1f}%",
            )

        console.print(table)

    # Instructions
    if active:
        console.print("\nResume with: [bold]mcf scrape-historical --year YEAR[/bold]")


# Daemon commands


@app.command(name="daemon")
def daemon_cmd(
    action: str = typer.Argument(..., help="Action: start, stop, or status"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Year to scrape (for start action)"),
    all_years: bool = typer.Option(False, "--all", help="Scrape all years (for start action)"),
    rate_limit: float = typer.Option(2.0, "--rate-limit", "-r", help="Initial requests per second"),
    max_rate_limit_retries: int = typer.Option(
        4,
        "--max-rate-limit-retries",
        help="Per-sequence retry cap for 429 responses",
    ),
    cooldown_seconds: float = typer.Option(
        30.0,
        "--cooldown-seconds",
        help="Global cooldown after repeated 429 responses",
    ),
    discover_bounds: bool = typer.Option(
        True,
        "--discover-bounds/--no-discover-bounds",
        help="Discover a tighter end bound before scanning",
    ),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """
    Manage the background scraper daemon.

    The daemon runs historical scraping in the background, surviving terminal
    closure and detecting sleep/wake cycles.

    Examples:
        mcf daemon start --year 2023     # Start scraping 2023 in background
        mcf daemon start --all           # Start scraping all years
        mcf daemon status                # Check daemon status
        mcf daemon stop                  # Stop the daemon
    """
    setup_logging(verbose)
    resolved_db_path = _resolve_daemon_db_path(db_path)

    if action == "status":
        db = _open_database(resolved_db_path, read_only=True)
        daemon = ScraperDaemon(db)
        status = daemon.status()

        console.print("\n[bold blue]Daemon Status[/bold blue]\n")

        if status["running"]:
            console.print(f"[green]● Running[/green] (PID {status['pid']})")
        else:
            console.print("[dim]○ Stopped[/dim]")

        console.print(f"PID file: {status['pidfile']}")
        console.print(f"Log file: {status['logfile']}")

        if status.get("last_heartbeat"):
            console.print(f"\nLast heartbeat: {status['last_heartbeat']}")

        if status.get("current_year"):
            console.print(f"Current year: {status['current_year']}")
            if status.get("current_seq"):
                console.print(f"Current sequence: {status['current_seq']:,}")

        if status.get("started_at"):
            console.print(f"Started at: {status['started_at']}")

    elif action == "start":
        if not year and not all_years:
            console.print("[red]Error: Must specify --year or --all for start action[/red]")
            raise typer.Exit(1)
        db = _open_database(resolved_db_path, read_only=True)
        daemon = ScraperDaemon(db)

        try:
            console.print("\n[bold blue]Starting Daemon[/bold blue]")

            if year:
                console.print(f"Year: [green]{year}[/green]")
            elif all_years:
                console.print("Mode: [green]All years (2019-2026)[/green]")

            console.print(f"Rate limit: {rate_limit} req/sec")
            console.print(f"Max 429 retries: {max_rate_limit_retries}")
            console.print(f"429 cooldown: {cooldown_seconds:.1f}s")
            console.print(f"Discover bounds: {'yes' if discover_bounds else 'no'}")
            console.print(f"Database: {resolved_db_path}")
            console.print()

            pid = daemon.start(
                year=year,
                all_years=all_years,
                rate_limit=rate_limit,
                db_path=resolved_db_path,
                max_rate_limit_retries=max_rate_limit_retries,
                cooldown_seconds=cooldown_seconds,
                discover_bounds=discover_bounds,
            )
            console.print(f"[green]Daemon started with PID {pid}[/green]")
            console.print(f"\nLogs: [cyan]{daemon.logfile}[/cyan]")
            console.print("\nCheck status: [bold]mcf daemon status[/bold]")
            console.print("Stop daemon: [bold]mcf daemon stop[/bold]")

        except DaemonAlreadyRunning as e:
            console.print(f"[yellow]{e}[/yellow]")
            console.print("Use [bold]mcf daemon stop[/bold] first")
            raise typer.Exit(1)
        except DaemonError as e:
            console.print(f"[red]{e}[/red]")
            console.print("Stop the other scrape process before starting the daemon.")
            raise typer.Exit(1)

    elif action == "stop":
        db = _open_database(resolved_db_path, read_only=True)
        daemon = ScraperDaemon(db)
        try:
            console.print("Stopping daemon...")
            daemon.stop()
            console.print("[green]Daemon stopped[/green]")
        except DaemonNotRunning:
            console.print("[yellow]No daemon is running[/yellow]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: start, stop, status")
        raise typer.Exit(1)


@app.command(name="_daemon-worker", hidden=True)
def daemon_worker(
    year: Optional[int] = typer.Option(None, "--year", "-y"),
    all_years: bool = typer.Option(False, "--all"),
    rate_limit: float = typer.Option(2.0, "--rate-limit", "-r"),
    max_rate_limit_retries: int = typer.Option(4, "--max-rate-limit-retries"),
    cooldown_seconds: float = typer.Option(30.0, "--cooldown-seconds"),
    discover_bounds: bool = typer.Option(True, "--discover-bounds/--no-discover-bounds"),
    db_path: Optional[str] = typer.Option(None, "--db"),
    pidfile: str = typer.Option("data/.scraper.pid", "--pidfile"),
    logfile: str = typer.Option("data/scraper_daemon.log", "--logfile"),
    heartbeat_interval: int = typer.Option(DEFAULT_HEARTBEAT_INTERVAL, "--heartbeat-interval"),
    wake_threshold: int = typer.Option(DEFAULT_WAKE_THRESHOLD, "--wake-threshold"),
) -> None:
    """Internal: daemon worker process. Do not call directly."""
    resolved_db_path = _resolve_daemon_db_path(db_path)
    db_target = resolve_database_target(resolved_db_path)
    db_exists = db_target.is_postgres or db_target.sqlite_path.exists()
    try:
        db = _open_database(resolved_db_path)
    except sqlite3.OperationalError as exc:
        if not db_exists or "locked" not in str(exc).lower():
            raise
        db = _open_database(resolved_db_path, ensure_schema=False)
    daemon = ScraperDaemon(
        db,
        pidfile=pidfile,
        logfile=logfile,
        heartbeat_interval=heartbeat_interval,
        wake_threshold=wake_threshold,
    )

    async def run_scraper():
        async with HistoricalScraper(
            db_path=resolved_db_path,
            requests_per_second=rate_limit,
            max_rate_limit_retries=max_rate_limit_retries,
            cooldown_seconds=cooldown_seconds,
            discover_bounds=discover_bounds,
        ) as scraper:
            if year:
                return await scraper.scrape_year(year, resume=True)
            else:
                return await scraper.scrape_all_years(resume=True)

    daemon.run_worker(run_scraper)


# Gap analysis commands


@app.command(name="gaps")
def show_gaps(
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Specific year to check"),
    all_years: bool = typer.Option(False, "--all", help="Check all years"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
) -> None:
    """
    Show gaps in scraped job sequences.

    Analyzes fetch_attempts to find missing sequence ranges that need
    to be retried.

    Examples:
        mcf gaps --year 2023     # Check gaps for 2023
        mcf gaps --all           # Check all years
    """
    db = _open_database(_resolve_cli_db_path(db_path), read_only=True)

    if not year and not all_years:
        console.print("[red]Error: Must specify --year or --all[/red]")
        raise typer.Exit(1)

    console.print("\n[bold blue]Gap Analysis[/bold blue]\n")

    years_to_check = [year] if year else sorted(YEAR_ESTIMATES.keys())

    total_gaps = 0
    total_missing = 0
    total_retryable = 0

    for y in years_to_check:
        gaps = db.get_missing_sequences(y)
        failed = db.get_failed_attempts(y)
        stats = db.get_attempt_stats(y)

        if not stats.get("total"):
            if year:  # Only show message if specific year requested
                console.print(f"[yellow]No fetch attempts recorded for year {y}[/yellow]")
            continue

        missing_count = sum(end - start + 1 for start, end in gaps)
        total_gaps += len(gaps)
        total_missing += missing_count
        total_retryable += len(failed)

        console.print(f"[bold]Year {y}:[/bold]")
        console.print(f"  Sequences attempted: {stats.get('total', 0):,}")
        console.print(f"  Found: [green]{stats.get('found', 0):,}[/green]")
        console.print(f"  Not found: {stats.get('not_found', 0):,}")
        console.print(f"  Skipped: {stats.get('skipped', 0):,}")
        console.print(f"  Errors: [red]{stats.get('error', 0):,}[/red]")
        console.print(f"  Rate limited: [yellow]{stats.get('rate_limited', 0):,}[/yellow]")

        if gaps:
            console.print(f"\n  [yellow]Gaps ({len(gaps)} ranges, {missing_count:,} sequences):[/yellow]")
            for start, end in gaps[:5]:  # Show first 5 gaps
                console.print(f"    {start:,} - {end:,} ({end - start + 1:,} missing)")
            if len(gaps) > 5:
                console.print(f"    ... and {len(gaps) - 5} more gaps")
        else:
            console.print("  [green]No gaps detected[/green]")

        if failed:
            console.print(f"\n  [red]Retryable attempts: {len(failed):,}[/red]")

        console.print()

    # Summary
    if all_years:
        console.print("[bold]Summary:[/bold]")
        console.print(f"  Total gaps: {total_gaps}")
        console.print(f"  Total missing sequences: {total_missing:,}")
        console.print(f"  Total retryable attempts: {total_retryable:,}")

        if total_missing + total_retryable > 0:
            console.print("\nRun [bold]mcf retry-gaps --all[/bold] to retry missing sequences")


@app.command(name="retry-gaps")
def retry_gaps(
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Specific year to retry"),
    all_years: bool = typer.Option(False, "--all", help="Retry all years"),
    rate_limit: float = typer.Option(2.0, "--rate-limit", "-r", help="Requests per second"),
    max_rate_limit_retries: int = typer.Option(
        4,
        "--max-rate-limit-retries",
        help="Per-sequence retry cap for 429 responses",
    ),
    cooldown_seconds: float = typer.Option(
        30.0,
        "--cooldown-seconds",
        help="Global cooldown after repeated 429 responses",
    ),
    discover_bounds: bool = typer.Option(
        True,
        "--discover-bounds/--no-discover-bounds",
        help="Discover tighter year bounds when initializing the scraper",
    ),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """
    Retry fetching jobs for missing/failed sequences.

    Finds gaps in fetch_attempts and retries each sequence.

    Examples:
        mcf retry-gaps --year 2023     # Retry gaps for 2023
        mcf retry-gaps --all           # Retry all years
    """
    setup_logging(verbose)
    resolved_db_path = _resolve_cli_db_path(db_path)

    if not year and not all_years:
        console.print("[red]Error: Must specify --year or --all[/red]")
        raise typer.Exit(1)

    console.print("\n[bold blue]Retrying Gaps[/bold blue]\n")

    async def run():
        async with HistoricalScraper(
            db_path=resolved_db_path,
            requests_per_second=rate_limit,
            max_rate_limit_retries=max_rate_limit_retries,
            cooldown_seconds=cooldown_seconds,
            discover_bounds=discover_bounds,
        ) as scraper:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.fields[stats]}"),
                console=console,
            ) as progress:
                task_id = None

                async def update_progress(p):
                    nonlocal task_id
                    if task_id is None:
                        task_id = progress.add_task(
                            f"Year {p.year}",
                            total=p.end_seq - p.start_seq + 1 if p.end_seq else None,
                            stats="",
                        )
                    completed = p.current_seq - p.start_seq if p.start_seq else 0
                    stats = f"Recovered: {p.jobs_found:,}"
                    progress.update(task_id, completed=completed, stats=stats)

                if year:
                    result = await scraper.retry_gaps(year, update_progress)
                    results = {year: result}
                else:
                    results = {}
                    for y in sorted(YEAR_ESTIMATES.keys()):
                        task_id = None
                        results[y] = await scraper.retry_gaps(y, update_progress)

        # Display results
        console.print("\n[bold green]Gap Retry Complete![/bold green]\n")

        table = Table(title="Results by Year", show_header=True)
        table.add_column("Year", style="cyan")
        table.add_column("Recovered", justify="right", style="green")
        table.add_column("Still Missing", justify="right", style="dim")

        total_recovered = 0
        total_missing = 0

        for y, p in sorted(results.items()):
            if p.jobs_found > 0 or p.jobs_not_found > 0:
                total_recovered += p.jobs_found
                total_missing += p.jobs_not_found

                table.add_row(
                    str(y),
                    f"{p.jobs_found:,}",
                    f"{p.jobs_not_found:,}",
                )

        if total_recovered > 0 or total_missing > 0:
            console.print(table)
            console.print(f"\n[bold]Total recovered:[/bold] [green]{total_recovered:,}[/green]")
        else:
            console.print("[green]No gaps to retry![/green]")

    asyncio.run(run())


@app.command(name="attempt-stats")
def attempt_stats(
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Specific year to show"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
) -> None:
    """
    Show fetch attempt statistics.

    Displays counts of found, not_found, error, and skipped attempts
    for tracking scraper completeness.

    Examples:
        mcf attempt-stats              # All years summary
        mcf attempt-stats --year 2023  # Specific year details
    """
    db = _open_database(_resolve_cli_db_path(db_path), read_only=True)

    console.print("\n[bold blue]Fetch Attempt Statistics[/bold blue]\n")

    if year:
        stats = db.get_attempt_stats(year)

        if not stats.get("total"):
            console.print(f"[yellow]No fetch attempts recorded for year {year}[/yellow]")
            return

        console.print(f"[bold]Year {year}:[/bold]")
        console.print(f"  Total attempts: {stats['total']:,}")
        console.print(f"  Found: [green]{stats.get('found', 0):,}[/green]")
        console.print(f"  Not found: {stats.get('not_found', 0):,}")
        console.print(f"  Skipped: {stats.get('skipped', 0):,}")
        console.print(f"  Errors: [red]{stats.get('error', 0):,}[/red]")
        console.print(f"  Rate limited: [yellow]{stats.get('rate_limited', 0):,}[/yellow]")

        if stats.get("min_sequence"):
            console.print(f"\n  Sequence range: {stats['min_sequence']:,} - {stats['max_sequence']:,}")
            console.print(f"  Range size: {stats['sequence_range']:,}")

            coverage = stats["total"] / stats["sequence_range"] * 100
            console.print(f"  Coverage: {coverage:.1f}%")

    else:
        all_stats = db.get_all_attempt_stats()

        if not all_stats:
            console.print("[yellow]No fetch attempts recorded[/yellow]")
            return

        table = Table(show_header=True)
        table.add_column("Year", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Found", justify="right", style="green")
        table.add_column("Not Found", justify="right")
        table.add_column("Skipped", justify="right", style="dim")
        table.add_column("Errors", justify="right", style="red")

        grand_total = 0
        grand_found = 0

        for y in sorted(all_stats.keys()):
            stats = all_stats[y]
            grand_total += stats.get("total", 0)
            grand_found += stats.get("found", 0)

            table.add_row(
                str(y),
                f"{stats.get('total', 0):,}",
                f"{stats.get('found', 0):,}",
                f"{stats.get('not_found', 0):,}",
                f"{stats.get('skipped', 0):,}",
                f"{stats.get('error', 0):,}",
            )

        console.print(table)
        console.print(f"\n[bold]Grand total:[/bold] {grand_total:,} attempts, {grand_found:,} jobs found")


# Embedding commands


@app.command(name="embed-generate")
def generate_embeddings(
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Jobs to process in each batch"),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip jobs that already have embeddings",
    ),
    build_index: bool = typer.Option(
        True,
        "--build-index/--no-build-index",
        help="Build FAISS indexes after embedding generation",
    ),
    index_dir: str = typer.Option("data/embeddings", "--index-dir", help="Directory for FAISS index files"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    since: Optional[str] = typer.Option(
        None, "--since", help="Only embed jobs posted on or after this date (YYYY-MM-DD)"
    ),
    embedding_backend: str = typer.Option(
        CLI_DEFAULT_EMBEDDING_BACKEND,
        "--embedding-backend",
        help="Embedding inference backend: torch or onnx",
    ),
    onnx_model_dir: Optional[str] = typer.Option(
        None,
        "--onnx-model-dir",
        help="Exported ONNX model directory when using --embedding-backend onnx",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Generate embeddings for all jobs and build FAISS indexes.

    This preprocesses all jobs to create semantic embeddings for similarity search,
    then builds FAISS indexes for efficient nearest-neighbor lookup.

    Examples:
        mcf embed-generate
        mcf embed-generate --batch-size 64
        mcf embed-generate --no-skip-existing  # Regenerate all
        mcf embed-generate --no-build-index    # Skip index building
        mcf embed-generate --since 2026-01-01  # Only 2026 jobs
    """
    setup_logging(verbose)
    since_date = _parse_since_date(since)
    resolved_db_path = _resolve_cli_db_path(db_path)

    console.print("\n[bold blue]Generating Embeddings[/bold blue]")
    console.print("━" * 40)

    db = _open_database(resolved_db_path)
    generator = _create_embedding_generator(
        backend=embedding_backend,
        onnx_model_dir=onnx_model_dir,
    )

    console.print(f"Model: [green]{generator.model_name}[/green] ({generator.DIMENSION} dimensions)")
    console.print(f"Backend: [green]{generator.backend_name}[/green]")
    console.print(f"Database: [green]{resolved_db_path}[/green]")
    if generator.backend_name == "onnx" and generator.onnx_model_dir is not None:
        console.print(f"ONNX dir: [green]{generator.onnx_model_dir}[/green]")
    if since_date:
        console.print(f"Since: [green]{since_date.isoformat()}[/green]")
    console.print()

    # Get initial count for progress
    if skip_existing:
        jobs_to_process = db.get_jobs_without_embeddings(
            limit=1000000,
            since=since_date,
            model_version=generator.model_name,
        )
        total_jobs = len(jobs_to_process)
        if total_jobs == 0:
            console.print("[green]✓ All jobs already have embeddings![/green]")
            # Still build index if requested and we have embeddings
            if build_index:
                emb_stats = db.get_embedding_stats()
                if emb_stats["job_embeddings"] > 0:
                    _build_faiss_indexes(db, generator, index_dir, console)
            return
        console.print(f"Jobs to process: [cyan]{total_jobs:,}[/cyan] (skipping existing)")
    else:
        if since_date:
            total_jobs = db.count_jobs_since(since_date)
        else:
            total_jobs = db.count_jobs()
        console.print(f"Jobs to process: [cyan]{total_jobs:,}[/cyan] (regenerating all)")

    console.print()

    # Phase 1: Generate embeddings
    console.print("[bold]Phase 1: Generating Embeddings[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed:,}/{task.total:,}"),
        TextColumn("[dim]({task.fields[speed]:.1f} jobs/sec)"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Jobs", total=total_jobs, speed=0.0)

        def on_progress(stats: EmbeddingStats):
            progress.update(
                task_id,
                completed=stats.jobs_processed,
                speed=stats.jobs_per_second,
            )

        stats = generator.generate_all(
            db,
            batch_size=batch_size,
            skip_existing=skip_existing,
            progress_callback=on_progress,
            since=since_date,
        )

    # Print embedding summary
    console.print("\n[bold green]✓ Embedding generation complete![/bold green]\n")

    table = Table(title="Embedding Summary", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Jobs processed", f"{stats.jobs_processed:,}")
    table.add_row("Jobs skipped", f"{stats.jobs_skipped:,}")
    if stats.jobs_failed > 0:
        table.add_row("Jobs failed", f"[red]{stats.jobs_failed:,}[/red]")
    table.add_row("Skills extracted", f"{stats.unique_skills:,}")
    table.add_row("Skill clusters", f"{stats.skill_clusters:,}")

    # Format elapsed time
    elapsed = stats.elapsed_seconds
    if elapsed >= 60:
        minutes, seconds = divmod(int(elapsed), 60)
        elapsed_str = f"{minutes}m {seconds}s"
    else:
        elapsed_str = f"{elapsed:.1f}s"
    table.add_row("Total time", elapsed_str)
    table.add_row("Speed", f"{stats.jobs_per_second:.1f} jobs/sec")

    console.print(table)

    # Phase 2: Build FAISS indexes
    if not build_index:
        console.print("\n[yellow]Skipping index building (--no-build-index)[/yellow]")
        return

    _build_faiss_indexes(db, generator, index_dir, console)


def _build_faiss_indexes(
    db: MCFDatabase,
    generator: EmbeddingGenerator,
    index_dir: str,
    console: Console,
) -> None:
    """Build FAISS indexes from embeddings in database."""

    console.print("\n[bold]Phase 2: Building FAISS Indexes[/bold]\n")

    index_path = Path(index_dir)
    index_manager = FAISSIndexManager(
        index_dir=index_path,
        model_version=generator.model_name,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load job embeddings from database
        task = progress.add_task("Loading job embeddings from database...", total=None)
        job_uuids, job_embeddings = db.get_all_embeddings("job", model_version=generator.model_name)

        if len(job_uuids) == 0:
            progress.update(task, description="[yellow]No job embeddings found[/yellow]")
            console.print("\n[yellow]No job embeddings found. Run embed-generate first.[/yellow]")
            return

        # Build job index
        progress.update(task, description=f"Building job index ({len(job_uuids):,} vectors)...")
        index_manager.build_job_index(job_embeddings, job_uuids)

        # Load and build skill index
        progress.update(task, description="Loading skill embeddings...")
        skill_names, skill_embeddings = db.get_all_embeddings("skill", model_version=generator.model_name)

        if len(skill_names) > 0:
            progress.update(task, description=f"Building skill index ({len(skill_names):,} skills)...")
            index_manager.build_skill_index(skill_embeddings, skill_names)

        # Generate company centroids and build company index
        progress.update(task, description="Generating company centroids...")
        company_centroids = generator.generate_company_centroids_from_db(db)
        if company_centroids:
            total_centroids = sum(len(c) for c in company_centroids.values())
            progress.update(
                task,
                description=(
                    f"Building company index ({len(company_centroids):,} companies, {total_centroids:,} centroids)..."
                ),
            )
            index_manager.build_company_index(company_centroids)

        # Save indexes to disk
        progress.update(task, description="Saving indexes to disk...")
        index_manager.save()

        progress.update(task, description="[green]✓ Complete[/green]")

    # Print index summary
    console.print("\n[bold green]✓ FAISS indexes built![/bold green]\n")

    index_stats = index_manager.get_stats()
    table = Table(title="Index Summary", show_header=True)
    table.add_column("Index", style="cyan")
    table.add_column("Vectors", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Type")

    if "jobs" in index_stats["indexes"]:
        job_idx = index_stats["indexes"]["jobs"]
        table.add_row(
            "jobs.index",
            f"{job_idx['total_vectors']:,}",
            f"{job_idx['estimated_memory_mb']:.1f} MB",
            job_idx["index_type"],
        )

    if "skills" in index_stats["indexes"]:
        skill_idx = index_stats["indexes"]["skills"]
        table.add_row(
            "skills.index",
            f"{skill_idx['total_skills']:,}",
            "-",
            skill_idx["index_type"],
        )

    if "companies" in index_stats["indexes"]:
        co_idx = index_stats["indexes"]["companies"]
        table.add_row(
            "companies.index",
            f"{co_idx['total_companies']:,} ({co_idx['total_centroids']:,} centroids)",
            "-",
            co_idx["index_type"],
        )

    console.print(table)
    console.print(f"\nIndexes saved to: [cyan]{index_dir}[/cyan]")


@app.command(name="embed-sync")
def sync_embeddings(
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Jobs to process in each batch"),
    update_index: bool = typer.Option(
        True,
        "--update-index/--no-update-index",
        help="Update FAISS indexes with new embeddings",
    ),
    index_dir: str = typer.Option("data/embeddings", "--index-dir", help="Directory for FAISS index files"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    since: Optional[str] = typer.Option(
        None, "--since", help="Only sync jobs posted on or after this date (YYYY-MM-DD)"
    ),
    embedding_backend: str = typer.Option(
        CLI_DEFAULT_EMBEDDING_BACKEND,
        "--embedding-backend",
        help="Embedding inference backend: torch or onnx",
    ),
    onnx_model_dir: Optional[str] = typer.Option(
        None,
        "--onnx-model-dir",
        help="Exported ONNX model directory when using --embedding-backend onnx",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Generate embeddings for new jobs and update indexes.

    Only processes jobs that don't have embeddings yet, then adds
    them to existing FAISS indexes for incremental updates.

    Examples:
        mcf embed-sync
        mcf scrape "data scientist" && mcf embed-sync  # Chain commands
        mcf embed-sync --no-update-index  # Skip index update
        mcf embed-sync --since 2026-02-20  # Only recent jobs
    """
    setup_logging(verbose)
    since_date = _parse_since_date(since)
    resolved_db_path = _resolve_cli_db_path(db_path)

    console.print("\n[bold blue]Syncing Embeddings[/bold blue]")
    console.print("━" * 40)

    db = _open_database(resolved_db_path)
    generator = _create_embedding_generator(
        backend=embedding_backend,
        onnx_model_dir=onnx_model_dir,
    )

    if since_date:
        console.print(f"Since: [green]{since_date.isoformat()}[/green]")
    console.print(f"Backend: [green]{generator.backend_name}[/green]")

    # Check how many jobs need embeddings
    jobs_to_process = db.get_jobs_without_embeddings(
        limit=1000000,
        since=since_date,
        model_version=generator.model_name,
    )
    total_new = len(jobs_to_process)

    if total_new == 0:
        console.print("[green]✓ All jobs are up-to-date![/green]")
        return

    console.print(f"New jobs to embed: [cyan]{total_new:,}[/cyan]")
    console.print()

    # Collect new UUIDs for index update
    new_uuids = [job["uuid"] for job in jobs_to_process]

    # Run with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed:,}/{task.total:,}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Embedding", total=total_new)

        def on_progress(stats: EmbeddingStats):
            progress.update(task_id, completed=stats.jobs_processed)

        stats = generator.generate_all(
            db,
            batch_size=batch_size,
            skip_existing=True,
            progress_callback=on_progress,
            since=since_date,
        )

    console.print(f"\n[green]✓ Synced {stats.jobs_processed:,} jobs in {stats.elapsed_seconds:.1f}s[/green]")

    # Update FAISS index if requested
    if not update_index:
        console.print("[yellow]Skipping index update (--no-update-index)[/yellow]")
        return

    _update_faiss_index(db, generator, index_dir, new_uuids, console)


def _update_faiss_index(
    db: MCFDatabase,
    generator: EmbeddingGenerator,
    index_dir: str,
    new_uuids: list[str],
    console: Console,
) -> None:
    """Add new embeddings to existing FAISS index."""
    import numpy as np

    index_path = Path(index_dir)
    index_manager = FAISSIndexManager(
        index_dir=index_path,
        model_version=generator.model_name,
    )

    # Try to load existing index
    if not index_manager.exists():
        console.print("\n[yellow]No existing FAISS index found.[/yellow]")
        console.print("Run [bold]mcf embed-generate[/bold] to build indexes.")
        return

    console.print("\n[bold]Updating FAISS Index[/bold]")

    try:
        index_manager.load()
    except IndexCompatibilityError as e:
        console.print(f"\n[red]Index compatibility error: {e}[/red]")
        console.print("Run [bold]mcf embed-generate --no-skip-existing[/bold] to rebuild.")
        return

    # Get embeddings for new jobs
    embeddings_dict = db.get_embeddings_for_uuids(new_uuids, model_version=generator.model_name)

    if not embeddings_dict:
        console.print("[yellow]No new embeddings to add to index[/yellow]")
        return

    # Prepare arrays for add_jobs
    uuids = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[uuid] for uuid in uuids], dtype=np.float32)

    # Add to index
    index_manager.add_jobs(embeddings, uuids)
    index_manager.save()

    console.print(f"[green]✓ Added {len(uuids):,} jobs to FAISS index[/green]")
    console.print(f"Total jobs in index: [cyan]{index_manager.indexes['jobs'].ntotal:,}[/cyan]")


@app.command(name="embed-status")
def embedding_status(
    index_dir: str = typer.Option("data/embeddings", "--index-dir", help="Directory for FAISS index files"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
) -> None:
    """
    Show embedding and FAISS index status.

    Displays statistics about embeddings in the database and FAISS indexes on disk.

    Example:
        mcf embed-status
    """
    db = _open_database(_resolve_cli_db_path(db_path))
    stats = db.get_embedding_stats()

    console.print("\n[bold blue]Embedding Status[/bold blue]")
    console.print("━" * 40)

    # Jobs section
    console.print("\n[bold]Jobs:[/bold]")
    console.print(f"  Total in database:     {stats['total_jobs']:,}")
    console.print(f"  With embeddings:       {stats['job_embeddings']:,}")

    coverage = stats["coverage_pct"]
    coverage_style = "green" if coverage >= 95 else "yellow" if coverage >= 80 else "red"
    console.print(f"  Coverage:              [{coverage_style}]{coverage:.1f}%[/{coverage_style}]")

    if stats["model_version"]:
        console.print(f"  Model version:         {stats['model_version']}")

    # Skills section
    console.print("\n[bold]Skills:[/bold]")
    console.print(f"  With embeddings:       {stats['skill_embeddings']:,}")

    # Companies section (if implemented)
    if stats.get("company_embeddings", 0) > 0:
        console.print("\n[bold]Companies:[/bold]")
        console.print(f"  With embeddings:       {stats['company_embeddings']:,}")

    # FAISS Index section
    console.print("\n[bold]FAISS Indexes:[/bold]")

    index_path = Path(index_dir)
    index_manager = FAISSIndexManager(
        index_dir=index_path,
        model_version=stats.get("model_version") or "all-MiniLM-L6-v2",
    )

    if index_manager.exists():
        try:
            index_manager.load()
            index_stats = index_manager.get_stats()

            table = Table(show_header=True, box=None)
            table.add_column("Index", style="cyan")
            table.add_column("Vectors", justify="right")
            table.add_column("Memory", justify="right")
            table.add_column("Status")

            # Job index
            if "jobs" in index_stats["indexes"]:
                job_idx = index_stats["indexes"]["jobs"]
                mem_str = f"{job_idx['estimated_memory_mb']:.1f} MB"
                # Check if index is in sync with embeddings
                in_sync = job_idx["total_vectors"] == stats["job_embeddings"]
                status = (
                    "[green]✓ ready[/green]"
                    if in_sync
                    else f"[yellow]⚠ {stats['job_embeddings'] - job_idx['total_vectors']:+,} out of sync[/yellow]"
                )
                table.add_row("jobs.index", f"{job_idx['total_vectors']:,}", mem_str, status)

            # Skill index
            if "skills" in index_stats["indexes"]:
                skill_idx = index_stats["indexes"]["skills"]
                in_sync = skill_idx["total_skills"] == stats["skill_embeddings"]
                status = "[green]✓ ready[/green]" if in_sync else "[yellow]⚠ out of sync[/yellow]"
                table.add_row("skills.index", f"{skill_idx['total_skills']:,}", "-", status)

            # Company index
            if "companies" in index_stats["indexes"]:
                company_idx = index_stats["indexes"]["companies"]
                table.add_row(
                    "companies.index",
                    f"{company_idx['total_centroids']:,} centroids",
                    "-",
                    "[green]✓ ready[/green]",
                )

            console.print(table)
            console.print(f"\n  Index directory: [dim]{index_dir}[/dim]")

        except IndexCompatibilityError as e:
            console.print(f"  [red]⚠ Index incompatible: {e}[/red]")
            console.print("  Run [bold]mcf embed-generate --no-skip-existing[/bold] to rebuild")
    else:
        console.print("  [yellow]No FAISS indexes found[/yellow]")
        console.print("  Run [bold]mcf embed-generate[/bold] to build indexes")

    # Check for cluster files
    cluster_dir = Path(index_dir)
    cluster_files = [
        "skill_clusters.pkl",
        "skill_to_cluster.pkl",
        "skill_cluster_centroids.pkl",
    ]

    console.print("\n[bold]Cluster Files:[/bold]")
    for filename in cluster_files:
        filepath = cluster_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            if size >= 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size >= 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            console.print(f"  {filename:<30} [green]✓ exists[/green] ({size_str})")
        else:
            console.print(f"  {filename:<30} [dim]not found[/dim]")

    # Missing embeddings warning
    if coverage < 100:
        missing = stats["total_jobs"] - stats["job_embeddings"]
        console.print(f"\n[yellow]⚠ {missing:,} jobs missing embeddings[/yellow]")
        console.print("Run [bold]mcf embed-sync[/bold] to generate missing embeddings")


@app.command(name="embed-upgrade")
def upgrade_embeddings(
    model: str = typer.Argument(..., help="New model name (e.g., all-mpnet-base-v2)"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Jobs to process in each batch"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    embedding_backend: str = typer.Option(
        CLI_DEFAULT_EMBEDDING_BACKEND,
        "--embedding-backend",
        help="Embedding inference backend: torch or onnx",
    ),
    onnx_model_dir: Optional[str] = typer.Option(
        None,
        "--onnx-model-dir",
        help="Exported ONNX model directory when using --embedding-backend onnx",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """
    Re-generate all embeddings with a new model version.

    This will:
    1. Delete all existing embeddings
    2. Generate new embeddings with the specified model
    3. Rebuild skill clusters

    Use this when upgrading to a better embedding model.

    Examples:
        mcf embed-upgrade all-mpnet-base-v2
        mcf embed-upgrade all-MiniLM-L12-v2 --yes
    """
    setup_logging(verbose)

    db = _open_database(_resolve_cli_db_path(db_path))
    stats = db.get_embedding_stats()

    current_model = stats.get("model_version") or "none"
    total_jobs = stats["total_jobs"]
    current_embeddings = stats["job_embeddings"]

    # Show warning
    console.print("\n[bold yellow]⚠️  This will regenerate ALL embeddings![/bold yellow]\n")
    console.print(f"Current model: [cyan]{current_model}[/cyan]")
    target_model_version = _create_embedding_generator(
        model_name=model,
        backend=embedding_backend,
        onnx_model_dir=onnx_model_dir,
    ).model_name
    console.print(f"New model:     [green]{target_model_version}[/green]")
    console.print(f"Backend:       [green]{embedding_backend}[/green]")
    console.print(f"Jobs to re-embed: [cyan]{total_jobs:,}[/cyan]")

    if current_embeddings > 0:
        console.print(f"Embeddings to delete: [red]{current_embeddings:,}[/red]")

    # Confirm
    if not confirm:
        console.print()
        proceed = typer.confirm("Continue?", default=False)
        if not proceed:
            console.print("[dim]Aborted[/dim]")
            raise typer.Exit(0)

    console.print()

    # Step 1: Delete existing embeddings
    if current_embeddings > 0:
        console.print("Deleting existing embeddings...")
        if current_model and current_model != "none":
            deleted = db.delete_embeddings_for_model(current_model)
            console.print(f"  Deleted {deleted:,} embeddings for model '{current_model}'")

        # Also delete any embeddings with different/no model version
        with db._connection() as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            if remaining > 0:
                conn.execute("DELETE FROM embeddings")
                console.print(f"  Deleted {remaining:,} remaining embeddings")

    # Step 2: Generate new embeddings with new model
    console.print(f"\nGenerating embeddings with [green]{target_model_version}[/green]...")
    console.print()

    generator = _create_embedding_generator(
        model_name=model,
        backend=embedding_backend,
        onnx_model_dir=onnx_model_dir,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed:,}/{task.total:,}"),
        TextColumn("[dim]({task.fields[speed]:.1f} jobs/sec)"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Jobs", total=total_jobs, speed=0.0)

        def on_progress(emb_stats: EmbeddingStats):
            progress.update(
                task_id,
                completed=emb_stats.jobs_processed,
                speed=emb_stats.jobs_per_second,
            )

        final_stats = generator.generate_all(
            db,
            batch_size=batch_size,
            skip_existing=False,  # Regenerate all
            progress_callback=on_progress,
        )

    # Summary
    console.print("\n[bold green]✓ Embedding upgrade complete![/bold green]\n")

    table = Table(title="Summary", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("New model", generator.model_name)
    table.add_row("Backend", generator.backend_name)
    table.add_row("Jobs processed", f"{final_stats.jobs_processed:,}")
    table.add_row("Skills clustered", f"{final_stats.unique_skills:,}")

    elapsed = final_stats.elapsed_seconds
    if elapsed >= 60:
        minutes, seconds = divmod(int(elapsed), 60)
        elapsed_str = f"{minutes}m {seconds}s"
    else:
        elapsed_str = f"{elapsed:.1f}s"
    table.add_row("Total time", elapsed_str)

    console.print(table)


@app.command(name="embed-export-onnx")
def export_embeddings_to_onnx(
    model: str = typer.Argument(EmbeddingGenerator.MODEL_NAME, help="Sentence-transformers model to export"),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory for the exported ONNX model bundle",
    ),
    opset: int = typer.Option(17, "--opset", help="ONNX opset version"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite an existing export"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Export a sentence-transformers encoder to an ONNX bundle for CPU inference."""
    setup_logging(verbose)

    export_dir = Path(output_dir) if output_dir else _default_onnx_model_dir(model)
    console.print("\n[bold blue]Exporting ONNX Embedding Model[/bold blue]")
    console.print(f"Model:      [green]{model}[/green]")
    console.print(f"Output dir: [green]{export_dir}[/green]")
    console.print(f"Opset:      [green]{opset}[/green]")
    console.print()

    model_path = export_sentence_transformer_to_onnx(
        model,
        export_dir,
        dimension=EmbeddingGenerator.DIMENSION,
        opset=opset,
        overwrite=overwrite,
    )

    console.print(f"[green]✓ Exported ONNX model to {model_path}[/green]")


@app.command(name="embed-compare-backends")
def compare_embedding_backends(
    model: str = typer.Option(EmbeddingGenerator.MODEL_NAME, "--model", help="Base embedding model name"),
    onnx_model_dir: str = typer.Option(..., "--onnx-model-dir", help="Exported ONNX model directory"),
    sample_size: int = typer.Option(24, "--sample-size", help="Representative texts to compare"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    index_dir: str = typer.Option("data/embeddings", "--index-dir", help="Torch FAISS index directory"),
    onnx_index_dir: Optional[str] = typer.Option(
        None,
        "--onnx-index-dir",
        help="Optional FAISS index directory rebuilt with ONNX embeddings",
    ),
    top_k: int = typer.Option(10, "--top-k", help="Top-k depth for search overlap"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Compare Torch and ONNX embedding outputs on representative text samples.

    When --onnx-index-dir is provided, also compares top-k search overlap between
    the existing Torch index directory and an ONNX-rebuilt index directory.
    """
    setup_logging(verbose)
    resolved_db_path = _resolve_cli_db_path(db_path)

    db = _open_database(resolved_db_path, read_only=True)
    torch_generator = _create_embedding_generator(model_name=model, backend="torch")
    onnx_generator = _create_embedding_generator(
        model_name=model,
        backend="onnx",
        onnx_model_dir=onnx_model_dir,
    )

    benchmark_queries = [
        "python developer",
        "machine learning engineer",
        "data scientist singapore",
        "full stack javascript react",
        "devops kubernetes aws",
    ]

    texts: list[str] = list(benchmark_queries)
    texts.extend(db.get_all_unique_skills()[: max(1, sample_size // 3)])
    for row in db.search_jobs(limit=max(1, sample_size)):
        title = row.get("title", "")
        description = (row.get("description") or "")[:240]
        skills = row.get("skills", "")
        texts.append(" ".join(part for part in (title, title, description, skills) if part))
        if len(texts) >= sample_size:
            break

    deduped_texts: list[str] = []
    seen: set[str] = set()
    for text in texts:
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_texts.append(normalized)
        if len(deduped_texts) >= sample_size:
            break

    torch_embeddings = torch_generator.backend.encode_batch(deduped_texts, batch_size=16, normalize_embeddings=True)
    onnx_embeddings = onnx_generator.backend.encode_batch(deduped_texts, batch_size=16, normalize_embeddings=True)
    cosine_scores = np.sum(torch_embeddings * onnx_embeddings, axis=1)

    table = Table(title="Embedding Backend Parity")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Samples", f"{len(deduped_texts):,}")
    table.add_row("Mean cosine", f"{float(np.mean(cosine_scores)):.6f}")
    table.add_row("Min cosine", f"{float(np.min(cosine_scores)):.6f}")
    table.add_row("Max cosine", f"{float(np.max(cosine_scores)):.6f}")
    table.add_row("Torch model", torch_generator.model_name)
    table.add_row("ONNX model", onnx_generator.model_name)
    console.print()
    console.print(table)

    if onnx_index_dir:
        torch_engine = SemanticSearchEngine(
            db_path=resolved_db_path,
            index_dir=Path(index_dir),
            model_version=model,
            embedding_backend="torch",
        )
        onnx_engine = SemanticSearchEngine(
            db_path=resolved_db_path,
            index_dir=Path(onnx_index_dir),
            model_version=model,
            embedding_backend="onnx",
            onnx_model_dir=onnx_model_dir,
        )
        torch_engine.load()
        onnx_engine.load()

        overlap_scores: list[float] = []
        for query in benchmark_queries:
            torch_results = torch_engine.search(SearchRequest(query=query, limit=top_k)).results
            onnx_results = onnx_engine.search(SearchRequest(query=query, limit=top_k)).results
            torch_uuids = {result.uuid for result in torch_results}
            onnx_uuids = {result.uuid for result in onnx_results}
            overlap = len(torch_uuids & onnx_uuids) / max(1, top_k)
            overlap_scores.append(overlap)

        overlap_table = Table(title="Search Top-K Overlap")
        overlap_table.add_column("Metric", style="cyan")
        overlap_table.add_column("Value", justify="right")
        overlap_table.add_row("Queries", f"{len(benchmark_queries):,}")
        overlap_table.add_row("Mean overlap", f"{float(np.mean(overlap_scores)):.3f}")
        overlap_table.add_row("Min overlap", f"{float(np.min(overlap_scores)):.3f}")
        overlap_table.add_row("Top-k", str(top_k))
        console.print()
        console.print(overlap_table)


# Semantic search command


@app.command(name="search-semantic")
def semantic_search_cli(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    salary_min: Optional[int] = typer.Option(None, "--salary-min", help="Minimum salary"),
    salary_max: Optional[int] = typer.Option(None, "--salary-max", help="Maximum salary"),
    company: Optional[str] = typer.Option(None, "--company", "-c", help="Filter by company"),
    employment_type: Optional[str] = typer.Option(None, "--employment-type", "-e", help="Filter by employment type"),
    region: Optional[str] = typer.Option(None, "--region", "-r", help="Filter by region"),
    alpha: float = typer.Option(0.7, "--alpha", help="Semantic vs keyword weight (0=keyword, 1=semantic)"),
    no_expand: bool = typer.Option(False, "--no-expand", help="Disable query expansion"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    index_dir: str = typer.Option("data/embeddings", "--index-dir", help="FAISS index directory"),
    embedding_backend: str = typer.Option(
        CLI_DEFAULT_EMBEDDING_BACKEND,
        "--embedding-backend",
        help="Embedding inference backend: torch or onnx",
    ),
    onnx_model_dir: Optional[str] = typer.Option(
        None,
        "--onnx-model-dir",
        help="Exported ONNX model directory when using --embedding-backend onnx",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Semantic search for jobs.

    Combines keyword matching with semantic similarity for better results
    than plain text search. Requires embeddings to be generated first
    (run 'embed-generate' if not done yet).

    Examples:
        mcf search-semantic "machine learning engineer"
        mcf search-semantic "python developer" --salary-min 8000
        mcf search-semantic "data scientist" --company Google
        mcf search-semantic "ML" --no-expand
        mcf search-semantic "AI engineer" --json
    """
    setup_logging(verbose)
    resolved_db_path = _resolve_cli_db_path(db_path)
    onnx_model_dir = _resolve_cli_onnx_model_dir(
        backend=embedding_backend,
        onnx_model_dir=onnx_model_dir,
    )
    _validate_cli_backend_config_or_exit(
        backend=embedding_backend,
        onnx_model_dir=onnx_model_dir,
    )

    engine = SemanticSearchEngine(
        resolved_db_path,
        Path(index_dir),
        embedding_backend=embedding_backend,
        onnx_model_dir=onnx_model_dir,
    )

    console.print("[dim]Loading search indexes...[/dim]")
    try:
        loaded = engine.load()
    except Exception as e:
        console.print(f"[red]Failed to load indexes: {e}[/red]")
        console.print("[yellow]Run 'mcf embed-generate' first to build indexes.[/yellow]")
        raise typer.Exit(1)

    if not loaded:
        console.print("[yellow]Running in degraded mode (keyword search only)[/yellow]")

    request = SearchRequest(
        query=query,
        limit=limit,
        salary_min=salary_min,
        salary_max=salary_max,
        company=company,
        employment_type=employment_type,
        region=region,
        alpha=alpha,
        expand_query=not no_expand,
    )

    response = engine.search(request)

    if json_output:
        _print_search_json(response, query)
    else:
        _display_search_results(response, query)


def _print_search_json(response: SearchResponse, query: str) -> None:
    """Print search results as JSON."""
    import json
    from dataclasses import asdict

    data = asdict(response)
    data["query"] = query

    # Convert date objects to strings for JSON serialization
    for result in data.get("results", []):
        if result.get("posted_date"):
            result["posted_date"] = str(result["posted_date"])

    console.print(json.dumps(data, indent=2))


def _display_search_results(response: SearchResponse, query: str) -> None:
    """Display search results as a Rich table."""
    console.print("\n[bold blue]Semantic Search Results[/bold blue]")
    console.print("━" * 70)

    # Query info
    console.print(f"\nQuery: [green]{query}[/green]")
    if response.query_expansion:
        console.print(f"Expanded: [dim]{', '.join(response.query_expansion)}[/dim]")
    console.print(f"Candidates: {response.total_candidates:,} jobs (after filters)")
    console.print(f"Search time: [cyan]{response.search_time_ms:.0f}ms[/cyan]")

    if response.degraded:
        console.print("[yellow]⚠ Running in degraded mode (keyword search only)[/yellow]")

    if response.cache_hit:
        console.print("[dim]Cache hit[/dim]")

    if not response.results:
        console.print("\n[yellow]No results found. Try broadening your search.[/yellow]")
        return

    # Results table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Score", justify="right", style="green", width=7)
    table.add_column("Title", max_width=40)
    table.add_column("Company", max_width=20)
    table.add_column("Salary", justify="right", width=15)

    for job in response.results:
        salary = ""
        if job.salary_min and job.salary_max:
            salary = f"${job.salary_min:,}-${job.salary_max:,}"
        elif job.salary_min:
            salary = f"${job.salary_min:,}+"

        table.add_row(
            f"{job.similarity_score:.3f}",
            job.title[:40],
            (job.company_name[:20] if job.company_name else "N/A"),
            salary,
        )

    console.print()
    console.print(table)
    console.print("\n[dim]Tip: Use --json for programmatic access[/dim]")


# API server command


@app.command(name="api-serve")
def serve_api(
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    index_dir: str = typer.Option("data/embeddings", "--index-dir", help="Path to FAISS indexes"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes (production)"),
    cors_origins: str = typer.Option(
        "http://localhost:3000,http://localhost:5173",
        "--cors",
        help="Comma-separated CORS origins",
    ),
    api_rate_limit: int = typer.Option(
        100,
        "--rate-limit",
        help="Max requests per minute per IP (0 to disable)",
    ),
    embedding_backend: str = typer.Option(
        CLI_DEFAULT_EMBEDDING_BACKEND,
        "--embedding-backend",
        help="Embedding inference backend: torch or onnx",
    ),
    search_backend: str = typer.Option(
        "faiss",
        "--search-backend",
        help="Vector search backend: faiss or pgvector",
    ),
    lean_hosted: bool = typer.Option(
        False,
        "--lean-hosted",
        help="Disable skill/company vector dependencies for a lean hosted slice",
    ),
    onnx_model_dir: Optional[str] = typer.Option(
        None,
        "--onnx-model-dir",
        help="Exported ONNX model directory when using --embedding-backend onnx",
    ),
) -> None:
    """
    Start the semantic search API server.

    Examples:
        mcf api-serve                     # Start on localhost:8000
        mcf api-serve --port 9000         # Custom port
        mcf api-serve --reload            # Auto-reload for dev
        mcf api-serve --workers 4         # Production mode
    """
    import uvicorn

    onnx_model_dir = _resolve_cli_onnx_model_dir(
        backend=embedding_backend,
        onnx_model_dir=onnx_model_dir,
    )
    _validate_cli_backend_config_or_exit(
        backend=embedding_backend,
        onnx_model_dir=onnx_model_dir,
    )
    resolved_db_path = _resolve_cli_db_path(db_path)

    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

    # Check prerequisites
    db_target = resolve_database_target(resolved_db_path)
    if db_target.is_sqlite and not db_target.sqlite_path.exists():
        console.print(f"[red]Error:[/red] Database not found: {resolved_db_path}")
        console.print("Run 'mcf scrape' first to populate the database.")
        raise typer.Exit(1)

    index_path = Path(index_dir)
    search_backend = search_backend.lower()
    if search_backend == "faiss" and not (index_path / "jobs.index").exists():
        console.print(f"[yellow]Warning:[/yellow] FAISS index not found in {index_dir}")
        console.print("API will run in degraded mode (keyword search only).")
        console.print("Run 'mcf embed-generate' to enable semantic search.")

    # Display startup info
    console.print("\n[bold green]Starting MCF Semantic Search API[/bold green]")
    console.print(f"  Database:   {resolved_db_path}")
    console.print(f"  Index dir:  {index_dir}")
    console.print(f"  Endpoint:   http://{host}:{port}")
    console.print(f"  API docs:   http://{host}:{port}/docs")
    console.print(f"  CORS:       {', '.join(origins)}")
    console.print(f"  Embedding:  {embedding_backend}")
    console.print(f"  Search:     {search_backend}")
    if lean_hosted:
        console.print("  Profile:    [yellow]Lean hosted slice[/yellow]")
    if onnx_model_dir:
        console.print(f"  ONNX dir:   {onnx_model_dir}")
    if api_rate_limit > 0:
        console.print(f"  Rate limit: {api_rate_limit} req/min per IP")
    else:
        console.print("  Rate limit: [yellow]disabled[/yellow]")
    if reload:
        console.print("  Mode:       [yellow]Development (auto-reload)[/yellow]")
    else:
        console.print(f"  Mode:       Production ({workers} worker{'s' if workers != 1 else ''})")
    console.print()

    # Pass config via environment so uvicorn workers can pick it up
    os.environ["MCF_DB_PATH"] = resolved_db_path
    if db_target.is_postgres:
        os.environ["DATABASE_URL"] = resolved_db_path
    else:
        os.environ.pop("DATABASE_URL", None)
    os.environ["MCF_INDEX_DIR"] = index_dir
    os.environ["MCF_CORS_ORIGINS"] = cors_origins
    os.environ["MCF_RATE_LIMIT_RPM"] = str(api_rate_limit)
    os.environ["MCF_EMBEDDING_BACKEND"] = embedding_backend
    os.environ["MCF_SEARCH_BACKEND"] = search_backend
    os.environ["MCF_LEAN_HOSTED"] = "1" if lean_hosted else "0"
    if onnx_model_dir:
        os.environ["MCF_ONNX_MODEL_DIR"] = str(onnx_model_dir)
    else:
        os.environ.pop("MCF_ONNX_MODEL_DIR", None)

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
        log_level="info",
    )


@app.command(name="db-backup")
def backup_database(
    source: str = typer.Option("data/mcf_jobs.db", "--source", help="Live SQLite source database"),
    backup_dir: str = typer.Option("data/backups", "--backup-dir", help="Backup output directory"),
    prefix: str = typer.Option("mcf_pre_postgres", "--prefix", help="Backup filename prefix"),
) -> None:
    """Create and verify a hot SQLite backup."""
    backup_path = create_sqlite_hot_backup(source, backup_dir, prefix=prefix)
    metadata = verify_sqlite_backup(backup_path)
    console.print(f"[green]Backup created:[/green] {backup_path}")
    console.print(f"Integrity: [cyan]{metadata.integrity_check}[/cyan]")
    console.print(f"Jobs: [cyan]{metadata.jobs_count:,}[/cyan]")
    console.print(f"Size: [cyan]{metadata.size_bytes:,} bytes[/cyan]")


@app.command(name="pg-migrate")
def migrate_postgres(
    source: str = typer.Option(..., "--source", help="SQLite backup path"),
    target: str = typer.Option(..., "--target", help="PostgreSQL DSN"),
    report_path: str = typer.Option(
        "data/backups/postgres_migration_report.json",
        "--report",
        help="Where to write the migration report",
    ),
    batch_size: int = typer.Option(5000, "--batch-size", help="Rows per batch"),
    audit_only: bool = typer.Option(False, "--audit-only", help="Only run the preflight audit"),
    truncate_first: bool = typer.Option(True, "--truncate-first/--no-truncate-first", help="Reset target tables first"),
) -> None:
    """Load a SQLite backup into PostgreSQL with anomaly reporting."""
    anomalies = audit_sqlite_source(source)
    console.print(f"Preflight anomalies: [cyan]{len(anomalies)}[/cyan]")
    if audit_only:
        for anomaly in anomalies[:20]:
            console.print(f"- {anomaly.table}.{anomaly.column} row={anomaly.row_id} issue={anomaly.issue}")
        return

    report = migrate_sqlite_backup_to_postgres(
        sqlite_path=source,
        postgres_dsn=target,
        batch_size=batch_size,
        truncate_first=truncate_first,
    )
    write_migration_report(report, report_path)
    console.print(f"[green]Migration complete[/green] report={report_path}")
    for table, count in sorted(report.copied_rows.items()):
        console.print(f"  {table}: {count:,}")
    console.print(f"  anomalies: {len(report.anomalies):,}")


@app.command(name="pg-seed-hosted")
def seed_hosted(
    source: str = typer.Option(..., "--source", help="Source PostgreSQL DSN"),
    target: str = typer.Option(..., "--target", help="Target PostgreSQL DSN"),
    min_posted_date: str = typer.Option(
        DEFAULT_HOSTED_SLICE_POLICY.min_posted_date.isoformat(),
        "--min-posted-date",
        help="Minimum posted date to retain in the hosted slice",
    ),
    max_age_days: int = typer.Option(
        DEFAULT_HOSTED_SLICE_POLICY.max_age_days,
        "--max-age-days",
        help="Maximum age in days for hosted rows",
    ),
) -> None:
    """Seed a lean hosted slice from local Postgres."""
    policy = HostedSlicePolicy(
        min_posted_date=date.fromisoformat(min_posted_date),
        max_age_days=max_age_days,
    )
    result = seed_hosted_slice_from_postgres(source_dsn=source, target_dsn=target, policy=policy)
    console.print_json(data=result)


@app.command(name="pg-purge-hosted")
def purge_hosted(
    target: str = typer.Option(..., "--target", help="Target PostgreSQL DSN"),
    min_posted_date: str = typer.Option(
        DEFAULT_HOSTED_SLICE_POLICY.min_posted_date.isoformat(),
        "--min-posted-date",
        help="Minimum posted date to retain in the hosted slice",
    ),
    max_age_days: int = typer.Option(
        DEFAULT_HOSTED_SLICE_POLICY.max_age_days,
        "--max-age-days",
        help="Maximum age in days for hosted rows",
    ),
) -> None:
    """Purge a hosted slice down to the lean Neon policy."""
    policy = HostedSlicePolicy(
        min_posted_date=date.fromisoformat(min_posted_date),
        max_age_days=max_age_days,
    )
    result = purge_hosted_slice(target_dsn=target, policy=policy)
    console.print_json(data=result)


@app.command(name="benchmark")
def run_benchmark(
    queries: int = typer.Option(100, "--queries", "-n", help="Number of benchmark queries"),
    warmup: int = typer.Option(10, "--warmup", help="Number of warmup queries"),
    embed_texts: int = typer.Option(100, "--embed-texts", help="Number of texts for embedding benchmark"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path or PostgreSQL DSN"),
    index_dir: str = typer.Option("data/embeddings", "--index-dir", help="FAISS index directory"),
    embedding_backend: str = typer.Option(
        CLI_DEFAULT_EMBEDDING_BACKEND,
        "--embedding-backend",
        help="Embedding inference backend: torch or onnx",
    ),
    onnx_model_dir: Optional[str] = typer.Option(
        None,
        "--onnx-model-dir",
        help="Exported ONNX model directory when using --embedding-backend onnx",
    ),
) -> None:
    """
    Run performance benchmarks on the semantic search system.

    Measures search latency, embedding generation time, and index loading.
    Checks results against performance targets:

    - Search latency p95 < 100ms
    - Cached search latency p95 < 20ms
    - Query embedding time < 50ms
    - Index load time < 5s

    Example:
        mcf benchmark --queries 50 --warmup 5
    """
    import subprocess

    onnx_model_dir = _resolve_cli_onnx_model_dir(
        backend=embedding_backend,
        onnx_model_dir=onnx_model_dir,
    )
    _validate_cli_backend_config_or_exit(
        backend=embedding_backend,
        onnx_model_dir=onnx_model_dir,
    )
    resolved_db_path = _resolve_cli_db_path(db_path)

    cmd = [
        sys.executable,
        "scripts/benchmark.py",
        "--queries",
        str(queries),
        "--warmup",
        str(warmup),
        "--embed-texts",
        str(embed_texts),
        "--db",
        resolved_db_path,
        "--index-dir",
        index_dir,
        "--embedding-backend",
        embedding_backend,
    ]
    if onnx_model_dir:
        cmd.extend(["--onnx-model-dir", str(onnx_model_dir)])

    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


if __name__ == "__main__":
    app()
