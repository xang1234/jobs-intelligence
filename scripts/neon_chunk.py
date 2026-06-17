#!/usr/bin/env python3
"""Export Neon jobs to a timestamped, gzipped JSONL chunk for the hosted-refresh
archive, then write CHUNK_FILE / CHUNK_ROWS to ./chunk.env for the publish step.

Two modes, chosen by the workflow via ARCHIVE_BASELINE:

  baseline (ARCHIVE_BASELINE=1)  First run for a year (no archive-<year> release
                                 exists yet): dump the whole jobs table, so rows
                                 that predate this scheme are captured before the
                                 90-day purge can delete them.
  delta    (default)             Every later run: only jobs changed this run
                                 (last_updated_at >= run start). Cheap, because
                                 upsert_job only stamps last_updated_at on insert
                                 or genuine change.

Concatenate every chunk (baseline + deltas) to restore the full year; upserts
are idempotent, so overlap and ordering are harmless.

Env in: NEON_DATABASE_URL, SCRAPE_YEAR, RUN_START_EPOCH, ARCHIVE_BASELINE.
"""
import datetime as dt
import gzip
import json
import os

import psycopg
from psycopg.rows import dict_row

dsn = os.environ["NEON_DATABASE_URL"]
year = int(os.environ["SCRAPE_YEAR"])
run_start = dt.datetime.fromtimestamp(int(os.environ["RUN_START_EPOCH"]), tz=dt.timezone.utc)
baseline = os.environ.get("ARCHIVE_BASELINE") == "1"

with psycopg.connect(dsn) as conn:
    row = conn.execute(
        "SELECT current_seq FROM historical_scrape_progress "
        "WHERE year = %s ORDER BY updated_at DESC LIMIT 1",
        (year,),
    ).fetchone()
    watermark = int(row[0]) if row and row[0] is not None else 0

    kind = "baseline" if baseline else "delta"
    # Timestamp to the second keeps same-day reruns from clobbering each other;
    # the concurrency group serializes runs, so starts are always >= 1s apart.
    chunk = f"mcf-{year}-{run_start:%Y%m%dT%H%M%SZ}-{kind}-upto-seq-{watermark}.jsonl.gz"

    if baseline:
        query, params = "SELECT * FROM jobs ORDER BY id", ()
    else:
        # last_updated_at is TIMESTAMPTZ and unindexed; compare it directly.
        query, params = "SELECT * FROM jobs WHERE last_updated_at >= %s ORDER BY id", (run_start,)

    rows = 0
    # Server-side cursor streams the rows so a full-table baseline doesn't have
    # to fit in runner memory.
    with conn.cursor(name="chunk_export", row_factory=dict_row) as cur, gzip.open(
        chunk, "wt", encoding="utf-8"
    ) as out:
        cur.execute(query, params)
        for record in cur:
            out.write(json.dumps(record, default=str) + "\n")
            rows += 1

with open("chunk.env", "w", encoding="utf-8") as fh:
    fh.write(f"CHUNK_FILE={chunk}\nCHUNK_ROWS={rows}\n")

print(f"chunk: {chunk} ({rows} rows, {kind}, watermark seq {watermark})")
