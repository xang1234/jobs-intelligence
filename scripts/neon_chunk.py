#!/usr/bin/env python3
"""Export the jobs changed during this Neon refresh run to a date-labeled,
gzipped JSONL chunk, then write CHUNK_FILE / CHUNK_ROWS to ./chunk.env for the
publish step to `source`.

The chunk's contents are the jobs whose `last_updated_at` moved this run, so the
filename is labeled by date (the honest content range) plus the scrape watermark
(`current_seq` — how far the resume got). Concatenate every daily chunk to
restore the full year; upserts are idempotent so order and overlap are harmless.

Env in: NEON_DATABASE_URL, SCRAPE_YEAR, RUN_START_EPOCH.
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

with psycopg.connect(dsn) as conn:
    row = conn.execute(
        "SELECT current_seq FROM historical_scrape_progress "
        "WHERE year = %s ORDER BY updated_at DESC LIMIT 1",
        (year,),
    ).fetchone()
    watermark = int(row[0]) if row and row[0] is not None else 0
    chunk = f"mcf-{year}-{run_start:%Y%m%d}-upto-seq-{watermark}.jsonl.gz"
    rows = 0
    # ponytail: client-side cursor; one day's delta is a few thousand rows.
    # Switch to conn.cursor(name=...) server-side only if a run ever scrapes huge.
    with conn.cursor(row_factory=dict_row) as cur, gzip.open(chunk, "wt", encoding="utf-8") as out:
        cur.execute(
            "SELECT * FROM jobs WHERE last_updated_at::timestamptz >= %s ORDER BY id",
            (run_start,),
        )
        for record in cur:
            out.write(json.dumps(record, default=str) + "\n")
            rows += 1

with open("chunk.env", "w", encoding="utf-8") as fh:
    fh.write(f"CHUNK_FILE={chunk}\nCHUNK_ROWS={rows}\n")

print(f"chunk: {chunk} ({rows} rows, scan watermark seq {watermark})")
