"""Generate static UI aggregate snapshots from the live API.

Run in CI (after the Neon slice is refreshed) or locally against the dev DB.
Uses the real FastAPI app via TestClient so snapshot shapes are byte-identical
to the live endpoints the frontend already consumes — zero schema drift.

Env (falls back to the app's own resolution, i.e. local data/mcf_jobs.db):
    DATABASE_URL / MCF_DATABASE_URL / MCF_DB_PATH
    MCF_SEARCH_BACKEND, MCF_LEAN_HOSTED, MCF_EMBEDDING_BACKEND, MCF_ONNX_MODEL_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app

OUT_DIR = Path("src/frontend/public/snapshots")

# filename -> (path, query params, required top-level keys)
SNAPSHOTS: dict[str, tuple[str, dict, list[str]]] = {
    "overview.json": ("/api/overview", {"months": 3}, ["headline_metrics", "rising_skills"]),
    "stats.json": ("/api/stats", {}, ["total_jobs"]),
    "skills_cloud.json": ("/api/skills/cloud", {"min_jobs": 10, "limit": 80}, ["items", "total_unique_skills"]),
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TestClient(app) as client:
        for filename, (path, params, required) in SNAPSHOTS.items():
            resp = client.get(path, params=params)
            resp.raise_for_status()
            data = resp.json()
            missing = [k for k in required if k not in data]
            if missing:
                print(f"ERROR: {filename} missing required keys {missing}", file=sys.stderr)
                return 1
            (OUT_DIR / filename).write_text(json.dumps(data, separators=(",", ":")))
            print(f"wrote {filename} ({(OUT_DIR / filename).stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
