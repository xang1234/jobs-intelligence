"""Deploy the Hugging Face Docker Space for the hosted API.

Reads secrets from /tmp/mcf-deploy.env by default. The env file should contain:

    export HF_TOKEN='hf_...'
    export NEON_DATABASE_URL='postgresql://...'

The script uploads deploy/huggingface-space/ to the Space repository, configures
runtime variables, stores the Neon DSN as a Space secret, and restarts the Space.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "xang1234/jobs-intelligence-api"
DEFAULT_SOURCE_REF = "master"
DEFAULT_SOURCE_REPO = "https://github.com/xang1234/jobs-intelligence.git"
DEFAULT_CORS_ORIGINS = (
    "https://jobs-intelligence.pages.dev,https://jobs.deepgradient.uk,https://deepgradient.uk,http://localhost:3000"
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :]
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def resolve_source_version(root: Path, source_ref: str) -> str:
    """Resolve a local git ref for the Space build checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--verify", source_ref],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return source_ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the Hugging Face Space backend")
    parser.add_argument("--env-file", default="/tmp/mcf-deploy.env", help="Path to deploy env file")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Space repo ID")
    parser.add_argument("--source-ref", default=DEFAULT_SOURCE_REF, help="Git ref cloned during Space build")
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO, help="Git source repo cloned during Space build")
    parser.add_argument(
        "--source-version",
        default=None,
        help="Source revision checked out during the Docker build. Defaults to the local git revision.",
    )
    parser.add_argument("--cors-origins", default=DEFAULT_CORS_ORIGINS, help="Comma-separated CORS origins")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))

    token = require_env("HF_TOKEN")
    neon_database_url = require_env("NEON_DATABASE_URL")

    root = Path(__file__).resolve().parents[1]
    space_dir = root / "deploy" / "huggingface-space"
    source_version = args.source_version or resolve_source_version(root, args.source_ref)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    api.add_space_secret(args.repo_id, "DATABASE_URL", neon_database_url, token=token)

    variables = {
        "MCF_SEARCH_BACKEND": "pgvector",
        "MCF_LEAN_HOSTED": "1",
        "MCF_EMBEDDING_BACKEND": "onnx",
        "MCF_CORS_ORIGINS": args.cors_origins,
        "MCF_RATE_LIMIT_RPM": "100",
        "SOURCE_REPO": args.source_repo,
        "SOURCE_REF": args.source_ref,
        "SOURCE_VERSION": source_version,
    }
    for key, value in variables.items():
        api.add_space_variable(args.repo_id, key, value, token=token)

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=str(space_dir),
        commit_message="Deploy Jobs Intelligence API Docker Space",
        token=token,
    )

    runtime = api.restart_space(args.repo_id, token=token)
    print(f"Space deployment triggered: https://huggingface.co/spaces/{args.repo_id}")
    print(f"Runtime stage: {getattr(runtime, 'stage', 'unknown')}")


if __name__ == "__main__":
    main()
