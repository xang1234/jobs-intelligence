"""CLI tests for ONNX default backend wiring."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from typer.testing import CliRunner

import src.cli as cli

runner = CliRunner()


def test_create_embedding_generator_defaults_to_onnx_dir(monkeypatch):
    captured: dict[str, object] = {}

    def fake_validate(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "validate_embedding_backend_config", fake_validate)

    generator = cli._create_embedding_generator()

    assert generator.backend_name == "onnx"
    assert generator.onnx_model_dir == Path("data/models/all-MiniLM-L6-v2-onnx")
    assert captured["backend"] == "onnx"
    assert captured["onnx_model_dir"] == Path("data/models/all-MiniLM-L6-v2-onnx")


def test_create_embedding_generator_prefers_env_onnx_dir(monkeypatch):
    captured: dict[str, object] = {}

    def fake_validate(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "validate_embedding_backend_config", fake_validate)
    monkeypatch.setenv("MCF_ONNX_MODEL_DIR", "/opt/mcf/models/all-MiniLM-L6-v2-onnx")

    generator = cli._create_embedding_generator()

    assert generator.backend_name == "onnx"
    assert generator.onnx_model_dir == Path("/opt/mcf/models/all-MiniLM-L6-v2-onnx")
    assert captured["onnx_model_dir"] == "/opt/mcf/models/all-MiniLM-L6-v2-onnx"


def test_api_serve_defaults_to_onnx(monkeypatch, temp_dir: Path):
    db_path = temp_dir / "smoke.db"
    db_path.touch()
    index_dir = temp_dir / "embeddings"
    index_dir.mkdir()
    (index_dir / "jobs.index").touch()
    calls: dict[str, object] = {}
    previous_backend = os.environ.get("MCF_EMBEDDING_BACKEND")
    previous_model_dir = os.environ.get("MCF_ONNX_MODEL_DIR")

    import uvicorn

    def fake_run(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "validate_embedding_backend_config", lambda **kwargs: None)

    try:
        result = runner.invoke(
            cli.app,
            ["api-serve", "--db", str(db_path), "--index-dir", str(index_dir)],
        )

        assert result.exit_code == 0
        assert calls["args"] == ("src.api.app:app",)
        assert os.environ["MCF_EMBEDDING_BACKEND"] == "onnx"
        assert os.environ["MCF_ONNX_MODEL_DIR"] == "data/models/all-MiniLM-L6-v2-onnx"
    finally:
        if previous_backend is None:
            os.environ.pop("MCF_EMBEDDING_BACKEND", None)
        else:
            os.environ["MCF_EMBEDDING_BACKEND"] = previous_backend
        if previous_model_dir is None:
            os.environ.pop("MCF_ONNX_MODEL_DIR", None)
        else:
            os.environ["MCF_ONNX_MODEL_DIR"] = previous_model_dir


def test_api_serve_respects_explicit_pgvector_backend(monkeypatch):
    calls: dict[str, object] = {}
    previous_search_backend = os.environ.get("MCF_SEARCH_BACKEND")
    previous_lean_hosted = os.environ.get("MCF_LEAN_HOSTED")
    previous_database_url = os.environ.get("DATABASE_URL")

    import uvicorn

    def fake_run(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "validate_embedding_backend_config", lambda **kwargs: None)
    monkeypatch.setenv("MCF_SEARCH_BACKEND", "faiss")

    try:
        result = runner.invoke(
            cli.app,
            [
                "api-serve",
                "--db",
                "postgresql://postgres:postgres@localhost:5432/mcf",
                "--search-backend",
                "pgvector",
                "--lean-hosted",
            ],
        )

        assert result.exit_code == 0
        assert calls["args"] == ("src.api.app:app",)
        assert os.environ["MCF_SEARCH_BACKEND"] == "pgvector"
        assert os.environ["MCF_LEAN_HOSTED"] == "1"
        assert os.environ["DATABASE_URL"] == "postgresql://postgres:postgres@localhost:5432/mcf"
    finally:
        if previous_search_backend is None:
            os.environ.pop("MCF_SEARCH_BACKEND", None)
        else:
            os.environ["MCF_SEARCH_BACKEND"] = previous_search_backend
        if previous_lean_hosted is None:
            os.environ.pop("MCF_LEAN_HOSTED", None)
        else:
            os.environ["MCF_LEAN_HOSTED"] = previous_lean_hosted
        if previous_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_database_url


def test_benchmark_defaults_to_onnx(monkeypatch):
    calls: dict[str, object] = {}

    def fake_run(cmd, *args, **kwargs):
        calls["cmd"] = cmd
        calls["args"] = args
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "validate_embedding_backend_config", lambda **kwargs: None)

    result = runner.invoke(cli.app, ["benchmark", "--queries", "1", "--warmup", "0"])

    assert result.exit_code == 0
    assert calls["cmd"][-4:] == [
        "--embedding-backend",
        "onnx",
        "--onnx-model-dir",
        "data/models/all-MiniLM-L6-v2-onnx",
    ]


def test_search_semantic_fails_cleanly_when_onnx_bundle_is_missing(monkeypatch):
    def fake_validate(**kwargs):
        raise FileNotFoundError("missing ONNX bundle")

    monkeypatch.setattr(cli, "validate_embedding_backend_config", fake_validate)

    result = runner.invoke(cli.app, ["search-semantic", "python engineer"])

    assert result.exit_code == 1
    assert "Invalid embedding backend configuration" in result.stdout
    assert "embed-export-onnx" in result.stdout


def test_benchmark_fails_cleanly_when_onnx_bundle_is_missing(monkeypatch):
    def fake_validate(**kwargs):
        raise FileNotFoundError("missing ONNX bundle")

    monkeypatch.setattr(cli, "validate_embedding_backend_config", fake_validate)

    result = runner.invoke(cli.app, ["benchmark", "--queries", "1", "--warmup", "0"])

    assert result.exit_code == 1
    assert "Invalid embedding backend configuration" in result.stdout
    assert "embed-export-onnx" in result.stdout
