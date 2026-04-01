"""Tests for embedding backend abstractions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.api.app import create_app
from src.mcf.database import MCFDatabase
from src.mcf.embeddings.backends import (
    EmbeddingBackend,
    normalize_backend_name,
    normalize_vectors,
    resolve_model_version,
)
from src.mcf.embeddings.generator import EmbeddingGenerator
from src.mcf.embeddings.pgvector_backend import PGVectorBackend
from src.mcf.embeddings.search_engine import SemanticSearchEngine
from src.mcf.pg_database import PostgresDatabase, build_pg_schema_sql
from tests.factories import generate_test_job


class DummyEmbeddingBackend(EmbeddingBackend):
    """Deterministic backend for contract tests."""

    def __init__(self, *, backend_name: str = "torch") -> None:
        super().__init__("all-MiniLM-L6-v2", backend_name=backend_name, dimension=384)
        self.calls = 0

    @property
    def raw_model(self):
        return self

    def _embed(self, text: str) -> np.ndarray:
        base = np.zeros(self.dimension, dtype=np.float32)
        base[0] = len(text)
        base[1] = sum(ord(char) for char in text) % 97
        return normalize_vectors(base)

    def encode_one(
        self,
        text: str,
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        self.calls += 1
        embedding = self._embed(text)
        if normalize_embeddings:
            return embedding.astype(np.float32)
        return (embedding * 3.0).astype(np.float32)

    def encode_batch(
        self,
        texts,
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del batch_size, show_progress_bar
        return np.vstack([self.encode_one(text, normalize_embeddings=normalize_embeddings) for text in texts]).astype(
            np.float32
        )


def test_model_version_suffix_for_onnx():
    assert resolve_model_version("all-MiniLM-L6-v2", "torch") == "all-MiniLM-L6-v2"
    assert resolve_model_version("all-MiniLM-L6-v2", "onnx") == "all-MiniLM-L6-v2+onnx"


def test_backend_name_validation():
    assert normalize_backend_name("Torch") == "torch"
    assert normalize_backend_name("onnx") == "onnx"


def test_generator_uses_backend_contract():
    generator = EmbeddingGenerator()
    backend = DummyEmbeddingBackend()
    generator._backend = backend

    job = generate_test_job(title="Data Scientist", skills=["Python", "SQL"])
    embedding = generator.generate_job_embedding(job)
    batch = generator.generate_skill_embeddings_batch(["Python", "SQL"])

    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0)
    assert set(batch.keys()) == {"Python", "SQL"}
    assert all(value.shape == (384,) for value in batch.values())
    assert generator.model.encode("backend alias").shape == (384,)


def test_search_engine_query_embedding_uses_backend_cache(temp_dir: Path):
    db = MCFDatabase(str(temp_dir / "query-cache.db"))
    db.upsert_job(generate_test_job())

    engine = SemanticSearchEngine(
        db_path=str(db.db_path),
        index_dir=temp_dir / "embeddings",
    )
    backend = DummyEmbeddingBackend()
    engine.generator._backend = backend

    first = engine._get_query_embedding("python developer")
    second = engine._get_query_embedding("python developer")

    assert first.shape == (384,)
    assert np.array_equal(first, second)
    assert backend.calls == 1


def test_onnx_generator_uses_distinct_model_version():
    generator = EmbeddingGenerator(
        backend="onnx",
        onnx_model_dir="data/embeddings",
    )
    assert generator.model_name == "all-MiniLM-L6-v2+onnx"
    assert generator.backend_name == "onnx"


def test_create_app_reads_embedding_backend_env(monkeypatch):
    monkeypatch.setenv("MCF_EMBEDDING_BACKEND", "onnx")
    monkeypatch.setenv("MCF_ONNX_MODEL_DIR", "/tmp/mcf-onnx")
    monkeypatch.setattr("src.api.app.validate_embedding_backend_config", lambda **kwargs: None)

    app = create_app()

    assert app.state.embedding_backend == "onnx"
    assert app.state.onnx_model_dir == "/tmp/mcf-onnx"


def test_create_app_prefers_database_url_env(monkeypatch):
    monkeypatch.delenv("MCF_DB_PATH", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/mcf")
    monkeypatch.setattr("src.api.app.validate_embedding_backend_config", lambda **kwargs: None)

    app = create_app()

    assert app.state.db_path == "postgresql://postgres:postgres@localhost:5432/mcf"


def test_pgvector_backend_requires_job_embeddings():
    class _FakeEmbeddingDB:
        def __init__(self, job_embeddings: int, *, vector_search_supported: bool = True):
            self.job_embeddings = job_embeddings
            self.vector_search_supported = vector_search_supported

        def get_embedding_stats(self):
            return {"job_embeddings": self.job_embeddings}

        def supports_vector_search(self):
            return self.vector_search_supported

    empty_backend = PGVectorBackend(_FakeEmbeddingDB(0), model_version="all-MiniLM-L6-v2")
    ready_backend = PGVectorBackend(_FakeEmbeddingDB(5), model_version="all-MiniLM-L6-v2")
    unsupported_backend = PGVectorBackend(
        _FakeEmbeddingDB(5, vector_search_supported=False),
        model_version="all-MiniLM-L6-v2",
    )

    assert empty_backend.exists() is False
    assert empty_backend.load() is False
    assert ready_backend.exists() is True
    assert ready_backend.load() is True
    assert unsupported_backend.exists() is False
    assert unsupported_backend.load() is False


def test_create_app_fails_fast_for_missing_onnx_model_dir():
    with pytest.raises(FileNotFoundError):
        create_app(embedding_backend="onnx", onnx_model_dir="/tmp/does-not-exist")


def test_postgres_database_decodes_bytea_embeddings():
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    decoded = PostgresDatabase._vector_from_value(vector.tobytes())

    assert np.allclose(decoded, vector)
    assert decoded.dtype == np.float32


def test_build_pg_schema_sql_falls_back_to_bytea_without_pgvector():
    schema = build_pg_schema_sql(pgvector_enabled=False)

    assert "embedding BYTEA NOT NULL" in schema
    assert "CREATE EXTENSION IF NOT EXISTS vector" not in schema
    assert "vector_cosine_ops" not in schema
