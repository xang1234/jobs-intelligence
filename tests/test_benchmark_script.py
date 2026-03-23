from pathlib import Path

from scripts import benchmark


class DummyManager:
    def __init__(self, index_dir: Path, model_version: str = "all-MiniLM-L6-v2") -> None:
        self.index_dir = Path(index_dir)
        self.model_version = model_version

    def exists(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def get_stats(self) -> dict:
        return {
            "indexes": {
                "jobs": {
                    "estimated_memory_mb": 12.5,
                    "total_vectors": 321,
                    "index_type": "IndexFlatIP",
                }
            }
        }


def test_benchmark_index_load_uses_supplied_model_version(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_manager(index_dir: Path, model_version: str = "all-MiniLM-L6-v2") -> DummyManager:
        captured["index_dir"] = Path(index_dir)
        captured["model_version"] = model_version
        return DummyManager(index_dir=index_dir, model_version=model_version)

    monkeypatch.setattr(benchmark, "FAISSIndexManager", fake_manager)

    result = benchmark.benchmark_index_load(tmp_path / "indexes", model_version="all-MiniLM-L6-v2+onnx")

    assert captured["index_dir"] == tmp_path / "indexes"
    assert captured["model_version"] == "all-MiniLM-L6-v2+onnx"
    assert result["index_type"] == "IndexFlatIP"
    assert result["n_vectors"] == 321


def test_main_fails_cleanly_for_missing_onnx_bundle(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "validate_embedding_backend_config",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing ONNX bundle")),
    )
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark.py"])

    result = benchmark.main()

    assert result == 1
