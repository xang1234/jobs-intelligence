"""
Embedding backend abstractions for semantic search.

The project currently uses sentence-transformers on top of PyTorch. This
module preserves that path while adding an ONNX Runtime implementation that
produces the same normalized float32 vectors for inference-only workloads.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_BACKEND = "torch"
ONNX_MODEL_SUFFIX = "+onnx"
SUPPORTED_EMBEDDING_BACKENDS = ("torch", "onnx")


def normalize_backend_name(name: str | None) -> str:
    """Normalize and validate an embedding backend name."""
    normalized = (name or DEFAULT_EMBEDDING_BACKEND).strip().lower()
    if normalized not in SUPPORTED_EMBEDDING_BACKENDS:
        supported = ", ".join(SUPPORTED_EMBEDDING_BACKENDS)
        raise ValueError(f"Unsupported embedding backend '{name}'. Expected one of: {supported}")
    return normalized


def resolve_base_model_name(model_name: str, backend: str) -> str:
    """Strip backend-specific suffixes from a model version string."""
    normalized_backend = normalize_backend_name(backend)
    if normalized_backend == "onnx" and model_name.endswith(ONNX_MODEL_SUFFIX):
        return model_name[: -len(ONNX_MODEL_SUFFIX)]
    return model_name


def resolve_model_version(model_name: str, backend: str) -> str:
    """Return the persisted model version for a backend."""
    normalized_backend = normalize_backend_name(backend)
    base_model = resolve_base_model_name(model_name, normalized_backend)
    if normalized_backend == "onnx":
        return f"{base_model}{ONNX_MODEL_SUFFIX}"
    return base_model


def default_onnx_model_dir(model_name: str) -> Path:
    """Return the standard local export directory for an ONNX model bundle."""
    safe_name = resolve_base_model_name(model_name, "onnx").replace("/", "--")
    return Path("data/models") / f"{safe_name}-onnx"


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize one or more vectors without changing shape."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        if norm == 0:
            return array
        return array / norm

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


class EmbeddingBackend(ABC):
    """Contract for embedding inference backends."""

    def __init__(
        self,
        model_name: str,
        *,
        backend_name: str,
        dimension: int,
    ) -> None:
        self.backend_name = normalize_backend_name(backend_name)
        self.base_model_name = resolve_base_model_name(model_name, self.backend_name)
        self.model_version = resolve_model_version(model_name, self.backend_name)
        self.dimension = dimension

    @abstractmethod
    def encode_one(
        self,
        text: str,
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode a single text string into a 1D vector."""

    @abstractmethod
    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode multiple text strings into a 2D matrix."""

    @property
    @abstractmethod
    def raw_model(self):
        """Return the underlying model/session object."""

    @property
    def device(self) -> str:
        return self.backend_name

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """SentenceTransformer-compatible convenience wrapper."""
        if isinstance(sentences, str):
            return self.encode_one(
                sentences,
                normalize_embeddings=normalize_embeddings,
            )

        return self.encode_batch(
            list(sentences),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )


class TorchSentenceTransformerBackend(EmbeddingBackend):
    """PyTorch-backed sentence-transformers inference."""

    def __init__(
        self,
        model_name: str,
        *,
        dimension: int,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name, backend_name="torch", dimension=dimension)
        self._device = device
        self._model = None

    @property
    def raw_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.base_model_name)
            self._model = SentenceTransformer(self.base_model_name, device=self._device)
            logger.info("Model loaded on device: %s", self._model.device)
        return self._model

    @property
    def device(self) -> str:
        return str(self.raw_model.device)

    def encode_one(
        self,
        text: str,
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        embedding = self.raw_model.encode(
            text,
            normalize_embeddings=normalize_embeddings,
        )
        return np.asarray(embedding, dtype=np.float32)

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        embeddings = self.raw_model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(embeddings, dtype=np.float32)


class OnnxEmbeddingBackend(EmbeddingBackend):
    """ONNX Runtime backend for mean-pooled transformer embeddings."""

    DEFAULT_MODEL_FILENAME = "model.onnx"

    def __init__(
        self,
        model_name: str,
        *,
        dimension: int,
        model_dir: str | Path | None = None,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        super().__init__(model_name, backend_name="onnx", dimension=dimension)
        self._provider = provider
        self._model_dir = self._resolve_model_dir(model_dir)
        self._tokenizer = None
        self._session = None
        self._session_input_names: tuple[str, ...] | None = None

    @staticmethod
    def _resolve_model_dir(model_dir: str | Path | None) -> Path:
        if model_dir is None:
            raise ValueError("ONNX backend requires --onnx-model-dir or MCF_ONNX_MODEL_DIR")

        path = Path(model_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ONNX model directory not found: {path}")
        return path

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def raw_model(self):
        if self._session is None:
            import onnxruntime as ort

            model_path = self._find_model_path()
            logger.info("Loading ONNX embedding model: %s", model_path)
            self._session = ort.InferenceSession(
                str(model_path),
                providers=[self._provider],
            )
            self._session_input_names = tuple(inp.name for inp in self._session.get_inputs())
            logger.info("ONNX model loaded with provider: %s", self._provider)
        return self._session

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        return self._tokenizer

    @property
    def device(self) -> str:
        return self._provider

    def _find_model_path(self) -> Path:
        candidates = [
            self.model_dir / self.DEFAULT_MODEL_FILENAME,
            self.model_dir / "onnx" / self.DEFAULT_MODEL_FILENAME,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        matches = sorted(self.model_dir.glob("*.onnx"))
        if matches:
            return matches[0]

        nested_matches = sorted((self.model_dir / "onnx").glob("*.onnx"))
        if nested_matches:
            return nested_matches[0]

        raise FileNotFoundError(f"No ONNX model file found in {self.model_dir}")

    def _mean_pool(self, last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask.astype(np.float32)[..., None]
        masked_embeddings = last_hidden_state.astype(np.float32) * mask
        token_counts = np.clip(mask.sum(axis=1), a_min=1.0, a_max=None)
        return masked_embeddings.sum(axis=1) / token_counts

    def _encode_chunk(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        session = self.raw_model
        encoded_inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="np",
        )

        feeds: dict[str, np.ndarray] = {}
        input_ids = np.asarray(encoded_inputs["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(encoded_inputs["attention_mask"], dtype=np.int64)

        for input_name in self._session_input_names or ():
            if input_name in encoded_inputs:
                feeds[input_name] = np.asarray(encoded_inputs[input_name], dtype=np.int64)
            elif input_name == "token_type_ids":
                feeds[input_name] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = session.run(None, feeds)
        last_hidden_state = np.asarray(outputs[0], dtype=np.float32)
        embeddings = self._mean_pool(last_hidden_state, attention_mask)
        if normalize_embeddings:
            embeddings = normalize_vectors(embeddings)
        return np.asarray(embeddings, dtype=np.float32)

    def encode_one(
        self,
        text: str,
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        embedding = self.encode_batch(
            [text],
            batch_size=1,
            normalize_embeddings=normalize_embeddings,
        )
        return np.asarray(embedding[0], dtype=np.float32)

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        iterator: Iterable[int]
        if show_progress_bar:
            from tqdm.auto import tqdm

            iterator = tqdm(
                range(0, len(texts), batch_size),
                desc="Encoding",
                leave=False,
            )
        else:
            iterator = range(0, len(texts), batch_size)

        chunks = []
        for start in iterator:
            chunk = texts[start : start + batch_size]
            chunks.append(
                self._encode_chunk(
                    chunk,
                    normalize_embeddings=normalize_embeddings,
                )
            )

        return np.vstack(chunks).astype(np.float32, copy=False)


def create_embedding_backend(
    *,
    backend: str,
    model_name: str,
    dimension: int,
    device: str | None = None,
    onnx_model_dir: str | Path | None = None,
) -> EmbeddingBackend:
    """Factory for concrete embedding backend implementations."""
    normalized_backend = normalize_backend_name(backend)
    if normalized_backend == "onnx":
        return OnnxEmbeddingBackend(
            model_name,
            dimension=dimension,
            model_dir=onnx_model_dir,
        )

    return TorchSentenceTransformerBackend(
        model_name,
        dimension=dimension,
        device=device,
    )


def validate_embedding_backend_config(
    *,
    backend: str,
    model_name: str,
    dimension: int,
    onnx_model_dir: str | Path | None = None,
) -> None:
    """Eagerly validate backend configuration without keeping the backend alive."""
    embedding_backend = create_embedding_backend(
        backend=backend,
        model_name=model_name,
        dimension=dimension,
        onnx_model_dir=onnx_model_dir,
    )
    if embedding_backend.backend_name == "onnx":
        # Force path/session validation so startup fails before serving requests.
        embedding_backend.raw_model


def export_sentence_transformer_to_onnx(
    model_name: str,
    output_dir: str | Path,
    *,
    dimension: int,
    opset: int = 17,
    overwrite: bool = False,
) -> Path:
    """
    Export the transformer encoder from a sentence-transformers model to ONNX.

    The exported model emits token-level hidden states. Mean pooling and vector
    normalization remain in the runtime backend so parity logic stays explicit.
    """
    output_path = Path(output_dir).expanduser().resolve()
    model_path = output_path / OnnxEmbeddingBackend.DEFAULT_MODEL_FILENAME

    if model_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing ONNX model at {model_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer
    from torch import nn

    torch_model = SentenceTransformer(resolve_base_model_name(model_name, "torch"), device="cpu")
    transformer_module = torch_model[0]
    auto_model = transformer_module.auto_model
    tokenizer = transformer_module.tokenizer

    class _FeatureExtractionWrapper(nn.Module):
        def __init__(self, model, input_names: Sequence[str]) -> None:
            super().__init__()
            self.model = model
            self.input_names = tuple(input_names)

        def forward(self, *inputs):
            kwargs = {name: value for name, value in zip(self.input_names, inputs)}
            return self.model(**kwargs).last_hidden_state

    tokenizer.save_pretrained(output_path)
    auto_model.config.save_pretrained(output_path)

    example = tokenizer("ONNX export sample", return_tensors="pt")
    input_names = [name for name in ("input_ids", "attention_mask", "token_type_ids") if name in example]
    wrapper = _FeatureExtractionWrapper(auto_model.eval(), input_names)

    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["last_hidden_state"] = {0: "batch", 1: "sequence"}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(example[name] for name in input_names),
            str(model_path),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset,
        )

    manifest = {
        "backend": "onnx",
        "base_model_name": resolve_base_model_name(model_name, "onnx"),
        "model_version": resolve_model_version(model_name, "onnx"),
        "dimension": dimension,
        "pooling": "mean",
        "normalize_embeddings": True,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "opset": opset,
    }
    with open(output_path / "embedding_backend_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    logger.info("Exported ONNX embedding model to %s", model_path)
    return model_path
