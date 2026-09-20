"""Local text-only embeddings using Tencent WeMM SentenceTransformer models."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import TYPE_CHECKING, final

from asyncer import asyncify
from pydantic import ConfigDict, TypeAdapter, ValidationError
from sentence_transformers import SentenceTransformer

from reflectlog.core.enums import WeMMDevice, WeMMModel
from reflectlog.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_TEXT_LIST_ADAPTER = TypeAdapter(list[str], config=ConfigDict(strict=True))


@dataclass(frozen=True, slots=True)
class WeMMEmbeddingConfig:
    """Validated settings required by the local WeMM adapter."""

    model: WeMMModel
    dimensions: int
    device: WeMMDevice
    batch_size: int

    def __post_init__(self) -> None:
        _ = self.model.resolve_dimensions(self.dimensions)
        if self.batch_size < 1:
            raise ConfigurationError("WeMM batch size must be positive")


@final
class WeMMEmbeddings:
    """Lazy, thread-safe, text-only WeMM embedding provider."""

    def __init__(self, config: WeMMEmbeddingConfig) -> None:
        self.config = config
        self._model: SentenceTransformer | None = None
        self._model_lock = threading.RLock()
        self._closed = False

    @property
    def is_loaded(self) -> bool:
        with self._model_lock:
            return self._model is not None

    def _get_model(self) -> SentenceTransformer:
        with self._model_lock:
            if self._closed:
                raise RuntimeError("WeMM embeddings are closed")
            model = self._model
            if model is None:
                device = (
                    None
                    if self.config.device is WeMMDevice.AUTO
                    else self.config.device.value
                )
                try:
                    model = SentenceTransformer(
                        self.config.model.value,
                        trust_remote_code=True,
                        device=device,
                    )
                except (OSError, RuntimeError) as exc:
                    raise RuntimeError("WeMM model load failed") from exc
                self._model = model
            return model

    @staticmethod
    def _query_text(value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("WeMM query must be text")
        return value

    @staticmethod
    def _document_texts(value: object) -> list[str]:
        try:
            return _TEXT_LIST_ADAPTER.validate_python(value)
        except ValidationError:
            raise TypeError("WeMM documents must be a list of text") from None

    def _validated_vectors(
        self,
        vectors: NDArray[np.float32],
        *,
        expected_count: int,
    ) -> list[list[float]]:
        if vectors.ndim != 2:
            raise RuntimeError("WeMM embedding output must be a matrix")
        count, width = vectors.shape
        if count != expected_count:
            raise RuntimeError("WeMM embedding batch count mismatch")
        if width == 0:
            raise RuntimeError("WeMM embedding produced an empty vector")
        if width != self.config.dimensions:
            raise RuntimeError("WeMM embedding width mismatch")

        validated: list[list[float]] = []
        for row_index in range(count):
            vector = [float(vectors[row_index, index]) for index in range(width)]
            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError("WeMM embedding values must be finite")
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise RuntimeError("WeMM embedding vector must be normalized")
            validated.append(vector)
        return validated

    def embed_query(self, text: str) -> list[float]:
        query = self._query_text(text)
        with self._model_lock:
            model = self._get_model()
            try:
                vectors = model.encode_query(
                    [query],
                    batch_size=self.config.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    truncate_dim=self.config.dimensions,
                )
            except (OSError, RuntimeError) as exc:
                raise RuntimeError("WeMM query inference failed") from exc
            return self._validated_vectors(vectors, expected_count=1)[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        documents = self._document_texts(texts)
        with self._model_lock:
            if self._closed:
                raise RuntimeError("WeMM embeddings are closed")
            if not documents:
                return []
            model = self._get_model()
            try:
                vectors = model.encode_document(
                    documents,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    truncate_dim=self.config.dimensions,
                )
            except (OSError, RuntimeError) as exc:
                raise RuntimeError("WeMM document inference failed") from exc
            return self._validated_vectors(vectors, expected_count=len(documents))

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncify(self.embed_query)(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncify(self.embed_documents)(texts)

    def close(self) -> None:
        with self._model_lock:
            if self._closed:
                return
            self._closed = True
            self._model = None
