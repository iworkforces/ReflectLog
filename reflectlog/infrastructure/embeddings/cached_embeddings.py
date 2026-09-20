"""Cached embeddings wrapper for query embedding LRU caching."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import threading
from typing import cast

from cachetools import LRUCache
from pydantic import BaseModel, ConfigDict, PrivateAttr

from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.types import Closable, Embeddings


@dataclass
class _SyncInFlight:
    event: threading.Event = field(default_factory=threading.Event)
    value: list[float] | None = None
    error: BaseException | None = None


class CachedEmbeddings(BaseModel):
    """LRU caching wrapper for any Embeddings provider.

    This wrapper caches `embed_query()` results using a SHA-256 hash of the query
    text as the cache key. This is useful for search operations where the same
    query may be executed multiple times (e.g., during result refinement).

    `embed_documents()` consults the same per-text LRU by default so add-path
    Phase 2 query embeds can be reused during Phase 3 persist. Providers with
    distinct query and document encoders can opt into role-separated keys.

    Thread-safety: cachetools.LRUCache is not thread-safe; access is locked.

    Example:
        ```python
        from reflectlog.infrastructure.cached_embeddings import CachedEmbeddings
        from reflectlog.infrastructure.qwen3_embedding import LangchainQwenEmbeddings

        base_embedder = LangchainQwenEmbeddings({...})
        cached_embedder = CachedEmbeddings(
            embedder=base_embedder,
            cache_size=100,
            enabled=True,
        )

        # First call computes embedding
        embedding1 = cached_embedder.embed_query("Python programming")

        # Second call returns cached result
        embedding2 = cached_embedder.embed_query("Python programming")
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Wrapped embeddings provider
    embedder: Embeddings

    # Cache configuration
    cache_size: int = 100  # Maximum number of cached embeddings
    enabled: bool = True  # Enable/disable caching
    role_separated: bool = False

    # Optional logger for cache hit/miss stats
    logger: IStructuredLogger | None = None

    _cache: LRUCache[str, list[float]] = PrivateAttr(
        default_factory=lambda: LRUCache(maxsize=100)
    )
    _cache_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _hits: int = PrivateAttr(default=0)
    _misses: int = PrivateAttr(default=0)
    _coalesced: int = PrivateAttr(default=0)
    _sync_inflight: dict[str, _SyncInFlight] = PrivateAttr(default_factory=dict)
    _async_inflight: dict[str, asyncio.Future[list[float]]] = PrivateAttr(
        default_factory=dict
    )
    _async_gate: asyncio.Lock | None = PrivateAttr(default=None)
    _closed: bool = PrivateAttr(default=False)

    def model_post_init(self, _context: object, /) -> None:
        """Bind LRU capacity to the configured cache_size."""
        self._cache = LRUCache(maxsize=max(1, self.cache_size))

    def _normalize_text(self, text: str) -> str:
        """Match embedder newline collapsing so cache keys align."""
        if self.role_separated:
            return text.replace("\r\n", "\n").replace("\n", " ")
        return text.replace("\n", " ")

    def _hash_query(self, text: str) -> str:
        """Compute SHA-256 hash of query text as cache key.

        SHA-256 is cryptographically secure and avoids MD5 collision risks.

        Args:
            text: Query text to hash.

        Returns:
            SHA-256 hex digest of the text.
        """
        normalized = self._normalize_text(text)
        if not self.role_separated:
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return hashlib.sha256(f"query\0{normalized}".encode()).hexdigest()

    def _hash_document(self, text: str) -> str:
        if not self.role_separated:
            return self._hash_query(text)
        return hashlib.sha256(
            f"document\0{self._normalize_text(text)}".encode()
        ).hexdigest()

    def _get_cached(self, cache_key: str) -> list[float] | None:
        """Get cached embedding if exists (LRU access).

        Args:
            cache_key: SHA-256 hash of the query text.

        Returns:
            Cached embedding if found, None otherwise.
        """
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._hits += 1
            else:
                self._misses += 1
            return cached

    def _set_cached(self, cache_key: str, embedding: list[float]) -> None:
        """Cache an embedding with LRU eviction (automatic).

        Args:
            cache_key: SHA-256 hash of the query text.
            embedding: Embedding vector to cache.
        """
        with self._cache_lock:
            self._cache[cache_key] = embedding

    def embed_query(self, text: str) -> list[float]:
        """Embed query text with LRU caching.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector (from cache or freshly computed).
        """
        if not self.enabled:
            return self.embedder.embed_query(text)

        cache_key = self._hash_query(text)
        leader = False
        flight: _SyncInFlight | None = None
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._hits += 1
                if self.logger:
                    self.logger.debug(
                        "Embedding cache HIT",
                        extra={
                            "cache_key": cache_key[:8],
                            "hits": self._hits,
                            "misses": self._misses,
                        },
                    )
                return cached
            existing = self._sync_inflight.get(cache_key)
            if existing is not None:
                self._coalesced += 1
                flight = existing
            else:
                flight = _SyncInFlight()
                self._sync_inflight[cache_key] = flight
                leader = True
                self._misses += 1

        if not leader:
            assert flight is not None
            _ = flight.event.wait()
            if flight.error is not None:
                raise flight.error
            if flight.value is None:
                raise RuntimeError("Embedding produced an empty vector")
            return flight.value

        assert flight is not None
        try:
            embedding = self.embedder.embed_query(text)
            if not embedding:
                raise RuntimeError("Embedding produced an empty vector")
            self._set_cached(cache_key, embedding)
            flight.value = embedding
            if self.logger:
                self.logger.debug(
                    "Embedding cache MISS",
                    extra={
                        "cache_key": cache_key[:8],
                        "hits": self._hits,
                        "misses": self._misses,
                        "cache_size": self._cache.currsize,
                    },
                )
            return embedding
        except BaseException as exc:
            flight.error = exc
            raise
        finally:
            with self._cache_lock:
                _ = self._sync_inflight.pop(cache_key, None)
            flight.event.set()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents, reusing per-text LRU entries from embed_query.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not self.enabled or not texts:
            return self.embedder.embed_documents(texts)

        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        for idx, text in enumerate(texts):
            cached = self._get_cached(self._hash_document(text))
            if cached is not None:
                results[idx] = cached
            else:
                miss_indices.append(idx)
                miss_texts.append(text)

        if miss_texts:
            computed = self.embedder.embed_documents(miss_texts)
            if len(computed) != len(miss_texts):
                raise RuntimeError(
                    "Embedding batch size mismatch for cached embed_documents"
                )
            for idx, embedding in zip(miss_indices, computed, strict=True):
                if not embedding:
                    raise RuntimeError("Empty embedding returned for cached document")
                self._set_cached(self._hash_document(texts[idx]), embedding)
                results[idx] = embedding

        filled: list[list[float]] = []
        for embedding in results:
            if embedding is None:
                raise RuntimeError("Missing cached embedding slot")
            filled.append(embedding)
        return filled

    async def aembed_query(self, text: str) -> list[float]:
        """Async version of embed_query with LRU caching.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector (from cache or freshly computed).
        """
        if not self.enabled:
            return await self.embedder.aembed_query(text)

        cache_key = self._hash_query(text)
        gate = self._ensure_async_gate()
        leader = False
        shared: asyncio.Future[list[float]] | None = None
        async with gate:
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._hits += 1
                    if self.logger:
                        self.logger.debug(
                            "Embedding cache HIT (async)",
                            extra={
                                "cache_key": cache_key[:8],
                                "hits": self._hits,
                                "misses": self._misses,
                            },
                        )
                    return cached
                existing_future = self._async_inflight.get(cache_key)
                if existing_future is not None:
                    self._coalesced += 1
                    shared = existing_future
                else:
                    shared = cast(
                        "asyncio.Future[list[float]]",
                        asyncio.get_running_loop().create_future(),
                    )
                    self._async_inflight[cache_key] = shared
                    leader = True
                    self._misses += 1

        if not leader:
            assert shared is not None
            return await asyncio.shield(shared)

        assert shared is not None
        try:
            embedding = await self.embedder.aembed_query(text)
            if not embedding:
                raise RuntimeError("Embedding produced an empty vector")
            self._set_cached(cache_key, embedding)
            if not shared.done():
                shared.set_result(embedding)
            if self.logger:
                self.logger.debug(
                    "Embedding cache MISS (async)",
                    extra={
                        "cache_key": cache_key[:8],
                        "hits": self._hits,
                        "misses": self._misses,
                        "cache_size": self._cache.currsize,
                    },
                )
            return embedding
        except BaseException as exc:
            if not shared.done():
                shared.set_exception(exc)
            raise
        finally:
            async with gate:
                _ = self._async_inflight.pop(cache_key, None)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_documents with the same per-text LRU.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not self.enabled or not texts:
            return await self.embedder.aembed_documents(texts)

        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        for idx, text in enumerate(texts):
            cached = self._get_cached(self._hash_document(text))
            if cached is not None:
                results[idx] = cached
            else:
                miss_indices.append(idx)
                miss_texts.append(text)

        if miss_texts:
            computed = await self.embedder.aembed_documents(miss_texts)
            if len(computed) != len(miss_texts):
                raise RuntimeError(
                    "Embedding batch size mismatch for cached aembed_documents"
                )
            for idx, embedding in zip(miss_indices, computed, strict=True):
                if not embedding:
                    raise RuntimeError("Empty embedding returned for cached document")
                self._set_cached(self._hash_document(texts[idx]), embedding)
                results[idx] = embedding

        filled: list[list[float]] = []
        for embedding in results:
            if embedding is None:
                raise RuntimeError("Missing cached embedding slot")
            filled.append(embedding)
        return filled

    def get_cache_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, and current cache size.
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "coalesced": self._coalesced,
            "size": self._cache.currsize,
            "max_size": self.cache_size,
        }

    def clear_cache(self) -> None:
        """Clear cache and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._coalesced = 0

    def close(self) -> None:
        with self._cache_lock:
            if self._closed:
                return
            self._closed = True
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._coalesced = 0
        if isinstance(self.embedder, Closable):
            self.embedder.close()

    def _ensure_async_gate(self) -> asyncio.Lock:
        gate = self._async_gate
        if gate is None:
            gate = asyncio.Lock()
            self._async_gate = gate
        return gate
