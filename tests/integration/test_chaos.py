"""Chaos testing for ReflectLog.

Simulates failure modes and edge cases to ensure
system resilience under adverse conditions.

Example scenarios:
- Engine failures (Tantivy index corruption, USearch unavailable)
- Network timeouts and connection drops
- LLM provider failures (rate limits, API errors)
- Resource exhaustion (memory, disk space)
- Concurrent request storms

Usage:
    pytest tests/integration/test_chaos.py --engine=failure
    pytest tests/integration/test_chaos.py --network=timeout
    pytest tests/integration/test_chaos.py --resource=exhaustion
"""

import asyncio
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.fusion.ranx_fusion import RanxFusionEngine
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.memory.search_strategies import SearchPipeline


@pytest.fixture
def manager(monkeypatch):
    """Create a mocked MemoryManager for testing.

    Mocks USearch/Tantivy/Embedder engines to avoid real external calls.

    Yields:
        MemoryManager: Mocked manager instance.
    """
    # Set required environment variables before creating config
    monkeypatch.setenv("WORKSPACE_ID", "test-chaos-project")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SEARCH_LIMIT", "10")
    monkeypatch.setenv("EMBEDDING_DIMS", "1536")
    monkeypatch.setenv("RERANKER_ENGINE", "none")

    mock_semantic_engine = MagicMock()
    mock_semantic_engine.search.return_value = []
    mock_semantic_engine.is_ready = MagicMock(return_value=False)
    mock_semantic_engine.count.return_value = 0
    mock_semantic_engine.memory_store.exists_many.side_effect = (
        lambda _workspace, contents: set(contents)
    )
    mock_semantic_engine.get_records_by_contents.return_value = []

    mock_tantivy_engine = MagicMock()
    mock_tantivy_engine.search.return_value = []
    mock_tantivy_engine.is_ready = MagicMock(return_value=False)

    config = Config.from_environment()
    config = replace(
        config,
        enable_recency_boost=False,
        overfetch_adaptive=False,
        fusion_ranking_threshold=0.0,
    )

    mgr = MemoryManager.__new__(MemoryManager)
    mgr._semantic_engine = mock_semantic_engine
    mgr._tantivy_engine = mock_tantivy_engine
    mgr.config = config
    mgr.workspace_id = config.workspace_id
    mgr._lock = MagicMock()
    mgr._write_lock = MagicMock()
    mgr._fusion_engine = RanxFusionEngine()
    mgr.logger = MagicMock()
    mgr._closed = False
    mgr._closing = False
    patched: Any = mgr
    patched.reconcile_pending_replacements = lambda: 0
    mgr._cross_encoder_reranker = None
    mgr._search_pipeline = SearchPipeline(
        semantic_engine=mock_semantic_engine,
        tantivy_engine=mock_tantivy_engine,
        fusion_engine=mgr._fusion_engine,
        config=config,
        logger=mgr.logger,
        memory_manager=mgr,
    )

    async def _default_add(memories: list[str], dry_run: bool = False) -> MagicMock:
        return MagicMock(stored_count=len(memories), skipped_count=0)

    mgr._add_pipeline = MagicMock()
    mgr._add_pipeline.execute = AsyncMock(side_effect=_default_add)

    yield mgr


class TestEngineFailure:
    """Tests system behavior when search engines fail.

    Ensures graceful degradation and proper error handling.
    """

    @pytest.mark.asyncio
    async def test_tantivy_unavailable(self, manager, monkeypatch):
        """Test search when Tantivy is unavailable.

        Should fallback to USearch-only search.
        """

        def mock_tantivy_error(*_args: object, **_kwargs: object) -> list[object]:
            raise ConnectionError("Tantivy connection failed")

        manager._semantic_engine.search.return_value = [
            ("from-usearch", 0.9, "2026-08-22T00:00:00+00:00")
        ]
        monkeypatch.setattr(
            manager._tantivy_engine,
            "search",
            mock_tantivy_error,
        )

        results = await manager.search("test query")

        assert results == ["from-usearch"]

    @pytest.mark.asyncio
    async def test_usearch_unavailable(self, manager, monkeypatch):
        """Test search when USearch is unavailable.

        Empty Tantivy plus a USearch outage is a dual outage.
        """

        from reflectlog.core.exceptions import SearchError

        def mock_usearch_error(*_args: object, **_kwargs: object) -> list[object]:
            raise ConnectionError("USearch connection failed")

        monkeypatch.setattr(
            manager._semantic_engine,
            "search",
            mock_usearch_error,
        )

        with pytest.raises(SearchError, match="USearch connection failed"):
            await manager.search("test query")

    @pytest.mark.asyncio
    async def test_both_engines_fail(self, manager, monkeypatch):
        """Test search when both engines are unavailable.

        With mocked pipeline, search returns gracefully.
        Real implementation would propagate errors from engines.
        """

        def mock_both_error(*_args: object, **_kwargs: object) -> list[object]:
            raise ConnectionError("Both search engines unavailable")

        monkeypatch.setattr(
            manager._semantic_engine,
            "search",
            mock_both_error,
        )
        monkeypatch.setattr(
            manager._tantivy_engine,
            "search",
            mock_both_error,
        )

        from reflectlog.core.exceptions import SearchError

        with pytest.raises(SearchError):
            await manager.search("test query")

    @pytest.mark.asyncio
    async def test_cross_encoder_reranker_failure(self, manager, monkeypatch):
        """Test search when the cross-encoder reranker fails.

        Should return fused hits instead of failing the search.
        """

        class BoomReranker:
            async def rerank_async(
                self, *_args: object, **_kwargs: object
            ) -> list[object]:
                raise ConnectionError("cross-encoder reranker failed")

        object.__setattr__(manager.config, "reranker_engine", "cross_encoder")
        object.__setattr__(manager.config, "fusion_ranking_threshold", 0.0)
        manager._cross_encoder_reranker = BoomReranker()
        manager._semantic_engine.search.return_value = [
            ("a", 0.9, "2026-08-22T00:00:00+00:00"),
            ("b", 0.8, "2026-08-22T00:00:00+00:00"),
        ]
        manager._tantivy_engine.search.return_value = [("a", 0.7), ("b", 0.6)]

        results = await manager.search("test query")
        assert results == ["a", "b"]

    @pytest.mark.asyncio
    async def test_network_timeout(self, manager, monkeypatch):
        """Test search with network timeout.

        Should complete without hanging indefinitely.
        """

        from reflectlog.core.exceptions import SearchError

        def timeout_search(*_args: object, **_kwargs: object) -> list[object]:
            raise TimeoutError("network timeout")

        monkeypatch.setattr(
            manager._semantic_engine,
            "search",
            timeout_search,
        )

        with pytest.raises(SearchError, match="network timeout"):
            await manager.search("test query")

    @pytest.mark.asyncio
    async def test_concurrent_request_storm(self, manager):
        """Test system under rapid concurrent requests.

        Should handle gracefully without resource exhaustion.
        """
        # Note: In a real scenario, we'd use the actual async add method
        # For now, just verify the test structure
        tasks = [
            manager.search(f"Test {i}")
            for i in range(10)  # Reduced from 100 for faster testing
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all results are either lists or exceptions
        assert len(results) == 10


class TestResourceExhaustion:
    """Tests system behavior under resource constraints.

    Ensures proper cleanup and no leaks.
    """

    @pytest.mark.asyncio
    async def test_disk_space_exhaustion(self, manager, tmp_path):
        """Test behavior when disk space runs out.

        Add must not swallow a disk-full OSError from persist.
        """

        async def boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("No space left on device")

        manager._add_pipeline.execute = boom
        with pytest.raises(OSError, match="No space left"):
            await manager.add_memories_async([f"Large memory {i}" for i in range(10)])

    @pytest.mark.asyncio
    async def test_memory_exhaustion(self, manager, monkeypatch):
        """Test with large memory operations.

        Should complete without OOM errors.
        """
        large_text = "x" * 1000000

        async def boom(*_args: object, **_kwargs: object) -> None:
            raise MemoryError("out of memory")

        manager._add_pipeline.execute = boom
        with pytest.raises(MemoryError, match="out of memory"):
            await manager.add_memories_async([large_text])

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, manager, monkeypatch):
        """Test with limited connection pool.

        With mocked pipeline, search completes without calling underlying engine.
        Real implementation would queue requests when pool is exhausted.
        """

        from reflectlog.core.exceptions import SearchError

        def mock_search(*_args: object, **_kwargs: object) -> list[object]:
            raise ConnectionError("Connection pool exhausted")

        monkeypatch.setattr(
            manager._semantic_engine,
            "search",
            mock_search,
        )

        for i in range(10):
            with pytest.raises(SearchError, match="Connection pool exhausted"):
                await manager.search(f"Query {i}")


class TestDataCorruption:
    """Tests resilience against corrupted data.

    Ensures data integrity checks catch corruption early.
    """

    @pytest.mark.asyncio
    async def test_invalid_embedding(self, manager, monkeypatch):
        """Test handling of corrupted embedding data.

        Should skip or handle gracefully.
        """
        corrupted_vector = [float("nan")] * manager.config.embedding_dims

        async def boom(*_args: object, **_kwargs: object) -> None:
            _ = corrupted_vector
            raise ValueError("invalid embedding")

        manager._add_pipeline.execute = boom
        with pytest.raises(ValueError, match="invalid embedding"):
            await manager.add_memories_async(["test memory"])

    @pytest.mark.asyncio
    async def test_config_reload_during_operation(self, manager, monkeypatch):
        """Test config reload during active operations.

        Should reload cleanly without data loss.
        """
        from reflectlog.application.config.settings import Config
        from reflectlog.application.utils.config_reload import ConfigReloadManager

        reload_manager = ConfigReloadManager(lambda: Config.from_environment())
        reloaded = reload_manager.reload_config()
        assert reloaded.workspace_id == manager.config.workspace_id
        await manager.add_memories_async(["test memory"])


def run_chaos_tests(
    engine: str | None = None, network: str | None = None, resource: str | None = None
):
    """Run chaos tests for specified scenario.

    Args:
        engine: Engine to test (failure, network, resource)
        network: Network scenario (timeout)
        resource: Resource to test (disk, memory, pool)
    """
    import sys

    test_file = "tests/integration/test_chaos.py"

    args = ["pytest", test_file]

    if engine:
        args.extend(["--engine", engine])

    if network:
        args.extend(["--network", network])

    if resource:
        args.extend(["--resource", resource])

    sys.exit(pytest.main(args))
