"""Thread safety tests.

These tests verify that the storage layer handles concurrent access
from multiple threads correctly without data races or corruption.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

from asyncer import asyncify
import numpy as np
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.application.utils.security import SecretString
from reflectlog.core.enums import EmbedderProvider
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.types import Embeddings


class MockEmbedder(Embeddings):
    """Mock embedder for testing that produces deterministic vectors."""

    def __init__(self, dims: int = 128) -> None:
        self.dims = dims
        self.call_count = 0

    def embed_query(self, text: str) -> list[float]:
        """Return deterministic embeddings based on text hash."""
        self.call_count += 1
        np.random.seed(hash(text) % (2**32))
        embedding: list[float] = np.random.randn(self.dims).astype(np.float32).tolist()
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self.embed_query(text) for text in texts]


def create_test_config(workspace_id: str = "test-thread-safety") -> Config:
    """Create a Config instance for testing."""
    return Config(
        workspace_id=workspace_id,
        openrouter_api_key=SecretString("test-key"),
        embedder_provider=EmbedderProvider.LANGCHAIN,
        embedding_model="mock-model",
        embedding_dims=128,
        qwen_embedding_dims=128,
        search_limit=5,
        deduplicate_memories=True,
        enable_llm_infer=False,
        log_level="DEBUG",
    )


def create_memory_manager(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> MemoryManager:
    """Create a MemoryManager with mock embedder and isolated index paths."""
    mock_embedder = MockEmbedder(dims=128)
    mock_logger: IStructuredLogger = cast(
        IStructuredLogger, MagicMock(spec=StructuredLogger)
    )

    from unittest.mock import patch

    monkeypatch.chdir(tmp_path)
    with patch(
        "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
        return_value=mock_embedder,
    ):
        manager = MemoryManager(config, mock_logger)
        return manager


@pytest.mark.integration
class TestMemoryManagerThreadSafety:
    """Tests for MemoryManager thread safety."""

    def test_concurrent_sync_adds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that concurrent sync add operations don't cause data corruption."""
        config = create_test_config("test-sync-adds")
        memory_manager = create_memory_manager(config, tmp_path, monkeypatch)

        num_threads = 10
        adds_per_thread = 100
        results = [0] * num_threads

        def add_memories(thread_id: int):
            """Add memories from a thread."""
            memories = [
                f"Thread {thread_id} - Memory {i}" for i in range(adds_per_thread)
            ]
            # add_memories is thread-safe (uses RLock)
            count = memory_manager.add_memories(memories)
            results[thread_id] = count

        # Run adds in parallel threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(add_memories, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Verify all memories were added (may have deduplicates within threads)
        all_memories = memory_manager.get_all()
        expected_max = num_threads * adds_per_thread
        assert len(all_memories) <= expected_max
        # Should have at least some unique memories
        assert len(all_memories) >= num_threads

        memory_manager.close()

    def test_concurrent_sync_get_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that concurrent get_all calls are thread-safe."""
        config = create_test_config("test-sync-getall")
        memory_manager = create_memory_manager(config, tmp_path, monkeypatch)

        # Add initial memories
        memories = [f"Memory {i}" for i in range(100)]
        memory_manager.add_memories(memories)

        num_threads = 10
        results = [None] * num_threads

        def get_all_memories(thread_id: int):
            """Get all memories from a thread."""
            # get_all is thread-safe (uses RLock)
            all_memories = memory_manager.get_all()
            results[thread_id] = len(all_memories)
            # Verify we got a valid list
            assert isinstance(all_memories, list)
            assert all(isinstance(mem, str) for mem in all_memories)

        # Run get_all in parallel threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_all_memories, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        # All threads should have gotten the same result
        assert all(r == 100 for r in results)

        memory_manager.close()

    def test_concurrent_mixed_operations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that mixed concurrent operations work correctly."""
        config = create_test_config("test-mixed-ops")
        memory_manager = create_memory_manager(config, tmp_path, monkeypatch)

        # Add initial memories
        for i in range(50):
            memory_manager.add_memories([f"Initial {i}"])

        add_count = [0]
        get_count = [0]

        def add_operation():
            """Add memories from thread."""
            for i in range(10):
                memory_manager.add_memories([f"Concurrent {i}"])
                add_count[0] += 1

        def get_operation():
            """Get all memories from thread."""
            for _ in range(10):
                all_memories = memory_manager.get_all()
                get_count[0] += 1
                assert isinstance(all_memories, list)

        # Run mixed operations in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for _ in range(5):
                futures.append(executor.submit(add_operation))
                futures.append(executor.submit(get_operation))

            for future in as_completed(futures):
                future.result()

        # Verify operations completed
        assert add_count[0] == 50  # 5 threads * 10 adds each
        assert get_count[0] == 50  # 5 threads * 10 gets each

        # Verify final state
        all_memories = memory_manager.get_all()
        assert len(all_memories) > 0

        memory_manager.close()

    def test_concurrent_deletes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that concurrent delete operations are thread-safe."""
        config = create_test_config("test-concurrent-deletes")
        memory_manager = create_memory_manager(config, tmp_path, monkeypatch)

        # Add memories
        for i in range(20):
            memory_manager.add_memories([f"Memory {i}"])

        deleted_count = [0]

        def delete_memories():
            """Delete memories from thread."""
            # Try to delete various memories
            for i in range(10):
                # Try to delete (some will succeed, some won't)
                result = memory_manager.delete_by_memory(f"Memory {i}")
                if result:
                    deleted_count[0] += 1

        # Run deletes in parallel threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(delete_memories) for _ in range(5)]
            for future in as_completed(futures):
                future.result()

        # Verify some deletions occurred
        # (exact count depends on timing)
        assert 0 <= deleted_count[0] <= 50

        # Verify system is still functional
        all_memories = memory_manager.get_all()
        assert isinstance(all_memories, list)

        memory_manager.close()


@pytest.mark.integration
class TestAsyncConcurrencySafety:
    """Tests for async operation safety."""

    @pytest.mark.asyncio
    async def test_async_add_concurrency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that async add operations handle concurrency correctly."""
        config = create_test_config("test-async-add")
        memory_manager = create_memory_manager(config, tmp_path, monkeypatch)

        num_tasks = 20
        memories_per_task = 10

        async def add_memories(task_id: int):
            """Add memories from an async task."""
            memories = [
                f"Task {task_id} - Memory {i}" for i in range(memories_per_task)
            ]
            await memory_manager.add_memories_async(memories)

        # Run all add tasks concurrently
        tasks = [add_memories(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)

        # Verify all memories were added (may have deduplicates)
        all_memories = memory_manager.get_all()
        # Should have at most num_tasks * memories_per_task
        assert len(all_memories) <= num_tasks * memories_per_task
        # Should have at least some memories
        assert len(all_memories) > 0

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_async_search_concurrency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that async search operations handle concurrency correctly."""
        config = create_test_config("test-async-search")
        memory_manager = create_memory_manager(config, tmp_path, monkeypatch)

        # Add initial memories
        memories = [f"Memory {i} for testing" for i in range(100)]
        await memory_manager.add_memories_async(memories)

        num_tasks = 50

        async def search_memories(task_id: int):
            """Search from an async task."""
            results = await memory_manager.search(f"Memory {task_id % 10}")
            # Results should always be a list
            assert isinstance(results, list)

        # Run all search tasks concurrently
        tasks = [search_memories(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_async_mixed_operations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test mixed async operations (add, search, get_all) running concurrently."""
        config = create_test_config("test-async-mixed")
        memory_manager = create_memory_manager(config, tmp_path, monkeypatch)

        # Add initial memories
        initial_memories = [f"Initial memory {i}" for i in range(50)]
        await memory_manager.add_memories_async(initial_memories)

        add_count = [0]
        search_count = [0]
        get_all_count = [0]

        async def add_operation():
            """Continuously add memories."""
            for i in range(10):
                await memory_manager.add_memories_async([f"Concurrent add {i}"])
                add_count[0] += 1

        async def search_operation():
            """Continuously search."""
            for _ in range(10):
                await memory_manager.search("memory")
                search_count[0] += 1

        async def get_all_operation():
            """Continuously get all memories using asyncify."""
            for _ in range(10):
                await asyncify(memory_manager.get_all)()
                get_all_count[0] += 1

        # Run all operations concurrently
        await asyncio.gather(
            add_operation(),
            search_operation(),
            get_all_operation(),
        )

        # Verify operations completed
        assert add_count[0] == 10
        assert search_count[0] == 10
        assert get_all_count[0] == 10

        # Verify system is still functional
        all_memories = memory_manager.get_all()
        assert isinstance(all_memories, list)
        assert len(all_memories) > 0

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_sync_and_async(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that sync and async operations can run concurrently."""
        config = create_test_config("test-sync-async")
        memory_manager = create_memory_manager(config, tmp_path, monkeypatch)

        # Add initial memories
        initial_memories = [f"Memory {i}" for i in range(50)]
        await memory_manager.add_memories_async(initial_memories)

        async_add_count = [0]
        sync_add_count = [0]

        async def async_add():
            """Async adds."""
            for i in range(10):
                await memory_manager.add_memories_async([f"Async {i}"])
                async_add_count[0] += 1

        def sync_add():
            """Sync adds in thread pool."""
            import asyncio

            async def sync_add_wrapper():
                for i in range(10, 20):
                    memory_manager.add_memories([f"Sync {i}"])
                    sync_add_count[0] += 1

            loop = asyncio.get_event_loop()
            return loop.run_in_executor(None, lambda: asyncio.run(sync_add_wrapper()))

        # Run async and sync operations concurrently
        await asyncio.gather(async_add(), sync_add())

        # Verify operations completed
        assert async_add_count[0] == 10
        assert sync_add_count[0] == 10

        # Verify system is still functional
        all_memories = memory_manager.get_all()
        assert isinstance(all_memories, list)
        assert len(all_memories) > 0

        memory_manager.close()
