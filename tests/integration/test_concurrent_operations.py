"""Concurrent operation integration tests.

These tests verify that the MemoryManager handles concurrent operations
correctly, including race conditions, duplicate detection, and data consistency.
"""

import asyncio
from collections import Counter
import os
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

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_USEARCH_CONCURRENCY_TESTS") != "1",
    reason="Set RUN_USEARCH_CONCURRENCY_TESTS=1 to run USearch concurrency tests",
)


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

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def create_test_config(workspace_id: str = "test-concurrent") -> Config:
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


def create_memory_manager(config: Config) -> MemoryManager:
    """Create a MemoryManager with mock embedder."""
    mock_embedder = MockEmbedder(dims=128)
    mock_logger: IStructuredLogger = cast(
        IStructuredLogger, MagicMock(spec=StructuredLogger)
    )

    from unittest.mock import patch

    with patch(
        "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
        return_value=mock_embedder,
    ):
        manager = MemoryManager(config, mock_logger)
        return manager


@pytest.mark.integration
class TestConcurrentAdds:
    """Tests for concurrent add operations."""

    @pytest.mark.asyncio
    async def test_parallel_add_same_memories(self):
        """Test that adding the same memories in parallel doesn't cause duplicates."""
        config = create_test_config("test-parallel-same")
        memory_manager = create_memory_manager(config)

        memories = ["Memory 1", "Memory 2", "Memory 3"]

        # Add the same memories multiple times in parallel
        tasks = [memory_manager.add_memories_async(memories) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should succeed
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Add operation failed: {result}")

        # Should only have 3 unique memories (deduplication should work)
        all_memories = memory_manager.get_all()
        memory_counts = Counter(all_memories)

        # Each memory should appear exactly once
        for mem in memories:
            assert memory_counts[mem] == 1, (
                f"Memory '{mem}' appeared {memory_counts[mem]} times instead of 1"
            )

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_parallel_add_different_memories(self):
        """Test that adding different memories in parallel preserves all memories."""
        config = create_test_config("test-parallel-different")
        memory_manager = create_memory_manager(config)

        # Create different memory sets
        memory_sets = [
            [f"Set 1 - Memory {i}" for i in range(3)],
            [f"Set 2 - Memory {i}" for i in range(3)],
            [f"Set 3 - Memory {i}" for i in range(3)],
        ]

        # Add all sets in parallel
        tasks = [
            memory_manager.add_memories_async(memories) for memories in memory_sets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should succeed
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Add operation failed: {result}")

        # Should have all 9 memories
        all_memories = memory_manager.get_all()
        assert len(all_memories) == 9

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_add_and_search(self):
        """Test that search works correctly while adds are happening."""
        config = create_test_config("test-concurrent-add-search")
        memory_manager = create_memory_manager(config)

        # Add initial memories
        initial_memories = ["Python", "JavaScript", "Go"]
        await memory_manager.add_memories_async(initial_memories)

        async def add_memories():
            """Continuously add memories."""
            for i in range(10):
                await memory_manager.add_memories_async([f"Language {i}"])
                await asyncio.sleep(0.01)

        async def search_memories():
            """Continuously search for memories."""
            for _ in range(10):
                results = await memory_manager.search("Python")
                # Search should always return results or empty list, never error
                assert isinstance(results, list)
                await asyncio.sleep(0.01)

        # Run adds and searches in parallel
        await asyncio.gather(add_memories(), search_memories())

        # Final search should work
        results = await memory_manager.search("Python")
        assert len(results) > 0

        memory_manager.close()


@pytest.mark.integration
class TestConcurrentDeletes:
    """Tests for concurrent delete operations."""

    @pytest.mark.asyncio
    async def test_parallel_delete_same_memory(self):
        """Test that deleting the same memory in parallel doesn't cause errors."""
        config = create_test_config("test-parallel-delete")
        memory_manager = create_memory_manager(config)

        # Add a memory
        memory = "Test memory to delete"
        await memory_manager.add_memories_async([memory])

        # Try to delete the same memory multiple times in parallel
        # Note: delete_by_memory is synchronous, so we use asyncify
        tasks = []
        for _ in range(5):
            tasks.append(asyncify(memory_manager.delete_by_memory)(memory))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should succeed (first deletes, others are no-ops)
        for result in results:
            if isinstance(result, Exception) and "not found" not in str(result).lower():
                pytest.fail(f"Delete operation failed unexpectedly: {result}")

        # Memory should be gone
        all_memories = memory_manager.get_all()
        assert memory not in all_memories

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_add_and_delete(self):
        """Test that adds and deletes can happen concurrently without issues."""
        config = create_test_config("test-concurrent-add-delete")
        memory_manager = create_memory_manager(config)

        # Add initial memories
        initial_memories = [f"Memory {i}" for i in range(10)]
        await memory_manager.add_memories_async(initial_memories)

        deleted_count = [0]  # Use list for mutability
        added_count = [0]

        async def add_new_memories():
            """Add new memories."""
            for i in range(10, 20):
                await memory_manager.add_memories_async([f"Memory {i}"])
                added_count[0] += 1
                await asyncio.sleep(0.01)

        async def delete_existing_memories():
            """Delete existing memories using search_for_removal."""
            for i in range(5):
                # Search for the memory
                candidates = memory_manager.search_for_removal(f"Memory {i}", limit=1)
                if candidates:
                    # Use asyncify since delete is sync
                    await asyncify(memory_manager.delete_by_memory)(
                        candidates[0]["memory"]
                    )
                    deleted_count[0] += 1
                await asyncio.sleep(0.01)

        # Run adds and deletes in parallel
        await asyncio.gather(add_new_memories(), delete_existing_memories())

        # Should have some memories left (exact count depends on timing)
        all_memories = memory_manager.get_all()
        assert len(all_memories) > 0

        memory_manager.close()


@pytest.mark.integration
class TestConcurrentSearches:
    """Tests for concurrent search operations."""

    @pytest.mark.asyncio
    async def test_parallel_search_same_query(self):
        """Test that parallel searches with the same query work correctly."""
        config = create_test_config("test-parallel-search")
        memory_manager = create_memory_manager(config)

        # Add memories
        memories = [
            "Python programming",
            "JavaScript web development",
            "Go systems programming",
        ]
        await memory_manager.add_memories_async(memories)

        # Search for "programming" multiple times in parallel
        tasks = [memory_manager.search("programming") for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All searches should succeed and return consistent results
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Search operation failed: {result}")
            # All searches should return at least one result (or empty list)
            assert isinstance(result, list)

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_parallel_search_different_queries(self):
        """Test that parallel searches with different queries work correctly."""
        config = create_test_config("test-search-different")
        memory_manager = create_memory_manager(config)

        # Add memories
        memories = [
            "Python is great for data science",
            "JavaScript is used for web development",
            "Go is excellent for systems programming",
            "Rust provides memory safety",
        ]
        await memory_manager.add_memories_async(memories)

        # Search for different terms in parallel
        queries = ["Python", "JavaScript", "Go", "Rust"]
        tasks = [memory_manager.search(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All searches should succeed
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Search for '{queries[i]}' failed: {result}")
            # Each search should find relevant results (or empty list)
            assert isinstance(result, list)

        memory_manager.close()


@pytest.mark.integration
class TestConcurrentGetAll:
    """Tests for concurrent get_all operations."""

    @pytest.mark.asyncio
    async def test_parallel_get_all(self):
        """Test that get_all works correctly when called in parallel."""
        config = create_test_config("test-parallel-getall")
        memory_manager = create_memory_manager(config)

        # Add memories
        memories = [f"Memory {i}" for i in range(10)]
        await memory_manager.add_memories_async(memories)

        # Call get_all multiple times concurrently using asyncify
        tasks = [asyncify(memory_manager.get_all)() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All calls should return the same results
        for result in results:
            assert len(result) == 10
            # Results should be independent (defensive copies)
            result.append("Should not affect original")

        # Verify original wasn't modified
        final = memory_manager.get_all()
        assert len(final) == 10

        memory_manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_get_all_and_add(self):
        """Test that get_all and add can work concurrently."""
        config = create_test_config("test-concurrent-getall-add")
        memory_manager = create_memory_manager(config)

        async def add_memories():
            """Add memories continuously."""
            for i in range(10):
                await memory_manager.add_memories_async([f"Memory {i}"])
                await asyncio.sleep(0.01)

        async def get_all_memories():
            """Get all memories continuously using thread pool."""
            import asyncio

            for _ in range(10):
                loop = asyncio.get_event_loop()
                all_memories = await loop.run_in_executor(None, memory_manager.get_all)
                assert isinstance(all_memories, list)
                await asyncio.sleep(0.01)

        await asyncio.gather(add_memories(), get_all_memories())

        memory_manager.close()


@pytest.mark.integration
class TestAsyncConcurrencySafety:
    """Tests for async operation safety."""

    @pytest.mark.asyncio
    async def test_async_add_concurrency(self):
        """Test that async add operations handle concurrency correctly."""
        config = create_test_config("test-async-add")
        memory_manager = create_memory_manager(config)

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
    async def test_async_search_concurrency(self):
        """Test that async search operations handle concurrency correctly."""
        config = create_test_config("test-async-search")
        memory_manager = create_memory_manager(config)

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
    async def test_async_mixed_operations(self):
        """Test mixed async operations (add, search, delete) running concurrently."""
        config = create_test_config("test-async-mixed")
        memory_manager = create_memory_manager(config)

        # Add initial memories
        initial_memories = [f"Initial memory {i}" for i in range(50)]
        await memory_manager.add_memories_async(initial_memories)

        deleted_count = [0]

        async def add_operation():
            """Continuously add memories."""
            for i in range(10):
                await memory_manager.add_memories_async([f"Concurrent add {i}"])
                await asyncio.sleep(0.001)

        async def search_operation():
            """Continuously search."""
            for _ in range(10):
                await memory_manager.search("memory")
                await asyncio.sleep(0.001)

        async def delete_operation():
            """Continuously remove memories."""
            for i in range(5):
                # Search for the memory
                candidates = memory_manager.search_for_removal(
                    f"Concurrent add {i}", limit=1
                )
                if candidates:
                    # Use asyncify since delete is sync
                    await asyncify(memory_manager.delete_by_memory)(
                        candidates[0]["memory"]
                    )
                    deleted_count[0] += 1
                await asyncio.sleep(0.001)

        async def get_all_operation():
            """Continuously get all memories using asyncify."""
            for _ in range(10):
                await asyncify(memory_manager.get_all)()
                await asyncio.sleep(0.001)

        # Run all operations concurrently
        await asyncio.gather(
            add_operation(),
            search_operation(),
            delete_operation(),
            get_all_operation(),
        )

        # Verify system is still functional
        all_memories = memory_manager.get_all()
        assert isinstance(all_memories, list)
        assert len(all_memories) > 0

        memory_manager.close()
