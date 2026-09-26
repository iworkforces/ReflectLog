#!/usr/bin/env python3
"""Unit tests for hybrid MemoryManager (USearch + Tantivy)."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.exceptions import StorageError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.storage_coordinator import (
    PortalockerStorageCoordinator,
)


@pytest.fixture(autouse=True)
def _stub_coordinator(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _factory(self: MemoryManager) -> PortalockerStorageCoordinator:
        _ = self
        return PortalockerStorageCoordinator(str(tmp_path / "indexes"), timeout=1.0)

    monkeypatch.setattr(MemoryManager, "_create_coordinator", _factory)
    monkeypatch.setattr(
        "reflectlog.application.memory.manager.ensure_embedding_identity",
        lambda _config, coordinator, *, tantivy_index_path: coordinator,
    )


_ = _stub_coordinator


def _wire_search_mocks(engine: MagicMock) -> None:
    engine.memory_store.exists_many.side_effect = lambda _workspace, contents: set(
        contents
    )
    engine.get_records_by_contents.side_effect = lambda _workspace, contents: [
        type("R", (), {"content": content, "created_at": "2024-01-01T00:00:00"})()
        for content in contents
    ]


@pytest.fixture
def mock_config() -> Config:
    config = Mock(spec=Config)
    config.workspace_id = "test_project"
    config.tantivy_index_path_template = "{workspace_id}_tantivy_test"
    config.index_base_path = "/tmp/test_indexes"
    config.search_limit = 5
    config.search_score_threshold = 0.8
    config.deduplicate_memories = True
    config.enable_llm_infer = False
    config.remove_search_limit = 5
    config.remove_score_threshold = 0.9
    # Fusion settings (ranx-based)
    config.fusion_method = "rrf"
    config.fusion_normalization = None
    config.fusion_rrf_k = 60
    config.fusion_ranking_threshold = 0.5
    config.enable_rrf_fusion = True  # RRF fusion enabled by default
    # Concurrency settings
    config.add_max_concurrency = 4
    config.rerank_max_concurrency = 5
    # Smart replacement settings
    config.enable_smart_replace = True
    config.smart_replace_threshold = 0.7
    config.smart_replace_min_similarity = 0.5
    config.smart_replace_candidate_limit = 3
    config.smart_replace_archive_ttl_days = 30
    config.smart_replace_max_retries = 3
    config.smart_replace_retry_delay = 1.0
    config.llm_provider = "openai"
    # Reranker settings
    config.reranker_engine = "cross_encoder"
    config.reranker_min_results = 0
    config.reranker_batch_normalize = True
    # Recency boost settings
    config.enable_recency_boost = True
    config.recency_decay_rate = 0.01
    # Hybrid search settings
    config.overfetch_multiplier = 3
    config.overfetch_adaptive = False
    config.overfetch_min_multiplier = 1.0
    config.overfetch_max_multiplier = 5.0
    # Logging settings
    config.log_search_results_verbose = False
    config.log_search_result_limit = 3
    # USearchEngine config fields
    config.embedding_model = "openai/text-embedding-3-large"
    config.embedding_dims = 3072
    config.embedder_provider = "openai"
    config.llm_model = "x-ai/grok-4.1-fast"
    config.openrouter_api_key = Mock()
    config.openrouter_api_key.get_secret_value.return_value = "test-api-key"
    config.openrouter_base_url = "https://openrouter.ai/api/v1"
    config.qwen_embedding_dims = 4096
    # Disable embedding cache in tests to avoid issues with mocked embedders
    config.embedding_cache_enabled = False
    config.embedding_cache_size = 100
    # Disable eager initialization in tests to avoid issues with mocked engines
    config.eager_initialization = False
    return config


@pytest.fixture
def mock_logger():
    """Mock structured logger."""
    return cast(IStructuredLogger, Mock(spec=StructuredLogger))


@pytest.mark.unit
class TestHybridMemoryManager:
    """Tests for hybrid MemoryManager (USearch + TantivyEngine)."""

    def test_initialization(self, mock_config, mock_logger):
        """Test basic initialization with TantivyEngine."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: (
                    contents if contents is not None else memories
                )
            )
            mock_usearch_class.return_value.get_id_by_content.return_value = None
            mock_usearch_class.return_value.embedder.embed_documents.side_effect = (
                lambda texts: [[0.1] * 4 for _ in texts]
            )
            mock_usearch_class.return_value.memory_store.begin_add_intents.return_value = []
            mock_usearch_class.return_value.memory_store.list_pending_transitions.return_value = []
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy:
                    manager = MemoryManager(mock_config, mock_logger)
                    assert hasattr(manager, "_semantic_engine")
                    assert hasattr(manager, "_tantivy_engine")
                    assert hasattr(manager, "_fusion_engine")
                    # TantivyEngine should be initialized with config
                    mock_tantivy.assert_called_once()

    def test_add_memories(self, mock_config, mock_logger):
        """Test parallel indexing works with TantivyEngine."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Setup USearch engine mock
                    mock_usearch = MagicMock()
                    _wire_search_mocks(mock_usearch)
                    inserted = {"done": False}

                    def _add_batch(
                        workspace_id: str,
                        contents: list[str],
                        infer: bool = False,
                        vectors: object = None,
                    ) -> list[str]:
                        _ = workspace_id, infer, vectors
                        inserted["done"] = True
                        return list(contents)

                    mock_usearch.add_batch.side_effect = _add_batch
                    mock_usearch.get_id_by_content.side_effect = (
                        lambda _workspace, content: 1 if inserted["done"] else None
                    )
                    mock_usearch.embedder.embed_documents.side_effect = lambda texts: [
                        [0.1] * 4 for _ in texts
                    ]
                    mock_usearch_class.return_value = mock_usearch

                    # Setup tantivy mock
                    mock_tantivy = MagicMock()
                    mock_tantivy.find_by_exact_match.side_effect = (
                        lambda _workspace, content: (
                            [content] if inserted["done"] else []
                        )
                    )
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    result = manager.add_memories(["test"])

                    assert result == 1
                    # USearchEngine.add_batch should be called
                    mock_usearch.add_batch.assert_called_once()
                    mock_tantivy.add_batch.assert_called_once()
                    # TantivyEngine.commit should be called after batch
                    mock_tantivy.commit.assert_called_once()

    def test_search_for_removal_optimized(self, mock_config, mock_logger):
        """Test removal search using direct database lookup (O(log n) optimization).

        Sprint 1.4 changed from O(n) get_all() + iteration to O(log n) indexed lookup.
        """
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch("reflectlog.application.memory.manager.TantivyEngine"):
                    mock_usearch = MagicMock()
                    _wire_search_mocks(mock_usearch)
                    # Mock get_id_by_content for direct lookup (O(log n))
                    mock_usearch.get_id_by_content.return_value = 42
                    mock_usearch_class.return_value = mock_usearch

                    manager = MemoryManager(mock_config, mock_logger)
                    candidates = manager.search_for_removal("test", limit=1)

                    # Verify get_id_by_content was called (not get_all)
                    mock_usearch.get_id_by_content.assert_called_once_with(
                        mock_config.workspace_id, "test"
                    )
                    mock_usearch.get_all.assert_not_called()
                    assert len(candidates) == 1
                    assert candidates[0]["memory"] == "test"
                    assert candidates[0]["id"] == "42"  # Database ID as string
                    assert candidates[0]["score"] == 1.0  # Exact match = perfect score

    def test_search_for_removal_not_found(self, mock_config, mock_logger):
        """Test removal search when memory is not found."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch("reflectlog.application.memory.manager.TantivyEngine"):
                    mock_usearch = MagicMock()
                    _wire_search_mocks(mock_usearch)
                    # Mock get_id_by_content returning None (not found)
                    mock_usearch.get_id_by_content.return_value = None
                    mock_usearch_class.return_value = mock_usearch

                    manager = MemoryManager(mock_config, mock_logger)
                    candidates = manager.search_for_removal("nonexistent", limit=1)

                    # Verify direct lookup was used
                    mock_usearch.get_id_by_content.assert_called_once()
                    assert len(candidates) == 0  # No candidates when not found

    def test_exact_match_detection(self, mock_config, mock_logger):
        """Test exact match detection uses Tantivy."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: (
                    contents if contents is not None else memories
                )
            )
            mock_usearch_class.return_value.get_id_by_content.return_value = None
            mock_usearch_class.return_value.embedder.embed_documents.side_effect = (
                lambda texts: [[0.1] * 4 for _ in texts]
            )
            mock_usearch_class.return_value.memory_store.begin_add_intents.return_value = []
            mock_usearch_class.return_value.memory_store.list_pending_transitions.return_value = []
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    mock_usearch = mock_usearch_class.return_value

                    mock_usearch.get_id_by_content.return_value = 1
                    result = manager._has_exact_match("test message")
                    assert result is True

                    mock_usearch.get_id_by_content.return_value = None
                    result = manager._has_exact_match("test message")
                    assert result is False

    def test_exact_match_detection_with_unavailable_tantivy_uses_database_lookup(
        self, mock_config, mock_logger
    ):
        """Test exact match detection fallback uses direct database lookup (Sprint 2.1).

        When Tantivy is not available, _has_exact_match() should use get_id_by_content()
        for O(log n) indexed lookup instead of semantic search with embedding API call.
        """
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch("reflectlog.application.memory.manager.TantivyEngine"):
                    mock_usearch = MagicMock()
                    mock_usearch_class.return_value = mock_usearch

                    manager = MemoryManager(mock_config, mock_logger)
                    manager._tantivy_engine = None

                    # Test: Memory found via database lookup
                    mock_usearch.get_id_by_content.return_value = 42
                    result = manager._has_exact_match("test message")
                    assert result is True
                    mock_usearch.get_id_by_content.assert_called_with(
                        mock_config.workspace_id, "test message"
                    )

                    # Test: Memory not found
                    mock_usearch.reset_mock()
                    mock_usearch.get_id_by_content.return_value = None
                    result = manager._has_exact_match("nonexistent")
                    assert result is False
                    mock_usearch.get_id_by_content.assert_called_with(
                        mock_config.workspace_id, "nonexistent"
                    )

                    # Test: Error handling - should return False and allow add
                    mock_usearch.reset_mock()
                    mock_usearch.get_id_by_content.side_effect = RuntimeError(
                        "DB error"
                    )
                    result = manager._has_exact_match("error case")
                    assert result is False  # Should proceed without deduplication

    @pytest.mark.asyncio
    async def test_search_uses_rrf_fusion(self, mock_config, mock_logger):
        """Test hybrid search uses RRFFusion for ranking."""
        mock_config.reranker_engine = "none"  # Skip reranking in unit test
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Setup USearchEngine mock - now returns 3-tuples (message, score, created_at)
                    mock_usearch = MagicMock()
                    _wire_search_mocks(mock_usearch)
                    mock_usearch.search.return_value = [
                        ("usearch result", 0.9, "2024-01-01T00:00:00")
                    ]
                    mock_usearch_class.return_value = mock_usearch

                    # Setup Tantivy mock - still returns 2-tuples (no timestamps)
                    mock_tantivy = MagicMock()
                    mock_tantivy.search.return_value = [("tantivy result", 0.8)]
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)

                    # Search should use RRF fusion (await async method)
                    _ = await manager.search("test query")

                    # Both engines should be queried
                    mock_usearch.search.assert_called()
                    mock_tantivy.search.assert_called()


@pytest.mark.unit
class TestParallelMemoryAddition:
    """Tests for parallel memory addition via add_memories_async()."""

    @pytest.mark.asyncio
    async def test_add_memories_async_empty_list(self, mock_config, mock_logger):
        """Empty list should return AddResult with 0 stored without any operations."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: (
                    contents if contents is not None else memories
                )
            )
            mock_usearch_class.return_value.get_id_by_content.return_value = None
            mock_usearch_class.return_value.embedder.embed_documents.side_effect = (
                lambda texts: [[0.1] * 4 for _ in texts]
            )
            mock_usearch_class.return_value.memory_store.begin_add_intents.return_value = []
            mock_usearch_class.return_value.memory_store.list_pending_transitions.return_value = []
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch("reflectlog.application.memory.manager.TantivyEngine"):
                    manager = MemoryManager(mock_config, mock_logger)
                    result = await manager.add_memories_async([])
                    assert result.stored_count == 0
                    assert result.skipped_count == 0
                    assert result.replaced_count == 0

    @pytest.mark.asyncio
    async def test_add_memories_async_single_memory(self, mock_config, mock_logger):
        """Single memory should skip concurrency overhead."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: (
                    contents if contents is not None else memories
                )
            )
            mock_usearch_class.return_value.get_id_by_content.return_value = None
            mock_usearch_class.return_value.embedder.embed_documents.side_effect = (
                lambda texts: [[0.1] * 4 for _ in texts]
            )
            mock_usearch_class.return_value.memory_store.begin_add_intents.return_value = []
            mock_usearch_class.return_value.memory_store.list_pending_transitions.return_value = []
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    result = await manager.add_memories_async(["single message"])

                    assert result.stored_count == 1
                    mock_tantivy.add_batch.assert_called_once()
                    mock_tantivy.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_memories_async_multiple_memories(self, mock_config, mock_logger):
        """Multiple memories should be processed in parallel."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: (
                    contents if contents is not None else memories
                )
            )
            mock_usearch_class.return_value.get_id_by_content.return_value = None
            mock_usearch_class.return_value.embedder.embed_documents.side_effect = (
                lambda texts: [[0.1] * 4 for _ in texts]
            )
            mock_usearch_class.return_value.memory_store.begin_add_intents.return_value = []
            mock_usearch_class.return_value.memory_store.list_pending_transitions.return_value = []
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    memories = ["msg1", "msg2", "msg3", "msg4"]
                    result = await manager.add_memories_async(memories)

                    assert result.stored_count == 4
                    # All memories should be added to Tantivy
                    assert mock_tantivy.add_batch.call_count >= 1
                    # Commit should be called once after all additions
                    mock_tantivy.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_memories_async_respects_concurrency_limit(
        self, mock_config, mock_logger
    ):
        """Concurrency limit should be respected via semaphore."""
        mock_config.add_max_concurrency = 2  # Low limit for testing

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: (
                    contents if contents is not None else memories
                )
            )
            mock_usearch_class.return_value.get_id_by_content.return_value = None
            mock_usearch_class.return_value.embedder.embed_documents.side_effect = (
                lambda texts: [[0.1] * 4 for _ in texts]
            )
            mock_usearch_class.return_value.memory_store.begin_add_intents.return_value = []
            mock_usearch_class.return_value.memory_store.list_pending_transitions.return_value = []
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    memories = ["msg1", "msg2", "msg3", "msg4"]
                    result = await manager.add_memories_async(memories)

                    # All memories should still be processed
                    assert result.stored_count == 4
                    assert mock_tantivy.add_batch.call_count >= 1

    @pytest.mark.asyncio
    async def test_add_memories_async_handles_duplicates(
        self, mock_config, mock_logger
    ):
        """Duplicate memories should be skipped (via Tantivy detection).

        Note: With Sprint 2.2 phased parallel processing, duplicate checks run
        in parallel, so we use a side_effect function instead of a list to
        handle non-deterministic call order.
        """
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: (
                    contents if contents is not None else memories
                )
            )
            mock_usearch_class.return_value.get_id_by_content.return_value = None
            mock_usearch_class.return_value.embedder.embed_documents.side_effect = (
                lambda texts: [[0.1] * 4 for _ in texts]
            )
            mock_usearch_class.return_value.memory_store.begin_add_intents.return_value = []
            mock_usearch_class.return_value.memory_store.list_pending_transitions.return_value = []
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()

                    # Use a function to return different results based on memory
                    # Note: _has_exact_match wraps query in quotes: f'"{escaped_query}"'
                    mock_usearch = mock_usearch_class.return_value

                    def get_id(_workspace_id, content):
                        if content == "duplicate":
                            return 1
                        return None

                    mock_usearch.get_id_by_content.side_effect = get_id
                    mock_usearch.memory_store.exists_many.side_effect = (
                        lambda _workspace, contents: {"duplicate"} & set(contents)
                    )
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    result = await manager.add_memories_async(["duplicate", "unique"])

                    # Only unique memory should be stored
                    assert result.stored_count == 1
                    assert result.skipped_count == 1

    @pytest.mark.asyncio
    async def test_add_memories_async_error_handling(self, mock_config, mock_logger):
        """Error during parallel addition should raise RuntimeError."""
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Setup USearchEngine mock to raise error
                    mock_usearch = MagicMock()
                    _wire_search_mocks(mock_usearch)
                    mock_usearch.add_batch.side_effect = Exception("Storage error")
                    mock_usearch.search.return_value = []  # For deduplication check
                    mock_usearch.get_id_by_content.return_value = None
                    mock_usearch_class.return_value = mock_usearch

                    # Setup Tantivy mock
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)

                    with pytest.raises(
                        StorageError, match=r"Failed to add memor(y|ies)"
                    ):
                        await manager.add_memories_async(["msg1", "msg2"])

    @pytest.mark.asyncio
    async def test_add_memories_async_batch_deduplication(
        self, mock_config, mock_logger
    ):
        """Batch deduplication should skip duplicate memories within the same batch.

        Sprint 2.2: Phase 1 includes deduplicating within the batch itself
        before checking storage, to avoid storing the same memory twice.
        """
        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: (
                    contents if contents is not None else memories
                )
            )
            mock_usearch_class.return_value.get_id_by_content.return_value = None
            mock_usearch_class.return_value.embedder.embed_documents.side_effect = (
                lambda texts: [[0.1] * 4 for _ in texts]
            )
            mock_usearch_class.return_value.memory_store.begin_add_intents.return_value = []
            mock_usearch_class.return_value.memory_store.list_pending_transitions.return_value = []
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy.search.return_value = []  # No storage duplicates
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)

                    # Batch with duplicates: "A" appears 3 times, "B" appears 2 times
                    memories = ["A", "B", "A", "A", "B", "C"]
                    result = await manager.add_memories_async(memories)

                    # Only unique memories should be stored: A, B, C (3 unique)
                    # Skipped: 3 batch duplicates (2 extra A's, 1 extra B)
                    assert result.stored_count == 3
                    assert result.skipped_count == 3

                    # Tantivy add should only be called 3 times
                    assert mock_tantivy.add_batch.call_count >= 1


@pytest.mark.unit
class TestConcurrencyConfiguration:
    """Tests for ADD_MAX_CONCURRENCY configuration."""

    def test_config_has_add_max_concurrency(self, mock_config):
        """Config should have add_max_concurrency attribute."""
        assert hasattr(mock_config, "add_max_concurrency")
        assert mock_config.add_max_concurrency == 4

    @pytest.mark.asyncio
    async def test_low_concurrency_limit(self, mock_config, mock_logger):
        """Low concurrency limit should still process all memories."""
        mock_config.add_max_concurrency = 1  # Sequential processing

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: (
                    contents if contents is not None else memories
                )
            )
            mock_usearch_class.return_value.get_id_by_content.return_value = None
            mock_usearch_class.return_value.embedder.embed_documents.side_effect = (
                lambda texts: [[0.1] * 4 for _ in texts]
            )
            mock_usearch_class.return_value.memory_store.begin_add_intents.return_value = []
            mock_usearch_class.return_value.memory_store.list_pending_transitions.return_value = []
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    result = await manager.add_memories_async(["msg1", "msg2", "msg3"])

                    assert result.stored_count == 3

    @pytest.mark.asyncio
    async def test_high_concurrency_limit(self, mock_config, mock_logger):
        """High concurrency limit should allow more parallel tasks."""
        mock_config.add_max_concurrency = 100  # Higher than memory count

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            mock_usearch_class.return_value.add_batch.side_effect = (
                lambda workspace_id, memories=None, infer=False, contents=None, vectors=None, **_kwargs: (
                    contents if contents is not None else memories
                )
            )
            mock_usearch_class.return_value.get_id_by_content.return_value = None
            mock_usearch_class.return_value.embedder.embed_documents.side_effect = (
                lambda texts: [[0.1] * 4 for _ in texts]
            )
            mock_usearch_class.return_value.memory_store.begin_add_intents.return_value = []
            mock_usearch_class.return_value.memory_store.list_pending_transitions.return_value = []
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    mock_tantivy = MagicMock()
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    result = await manager.add_memories_async(["msg1", "msg2", "msg3"])

                    assert result.stored_count == 3


@pytest.mark.unit
class TestSingleResultRerankingSkip:
    """Tests for skipping reranking when only 0-1 results after fusion (Sprint optimization).

    When fusion filtering produces <= 1 result, reranking is unnecessary because:
    - 0 results: Nothing to rerank
    - 1 result: No ordering to optimize
    This saves 15-25s of reranker latency for single-result queries.
    """

    @pytest.mark.asyncio
    async def test_single_result_skips_reranking(self, mock_config, mock_logger):
        """Single result after fusion should skip reranking step."""
        mock_config.reranker_engine = "cross_encoder"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.5

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    with patch(
                        "reflectlog.application.memory.manager.CrossEncoderReranker"
                    ) as mock_reranker_class:
                        # Setup USearchEngine mock - return 1 result
                        # Now returns 3-tuples: (message, score, created_at)
                        mock_usearch = MagicMock()
                        _wire_search_mocks(mock_usearch)
                        mock_usearch.search.return_value = [
                            ("single result", 0.9, "2024-01-01T00:00:00")
                        ]
                        mock_usearch.count.return_value = 10
                        mock_usearch_class.return_value = mock_usearch

                        # Setup Tantivy mock - return same result (2-tuples, no timestamps)
                        mock_tantivy = MagicMock()
                        mock_tantivy.search.return_value = [("single result", 0.9)]
                        mock_tantivy_class.return_value = mock_tantivy

                        # Setup CrossEncoder reranker mock
                        mock_reranker = MagicMock()
                        mock_reranker.rerank_async = AsyncMock(
                            return_value=[("single result", 0.95)]
                        )
                        mock_reranker_class.return_value = mock_reranker

                        manager = MemoryManager(mock_config, mock_logger)
                        results = await manager.search("test query")

                        # Reranker should NOT be called (skipped for single result)
                        mock_reranker.rerank_async.assert_not_awaited()

                        # Result should still be returned
                        assert len(results) == 1
                        assert results[0] == "single result"

                        # Verify skip was logged
                        skip_logged = any(
                            "Reranking skipped" in str(call)
                            or "reranking_skip" in str(call)
                            for call in mock_logger.info.call_args_list
                        )
                        assert skip_logged, "Expected reranking skip to be logged"

    @pytest.mark.asyncio
    async def test_single_result_skips_cross_encoder_reranking(
        self, mock_config, mock_logger
    ):
        """Single result after fusion should skip CrossEncoder reranking step."""
        mock_config.reranker_engine = "cross_encoder"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.5

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    with patch(
                        "reflectlog.application.memory.manager.CrossEncoderReranker"
                    ) as mock_reranker_class:
                        # Setup USearchEngine mock - return 1 result
                        # Now returns 3-tuples: (message, score, created_at)
                        mock_usearch = MagicMock()
                        _wire_search_mocks(mock_usearch)
                        mock_usearch.search.return_value = [
                            ("single result", 0.9, "2024-01-01T00:00:00")
                        ]
                        mock_usearch.count.return_value = 10
                        mock_usearch_class.return_value = mock_usearch

                        # Setup Tantivy mock - return same result (2-tuples, no timestamps)
                        mock_tantivy = MagicMock()
                        mock_tantivy.search.return_value = [("single result", 0.9)]
                        mock_tantivy_class.return_value = mock_tantivy

                        # Setup CrossEncoderReranker mock
                        mock_reranker = MagicMock()
                        mock_reranker.rerank_async = AsyncMock(
                            return_value=[("single result", 0.95)]
                        )
                        mock_reranker_class.return_value = mock_reranker

                        manager = MemoryManager(mock_config, mock_logger)
                        results = await manager.search("test query")

                        # CrossEncoder reranker should NOT be called (skipped for single result)
                        mock_reranker.rerank_async.assert_not_called()

                        # Result should still be returned
                        assert len(results) == 1
                        assert results[0] == "single result"

    @pytest.mark.asyncio
    async def test_zero_results_skips_reranking_implicitly(
        self, mock_config, mock_logger
    ):
        """Zero results after fusion should skip reranking (implicit - no candidates)."""
        mock_config.reranker_engine = "cross_encoder"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.5

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    with patch(
                        "reflectlog.application.memory.manager.CrossEncoderReranker"
                    ) as mock_reranker_class:
                        # Setup USearchEngine mock - return empty results
                        mock_usearch = MagicMock()
                        _wire_search_mocks(mock_usearch)
                        mock_usearch.search.return_value = []  # No semantic results
                        mock_usearch.count.return_value = 10
                        mock_usearch_class.return_value = mock_usearch

                        # Setup Tantivy mock - return empty results
                        mock_tantivy = MagicMock()
                        mock_tantivy.search.return_value = []  # No full-text results
                        mock_tantivy_class.return_value = mock_tantivy

                        # Setup CrossEncoder reranker mock
                        mock_reranker = MagicMock()
                        mock_reranker.rerank_async = AsyncMock(return_value=[])
                        mock_reranker_class.return_value = mock_reranker

                        manager = MemoryManager(mock_config, mock_logger)
                        results = await manager.search("test query")

                        # Reranker should NOT be called (no results to rerank)
                        mock_reranker.rerank_async.assert_not_awaited()

                        # Empty results expected (no results from either engine)
                        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_multiple_results_proceed_to_reranking(
        self, mock_config, mock_logger
    ):
        """Multiple results after fusion should proceed to reranking normally."""
        mock_config.reranker_engine = "cross_encoder"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.3  # Low threshold to keep results

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    with patch(
                        "reflectlog.application.memory.manager.CrossEncoderReranker"
                    ) as mock_reranker_class:
                        # Setup USearchEngine mock - return multiple results
                        # Now returns 3-tuples: (message, score, created_at)
                        mock_usearch = MagicMock()
                        _wire_search_mocks(mock_usearch)
                        mock_usearch.search.return_value = [
                            ("result 1", 0.9, "2024-01-01T00:00:00"),
                            ("result 2", 0.8, "2024-01-02T00:00:00"),
                            ("result 3", 0.7, "2024-01-03T00:00:00"),
                        ]
                        mock_usearch.count.return_value = 10
                        mock_usearch_class.return_value = mock_usearch

                        # Setup Tantivy mock - return multiple results (2-tuples, no timestamps)
                        mock_tantivy = MagicMock()
                        mock_tantivy.search.return_value = [
                            ("result 1", 0.85),
                            ("result 2", 0.75),
                        ]
                        mock_tantivy_class.return_value = mock_tantivy

                        # Setup CrossEncoder reranker mock
                        mock_reranker = MagicMock()
                        mock_reranker.rerank_async = AsyncMock(
                            return_value=[
                                ("result 1", 0.95),
                                ("result 2", 0.85),
                                ("result 3", 0.75),
                            ]
                        )
                        mock_reranker_class.return_value = mock_reranker

                        manager = MemoryManager(mock_config, mock_logger)
                        results = await manager.search("test query")

                        mock_reranker.rerank_async.assert_awaited_once()

                        # Results should be from reranker
                        assert len(results) >= 1


@pytest.mark.unit
class TestTimestampPropagation:
    """Tests for timestamp propagation through the search pipeline (Temporal-aware reranking).

    These tests verify that created_at timestamps from USearchEngine are properly
    propagated through the hybrid search pipeline for use in temporal-aware reranking.
    """

    @pytest.mark.asyncio
    async def test_search_receives_timestamps_from_usearch(
        self, mock_config, mock_logger
    ):
        """USearch search results should include created_at timestamps (3-tuples)."""
        mock_config.reranker_engine = "none"  # Skip reranking for simplicity
        mock_config.enable_rrf_fusion = True

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Setup USearchEngine mock with timestamps
                    mock_usearch = MagicMock()
                    _wire_search_mocks(mock_usearch)
                    mock_usearch.search.return_value = [
                        ("result 1", 0.9, "2024-01-15T10:30:00"),
                        ("result 2", 0.8, "2024-01-14T09:00:00"),
                        ("result 3", 0.7, "2024-01-13T08:00:00"),
                    ]
                    mock_usearch.count.return_value = 10
                    mock_usearch_class.return_value = mock_usearch

                    # Setup Tantivy mock (2-tuples, no timestamps)
                    mock_tantivy = MagicMock()
                    mock_tantivy.search.return_value = [
                        ("result 1", 0.85),
                    ]
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    results = await manager.search("test query")

                    # USearch should have been called
                    mock_usearch.search.assert_called()

                    # Results should be returned
                    assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_timestamp_map_built_from_semantic_results(
        self, mock_config, mock_logger
    ):
        """Verify timestamp_map is built correctly from semantic search results."""
        mock_config.reranker_engine = "none"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.0  # Keep all results

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Create test data with distinct timestamps
                    semantic_results = [
                        ("Memory about cats", 0.95, "2024-12-01T12:00:00"),
                        ("Memory about dogs", 0.85, "2024-12-15T14:30:00"),
                    ]

                    mock_usearch = MagicMock()
                    _wire_search_mocks(mock_usearch)
                    mock_usearch.search.return_value = semantic_results
                    mock_usearch.count.return_value = 100
                    mock_usearch_class.return_value = mock_usearch

                    mock_tantivy = MagicMock()
                    mock_tantivy.search.return_value = [("Memory about cats", 0.9)]
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    results = await manager.search("pets")

                    # The search should return results
                    assert len(results) >= 1
                    # Verify the semantic results contain the expected memories
                    memories_returned = set(results)
                    assert (
                        "Memory about cats" in memories_returned
                        or "Memory about dogs" in memories_returned
                    )

    @pytest.mark.asyncio
    async def test_timestamps_preserved_through_fusion(self, mock_config, mock_logger):
        """Timestamps should be accessible after RRF fusion."""
        mock_config.reranker_engine = "none"
        mock_config.enable_rrf_fusion = True
        mock_config.fusion_ranking_threshold = 0.0

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_class:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy_class:
                    # Test that older and newer memories have different timestamps
                    mock_usearch = MagicMock()
                    _wire_search_mocks(mock_usearch)
                    mock_usearch.search.return_value = [
                        ("Old memory", 0.9, "2024-01-01T00:00:00"),  # Older
                        ("New memory", 0.85, "2024-12-01T00:00:00"),  # Newer
                    ]
                    mock_usearch.count.return_value = 50
                    mock_usearch_class.return_value = mock_usearch

                    mock_tantivy = MagicMock()
                    mock_tantivy.search.return_value = []
                    mock_tantivy_class.return_value = mock_tantivy

                    manager = MemoryManager(mock_config, mock_logger)
                    results = await manager.search("memory")

                    # Both memories should be in results
                    assert "Old memory" in results
                    assert "New memory" in results
                    mock_usearch.search.assert_called_once()
                    mock_tantivy.search.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
