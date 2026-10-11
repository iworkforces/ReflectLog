#!/usr/bin/env python3
"""Unit tests for add_phases.py - 3-phase parallel add pipeline.

Covers uncovered lines: 291, 417-524, 588-663, 670, 676, 719,
723-731, 759-768, 782-815, 824, 850-900.
"""

import asyncio
import threading
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.add_phases import (
    AddPipeline,
    AddResult,
    DuplicateDetectionPhase,
    Phase1Result,
    Phase2Result,
    Phase3Result,
    ReplacementInfo,
    SmartReplacementPhase,
    StoragePhase,
    _persist_list_for_add,
)
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.enums import TransitionKind, TransitionStatus
from reflectlog.core.exceptions import InitializationError, StorageError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.types import ReplacementTransition, ReplacementTransitionRequest
from reflectlog.infrastructure.tantivy_engine import TantivyEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> Config:
    """Minimal mock Config for add_phases tests."""
    config = Mock(spec=Config)
    config.workspace_id = "test_project"
    config.add_max_concurrency = 4
    config.rerank_max_concurrency = 5
    config.smart_replace_candidate_limit = 3
    config.smart_replace_min_similarity = 0.5
    config.enable_llm_infer = False
    config.deduplicate_memories = True
    return config


@pytest.fixture
def mock_logger():
    """Mock structured logger."""
    return cast(IStructuredLogger, Mock(spec=StructuredLogger))


@pytest.fixture
def mock_semantic_engine():
    """Mock ISemanticSearchEngine."""
    engine = MagicMock()
    engine.add = MagicMock(return_value=None)
    engine.add_batch = MagicMock(
        side_effect=lambda workspace_id, contents, infer=False, vectors=None, **_kwargs: (
            contents
        )
    )
    engine.search = MagicMock(return_value=[])
    engine.delete = MagicMock(return_value=None)
    engine.commit = MagicMock(return_value=None)
    engine.get_id_by_content = MagicMock(return_value=None)
    engine.contains_id = MagicMock(return_value=False)
    engine.count = MagicMock(return_value=0)
    engine.is_ready = MagicMock(return_value=True)
    engine.embedder.embed_documents.side_effect = lambda texts: [
        [0.1] * 4 for _ in texts
    ]
    engine.embedder.aembed_documents = AsyncMock(
        side_effect=lambda texts: [[0.1] * 4 for _ in texts]
    )
    engine.memory_store = MagicMock()
    engine.memory_store.exists_many = MagicMock(return_value=set())
    engine.memory_store.begin_add_intents = MagicMock(return_value=[])
    engine.memory_store.begin_delete_intents = MagicMock(return_value=[])
    engine.memory_store.begin_replacement_transitions = MagicMock(return_value=[])
    engine.memory_store.list_pending_transitions = MagicMock(return_value=[])
    engine.memory_store.has_later_intent = MagicMock(return_value=False)
    engine.memory_store.get_transition_for_old_memory = MagicMock(return_value=None)
    engine.memory_store.complete_replacement_transition = MagicMock()
    engine.memory_store.get = MagicMock(return_value=None)
    return engine


@pytest.fixture
def mock_tantivy_engine():
    """Mock TantivyEngine."""
    engine = MagicMock()
    engine.add = MagicMock(return_value=None)
    engine.add_batch = MagicMock(return_value=None)
    engine.delete = MagicMock(return_value=True)
    engine.delete_batch = MagicMock(
        side_effect=lambda _workspace, contents, verify_exists=True: len(contents)
    )
    engine.find_by_exact_match = MagicMock(side_effect=lambda _ws, content: [content])
    engine.get_all = MagicMock(return_value=[])
    engine.commit = MagicMock(return_value=None)
    engine.search = MagicMock(return_value=[])
    engine.is_ready = MagicMock(return_value=True)
    return engine


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataclasses:
    """Tests for dataclass constructors and defaults."""

    def test_replacement_info_defaults(self):
        """ReplacementInfo should have default similarity_score of 0.0."""
        info = ReplacementInfo(
            old_memory="old",
            new_memory="new",
            confidence=0.9,
            reason="updated",
        )
        assert info.similarity_score == 0.0
        assert info.old_memory == "old"
        assert info.confidence == 0.9

    def test_add_result_defaults(self):
        """AddResult defaults should be zero counts and empty list."""
        result = AddResult()
        assert result.stored_count == 0
        assert result.skipped_count == 0
        assert result.replaced_count == 0
        assert result.replacements == []

    def test_phase1_result_fields(self):
        """Phase1Result should store all fields."""
        result = Phase1Result(
            unique_memories=["a", "b"],
            storage_duplicates=["c"],
            batch_duplicates_count=1,
            duration=0.5,
        )
        assert len(result.unique_memories) == 2
        assert result.batch_duplicates_count == 1

    def test_phase2_result_fields(self):
        """Phase2Result should store all fields."""
        result = Phase2Result(
            replacement_map={},
            total_replacements=0,
            duration=0.1,
        )
        assert result.total_replacements == 0

    def test_phase3_result_fields(self):
        """Phase3Result should store all fields."""
        result = Phase3Result(
            stored_count=2,
            replaced_count=1,
            replacements=[],
            duration=0.3,
        )
        assert result.stored_count == 2
        assert result.replaced_count == 1


# ---------------------------------------------------------------------------
# DuplicateDetectionPhase Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDuplicateDetectionPhase:
    """Tests for Phase 1: Duplicate Detection."""

    async def test_execute_no_duplicates(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """All unique memories should pass through."""
        phase = DuplicateDetectionPhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )

        # No exact match found
        mock_semantic_engine.get_id_by_content.return_value = None
        mock_tantivy_engine.search.return_value = []

        result = await phase.execute(["msg1", "msg2"])

        assert len(result.unique_memories) == 2
        assert result.batch_duplicates_count == 0
        assert len(result.storage_duplicates) == 0

    async def test_execute_with_batch_duplicates(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Duplicate memories within batch should be deduplicated."""
        phase = DuplicateDetectionPhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        mock_tantivy_engine.search.return_value = []

        result = await phase.execute(["msg1", "msg2", "msg1", "msg2"])

        assert len(result.unique_memories) == 2
        assert result.batch_duplicates_count == 2
        mock_logger.info.assert_called()

    async def test_execute_with_storage_duplicates(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Memories already in storage should be marked as duplicates."""
        phase = DuplicateDetectionPhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )

        def get_id(workspace_id, content):
            if content == "existing":
                return 1
            return None

        mock_semantic_engine.get_id_by_content.side_effect = get_id
        mock_semantic_engine.memory_store.exists_many.return_value = {"existing"}

        result = await phase.execute(["existing", "new_msg"])

        assert "new_msg" in result.unique_memories
        assert "existing" in result.storage_duplicates

    async def test_execute_uses_class_defined_exists_many(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        class Store:
            def exists_many(self, workspace_id: str, contents: list[str]) -> set[str]:
                return {"existing"}

        mock_semantic_engine.memory_store = Store()
        phase = DuplicateDetectionPhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(["existing", "new_msg"])
        assert "existing" in result.storage_duplicates
        assert "new_msg" in result.unique_memories
        mock_semantic_engine.get_id_by_content.assert_not_called()

    async def test_execute_uses_store_exists_many(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        mock_semantic_engine.memory_store.exists_many.return_value = set()
        phase = DuplicateDetectionPhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(["msg1"])
        assert result.unique_memories == ["msg1"]
        mock_semantic_engine.memory_store.exists_many.assert_called_once()
        mock_semantic_engine.get_id_by_content.assert_not_called()


# ---------------------------------------------------------------------------
# SmartReplacementPhase Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSmartReplacementPhase:
    """Tests for Phase 2: Smart Replacement Detection."""

    def test_get_smart_replacer_when_memory_manager_is_none(
        self,
        mock_semantic_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        """No memory manager means no smart replacer."""
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=None,
        )
        assert phase._get_smart_replacer() is None
        assert phase.smart_replace_enabled is False

    def test_get_smart_replacer_returns_replacer(
        self,
        mock_semantic_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        """Memory manager exposes the configured smart replacer."""
        mock_mm = MagicMock()
        replacer = MagicMock()
        mock_mm.smart_replacer = replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        assert phase._get_smart_replacer() is replacer
        assert phase.smart_replace_enabled is True

    async def test_execute_no_smart_replacer(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """Phase 2 should return empty results when no smart replacer (line 311-316)."""
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=None,
        )
        result = await phase.execute(["msg1"])

        assert result.total_replacements == 0
        assert result.replacement_map == {}

    async def test_execute_empty_memories(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """Phase 2 should return early on empty memories list."""
        mock_mm = MagicMock()
        mock_mm.smart_replacer = MagicMock()
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        result = await phase.execute([])

        assert result.total_replacements == 0

    async def test_check_for_replacement_no_similar_results(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """No similar results from semantic search returns empty list."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = []

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []

    async def test_check_for_replacement_below_threshold(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """Candidates below similarity threshold should be filtered out."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        # Score 0.3 is below threshold of 0.5
        mock_semantic_engine.search.return_value = [
            ("old memory", 0.3, "2024-01-01T00:00:00"),
        ]

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []

    async def test_check_for_replacement_same_message_filtered(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """Candidates that are identical to new_memory should be filtered out."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        # Same memory with high score - should be filtered (mem != new_memory)
        mock_semantic_engine.search.return_value = [
            ("new memory", 1.0, "2024-01-01T00:00:00"),
        ]

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []

    async def test_check_for_replacement_should_replace(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """LLM confirms replacement - should return ReplacementInfo (lines 417-471)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = [
            ("old memory about python", 0.85, "2024-01-01T00:00:00"),
        ]
        # LLM says replace
        mock_replacer.check_replacement = AsyncMock(
            return_value=(True, 0.95, "Updated information")
        )

        result = await phase._check_for_replacement(
            "new memory about python", mock_replacer
        )

        assert len(result) == 1
        assert result[0].old_memory == "old memory about python"
        assert result[0].new_memory == "new memory about python"
        assert result[0].confidence == 0.95
        assert result[0].reason == "Updated information"
        assert result[0].similarity_score == 0.85

    async def test_check_for_replacement_should_not_replace(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """LLM says no replacement - returns empty list (lines 472-482)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = [
            ("old memory", 0.85, "2024-01-01T00:00:00"),
        ]
        mock_replacer.check_replacement = AsyncMock(
            return_value=(False, 0.3, "Different topic")
        )

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []

    async def test_check_for_replacement_llm_error_graceful(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """LLM call failure should be handled gracefully (lines 483-493)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = [
            ("old memory", 0.85, "2024-01-01T00:00:00"),
        ]
        mock_replacer.check_replacement = AsyncMock(
            side_effect=RuntimeError("LLM API timeout")
        )

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []
        mock_logger.warning.assert_called()

    async def test_check_for_replacement_outer_exception(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """Outer exception in _check_for_replacement is caught (lines 514-524)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        # Semantic search itself throws
        mock_semantic_engine.search.side_effect = RuntimeError("Search engine failed")

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []
        mock_logger.warning.assert_called()

    async def test_execute_with_replacements(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """Full execute with replacement detection (lines 417-510)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = [
            ("old memory", 0.85, "2024-01-01T00:00:00"),
        ]
        mock_replacer.check_replacement = AsyncMock(
            return_value=(True, 0.9, "Updated info")
        )

        result = await phase.execute(["new memory"])

        assert result.total_replacements == 1
        assert "new memory" in result.replacement_map
        assert len(result.replacement_map["new memory"]) == 1

    async def test_execute_multiple_candidates(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """Multiple candidates above threshold are all checked (lines 505-510)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = [
            ("candidate 1", 0.9, "2024-01-01T00:00:00"),
            ("candidate 2", 0.85, "2024-01-02T00:00:00"),
        ]
        # First candidate should be replaced, second not
        call_count = 0

        def mock_check(new_memory, existing_memory):
            nonlocal call_count
            call_count += 1
            if existing_memory == "candidate 1":
                return (True, 0.95, "Replace this")
            return (False, 0.2, "Keep this")

        mock_replacer.check_replacement = AsyncMock(side_effect=mock_check)

        result = await phase.execute(["new memory"])

        assert result.total_replacements == 1


# ---------------------------------------------------------------------------
# StoragePhase Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStoragePhase:
    """Tests for Phase 3: Sequential Storage."""

    async def test_execute_awaits_ordered_vectors_before_exclusive_lease(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        coordinator = MagicMock()
        memories = ["first", "second", "third"]

        async def async_vectors(texts: list[str]) -> list[list[float]]:
            assert texts == memories
            coordinator.acquire.assert_not_called()

            return [[float(index + 1)] for index in range(len(texts))]

        mock_semantic_engine.embedder.aembed_documents.side_effect = async_vectors
        phase = StoragePhase(
            mock_semantic_engine,
            mock_tantivy_engine,
            mock_config,
            mock_logger,
            coordinator=coordinator,
        )

        result = await phase.execute(memories, {})

        assert result.stored_count == 3
        mock_semantic_engine.embedder.aembed_documents.assert_awaited_once_with(
            memories
        )
        mock_semantic_engine.embedder.embed_documents.assert_not_called()
        assert mock_semantic_engine.add_batch.call_args.kwargs["vectors"] == [
            [1.0],
            [2.0],
            [3.0],
        ]
        coordinator.acquire.assert_called_once()

    @pytest.mark.parametrize("vectors", [[], [[1.0]], [[1.0], []]])
    async def test_execute_rejects_invalid_async_vectors_before_persist(
        self,
        vectors: list[list[float]],
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        coordinator = MagicMock()
        mock_semantic_engine.embedder.aembed_documents.side_effect = None
        mock_semantic_engine.embedder.aembed_documents.return_value = vectors
        phase = StoragePhase(
            mock_semantic_engine,
            mock_tantivy_engine,
            mock_config,
            mock_logger,
            coordinator=coordinator,
        )

        with pytest.raises(StorageError, match="Embedding"):
            await phase.execute(["first", "second"], {})

        coordinator.acquire.assert_not_called()
        mock_semantic_engine.add_batch.assert_not_called()
        mock_semantic_engine.memory_store.begin_add_intents.assert_not_called()

    async def test_cancelled_async_embedding_does_not_start_persist(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        entered = asyncio.Event()

        async def blocked_embed(_texts: list[str]) -> list[list[float]]:
            entered.set()
            await asyncio.Event().wait()
            return [[1.0]]

        mock_semantic_engine.embedder.aembed_documents.side_effect = blocked_embed
        phase = StoragePhase(
            mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
        )
        task = asyncio.create_task(phase.execute(["first"], {}))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        mock_semantic_engine.add_batch.assert_not_called()

    def test_sync_persist_keeps_sync_embedding_contract(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        phase = StoragePhase(
            mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
        )

        count, replacements = phase._persist_replacements(["first"], {})

        assert count == 1
        assert replacements == []
        mock_semantic_engine.embedder.embed_documents.assert_called_once_with(["first"])
        mock_semantic_engine.embedder.aembed_documents.assert_not_awaited()
        assert mock_semantic_engine.add_batch.call_args.kwargs["vectors"] == [[0.1] * 4]

    def test_add_intents_complete_only_after_both_backends_converge_with_one_scan(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        contents = ["ready", "missing tantivy", "missing sqlite", "ready too"]
        intents = [
            ReplacementTransition(
                id=index,
                workspace_id="test_project",
                old_memory_id=0,
                old_content="",
                new_content=content,
                archive_id=0,
                reason="",
                confidence=0,
                status=TransitionStatus.PENDING,
                kind=TransitionKind.ADD,
            )
            for index, content in enumerate(contents, 1)
        ]
        mock_semantic_engine.get_id_by_content.side_effect = (
            lambda _workspace, content: None if content == "missing sqlite" else 1
        )
        mock_tantivy_engine.get_all.return_value = [
            "ready",
            "missing sqlite",
            "ready too",
            "Missing tantivy",
        ]
        phase = StoragePhase(
            mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
        )

        phase._complete_add_intents(intents)

        mock_tantivy_engine.get_all.assert_called_once_with("test_project")
        mock_tantivy_engine.find_by_exact_match.assert_not_called()
        assert (
            mock_semantic_engine.memory_store.complete_replacement_transition.call_args_list
            == [
                ((1,), {}),
                ((4,), {}),
            ]
        )

    def test_add_intents_empty_batch_skips_scan(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        phase = StoragePhase(
            mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
        )

        phase._complete_add_intents([])

        mock_tantivy_engine.get_all.assert_not_called()
        mock_semantic_engine.memory_store.complete_replacement_transition.assert_not_called()

    def test_add_intents_snapshot_failure_leaves_intents_pending(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        intent = ReplacementTransition(
            id=1,
            workspace_id="test_project",
            old_memory_id=0,
            old_content="",
            new_content="ready",
            archive_id=0,
            reason="",
            confidence=0,
            status=TransitionStatus.PENDING,
            kind=TransitionKind.ADD,
        )
        mock_tantivy_engine.get_all.side_effect = RuntimeError("snapshot failed")
        phase = StoragePhase(
            mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
        )

        with pytest.raises(RuntimeError, match="snapshot failed"):
            phase._complete_add_intents([intent])

        mock_semantic_engine.memory_store.complete_replacement_transition.assert_not_called()

    async def test_execute_no_replacements(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Simple storage without replacements."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(["msg1", "msg2"], replacement_map={})

        assert result.stored_count == 2
        assert result.replaced_count == 0
        mock_semantic_engine.add_batch.assert_called_once()
        mock_semantic_engine.commit.assert_called_once()
        mock_tantivy_engine.commit.assert_called_once()

    async def test_execute_dry_run_no_storage(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Dry run should not write to storage (line 670)."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(["msg1"], replacement_map={}, dry_run=True)

        assert result.stored_count == 1
        mock_semantic_engine.add_batch.assert_not_called()
        mock_semantic_engine.commit.assert_not_called()
        mock_tantivy_engine.commit.assert_not_called()

    async def test_execute_dry_run_with_replacements(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Dry run should record replacements without deleting (lines 660-663)."""
        replacement_info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
            similarity_score=0.85,
        )
        mock_semantic_engine.get_id_by_content.return_value = 42
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(
            ["new msg"],
            replacement_map={"new msg": [replacement_info]},
            dry_run=True,
        )

        assert result.replaced_count == 1
        assert len(result.replacements) == 1
        assert result.stored_count == 1
        # No actual deletes or durable records in dry_run
        mock_semantic_engine.delete.assert_not_called()
        mock_semantic_engine.memory_store.begin_replacement_transitions.assert_not_called()
        mock_semantic_engine.memory_store.archive.assert_not_called()

    async def test_execute_with_replacement_success(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Replacement flow: archive + delete + add (lines 588-649)."""
        replacement_info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
            similarity_score=0.85,
        )
        live_ids = {"old msg": 42, "new msg": 99}

        def get_id(_ws: str, content: str) -> int | None:
            return live_ids.get(content)

        def delete(*, memory_id: str) -> None:
            if memory_id == "42":
                live_ids.pop("old msg", None)

        mock_semantic_engine.get_id_by_content.side_effect = get_id
        mock_semantic_engine.delete.side_effect = delete
        mock_semantic_engine.index = {99}
        mock_semantic_engine.contains_id.side_effect = lambda memory_id: (
            memory_id in mock_semantic_engine.index
        )
        mock_tantivy_engine.find_by_exact_match.side_effect = lambda _ws, content: (
            [content] if content == "new msg" else []
        )
        mock_semantic_engine.memory_store.begin_replacement_transitions.return_value = [
            ReplacementTransition(
                id=9,
                workspace_id="test_project",
                old_memory_id=42,
                old_content="old msg",
                new_content="new msg",
                archive_id=100,
                reason="updated",
                confidence=0.9,
                status=TransitionStatus.PENDING,
            )
        ]

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(
            ["new msg"],
            replacement_map={"new msg": [replacement_info]},
        )

        assert result.replaced_count == 1
        assert result.stored_count == 1
        assert len(result.replacements) == 1
        mock_semantic_engine.memory_store.begin_replacement_transitions.assert_called_once()
        mock_semantic_engine.delete.assert_called_once_with(memory_id="42")
        mock_tantivy_engine.delete_batch.assert_called_once_with(
            "test_project", ["old msg"], verify_exists=True
        )
        mock_semantic_engine.memory_store.complete_replacement_transition.assert_called_once_with(
            9
        )

    async def test_execute_replacement_old_memory_not_found(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """When old memory ID not found, skip replacement (lines 625-633)."""
        replacement_info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
        )
        mock_semantic_engine.get_id_by_content.return_value = None
        mock_semantic_engine.memory_store.archive.return_value = 100

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(
            ["new msg"],
            replacement_map={"new msg": [replacement_info]},
        )

        # Replacement skipped because msg_id was None
        assert result.replaced_count == 0
        mock_semantic_engine.delete.assert_not_called()
        mock_semantic_engine.memory_store.begin_replacement_transitions.assert_not_called()
        mock_logger.warning.assert_called()

    async def test_execute_replacement_delete_error(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Delete error during replacement is caught gracefully (lines 650-659)."""
        replacement_info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
        )
        mock_semantic_engine.get_id_by_content.return_value = 42
        mock_semantic_engine.index = set()
        mock_semantic_engine.memory_store.begin_replacement_transitions.return_value = [
            ReplacementTransition(
                id=9,
                workspace_id="test_project",
                old_memory_id=42,
                old_content="old msg",
                new_content="new msg",
                archive_id=100,
                reason="updated",
                confidence=0.9,
                status=TransitionStatus.PENDING,
            )
        ]
        mock_semantic_engine.delete.side_effect = RuntimeError("Delete failed")

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        with pytest.raises(StorageError, match="Failed to delete old memory"):
            await phase.execute(
                ["new msg"],
                replacement_map={"new msg": [replacement_info]},
            )

        mock_semantic_engine.memory_store.complete_replacement_transition.assert_not_called()
        mock_logger.warning.assert_called()

    async def test_execute_batch_stored_fewer_than_expected(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Warning when batch add returns fewer memories (line 676)."""
        # Clear side_effect from fixture so return_value is used
        mock_semantic_engine.add_batch.side_effect = None
        mock_semantic_engine.add_batch.return_value = ["msg1"]

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(["msg1", "msg2"], replacement_map={})

        assert result.stored_count == 1
        mock_logger.warning.assert_called()

    async def test_execute_no_tantivy(
        self, mock_semantic_engine: MagicMock, mock_config: Config, mock_logger: Mock
    ):
        """Storage without tantivy engine should still work."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(["msg1"], replacement_map={})

        assert result.stored_count == 1
        mock_semantic_engine.commit.assert_called_once()

    # -----------------------------------------------------------------------
    # _add_memories_batch Tests (lines 709-755)
    # -----------------------------------------------------------------------

    def test_add_memories_batch_no_write_lock_no_tantivy(
        self,
        mock_semantic_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        """Batch add without a write lock or Tantivy still persists to USearch."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )
        inserted = phase._add_memories_batch(["a", "b"])
        assert inserted == ["a", "b"]
        mock_semantic_engine.add_batch.assert_called_once()

    def test_add_memories_batch_with_write_lock_no_tantivy(
        self,
        mock_semantic_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        """Batch add acquires the provided write lock."""
        write_lock = threading.Lock()
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
            write_lock=write_lock,
        )
        inserted = phase._add_memories_batch(["only"])
        assert inserted == ["only"]
        mock_semantic_engine.add_batch.assert_called_once()

    def test_add_memories_batch_exception_raises_storage_error(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        """Engine failures are wrapped in StorageError."""
        mock_semantic_engine.add_batch.side_effect = RuntimeError("boom")
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )
        with pytest.raises(StorageError, match="Failed to add memory batch"):
            phase._add_memories_batch(["x"])

    def test_add_memories_unlocked_passes_vectors(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        """Precomputed vectors are forwarded to add_batch."""
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )
        inserted = phase._add_memories_unlocked(["a", "b"], vectors)
        assert inserted == ["a", "b"]
        kwargs = mock_semantic_engine.add_batch.call_args.kwargs
        assert kwargs["vectors"] == vectors

    def test_add_memories_unlocked_uses_class_defined_add_batch(
        self,
        mock_semantic_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        """Class-level add_batch is used; MagicMock auto-attrs are ignored."""

        class FakeTantivy:
            def __init__(self) -> None:
                self.added: list[str] = []
                self.batched: list[list[str]] = []

            def add(self, workspace_id: str, content: str) -> None:
                self.added.append(content)

            def add_batch(self, workspace_id: str, contents: list[str]) -> int:
                self.batched.append(list(contents))
                return len(contents)

        tantivy = FakeTantivy()
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=cast(TantivyEngine, tantivy),
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )
        inserted = phase._add_memories_unlocked(["a", "b"])
        assert inserted == ["a", "b"]
        assert tantivy.batched == [["a", "b"]]
        assert tantivy.added == []

    # -----------------------------------------------------------------------
    # _has_exact_match Tests (line 824)
    # -----------------------------------------------------------------------

    def test_has_exact_match_delegates(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        """Exact match is true when the semantic store has an id."""
        mock_semantic_engine.get_id_by_content.return_value = 99
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )
        assert phase._has_exact_match("hello") is True

    def test_has_exact_match_no_match(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        """Exact match is false when no id is found."""
        mock_semantic_engine.get_id_by_content.return_value = None
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )
        assert phase._has_exact_match("hello") is False

    async def test_records_all_transitions_before_any_delete(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Every intended old is recorded before the first index delete."""
        order: list[str] = []
        mock_semantic_engine.get_id_by_content.side_effect = lambda _ws, content: {
            "old-a": 11,
            "old-b": 12,
            "new msg": 99,
        }.get(content)
        mock_semantic_engine.index = {99}
        mock_tantivy_engine.find_by_exact_match.side_effect = lambda _ws, content: (
            [content] if content == "new msg" else []
        )

        def record(
            requests: list[ReplacementTransitionRequest],
        ) -> list[ReplacementTransition]:
            recorded: list[ReplacementTransition] = []
            for request in requests:
                order.append("record")
                recorded.append(
                    ReplacementTransition(
                        id=len(order),
                        workspace_id=request.workspace_id,
                        old_memory_id=request.old_memory_id,
                        old_content=request.old_content,
                        new_content=request.new_content,
                        archive_id=100,
                        reason=request.reason,
                        confidence=request.confidence,
                        status=TransitionStatus.PENDING,
                    )
                )
            return recorded

        def delete(*, memory_id: str) -> None:
            order.append(f"delete:{memory_id}")

        mock_semantic_engine.memory_store.begin_replacement_transitions.side_effect = (
            record
        )
        mock_semantic_engine.delete.side_effect = delete

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        infos = [
            ReplacementInfo(
                old_memory="old-a", new_memory="new msg", confidence=0.9, reason="a"
            ),
            ReplacementInfo(
                old_memory="old-b", new_memory="new msg", confidence=0.9, reason="b"
            ),
        ]
        result = await phase.execute(["new msg"], replacement_map={"new msg": infos})
        assert result.replaced_count == 2
        assert order[:2] == ["record", "record"]
        assert order[2].startswith("delete:")

    async def test_archive_failure_prevents_delete(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """An archive/transition failure must not delete the old memory."""
        replacement_info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
        )
        mock_semantic_engine.get_id_by_content.return_value = 42
        mock_semantic_engine.memory_store.begin_replacement_transitions.side_effect = (
            StorageError("archive failed")
        )

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        with pytest.raises(StorageError, match="Failed to record replacement"):
            await phase.execute(
                ["new msg"],
                replacement_map={"new msg": [replacement_info]},
            )

        mock_semantic_engine.delete.assert_not_called()
        mock_tantivy_engine.delete_batch.assert_not_called()

    async def test_keeps_one_successor_per_old_id(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Two successors for one old id keep the highest-confidence winner."""
        loser = ReplacementInfo(
            old_memory="old msg",
            new_memory="low",
            confidence=0.4,
            reason="weak",
        )
        winner = ReplacementInfo(
            old_memory="old msg",
            new_memory="high",
            confidence=0.95,
            reason="strong",
        )
        live_ids = {"old msg": 42}

        def get_id(_ws: str, content: str) -> int | None:
            return live_ids.get(content)

        def delete(*, memory_id: str) -> None:
            if memory_id == "42":
                live_ids.pop("old msg", None)

        mock_semantic_engine.get_id_by_content.side_effect = get_id
        mock_semantic_engine.delete.side_effect = delete
        mock_semantic_engine.index = set()
        mock_semantic_engine.add_batch.side_effect = None
        mock_semantic_engine.add_batch.return_value = ["low", "high"]
        mock_semantic_engine.memory_store.begin_replacement_transitions.return_value = [
            ReplacementTransition(
                id=1,
                workspace_id="test_project",
                old_memory_id=42,
                old_content="old msg",
                new_content="high",
                archive_id=1,
                reason="strong",
                confidence=0.95,
                status=TransitionStatus.PENDING,
            )
        ]

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(
            ["low", "high"],
            {"low": [loser], "high": [winner]},
        )

        assert result.stored_count == 2
        assert result.replaced_count == 1
        assert result.replacements[0].new_memory == "high"
        mock_semantic_engine.memory_store.begin_replacement_transitions.assert_called_once()
        recorded = (
            mock_semantic_engine.memory_store.begin_replacement_transitions.call_args[
                0
            ][0]
        )
        assert recorded[0].old_memory_id == 42
        assert recorded[0].new_content == "high"
        assert recorded[0].reason == "strong"
        mock_semantic_engine.add_batch.assert_called_once_with(
            workspace_id="test_project",
            contents=["low", "high"],
            infer=False,
            vectors=[[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
        )

    async def test_dry_run_collapses_two_successors(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Dry-run replaced_count matches the one-successor plan."""
        loser = ReplacementInfo(
            old_memory="old msg",
            new_memory="low",
            confidence=0.4,
            reason="weak",
        )
        winner = ReplacementInfo(
            old_memory="old msg",
            new_memory="high",
            confidence=0.95,
            reason="strong",
        )
        mock_semantic_engine.get_id_by_content.return_value = 42

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(
            ["low", "high"],
            {"low": [loser], "high": [winner]},
            dry_run=True,
        )

        assert result.stored_count == 2
        assert result.replaced_count == 1
        assert result.replacements[0].confidence == 0.95
        mock_semantic_engine.memory_store.begin_replacement_transitions.assert_not_called()
        mock_semantic_engine.delete.assert_not_called()

    async def test_skips_successor_that_conflicts_with_existing_transition(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """A leftover exclusive row must not abort unrelated memories."""
        replacement_info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
        )
        mock_semantic_engine.get_id_by_content.return_value = 42
        mock_semantic_engine.add_batch.side_effect = None
        mock_semantic_engine.add_batch.return_value = ["new msg", "other"]
        mock_semantic_engine.memory_store.get_transition_for_old_memory.return_value = (
            ReplacementTransition(
                id=1,
                workspace_id="test_project",
                old_memory_id=42,
                old_content="old msg",
                new_content="already recorded",
                archive_id=1,
                reason="first intent",
                confidence=0.8,
                status=TransitionStatus.PENDING,
            )
        )

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(
            ["new msg", "other"],
            {"new msg": [replacement_info]},
        )

        assert result.stored_count == 2
        assert result.replaced_count == 0
        mock_semantic_engine.memory_store.begin_replacement_transitions.assert_not_called()
        mock_semantic_engine.delete.assert_not_called()
        mock_semantic_engine.add_batch.assert_called_once_with(
            workspace_id="test_project",
            contents=["new msg", "other"],
            infer=False,
            vectors=[[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
        )

    async def test_reconcile_runs_at_persist_start_when_write_lock_set(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """Start-of-persist reconcile is invoked when a write lock is present."""
        lock = threading.Lock()
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
            write_lock=lock,
        )
        with patch(
            "reflectlog.application.memory.add_phases.reconcile_pending_replacements",
            return_value=0,
        ) as reconcile:
            result = await phase.execute([], {})

        assert result.stored_count == 0
        mock_semantic_engine.embedder.aembed_documents.assert_not_awaited()
        reconcile.assert_called_once()

    async def test_reconcile_skipped_without_write_lock(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ):
        """No write lock means start-of-persist reconcile is a no-op."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )
        with patch(
            "reflectlog.application.memory.add_phases.reconcile_pending_replacements",
            return_value=0,
        ) as reconcile:
            result = await phase.execute(["msg1"], {})

        assert result.stored_count == 1
        reconcile.assert_not_called()


# ---------------------------------------------------------------------------
# AddPipeline Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddPipeline:
    """Tests for the AddPipeline orchestrator."""

    async def test_allow_replace_false_skips_phase2(
        self, mock_config: Config, mock_logger: Mock
    ) -> None:
        phase1 = MagicMock(spec=DuplicateDetectionPhase)
        phase1.execute = AsyncMock(
            return_value=Phase1Result(
                unique_memories=["new"],
                storage_duplicates=[],
                batch_duplicates_count=0,
                duration=0.0,
            )
        )
        phase2 = MagicMock(spec=SmartReplacementPhase)
        phase2.execute = AsyncMock(
            side_effect=AssertionError("replacement must not run")
        )
        phase3 = MagicMock(spec=StoragePhase)
        phase3.execute = AsyncMock(
            return_value=Phase3Result(
                stored_count=1, replaced_count=0, replacements=[], duration=0.0
            )
        )
        pipeline = AddPipeline(phase1, phase2, phase3, mock_config, mock_logger)
        result = await pipeline.execute(["new"], allow_replace=False)
        assert result.stored_count == 1
        assert result.replaced_count == 0
        phase2.execute.assert_not_awaited()
        phase3.execute.assert_awaited_once_with(["new"], {}, False)

    async def test_execute_empty_memories(self, mock_config: Config, mock_logger: Mock):
        """Empty memories list returns default AddResult."""
        phase1 = MagicMock()
        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = None
        phase3 = MagicMock()

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        result = await pipeline.execute([])

        assert result.stored_count == 0
        assert result.skipped_count == 0
        phase1.execute.assert_not_called()

    async def test_execute_full_pipeline(self, mock_config: Config, mock_logger: Mock):
        """Full pipeline: Phase 1 → Phase 2 → Phase 3."""
        phase1 = MagicMock()
        phase1.execute = AsyncMock(
            return_value=Phase1Result(
                unique_memories=["msg1"],
                storage_duplicates=["dup"],
                batch_duplicates_count=1,
                duration=0.1,
            )
        )

        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = None
        phase2.execute = AsyncMock(
            return_value=Phase2Result(
                replacement_map={},
                total_replacements=0,
                duration=0.1,
            )
        )

        phase3 = MagicMock()
        phase3.execute = AsyncMock(
            return_value=Phase3Result(
                stored_count=1,
                replaced_count=0,
                replacements=[],
                duration=0.1,
            )
        )

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        result = await pipeline.execute(["msg1", "dup", "msg1"])

        assert result.stored_count == 1
        # skipped = storage_duplicates(1) + batch_duplicates(1) = 2
        assert result.skipped_count == 2

    async def test_execute_pipeline_error_live_mode(
        self, mock_config: Config, mock_logger: Mock
    ):
        """Pipeline error in live mode raises StorageError (lines 988-1003)."""
        phase1 = MagicMock()
        phase1.execute = MagicMock(side_effect=RuntimeError("Phase 1 failed"))

        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = None
        phase3 = MagicMock()

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )

        with pytest.raises(StorageError, match="Failed to add memories"):
            await pipeline.execute(["msg1"], dry_run=False)

        mock_logger.error.assert_called()

    async def test_execute_pipeline_error_dry_run_mode(
        self, mock_config: Config, mock_logger: Mock
    ):
        """Pipeline error in dry_run mode does not raise (lines 1002-1003)."""
        phase1 = MagicMock()
        phase1.execute = MagicMock(side_effect=RuntimeError("Phase 1 failed"))

        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = None
        phase3 = MagicMock()

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        # Should NOT raise in dry_run mode
        result = await pipeline.execute(["msg1"], dry_run=True)

        assert result.stored_count == 0
        mock_logger.error.assert_called()

    async def test_execute_dry_run_full_pipeline(
        self, mock_config: Config, mock_logger: Mock
    ):
        """Full pipeline in dry_run mode."""
        phase1 = MagicMock()
        phase1.execute = AsyncMock(
            return_value=Phase1Result(
                unique_memories=["msg1"],
                storage_duplicates=[],
                batch_duplicates_count=0,
                duration=0.1,
            )
        )

        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = MagicMock()
        phase2.execute = AsyncMock(
            return_value=Phase2Result(
                replacement_map={},
                total_replacements=0,
                duration=0.1,
            )
        )

        phase3 = MagicMock()
        phase3.execute = AsyncMock(
            return_value=Phase3Result(
                stored_count=1,
                replaced_count=0,
                replacements=[],
                duration=0.1,
            )
        )

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        result = await pipeline.execute(["msg1"], dry_run=True)

        assert result.stored_count == 1
        phase3.execute.assert_called_once()

    async def test_execute_with_replacements(
        self, mock_config: Config, mock_logger: Mock
    ):
        """Pipeline with replacements propagates results."""
        replacement = ReplacementInfo(
            old_memory="old",
            new_memory="new",
            confidence=0.9,
            reason="updated",
            similarity_score=0.85,
        )

        phase1 = MagicMock()
        phase1.execute = AsyncMock(
            return_value=Phase1Result(
                unique_memories=["new"],
                storage_duplicates=[],
                batch_duplicates_count=0,
                duration=0.1,
            )
        )

        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = MagicMock()
        phase2.execute = AsyncMock(
            return_value=Phase2Result(
                replacement_map={"new": [replacement]},
                total_replacements=1,
                duration=0.1,
            )
        )

        phase3 = MagicMock()
        phase3.execute = AsyncMock(
            return_value=Phase3Result(
                stored_count=1,
                replaced_count=1,
                replacements=[replacement],
                duration=0.1,
            )
        )

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        result = await pipeline.execute(["new"])

        assert result.stored_count == 1
        assert result.replaced_count == 1
        assert len(result.replacements) == 1
        assert result.replacements[0].old_memory == "old"


class TestPersistListForAdd:
    """Replacement olds and dry-run must not treat storage dups as stored."""

    def test_dry_run_skips_all_storage_duplicates(self) -> None:
        persist, skipped = _persist_list_for_add(
            unique_memories=["new"],
            storage_duplicates=["old"],
            replacement_map={},
            dry_run=True,
        )
        assert persist == ["new"]
        assert skipped == 1

    def test_live_excludes_replacement_olds(self) -> None:
        persist, skipped = _persist_list_for_add(
            unique_memories=["I moved to Boston"],
            storage_duplicates=["I live in NYC"],
            replacement_map={
                "I moved to Boston": [
                    ReplacementInfo(
                        old_memory="I live in NYC",
                        new_memory="I moved to Boston",
                        confidence=0.9,
                        reason="updated",
                    )
                ]
            },
            dry_run=False,
        )
        assert persist == ["I moved to Boston"]
        assert skipped == 1

    def test_live_keeps_unrelated_storage_duplicates(self) -> None:
        persist, skipped = _persist_list_for_add(
            unique_memories=["fresh"],
            storage_duplicates=["already there"],
            replacement_map={},
            dry_run=False,
        )
        assert persist == ["fresh", "already there"]
        assert skipped == 0

    async def test_pipeline_does_not_persist_replacement_old(
        self, mock_config: Config, mock_logger: Mock
    ) -> None:
        replacement = ReplacementInfo(
            old_memory="I live in NYC",
            new_memory="I moved to Boston",
            confidence=0.9,
            reason="updated",
        )
        phase1 = MagicMock()
        phase1.execute = AsyncMock(
            return_value=Phase1Result(
                unique_memories=["I moved to Boston"],
                storage_duplicates=["I live in NYC"],
                batch_duplicates_count=0,
                duration=0.1,
            )
        )
        phase2 = MagicMock()
        phase2.execute = AsyncMock(
            return_value=Phase2Result(
                replacement_map={"I moved to Boston": [replacement]},
                total_replacements=1,
                duration=0.1,
            )
        )
        phase3 = MagicMock()
        phase3.execute = AsyncMock(
            return_value=Phase3Result(
                stored_count=1,
                replaced_count=1,
                replacements=[replacement],
                duration=0.1,
            )
        )
        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        result = await pipeline.execute(["I moved to Boston", "I live in NYC"])
        assert result.stored_count == 1
        assert result.skipped_count == 1
        called = phase3.execute.await_args
        assert called is not None
        persisted = called.args[0]
        assert persisted == ["I moved to Boston"]
        assert "I live in NYC" not in persisted

    async def test_dry_run_counts_storage_duplicate_as_skipped(
        self, mock_config: Config, mock_logger: Mock
    ) -> None:
        phase1 = MagicMock()
        phase1.execute = AsyncMock(
            return_value=Phase1Result(
                unique_memories=["fresh"],
                storage_duplicates=["already"],
                batch_duplicates_count=0,
                duration=0.1,
            )
        )
        phase2 = MagicMock()
        phase2.execute = AsyncMock(
            return_value=Phase2Result(
                replacement_map={},
                total_replacements=0,
                duration=0.1,
            )
        )
        phase3 = MagicMock()
        phase3.execute = AsyncMock(
            return_value=Phase3Result(
                stored_count=1,
                replaced_count=0,
                replacements=[],
                duration=0.1,
            )
        )
        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        result = await pipeline.execute(["fresh", "already"], dry_run=True)
        assert result.stored_count == 1
        assert result.skipped_count == 1
        called = phase3.execute.await_args
        assert called is not None
        persisted = called.args[0]
        assert persisted == ["fresh"]


@pytest.mark.unit
class TestStoragePhaseCoordination:
    def test_revalidate_drops_stale_duplicate(self, mock_config, mock_logger) -> None:
        from reflectlog.application.memory.add_phases import StoragePhase

        semantic = MagicMock()
        semantic.get_id_by_content.return_value = 9
        semantic.embedder.embed_documents.return_value = [[0.1, 0.2]]
        semantic.add_batch.return_value = []
        semantic.memory_store.begin_add_intents.return_value = []
        phase = StoragePhase(
            semantic_engine=semantic,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
        )
        mock_config.deduplicate_memories = True
        kept, vectors = phase._revalidate_persist_inputs(["dup"], [[0.1, 0.2]], {})
        assert kept == []
        assert vectors == []


REFUSAL = "Restore a consistent backup or rebuild this workspace offline."


@pytest.mark.unit
class TestRefusedIndexIsNeverWrapped:
    """InitializationError keeps its type and operator text through the add path."""

    def test_add_batch_refusal_is_not_wrapped_in_storage_error(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        refusal = InitializationError(REFUSAL)
        mock_semantic_engine.add_batch.side_effect = refusal
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )

        with pytest.raises(InitializationError) as raised:
            phase._add_memories_batch(["x"])

        assert raised.value is refusal

    def test_replacement_delete_refusal_is_not_wrapped_in_storage_error(
        self,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Config,
        mock_logger: Mock,
    ) -> None:
        refusal = InitializationError(REFUSAL)
        mock_semantic_engine.delete.side_effect = refusal
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        transition = ReplacementTransition(
            id=9,
            workspace_id="test_project",
            old_memory_id=42,
            old_content="old msg",
            new_content="new msg",
            archive_id=100,
            reason="updated",
            confidence=0.9,
            status=TransitionStatus.PENDING,
        )

        with pytest.raises(InitializationError) as raised:
            phase._delete_recorded_olds([transition])

        assert raised.value is refusal

    async def test_pipeline_does_not_wrap_a_refusal(
        self, mock_config: Config, mock_logger: Mock
    ) -> None:
        refusal = InitializationError(REFUSAL)
        phase1 = MagicMock(spec=DuplicateDetectionPhase)
        phase1.execute = AsyncMock(side_effect=refusal)
        phase2 = MagicMock(spec=SmartReplacementPhase)
        phase2._get_smart_replacer.return_value = None
        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=MagicMock(spec=StoragePhase),
            config=mock_config,
            logger=mock_logger,
        )

        with pytest.raises(InitializationError) as raised:
            await pipeline.execute(["msg1"], dry_run=False)

        assert raised.value is refusal
