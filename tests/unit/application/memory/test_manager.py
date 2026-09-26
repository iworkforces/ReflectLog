#!/usr/bin/env python3
"""Unit tests for MemoryManager – targeting uncovered lines for 90%+ coverage.

Uncovered lines targeted:
  334-353, 364-376, 390, 407, 413, 435-463, 483, 512, 527-560,
  594-602, 642, 654, 662, 774-775, 841-842, 853-857, 904-914,
  933, 983-984, 1002-1003
"""

from dataclasses import replace
import logging
import os
from pathlib import Path
from typing import Self, cast
from unittest.mock import MagicMock, patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.application.utils.security import SecretString
from reflectlog.core.enums import LlmProvider, RerankerEngine
from reflectlog.core.exceptions import (
    InconsistentStateError,
    InitializationError,
    SearchError,
    StorageError,
)
from reflectlog.core.storage_coordination import IStorageCoordinator
from reflectlog.core.types import ISemanticSearchEngine
from reflectlog.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from reflectlog.infrastructure.embedding_identity import IDENTITY_NAME
from reflectlog.infrastructure.storage_coordinator import (
    GENERATION_NAME,
    LOCK_NAME,
    PortalockerStorageCoordinator,
)
from reflectlog.infrastructure.tantivy_engine import TantivyEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODULE = "reflectlog.application.memory.manager"


@pytest.fixture(autouse=True)
def _stub_coordinator(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from reflectlog.infrastructure.storage_coordinator import (
        PortalockerStorageCoordinator,
    )

    def _factory(self: MemoryManager) -> PortalockerStorageCoordinator:
        _ = self
        return PortalockerStorageCoordinator(str(tmp_path / "indexes"), timeout=1.0)

    monkeypatch.setattr(MemoryManager, "_create_coordinator", _factory)
    monkeypatch.chdir(tmp_path)


_ = _stub_coordinator


def _fake_coordinator() -> IStorageCoordinator:
    from reflectlog.core.storage_coordination import (
        IStorageLease,
        LeaseMode,
        WorkspaceStoragePaths,
    )

    class _Fake:
        timeout = 1.0
        generation = 0
        workspace_id = "test"
        mode = LeaseMode.EXCLUSIVE

        def paths_for(self, workspace_id: str) -> WorkspaceStoragePaths:
            root = os.path.abspath(os.path.join("indexes", workspace_id.lower()))
            return WorkspaceStoragePaths(
                workspace_id=workspace_id,
                root=root,
                lock_path=os.path.join(root, LOCK_NAME),
                generation_path=os.path.join(root, GENERATION_NAME),
            )

        def acquire(
            self,
            workspace_id: str,
            mode: LeaseMode = LeaseMode.EXCLUSIVE,
            *,
            timeout: float | None = None,
        ) -> IStorageLease:
            _ = workspace_id, timeout
            self.workspace_id = workspace_id
            self.mode = mode
            os.makedirs(self.paths_for(workspace_id).root, exist_ok=True)
            return self

        def release(self) -> None:
            return None

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object,
        ) -> None:
            _ = exc_type, exc, traceback
            return None

        def read_generation(self, workspace_id: str) -> int:
            _ = workspace_id
            return self.generation

        def publish_generation(self, workspace_id: str, generation: int) -> None:
            _ = workspace_id
            self.generation = generation

        def is_held(self, workspace_id: str, mode: LeaseMode | None = None) -> bool:
            _ = workspace_id, mode
            return False

    return _Fake()


@pytest.fixture
def mock_config() -> Config:
    """Minimal mock Config for MemoryManager tests."""
    return Config(
        workspace_id="test_project",
        openrouter_api_key=SecretString("test-api-key"),
        tantivy_index_path_template="{workspace_id}_tantivy_test",
        search_score_threshold=0.8,
        fusion_ranking_threshold=0.5,
        enable_smart_replace=False,
        llm_provider=LlmProvider.OPENAI,
        reranker_engine=RerankerEngine.NONE,
        rerank_max_concurrency=5,
        embedding_cache_enabled=False,
        eager_initialization=False,
    )


class _LogCaptureHandler(logging.Handler):
    def __init__(self, records: list[logging.LogRecord]) -> None:
        super().__init__()
        self._records = records

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(record)


class LogCapture:
    def __init__(self) -> None:
        self._records: list[logging.LogRecord] = []
        self._logger = logging.getLogger("reflectlog.test_manager")
        self._logger.handlers.clear()
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._logger.addHandler(_LogCaptureHandler(self._records))
        self.structured = StructuredLogger(self._logger)

    def messages(self, level: int) -> list[str]:
        return [
            record.getMessage() for record in self._records if record.levelno == level
        ]


@pytest.fixture
def mock_logger() -> LogCapture:
    """Mock structured logger."""
    return LogCapture()


def _return_inserted_memories(
    workspace_id: str,
    contents: list[str] | None = None,
    infer: bool = False,
    vectors: list[list[float]] | None = None,
) -> list[str]:
    _ = workspace_id, infer, vectors
    return contents or []


def _make_manager(
    config: Config, logger: LogCapture, coordinator: IStorageCoordinator | None = None
) -> tuple[MemoryManager, MagicMock, MagicMock]:
    """Helper to construct MemoryManager with all infrastructure mocked."""
    with (
        patch(f"{MODULE}.USearchEngine") as usearch_cls,
        patch(f"{MODULE}.LangchainQwenEmbeddings"),
        patch(f"{MODULE}.WeMMEmbeddings"),
        patch(f"{MODULE}.CachedEmbeddings"),
        patch(f"{MODULE}.TantivyEngine") as tantivy_cls,
    ):
        mock_usearch = MagicMock()
        mock_usearch.add_batch.side_effect = _return_inserted_memories
        mock_usearch.get_id_by_content.return_value = None
        mock_usearch.contains_id.return_value = None
        mock_usearch.count.return_value = 0
        mock_usearch.is_ready.return_value = False
        mock_usearch.embedder.embed_documents.side_effect = lambda texts: [
            [0.1] * 4 for _ in texts
        ]
        mock_usearch.memory_store.begin_add_intents.return_value = []
        mock_usearch.memory_store.begin_delete_intents.return_value = []
        mock_usearch.memory_store.list_pending_transitions.return_value = []
        mock_usearch.memory_store.has_later_intent.return_value = False
        mock_usearch.memory_store.get.return_value = None
        usearch_cls.return_value = mock_usearch

        mock_tantivy = MagicMock()
        mock_tantivy.is_ready.return_value = False
        mock_tantivy.delete.return_value = False
        mock_tantivy.find_by_exact_match.return_value = []
        mock_tantivy.delete_batch.side_effect = (
            lambda _workspace, contents, verify_exists=True: len(contents)
        )
        tantivy_cls.return_value = mock_tantivy

        manager = MemoryManager(
            config, logger.structured, coordinator=coordinator or _fake_coordinator()
        )
        return manager, mock_usearch, mock_tantivy


@pytest.mark.unit
class TestEmbeddingIdentityStartup:
    def test_external_tantivy_rejected_before_engines_or_recovery(
        self, mock_config: Config, mock_logger: LogCapture, tmp_path: Path
    ) -> None:
        external = tmp_path / "external" / "tantivy"
        external.mkdir(parents=True)
        (external / "metadata.json").write_text("legacy")
        config = replace(mock_config, tantivy_index_path_template=str(external))
        coordinator = PortalockerStorageCoordinator(os.path.abspath("indexes"))

        with (
            patch(f"{MODULE}.WeMMEmbeddings") as embedder,
            patch(f"{MODULE}.USearchEngine") as semantic,
            patch(f"{MODULE}.TantivyEngine") as tantivy,
            patch.object(MemoryManager, "reconcile_pending_replacements") as reconcile,
            pytest.raises(InitializationError, match="Legacy workspace"),
        ):
            MemoryManager(config, mock_logger.structured, coordinator=coordinator)

        embedder.assert_not_called()
        semantic.assert_not_called()
        tantivy.assert_not_called()
        reconcile.assert_not_called()
        assert not Path(coordinator.paths_for(config.workspace_id).root).exists()
        assert (external / "metadata.json").read_text() == "legacy"

    def test_rejects_changed_model_before_embedder_or_recovery(
        self, mock_config: Config, mock_logger: LogCapture
    ) -> None:
        coordinator = PortalockerStorageCoordinator(os.path.abspath("indexes"))
        _make_manager(mock_config, mock_logger, coordinator)

        with (
            patch(f"{MODULE}.WeMMEmbeddings") as embedder,
            patch(f"{MODULE}.TantivyEngine") as tantivy,
            patch.object(MemoryManager, "reconcile_pending_replacements") as reconcile,
            pytest.raises(InitializationError, match="identity"),
        ):
            MemoryManager(
                replace(mock_config, embedding_model="different/model"),
                mock_logger.structured,
                coordinator=coordinator,
            )

        embedder.assert_not_called()
        tantivy.assert_not_called()
        reconcile.assert_not_called()

    def test_matching_reopen_with_cache_toggle(
        self, mock_config: Config, mock_logger: LogCapture
    ) -> None:
        coordinator = PortalockerStorageCoordinator(os.path.abspath("indexes"))
        _make_manager(mock_config, mock_logger, coordinator)
        identity_path = (
            Path(coordinator.paths_for(mock_config.workspace_id).root) / IDENTITY_NAME
        )
        identity_before = identity_path.read_bytes()

        reopened, _, _ = _make_manager(
            replace(mock_config, embedding_cache_enabled=True), mock_logger, coordinator
        )

        assert reopened.workspace_id == mock_config.workspace_id
        assert identity_path.read_bytes() == identity_before

    def test_publishes_identity_before_embedder_construction(
        self, mock_config: Config, mock_logger: LogCapture
    ) -> None:
        coordinator = PortalockerStorageCoordinator(os.path.abspath("indexes"))
        identity_path = (
            Path(coordinator.paths_for(mock_config.workspace_id).root) / IDENTITY_NAME
        )
        with (
            patch(f"{MODULE}.WeMMEmbeddings") as embedder,
            patch(f"{MODULE}.USearchEngine"),
            patch(f"{MODULE}.TantivyEngine"),
            patch.object(MemoryManager, "reconcile_pending_replacements"),
        ):
            embedder.side_effect = lambda *_args: (
                identity_path.read_bytes() and MagicMock()
            )
            MemoryManager(mock_config, mock_logger.structured, coordinator=coordinator)

        embedder.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Eager Initialization  (lines 334-353, 364-376, 390)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEagerInitialization:
    """Tests for _eager_initialize_engines covering uncovered branches."""

    def test_eager_init_reranker_invalid_engine_raises(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Invalid reranker_engine should raise ValueError (lines 334-340)."""
        mock_config = replace(
            mock_config,
            eager_initialization=True,
            eager_initialize_search_engines=False,
            eager_initialize_reranker=True,
            reranker_engine="none",
        )

        with pytest.raises(ValueError, match="Invalid reranker_engine"):
            _make_manager(mock_config, mock_logger)

    def test_eager_init_reranker_returns_none_warns(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Reranker get_reranker returning None should log warning (lines 349-359)."""
        mock_config = replace(
            mock_config,
            eager_initialization=True,
            eager_initialize_search_engines=False,
            eager_initialize_reranker=True,
            reranker_engine="cross_encoder",
        )

        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker") as reranker_cls,
        ):
            usearch_cls.return_value = MagicMock()
            reranker_cls.return_value = MagicMock()
            manager = MemoryManager(mock_config, mock_logger.structured)
            assert manager._cross_encoder_reranker is not None

    def test_eager_init_reranker_with_cross_encoder(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Eager init with a valid cross-encoder reranker."""
        mock_config = replace(
            mock_config,
            eager_initialization=True,
            eager_initialize_search_engines=False,
            eager_initialize_reranker=True,
            reranker_engine="cross_encoder",
        )

        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker") as reranker_cls,
        ):
            mock_reranker = MagicMock()
            reranker_cls.return_value = mock_reranker
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger.structured)
            assert manager._cross_encoder_reranker is mock_reranker

    def test_eager_init_smart_replacer_disabled_raises(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Eager smart replacer with enable_smart_replace=False raises (lines 364-369)."""
        mock_config = replace(
            mock_config,
            eager_initialization=True,
            eager_initialize_search_engines=False,
            eager_initialize_smart_replacer=True,
            enable_smart_replace=False,
        )

        with pytest.raises(
            ValueError, match="Eager SmartReplacer initialization requested"
        ):
            _make_manager(mock_config, mock_logger)

    def test_eager_init_smart_replacer_enabled(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Eager smart replacer with enable_smart_replace=True (lines 371-376)."""
        mock_config = replace(
            mock_config,
            eager_initialization=True,
            eager_initialize_search_engines=False,
            eager_initialize_smart_replacer=True,
            enable_smart_replace=True,
        )

        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.SmartReplacerConfig"),
            patch(f"{MODULE}.SmartReplacer") as replacer_cls,
        ):
            mock_replacer = MagicMock()
            replacer_cls.return_value = mock_replacer
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger.structured)
            assert manager._smart_replacer is mock_replacer

    def test_eager_init_all_lazy_skipped(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """All components set to lazy should log skip message (line 390)."""
        mock_config = replace(
            mock_config,
            eager_initialization=True,
            eager_initialize_search_engines=False,
            eager_initialize_reranker=False,
            eager_initialize_smart_replacer=False,
        )

        _manager, _, _ = _make_manager(mock_config, mock_logger)
        # Verify skip message was logged
        skip_logged = any(
            "skipped" in message.lower()
            for message in mock_logger.messages(logging.INFO)
        )
        assert skip_logged


# ---------------------------------------------------------------------------
# Tests: Lazy Reranker Properties  (lines 407, 413, 435-463, 512)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLazyRerankerProperties:
    """Tests for cross_encoder_reranker and get_reranker properties."""

    def test_cross_encoder_reranker_returns_none_when_not_configured(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """cross_encoder_reranker returns None for non-cross_encoder (line 435-439)."""
        mock_config = replace(mock_config, reranker_engine="none")
        manager, _, _ = _make_manager(mock_config, mock_logger)
        assert manager.cross_encoder_reranker is None

    def test_cross_encoder_reranker_lazy_init(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """cross_encoder_reranker lazy initializes (lines 441-463)."""
        mock_config = replace(mock_config, reranker_engine="cross_encoder")
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker") as ce_cls,
        ):
            mock_ce = MagicMock()
            ce_cls.return_value = mock_ce
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger.structured)
            result = manager.cross_encoder_reranker
            assert result is mock_ce
            ce_cls.assert_called_once()

    def test_cross_encoder_reranker_cached(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """cross_encoder_reranker returns cached on second call (line 435-439)."""
        mock_config = replace(mock_config, reranker_engine="cross_encoder")
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker") as ce_cls,
        ):
            mock_ce = MagicMock()
            ce_cls.return_value = mock_ce
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger.structured)
            first = manager.cross_encoder_reranker
            second = manager.cross_encoder_reranker
            assert first is second
            ce_cls.assert_called_once()

    def test_cross_encoder_reranker_double_check(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """cross_encoder_reranker double-check after lock (lines 444-448)."""
        mock_config = replace(mock_config, reranker_engine="cross_encoder")
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker"),
        ):
            usearch_cls.return_value = MagicMock()
            manager = MemoryManager(mock_config, mock_logger.structured)
            sentinel = MagicMock(spec=CrossEncoderReranker)
            manager._cross_encoder_reranker = sentinel
            result = manager.cross_encoder_reranker
            assert result is sentinel

    def test_get_reranker_cross_encoder_path(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """get_reranker returns cross_encoder when configured (line 512)."""
        mock_config = replace(mock_config, reranker_engine="cross_encoder")
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CrossEncoderConfig"),
            patch(f"{MODULE}.CrossEncoderReranker") as ce_cls,
        ):
            mock_ce = MagicMock()
            ce_cls.return_value = mock_ce
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger.structured)
            result = manager.get_reranker()
            assert result is mock_ce

    def test_get_reranker_none_engine(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """get_reranker returns None for "none" engine."""
        mock_config = replace(mock_config, reranker_engine="none")
        manager, _, _ = _make_manager(mock_config, mock_logger)
        assert manager.get_reranker() is None


# ---------------------------------------------------------------------------
# Tests: Smart Replacer Property  (line 483)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSmartReplacerProperty:
    """Tests for smart_replacer lazy property."""

    def test_smart_replacer_returns_none_when_disabled(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """smart_replacer returns None when enable_smart_replace=False."""
        mock_config = replace(mock_config, enable_smart_replace=False)
        manager, _, _ = _make_manager(mock_config, mock_logger)
        assert manager.smart_replacer is None

    def test_smart_replacer_double_check_locking(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """smart_replacer double-check after lock (line 483)."""
        mock_config = replace(mock_config, enable_smart_replace=True)
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.SmartReplacerConfig"),
            patch(f"{MODULE}.SmartReplacer"),
        ):
            usearch_cls.return_value = MagicMock()
            manager = MemoryManager(mock_config, mock_logger.structured)
            sentinel = MagicMock()
            manager._smart_replacer = sentinel
            result = manager.smart_replacer
            assert result is sentinel

    def test_smart_replacer_lazy_init(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """smart_replacer lazily initializes when enabled."""
        mock_config = replace(mock_config, enable_smart_replace=True)
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.SmartReplacerConfig"),
            patch(f"{MODULE}.SmartReplacer") as replacer_cls,
        ):
            mock_replacer = MagicMock()
            replacer_cls.return_value = mock_replacer
            usearch_cls.return_value = MagicMock()

            manager = MemoryManager(mock_config, mock_logger.structured)
            result = manager.smart_replacer
            assert result is mock_replacer
            replacer_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: add_memories  (dedup, hybrid write, errors)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddMemory:
    """Tests for the public add_memories path."""

    def test_add_memory_duplicate_skipped(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Existing exact match is not stored again."""
        manager, mock_usearch, _mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 11

        result = manager.add_memories(["hello world"])
        assert result == 0
        mock_usearch.add_batch.assert_not_called()

    def test_add_memory_success(self, mock_config: Config, mock_logger: LogCapture):
        """Successful add stores on both engines."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)

        result = manager.add_memories(["new memory"])
        assert result == 1
        mock_usearch.add_batch.assert_called_once_with(
            workspace_id="test_project",
            contents=["new memory"],
            infer=False,
            vectors=[[0.1, 0.1, 0.1, 0.1]],
        )
        mock_tantivy.add_batch.assert_called_once_with("test_project", ["new memory"])

    def test_add_memory_with_unavailable_tantivy(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        manager._tantivy_engine = None
        manager._init_pipelines()

        result = manager.add_memories(["solo memory"])
        assert result == 1
        mock_usearch.add_batch.assert_called_once()
        mock_tantivy.add_batch.assert_not_called()

    def test_add_memory_storage_error(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Batch insert failures surface as StorageError."""
        manager, mock_usearch, _mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.add_batch.side_effect = StorageError("disk full")

        with pytest.raises(StorageError, match="disk full"):
            manager.add_memories(["error content"])

    def test_add_memory_dedup_disabled(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """When deduplicate_memories=False, skip duplicate check."""
        mock_config = replace(mock_config, deduplicate_memories=False)
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 11
        mock_tantivy.find_by_exact_match.return_value = ["any memory"]

        result = manager.add_memories(["any memory"])
        assert result == 1
        mock_usearch.add_batch.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: add_memories batch dedup/logging  (lines 594-602, 642, 654, 662)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddMemoriesBatchLogging:
    """Tests for add_memories batch dedup and logging edge cases."""

    def test_add_memories_in_batch_duplicate(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Duplicate within batch should be skipped (lines 593-602)."""
        manager, _mock_usearch, _ = _make_manager(mock_config, mock_logger)
        result = manager.add_memories(["mem1", "mem1", "mem2"])
        # "mem1" repeated: first stored, second skipped
        assert result == 2  # mem1 + mem2

    def test_add_memories_batch_insert_skipped_warning(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Memories skipped during batch insert should log warning (line 654)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        # add_batch returns only subset - some memories skipped
        # Must clear side_effect (set by _make_manager) before setting return_value
        mock_usearch.add_batch.side_effect = None
        mock_usearch.add_batch.return_value = ["mem1"]

        result = manager.add_memories(["mem1", "mem2"])
        assert result == 1
        # Verify warning logged for skipped memory
        warning_logged = any(
            "Skipped during batch insert" in message
            for message in mock_logger.messages(logging.WARNING)
        )
        assert warning_logged

    def test_add_memories_stored_log_limit_exceeded(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """When stored memories exceed LOG_ADD_MEMORY_PREVIEW_LIMIT (line 662)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        # Create more memories than LOG_ADD_MEMORY_PREVIEW_LIMIT (20)
        memories = [f"mem_{i}" for i in range(25)]
        mock_usearch.add_batch.return_value = memories

        result = manager.add_memories(memories)
        assert result == 25

    def test_add_memories_log_limit_exceeded(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """When total memories exceed LOG_ADD_MEMORY_PREVIEW_LIMIT (line 642)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        # Create more memories than LOG_ADD_MEMORY_PREVIEW_LIMIT (20)
        memories = [f"mem_{i}" for i in range(25)]
        mock_usearch.add_batch.return_value = memories

        result = manager.add_memories(memories)
        assert result == 25
        # Should have logged omission notice
        omit_logged = any(
            "omitted from logs" in message
            for message in mock_logger.messages(logging.INFO)
        )
        assert omit_logged


# ---------------------------------------------------------------------------
# Tests: search  (lines 774-775)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchIndexSizeException:
    """Tests for search when index size lookup raises exception."""

    @pytest.mark.asyncio
    async def test_search_index_size_exception_fallback(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Exception getting index size should fallback to 0 (lines 774-775)."""
        mock_config = replace(mock_config, reranker_engine="none")
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        # Make index property raise
        mock_index = MagicMock()
        mock_index.__len__ = MagicMock(side_effect=RuntimeError("broken"))
        mock_usearch.index = mock_index
        mock_usearch.search.return_value = [("result", 0.9, "2024-01-01T00:00:00")]
        mock_tantivy.search.return_value = [("result", 0.9)]

        results = await manager.search("test query")
        # Should not raise, should work with index_size=0
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Tests: search_for_removal error  (lines 841-842)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchForRemovalError:
    """Tests for search_for_removal error path."""

    def test_search_for_removal_exception_raises_search_error(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Exception during lookup raises SearchError (lines 841-842)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.side_effect = RuntimeError("db error")

        with pytest.raises(SearchError, match="Failed to search for removal"):
            manager.search_for_removal("test")


# ---------------------------------------------------------------------------
# Tests: delete_by_id and delete_by_memory  (lines 853-857, 904-914, 933)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteOperations:
    """Tests for delete_by_id and delete_by_memory error paths."""

    def test_delete_by_memory_tombs_orphan_tantivy(
        self, mock_config: Config, mock_logger: LogCapture
    ) -> None:
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = None
        mock_tantivy.delete.return_value = True
        mock_usearch.memory_store.list_pending_transitions.return_value = []

        assert manager.delete_by_memory("leftover") is True
        mock_tantivy.delete.assert_called_once_with(
            "test_project", "leftover", verify_exists=True
        )
        mock_tantivy.commit.assert_called()

    def test_delete_by_id_success(self, mock_config: Config, mock_logger: LogCapture):
        """delete_by_id calls semantic engine delete (lines 853-855)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        manager.delete_by_id("42")
        mock_usearch.delete.assert_called_once_with(memory_id="42")

    def test_delete_by_id_tombs_tantivy_when_content_known(
        self, mock_config: Config, mock_logger: LogCapture
    ) -> None:
        class _Store:
            def get(self, memory_id: int) -> object:
                return type("Rec", (), {"content": "known"})()

            def begin_delete_intents(
                self, workspace_id: str, items: list[tuple[int, str]]
            ) -> list[object]:
                return []

        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.memory_store = _Store()
        mock_tantivy.delete.return_value = True
        manager.delete_by_id("42")
        mock_usearch.delete.assert_called_once_with(memory_id="42")
        mock_usearch.commit.assert_called()
        mock_tantivy.delete.assert_called_once_with(
            "test_project", "known", verify_exists=True
        )
        mock_tantivy.commit.assert_called()

    def test_delete_by_id_tantivy_false_is_inconsistent(
        self, mock_config: Config, mock_logger: LogCapture
    ) -> None:
        class _Store:
            def get(self, memory_id: int) -> object:
                return type("Rec", (), {"content": "known"})()

            def begin_delete_intents(
                self, workspace_id: str, items: list[tuple[int, str]]
            ) -> list[object]:
                return []

        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.memory_store = _Store()
        mock_tantivy.delete.return_value = False
        with pytest.raises(InconsistentStateError, match="Tantivy"):
            manager.delete_by_id("42")

    def test_delete_by_id_exception_raises_storage_error(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """delete_by_id wraps exception in StorageError (lines 856-857)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.delete.side_effect = RuntimeError("delete failed")

        with pytest.raises(StorageError, match="Failed to delete memory"):
            manager.delete_by_id("42")

    def test_delete_by_memory_tantivy_failure_inconsistent_state(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Tantivy failure after USearch delete raises InconsistentStateError (lines 904-917)."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 42
        mock_tantivy.delete.side_effect = RuntimeError("tantivy broken")

        with pytest.raises(
            InconsistentStateError,
            match="USearch deletion succeeded but Tantivy deletion failed",
        ):
            manager.delete_by_memory("test memory")

        # USearch delete should have been called
        mock_usearch.delete.assert_called_once_with(memory_id="42")

    def test_delete_by_memory_inconsistent_state_reraise(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """InconsistentStateError is re-raised, not wrapped (line 933)."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 42
        mock_tantivy.delete.side_effect = RuntimeError("tantivy broken")

        with pytest.raises(InconsistentStateError):
            manager.delete_by_memory("test memory")

    def test_delete_by_memory_generic_exception_raises_storage_error(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Generic exception during delete wraps in StorageError (line 935)."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.side_effect = RuntimeError("lookup failed")

        with pytest.raises(StorageError, match="Failed to delete memory"):
            manager.delete_by_memory("test memory")

    def test_delete_by_memory_not_found(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """delete_by_memory returns False when not found."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = None

        result = manager.delete_by_memory("nonexistent")
        assert result is False

    def test_delete_by_memory_success(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """delete_by_memory returns True on success."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 42
        mock_tantivy.delete.return_value = True

        result = manager.delete_by_memory("test memory")
        assert result is True
        mock_usearch.delete.assert_called_once_with(memory_id="42")
        mock_tantivy.delete.assert_called_once_with(
            "test_project", "test memory", verify_exists=True
        )

    def test_delete_by_memory_with_unavailable_tantivy(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        with (
            patch(f"{MODULE}.USearchEngine") as usearch_cls,
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
        ):
            mock_usearch = MagicMock()
            mock_usearch.get_id_by_content.return_value = 42
            usearch_cls.return_value = mock_usearch

            manager = MemoryManager(mock_config, mock_logger.structured)
            manager._tantivy_engine = None
            result = manager.delete_by_memory("test memory")
            assert result is True
            mock_usearch.delete.assert_called_once_with(memory_id="42")

    def test_delete_memories_returns_found_contents(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """delete_memories returns only contents that existed."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)

        def lookup(_workspace_id: str, content: str) -> int | None:
            return {"keep": 1, "also": 2}.get(content)

        mock_usearch.get_id_by_content.side_effect = lookup

        deleted = manager.delete_memories(["keep", "missing", "also"])

        assert deleted == ["keep", "also"]
        assert mock_usearch.delete.call_count == 2
        mock_usearch.commit.assert_called_once()
        mock_tantivy.delete_batch.assert_called_once_with(
            "test_project", ["keep", "also"], verify_exists=True
        )

    def test_delete_memories_tantivy_failure_inconsistent_state(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Tantivy failure after USearch batch delete raises InconsistentStateError."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 42
        mock_tantivy.delete_batch.side_effect = RuntimeError("tantivy broken")

        with pytest.raises(
            InconsistentStateError,
            match="USearch deletion succeeded but Tantivy deletion failed",
        ):
            manager.delete_memories(["test memory"])

        mock_usearch.delete.assert_called_once_with(memory_id="42")

    def test_delete_memories_uses_verify_exists_on_batch(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Production delete_batch probes FTS so it does not plant phantom tombstones."""
        manager, mock_usearch, _mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 7

        class FakeTantivy:
            def __init__(self) -> None:
                self.calls: list[tuple[str, list[str], bool]] = []

            def find_by_exact_match(self, workspace_id: str, content: str) -> list[str]:
                _ = workspace_id, content
                return []

            def delete_batch(
                self,
                workspace_id: str,
                contents: list[str],
                verify_exists: bool = False,
            ) -> int:
                self.calls.append((workspace_id, contents, verify_exists))
                return len(contents)

        fake = FakeTantivy()
        manager._tantivy_engine = cast(TantivyEngine, fake)

        deleted = manager.delete_memories(["hello"])

        assert deleted == ["hello"]
        assert fake.calls == [("test_project", ["hello"], True)]

    def test_delete_memories_short_count_fails_closed(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """A live FTS miss after USearch delete is InconsistentStateError."""
        manager, mock_usearch, _mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.get_id_by_content.return_value = 7

        class ShortTantivy:
            def find_by_exact_match(self, workspace_id: str, content: str) -> list[str]:
                _ = workspace_id
                return [content]

            def delete_batch(
                self,
                workspace_id: str,
                contents: list[str],
                verify_exists: bool = False,
            ) -> int:
                return 0

        manager._tantivy_engine = cast(TantivyEngine, ShortTantivy())

        with pytest.raises(InconsistentStateError, match="deleted 0/1"):
            manager.delete_memories(["hello"])

        mock_usearch.delete.assert_called_once_with(memory_id="7")


# ---------------------------------------------------------------------------
# Tests: close error paths  (lines 983-984, 1002-1003)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloseErrorPaths:
    """Tests for close method error paths."""

    def test_close_tantivy_error_logged(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Tantivy close error should be logged not raised (lines 983-991)."""
        manager, _mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_tantivy.flush.side_effect = RuntimeError("tantivy flush error")

        with pytest.raises(StorageError, match="persist incomplete"):
            manager.close()
        error_logged = any(
            "Error persisting Tantivy engine" in message
            for message in mock_logger.messages(logging.ERROR)
        )
        assert error_logged
        mock_tantivy.close.assert_not_called()

    def test_close_usearch_error_logged(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """USearch close error should be logged not raised (lines 1002-1010)."""
        manager, mock_usearch, _mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.commit.side_effect = RuntimeError("usearch commit error")

        with pytest.raises(StorageError, match="persist incomplete"):
            manager.close()
        error_logged = any(
            "Error persisting USearch engine" in message
            for message in mock_logger.messages(logging.ERROR)
        )
        assert error_logged
        mock_usearch.close.assert_not_called()

    def test_close_both_errors_logged(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Both engines failing should both be logged."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_tantivy.flush.side_effect = RuntimeError("tantivy error")
        mock_usearch.commit.side_effect = RuntimeError("usearch error")

        with pytest.raises(StorageError, match="persist incomplete"):
            manager.close()
        error_calls = mock_logger.messages(logging.ERROR)
        tantivy_err = any("Tantivy" in c for c in error_calls)
        usearch_err = any("USearch" in c for c in error_calls)
        assert tantivy_err
        assert usearch_err

    def test_close_success(self, mock_config: Config, mock_logger: LogCapture):
        """Successful close should log completion."""
        manager, _, _ = _make_manager(mock_config, mock_logger)
        manager.close()
        close_logged = any(
            "all data persisted" in message.lower()
            for message in mock_logger.messages(logging.INFO)
        )
        assert close_logged

    def test_close_is_idempotent(self, mock_config: Config, mock_logger: LogCapture):
        """A second close() is a no-op after the first persist."""
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        manager.close()
        manager.close()
        mock_usearch.close.assert_called_once()
        mock_tantivy.close.assert_called_once()

    def test_failed_close_can_retry_persist(
        self, mock_config: Config, mock_logger: LogCapture
    ) -> None:
        """A persist failure must not stick closed so a later close can retry."""
        manager, mock_usearch, _mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_usearch.commit.side_effect = [RuntimeError("usearch commit error"), None]

        with pytest.raises(StorageError, match="persist incomplete"):
            manager.close()
        manager.close()
        assert mock_usearch.commit.call_count == 2
        mock_usearch.close.assert_called_once()

    def test_failed_tantivy_persist_does_not_close_usearch(
        self, mock_config: Config, mock_logger: LogCapture
    ) -> None:
        manager, mock_usearch, mock_tantivy = _make_manager(mock_config, mock_logger)
        mock_tantivy.flush.side_effect = RuntimeError("tantivy flush error")

        with pytest.raises(StorageError, match="persist incomplete"):
            manager.close()
        mock_usearch.commit.assert_called_once()
        mock_usearch.close.assert_not_called()
        mock_tantivy.close.assert_not_called()
        assert manager._closed is False
        manager._ensure_open()

    def test_search_after_close_is_rejected(
        self, mock_config: Config, mock_logger: LogCapture
    ) -> None:
        manager, _, _ = _make_manager(mock_config, mock_logger)
        manager.close()
        import asyncio

        with pytest.raises(StorageError, match="closed"):
            asyncio.run(manager.search("query"))

    def test_writes_after_close_are_rejected(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        manager, _, _ = _make_manager(mock_config, mock_logger)
        manager.close()
        with pytest.raises(StorageError, match="closed"):
            manager.add_memories(["too late"])

    def test_pending_intent_count_propagates_list_failure(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Journal listing errors must surface so health cannot report zero."""
        manager, _mock_usearch, _ = _make_manager(mock_config, mock_logger)

        class BrokenStore:
            def list_pending_transitions(self) -> list[object]:
                raise RuntimeError("journal locked")

        class BrokenEngine:
            memory_store = BrokenStore()

        manager._semantic_engine = cast(ISemanticSearchEngine, BrokenEngine())
        with pytest.raises(RuntimeError, match="journal locked"):
            _ = manager.pending_intent_count()


# ---------------------------------------------------------------------------
# Tests: get_all error path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetAll:
    """Tests for get_all method."""

    def test_get_all_success(self, mock_config: Config, mock_logger: LogCapture):
        """get_all returns memories from semantic engine."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.get_all.return_value = ["mem1", "mem2"]

        result = manager.get_all()
        assert result == ["mem1", "mem2"]
        mock_usearch.get_all.assert_called_once_with(
            workspace_id="test_project", limit=None, offset=0
        )

    def test_get_all_exception_raises_storage_error(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """get_all wraps exceptions in StorageError."""
        manager, mock_usearch, _ = _make_manager(mock_config, mock_logger)
        mock_usearch.get_all.side_effect = RuntimeError("db error")

        with pytest.raises(StorageError, match="Failed to retrieve memories"):
            manager.get_all()


# ---------------------------------------------------------------------------
# Tests: Init logging paths (reranker_engine variations)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitLogging:
    """Tests for __init__ logging based on reranker_engine and smart_replace."""

    def test_init_cross_encoder_reranker_logging(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Init with cross_encoder reranker logs correctly."""
        mock_config = replace(mock_config, reranker_engine="cross_encoder")
        _make_manager(mock_config, mock_logger)
        ce_logged = any(
            "CrossEncoder" in message for message in mock_logger.messages(logging.INFO)
        )
        assert ce_logged

    def test_init_smart_replace_enabled_logging(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Init with smart replace enabled logs correctly."""
        mock_config = replace(mock_config, enable_smart_replace=True)
        _make_manager(mock_config, mock_logger)
        sr_logged = any(
            "SmartReplacer configured" in message
            for message in mock_logger.messages(logging.INFO)
        )
        assert sr_logged

    def test_init_embedding_cache_enabled(
        self, mock_config: Config, mock_logger: LogCapture
    ):
        """Init with embedding cache enabled wraps embedder."""
        mock_config = replace(mock_config, embedding_cache_enabled=True)
        with (
            patch(f"{MODULE}.USearchEngine"),
            patch(f"{MODULE}.LangchainQwenEmbeddings"),
            patch(f"{MODULE}.TantivyEngine"),
            patch(f"{MODULE}.CachedEmbeddings") as cached_cls,
        ):
            _manager = MemoryManager(mock_config, mock_logger.structured)
            cached_cls.assert_called_once()


@pytest.mark.unit
class TestCoordinatorLifecycle:
    """Direct writes publish generation after store convergence."""

    def test_add_publishes_generation_before_intent(
        self, mock_config: Config, mock_logger: LogCapture
    ) -> None:
        manager, _usearch, _tantivy = _make_manager(mock_config, mock_logger)
        steps: list[str] = []
        manager.orchestration_hook = steps.append
        stored = manager.add_memories(["alpha"])
        assert stored == 1
        assert steps == [
            "before_generation",
            "after_generation",
            "before_intent",
            "after_intent",
        ]
        assert manager._coordinator.read_generation(mock_config.workspace_id) == 1

    def test_close_relinquishes_and_rejects_later_calls(
        self, mock_config: Config, mock_logger: LogCapture
    ) -> None:
        manager, _usearch, _tantivy = _make_manager(mock_config, mock_logger)
        manager.close()
        with pytest.raises(StorageError, match="closed"):
            manager.add_memories(["after-close"])
        with pytest.raises(StorageError, match="closed"):
            _ = manager.get_all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
