"""Canonical SearchPipeline tests: behavior plus event-loop responsiveness.

These tests target ``search_strategies.SearchPipeline`` — the single
production search path used by MemoryManager.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import threading
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.memory.search_strategies import (
    SearchContext,
    SearchPipeline,
    calculate_adaptive_overfetch,
)
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.exceptions import InitializationError, SearchError
from reflectlog.core.logging import IStructuredLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BACKEND_WAIT_TIMEOUT = 1.0
_TS = "2025-01-01T00:00:00Z"


def _make_config(
    *,
    fusion_ranking_threshold: float = 0.1,
    reranker_engine: str = "none",
) -> Mock:
    config = Mock(spec=Config)
    config.workspace_id = "test_project"
    config.fusion_ranking_threshold = fusion_ranking_threshold
    config.fusion_rrf_k = 60
    config.reranker_engine = reranker_engine
    config.search_score_threshold = 0.0
    config.cross_encoder_top_k = 20
    config.cross_encoder_model = "BAAI/bge-reranker-v2-m3"
    config.overfetch_multiplier = 3
    config.overfetch_adaptive = True
    config.overfetch_min_multiplier = 1.5
    config.overfetch_max_multiplier = 3.0
    return config


def _make_logger() -> IStructuredLogger:
    return cast(IStructuredLogger, Mock(spec=StructuredLogger))


def _make_context(
    *,
    query: str = "test query",
    limit: int = 5,
    overfetch_limit: int = 15,
    enable_rrf_fusion: bool = True,
    reranker_engine: str = "none",
    workspace_id: str = "test_project",
) -> SearchContext:
    return SearchContext(
        query=query,
        limit=limit,
        overfetch_limit=overfetch_limit,
        enable_rrf_fusion=enable_rrf_fusion,
        reranker_engine=reranker_engine,
        workspace_id=workspace_id,
    )


def _make_pipeline(
    *,
    semantic: Any,
    tantivy: Any = None,
    fusion: Any = None,
    config: Mock | None = None,
    logger: IStructuredLogger | None = None,
    memory_manager: Any = None,
) -> SearchPipeline:
    if fusion is None:
        fusion_engine = MagicMock()
        fusion_engine.method = "rrf"
        fusion_engine.fuse = MagicMock(return_value=[])
    else:
        fusion_engine = fusion
    if memory_manager is None:
        manager = MagicMock()
        manager.cross_encoder_reranker = None
    else:
        manager = memory_manager
    if isinstance(semantic, MagicMock):
        semantic.is_ready.return_value = False
        exists_many = semantic.memory_store.exists_many
        if (
            isinstance(exists_many, MagicMock)
            and exists_many.side_effect is None
            and isinstance(exists_many.return_value, MagicMock)
        ):
            exists_many.side_effect = lambda _workspace_id, contents: set(contents)
    if isinstance(tantivy, MagicMock):
        tantivy.is_ready.return_value = False
    return SearchPipeline(
        semantic_engine=semantic,
        tantivy_engine=tantivy,
        fusion_engine=fusion_engine,
        config=config or _make_config(),
        logger=logger or _make_logger(),
        memory_manager=manager,
    )


class ControllableBackend:
    """Synchronous backend that can block until the event loop progresses."""

    def __init__(
        self,
        results: list[object],
        *,
        entered: threading.Event,
        loop_progressed: threading.Event,
        barrier: threading.Barrier | None = None,
        thread_ids: list[int] | None = None,
        fail: Exception | None = None,
        init_entered: threading.Event | None = None,
        init_barrier: threading.Barrier | None = None,
        init_thread_ids: list[int] | None = None,
        init_release: threading.Event | None = None,
        block_on_init: bool = False,
        finished: threading.Event | None = None,
    ) -> None:
        self.results = results
        self.entered = entered
        self.loop_progressed = loop_progressed
        self.barrier = barrier
        self.thread_ids = thread_ids
        self.fail = fail
        self.init_entered = init_entered
        self.init_barrier = init_barrier
        self.init_thread_ids = init_thread_ids
        self.init_release = init_release
        self.block_on_init = block_on_init
        self.finished = finished
        self.search_calls: list[tuple[str, str, int]] = []

    @property
    def memory_store(self) -> object:
        return self

    def exists_many(self, workspace_id: str, contents: list[str]) -> set[str]:
        _ = workspace_id
        return set(contents)

    def get_records_by_contents(
        self, workspace_id: str, contents: list[str]
    ) -> list[object]:
        _ = workspace_id
        records: list[object] = []
        for item in self.results:
            if isinstance(item, tuple) and len(item) >= 3 and item[0] in contents:
                records.append(
                    type(
                        "Stored",
                        (),
                        {"content": item[0], "created_at": item[2]},
                    )()
                )
        return records

    def is_ready(self) -> bool:
        return False

    def ensure_initialized(self) -> None:
        if self.init_thread_ids is not None:
            self.init_thread_ids.append(threading.get_ident())
        if self.init_entered is not None:
            self.init_entered.set()
        if self.init_barrier is not None:
            _ = self.init_barrier.wait(timeout=_BACKEND_WAIT_TIMEOUT)
        release = (
            self.init_release
            if self.init_release is not None
            else (self.loop_progressed if self.block_on_init else None)
        )
        if release is not None and not release.wait(timeout=_BACKEND_WAIT_TIMEOUT):
            raise TimeoutError("event loop did not progress while init blocked")

    def search(self, query: str, workspace_id: str, limit: int) -> list[object]:
        self.search_calls.append((query, workspace_id, limit))
        if self.thread_ids is not None:
            self.thread_ids.append(threading.get_ident())
        self.entered.set()
        try:
            if self.barrier is not None:
                _ = self.barrier.wait(timeout=_BACKEND_WAIT_TIMEOUT)
            if not self.loop_progressed.wait(timeout=_BACKEND_WAIT_TIMEOUT):
                raise TimeoutError("event loop did not progress while backend blocked")
            if self.fail is not None:
                raise self.fail
            return list(self.results)
        finally:
            if self.finished is not None:
                self.finished.set()


# ---------------------------------------------------------------------------
# Single canonical pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCanonicalPipelineIdentity:
    """Guard against a second production search implementation."""

    def test_search_pipeline_module_removed(self) -> None:
        """Compatibility re-export module no longer exists."""
        import importlib

        with pytest.raises(ModuleNotFoundError):
            _ = importlib.import_module("reflectlog.application.memory.search_pipeline")
        with pytest.raises(ModuleNotFoundError):
            _ = importlib.import_module("reflectlog.core.search")
        with pytest.raises(ModuleNotFoundError):
            _ = importlib.import_module("reflectlog.infrastructure.search.base")
        with pytest.raises(ModuleNotFoundError):
            _ = importlib.import_module("reflectlog.application.memory.protocols")
        with pytest.raises(ModuleNotFoundError):
            _ = importlib.import_module("reflectlog.application.types")

    def test_deleted_reexport_paths_are_gone(self) -> None:
        """Old barrel imports must fail instead of silently returning."""
        import reflectlog
        import reflectlog.application.tools as tools_pkg
        from reflectlog.utility import utility as utility_mod

        assert not hasattr(reflectlog, "main")
        assert not hasattr(tools_pkg, "SearchTool")
        for name in (
            "AssistantMessage",
            "ClaudeAgentOptions",
            "ResultMessage",
            "TextBlock",
        ):
            assert name not in utility_mod.__all__
            assert not hasattr(utility_mod, name)

    def test_manager_wires_strategies_pipeline(self) -> None:
        """MemoryManager._init_pipelines() uses search_strategies.SearchPipeline."""
        config = MagicMock()
        config.workspace_id = "test_project"
        config.tantivy_index_path_template = "{workspace_id}_tantivy_test"
        config.enable_smart_replace = False
        config.reranker_engine = "none"
        config.embedding_cache_enabled = False
        config.eager_initialization = False
        config.fusion_method = "rrf"
        config.fusion_normalization = None
        config.fusion_rrf_k = 60
        config.openrouter_api_key.get_secret_value.return_value = "test-key"

        with (
            patch("reflectlog.application.memory.manager.USearchEngine"),
            patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"),
            patch("reflectlog.application.memory.manager.TantivyEngine"),
        ):
            manager = MemoryManager(config, _make_logger())

        assert isinstance(manager._search_pipeline, SearchPipeline)
        assert (
            type(manager._search_pipeline).__module__
            == "reflectlog.application.memory.search_strategies"
        )

    def test_search_engine_status_reports_pending_and_ready(self) -> None:
        pending = MagicMock()
        pending.is_ready.return_value = False
        ready = MagicMock()
        ready.is_ready.return_value = True
        config = MagicMock()
        config.workspace_id = "test_project"
        config.tantivy_index_path_template = "{workspace_id}_tantivy_test"
        config.enable_smart_replace = False
        config.reranker_engine = "none"
        config.embedding_cache_enabled = False
        config.eager_initialization = False
        config.fusion_method = "rrf"
        config.fusion_normalization = None
        config.fusion_rrf_k = 60
        config.openrouter_api_key.get_secret_value.return_value = "test-key"

        with (
            patch("reflectlog.application.memory.manager.USearchEngine"),
            patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"),
            patch("reflectlog.application.memory.manager.TantivyEngine"),
        ):
            manager = MemoryManager(config, _make_logger())

        class PendingEngine:
            def is_ready(self) -> bool:
                return False

        class ReadyEngine:
            def is_ready(self) -> bool:
                return True

        manager._semantic_engine = cast(Any, PendingEngine())
        manager._tantivy_engine = None
        assert manager.search_engine_status() == {
            "semantic_engine": "pending",
            "tantivy_engine": "disabled",
        }
        manager._semantic_engine = cast(Any, ReadyEngine())
        manager._tantivy_engine = cast(Any, ReadyEngine())
        assert manager.search_engine_status() == {
            "semantic_engine": "initialized",
            "tantivy_engine": "initialized",
        }

    def test_search_engine_status_treats_is_ready_errors_as_pending(self) -> None:
        broken = MagicMock()
        broken.is_ready.side_effect = RuntimeError("peek failed")
        missing = MagicMock()
        missing.is_ready = None
        config = MagicMock()
        config.workspace_id = "test_project"
        config.tantivy_index_path_template = "{workspace_id}_tantivy_test"
        config.enable_smart_replace = False
        config.reranker_engine = "none"
        config.embedding_cache_enabled = False
        config.eager_initialization = False
        config.fusion_method = "rrf"
        config.fusion_normalization = None
        config.fusion_rrf_k = 60
        config.openrouter_api_key.get_secret_value.return_value = "test-key"

        with (
            patch("reflectlog.application.memory.manager.USearchEngine"),
            patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"),
            patch("reflectlog.application.memory.manager.TantivyEngine"),
        ):
            manager = MemoryManager(config, _make_logger())

        manager._semantic_engine = broken
        manager._tantivy_engine = missing
        assert manager.search_engine_status() == {
            "semantic_engine": "pending",
            "tantivy_engine": "pending",
        }

    def test_logger_required(self) -> None:
        """Construction still requires a logger."""
        with pytest.raises(ValueError, match="logger is required"):
            _ = SearchPipeline(
                semantic_engine=MagicMock(),
                tantivy_engine=None,
                fusion_engine=MagicMock(),
                config=_make_config(),
                logger=None,
                memory_manager=None,
            )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnavailableTantivyBehavior:
    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = []
        pipeline = _make_pipeline(semantic=semantic)

        result = await pipeline.execute(_make_context(enable_rrf_fusion=False))

        assert result.memories == []
        assert result.timestamp_map == {}
        assert result.semantic_results == []
        assert result.tantivy_results == []

    @pytest.mark.asyncio
    async def test_single_result_preserves_order_and_timestamp(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("only", 0.9, _TS)]
        pipeline = _make_pipeline(semantic=semantic)

        result = await pipeline.execute(_make_context(enable_rrf_fusion=False))

        assert result.memories == ["only"]
        assert result.timestamp_map == {"only": _TS}
        semantic.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_results_keep_backend_order(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [
            ("alpha", 0.9, _TS),
            ("beta", 0.7, "2025-02-01T00:00:00Z"),
        ]
        pipeline = _make_pipeline(semantic=semantic)

        result = await pipeline.execute(_make_context(limit=5, enable_rrf_fusion=False))

        assert result.memories == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_engine_failure_raises_search_error(self) -> None:
        semantic = MagicMock()
        semantic.search.side_effect = RuntimeError("engine boom")
        pipeline = _make_pipeline(semantic=semantic)

        with pytest.raises(SearchError, match="Failed to execute search"):
            await pipeline.execute(_make_context())


@pytest.mark.unit
class TestHybridFusionAndFilter:
    """RRF vs concatenation, thresholds, limits — migrated orchestration tests."""

    @pytest.mark.asyncio
    async def test_rrf_calls_fusion_engine_and_respects_limit(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [
            ("hit1", 0.9, _TS),
            ("hit2", 0.8, _TS),
            ("hit3", 0.7, _TS),
        ]
        tantivy = MagicMock()
        tantivy.search.return_value = [("hit2", 0.95), ("hit4", 0.6)]
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [
            ("hit2", 0.5),
            ("hit1", 0.4),
            ("hit4", 0.3),
            ("hit3", 0.2),
        ]
        config = _make_config(fusion_ranking_threshold=0.0)
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=True, limit=2, overfetch_limit=10)
        )

        fusion.fuse.assert_called_once_with(
            [("hit1", 0.9), ("hit2", 0.8), ("hit3", 0.7)],
            [("hit2", 0.95), ("hit4", 0.6)],
        )
        assert result.memories == ["hit2", "hit1"]

    @pytest.mark.asyncio
    async def test_concatenation_semantic_first_not_score_sort(self) -> None:
        """Canonical concat keeps semantic hits first, even if Tantivy scores higher."""
        semantic = MagicMock()
        semantic.search.return_value = [("semantic-low", 0.2, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = [("tantivy-high", 0.99)]
        fusion = MagicMock()
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy, fusion=fusion)

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=False, limit=5, overfetch_limit=10)
        )

        fusion.fuse.assert_not_called()
        assert result.memories == ["semantic-low", "tantivy-high"]

    @pytest.mark.asyncio
    async def test_concatenation_deduplicates_and_respects_limit(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("shared", 0.8, _TS), ("s2", 0.7, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = [("shared", 0.99), ("t2", 0.5)]
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy)

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=False, limit=2, overfetch_limit=2)
        )

        assert result.memories == ["shared", "t2"]

    @pytest.mark.asyncio
    async def test_empty_backends_return_empty(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = []
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = []
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy, fusion=fusion)

        result = await pipeline.execute(_make_context(enable_rrf_fusion=True))

        assert result.memories == []

    @pytest.mark.asyncio
    async def test_fusion_threshold_keeps_scores_at_or_above(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("keep", 0.9, _TS), ("drop", 0.8, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = [("other", 0.2)]
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("keep", 0.5), ("drop", 0.3)]
        config = _make_config(fusion_ranking_threshold=0.5)
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(_make_context(enable_rrf_fusion=True))

        assert result.memories == ["keep"]

    @pytest.mark.asyncio
    async def test_fusion_threshold_filters_all_returns_empty(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("low", 0.9, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = [("other", 0.2)]
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("low", 0.01)]
        config = _make_config(fusion_ranking_threshold=0.02)
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(_make_context(enable_rrf_fusion=True))

        assert result.memories == []
        assert result.semantic_results == [("low", 0.9, _TS)]

    @pytest.mark.asyncio
    async def test_concatenation_skips_threshold_filter(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("low", 0.01, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        config = _make_config(fusion_ranking_threshold=0.9)
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy, config=config)

        result = await pipeline.execute(_make_context(enable_rrf_fusion=False))

        assert result.memories == ["low"]

    @pytest.mark.asyncio
    async def test_single_backend_skips_fusion_threshold(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("cosine-hit", 0.65, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("cosine-hit", 0.65)]
        config = _make_config(fusion_ranking_threshold=0.8)
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(_make_context(enable_rrf_fusion=True))

        assert result.memories == ["cosine-hit"]

    def test_filter_stage_empty_and_zero_threshold(self) -> None:
        pipeline = _make_pipeline(semantic=MagicMock())
        assert pipeline._filter_by_fusion_threshold([], 0.5, "q") == []
        kept = pipeline._filter_by_fusion_threshold([("a", 0.0), ("b", 0.01)], 0.0, "q")
        assert kept == [("a", 0.0), ("b", 0.01)]


@pytest.mark.unit
class TestBackendFailureContracts:
    """Hybrid fallback and exception contracts stay unchanged."""

    @pytest.mark.asyncio
    async def test_semantic_search_failure_falls_back_to_tantivy(self) -> None:
        semantic = MagicMock()
        semantic.search.side_effect = RuntimeError("embed fail")
        tantivy = MagicMock()
        tantivy.search.return_value = [("from-tantivy", 0.8)]
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("from-tantivy", 0.4)]
        config = _make_config(fusion_ranking_threshold=0.0)
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(_make_context(enable_rrf_fusion=True))

        assert result.memories == ["from-tantivy"]
        assert result.semantic_results == []
        fusion.fuse.assert_called_once_with([], [("from-tantivy", 0.8)])

    @pytest.mark.asyncio
    async def test_semantic_init_failure_falls_back_to_tantivy(self) -> None:
        semantic = MagicMock()
        semantic.ensure_initialized.side_effect = RuntimeError("init fail")
        tantivy = MagicMock()
        tantivy.search.return_value = [("from-tantivy", 0.8)]
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("from-tantivy", 0.8)]
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy, fusion=fusion)

        result = await pipeline.execute(_make_context())

        assert result.memories == ["from-tantivy"]
        assert result.semantic_results == []

    @pytest.mark.asyncio
    async def test_tantivy_search_failure_falls_back_to_semantic(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("ok", 0.9, _TS)]
        tantivy = MagicMock()
        tantivy.search.side_effect = RuntimeError("tantivy boom")
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("ok", 0.9)]
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy, fusion=fusion)

        result = await pipeline.execute(_make_context())

        assert result.memories == ["ok"]
        assert result.tantivy_results == []

    @pytest.mark.asyncio
    async def test_both_backend_search_failures_raise_search_error(self) -> None:
        semantic = MagicMock()
        semantic.search.side_effect = RuntimeError("embed fail")
        tantivy = MagicMock()
        tantivy.search.side_effect = RuntimeError("tantivy boom")
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy)

        with pytest.raises(SearchError, match="Failed to execute search"):
            await pipeline.execute(_make_context())

    @pytest.mark.asyncio
    async def test_tantivy_init_failure_falls_back_to_semantic(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("ok", 0.9, _TS)]
        tantivy = MagicMock()
        tantivy.ensure_initialized.side_effect = RuntimeError("tantivy init fail")
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("ok", 0.9)]
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy, fusion=fusion)

        result = await pipeline.execute(_make_context())

        assert result.memories == ["ok"]
        assert result.tantivy_results == []

    @pytest.mark.asyncio
    async def test_semantic_error_and_empty_tantivy_success_raises_search_error(
        self,
    ) -> None:
        cause = RuntimeError("embed fail")
        semantic = MagicMock()
        semantic.search.side_effect = cause
        semantic.memory_store.exists_many.side_effect = lambda *_args: set()
        tantivy = MagicMock()
        tantivy.search.return_value = []
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy)

        with pytest.raises(SearchError, match="Failed to execute search") as exc_info:
            await pipeline.execute(_make_context())
        assert exc_info.value.__cause__ is cause

    @pytest.mark.asyncio
    async def test_semantic_error_and_stale_only_fts_raises_search_error(
        self,
    ) -> None:
        cause = RuntimeError("embed fail")
        semantic = MagicMock()
        semantic.search.side_effect = cause
        semantic.memory_store.exists_many.side_effect = lambda *_args: set()
        tantivy = MagicMock()
        tantivy.search.return_value = [("deleted text", 4.0)]
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy)

        with pytest.raises(SearchError, match="Failed to execute search") as exc_info:
            await pipeline.execute(_make_context())
        assert exc_info.value.__cause__ is cause

    async def test_semantic_error_and_unavailable_tantivy_raises_search_error(
        self,
    ) -> None:
        semantic = MagicMock()
        semantic.search.side_effect = RuntimeError("embed fail")
        pipeline = _make_pipeline(semantic=semantic, tantivy=None)

        with pytest.raises(SearchError, match="Failed to execute search"):
            await pipeline.execute(_make_context())

    async def test_fts_hits_missing_from_sqlite_are_dropped(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = []
        semantic.memory_store.exists_many.side_effect = lambda *_args: set()
        tantivy = MagicMock()
        tantivy.search.return_value = [("deleted text", 4.0)]
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("deleted text", 0.01)]
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy, fusion=fusion)

        result = await pipeline.execute(_make_context())
        assert result.tantivy_results == []
        fusion.fuse.assert_called_once_with([], [])
        assert result.memories == []

    async def test_fts_keeps_live_sqlite_hits_and_drops_deleted(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("keep me", 0.9, _TS)]
        semantic.memory_store.exists_many.side_effect = lambda _workspace_id, contents: {
            item for item in contents if item == "keep me"
        }
        tantivy = MagicMock()
        tantivy.search.return_value = [("keep me", 4.0), ("deleted text", 3.0)]
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("keep me", 0.4), ("deleted text", 0.01)]
        pipeline = _make_pipeline(
            semantic=semantic,
            tantivy=tantivy,
            fusion=fusion,
            config=_make_config(fusion_ranking_threshold=0.0),
        )

        result = await pipeline.execute(_make_context())
        assert result.tantivy_results == [("keep me", 4.0)]
        fusion.fuse.assert_called_once_with([("keep me", 0.9)], [("keep me", 4.0)])
        assert result.memories == ["keep me"]

    async def test_fts_hits_dropped_when_exists_many_is_not_a_string_set(
        self,
    ) -> None:
        semantic = MagicMock()
        semantic.search.return_value = []
        semantic.memory_store.exists_many.side_effect = None
        semantic.memory_store.exists_many.return_value = ["not", "a", "set"]
        tantivy = MagicMock()
        tantivy.search.return_value = [("maybe live", 4.0)]
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("maybe live", 0.01)]
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy, fusion=fusion)

        result = await pipeline.execute(_make_context())
        assert result.tantivy_results == []
        fusion.fuse.assert_called_once_with([], [])
        assert result.memories == []

    async def test_tantivy_error_and_empty_semantic_raises_search_error(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = []
        tantivy = MagicMock()
        tantivy.search.side_effect = RuntimeError("tantivy boom")
        pipeline = _make_pipeline(semantic=semantic, tantivy=tantivy)

        with pytest.raises(SearchError, match="Failed to execute search"):
            await pipeline.execute(_make_context())

    @pytest.mark.asyncio
    async def test_search_score_threshold_drops_weak_semantic_hits(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [
            ("strong", 0.9, _TS),
            ("weak", 0.1, _TS),
        ]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("strong", 0.9)]
        config = _make_config()
        config.search_score_threshold = 0.5
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(_make_context())

        assert result.memories == ["strong"]
        assert all(msg != "weak" for msg, _, _ in result.semantic_results)

    @pytest.mark.asyncio
    async def test_unavailable_tantivy_returns_empty(self) -> None:
        pipeline = _make_pipeline(semantic=MagicMock(), tantivy=None)
        assert await pipeline._search_tantivy("q", 10, "proj") == ([], None)


@pytest.mark.unit
class TestRerankerSettings:
    """Lazy LLM, cross-encoder, none, and skip-on-single-result contracts."""

    @pytest.mark.asyncio
    async def test_none_passthrough(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("a", 0.9, _TS), ("b", 0.8, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("a", 0.4), ("b", 0.3)]
        config = _make_config(fusion_ranking_threshold=0.0, reranker_engine="none")
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=True, reranker_engine="none")
        )

        assert result.memories == ["a", "b"]

    @pytest.mark.asyncio
    async def test_cross_encoder_reranker_applied(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("a", 0.9, _TS), ("b", 0.8, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("a", 0.4), ("b", 0.3)]
        config = _make_config(
            fusion_ranking_threshold=0.0, reranker_engine="cross_encoder"
        )
        encoder = AsyncMock()
        encoder.rerank_async = AsyncMock(return_value=[("b", 0.95), ("a", 0.2)])
        manager = MagicMock()
        manager.cross_encoder_reranker = encoder
        pipeline = _make_pipeline(
            semantic=semantic,
            tantivy=tantivy,
            fusion=fusion,
            config=config,
            memory_manager=manager,
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=True, reranker_engine="cross_encoder")
        )

        encoder.rerank_async.assert_awaited_once_with(
            "test query",
            [("a", 0.4), ("b", 0.3)],
            {"a": _TS, "b": _TS},
            top_k=20,
        )
        assert result.memories == ["b", "a"]

    @pytest.mark.asyncio
    async def test_single_result_skips_reranker(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("only", 0.9, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = []
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("only", 0.4)]
        config = _make_config(
            fusion_ranking_threshold=0.0, reranker_engine="cross_encoder"
        )
        encoder = AsyncMock()
        encoder.rerank_async = AsyncMock(return_value=[("only", 0.99)])
        manager = MagicMock()
        manager.cross_encoder_reranker = encoder
        pipeline = _make_pipeline(
            semantic=semantic,
            tantivy=tantivy,
            fusion=fusion,
            config=config,
            memory_manager=manager,
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=True, reranker_engine="cross_encoder")
        )

        encoder.rerank_async.assert_not_awaited()
        assert result.memories == ["only"]

    @pytest.mark.asyncio
    async def test_empty_after_filter_skips_reranker(self) -> None:
        semantic = MagicMock()
        semantic.search.return_value = [("low", 0.9, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = [("other", 0.2)]
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("low", 0.01)]
        config = _make_config(
            fusion_ranking_threshold=0.02, reranker_engine="cross_encoder"
        )
        encoder = AsyncMock()
        manager = MagicMock()
        manager.cross_encoder_reranker = encoder
        pipeline = _make_pipeline(
            semantic=semantic,
            tantivy=tantivy,
            fusion=fusion,
            config=config,
            memory_manager=manager,
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=True, reranker_engine="cross_encoder")
        )

        encoder.rerank_async.assert_not_awaited()
        assert result.memories == []


# ---------------------------------------------------------------------------
# Responsiveness: event-loop ticker + overlapping hybrid backends
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchResponsiveness:
    """Slow sync fakes must not stall the AnyIO/asyncio event loop."""

    @pytest.mark.asyncio
    async def test_unavailable_tantivy_ticker_advances(self) -> None:
        entered = threading.Event()
        loop_progressed = threading.Event()
        loop_thread = threading.get_ident()
        worker_threads: list[int] = []
        ticks_after_enter = 0

        backend = ControllableBackend(
            [("mem", 0.9, _TS)],
            entered=entered,
            loop_progressed=loop_progressed,
            thread_ids=worker_threads,
        )
        pipeline = _make_pipeline(semantic=backend)

        async def ticker() -> None:
            nonlocal ticks_after_enter
            while True:
                if entered.is_set():
                    ticks_after_enter += 1
                    loop_progressed.set()
                await asyncio.sleep(0)
                if loop_progressed.is_set() and ticks_after_enter >= 3:
                    break

        ticker_task = asyncio.create_task(ticker())
        search_task = asyncio.create_task(
            pipeline.execute(_make_context(enable_rrf_fusion=False))
        )
        result = await asyncio.wait_for(search_task, timeout=5)
        ticker_task.cancel()
        with suppress(asyncio.CancelledError):
            await ticker_task

        assert result.memories == ["mem"]
        assert entered.is_set()
        assert ticks_after_enter >= 1
        assert worker_threads
        assert worker_threads[0] != loop_thread

    @pytest.mark.asyncio
    async def test_hybrid_backends_overlap_and_ticker_advances(self) -> None:
        semantic_entered = threading.Event()
        tantivy_entered = threading.Event()
        loop_progressed = threading.Event()
        barrier = threading.Barrier(2, timeout=_BACKEND_WAIT_TIMEOUT)
        loop_thread = threading.get_ident()
        semantic_threads: list[int] = []
        tantivy_threads: list[int] = []
        ticks_after_both = 0

        semantic = ControllableBackend(
            [("sem", 0.9, _TS)],
            entered=semantic_entered,
            loop_progressed=loop_progressed,
            barrier=barrier,
            thread_ids=semantic_threads,
        )
        tantivy = ControllableBackend(
            [("tan", 0.8)],
            entered=tantivy_entered,
            loop_progressed=loop_progressed,
            barrier=barrier,
            thread_ids=tantivy_threads,
        )
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("sem", 0.4), ("tan", 0.3)]
        config = _make_config(fusion_ranking_threshold=0.0)
        pipeline = _make_pipeline(
            semantic=semantic, tantivy=tantivy, fusion=fusion, config=config
        )

        async def ticker() -> None:
            nonlocal ticks_after_both
            while True:
                if semantic_entered.is_set() and tantivy_entered.is_set():
                    ticks_after_both += 1
                    loop_progressed.set()
                await asyncio.sleep(0)
                if loop_progressed.is_set() and ticks_after_both >= 3:
                    break

        ticker_task = asyncio.create_task(ticker())
        result = await asyncio.wait_for(
            pipeline.execute(_make_context()),
            timeout=5,
        )
        ticker_task.cancel()
        with suppress(asyncio.CancelledError):
            await ticker_task

        assert result.memories == ["sem", "tan"]
        assert semantic_entered.is_set()
        assert tantivy_entered.is_set()
        assert ticks_after_both >= 1
        assert semantic_threads and tantivy_threads
        assert semantic_threads[0] != loop_thread
        assert tantivy_threads[0] != loop_thread
        assert semantic_threads[0] != tantivy_threads[0]
        assert semantic.search_calls[0][2] == 15
        assert tantivy.search_calls[0][2] == 15

    @pytest.mark.asyncio
    async def test_hybrid_init_and_search_hops_stay_off_loop(self) -> None:
        semantic_init = threading.Event()
        tantivy_init = threading.Event()
        semantic_search = threading.Event()
        tantivy_search = threading.Event()
        init_release = threading.Event()
        search_release = threading.Event()
        init_barrier = threading.Barrier(2, timeout=_BACKEND_WAIT_TIMEOUT)
        search_barrier = threading.Barrier(2, timeout=_BACKEND_WAIT_TIMEOUT)
        loop_thread = threading.get_ident()
        semantic_init_threads: list[int] = []
        tantivy_init_threads: list[int] = []
        semantic_search_threads: list[int] = []
        tantivy_search_threads: list[int] = []

        semantic = ControllableBackend(
            [("sem", 0.9, _TS)],
            entered=semantic_search,
            loop_progressed=search_release,
            barrier=search_barrier,
            thread_ids=semantic_search_threads,
            init_entered=semantic_init,
            init_barrier=init_barrier,
            init_thread_ids=semantic_init_threads,
            init_release=init_release,
            block_on_init=True,
        )
        tantivy = ControllableBackend(
            [("tan", 0.8)],
            entered=tantivy_search,
            loop_progressed=search_release,
            barrier=search_barrier,
            thread_ids=tantivy_search_threads,
            init_entered=tantivy_init,
            init_barrier=init_barrier,
            init_thread_ids=tantivy_init_threads,
            init_release=init_release,
            block_on_init=True,
        )
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("sem", 0.4), ("tan", 0.3)]
        pipeline = _make_pipeline(
            semantic=semantic,
            tantivy=tantivy,
            fusion=fusion,
            config=_make_config(fusion_ranking_threshold=0.0),
        )

        async def ticker() -> None:
            while True:
                if semantic_init.is_set() and tantivy_init.is_set():
                    init_release.set()
                if semantic_search.is_set() and tantivy_search.is_set():
                    search_release.set()
                    break
                await asyncio.sleep(0)

        ticker_task = asyncio.create_task(ticker())
        result = await asyncio.wait_for(
            pipeline.execute(_make_context()),
            timeout=5,
        )
        ticker_task.cancel()
        with suppress(asyncio.CancelledError):
            await ticker_task

        assert result.memories == ["sem", "tan"]
        assert semantic_init_threads[0] != loop_thread
        assert tantivy_init_threads[0] != loop_thread
        assert semantic_init_threads[0] != tantivy_init_threads[0]
        assert semantic_search_threads[0] != loop_thread
        assert tantivy_search_threads[0] != loop_thread
        assert semantic_search_threads[0] != tantivy_search_threads[0]

    @pytest.mark.asyncio
    async def test_fusion_offload_keeps_loop_responsive(self) -> None:
        entered = threading.Event()
        loop_progressed = threading.Event()
        loop_thread = threading.get_ident()
        fuse_threads: list[int] = []
        ticks_after_enter = 0

        class SlowFusion:
            method = "rrf"

            def fuse(
                self,
                *_result_sets: list[tuple[str, float]],
            ) -> list[tuple[str, float]]:
                fuse_threads.append(threading.get_ident())
                entered.set()
                if not loop_progressed.wait(timeout=_BACKEND_WAIT_TIMEOUT):
                    raise TimeoutError("event loop did not progress during fuse")
                return [("mem", 0.4)]

        semantic = MagicMock()
        semantic.search.return_value = [("mem", 0.9, _TS)]
        tantivy = MagicMock()
        tantivy.search.return_value = [("other", 0.8)]
        pipeline = _make_pipeline(
            semantic=semantic,
            tantivy=tantivy,
            fusion=SlowFusion(),
            config=_make_config(fusion_ranking_threshold=0.0),
        )

        async def ticker() -> None:
            nonlocal ticks_after_enter
            while True:
                if entered.is_set():
                    ticks_after_enter += 1
                    loop_progressed.set()
                await asyncio.sleep(0)
                if loop_progressed.is_set() and ticks_after_enter >= 3:
                    break

        ticker_task = asyncio.create_task(ticker())
        result = await asyncio.wait_for(
            pipeline.execute(_make_context()),
            timeout=5,
        )
        ticker_task.cancel()
        with suppress(asyncio.CancelledError):
            await ticker_task

        assert result.memories == ["mem"]
        assert ticks_after_enter >= 1
        assert fuse_threads
        assert fuse_threads[0] != loop_thread

    @pytest.mark.asyncio
    async def test_cancel_waits_for_worker_and_does_not_abort_it(self) -> None:
        entered = threading.Event()
        loop_progressed = threading.Event()
        finished = threading.Event()
        backend = ControllableBackend(
            [("mem", 0.9, _TS)],
            entered=entered,
            loop_progressed=loop_progressed,
            finished=finished,
        )
        pipeline = _make_pipeline(semantic=backend)
        search_task = asyncio.create_task(pipeline.execute(_make_context()))
        deadline = asyncio.get_running_loop().time() + _BACKEND_WAIT_TIMEOUT
        while not entered.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(0)
        assert entered.is_set()
        search_task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        assert not search_task.done()
        assert not finished.is_set()
        loop_progressed.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(search_task, timeout=5)
        assert finished.wait(timeout=_BACKEND_WAIT_TIMEOUT)

    @pytest.mark.asyncio
    async def test_hybrid_cancel_waits_for_both_workers(self) -> None:
        semantic_entered = threading.Event()
        tantivy_entered = threading.Event()
        loop_progressed = threading.Event()
        semantic_finished = threading.Event()
        tantivy_finished = threading.Event()
        barrier = threading.Barrier(2, timeout=_BACKEND_WAIT_TIMEOUT)
        semantic = ControllableBackend(
            [("sem", 0.9, _TS)],
            entered=semantic_entered,
            loop_progressed=loop_progressed,
            barrier=barrier,
            finished=semantic_finished,
        )
        tantivy = ControllableBackend(
            [("tan", 0.8)],
            entered=tantivy_entered,
            loop_progressed=loop_progressed,
            barrier=barrier,
            finished=tantivy_finished,
        )
        pipeline = _make_pipeline(
            semantic=semantic,
            tantivy=tantivy,
            fusion=MagicMock(method="rrf", fuse=MagicMock(return_value=[("sem", 0.4)])),
            config=_make_config(fusion_ranking_threshold=0.0),
        )
        search_task = asyncio.create_task(pipeline.execute(_make_context()))
        deadline = asyncio.get_running_loop().time() + _BACKEND_WAIT_TIMEOUT
        while not (semantic_entered.is_set() and tantivy_entered.is_set()):
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(0)
        assert semantic_entered.is_set() and tantivy_entered.is_set()
        search_task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        assert not search_task.done()
        assert not semantic_finished.is_set()
        assert not tantivy_finished.is_set()
        loop_progressed.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(search_task, timeout=5)
        assert semantic_finished.wait(timeout=_BACKEND_WAIT_TIMEOUT)
        assert tantivy_finished.wait(timeout=_BACKEND_WAIT_TIMEOUT)

    @pytest.mark.asyncio
    async def test_manager_search_index_lookup_stays_off_loop(self) -> None:
        entered = threading.Event()
        loop_progressed = threading.Event()
        loop_thread = threading.get_ident()
        index_threads: list[int] = []
        ticks_after_enter = 0
        index_size = 10000

        class BlockingIndexEngine:
            def __init__(self) -> None:
                self.search = MagicMock(return_value=[("mem", 0.9, _TS)])
                self.ensure_initialized = MagicMock()

            def is_ready(self) -> bool:
                return False

            @property
            def memory_store(self) -> object:
                return self

            def exists_many(self, workspace_id: str, contents: list[str]) -> set[str]:
                _ = workspace_id
                return set(contents)

            def count(self, workspace_id: str) -> int:
                _ = workspace_id
                index_threads.append(threading.get_ident())
                entered.set()
                if not loop_progressed.wait(timeout=_BACKEND_WAIT_TIMEOUT):
                    raise TimeoutError("event loop did not progress during index get")
                return index_size

            @property
            def _index(self) -> range:
                index_threads.append(threading.get_ident())
                entered.set()
                if not loop_progressed.wait(timeout=_BACKEND_WAIT_TIMEOUT):
                    raise TimeoutError("event loop did not progress during index get")
                return range(index_size)

            @property
            def index(self) -> range:
                return self._index

        config = MagicMock()
        config.workspace_id = "test_project"
        config.tantivy_index_path_template = "{workspace_id}_tantivy_test"
        config.enable_smart_replace = False
        config.reranker_engine = "none"
        config.embedding_cache_enabled = False
        config.eager_initialization = False
        config.fusion_method = "rrf"
        config.fusion_normalization = None
        config.fusion_rrf_k = 60
        config.enable_rrf_fusion = True
        config.fusion_ranking_threshold = 0.0
        config.search_limit = 10
        config.overfetch_adaptive = True
        config.overfetch_max_multiplier = 3.0
        config.overfetch_min_multiplier = 1.5
        config.overfetch_multiplier = 3
        config.openrouter_api_key.get_secret_value.return_value = "test-key"

        with (
            patch("reflectlog.application.memory.manager.USearchEngine"),
            patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"),
            patch("reflectlog.application.memory.manager.TantivyEngine"),
        ):
            manager = MemoryManager(config, _make_logger())

        semantic = BlockingIndexEngine()
        tantivy = MagicMock()
        tantivy.search.return_value = []
        tantivy.ensure_initialized = MagicMock()
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("mem", 0.4)]
        manager._semantic_engine = cast(Any, semantic)
        manager._tantivy_engine = tantivy
        manager._search_pipeline = SearchPipeline(
            semantic_engine=cast(Any, semantic),
            tantivy_engine=tantivy,
            fusion_engine=fusion,
            config=config,
            logger=manager.logger,
            memory_manager=manager,
        )

        async def ticker() -> None:
            nonlocal ticks_after_enter
            while True:
                if entered.is_set():
                    ticks_after_enter += 1
                    loop_progressed.set()
                await asyncio.sleep(0)
                if loop_progressed.is_set() and ticks_after_enter >= 3:
                    break

        ticker_task = asyncio.create_task(ticker())
        results = await asyncio.wait_for(manager.search("q", limit=10), timeout=5)
        ticker_task.cancel()
        with suppress(asyncio.CancelledError):
            await ticker_task

        expected_overfetch = calculate_adaptive_overfetch(10, index_size, config)
        zero_overfetch = calculate_adaptive_overfetch(10, 0, config)
        assert expected_overfetch != zero_overfetch
        assert results == ["mem"]
        assert ticks_after_enter >= 1
        assert index_threads
        assert index_threads[0] != loop_thread
        semantic.search.assert_called_once_with(
            query="q", workspace_id="test_project", limit=expected_overfetch
        )
        tantivy.search.assert_called_once_with("q", "test_project", expected_overfetch)

    @pytest.mark.asyncio
    async def test_manager_search_recovery_offload(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        loop_progressed = threading.Event()
        loop_thread = threading.get_ident()
        recover_threads: list[int] = []
        ticks_after_enter = 0

        config = MagicMock()
        config.workspace_id = "test_project"
        config.tantivy_index_path_template = "{workspace_id}_tantivy_test"
        config.enable_smart_replace = False
        config.reranker_engine = "none"
        config.embedding_cache_enabled = False
        config.eager_initialization = False
        config.fusion_method = "rrf"
        config.fusion_normalization = None
        config.fusion_rrf_k = 60
        config.enable_rrf_fusion = True
        config.fusion_ranking_threshold = 0.0
        config.search_limit = 10
        config.overfetch_adaptive = False
        config.overfetch_max_multiplier = 3.0
        config.overfetch_min_multiplier = 1.5
        config.overfetch_multiplier = 3
        config.openrouter_api_key.get_secret_value.return_value = "test-key"

        with (
            patch("reflectlog.application.memory.manager.USearchEngine"),
            patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"),
            patch("reflectlog.application.memory.manager.TantivyEngine"),
        ):
            manager = MemoryManager(config, _make_logger())

        semantic = MagicMock()
        semantic.search.return_value = [("mem", 0.9, _TS)]
        semantic.count.return_value = 1
        semantic.ensure_initialized = MagicMock()
        semantic.is_ready.return_value = True
        semantic.memory_store.exists_many.return_value = {"mem"}
        fusion = MagicMock()
        fusion.method = "rrf"
        fusion.fuse.return_value = [("mem", 0.4)]
        manager._semantic_engine = semantic
        manager._tantivy_engine = None
        manager._search_pipeline = SearchPipeline(
            semantic_engine=cast(Any, semantic),
            tantivy_engine=None,
            fusion_engine=fusion,
            config=config,
            logger=manager.logger,
            memory_manager=manager,
        )

        def blocking_reconcile() -> int:
            recover_threads.append(threading.get_ident())
            entered.set()
            if not release.wait(timeout=_BACKEND_WAIT_TIMEOUT):
                raise TimeoutError("recovery was not released")
            return 0

        patched: Any = manager
        patched.reconcile_pending_replacements = blocking_reconcile

        async def ticker() -> None:
            nonlocal ticks_after_enter
            while True:
                if entered.is_set():
                    ticks_after_enter += 1
                    loop_progressed.set()
                    if ticks_after_enter >= 3:
                        release.set()
                await asyncio.sleep(0)
                if release.is_set() and ticks_after_enter >= 3:
                    break

        ticker_task = asyncio.create_task(ticker())
        _ = await asyncio.wait_for(manager.search("q", limit=5), timeout=5)
        ticker_task.cancel()
        with suppress(asyncio.CancelledError):
            await ticker_task

        assert ticks_after_enter >= 3
        assert recover_threads
        assert recover_threads[0] != loop_thread
        semantic.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_search_recovery_initialization_error_aborts(self) -> None:
        config = MagicMock()
        config.workspace_id = "test_project"
        config.tantivy_index_path_template = "{workspace_id}_tantivy_test"
        config.enable_smart_replace = False
        config.reranker_engine = "none"
        config.embedding_cache_enabled = False
        config.eager_initialization = False
        config.fusion_method = "rrf"
        config.fusion_normalization = None
        config.fusion_rrf_k = 60
        config.enable_rrf_fusion = True
        config.fusion_ranking_threshold = 0.0
        config.search_limit = 10
        config.overfetch_adaptive = False
        config.overfetch_max_multiplier = 3.0
        config.overfetch_min_multiplier = 1.5
        config.overfetch_multiplier = 3
        config.openrouter_api_key.get_secret_value.return_value = "test-key"

        with (
            patch("reflectlog.application.memory.manager.USearchEngine"),
            patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"),
            patch("reflectlog.application.memory.manager.TantivyEngine"),
        ):
            manager = MemoryManager(config, _make_logger())

        semantic = MagicMock()
        manager._semantic_engine = semantic
        manager._tantivy_engine = None
        cause = InitializationError("index missing")

        def boom() -> int:
            raise cause

        patched: Any = manager
        patched.reconcile_pending_replacements = boom
        with pytest.raises(InitializationError) as exc_info:
            _ = await manager.search("q")
        assert exc_info.value is cause
        semantic.search.assert_not_called()


# ---------------------------------------------------------------------------
# Adaptive overfetch remains the canonical policy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCanonicalOverfetch:
    """Old limit-tier overfetch is gone; adaptive policy is the only one."""

    def test_adaptive_policy_used_not_limit_tiers(self) -> None:
        config = _make_config()
        config.overfetch_adaptive = True
        config.overfetch_max_multiplier = 3.0
        # Competing pipeline used min(limit*3, 50) for limit<=10 → 30 for limit=10.
        # Canonical adaptive uses max multiplier for small indexes: 10*3=30, then
        # the same number happens to match — check a large-index case instead.
        large = calculate_adaptive_overfetch(10, 10000, config)
        assert large == 15  # 10 * min_multiplier 1.5, above MIN_OVERFETCH_LIMIT

    def test_minimum_overfetch_floor(self) -> None:
        config = _make_config()
        config.overfetch_adaptive = False
        config.overfetch_multiplier = 1
        assert calculate_adaptive_overfetch(1, 0, config) >= 8
