"""Unit tests for search_strategies.py - 4-step search pipeline.

Covers uncovered lines: 128-137, 295, 365, 374-378, 451, 457, 475,
493-495, 538-573, 671-683.
"""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import anyio
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.search_strategies import (
    MIN_OVERFETCH_LIMIT,
    SearchContext,
    SearchPipeline,
    SearchResult,
    calculate_adaptive_overfetch,
)
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.exceptions import SearchError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchExplanations:
    @pytest.mark.parametrize("fusion", [True, False])
    async def test_filtered_source_ranks_and_candidate_scores(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_fusion_engine: MagicMock,
        mock_config: Mock,
        fusion: bool,
    ) -> None:
        mock_config.search_score_threshold = 0.5
        mock_config.fusion_ranking_threshold = 0.0
        mock_semantic_engine.search.return_value = [
            ("shared", 0.9, "ts"),
            ("weak", 0.2, "ts"),
            ("semantic", 0.7, "ts"),
        ]
        mock_tantivy_engine.search.return_value = [
            ("stale", 12.0),
            ("weak", 9.0),
            ("shared", 7.0),
        ]
        mock_semantic_engine.memory_store.exists_many.side_effect = (
            lambda _workspace, contents: set(contents) - {"stale"}
        )
        mock_fusion_engine.fuse.return_value = [
            ("shared", 0.03),
            ("weak", 0.02),
            ("semantic", 0.01),
        ]

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=fusion), explain=True
        )

        assert result.explanations is not None
        shared = result.explanations["shared"]
        assert shared.semantic is not None and shared.fulltext is not None
        assert (shared.semantic.rank, shared.semantic.score) == (1, 0.9)
        assert (shared.fulltext.rank, shared.fulltext.score) == (2, 7.0)
        assert result.explanations["weak"].semantic is None
        assert result.explanations["semantic"].fulltext is None
        assert result.explanations["semantic"].semantic is not None
        assert result.explanations["semantic"].semantic.rank == 2
        assert shared.pre_rerank_score == (0.03 if fusion else 0.9)
        assert shared.pre_rerank_score_kind == ("rrf" if fusion else "concatenated")
        assert shared.final_score_kind == shared.pre_rerank_score_kind
        assert result.stage_outcomes is not None
        assert result.stage_outcomes.semantic == "applied"
        assert result.stage_outcomes.fulltext == "applied"
        assert result.stage_outcomes.reranker == "disabled"

    @pytest.mark.parametrize("engine", ["cross_encoder", "openrouter"])
    @pytest.mark.parametrize(
        "outcome", ["failed", "degraded", "applied", "unavailable", "filtered"]
    )
    async def test_reranker_outcomes_and_pre_rerank_snapshot(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_config: Mock,
        mock_memory_manager: MagicMock,
        engine: str,
        outcome: str,
    ) -> None:
        mock_config.reranker_engine = engine
        mock_semantic_engine.search.return_value = [("a", 0.9, "ts"), ("b", 0.8, "ts")]
        reranker = MagicMock()
        reranker.rerank_async = AsyncMock(return_value=None)
        if outcome == "failed":
            reranker.rerank_async.side_effect = RuntimeError("private failure")
        elif outcome == "applied":
            reranker.rerank_async.return_value = [("b", 0.95)]
        elif outcome == "filtered":
            reranker.rerank_async.return_value = []
        if engine == "cross_encoder":
            mock_memory_manager.cross_encoder_reranker = (
                None if outcome == "unavailable" else reranker
            )
        else:
            mock_memory_manager.openrouter_reranker = (
                None if outcome == "unavailable" else reranker
            )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=False), explain=True
        )

        assert result.stage_outcomes is not None
        assert result.stage_outcomes.reranker == (
            "applied" if outcome == "filtered" else outcome
        )
        assert result.explanations is not None
        if outcome == "filtered":
            assert result.memories == [] and result.scores == []
            assert result.explanations == {}
            assert result.score_kind == "none"
            reranker.rerank_async.assert_awaited_once()
            return
        assert result.explanations["b"].pre_rerank_score == 0.8
        assert result.explanations["b"].pre_rerank_score_kind == "semantic"
        assert result.explanations["b"].final_score_kind == (
            "reranker" if outcome == "applied" else "semantic"
        )
        assert reranker.rerank_async.await_count == (
            0 if outcome == "unavailable" else 1
        )

    @pytest.mark.parametrize(
        "count,expected", [(0, "empty"), (1, "skipped_single_hit")]
    )
    async def test_small_candidate_sets(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        count: int,
        expected: str,
    ) -> None:
        mock_semantic_engine.search.return_value = [("a", 0.9, "ts")][:count]

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=False), explain=True
        )

        assert result.stage_outcomes is not None
        assert result.stage_outcomes.reranker == expected

    @pytest.mark.parametrize("failed_backend", ["semantic", "fulltext"])
    async def test_backend_fallback_uses_original_errors(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        failed_backend: str,
    ) -> None:
        mock_semantic_engine.search.return_value = [("a", 0.9, "ts")]
        mock_tantivy_engine.search.return_value = [("a", 7.0)]
        failed = (
            mock_semantic_engine
            if failed_backend == "semantic"
            else mock_tantivy_engine
        )
        failed.search.side_effect = RuntimeError("private failure")

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=False), explain=True
        )

        assert result.stage_outcomes is not None
        assert (result.stage_outcomes.semantic, result.stage_outcomes.fulltext) == (
            ("failed", "applied")
            if failed_backend == "semantic"
            else ("applied", "failed")
        )
        mock_semantic_engine.search.assert_called_once()
        mock_tantivy_engine.search.assert_called_once()
        assert result.memories == ["a"]

    async def test_missing_fulltext_backend_is_unavailable(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
    ) -> None:
        pipeline._tantivy_engine = None
        mock_semantic_engine.search.return_value = [("a", 0.9, "ts")]

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=False), explain=True
        )

        assert result.stage_outcomes is not None
        assert result.stage_outcomes.fulltext == "unavailable"
        assert result.explanations is not None
        assert result.explanations["a"].fulltext is None

    async def test_threshold_and_live_filter_leave_empty_backends(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Mock,
    ) -> None:
        mock_config.search_score_threshold = 0.8
        mock_semantic_engine.search.return_value = [("weak", 0.2, "ts")]
        mock_tantivy_engine.search.return_value = [("stale", 7.0)]
        mock_semantic_engine.memory_store.exists_many.side_effect = (
            lambda _workspace, _contents: set[str]()
        )

        result = await pipeline.execute(
            _make_context(enable_rrf_fusion=False), explain=True
        )

        assert result.stage_outcomes is not None
        assert (
            result.stage_outcomes.semantic,
            result.stage_outcomes.fulltext,
            result.stage_outcomes.reranker,
        ) == ("empty", "empty", "empty")
        assert result.explanations == {}

    async def test_empty_fusion_threshold_result_has_stage_outcomes(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_fusion_engine: MagicMock,
    ) -> None:
        mock_semantic_engine.search.return_value = [("a", 0.9, "ts")]
        mock_tantivy_engine.search.return_value = [("b", 7.0)]
        mock_fusion_engine.fuse.return_value = []

        result = await pipeline.execute(_make_context(), explain=True)

        assert result.stage_outcomes is not None
        assert (
            result.stage_outcomes.semantic,
            result.stage_outcomes.fulltext,
            result.stage_outcomes.reranker,
        ) == ("applied", "applied", "empty")
        assert result.explanations == {}

    @pytest.mark.parametrize("fallback_outcome", ["failed", "degraded"])
    async def test_concurrent_explanations_are_request_local(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_config: Mock,
        mock_memory_manager: MagicMock,
        fallback_outcome: str,
    ) -> None:
        mock_config.reranker_engine = "openrouter"
        mock_semantic_engine.search.return_value = [("a", 0.9, "ts"), ("b", 0.8, "ts")]
        entered = anyio.Event()
        release = anyio.Event()
        results: dict[str, SearchResult] = {}

        async def rerank(
            query: str,
            _results: list[tuple[str, float]],
            _timestamps: dict[str, str],
            *,
            top_k: int,
        ) -> list[tuple[str, float]] | None:
            if query == "degraded":
                entered.set()
                await release.wait()
                if fallback_outcome == "failed":
                    raise RuntimeError("private failure")
                return None
            await entered.wait()
            release.set()
            return [("b", 0.99)]

        mock_memory_manager.openrouter_reranker = MagicMock(
            rerank_async=AsyncMock(side_effect=rerank)
        )

        async def search(query: str) -> None:
            results[query] = await pipeline.execute(
                _make_context(query=query, enable_rrf_fusion=False), explain=True
            )

        async with anyio.create_task_group() as group:
            group.start_soon(search, "degraded")
            group.start_soon(search, "applied")

        degraded = results["degraded"]
        applied = results["applied"]
        assert (
            degraded.stage_outcomes is not None and applied.stage_outcomes is not None
        )
        assert degraded.stage_outcomes.reranker == fallback_outcome
        assert applied.stage_outcomes.reranker == "applied"
        assert degraded.explanations is not None and applied.explanations is not None
        assert degraded.explanations["b"].final_score_kind == "semantic"
        assert applied.explanations["b"].final_score_kind == "reranker"

    async def test_opt_in_preserves_results_and_default_metadata(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
    ) -> None:
        mock_semantic_engine.search.return_value = [("a", 0.9, "ts"), ("b", 0.8, "ts")]

        plain = await pipeline.execute(_make_context(enable_rrf_fusion=False))
        explained = await pipeline.execute(
            _make_context(enable_rrf_fusion=False), explain=True
        )

        assert plain.explanations is None and plain.stage_outcomes is None
        assert (
            plain.memories,
            plain.scores,
            plain.score_kind,
            plain.timestamp_map,
        ) == (
            explained.memories,
            explained.scores,
            explained.score_kind,
            explained.timestamp_map,
        )


@pytest.fixture
def mock_config() -> Mock:
    """Minimal mock Config for search_strategies tests."""
    config = Mock(spec=Config)
    config.workspace_id = "test_project"
    config.fusion_ranking_threshold = 0.1
    config.fusion_rrf_k = 60
    config.reranker_engine = "none"
    config.search_score_threshold = 0.0
    config.cross_encoder_top_k = 20
    config.cross_encoder_model = "BAAI/bge-reranker-v2-m3"
    config.overfetch_multiplier = 3
    config.overfetch_adaptive = True
    config.overfetch_min_multiplier = 1.5
    config.overfetch_max_multiplier = 3.0
    return config


@pytest.fixture
def mock_logger() -> Mock:
    """Mock structured logger."""
    return Mock(spec=StructuredLogger)


@pytest.fixture
def mock_semantic_engine() -> MagicMock:
    """Mock ISemanticSearchEngine."""
    engine = MagicMock()
    engine.search = MagicMock(return_value=[])
    engine.ensure_initialized = MagicMock()
    engine.is_ready = MagicMock(return_value=True)
    engine.contains_id = MagicMock(return_value=False)
    engine.count = MagicMock(return_value=0)
    engine.memory_store.exists_many.side_effect = lambda _workspace, contents: set(
        contents
    )
    engine.get_records_by_contents.return_value = []
    return engine


@pytest.fixture
def mock_tantivy_engine() -> MagicMock:
    """Mock TantivyEngine."""
    engine = MagicMock()
    engine.search = MagicMock(return_value=[])
    engine.ensure_initialized = MagicMock()
    engine.is_ready = MagicMock(return_value=True)
    return engine


@pytest.fixture
def mock_fusion_engine() -> MagicMock:
    """Mock FusionEngine."""
    engine = MagicMock()
    engine.method = "rrf"
    engine.fuse = MagicMock(return_value=[])
    return engine


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Mock MemoryManager for lazy reranker fetching."""
    manager = MagicMock()
    manager.cross_encoder_reranker = None
    manager.openrouter_reranker = None
    return manager


@pytest.fixture
def pipeline(
    mock_semantic_engine: MagicMock,
    mock_tantivy_engine: MagicMock,
    mock_fusion_engine: MagicMock,
    mock_config: Mock,
    mock_logger: Mock,
    mock_memory_manager: MagicMock,
) -> SearchPipeline:
    """Construct a SearchPipeline with mocked dependencies."""
    return SearchPipeline(
        semantic_engine=mock_semantic_engine,
        tantivy_engine=mock_tantivy_engine,
        fusion_engine=mock_fusion_engine,
        config=cast(Config, mock_config),
        logger=mock_logger,
        memory_manager=mock_memory_manager,
    )


def _make_context(
    *,
    query: str = "test query",
    limit: int = 5,
    overfetch_limit: int = 15,
    enable_rrf_fusion: bool = True,
    reranker_engine: str = "none",
    workspace_id: str = "test_project",
) -> SearchContext:
    """Build a SearchContext with sensible defaults."""
    return SearchContext(
        query=query,
        limit=limit,
        overfetch_limit=overfetch_limit,
        enable_rrf_fusion=enable_rrf_fusion,
        reranker_engine=reranker_engine,
        workspace_id=workspace_id,
    )


# ---------------------------------------------------------------------------
# TestSearchContext & TestSearchResult
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchContext:
    """Tests for SearchContext dataclass."""

    def test_construction(self) -> None:
        """Fields should be stored correctly."""
        ctx = _make_context(query="hello", limit=10, workspace_id="proj")
        assert ctx.query == "hello"
        assert ctx.limit == 10
        assert ctx.workspace_id == "proj"


@pytest.mark.parametrize(
    ("semantic", "fulltext", "rrf", "rerank", "expected"),
    [
        (True, True, True, "none", "rrf"),
        (True, False, True, "none", "semantic"),
        (False, True, True, "none", "fulltext"),
        (True, True, False, "none", "concatenated"),
        (True, True, True, "success", "reranker"),
        (True, True, True, "exception", "rrf"),
        (True, True, True, "disabled", "rrf"),
        (True, True, True, "null", "rrf"),
        (False, False, True, "none", "none"),
    ],
)
async def test_score_kind_matrix(
    pipeline: SearchPipeline,
    mock_config: Mock,
    mock_memory_manager: MagicMock,
    semantic: bool,
    fulltext: bool,
    rrf: bool,
    rerank: str,
    expected: str,
) -> None:
    semantic_hits = [("a", 0.9, "time"), ("b", 0.8, "time")] if semantic else []
    fulltext_hits = [("a", 4.0), ("b", 3.0)] if fulltext else []
    candidates = [("a", 0.03), ("b", 0.02)] if semantic or fulltext else []
    mock_config.reranker_engine = "cross_encoder" if rerank != "none" else "none"
    if rerank not in ("none", "disabled"):
        reranker = Mock()
        reranker.rerank_async = AsyncMock(
            return_value=None if rerank == "null" else [("b", 0.97), ("a", 0.92)]
        )
        if rerank == "exception":
            reranker.rerank_async.side_effect = RuntimeError("failure")
        mock_memory_manager.cross_encoder_reranker = reranker
    with (
        patch.object(
            pipeline,
            "_step1_parallel_search",
            AsyncMock(return_value=(semantic_hits, fulltext_hits, None, None)),
        ),
        patch.object(
            pipeline, "_step2_fusion_or_concatenate", AsyncMock(return_value=candidates)
        ),
        patch.object(
            pipeline, "_step3_fusion_threshold", AsyncMock(return_value=candidates)
        ),
    ):
        result = await pipeline.execute(_make_context(enable_rrf_fusion=rrf))
    expected_hits = [("b", 0.97), ("a", 0.92)] if rerank == "success" else candidates
    assert result.score_kind == expected
    assert list(zip(result.memories, result.scores, strict=True)) == expected_hits


@pytest.mark.unit
class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_construction(self) -> None:
        """Fields should be stored correctly."""
        result = SearchResult(
            memories=["a", "b"],
            timestamp_map={"a": "2025-01-01T00:00:00Z"},
            semantic_results=[("a", 0.9, "2025-01-01T00:00:00Z")],
            tantivy_results=[("b", 0.8)],
        )
        assert result.memories == ["a", "b"]
        assert len(result.semantic_results) == 1
        assert len(result.tantivy_results) == 1


# ---------------------------------------------------------------------------
# TestSearchPipelineExecute – lines 128-137 (error path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchPipelineExecute:
    """Tests for SearchPipeline.execute()."""

    @pytest.mark.asyncio
    async def test_execute_raises_search_error_on_failure(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
    ) -> None:
        """execute() wraps unexpected exceptions in SearchError (lines 128-137)."""
        mock_semantic_engine.search.side_effect = RuntimeError("engine boom")
        mock_tantivy_engine.search.side_effect = RuntimeError("engine boom")

        ctx = _make_context()

        with pytest.raises(SearchError, match="Failed to execute search"):
            await pipeline.execute(ctx)

    @pytest.mark.asyncio
    async def test_execute_queries_all_backends(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_fusion_engine: MagicMock,
    ) -> None:
        """Hybrid search invokes full pipeline."""
        mock_semantic_engine.search.return_value = [
            ("msg1", 0.9, "2025-01-01T00:00:00Z"),
        ]
        mock_tantivy_engine.search.return_value = [("msg2", 0.8)]
        mock_fusion_engine.fuse.return_value = [("msg1", 0.5), ("msg2", 0.3)]

        ctx = _make_context(enable_rrf_fusion=True)

        result = await pipeline.execute(ctx)

        assert len(result.memories) >= 1

    @pytest.mark.asyncio
    async def test_default_completes_timestamps_without_reranker(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_fusion_engine: MagicMock,
    ) -> None:
        mock_semantic_engine.search.return_value = [("semantic", 0.9, "semantic-time")]
        mock_tantivy_engine.search.return_value = [("lexical", 0.8)]
        mock_fusion_engine.fuse.return_value = [
            ("semantic", 0.5),
            ("lexical", 0.4),
        ]
        record = Mock(content="lexical", created_at="lexical-time")
        mock_semantic_engine.get_records_by_contents.return_value = [record]

        result = await pipeline.execute(_make_context())

        assert result.timestamp_map == {
            "semantic": "semantic-time",
            "lexical": "lexical-time",
        }
        mock_semantic_engine.get_records_by_contents.assert_called_once_with(
            "test_project", ["lexical"]
        )

    @pytest.mark.asyncio
    async def test_memories_only_without_reranker_skips_timestamp_lookup(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_fusion_engine: MagicMock,
    ) -> None:
        mock_semantic_engine.search.return_value = [("semantic", 0.9, "semantic-time")]
        mock_tantivy_engine.search.return_value = [("lexical", 0.8)]
        mock_fusion_engine.fuse.return_value = [
            ("semantic", 0.5),
            ("lexical", 0.4),
        ]

        result = await pipeline.execute(_make_context(), include_timestamp_map=False)

        assert result.memories == ["semantic", "lexical"]
        assert result.timestamp_map == {"semantic": "semantic-time"}
        mock_semantic_engine.get_records_by_contents.assert_not_called()

    @pytest.mark.asyncio
    async def test_memories_only_with_reranker_completes_timestamps(
        self,
        pipeline: SearchPipeline,
        mock_config: Mock,
        mock_memory_manager: MagicMock,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_fusion_engine: MagicMock,
    ) -> None:
        mock_config.reranker_engine = "cross_encoder"
        reranker = MagicMock()
        reranker.rerank_async = AsyncMock(return_value=[("lexical", 0.9)])
        mock_memory_manager.cross_encoder_reranker = reranker
        mock_semantic_engine.search.return_value = [("semantic", 0.9, "semantic-time")]
        mock_tantivy_engine.search.return_value = [("lexical", 0.8)]
        mock_fusion_engine.fuse.return_value = [
            ("semantic", 0.5),
            ("lexical", 0.4),
        ]
        mock_semantic_engine.get_records_by_contents.return_value = [
            Mock(content="lexical", created_at="lexical-time")
        ]

        result = await pipeline.execute(_make_context(), include_timestamp_map=False)

        assert result.memories == ["lexical"]
        assert result.timestamp_map == {
            "semantic": "semantic-time",
            "lexical": "lexical-time",
        }
        mock_semantic_engine.get_records_by_contents.assert_called_once_with(
            "test_project", ["lexical"]
        )
        assert reranker.rerank_async.await_args is not None
        assert reranker.rerank_async.await_args.args[2] == result.timestamp_map


# ---------------------------------------------------------------------------
# TestSearchTantivy – line 295 (tantivy_engine is None)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchTantivy:
    """Tests for _search_tantivy()."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_tantivy_is_unavailable(
        self,
        mock_config: Mock,
        mock_logger: Mock,
        mock_fusion_engine: MagicMock,
        mock_semantic_engine: MagicMock,
        mock_memory_manager: MagicMock,
    ) -> None:
        pipeline = SearchPipeline(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            fusion_engine=mock_fusion_engine,
            config=cast(Config, mock_config),
            logger=mock_logger,
            memory_manager=mock_memory_manager,
        )

        result = await pipeline._search_tantivy("query", 10, "proj")
        assert result == ([], None)


# ---------------------------------------------------------------------------
# TestConcatenateResults – lines 365, 374-378
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConcatenateResults:
    """Tests for _concatenate_results()."""

    def test_limit_reached_during_semantic(self, pipeline: SearchPipeline) -> None:
        """Stops adding semantic results once limit is reached (line 365)."""
        semantic = [("s1", 0.9), ("s2", 0.8), ("s3", 0.7)]
        tantivy = [("t1", 0.6)]

        combined = pipeline._concatenate_results(semantic, tantivy, limit=2)

        assert len(combined) == 2
        assert combined[0][0] == "s1"
        assert combined[1][0] == "t1"

    def test_limit_reached_during_tantivy(self, pipeline: SearchPipeline) -> None:
        """Stops adding tantivy results once limit is reached (lines 374-378)."""
        semantic = [("s1", 0.9)]
        tantivy = [("t1", 0.6), ("t2", 0.5), ("t3", 0.4)]

        combined = pipeline._concatenate_results(semantic, tantivy, limit=2)

        assert len(combined) == 2
        assert combined[0][0] == "s1"
        assert combined[1][0] == "t1"

    def test_deduplication_across_engines(self, pipeline: SearchPipeline) -> None:
        """Duplicate memories across engines are skipped."""
        semantic = [("shared", 0.9)]
        tantivy = [("shared", 0.6), ("unique", 0.5)]

        combined = pipeline._concatenate_results(semantic, tantivy, limit=10)

        assert len(combined) == 2
        msgs = [m for m, _ in combined]
        assert msgs == ["shared", "unique"]

    def test_user_limit_keeps_lexical_slot_when_overfetching(
        self, pipeline: SearchPipeline
    ) -> None:
        """Reserve the lexical slot in the user-visible page, not only overfetch."""
        semantic = [("s1", 0.9), ("s2", 0.8), ("s3", 0.7), ("s4", 0.6)]
        tantivy = [("t1", 0.5)]

        combined = pipeline._concatenate_results(
            semantic, tantivy, limit=5, user_limit=2
        )

        assert [msg for msg, _ in combined[:2]] == ["s1", "t1"]
        assert "t1" in {msg for msg, _ in combined}


# ---------------------------------------------------------------------------
# TestGetReranker – lines 451, 457
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetReranker:
    """Tests for _get_reranker()."""

    def test_returns_none_when_memory_manager_is_none(
        self,
        mock_config: Mock,
        mock_logger: Mock,
        mock_fusion_engine: MagicMock,
        mock_semantic_engine: MagicMock,
    ) -> None:
        """_get_reranker() returns (None, None) when memory_manager is None (line 451)."""
        pipeline = SearchPipeline(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            fusion_engine=mock_fusion_engine,
            config=cast(Config, mock_config),
            logger=mock_logger,
            memory_manager=None,
        )

        rtype, rinstance = pipeline._get_reranker()
        assert rtype is None
        assert rinstance is None

    def test_returns_cross_encoder_reranker(
        self,
        pipeline: SearchPipeline,
        mock_config: Mock,
        mock_memory_manager: MagicMock,
    ) -> None:
        """Returns ("cross_encoder", reranker) when configured (line 457)."""
        mock_config.reranker_engine = "cross_encoder"
        mock_reranker = MagicMock()
        mock_memory_manager.cross_encoder_reranker = mock_reranker

        rtype, rinstance = pipeline._get_reranker()
        assert rtype == "cross_encoder"
        assert rinstance is mock_reranker

    def test_returns_none_when_reranker_is_none(
        self,
        pipeline: SearchPipeline,
        mock_config: Mock,
        mock_memory_manager: MagicMock,
    ) -> None:
        """Returns (None, None) when no lazy reranker instance is available."""
        mock_config.reranker_engine = "cross_encoder"
        mock_memory_manager.cross_encoder_reranker = None

        rtype, rinstance = pipeline._get_reranker()
        assert rtype is None
        assert rinstance is None


# ---------------------------------------------------------------------------
# TestStep4Reranking – line 475 (cross_encoder path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStep4Reranking:
    """Tests for _step4_reranking()."""

    @pytest.mark.asyncio
    async def test_cross_encoder_path(
        self,
        pipeline: SearchPipeline,
        mock_config: Mock,
        mock_memory_manager: MagicMock,
    ) -> None:
        """_step4_reranking() delegates to _rerank_cross_encoder (line 475)."""
        mock_config.reranker_engine = "cross_encoder"
        mock_reranker = AsyncMock()
        mock_reranker.rerank_async = AsyncMock(return_value=[("msg1", 0.95)])
        mock_memory_manager.cross_encoder_reranker = mock_reranker

        ctx = _make_context(reranker_engine="cross_encoder")
        results = [("msg1", 0.5), ("msg2", 0.3)]
        timestamp_map = {"msg1": "2025-01-01T00:00:00Z"}

        reranked = await pipeline._step4_reranking(ctx, results, timestamp_map, 4)

        assert reranked == ([("msg1", 0.95)], True)

    @pytest.mark.asyncio
    async def test_no_reranker_returns_original(
        self,
        pipeline: SearchPipeline,
        mock_config: Mock,
        mock_memory_manager: MagicMock,
    ) -> None:
        """When no reranker is configured, returns results unchanged."""
        mock_config.reranker_engine = "none"
        mock_memory_manager.cross_encoder_reranker = None

        ctx = _make_context(reranker_engine="none")
        results = [("msg1", 0.5), ("msg2", 0.3)]

        reranked = await pipeline._step4_reranking(ctx, results, {}, 4)
        assert reranked == (results, False)

    async def test_openrouter_dispatch_and_failure_fallback(
        self,
        pipeline: SearchPipeline,
        mock_config: Mock,
        mock_memory_manager: MagicMock,
        mock_logger: Mock,
    ) -> None:
        mock_config.reranker_engine = "openrouter"
        reranker = Mock()
        reranker.rerank_async = AsyncMock(return_value=[("second", 0.95)])
        mock_memory_manager.openrouter_reranker = reranker
        context = _make_context(reranker_engine="openrouter", limit=5)
        candidates = [("first", 0.04), ("second", 0.03)]
        stamps = {"first": "2025-01-01T00:00:00Z"}
        assert await pipeline._step4_reranking(context, candidates, stamps, 4) == (
            [("second", 0.95)],
            True,
        )
        reranker.rerank_async.assert_awaited_once_with(
            "test query", candidates, stamps, top_k=5
        )

        reranker.rerank_async.side_effect = RuntimeError("private response body")
        assert await pipeline._step4_reranking(context, candidates, stamps, 4) == (
            candidates,
            False,
        )
        warning = mock_logger.warning.call_args
        assert warning.kwargs["extra"]["error_type"] == "RuntimeError"
        assert "private response body" not in str(warning)


# ---------------------------------------------------------------------------
# TestRerankCrossEncoder – lines 538-573
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRerankCrossEncoder:
    """Tests for _rerank_cross_encoder()."""

    @pytest.mark.asyncio
    async def test_fallback_when_no_reranker_provided(
        self,
        pipeline: SearchPipeline,
        mock_config: Mock,
        mock_memory_manager: MagicMock,
    ) -> None:
        """Returns results unmodified when cross_encoder_reranker is None (lines 538-541)."""
        mock_config.reranker_engine = "none"
        mock_memory_manager.cross_encoder_reranker = None

        ctx = _make_context()
        results = [("msg1", 0.5), ("msg2", 0.3)]

        reranked = await pipeline._rerank_cross_encoder(
            ctx, results, step_num=4, cross_encoder_reranker=None
        )
        assert reranked == (results, False)

    @pytest.mark.asyncio
    async def test_cross_encoder_reranking_with_provided_reranker(
        self, pipeline: SearchPipeline, mock_config: Mock, mock_logger: Mock
    ) -> None:
        """Full cross-encoder reranking path (lines 543-573)."""
        mock_reranker = AsyncMock()
        mock_reranker.rerank_async = AsyncMock(return_value=[("msg1", 0.95)])

        ctx = _make_context()
        results = [("msg1", 0.5), ("msg2", 0.3)]

        reranked = await pipeline._rerank_cross_encoder(
            ctx, results, step_num=4, cross_encoder_reranker=mock_reranker
        )

        assert reranked == ([("msg1", 0.95)], True)
        mock_reranker.rerank_async.assert_awaited_once_with(
            ctx.query, results, top_k=20
        )

        # Verify logging calls happened
        assert mock_logger.info.call_count >= 2


# ---------------------------------------------------------------------------
# TestCalculateAdaptiveOverfetch – lines 671-683
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculateAdaptiveOverfetch:
    """Tests for calculate_adaptive_overfetch()."""

    def test_static_multiplier_when_adaptive_disabled(self, mock_config: Mock) -> None:
        """Uses static multiplier when adaptive is disabled."""
        mock_config.overfetch_adaptive = False
        mock_config.overfetch_multiplier = 3

        result = calculate_adaptive_overfetch(5, 500, mock_config)
        assert result == max(5 * 3, MIN_OVERFETCH_LIMIT)

    def test_static_multiplier_when_index_empty(self, mock_config: Mock) -> None:
        """Uses static multiplier when index_size is 0."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_multiplier = 3

        result = calculate_adaptive_overfetch(5, 0, mock_config)
        assert result == max(5 * 3, MIN_OVERFETCH_LIMIT)

    def test_max_multiplier_for_small_index(self, mock_config: Mock) -> None:
        """Uses max multiplier for small indexes (<= 100)."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_max_multiplier = 3.0
        mock_config.overfetch_min_multiplier = 1.5

        result = calculate_adaptive_overfetch(10, 50, mock_config)
        assert result == int(10 * 3.0)

    def test_min_multiplier_for_large_index(self, mock_config: Mock) -> None:
        """Uses min multiplier for large indexes (>= 10000)."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_max_multiplier = 3.0
        mock_config.overfetch_min_multiplier = 1.5

        result = calculate_adaptive_overfetch(10, 10000, mock_config)
        assert result == max(int(10 * 1.5), MIN_OVERFETCH_LIMIT)

    def test_logarithmic_interpolation_mid_index(self, mock_config: Mock) -> None:
        """Logarithmic interpolation for mid-range index (lines 671-683)."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_max_multiplier = 3.0
        mock_config.overfetch_min_multiplier = 1.5

        result = calculate_adaptive_overfetch(10, 1000, mock_config)

        # Should be between min and max multiplier
        assert int(10 * 1.5) <= result <= int(10 * 3.0)
        # And above the minimum overfetch limit
        assert result >= MIN_OVERFETCH_LIMIT

    def test_minimum_overfetch_limit_enforced(self, mock_config: Mock) -> None:
        """Result is never below MIN_OVERFETCH_LIMIT."""
        mock_config.overfetch_adaptive = False
        mock_config.overfetch_multiplier = 1

        result = calculate_adaptive_overfetch(1, 0, mock_config)
        assert result >= MIN_OVERFETCH_LIMIT

    def test_interpolation_at_boundary_100(self, mock_config: Mock) -> None:
        """Exact boundary at index_size=100 yields max multiplier."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_max_multiplier = 3.0
        mock_config.overfetch_min_multiplier = 1.5

        result = calculate_adaptive_overfetch(10, 100, mock_config)
        assert result == int(10 * 3.0)

    def test_interpolation_at_boundary_10000(self, mock_config: Mock) -> None:
        """Exact boundary at index_size=10000 yields min multiplier."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_max_multiplier = 3.0
        mock_config.overfetch_min_multiplier = 1.5

        result = calculate_adaptive_overfetch(10, 10000, mock_config)
        assert result == max(int(10 * 1.5), MIN_OVERFETCH_LIMIT)

    def test_quality_multiplier_raises_adaptive_max(self, mock_config: Mock) -> None:
        """QUALITY overfetch_multiplier=5 must raise the adaptive max."""
        mock_config.overfetch_adaptive = True
        mock_config.overfetch_multiplier = 5
        mock_config.overfetch_max_multiplier = 3.0
        mock_config.overfetch_min_multiplier = 1.5

        result = calculate_adaptive_overfetch(10, 50, mock_config)
        assert result == int(10 * 5)


@pytest.mark.unit
class TestEffectiveFusionThreshold:
    def test_leftover_zero_one_gate_is_ignored(
        self, mock_config: Mock, pipeline: SearchPipeline
    ) -> None:
        mock_config.fusion_ranking_threshold = 0.8
        mock_config.fusion_rrf_k = 60
        raw = [("a", 1.0 / 61), ("b", 1.0 / 62)]
        assert pipeline._effective_fusion_threshold(raw) == 0.0

    def test_raw_rrf_scale_threshold_is_kept(
        self, mock_config: Mock, pipeline: SearchPipeline
    ) -> None:
        mock_config.fusion_ranking_threshold = 0.01
        mock_config.fusion_rrf_k = 60
        raw = [("a", 1.0 / 61), ("b", 1.0 / 62)]
        assert pipeline._effective_fusion_threshold(raw) == 0.01

    def test_normalized_mock_scores_keep_configured_gate(
        self, mock_config: Mock, pipeline: SearchPipeline
    ) -> None:
        mock_config.fusion_ranking_threshold = 0.5
        mock_config.fusion_rrf_k = 60
        assert pipeline._effective_fusion_threshold([("a", 0.5), ("b", 0.3)]) == 0.5

    def test_weighted_raw_rrf_ignores_leftover_gate(
        self, mock_config: Mock, pipeline: SearchPipeline
    ) -> None:
        mock_config.fusion_ranking_threshold = 0.8
        mock_config.fusion_rrf_k = 60
        mock_config.fusion_weights = [2.0, 1.0]
        raw = [("a", 2.0 / 61), ("b", 1.0 / 61)]
        assert pipeline._effective_fusion_threshold(raw, n_backends=2) == 0.0


# ---------------------------------------------------------------------------
# TestHybridSearchIntegration – end-to-end pipeline paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHybridSearchIntegration:
    """Integration-style tests covering full pipeline flows."""

    @pytest.mark.asyncio
    async def test_rrf_fusion_filters_all_results(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_fusion_engine: MagicMock,
        mock_config: Mock,
    ) -> None:
        """When all fused results are below threshold, returns empty."""
        mock_semantic_engine.search.return_value = [
            ("msg1", 0.9, "2025-01-01T00:00:00Z"),
        ]
        mock_tantivy_engine.search.return_value = [("msg2", 0.8)]
        mock_fusion_engine.fuse.return_value = [("msg1", 0.05), ("msg2", 0.01)]
        mock_config.fusion_ranking_threshold = 0.5

        ctx = _make_context(enable_rrf_fusion=True)
        result = await pipeline.execute(ctx)

        assert result.memories == []

    @pytest.mark.asyncio
    async def test_single_result_skips_reranking(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_fusion_engine: MagicMock,
        mock_config: Mock,
        mock_logger: Mock,
    ) -> None:
        """A single result after fusion skips reranking."""
        mock_semantic_engine.search.return_value = [
            ("msg1", 0.9, "2025-01-01T00:00:00Z"),
        ]
        mock_tantivy_engine.search.return_value = []
        mock_fusion_engine.fuse.return_value = [("msg1", 0.5)]
        mock_config.fusion_ranking_threshold = 0.1

        ctx = _make_context(enable_rrf_fusion=True)
        result = await pipeline.execute(ctx)

        assert result.memories == ["msg1"]
        # Check skip-reranking log was called
        skip_logged = any(
            "skipped" in str(call).lower() for call in mock_logger.info.call_args_list
        )
        assert skip_logged

    @pytest.mark.asyncio
    async def test_concatenation_mode_skips_fusion_threshold(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_tantivy_engine: MagicMock,
        mock_config: Mock,
        mock_logger: Mock,
    ) -> None:
        """When RRF fusion disabled, uses concatenation (no threshold step)."""
        mock_semantic_engine.search.return_value = [
            ("msg1", 0.9, "2025-01-01T00:00:00Z"),
        ]
        mock_tantivy_engine.search.return_value = [("msg2", 0.8)]

        ctx = _make_context(
            enable_rrf_fusion=False,
        )
        result = await pipeline.execute(ctx)

        assert "msg1" in result.memories
        assert "msg2" in result.memories


# ---------------------------------------------------------------------------
# TestSearchSemantic – error fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchSemantic:
    """Tests for _search_semantic()."""

    @pytest.mark.asyncio
    async def test_fallback_on_error(
        self,
        pipeline: SearchPipeline,
        mock_semantic_engine: MagicMock,
        mock_logger: Mock,
    ) -> None:
        """Returns empty list and logs warning on exception."""
        mock_semantic_engine.search.side_effect = RuntimeError("embed fail")

        result = await pipeline._search_semantic("query", 10, "proj")

        assert result[0] == []
        assert isinstance(result[1], RuntimeError)
        mock_logger.warning.assert_called_once()


@pytest.mark.unit
class TestCompleteTimestampMap:
    """_complete_timestamp_map must keep stamps it already has."""

    def test_keeps_semantic_stamps_when_one_lookup_misses(
        self, pipeline: SearchPipeline, mock_semantic_engine: MagicMock
    ) -> None:
        mock_semantic_engine.get_records_by_contents.return_value = []

        completed = pipeline._complete_timestamp_map(
            {"known": "2026-01-01T00:00:00+00:00"},
            ["known", "fts-only"],
            "proj",
        )

        assert completed == {"known": "2026-01-01T00:00:00+00:00"}
        mock_semantic_engine.get_records_by_contents.assert_called_once_with(
            "proj", ["fts-only"]
        )
        mock_semantic_engine.get_id_by_content.assert_not_called()

    def test_lookup_exception_raises_search_error(
        self, pipeline: SearchPipeline, mock_semantic_engine: MagicMock
    ) -> None:
        from reflectlog.core.exceptions import SearchError

        cause = RuntimeError("store down")
        mock_semantic_engine.get_records_by_contents.side_effect = cause

        with pytest.raises(SearchError, match="candidate timestamps") as exc_info:
            pipeline._complete_timestamp_map(
                {"known": "2026-01-01T00:00:00+00:00"},
                ["known", "fts-only"],
                "proj",
            )
        assert exc_info.value.__cause__ is cause

    def test_fills_missing_from_store(
        self, pipeline: SearchPipeline, mock_semantic_engine: MagicMock
    ) -> None:
        class _Stored:
            id = 7
            workspace_id = "proj"
            content = "fts-only"
            created_at = "2026-02-01T00:00:00+00:00"

        mock_semantic_engine.get_records_by_contents.return_value = [_Stored()]

        completed = pipeline._complete_timestamp_map(
            {"known": "2026-01-01T00:00:00+00:00"},
            ["known", "fts-only"],
            "proj",
        )

        assert completed == {
            "known": "2026-01-01T00:00:00+00:00",
            "fts-only": "2026-02-01T00:00:00+00:00",
        }
        mock_semantic_engine.get_id_by_content.assert_not_called()
