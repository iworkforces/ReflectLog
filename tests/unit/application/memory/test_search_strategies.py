"""Unit tests for search_strategies.py - 4-step search pipeline.

Covers uncovered lines: 128-137, 295, 365, 374-378, 451, 457, 475,
493-495, 538-573, 671-683.
"""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock

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

        assert reranked == [("msg1", 0.95)]

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
        assert reranked == results


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
        assert reranked == results

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

        assert reranked == [("msg1", 0.95)]
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
