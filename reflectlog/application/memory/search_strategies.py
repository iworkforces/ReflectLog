"""Search pipeline strategies for hybrid semantic + full-text search.

This module extracts the search logic from MemoryManager into a separate
concern-focused module. It implements the 4-step search pipeline:
1. Parallel Search (USearch + Tantivy)
2. RRF Fusion or Concatenation
3. Fusion Threshold Filtering (when RRF enabled)
4. Reranking (CrossEncoder)
"""

from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from reflectlog.infrastructure.cross_encoder_reranker import CrossEncoderReranker
    from reflectlog.infrastructure.tantivy_engine import TantivyEngine

    from ...core.logging import IStructuredLogger
    from ...core.types import ISemanticSearchEngine
    from ..config.settings import Config
    from .fusion.base import FusionEngine

from asyncer import (
    asyncify,
    create_task_group,
)

from reflectlog.core.enums import RerankerEngine
from reflectlog.core.exceptions import SearchError

from ..utils.logging import format_fusion_score_status

# Constants for magic numbers (documented for maintainability)
MIN_OVERFETCH_LIMIT = 8  # Floor so tiny limits still have fusion diversity
TANTIVY_SCORE_DIVISOR = 10.0  # Tantivy BM25 scores typically 0-10+, normalize to 0-1
LOG_QUERY_TRUNCATE_LENGTH = 100


@dataclass
class SearchContext:
    """Context object for search pipeline execution.

    Attributes:
        query: The search query string.
        limit: Maximum number of results to return.
        overfetch_limit: Number of candidates to fetch for better fusion quality.
        enable_rrf_fusion: Whether RRF fusion is enabled (vs concatenation).
        reranker_engine: The reranker engine to use ("cross_encoder" or "none").
        workspace_id: Workspace identifier for logging.
    """

    query: str
    limit: int
    overfetch_limit: int
    enable_rrf_fusion: bool
    reranker_engine: RerankerEngine | str
    workspace_id: str


@dataclass
class SearchResult:
    """Result of search pipeline execution.

    Attributes:
        memories: List of memory strings.
        timestamp_map: Mapping from memory to created_at timestamp.
        semantic_results: Original semantic search results.
        tantivy_results: Original Tantivy search results.
    """

    memories: list[str]
    timestamp_map: dict[str, str]
    semantic_results: list[tuple[str, float, str]]
    tantivy_results: list[tuple[str, float]]


class RerankerProvider(Protocol):
    @property
    def cross_encoder_reranker(self) -> CrossEncoderReranker | None: ...


class SearchPipeline:
    """Hybrid search pipeline orchestrator.

    Implements a 4-step search pipeline:
    1. Parallel semantic + full-text search
    2. RRF fusion or result concatenation
    3. Fusion threshold filtering (when RRF enabled)
    4. Reranking (CrossEncoder)

    Thread-safe: Uses MemoryManager's RLock for state changes.
    """

    logger: IStructuredLogger

    def __init__(
        self,
        semantic_engine: ISemanticSearchEngine,
        tantivy_engine: TantivyEngine | None,
        fusion_engine: FusionEngine,
        config: Config,
        logger: IStructuredLogger | None,
        memory_manager: RerankerProvider | None,
    ) -> None:
        """Initialize search pipeline.

        Args:
            semantic_engine: USearchEngine for semantic search.
            tantivy_engine: TantivyEngine for full-text search (optional).
            fusion_engine: RanxFusionEngine for result fusion.
            config: Application configuration.
            logger: Structured logger instance.
            memory_manager: MemoryManager instance for lazy reranker fetching (optional).
        """
        super().__init__()
        if logger is None:
            raise ValueError("logger is required")

        self._semantic_engine = semantic_engine
        self._tantivy_engine = tantivy_engine
        self._fusion_engine = fusion_engine
        self.config = config
        self.logger = logger
        self._memory_manager = memory_manager

    async def execute(self, context: SearchContext) -> SearchResult:
        """Execute the full search pipeline.

        Args:
            context: Search context with query, limit, and configuration.

        Returns:
            SearchResult with memories and metadata.

        Raises:
            SearchError: If search operation fails.

        Note:
            Cancelling this await does not abort native USearch or Tantivy
            work already running in a worker thread.
        """
        try:
            return await self._execute_hybrid_search(context)

        except SearchError:
            raise
        except Exception as e:
            self.logger.error(
                "Search pipeline failed",
                extra={
                    "workspace_id": context.workspace_id,
                    "query_length": len(context.query),
                    "error": str(e),
                },
            )
            raise SearchError(f"Failed to execute search: {e}") from e

    async def _execute_hybrid_search(self, context: SearchContext) -> SearchResult:
        """Execute 4-step hybrid search pipeline."""
        # Step 1: Parallel Search
        (
            semantic_results,
            tantivy_results,
            semantic_error,
            tantivy_error,
        ) = await self._step1_parallel_search(context)
        semantic_results = self._filter_semantic_threshold(semantic_results)
        tantivy_results = await asyncify(self._filter_live_hits)(
            tantivy_results, context.workspace_id
        )
        if semantic_error is not None and not tantivy_results:
            raise SearchError(
                f"Failed to execute search: {semantic_error}"
            ) from semantic_error
        if tantivy_error is not None and not semantic_results:
            raise SearchError(
                f"Failed to execute search: {tantivy_error}"
            ) from tantivy_error

        timestamp_map: dict[str, str] = {
            msg: created_at for msg, _, created_at in semantic_results
        }

        if self.config.log_search_results_verbose:
            self._log_search_results(context, semantic_results, tantivy_results)

        # Step 2: RRF Fusion or Concatenation
        hybrid_results = await self._step2_fusion_or_concatenate(
            context, semantic_results, tantivy_results
        )
        hybrid_results = await asyncify(self._filter_live_hits)(
            hybrid_results, context.workspace_id
        )

        # Step 3: Fusion threshold applies only to fused (2+ backend) RRF output.
        # A single unfused list keeps backend scores (cosine/BM25), which must
        # not be compared to fusion_ranking_threshold=0.8.
        backends_used = int(bool(semantic_results)) + int(bool(tantivy_results))
        if context.enable_rrf_fusion and backends_used >= 2:
            hybrid_results = await self._step3_fusion_threshold(
                context, hybrid_results, backends_used
            )
            # Handle case where all results were filtered out
            if not hybrid_results:
                return SearchResult(
                    memories=[],
                    timestamp_map={},
                    semantic_results=semantic_results,
                    tantivy_results=tantivy_results,
                )

        # Step 4: Reranking (or skip if 0-1 results)
        rerank_step_num = 4 if context.enable_rrf_fusion else 3
        if len(hybrid_results) <= 1:
            self._log_skip_reranking(len(hybrid_results), rerank_step_num)
        else:
            timestamp_map = await asyncify(self._complete_timestamp_map)(
                timestamp_map,
                [msg for msg, _ in hybrid_results],
                context.workspace_id,
            )
            hybrid_results = await self._step4_reranking(
                context, hybrid_results, timestamp_map, rerank_step_num
            )

        # Extract final memories
        memories = [msg for msg, _ in hybrid_results[: context.limit]]

        # Final summary
        self.logger.info(
            f"SEARCH COMPLETE: {len(memories)} result(s) returned",
            extra={
                "workspace_id": context.workspace_id,
                "query_length": len(context.query),
                "result_count": len(memories),
                "top_score": hybrid_results[0][1] if hybrid_results else 0.0,
            },
        )

        return SearchResult(
            memories=memories,
            timestamp_map=timestamp_map,
            semantic_results=semantic_results,
            tantivy_results=tantivy_results,
        )

    async def _step1_parallel_search(
        self, context: SearchContext
    ) -> tuple[
        list[tuple[str, float, str]],
        list[tuple[str, float]],
        Exception | None,
        Exception | None,
    ]:
        """Step 1: Execute parallel search on both engines."""
        self.logger.info(
            "STEP 1: Executing parallel search engines...",
            extra={"step": "parallel_search"},
        )

        # asyncer task group required: soonify captures SoonValue results
        # from parallel USearch + Tantivy searches on worker threads.
        soon_semantic = None
        soon_tantivy = None
        async with create_task_group() as tg:
            soon_semantic = tg.soonify(self._search_semantic)(
                context.query, context.overfetch_limit, context.workspace_id
            )
            soon_tantivy = tg.soonify(self._search_tantivy)(
                context.query, context.overfetch_limit, context.workspace_id
            )

        assert soon_semantic is not None
        assert soon_tantivy is not None

        semantic_results, semantic_error = soon_semantic.value
        tantivy_results, tantivy_error = soon_tantivy.value

        self.logger.info(
            f"Both engines completed (semantic: {len(semantic_results)}, tantivy: {len(tantivy_results)})",
            extra={
                "workspace_id": context.workspace_id,
                "query_length": len(context.query),
                "semantic_count": len(semantic_results),
                "tantivy_count": len(tantivy_results),
            },
        )

        return semantic_results, tantivy_results, semantic_error, tantivy_error

    async def _search_semantic(
        self, query: str, limit: int, workspace_id: str
    ) -> tuple[list[tuple[str, float, str]], Exception | None]:
        """Execute semantic search on USearchEngine.

        Returns (results, error). A failed search yields [] plus the error
        so the caller can raise if every backend failed.
        """
        try:
            # Class-defined is_ready (MRO walk) so MagicMock auto-attrs
            # cannot skip init, but subclasses that inherit it still work.
            if not self._semantic_engine.is_ready():
                await asyncify(self._semantic_engine.ensure_initialized)()
            results: list[tuple[str, float, str]] = await asyncify(
                self._semantic_engine.search
            )(
                query=query,
                workspace_id=workspace_id,
                limit=limit,
            )
            return results, None
        except Exception as e:
            self.logger.warning(
                "Semantic search failed - falling back to Tantivy full-text only",
                extra={
                    "workspace_id": workspace_id,
                    "query_length": len(query),
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "fallback_behavior": "empty_semantic_results",
                    "note": "Search will continue with Tantivy results only",
                },
            )
            return [], e

    async def _search_tantivy(
        self, query: str, limit: int, workspace_id: str
    ) -> tuple[list[tuple[str, float]], Exception | None]:
        """Execute full-text search on Tantivy engine."""
        if self._tantivy_engine is None:
            return [], None
        try:
            if not self._tantivy_engine.is_ready():
                await asyncify(self._tantivy_engine.ensure_initialized)()
            tantivy_results: list[tuple[str, float]] = await asyncify(
                self._tantivy_engine.search
            )(query, workspace_id, limit)
            return tantivy_results, None
        except Exception as e:
            self.logger.warning(
                "Tantivy search failed - continuing with semantic results only",
                extra={
                    "workspace_id": workspace_id,
                    "query_length": len(query),
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            return [], e

    async def _step2_fusion_or_concatenate(
        self,
        context: SearchContext,
        semantic_results: list[tuple[str, float, str]],
        tantivy_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Step 2: Combine results using RRF fusion or concatenation."""
        # Convert semantic_results from 3-tuples to 2-tuples for fusion
        semantic_results_2tuple = [(msg, score) for msg, score, _ in semantic_results]

        if context.enable_rrf_fusion:
            return await self._fuse_rrf(
                context, semantic_results_2tuple, tantivy_results
            )
        else:
            return self._concatenate_results(
                semantic_results_2tuple,
                tantivy_results,
                context.overfetch_limit,
                user_limit=context.limit,
            )

    async def _fuse_rrf(
        self,
        context: SearchContext,
        semantic_results: list[tuple[str, float]],
        tantivy_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Fuse results using RRF algorithm."""
        self.logger.info(
            "STEP 2: RRF Fusion (combining results)...",
            extra={"step": "fusion", "mode": "rrf"},
        )

        hybrid_results = await asyncify(self._fusion_engine.fuse)(
            semantic_results, tantivy_results
        )

        if hybrid_results:
            top_rrf = hybrid_results[0][1] if hybrid_results else 0.0
            self.logger.info(
                f"Combined {len(hybrid_results)} unique result(s) using {self._fusion_engine.method.upper()} algorithm",
                extra={
                    "workspace_id": context.workspace_id,
                    "query_length": len(context.query),
                    "engine": "fusion",
                    "result_count": len(hybrid_results),
                    "method": self._fusion_engine.method,
                    "top_rrf_score": top_rrf,
                },
            )

        return hybrid_results

    def _concatenate_results(
        self,
        semantic_results: list[tuple[str, float]],
        tantivy_results: list[tuple[str, float]],
        limit: int,
        user_limit: int | None = None,
    ) -> list[tuple[str, float]]:
        """Concatenate semantic + tantivy results without RRF fusion."""
        self.logger.info(
            "STEP 2: Concatenate results (RRF fusion disabled)...",
            extra={"step": "concatenate", "mode": "concatenate"},
        )

        seen_memories: set[str] = set()
        combined: list[tuple[str, float]] = []

        page_limit = user_limit if user_limit is not None else limit
        reserved_lexical = 1 if tantivy_results else 0
        semantic_budget = max(0, page_limit - reserved_lexical)

        for msg, score in semantic_results:
            if len(combined) >= semantic_budget:
                break
            if msg not in seen_memories:
                seen_memories.add(msg)
                combined.append((msg, score))

        semantic_count = len(combined)

        for msg, score in tantivy_results:
            if len(combined) >= page_limit:
                break
            if msg not in seen_memories:
                seen_memories.add(msg)
                combined.append((msg, score))

        if limit > page_limit:
            for msg, score in semantic_results:
                if len(combined) >= limit:
                    break
                if msg not in seen_memories:
                    seen_memories.add(msg)
                    combined.append((msg, score))
            for msg, score in tantivy_results:
                if len(combined) >= limit:
                    break
                if msg not in seen_memories:
                    seen_memories.add(msg)
                    combined.append((msg, score))

        for msg, score in semantic_results:
            if len(combined) >= limit:
                break
            if msg not in seen_memories:
                seen_memories.add(msg)
                combined.append((msg, score))

        tantivy_added = len(combined) - semantic_count
        duplicates_skipped = len(tantivy_results) - tantivy_added

        self.logger.info(
            f"Combined {len(combined)} result(s): {semantic_count} semantic + {tantivy_added} tantivy ({duplicates_skipped} duplicates skipped)",
            extra={
                "semantic_count": semantic_count,
                "tantivy_added": tantivy_added,
                "duplicates_skipped": duplicates_skipped,
                "total_count": len(combined),
            },
        )

        return combined

    async def _step3_fusion_threshold(
        self,
        context: SearchContext,
        results: list[tuple[str, float]],
        n_backends: int = 2,
    ) -> list[tuple[str, float]]:
        """Step 3: Filter by fusion threshold."""
        threshold = self._effective_fusion_threshold(results, n_backends)
        self.logger.info(
            f"STEP 3: Filtering (threshold >= {threshold})...",
            extra={
                "step": "filtering",
                "threshold": threshold,
            },
        )

        return self._filter_by_fusion_threshold(results, threshold, context.query)

    def _effective_fusion_threshold(
        self, results: list[tuple[str, float]], n_backends: int = 2
    ) -> float:
        """Ignore leftover 0-1 gates when the fused scores are raw RRF."""
        try:
            threshold = float(self.config.fusion_ranking_threshold)
            k = max(1, int(self.config.fusion_rrf_k))
        except TypeError, ValueError:
            return 0.0
        if not results:
            return threshold
        max_score = max(score for _msg, score in results)
        weights = self.config.fusion_weights
        weight_sum = 0.0
        if isinstance(weights, list) and weights:
            try:
                for weight in cast("list[object]", weights):
                    if not isinstance(weight, (int, float)):
                        raise TypeError("fusion weight is not numeric")
                    weight_sum += float(weight)
            except TypeError, ValueError:
                weight_sum = 0.0
        if weight_sum <= 0.0:
            weight_sum = float(max(2, n_backends))
        max_raw = weight_sum / (k + 1)
        if threshold > max_raw and max_score <= max_raw:
            self.logger.warning(
                "Ignoring leftover 0-1 fusion_ranking_threshold against raw RRF",
                extra={"configured": threshold, "max_raw_rrf": max_raw},
            )
            return 0.0
        return threshold

    def _filter_by_fusion_threshold(
        self, results: list[tuple[str, float]], threshold: float, query: str
    ) -> list[tuple[str, float]]:
        """Filter fusion results by fusion score threshold."""
        if not results:
            return results

        # Apply threshold filter
        filtered = [(msg, score) for msg, score in results if score >= threshold]
        filtered_out = len(results) - len(filtered)

        # Log summary
        self.logger.info(
            f"Kept {len(filtered)}/{len(results)} result(s), filtered {filtered_out}",
            extra={
                "fusion_threshold": threshold,
                "kept_count": len(filtered),
                "filtered_count": filtered_out,
            },
        )

        if not self.config.log_search_results_verbose:
            return filtered

        # Show filtered results with status
        for idx, (_memory, score) in enumerate(results[: min(5, len(results))], 1):
            status, interpretation = format_fusion_score_status(score, threshold)
            status_indicator = "[KEEP]" if score >= threshold else "[FILTER]"
            self.logger.info(
                f"[{idx}] {status_indicator} score={score:.4f} ({interpretation})",
                extra={"fusion_score": score, "fusion_status": status},
            )

        return filtered

    def _get_cross_encoder_reranker(self) -> CrossEncoderReranker | None:
        if (
            self._memory_manager is None
            or self.config.reranker_engine != RerankerEngine.CROSS_ENCODER
        ):
            return None
        return self._memory_manager.cross_encoder_reranker

    def _get_reranker(
        self,
    ) -> tuple[RerankerEngine, CrossEncoderReranker] | tuple[None, None]:
        cross_encoder_reranker = self._get_cross_encoder_reranker()
        if cross_encoder_reranker is not None:
            return (RerankerEngine.CROSS_ENCODER, cross_encoder_reranker)

        return (None, None)

    async def _step4_reranking(
        self,
        context: SearchContext,
        results: list[tuple[str, float]],
        timestamp_map: dict[str, str],
        step_num: int,
    ) -> list[tuple[str, float]]:
        """Step 4: Rerank results using the cross-encoder when enabled."""
        match self._get_reranker():
            case ("cross_encoder", cross_encoder_reranker):
                return await self._rerank_cross_encoder(
                    context,
                    results,
                    step_num,
                    cross_encoder_reranker,
                    timestamp_map=timestamp_map,
                )
            case _:
                return results

    async def _rerank_cross_encoder(
        self,
        context: SearchContext,
        results: list[tuple[str, float]],
        step_num: int,
        cross_encoder_reranker: CrossEncoderReranker | None = None,
        timestamp_map: dict[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        """Rerank using CrossEncoder."""
        # Use provided reranker or fetch via _get_reranker
        if cross_encoder_reranker is None:
            cross_encoder_reranker = self._get_cross_encoder_reranker()
            if cross_encoder_reranker is None:
                return results

        self.logger.info(
            f"STEP {step_num}: CrossEncoder Reranking ({len(results)} candidates)...",
            extra={
                "step": "reranking",
                "step_num": step_num,
                "engine": "cross_encoder",
                "candidate_count": len(results),
            },
        )

        rerank_start = time.time()
        pre_rerank_count = len(results)

        try:
            if timestamp_map is None:
                reranked_results = await cross_encoder_reranker.rerank_async(
                    context.query, results, top_k=self._cross_encoder_top_k(context)
                )
            else:
                reranked_results = await cross_encoder_reranker.rerank_async(
                    context.query,
                    results,
                    timestamp_map,
                    top_k=self._cross_encoder_top_k(context),
                )
        except Exception as exc:
            self.logger.warning(
                "CrossEncoder failed; returning fused results",
                extra={"error": str(exc), "candidate_count": len(results)},
            )
            return results
        results = reranked_results

        rerank_duration = (time.time() - rerank_start) * 1000

        self.logger.info(
            f"Kept {len(results)}/{pre_rerank_count} result(s) after CrossEncoder scoring",
            extra={
                "kept_count": len(results),
                "filtered_count": pre_rerank_count - len(results),
                "model": self.config.cross_encoder_model,
            },
        )
        self.logger.info(
            f"CrossEncoder reranking completed in {rerank_duration:.0f}ms",
            extra={"rerank_duration_ms": rerank_duration},
        )

        return results

    def _log_skip_reranking(self, result_count: int, step_num: int) -> None:
        """Log when reranking is skipped due to 0-1 results."""
        if result_count == 1:
            self.logger.info(
                f"STEP {step_num}: Reranking skipped (single result - reranking unnecessary)",
                extra={
                    "step": "reranking_skip",
                    "step_num": step_num,
                    "result_count": 1,
                    "reason": "single result - reranking unnecessary",
                },
            )

    def _log_search_results(
        self,
        context: SearchContext,
        semantic_results: list[tuple[str, float, str]],
        tantivy_results: list[tuple[str, float]],
    ) -> None:
        """Log search results from both engines."""
        self.logger.info("─" * 50, extra={"section": "search_engines"})

        # Log USearch results
        self.logger.info("USEARCH SEMANTIC ENGINE:", extra={"engine": "usearch"})
        if semantic_results:
            top_score = semantic_results[0][1] if semantic_results else 0.0
            self.logger.info(
                f"Found {len(semantic_results)} result(s), best score: {top_score:.4f}",
                extra={"result_count": len(semantic_results), "top_score": top_score},
            )
            for idx, (_memory, score, _) in enumerate(
                semantic_results[: min(3, len(semantic_results))], 1
            ):
                self.logger.info(
                    f"[{idx}] score={score:.4f}",
                    extra={"result_index": idx, "score": score},
                )
        else:
            self.logger.info("No results found", extra={"result_count": 0})

        # Log Tantivy results
        self.logger.info("TANTIVY FULL-TEXT ENGINE:", extra={"engine": "tantivy"})
        if tantivy_results:
            top_score = tantivy_results[0][1] if tantivy_results else 0.0
            self.logger.info(
                f"Found {len(tantivy_results)} result(s), best BM25 score: {top_score:.4f}",
                extra={"result_count": len(tantivy_results), "top_score": top_score},
            )
            for idx, (_memory, score) in enumerate(
                tantivy_results[: min(3, len(tantivy_results))], 1
            ):
                self.logger.info(
                    f"[{idx}] score={score:.4f}",
                    extra={"result_index": idx, "score": score},
                )
        else:
            self.logger.info("No results found", extra={"result_count": 0})

        self.logger.info("─" * 50, extra={"section": "fusion"})

    def _complete_timestamp_map(
        self,
        timestamp_map: dict[str, str],
        contents: list[str],
        workspace_id: str,
    ) -> dict[str, str]:
        """Resolve created_at for every candidate. Empty map disables recency."""
        completed = {
            content: stamp for content, stamp in timestamp_map.items() if stamp
        }
        missing = [content for content in contents if content not in completed]
        if not missing:
            return completed

        try:
            records = self._semantic_engine.get_records_by_contents(
                workspace_id, missing
            )
        except Exception as exc:
            raise SearchError("Failed to load candidate timestamps") from exc
        for stored in records:
            created_at = stored.created_at
            if not created_at:
                continue
            completed[stored.content] = created_at
        return completed

    def _filter_live_hits(
        self,
        results: list[tuple[str, float]],
        workspace_id: str,
    ) -> list[tuple[str, float]]:
        """Drop FTS hits that are no longer in the SQLite source of truth."""
        if not results:
            return results
        try:
            store = self._semantic_engine.memory_store
            present = _string_set(
                store.exists_many(workspace_id, [memory for memory, _score in results])
            )
        except AttributeError, TypeError:
            return []
        if present is None:
            return []
        return [(memory, score) for memory, score in results if memory in present]

    def _filter_semantic_threshold(
        self, results: list[tuple[str, float, str]]
    ) -> list[tuple[str, float, str]]:
        """Drop USearch hits below SEARCH_SCORE_THRESHOLD before fusion."""
        try:
            threshold = float(self.config.search_score_threshold)
        except TypeError, ValueError:
            return results
        if threshold <= 0:
            return results
        return [
            (memory, score, created_at)
            for memory, score, created_at in results
            if score >= threshold
        ]

    def _cross_encoder_top_k(self, context: SearchContext) -> int:
        """CE window is at least the user limit and the configured top_k."""
        try:
            configured = int(self.config.cross_encoder_top_k)
        except TypeError, ValueError:
            configured = context.limit
        return max(1, configured, context.limit)


def _string_set(raw: object) -> set[str] | None:
    """Return a string set, or None when the value is not a real content set."""
    if not isinstance(raw, set):
        return None
    members: list[object] = list(cast("set[object]", raw))
    contents: set[str] = set()
    for item in members:
        if not isinstance(item, str):
            return None
        contents.add(item)
    return contents


def calculate_adaptive_overfetch(limit: int, index_size: int, config: Config) -> int:
    """Calculate adaptive overfetch limit based on index size.

    For small indexes, we use a higher multiplier to ensure diversity.
    For large indexes, we use a lower multiplier since there are enough
    documents for good fusion quality.

    The multiplier scales logarithmically:
    - Index size <= 100: max multiplier (3.0x default)
    - Index size >= 10,000: min multiplier (1.5x default)
    - Between: logarithmic interpolation

    Args:
        limit: The base search limit.
        index_size: Current size of the search index.
        config: Application configuration.

    Returns:
        Calculated overfetch limit (minimum MIN_OVERFETCH_LIMIT).
    """
    # If adaptive is disabled or index is empty, use static multiplier
    if not config.overfetch_adaptive or index_size == 0:
        return max(limit * config.overfetch_multiplier, MIN_OVERFETCH_LIMIT)

    # Bounds for index size (log scale interpolation)
    small_index = 100  # At or below this: use max multiplier
    large_index = 10000  # At or above this: use min multiplier

    min_mult = config.overfetch_min_multiplier
    max_mult = max(config.overfetch_max_multiplier, float(config.overfetch_multiplier))

    if index_size <= small_index:
        multiplier = max_mult
    elif index_size >= large_index:
        multiplier = min_mult
    else:
        # Logarithmic interpolation between bounds
        log_small = math.log(small_index)
        log_large = math.log(large_index)
        log_current = math.log(index_size)

        # Normalize to [0, 1] range
        t = (log_current - log_small) / (log_large - log_small)

        # Linear interpolation from max_mult to min_mult
        multiplier = max_mult - t * (max_mult - min_mult)

    overfetch_limit = int(limit * multiplier)
    return max(overfetch_limit, MIN_OVERFETCH_LIMIT)
