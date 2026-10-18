import asyncio
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Never

import anyio
from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import Tool
from pydantic import TypeAdapter
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.memory.search_evidence import StageOutcome
from reflectlog.application.memory.search_strategies import SearchContext, SearchResult
from reflectlog.application.memory.workspace_registry import WorkspaceRegistry
from reflectlog.core.enums import RerankerEngine, SearchComponent
from reflectlog.core.exceptions import SearchError
from reflectlog.core.search_health import SearchFailureSnapshot
from reflectlog.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from reflectlog.infrastructure.openrouter_reranker import OpenRouterReranker
from reflectlog.infrastructure.usearch_engine import USearchEngine
from tests.integration.test_memory_manager_usearch import (
    cleanup_manager,
    create_memory_manager,
    create_usearch_config,
)

pytestmark = pytest.mark.integration
SECRET = "SENTINEL-secret-exception"
QUERY = "healthprobe SENTINEL-private-query"
MEMORIES = ["healthprobe SENTINEL-private-memory", "healthprobe second document"]


def fail_search(*args: object, **kwargs: object) -> Never:
    raise OSError(SECRET)


async def fail_rerank(*args: object, **kwargs: object) -> Never:
    raise ValueError(SECRET)


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryManager]:
    monkeypatch.chdir(tmp_path)
    config = replace(
        create_usearch_config(str(tmp_path / "indexes")),
        enable_smart_replace=False,
        search_score_threshold=0.0,
    )
    active, _ = create_memory_manager(config)
    try:
        yield active
    finally:
        cleanup_manager(active)


async def health(manager: MemoryManager) -> dict[str, dict[str, int | str | None]]:
    return manager.search_failure_snapshot().to_dict()


def assert_failures(
    payload: dict[str, dict[str, int | str | None]], counts: dict[SearchComponent, int]
) -> None:
    records = payload
    assert set(records) == {component.value for component in SearchComponent}
    for component in SearchComponent:
        record = records[component.value]
        assert set(record) == {"count", "last_failure_at", "exception_type"}
        assert record["count"] == counts.get(component, 0)
        if component in counts:
            raw_timestamp = record["last_failure_at"]
            assert isinstance(raw_timestamp, str)
            timestamp = datetime.fromisoformat(raw_timestamp)
            assert timestamp.utcoffset() == timedelta(0)
            assert record["exception_type"] == (
                "OSError"
                if component in (SearchComponent.SEMANTIC, SearchComponent.TANTIVY)
                else "ValueError"
            )
        else:
            assert record == {
                "count": 0,
                "last_failure_at": None,
                "exception_type": None,
            }
    serialized = json.dumps(payload)
    for private in (SECRET, QUERY, *MEMORIES):
        assert private not in serialized


@pytest.mark.parametrize("component", list(SearchComponent))
async def test_each_real_pipeline_failure_is_private_and_counted_once(
    manager: MemoryManager, monkeypatch: pytest.MonkeyPatch, component: SearchComponent
) -> None:
    assert manager.add_memories(MEMORIES) == 2
    match component:
        case SearchComponent.SEMANTIC:
            monkeypatch.setattr(type(manager._semantic_engine), "search", fail_search)
        case SearchComponent.TANTIVY:
            assert manager._tantivy_engine is not None
            monkeypatch.setattr(type(manager._tantivy_engine), "search", fail_search)
        case SearchComponent.CROSS_ENCODER:
            manager.config = replace(
                manager.config, reranker_engine=RerankerEngine.CROSS_ENCODER
            )
            manager._search_pipeline.config = manager.config
            assert manager.cross_encoder_reranker is not None
            monkeypatch.setattr(CrossEncoderReranker, "rerank_async", fail_rerank)
        case SearchComponent.OPENROUTER_RERANKER:
            manager.config = replace(
                manager.config,
                reranker_engine=RerankerEngine.OPENROUTER,
                openrouter_rerank_model="test-model",
            )
            manager._search_pipeline.config = manager.config
            assert manager.openrouter_reranker is not None
            monkeypatch.setattr(OpenRouterReranker, "rerank_async", fail_rerank)

    results = await manager.search(QUERY)

    assert isinstance(results, list)
    assert results and all(isinstance(result, str) for result in results)
    assert_failures(await health(manager), {component: 1})


async def test_clean_workspace_has_four_zero_records(manager: MemoryManager) -> None:
    assert_failures(await health(manager), {})


@pytest.mark.parametrize(
    "mode", ["tantivy_disabled", "unconfigured", "absent", "single", "empty"]
)
async def test_benign_paths_leave_zero_records(
    manager: MemoryManager, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    assert manager.add_memories(MEMORIES[:1] if mode == "single" else MEMORIES) >= 1
    if mode == "tantivy_disabled":
        manager._search_pipeline._tantivy_engine = None
    if mode in ("absent", "single", "empty"):
        manager.config = replace(
            manager.config,
            reranker_engine=RerankerEngine.OPENROUTER,
            openrouter_rerank_model="test-model",
        )
        manager._search_pipeline.config = manager.config
        if mode == "absent":
            monkeypatch.setattr(
                MemoryManager, "openrouter_reranker", property(lambda self: None)
            )
        else:
            monkeypatch.setattr(OpenRouterReranker, "rerank_async", fail_rerank)
    if mode == "empty":
        manager.delete_memories(MEMORIES)

    results = await manager.search(QUERY)

    assert isinstance(results, list)
    assert_failures(await health(manager), {})


async def test_dual_failure_records_both_and_still_raises(
    manager: MemoryManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert manager.add_memories(MEMORIES) == 2
    assert manager._tantivy_engine is not None
    monkeypatch.setattr(type(manager._semantic_engine), "search", fail_search)
    monkeypatch.setattr(type(manager._tantivy_engine), "search", fail_search)

    with pytest.raises(SearchError):
        await manager.search(QUERY)

    assert_failures(
        await health(manager), {SearchComponent.SEMANTIC: 1, SearchComponent.TANTIVY: 1}
    )


def fail_prepare(*args: object, **kwargs: object) -> Never:
    raise OSError(SECRET)


@pytest.fixture
def lazy_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MemoryManager]:
    """A populated workspace reopened without eager initialization."""
    monkeypatch.chdir(tmp_path)
    config = replace(
        create_usearch_config(str(tmp_path / "indexes")),
        enable_smart_replace=False,
        search_score_threshold=0.0,
    )
    seeded, _ = create_memory_manager(config)
    try:
        assert seeded.add_memories(MEMORIES) == 2
    finally:
        seeded.close()
    active, _ = create_memory_manager(replace(config, eager_initialization=False))
    try:
        yield active
    finally:
        cleanup_manager(active)


async def test_lazy_semantic_init_failure_falls_back_to_fulltext_and_is_counted(
    lazy_manager: MemoryManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = lazy_manager._semantic_engine
    assert isinstance(engine, USearchEngine)
    assert engine._index is None
    monkeypatch.setattr(USearchEngine, "_prepare_index_files", fail_prepare)

    results = await lazy_manager.search(QUERY)

    assert results and all(isinstance(result, str) for result in results)
    failures = lazy_manager.search_failure_snapshot()
    assert failures.semantic.count == 1
    assert failures.semantic.exception_type == "RuntimeError"
    assert failures.tantivy.count == 0
    assert SECRET not in json.dumps(await health(lazy_manager))


async def test_lazy_dual_init_outage_raises_search_error_and_counts_both(
    lazy_manager: MemoryManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert lazy_manager._tantivy_engine is not None
    monkeypatch.setattr(USearchEngine, "_prepare_index_files", fail_prepare)
    monkeypatch.setattr(type(lazy_manager._tantivy_engine), "search", fail_search)

    with pytest.raises(SearchError):
        await lazy_manager.search(QUERY)

    failures = lazy_manager.search_failure_snapshot()
    assert failures.semantic.count == 1
    assert failures.semantic.exception_type == "RuntimeError"
    assert failures.tantivy.count == 1
    assert failures.tantivy.exception_type == "OSError"


async def test_concurrent_real_searches_have_exact_failure_total(
    manager: MemoryManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert manager.add_memories(MEMORIES) == 2
    monkeypatch.setattr(type(manager._semantic_engine), "search", fail_search)

    results = await asyncio.gather(*(manager.search(QUERY) for _ in range(40)))

    assert all(isinstance(result, list) and result for result in results)
    assert_failures(await health(manager), {SearchComponent.SEMANTIC: 40})


async def test_real_registry_idle_eviction_resets_failure_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = replace(
        create_usearch_config(str(tmp_path / "indexes")),
        enable_smart_replace=False,
        search_score_threshold=0.0,
    )
    now = [0.0]

    def factory(concrete: Config) -> MemoryManager:
        return create_memory_manager(concrete)[0]

    registry = WorkspaceRegistry(config, factory, clock=lambda: now[0], idle_ttl=1)
    try:
        async with registry.acquire(config.workspace_id) as first:
            assert first.add_memories(MEMORIES) == 2
            with monkeypatch.context() as fault:
                fault.setattr(type(first._semantic_engine), "search", fail_search)
                await first.search(QUERY)
            assert_failures(await health(first), {SearchComponent.SEMANTIC: 1})
        now[0] = 2.0
        await registry.prune()

        async with registry.acquire(config.workspace_id) as second:
            assert second is not first
            assert second.search_failure_snapshot() == SearchFailureSnapshot()
            assert_failures(await health(second), {})
    finally:
        await registry.close()


@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "concatenated",
        "semantic_failed",
        "fulltext_failed",
        "reranker_failed",
        "applied",
        "degraded",
        "unavailable",
        "single",
        "empty",
        "disabled",
    ],
)
async def test_explained_real_records_preserve_ranking(
    manager: MemoryManager, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    manager.add_memories(MEMORIES[:1] if mode == "single" else MEMORIES)
    expected_semantic = StageOutcome.APPLIED
    expected_fulltext = StageOutcome.APPLIED
    expected_reranker = StageOutcome.DISABLED
    if mode == "concatenated":
        manager.config = replace(manager.config, enable_rrf_fusion=False)
        manager._search_pipeline.config = manager.config
    if mode == "semantic_failed":
        monkeypatch.setattr(type(manager._semantic_engine), "search", fail_search)
        expected_semantic = StageOutcome.FAILED
    if mode == "fulltext_failed":
        assert manager._tantivy_engine is not None
        monkeypatch.setattr(type(manager._tantivy_engine), "search", fail_search)
        expected_fulltext = StageOutcome.FAILED
    if mode == "disabled":
        manager._search_pipeline._tantivy_engine = None
        expected_fulltext = StageOutcome.UNAVAILABLE
    if mode in (
        "reranker_failed",
        "applied",
        "degraded",
        "unavailable",
        "single",
        "empty",
    ):
        manager.config = replace(
            manager.config,
            reranker_engine=RerankerEngine.OPENROUTER,
            openrouter_rerank_model="test-model",
        )
        manager._search_pipeline.config = manager.config
        expected_reranker = StageOutcome.FAILED
        monkeypatch.setattr(OpenRouterReranker, "rerank_async", fail_rerank)
        if mode == "applied":

            async def applied(
                _self: OpenRouterReranker,
                query: str,
                candidates: list[tuple[str, float]],
                *args: object,
                **kwargs: object,
            ) -> list[tuple[str, float]]:
                return [
                    (content, 0.9 - index * 0.1)
                    for index, (content, _) in enumerate(reversed(candidates))
                ]

            monkeypatch.setattr(OpenRouterReranker, "rerank_async", applied)
            expected_reranker = StageOutcome.APPLIED
        if mode == "degraded":

            async def degraded(*args: object, **kwargs: object) -> None:
                return None

            monkeypatch.setattr(OpenRouterReranker, "rerank_async", degraded)
            expected_reranker = StageOutcome.DEGRADED
        if mode == "unavailable":
            monkeypatch.setattr(
                MemoryManager, "openrouter_reranker", property(lambda self: None)
            )
            expected_reranker = StageOutcome.UNAVAILABLE
        if mode == "single":
            expected_reranker = StageOutcome.SKIPPED_SINGLE_HIT
        if mode == "empty":
            manager.delete_memories(MEMORIES)
            expected_semantic = expected_fulltext = expected_reranker = (
                StageOutcome.EMPTY
            )
    plain = await manager.search_records(QUERY)

    explained = await manager.search_records(QUERY, explain=True)

    assert [
        (row.id, row.content, row.score, row.created_at) for row in explained.records
    ] == [(row.id, row.content, row.score, row.created_at) for row in plain.records]
    assert explained.score_kind == plain.score_kind
    assert plain.stage_outcomes is None
    outcomes = explained.stage_outcomes
    assert outcomes is not None
    assert (outcomes.semantic, outcomes.fulltext, outcomes.reranker) == (
        expected_semantic,
        expected_fulltext,
        expected_reranker,
    )
    for row in explained.records:
        assert row.explanation is not None
        assert row.explanation.final_score_kind == explained.score_kind
        for contribution in (row.explanation.semantic, row.explanation.fulltext):
            assert contribution is None or contribution.rank >= 1


@pytest.mark.parametrize("delete_all", [False, True])
async def test_explained_records_drop_deleted_pipeline_hit(
    manager: MemoryManager, monkeypatch: pytest.MonkeyPatch, delete_all: bool
) -> None:
    manager.add_memories(MEMORIES)
    execute = manager._search_pipeline.execute
    captured: list[SearchResult] = []

    async def delete_after_search(
        context: SearchContext,
        *,
        include_timestamp_map: bool = True,
        explain: bool = False,
    ) -> SearchResult:
        result = await execute(
            context, include_timestamp_map=include_timestamp_map, explain=explain
        )
        captured.append(result)
        manager.delete_memories(result.memories if delete_all else [result.memories[0]])
        return result

    monkeypatch.setattr(manager._search_pipeline, "execute", delete_after_search)

    result = await manager.search_records(QUERY, explain=True)

    pipeline = captured[0]
    assert pipeline.explanations is not None
    assert len(result.records) == (0 if delete_all else 1)
    if not delete_all:
        assert result.records[0].content == pipeline.memories[1]
        assert (
            result.records[0].explanation == pipeline.explanations[pipeline.memories[1]]
        )


async def test_overlapping_explanations_are_request_local(
    manager: MemoryManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager.add_memories(MEMORIES)
    manager.config = replace(
        manager.config,
        reranker_engine=RerankerEngine.OPENROUTER,
        openrouter_rerank_model="test-model",
    )
    manager._search_pipeline.config = manager.config
    entered = anyio.Event()
    release = anyio.Event()
    outcomes: dict[str, StageOutcome] = {}

    async def rerank(
        _self: OpenRouterReranker, query: str, *args: object, **kwargs: object
    ) -> None:
        if query == QUERY:
            entered.set()
            await release.wait()
            return None
        raise ValueError(SECRET)

    async def search(query: str) -> None:
        result = await manager.search_records(query, explain=True)
        assert result.stage_outcomes is not None
        outcomes[query] = result.stage_outcomes.reranker

    monkeypatch.setattr(OpenRouterReranker, "rerank_async", rerank)

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(search, QUERY)
            await entered.wait()
            await search("healthprobe other")
            release.set()

    assert outcomes == {
        QUERY: StageOutcome.DEGRADED,
        "healthprobe other": StageOutcome.FAILED,
    }


async def test_registered_mcp_search_explanation(
    manager: MemoryManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(_self: FastMCPServer, config: Config) -> MemoryManager:
        return create_memory_manager(config)[0]

    monkeypatch.setattr(FastMCPServer, "_create_manager", factory)
    server = FastMCPServer(manager.config)
    try:
        async with server._registry.acquire(manager.workspace_id) as active:
            active.add_memories(MEMORIES)
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            search_tool = next(tool for tool in tools if tool.name == "search_memories")
            assert isinstance(search_tool, Tool)
            schema = search_tool.input_schema
            assert schema["properties"]["explain"]["default"] is False
            assert schema["properties"]["explain"]["type"] == "boolean"
            arguments: dict[str, object] = {
                "query": QUERY,
                "workspace_id": manager.workspace_id,
            }
            plain = (
                await client.call_tool("search_memories", arguments)
            ).structured_content
            explained = (
                await client.call_tool(
                    "search_memories", {**arguments, "explain": True}
                )
            ).structured_content
            assert plain is not None and explained is not None
            assert "stage_outcomes" not in plain
            assert explained["stage_outcomes"] == {
                "semantic": "applied",
                "fulltext": "applied",
                "reranker": "disabled",
            }
            rows = TypeAdapter(list[dict[str, object]]).validate_python(
                explained["results"]
            )
            assert [
                {key: value for key, value in row.items() if key != "explanation"}
                for row in rows
            ] == plain["results"]
            assert all("explanation" in row for row in rows)
            with pytest.raises(ToolError, match="validation"):
                await client.call_tool("search_memories", {**arguments, "limit": 0})
    finally:
        await server.aclose()
