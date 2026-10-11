import asyncio
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Never

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import MemoryManager
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
