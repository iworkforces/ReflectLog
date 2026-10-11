"""Unseeded workspace recovery through real storage and manager surfaces."""

from pathlib import Path
from unittest.mock import patch

import pytest
import tantivy
from usearch.index import Index

from reflectlog.application.memory.manager import MemoryManager
from reflectlog.core.exceptions import InitializationError
from reflectlog.infrastructure.memory_store import read_memory_store_snapshot
from reflectlog.infrastructure.tantivy_engine import TantivyEngine
from reflectlog.infrastructure.usearch_engine import USearchEngine
from tests.integration.test_memory_manager_usearch import (
    MockEmbedder,
    cleanup_manager,
    create_memory_manager,
    create_usearch_config,
)
from tests.integration.test_replacement_recovery import (
    NEW,
    _abandon_without_persist,
    _crash_after_insert,
    _crash_after_tantivy_add,
)

SECOND = "Track astronomy observations in a separate notebook"


@pytest.mark.parametrize("failure_backend", ["tantivy", "vector"])
async def test_first_add_batch_staging_failure_holds_generation(
    tmp_path: Path, failure_backend: str
) -> None:
    config = create_usearch_config(str(tmp_path))
    manager, logger = create_memory_manager(config)
    engine = manager._semantic_engine
    assert isinstance(engine, USearchEngine)
    generation = manager._coordinator.read_generation(manager.workspace_id)
    with _crash_after_insert(manager), pytest.raises(Exception):
        await manager._storage_phase.execute([NEW, SECOND], {})
    _abandon_without_persist(manager)
    stages: list[str] = []
    failed = False
    original_add = TantivyEngine.add
    original_batch = USearchEngine.add_batch

    def flaky_add(self: TantivyEngine, workspace_id: str, content: str) -> None:
        nonlocal failed
        if content == SECOND and not failed:
            failed = True
            raise RuntimeError("full-text staging interrupted")
        original_add(self, workspace_id, content)

    def flaky_batch(
        self: USearchEngine,
        workspace_id: str,
        memories: list[str],
        *,
        infer: bool = True,
        vectors: list[list[float]] | None = None,
    ) -> list[str]:
        nonlocal failed
        if SECOND in memories and not failed:
            failed = True
            raise RuntimeError("vector staging interrupted")
        return original_batch(
            self, workspace_id, memories, infer=infer, vectors=vectors
        )

    fault = (
        patch.object(TantivyEngine, "add", flaky_add)
        if failure_backend == "tantivy"
        else patch.object(USearchEngine, "add_batch", flaky_batch)
    )
    with (
        fault,
        patch(
            "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
            return_value=MockEmbedder(),
        ),
    ):
        degraded = MemoryManager(config, logger, orchestration_hook=stages.append)
    try:
        assert failed
        assert "before_generation" not in stages
        assert (
            degraded._coordinator.read_generation(degraded.workspace_id) == generation
        )
        assert degraded.pending_intent_count() == 2
        pending = read_memory_store_snapshot(
            engine.config.db_path, manager.workspace_id
        ).pending_transitions
        assert sorted(item.new_content for item in pending) == sorted([NEW, SECOND])
        if failure_backend == "tantivy":
            assert sorted(degraded.get_all()) == sorted([NEW, SECOND])
        else:
            # The re-index step drops an unindexed row before re-adding it, so
            # SECOND lives only in its pending journal row until the retry.
            assert degraded.get_all() == [NEW]
    finally:
        degraded.close()

    records: list[tuple[set[int], set[int], bool]] = []

    def ordering(stage: str) -> None:
        if stage == "before_generation":
            fresh = read_memory_store_snapshot(
                engine.config.db_path, manager.workspace_id
            )
            published = Index.restore(engine.config.index_path)
            assert published is not None
            fulltext = tantivy.Index.open(
                config.tantivy_index_path_template.format(
                    workspace_id=manager.workspace_id
                )
            )
            visible = all(
                len(
                    fulltext.searcher()
                    .search(
                        query=fulltext.parse_query(
                            f'"{content}"', default_field_names=["content"]
                        ),
                        limit=5,
                    )
                    .hits
                )
                == 1
                for content in [NEW, SECOND]
            )
            records.append(
                (
                    {int(key) for key in published.keys},
                    set(fresh.contents_by_id),
                    visible,
                )
            )

    with patch(
        "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
        return_value=MockEmbedder(),
    ):
        recovered = MemoryManager(config, logger, orchestration_hook=ordering)
    try:
        assert records
        assert all(
            vector_ids == sqlite_ids and visible
            for vector_ids, sqlite_ids, visible in records
        )
        assert recovered.pending_intent_count() == 0
        assert (
            recovered._coordinator.read_generation(recovered.workspace_id) > generation
        )
        assert sorted(recovered.get_all()) == sorted([NEW, SECOND])
        recovered_engine = recovered._semantic_engine
        assert isinstance(recovered_engine, USearchEngine)
        fresh = read_memory_store_snapshot(engine.config.db_path, manager.workspace_id)
        assert {int(key) for key in recovered_engine.index.keys} == set(
            fresh.contents_by_id
        )
        for content in [NEW, SECOND]:
            assert content in await recovered.search(content)
    finally:
        cleanup_manager(recovered)


@pytest.mark.parametrize("contents", [[NEW], [NEW, SECOND]])
@pytest.mark.parametrize("seam", ["sqlite", "tantivy", "before_replace"])
async def test_first_add_crash_reopens(
    tmp_path: Path, seam: str, contents: list[str]
) -> None:
    config = create_usearch_config(str(tmp_path))
    manager, logger = create_memory_manager(config)
    engine = manager._semantic_engine
    assert isinstance(engine, USearchEngine)
    initial_generation = manager._coordinator.read_generation(manager.workspace_id)

    def crash(stage: str) -> None:
        if stage == "before_replace":
            raise RuntimeError("first publish interrupted")

    if seam == "before_replace":
        engine.publish_hook = crash
        with pytest.raises(Exception):
            await manager._storage_phase.execute(contents, {})
    else:
        injector = _crash_after_insert if seam == "sqlite" else _crash_after_tantivy_add
        with injector(manager), pytest.raises(Exception):
            await manager._storage_phase.execute(contents, {})
    assert not Path(engine.config.index_path).exists()
    assert (
        manager._coordinator.read_generation(manager.workspace_id) == initial_generation
    )
    snapshot = read_memory_store_snapshot(engine.config.db_path, manager.workspace_id)
    assert sorted(snapshot.contents_by_id.values()) == sorted(contents)
    assert len(snapshot.pending_transitions) == len(contents)
    _abandon_without_persist(manager)

    stages: list[str] = []
    records: list[tuple[set[int], set[int], list[str], bool, int, int]] = []

    def ordering(stage: str) -> None:
        stages.append(stage)
        if stage == "before_generation":
            fresh = read_memory_store_snapshot(
                engine.config.db_path, manager.workspace_id
            )
            published = Index.restore(engine.config.index_path)
            fulltext = tantivy.Index.open(
                config.tantivy_index_path_template.format(
                    workspace_id=manager.workspace_id
                )
            )
            result = fulltext.searcher().search(
                query=fulltext.parse_query("spaces", default_field_names=["content"]),
                limit=5,
            )
            all_visible = all(
                len(
                    fulltext.searcher()
                    .search(
                        query=fulltext.parse_query(
                            f'"{content}"', default_field_names=["content"]
                        ),
                        limit=5,
                    )
                    .hits
                )
                == 1
                for content in contents
            )
            records.append(
                (
                    set()
                    if published is None
                    else {int(key) for key in published.keys},
                    set(fresh.contents_by_id),
                    sorted(fresh.contents_by_id.values()),
                    all_visible,
                    len(result.hits),
                    manager._coordinator.read_generation(manager.workspace_id),
                )
            )

    with patch(
        "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
        return_value=MockEmbedder(),
    ):
        reopened = MemoryManager(config, logger, orchestration_hook=ordering)
    try:
        assert records
        for (
            published_ids,
            sqlite_ids,
            stored_contents,
            all_visible,
            spaces_hits,
            _,
        ) in records:
            assert published_ids == sqlite_ids
            assert stored_contents == sorted(contents)
            assert all_visible
            assert spaces_hits == 1
        assert records[0][-1] == initial_generation
        recovered = reopened._semantic_engine
        assert isinstance(recovered, USearchEngine)
        assert sorted(reopened.get_all()) == sorted(contents)
        for content in contents:
            assert content in await reopened.search(content)
        saved = Index.restore(recovered.config.index_path)
        assert saved is not None
        ids = set(
            read_memory_store_snapshot(
                engine.config.db_path, manager.workspace_id
            ).contents_by_id
        )
        assert {int(key) for key in saved.keys} == ids
        assert {int(key) for key in recovered.index.keys} == ids
        recovered.verify_index_integrity()
        fulltext_engine = reopened._tantivy_engine
        assert isinstance(fulltext_engine, TantivyEngine)
        for content in contents:
            assert fulltext_engine.find_by_exact_match(
                reopened.workspace_id, content
            ) == [content]
        assert reopened.pending_intent_count() == 0
        assert recovered.memory_store.list_pending_transitions() == []
        generation = reopened._coordinator.read_generation(reopened.workspace_id)
        assert generation > initial_generation
        assert stages.index("before_generation") < stages.index("after_generation")
        assert not recovered._unpublished_bootstrap
        reopened.close()
        second, _ = create_memory_manager(config)
        try:
            assert sorted(second.get_all()) == sorted(contents)
            for content in contents:
                assert second.get_id_by_content(content) in ids
                assert content in await second.search(content)
            assert second.pending_intent_count() == 0
            assert (
                second._coordinator.read_generation(second.workspace_id) == generation
            )
        finally:
            cleanup_manager(second)
    finally:
        reopened.close()


@pytest.mark.parametrize("shutdown", ["abandon", "close"])
async def test_failed_embedding_keeps_unpublished_bootstrap(
    tmp_path: Path, shutdown: str
) -> None:
    config = create_usearch_config(str(tmp_path))
    manager, _ = create_memory_manager(config)
    with _crash_after_insert(manager), pytest.raises(Exception):
        await manager._storage_phase.execute([NEW], {})
    _abandon_without_persist(manager)

    with patch.object(
        MockEmbedder, "embed_documents", side_effect=RuntimeError("offline")
    ):
        degraded, _ = create_memory_manager(config)
    engine = degraded._semantic_engine
    assert isinstance(engine, USearchEngine)
    try:
        assert degraded.get_all() == [NEW]
        assert degraded.pending_intent_count() == 1
        assert not Path(engine.config.index_path).exists()
        engine.verify_index_integrity()
    finally:
        if shutdown == "close":
            degraded.close()
        else:
            _abandon_without_persist(degraded)
    assert not Path(engine.config.index_path).exists()
    recovered, _ = create_memory_manager(config)
    try:
        assert recovered.get_all() == [NEW]
        assert NEW in await recovered.search(NEW)
        assert recovered.pending_intent_count() == 0
    finally:
        cleanup_manager(recovered)


@pytest.mark.parametrize(
    "case",
    ["no_journal", "partial", "completed", "foreign_only", "foreign_pending", "delete"],
)
def test_manager_refuses_unexplained_missing_index(tmp_path: Path, case: str) -> None:
    config = create_usearch_config(str(tmp_path))
    manager, _ = create_memory_manager(config)
    engine = manager._semantic_engine
    assert isinstance(engine, USearchEngine)
    workspace = manager.workspace_id
    store = engine.memory_store
    memory_id = store.insert(workspace, NEW)
    if case in {"partial", "foreign_pending"}:
        store.begin_add_intents(workspace, [NEW])
    if case == "partial":
        store.insert(workspace, "unexplained")
    if case == "completed":
        intents = store.begin_add_intents(workspace, [NEW])
        store.complete_replacement_transition(intents[0].id)
    if case in {"foreign_only", "foreign_pending"}:
        store.begin_add_intents("foreign", [NEW])
    if case == "delete":
        store.begin_delete_intents(workspace, [(memory_id, NEW)])
    before = read_memory_store_snapshot(engine.config.db_path, workspace)
    _abandon_without_persist(manager)
    with pytest.raises(
        InitializationError, match="USearch index is missing but SQLite has"
    ):
        create_memory_manager(config)
    after = read_memory_store_snapshot(engine.config.db_path, workspace)
    assert after.contents_by_id == before.contents_by_id
    assert not Path(engine.config.index_path).exists()
