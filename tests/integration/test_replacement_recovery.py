"""Integration tests for restart-safe smart replacement.

Each crash-injection case records a durable SQLite transition, stops at a
persistence boundary, then constructs a fresh MemoryManager over the same
files to prove convergence to one active replacement plus an audit row.
"""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
import tempfile
from unittest.mock import patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.add_phases import ReplacementInfo
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.core.exceptions import StorageError
from reflectlog.infrastructure.memory_store import MemoryStore
from reflectlog.infrastructure.tantivy_engine import TantivyEngine
from reflectlog.infrastructure.usearch_engine import USearchEngine
from tests.integration.test_memory_manager_usearch import (
    cleanup_manager,
    create_memory_manager,
    create_usearch_config,
)

OLD = "Prefer tabs for indentation in this repository"
NEW = "Prefer spaces for indentation in this repository"

Injector = Callable[[MemoryManager], AbstractContextManager[None]]


def _replacement() -> ReplacementInfo:
    return ReplacementInfo(
        old_memory=OLD,
        new_memory=NEW,
        confidence=0.93,
        reason="updated convention",
        similarity_score=0.88,
    )


def _config(tmpdir: str) -> Config:
    return create_usearch_config(tmpdir)


@contextmanager
def _manager(tmpdir: str) -> Generator[MemoryManager]:
    manager, _logger = create_memory_manager(_config(tmpdir))
    try:
        yield manager
    finally:
        cleanup_manager(manager)


def _abandon_without_persist(manager: MemoryManager) -> None:
    """Drop in-memory indexes without commit, like a SIGKILL."""
    engine = manager._semantic_engine
    store = engine.memory_store
    store.close()
    if isinstance(engine, USearchEngine) and engine._index is not None:
        engine._index = None
    tantivy = manager._tantivy_engine
    if not isinstance(tantivy, TantivyEngine):
        return
    if tantivy._writer is not None:
        tantivy._writer = None
    tantivy._searcher = None
    tantivy._index = None


def _assert_converged(manager: MemoryManager) -> None:
    memories = manager.get_all()
    assert memories == [NEW]
    new_id = manager.get_id_by_content(NEW)
    assert new_id is not None
    engine = manager._semantic_engine
    assert isinstance(engine, USearchEngine)
    index = engine.index
    assert index is not None
    assert new_id in index
    assert manager.get_id_by_content(OLD) is None

    store = manager._semantic_engine.memory_store
    assert isinstance(store, MemoryStore)
    archives = store.get_archived(manager.workspace_id)
    assert len(archives) == 1
    assert archives[0].content == OLD
    assert archives[0].replaced_by == NEW
    assert archives[0].reason == "updated convention"
    assert archives[0].confidence == 0.93
    assert store.list_pending_transitions() == []

    tantivy = manager._tantivy_engine
    if tantivy is not None:
        new_hits = tantivy.find_by_exact_match(manager.workspace_id, NEW)
        assert NEW in new_hits
        assert len(set(new_hits)) == 1
        assert not tantivy.find_by_exact_match(manager.workspace_id, OLD)


async def _replace(manager: MemoryManager) -> None:
    result = await manager._storage_phase.execute(
        [NEW],
        {NEW: [_replacement()]},
    )
    assert result.stored_count == 1
    assert result.replaced_count == 1
    assert result.replacements[0].old_memory == OLD


@contextmanager
def _crash_after_transition(_manager: MemoryManager) -> Generator[None]:
    def boom(self: USearchEngine, memory_id: str) -> None:
        raise RuntimeError("crash after transition")

    with patch.object(USearchEngine, "delete", boom):
        yield


@contextmanager
def _crash_after_semantic_delete(_manager: MemoryManager) -> Generator[None]:
    original = USearchEngine.delete

    def boom(self: USearchEngine, memory_id: str) -> None:
        original(self, memory_id)
        raise RuntimeError("crash after semantic delete")

    with patch.object(USearchEngine, "delete", boom):
        yield


@contextmanager
def _crash_after_tantivy_delete(_manager: MemoryManager) -> Generator[None]:
    original = TantivyEngine.delete_batch

    def boom(
        self: TantivyEngine,
        workspace_id: str,
        contents: list[str],
        *,
        verify_exists: bool = True,
    ) -> int:
        deleted = original(self, workspace_id, contents, verify_exists=verify_exists)
        raise RuntimeError("crash after tantivy delete")
        return deleted

    with patch.object(TantivyEngine, "delete_batch", boom):
        yield


@contextmanager
def _crash_after_insert(_manager: MemoryManager) -> Generator[None]:
    original = USearchEngine.add_batch

    def boom(
        self: USearchEngine,
        workspace_id: str,
        contents: list[str],
        infer: bool,
        vectors: list[list[float]] | None = None,
    ) -> list[str]:
        stored = original(self, workspace_id, contents, infer, vectors=vectors)
        raise RuntimeError("crash after insert")
        return stored

    with patch.object(USearchEngine, "add_batch", boom):
        yield


@contextmanager
def _crash_after_tantivy_add(_manager: MemoryManager) -> Generator[None]:
    original = TantivyEngine.add_batch

    def boom(self: TantivyEngine, workspace_id: str, contents: list[str]) -> None:
        original(self, workspace_id, contents)
        raise RuntimeError("crash after tantivy add")

    with patch.object(TantivyEngine, "add_batch", boom):
        yield


@contextmanager
def _crash_before_usearch_save(_manager: MemoryManager) -> Generator[None]:
    def boom(self: USearchEngine) -> None:
        raise RuntimeError("crash before usearch save")

    with patch.object(USearchEngine, "commit", boom):
        yield


@contextmanager
def _crash_after_semantic_delete_once(_manager: MemoryManager) -> Generator[None]:
    original = USearchEngine.delete
    fired = {"done": False}

    def boom(self: USearchEngine, memory_id: str) -> None:
        original(self, memory_id)
        if not fired["done"]:
            fired["done"] = True
            raise RuntimeError("crash after semantic delete")

    with patch.object(USearchEngine, "delete", boom):
        yield


@contextmanager
def _crash_after_complete(_manager: MemoryManager) -> Generator[None]:
    original = MemoryStore.complete_replacement_transition

    def boom(self: MemoryStore, transition_id: int) -> None:
        original(self, transition_id)
        raise RuntimeError("crash after complete")

    with patch.object(MemoryStore, "complete_replacement_transition", boom):
        yield


async def _crash_and_reopen(
    tmpdir: str,
    inject: Injector,
) -> None:
    config = _config(tmpdir)
    first, _ = create_memory_manager(config)
    try:
        assert first.add_memories([OLD]) == 1
        with inject(first):
            with pytest.raises((StorageError, RuntimeError)):
                _ = await first._storage_phase.execute(
                    [NEW],
                    {NEW: [_replacement()]},
                )
    finally:
        _abandon_without_persist(first)

    second, _ = create_memory_manager(config)
    try:
        _assert_converged(second)
    finally:
        second.close()

    third, _ = create_memory_manager(config)
    try:
        _assert_converged(third)
    finally:
        cleanup_manager(third)


@pytest.mark.integration
class TestReplacementRecoveryIntegration:
    """Crash-injection + reopen proves restart-safe replacement."""

    async def test_normal_replacement_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _manager(tmpdir) as manager:
                assert manager.add_memories([OLD]) == 1
                dry = await manager._storage_phase.execute(
                    [NEW],
                    {NEW: [_replacement()]},
                    dry_run=True,
                )
                assert dry.stored_count == 1
                assert dry.replaced_count == 1
                assert manager.get_all() == [OLD]
                store = manager._semantic_engine.memory_store
                assert isinstance(store, MemoryStore)
                assert store.get_archived(manager.workspace_id) == []

                await _replace(manager)
                _assert_converged(manager)

    async def test_crash_after_transition_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_transition)

    async def test_crash_after_semantic_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_semantic_delete)

    async def test_crash_after_tantivy_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_tantivy_delete)

    async def test_crash_after_new_memory_insert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_insert)

    async def test_crash_after_tantivy_add(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_tantivy_add)

    async def test_crash_before_usearch_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_before_usearch_save)

    async def test_crash_after_complete_then_repeat_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await _crash_and_reopen(tmpdir, _crash_after_complete)

    async def test_live_reconcile_after_failed_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _manager(tmpdir) as manager:
                assert manager.add_memories([OLD]) == 1
                with _crash_after_semantic_delete_once(manager):
                    with pytest.raises((StorageError, RuntimeError)):
                        _ = await manager._storage_phase.execute(
                            [NEW],
                            {NEW: [_replacement()]},
                        )
                retry = await manager._storage_phase.execute([], {})
                assert retry.stored_count == 0
                _assert_converged(manager)

    async def test_records_all_olds_before_first_delete(self) -> None:
        old_a = "Convention A is stale"
        old_b = "Convention B is stale"
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _config(tmpdir)
            first, _ = create_memory_manager(config)
            try:
                assert first.add_memories([old_a, old_b]) == 2
                with _crash_after_transition(first):
                    with pytest.raises((StorageError, RuntimeError)):
                        _ = await first._storage_phase.execute(
                            [NEW],
                            {
                                NEW: [
                                    ReplacementInfo(
                                        old_memory=old_a,
                                        new_memory=NEW,
                                        confidence=0.9,
                                        reason="updated",
                                    ),
                                    ReplacementInfo(
                                        old_memory=old_b,
                                        new_memory=NEW,
                                        confidence=0.9,
                                        reason="updated",
                                    ),
                                ]
                            },
                        )
            finally:
                _abandon_without_persist(first)

            second, _ = create_memory_manager(config)
            try:
                memories = set(second.get_all())
                assert NEW in memories
                assert old_a not in memories
                assert old_b not in memories
                store = second._semantic_engine.memory_store
                assert isinstance(store, MemoryStore)
                assert store.list_pending_transitions() == []
                assert len(store.get_archived(second.workspace_id)) == 2
            finally:
                cleanup_manager(second)

    async def test_two_successors_keep_highest_confidence(self) -> None:
        loser = "Prefer two spaces for indentation in this repository"
        winner = NEW
        with tempfile.TemporaryDirectory() as tmpdir:
            with _manager(tmpdir) as manager:
                assert manager.add_memories([OLD]) == 1
                result = await manager._storage_phase.execute(
                    [loser, winner],
                    {
                        loser: [
                            ReplacementInfo(
                                old_memory=OLD,
                                new_memory=loser,
                                confidence=0.4,
                                reason="weaker convention",
                                similarity_score=0.7,
                            )
                        ],
                        winner: [
                            ReplacementInfo(
                                old_memory=OLD,
                                new_memory=winner,
                                confidence=0.95,
                                reason="updated convention",
                                similarity_score=0.88,
                            )
                        ],
                    },
                )
                assert result.stored_count == 2
                assert result.replaced_count == 1
                assert result.replacements[0].new_memory == winner
                memories = set(manager.get_all())
                assert memories == {loser, winner}
                assert manager.get_id_by_content(OLD) is None
                loser_id = manager.get_id_by_content(loser)
                winner_id = manager.get_id_by_content(winner)
                engine = manager._semantic_engine
                assert isinstance(engine, USearchEngine)
                index = engine.index
                assert loser_id is not None
                assert winner_id is not None
                assert index is not None
                assert loser_id in index
                assert winner_id in index
                store = manager._semantic_engine.memory_store
                assert isinstance(store, MemoryStore)
                archives = store.get_archived(manager.workspace_id)
                assert len(archives) == 1
                assert archives[0].replaced_by == winner
                assert store.list_pending_transitions() == []

    async def test_leftover_exclusive_heals_then_stores_unrelated(self) -> None:
        leftover_new = "Prefer two spaces for indentation in this repository"
        unrelated = "Prefer 100-character lines in this repository"
        with tempfile.TemporaryDirectory() as tmpdir:
            with _manager(tmpdir) as manager:
                assert manager.add_memories([OLD]) == 1
                old_id = manager.get_id_by_content(OLD)
                assert old_id is not None
                store = manager._semantic_engine.memory_store
                assert isinstance(store, MemoryStore)
                _ = store.begin_replacement_transition(
                    old_memory_id=old_id,
                    workspace_id=manager.workspace_id,
                    old_content=OLD,
                    new_content=leftover_new,
                    reason="first intent",
                    confidence=0.8,
                )
                result = await manager._storage_phase.execute(
                    [NEW, unrelated],
                    {
                        NEW: [
                            ReplacementInfo(
                                old_memory=OLD,
                                new_memory=NEW,
                                confidence=0.95,
                                reason="second intent",
                            )
                        ]
                    },
                )
                assert result.stored_count == 2
                assert result.replaced_count == 0
                memories = set(manager.get_all())
                assert leftover_new in memories
                assert NEW in memories
                assert unrelated in memories
                assert OLD not in memories
                assert store.list_pending_transitions() == []
