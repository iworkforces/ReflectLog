"""Unit tests for durable replacement transitions on MemoryStore."""

import os
from pathlib import Path
import tempfile

import pytest

from reflectlog.core.enums import TransitionKind
from reflectlog.core.exceptions import StorageError
from reflectlog.core.types import ReplacementTransition
from reflectlog.infrastructure.memory_store import (
    TRANSITION_COMPLETED,
    TRANSITION_PENDING,
    MemoryStore,
)


@pytest.mark.parametrize("completed", [False, True])
def test_successor_is_single_hop_and_does_not_replay_pending_transition(
    tmp_path: Path, completed: bool
) -> None:
    store = MemoryStore(db_path=str(tmp_path / "successor.db"))
    try:
        transition = _begin(store)
        if completed:
            store.complete_replacement_transition(transition.id)
        before = store.list_pending_transitions()
        assert store.get_record_with_successor("proj1", 11).successor_id is None
        successor_id = store.insert("proj1", "Use spaces")
        assert successor_id is not None
        assert store.get_record_with_successor("proj1", 11).successor_id == successor_id
        store.archive(
            successor_id, "proj1", "Use spaces", "Use four spaces", "private", 0.9
        )
        assert store.delete(successor_id)
        store.insert("proj1", "Use four spaces")
        assert store.get_record_with_successor("proj1", 11).successor_id is None
        readded_id = store.insert("proj1", "Use spaces")
        assert readded_id != successor_id
        assert store.get_record_with_successor("proj1", 11).successor_id == readded_id
        assert store.list_pending_transitions() == before
    finally:
        store.close()


def _begin(store: MemoryStore) -> ReplacementTransition:
    return store.begin_replacement_transition(
        old_memory_id=11,
        workspace_id="proj1",
        old_content="Use tabs",
        new_content="Use spaces",
        reason="updated convention",
        confidence=0.95,
    )


@pytest.mark.unit
class TestBeginReplacementTransition:
    """begin_replacement_transition is one SQLite transaction."""

    def test_records_archive_and_pending_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            transition = _begin(store)

            assert transition.status == TRANSITION_PENDING
            assert transition.old_memory_id == 11
            assert transition.old_content == "Use tabs"
            assert transition.new_content == "Use spaces"
            assert transition.reason == "updated convention"
            assert transition.confidence == 0.95
            assert transition.workspace_id == "proj1"

            archives = store.get_archived("proj1")
            assert len(archives) == 1
            assert archives[0].id == transition.archive_id
            assert archives[0].original_id == 11
            assert archives[0].content == "Use tabs"
            assert archives[0].replaced_by == "Use spaces"
            store.close()

    def test_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            first = _begin(store)
            second = _begin(store)

            assert first.id == second.id
            assert first.archive_id == second.archive_id
            assert len(store.get_archived("proj1")) == 1
            assert len(store.list_pending_transitions()) == 1
            store.close()

    def test_dedupes_archive_pairs_before_unique_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            cursor = store.connection.cursor()
            _ = cursor.execute("DROP INDEX IF EXISTS idx_archived_original_replaced")
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, confidence) "
                "VALUES (1, 'proj1', 'old', 'new', 'r', 0.9), "
                "(1, 'proj1', 'old', 'new', 'r', 0.8)"
            )
            store.connection.commit()
            cursor.close()
            store.close()

            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            archives = store.get_archived("proj1")
            assert len(archives) == 1
            store.close()

    def test_rejects_second_successor_for_same_old_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            _ = _begin(store)
            with pytest.raises(StorageError, match="already has a replacement"):
                _ = store.begin_replacement_transition(
                    old_memory_id=11,
                    workspace_id="proj1",
                    old_content="Use tabs",
                    new_content="Use two spaces",
                    reason="conflict",
                    confidence=0.5,
                )
            assert len(store.list_pending_transitions()) == 1
            assert store.list_pending_transitions()[0].new_content == "Use spaces"
            store.close()

    def test_batch_records_all_or_none(self) -> None:
        from reflectlog.core.types import ReplacementTransitionRequest

        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            _ = store.begin_replacement_transitions(
                [
                    ReplacementTransitionRequest(
                        old_memory_id=1,
                        workspace_id="proj1",
                        old_content="old-a",
                        new_content="new",
                        reason="updated",
                        confidence=0.9,
                    ),
                    ReplacementTransitionRequest(
                        old_memory_id=2,
                        workspace_id="proj1",
                        old_content="old-b",
                        new_content="new",
                        reason="updated",
                        confidence=0.9,
                    ),
                ]
            )
            assert len(store.list_pending_transitions()) == 2
            store.close()

    def test_rolls_back_archive_when_transition_insert_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            _ = store.connection
            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE replacement_transitions")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="replacement transition"):
                _ = _begin(store)

            assert store.get_archived("proj1") == []
            store.close()


@pytest.mark.unit
class TestPendingTransitionLifecycle:
    """list/complete are idempotent and leave an audit row."""

    def test_list_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            transition = _begin(store)

            pending = store.list_pending_transitions()
            assert len(pending) == 1
            assert pending[0].id == transition.id
            assert store.is_pending_transition(transition.id)
            assert not store.is_pending_transition(transition.id + 1)

            store.complete_replacement_transition(transition.id)
            store.complete_replacement_transition(transition.id)

            assert not store.is_pending_transition(transition.id)
            assert store.list_pending_transitions() == []
            archives = store.get_archived("proj1")
            assert len(archives) == 1
            assert archives[0].replaced_by == "Use spaces"
            store.close()

    def test_complete_keeps_row_for_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            transition = _begin(store)
            store.complete_replacement_transition(transition.id)

            cursor = store.connection.cursor()
            _ = cursor.execute(
                "SELECT status FROM replacement_transitions WHERE id = ?",
                (transition.id,),
            )
            row = cursor.fetchone()
            cursor.close()
            assert row is not None
            assert row[0] == TRANSITION_COMPLETED
            store.close()

    def test_lookup_by_old_memory_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            transition = _begin(store)
            found = store.get_transition_for_old_memory("proj1", 11)
            assert found is not None
            assert found.id == transition.id
            assert store.get_transition_for_old_memory("proj1", 99) is None
            store.close()

    def test_complete_missing_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            _ = store.connection
            with pytest.raises(StorageError, match="was not pending"):
                store.complete_replacement_transition(999)
            store.close()


@pytest.mark.unit
class TestAddAndDeleteIntents:
    """Ordinary add/delete intents share the replacement_transitions table."""

    def test_begin_add_intents_are_pending_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            first = store.begin_add_intents("proj1", ["hello", "hello"])
            second = store.begin_add_intents("proj1", ["hello"])
            assert len(first) == 1
            assert first[0].kind == TransitionKind.ADD
            assert first[0].new_content == "hello"
            assert second[0].id == first[0].id
            assert len(store.list_pending_transitions()) == 1
            store.close()

    def test_begin_delete_intents_are_id_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            rows = store.begin_delete_intents("proj1", [(11, "hello")])
            assert len(rows) == 1
            assert rows[0].kind == TransitionKind.DELETE
            assert rows[0].old_memory_id == 11
            assert rows[0].old_content == "hello"
            store.close()

    def test_has_later_intent_sees_completed_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            added = store.begin_add_intents("proj1", ["hello"])[0]
            deleted = store.begin_delete_intents("proj1", [(11, "hello")])[0]
            store.complete_replacement_transition(deleted.id)
            assert store.has_later_intent(
                workspace_id="proj1",
                kind=TransitionKind.DELETE,
                content="hello",
                after_id=added.id,
            )
            store.close()

    def test_reopen_keeps_later_pending_adds(self) -> None:
        """Connect-time dedupe must not drop add intents that share old_memory_id=0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=path)
            first = store.begin_add_intents("proj1", ["one"])[0]
            store.complete_replacement_transition(first.id)
            second = store.begin_add_intents("proj1", ["two"])[0]
            store.close()

            reopened = MemoryStore(db_path=path)
            pending = reopened.list_pending_transitions()
            assert [row.new_content for row in pending] == ["two"]
            assert pending[0].id == second.id
            reopened.close()

    def test_reopen_keeps_two_pending_adds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=path)
            first = store.begin_add_intents("proj1", ["one"])[0]
            store.complete_replacement_transition(first.id)
            _ = store.begin_add_intents("proj1", ["two", "three"])
            store.close()
            reopened = MemoryStore(db_path=path)
            pending = {row.new_content for row in reopened.list_pending_transitions()}
            assert pending == {"two", "three"}
            reopened.close()

    def test_readd_after_delete_gets_new_intent_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            first = store.begin_add_intents("proj1", ["hello"])[0]
            deleted = store.begin_delete_intents("proj1", [(11, "hello")])[0]
            store.complete_replacement_transition(deleted.id)
            second = store.begin_add_intents("proj1", ["hello"])[0]
            assert second.id > first.id
            assert second.id > deleted.id
            store.close()

    def test_delete_and_replace_can_coexist_for_same_old_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            replaced = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj1",
                old_content="hello",
                new_content="hello v2",
                reason="updated",
                confidence=0.9,
            )
            deleted = store.begin_delete_intents("proj1", [(11, "hello")])[0]
            assert deleted.id != replaced.id
            assert deleted.kind == TransitionKind.DELETE
            assert replaced.kind == TransitionKind.REPLACE
            store.close()

    def test_has_later_intent_delete_sees_later_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            added = store.begin_add_intents("proj1", ["hello"])[0]
            _ = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj1",
                old_content="hello",
                new_content="hello v2",
                reason="updated",
                confidence=0.9,
            )
            assert store.has_later_intent(
                workspace_id="proj1",
                kind=TransitionKind.DELETE,
                content="hello",
                after_id=added.id,
            )
            store.close()


@pytest.mark.unit
class TestTransitionRowParse:
    """Corrupt journal kinds must fail closed."""

    def test_unknown_kind_raises_storage_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            _ = store.connection.execute(
                """
                INSERT INTO replacement_transitions
                    (workspace_id, old_memory_id, old_content, new_content,
                     archive_id, reason, confidence, status, kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "proj1",
                    11,
                    "old",
                    "new",
                    0,
                    "r",
                    0.9,
                    TRANSITION_PENDING,
                    "bogus",
                ),
            )
            store.connection.commit()
            with pytest.raises(StorageError, match="Invalid replacement transition"):
                _ = store.list_pending_transitions()
            store.close()
