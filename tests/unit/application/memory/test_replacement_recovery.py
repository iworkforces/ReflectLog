"""Unit tests for restart reconciliation of unfinished replacements."""

import os
import tempfile
import threading
from unittest.mock import MagicMock

import pytest

from reflectlog.application.memory.replacement_recovery import (
    apply_pending_transition,
    reconcile_pending_replacements,
    replacement_converged,
)
from reflectlog.core.enums import TransitionKind, TransitionStatus
from reflectlog.core.storage_coordination import LeaseMode, WorkspaceStoragePaths
from reflectlog.core.types import ReplacementTransition
from reflectlog.infrastructure.memory_store import MemoryStore


def _stub_contains(semantic: MagicMock) -> None:
    def _contains(memory_id: int) -> bool:
        index = semantic.index
        if isinstance(index, (set, dict)):
            return memory_id in index
        return False

    semantic.contains_id.side_effect = _contains


def _stub_journal(
    semantic: MagicMock,
    *rows: ReplacementTransition,
) -> None:
    semantic.memory_store.list_pending_transitions.return_value = (
        list(rows) if rows else [_transition()]
    )
    semantic.memory_store.has_later_intent.return_value = False
    _stub_contains(semantic)


def _transition() -> ReplacementTransition:
    return ReplacementTransition(
        id=3,
        workspace_id="proj",
        old_memory_id=11,
        old_content="old convention",
        new_content="new convention",
        archive_id=8,
        reason="updated",
        confidence=0.9,
        status=TransitionStatus.PENDING,
    )


@pytest.mark.unit
class TestApplyPendingTransition:
    """apply_pending_transition is idempotent per backend."""

    def test_deletes_old_and_inserts_missing_new(self) -> None:
        semantic = MagicMock()
        _stub_journal(semantic)
        semantic.get_id_by_content.side_effect = [None, 99, None, 99, None]
        semantic.index = {99}
        tantivy = MagicMock()
        seen_new = {"yes": False}

        def find(_pid: str, content: str) -> list[str]:
            if content == "new convention" and seen_new["yes"]:
                return [content]
            return []

        def add(_pid: str, content: str) -> None:
            if content == "new convention":
                seen_new["yes"] = True

        tantivy.find_by_exact_match.side_effect = find
        tantivy.add.side_effect = add
        logger = MagicMock()

        _ = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=tantivy,
            logger=logger,
        )

        semantic.delete.assert_called_once_with(memory_id="11")
        tantivy.delete.assert_called_once_with(
            "proj", "old convention", verify_exists=False
        )
        semantic.add.assert_called_once_with(
            workspace_id="proj",
            content="new convention",
            infer=False,
        )
        tantivy.add.assert_called_once_with("proj", "new convention")
        tantivy.commit.assert_called_once()
        semantic.commit.assert_called_once()
        semantic.memory_store.complete_replacement_transition.assert_called_once_with(3)

    def test_skips_insert_when_replacement_already_present(self) -> None:
        semantic = MagicMock()
        _stub_journal(semantic)

        def get_id(_workspace_id: str, content: str) -> int | None:
            return 99 if content == "new convention" else None

        semantic.get_id_by_content.side_effect = get_id
        semantic.index = {99}
        tantivy = MagicMock()

        def find_existing(_workspace_id: str, content: str) -> list[str]:
            return [content] if content == "new convention" else []

        tantivy.find_by_exact_match.side_effect = find_existing

        _ = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=tantivy,
            logger=MagicMock(),
        )

        semantic.add.assert_not_called()
        tantivy.add.assert_not_called()
        semantic.memory_store.complete_replacement_transition.assert_called_once_with(3)

    def test_reindexes_missing_vector(self) -> None:
        semantic = MagicMock()
        _stub_journal(semantic)

        def get_new_id(_workspace_id: str, content: str) -> int | None:
            return 7 if content == "new convention" else None

        semantic.get_id_by_content.side_effect = get_new_id
        semantic.index = set()

        def add_and_index(**_kwargs: object) -> None:
            semantic.index.add(7)

        semantic.add.side_effect = add_and_index

        _ = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=None,
            logger=MagicMock(),
        )

        semantic.delete.assert_any_call(memory_id="11")
        semantic.delete.assert_any_call(memory_id="7")
        semantic.add.assert_called_once_with(
            workspace_id="proj",
            content="new convention",
            infer=False,
        )
        semantic.memory_store.complete_replacement_transition.assert_called_once()

    def test_skips_tantivy_delete_when_old_text_was_readded(self) -> None:
        semantic = MagicMock()
        _stub_journal(semantic)

        def get_current_id(_workspace_id: str, content: str) -> int:
            return 22 if content == "old convention" else 99

        semantic.get_id_by_content.side_effect = get_current_id
        semantic.index = {99}
        tantivy = MagicMock()

        def find_all(_workspace_id: str, content: str) -> list[str]:
            return [content]

        tantivy.find_by_exact_match.side_effect = find_all

        completed = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=tantivy,
            logger=MagicMock(),
        )

        assert completed is True
        tantivy.delete.assert_not_called()
        semantic.memory_store.complete_replacement_transition.assert_called_once()

    def test_leaves_pending_when_tantivy_still_has_old(self) -> None:
        semantic = MagicMock()
        _stub_journal(semantic)

        def get_replacement_id(_workspace_id: str, content: str) -> int | None:
            return 99 if content == "new convention" else None

        semantic.get_id_by_content.side_effect = get_replacement_id
        semantic.index = {99}
        tantivy = MagicMock()

        def find_all(_workspace_id: str, content: str) -> list[str]:
            return [content]

        tantivy.find_by_exact_match.side_effect = find_all

        completed = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=tantivy,
            logger=MagicMock(),
        )

        assert completed is False
        semantic.memory_store.complete_replacement_transition.assert_not_called()

    def test_does_not_complete_when_indexes_disagree(self) -> None:
        semantic = MagicMock()
        _stub_journal(semantic)
        semantic.get_id_by_content.return_value = None
        semantic.index = set()
        tantivy = MagicMock()
        tantivy.find_by_exact_match.return_value = []
        semantic.add.side_effect = RuntimeError("embedder down")

        with pytest.raises(RuntimeError, match="embedder down"):
            _ = apply_pending_transition(
                _transition(),
                semantic_engine=semantic,
                tantivy_engine=tantivy,
                logger=MagicMock(),
            )
        semantic.memory_store.complete_replacement_transition.assert_not_called()

    def test_works_without_tantivy(self) -> None:
        semantic = MagicMock()
        _stub_journal(semantic)
        semantic.get_id_by_content.side_effect = [None, 7, 7, None]
        semantic.index = {7}

        _ = apply_pending_transition(
            _transition(),
            semantic_engine=semantic,
            tantivy_engine=None,
            logger=MagicMock(),
        )

        semantic.add.assert_called_once()
        semantic.commit.assert_called_once()
        semantic.memory_store.complete_replacement_transition.assert_called_once()

    def test_converged_requires_old_id_gone(self) -> None:
        semantic = MagicMock()
        _stub_journal(semantic)

        def get_live_id(_workspace_id: str, content: str) -> int:
            return 11 if content == "old convention" else 99

        semantic.get_id_by_content.side_effect = get_live_id
        semantic.index = {99}
        assert (
            replacement_converged(
                _transition(), semantic_engine=semantic, tantivy_engine=None
            )
            is False
        )

    def test_converged_when_old_text_live_under_new_id(self) -> None:
        semantic = MagicMock()
        _stub_journal(semantic)

        def get_live_id(_workspace_id: str, content: str) -> int:
            return 22 if content == "old convention" else 99

        semantic.get_id_by_content.side_effect = get_live_id
        semantic.index = {99}
        tantivy = MagicMock()

        def find_all(_workspace_id: str, content: str) -> list[str]:
            return [content]

        tantivy.find_by_exact_match.side_effect = find_all
        assert (
            replacement_converged(
                _transition(), semantic_engine=semantic, tantivy_engine=tantivy
            )
            is True
        )

    def test_converged_false_when_index_missing(self) -> None:
        semantic = MagicMock()
        semantic.get_id_by_content.return_value = 99
        semantic.contains_id.return_value = None
        assert (
            replacement_converged(
                _transition(), semantic_engine=semantic, tantivy_engine=None
            )
            is False
        )


@pytest.mark.unit
class TestReconcilePendingReplacements:
    """Startup reconciliation respects lock order and skips empty stores."""

    def test_noops_for_mock_store(self) -> None:
        semantic = MagicMock()
        semantic.memory_store = MagicMock()
        semantic.memory_store.db_path = ""
        count = reconcile_pending_replacements(
            semantic_engine=semantic,
            tantivy_engine=None,
            write_lock=threading.Lock(),
            lock=threading.RLock(),
            logger=MagicMock(),
        )
        assert count == 0

    def test_skips_when_store_db_is_missing(self) -> None:
        logger = MagicMock()
        store = MagicMock()
        store.db_path = "/no/such/memories.db"
        store.list_pending_transitions.return_value = []
        semantic = MagicMock()
        semantic.memory_store = store
        count = reconcile_pending_replacements(
            semantic_engine=semantic,
            tantivy_engine=None,
            write_lock=threading.Lock(),
            lock=threading.RLock(),
            logger=logger,
        )
        assert count == 0

    def test_failed_precompute_leaves_add_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            _ = store.begin_add_intents("proj", ["ghost-content"])
            semantic = MagicMock()
            semantic.memory_store = store
            semantic.embedder.embed_documents.side_effect = RuntimeError(
                "provider down"
            )
            semantic.get_id_by_content.return_value = None
            semantic.ensure_initialized = MagicMock()
            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )
            assert count == 0
            pending = store.list_pending_transitions()
            assert len(pending) == 1
            assert pending[0].new_content == "ghost-content"
            semantic.add.assert_not_called()
            semantic.add_batch.assert_not_called()
            store.close()

    def test_pending_add_indexes_document_role_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            _ = store.begin_add_intents("proj", ["pending add"])
            semantic = MagicMock()
            semantic.memory_store = store
            semantic.embedder.embed_query.return_value = [1.0, 0.0]
            semantic.embedder.embed_documents.return_value = [[0.0, 1.0]]
            semantic.index = set()
            live: dict[str, int] = {}
            semantic.get_id_by_content.side_effect = lambda _workspace_id, content: (
                live.get(content)
            )
            semantic.contains_id.side_effect = lambda memory_id: (
                memory_id in semantic.index
            )

            def add_batch(
                _workspace_id: str,
                contents: list[str],
                *,
                infer: bool,
                vectors: list[list[float]],
            ) -> list[str]:
                _ = infer, vectors
                live[contents[0]] = 21
                semantic.index.add(21)
                return contents

            semantic.add_batch.side_effect = add_batch

            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )

            assert count == 1
            semantic.embedder.embed_documents.assert_called_once_with(["pending add"])
            semantic.embedder.embed_query.assert_not_called()
            semantic.add_batch.assert_called_once_with(
                "proj",
                ["pending add"],
                infer=False,
                vectors=[[0.0, 1.0]],
            )
            store.close()

    def test_pending_replace_indexes_document_role_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            _ = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="old convention",
                new_content="replacement document",
                reason="updated",
                confidence=0.9,
            )
            semantic = MagicMock()
            semantic.memory_store = store
            semantic.embedder.embed_query.return_value = [1.0, 0.0]
            semantic.embedder.embed_documents.return_value = [[0.0, 1.0]]
            semantic.index = set()
            live: dict[str, int] = {}
            semantic.get_id_by_content.side_effect = lambda _workspace_id, content: (
                live.get(content)
            )
            semantic.contains_id.side_effect = lambda memory_id: (
                memory_id in semantic.index
            )

            def add_batch(
                _workspace_id: str,
                contents: list[str],
                *,
                infer: bool,
                vectors: list[list[float]],
            ) -> list[str]:
                _ = infer, vectors
                live[contents[0]] = 22
                semantic.index.add(22)
                return contents

            semantic.add_batch.side_effect = add_batch
            semantic.delete.side_effect = lambda *, memory_id: semantic.index.discard(
                int(memory_id)
            )

            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )

            assert count == 1
            semantic.embedder.embed_documents.assert_called_once_with(
                ["replacement document"]
            )
            semantic.embedder.embed_query.assert_not_called()
            semantic.add_batch.assert_called_once_with(
                "proj",
                ["replacement document"],
                infer=False,
                vectors=[[0.0, 1.0]],
            )
            store.close()

    @pytest.mark.parametrize(
        "document_vectors",
        [[[0.0, 1.0]], [[0.0, 1.0], []]],
    )
    def test_invalid_document_batch_leaves_all_adds_pending(
        self, document_vectors: list[list[float]]
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            _ = store.begin_add_intents("proj", ["first add", "second add"])
            semantic = MagicMock()
            semantic.memory_store = store
            semantic.embedder.embed_query.return_value = [1.0, 0.0]
            semantic.embedder.embed_documents.return_value = document_vectors
            semantic.get_id_by_content.return_value = None
            semantic.contains_id.return_value = False

            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )

            assert count == 0
            assert len(store.list_pending_transitions()) == 2
            semantic.add_batch.assert_not_called()
            semantic.embedder.embed_query.assert_not_called()
            store.close()

    def test_noops_when_nothing_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            _ = store.connection
            semantic = MagicMock()
            semantic.memory_store = store
            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )
            assert count == 0
            store.close()

    def test_acquires_write_lock_before_lock(self) -> None:
        order: list[str] = []

        class OrderLock:
            def __init__(self, name: str, inner: threading.Lock | threading.RLock):
                self.name = name
                self.inner = inner

            def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
                order.append(self.name)
                return self.inner.acquire(blocking, timeout)

            def release(self) -> None:
                self.inner.release()

            def __enter__(self) -> OrderLock:
                _ = self.acquire()
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: object,
            ) -> None:
                self.release()

        write_lock = OrderLock("write", threading.Lock())
        inner_lock = OrderLock("inner", threading.RLock())
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            planted = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="old convention",
                new_content="new convention",
                reason="updated",
                confidence=0.9,
            )
            semantic = MagicMock()
            semantic.memory_store = store
            _stub_contains(semantic)

            def get_id(_workspace_id: str, content: str) -> int | None:
                return 99 if content == "new convention" else None

            semantic.get_id_by_content.side_effect = get_id
            semantic.index = {99}

            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=write_lock,
                lock=inner_lock,
                logger=MagicMock(),
            )

            assert planted.id == 1
            assert count == 1
            assert order == ["write", "inner"]
            semantic.delete.assert_called_once_with(memory_id="11")
            assert store.list_pending_transitions() == []
            store.close()

    def test_repeated_reconcile_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            _ = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="old convention",
                new_content="new convention",
                reason="updated",
                confidence=0.9,
            )
            semantic = MagicMock()
            semantic.memory_store = store
            _stub_contains(semantic)
            semantic.get_id_by_content.return_value = 99
            semantic.index = {99: object()}

            first = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )
            second = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )

            assert first == 1
            assert second == 0
            assert store.list_pending_transitions() == []
            assert len(store.get_archived("proj")) == 1
            store.close()

    def test_count_skips_transitions_that_stay_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            _ = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="old convention",
                new_content="new convention",
                reason="updated",
                confidence=0.9,
            )
            semantic = MagicMock()
            semantic.memory_store = store
            _stub_contains(semantic)

            def get_replacement_id(_workspace_id: str, content: str) -> int | None:
                return 99 if content == "new convention" else None

            semantic.get_id_by_content.side_effect = get_replacement_id
            semantic.index = {99}
            tantivy = MagicMock()

            def find_all(_workspace_id: str, content: str) -> list[str]:
                return [content]

            tantivy.find_by_exact_match.side_effect = find_all

            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=tantivy,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )

            assert count == 0
            assert len(store.list_pending_transitions()) == 1
            store.close()

    def test_stale_add_is_completed_when_later_replace_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            added = store.begin_add_intents("proj", ["hello"])[0]
            _ = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="hello",
                new_content="hello v2",
                reason="updated",
                confidence=0.9,
            )
            semantic = MagicMock()
            semantic.memory_store = store
            semantic.get_id_by_content.return_value = None
            completed = apply_pending_transition(
                added,
                semantic_engine=semantic,
                tantivy_engine=None,
                logger=MagicMock(),
            )
            assert completed is True
            semantic.add.assert_not_called()
            assert all(
                row.kind != TransitionKind.ADD
                for row in store.list_pending_transitions()
            )
            store.close()

    def test_later_delete_in_other_workspace_does_not_suppress_add(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            added = store.begin_add_intents("proj-a", ["hello"])[0]
            _ = store.begin_delete_intents("proj-b", [(11, "hello")])
            semantic = MagicMock()
            semantic.memory_store = store
            _stub_contains(semantic)
            semantic.get_id_by_content.return_value = None
            semantic.add.return_value = None
            semantic.index = {1}
            completed = apply_pending_transition(
                added,
                semantic_engine=semantic,
                tantivy_engine=None,
                logger=MagicMock(),
            )
            semantic.add.assert_called()
            assert completed is False
            assert added.id in {row.id for row in store.list_pending_transitions()}
            store.close()

    def test_stale_replace_is_completed_when_later_delete_of_new_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            replaced = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="hello",
                new_content="hello v2",
                reason="updated",
                confidence=0.9,
            )
            deleted = store.begin_delete_intents("proj", [(22, "hello v2")])[0]
            store.complete_replacement_transition(deleted.id)
            semantic = MagicMock()
            semantic.memory_store = store
            _stub_contains(semantic)
            semantic.get_id_by_content.return_value = None
            semantic.index = set()
            completed = apply_pending_transition(
                replaced,
                semantic_engine=semantic,
                tantivy_engine=None,
                logger=MagicMock(),
            )
            assert completed is True
            semantic.add.assert_not_called()
            assert all(
                row.id != replaced.id for row in store.list_pending_transitions()
            )
            store.close()

    def test_completed_later_delete_supersedes_pending_add(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            added = store.begin_add_intents("proj", ["hello"])[0]
            deleted = store.begin_delete_intents("proj", [(11, "hello")])[0]
            store.complete_replacement_transition(deleted.id)
            semantic = MagicMock()
            semantic.memory_store = store
            semantic.get_id_by_content.return_value = None
            completed = apply_pending_transition(
                added,
                semantic_engine=semantic,
                tantivy_engine=None,
                logger=MagicMock(),
            )
            assert completed is True
            semantic.add.assert_not_called()
            store.close()

    def test_later_add_of_old_text_is_not_superseded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            replaced = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="I live in NYC",
                new_content="I moved to Boston",
                reason="updated",
                confidence=0.9,
            )
            added = store.begin_add_intents("proj", ["I live in NYC"])[0]
            assert added.id > replaced.id
            live: dict[str, int] = {}
            semantic = MagicMock()
            semantic.memory_store = store
            semantic.index = set()

            def get_id(_workspace_id: str, content: str) -> int | None:
                return live.get(content)

            def add(workspace_id: str, content: str, infer: bool = False) -> None:
                _ = (workspace_id, infer)
                live[content] = 33
                semantic.index.add(33)

            semantic.get_id_by_content.side_effect = get_id
            semantic.add.side_effect = add
            _stub_contains(semantic)
            completed = apply_pending_transition(
                added,
                semantic_engine=semantic,
                tantivy_engine=None,
                logger=MagicMock(),
            )
            assert completed is True
            assert live == {"I live in NYC": 33}
            assert added.id not in {row.id for row in store.list_pending_transitions()}
            store.close()

    def test_apply_replace_completes_later_add_of_old_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            replaced = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="I live in NYC",
                new_content="I moved to Boston",
                reason="updated",
                confidence=0.9,
            )
            added = store.begin_add_intents("proj", ["I live in NYC"])[0]
            semantic = MagicMock()
            semantic.memory_store = store
            _stub_contains(semantic)
            semantic.get_id_by_content.side_effect = lambda workspace_id, content: (
                22 if content == "I moved to Boston" else None
            )
            semantic.index = {22}
            semantic.contains_id.side_effect = lambda memory_id: memory_id == 22
            completed = apply_pending_transition(
                replaced,
                semantic_engine=semantic,
                tantivy_engine=None,
                logger=MagicMock(),
            )
            assert completed is True
            pending_ids = {row.id for row in store.list_pending_transitions()}
            assert added.id not in pending_ids
            store.close()

    def test_reconcile_replace_then_later_add_does_not_reinsert_old_text(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            replaced = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="I live in NYC",
                new_content="I moved to Boston",
                reason="updated",
                confidence=0.9,
            )
            added = store.begin_add_intents("proj", ["I live in NYC"])[0]
            assert added.id > replaced.id
            live: dict[str, int] = {}

            semantic = MagicMock()
            semantic.memory_store = store
            semantic.embedder.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
            semantic.index = set()

            def get_id(_workspace_id: str, content: str) -> int | None:
                return live.get(content)

            def add(workspace_id: str, content: str, infer: bool = False) -> None:
                _ = (workspace_id, infer)
                live[content] = 100 + len(live)
                semantic.index.add(live[content])

            def add_batch(
                workspace_id: str,
                contents: list[str],
                infer: bool = False,
                vectors: list[list[float]] | None = None,
            ) -> list[str]:
                _ = (infer, vectors)
                for content in contents:
                    add(workspace_id, content)
                return list(contents)

            def delete(*, memory_id: str) -> None:
                target = int(memory_id)
                for content, mem_id in list(live.items()):
                    if mem_id == target:
                        del live[content]
                        semantic.index.discard(target)

            semantic.get_id_by_content.side_effect = get_id
            semantic.add.side_effect = add
            semantic.add_batch.side_effect = add_batch
            semantic.delete.side_effect = delete
            semantic.contains_id.side_effect = lambda memory_id: (
                memory_id in semantic.index
            )

            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )
            assert count >= 1
            assert "I live in NYC" not in live
            assert "I moved to Boston" in live
            pending_ids = {row.id for row in store.list_pending_transitions()}
            assert added.id not in pending_ids
            assert replaced.id not in pending_ids
            store.close()

    def test_apply_completed_transition_does_not_insert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            added = store.begin_add_intents("proj", ["I live in NYC"])[0]
            store.complete_replacement_transition(added.id)
            semantic = MagicMock()
            semantic.memory_store = store
            completed = apply_pending_transition(
                added,
                semantic_engine=semantic,
                tantivy_engine=None,
                logger=MagicMock(),
            )
            assert completed is False
            semantic.add.assert_not_called()
            store.close()

    def test_reconcile_uses_coordinator_lease(self) -> None:
        acquired: list[str] = []

        class _Lease:
            workspace_id = "ws"
            mode = LeaseMode.EXCLUSIVE

            def release(self) -> None:
                return None

            def __enter__(self) -> _Lease:
                acquired.append("lease")
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: object,
            ) -> None:
                _ = exc_type, exc, traceback
                return None

        class _Coordinator:
            timeout = 1.0
            published: list[int] = []

            def acquire(
                self,
                workspace_id: str,
                mode: LeaseMode = LeaseMode.EXCLUSIVE,
                *,
                timeout: float | None = None,
            ) -> _Lease:
                _ = timeout
                lease = _Lease()
                lease.workspace_id = workspace_id
                lease.mode = mode
                return lease

            def read_generation(self, workspace_id: str) -> int:
                _ = workspace_id
                return 0

            def publish_generation(self, workspace_id: str, generation: int) -> None:
                _ = workspace_id
                self.published.append(generation)

            def is_held(self, workspace_id: str, mode: LeaseMode | None = None) -> bool:
                _ = workspace_id, mode
                return False

            def paths_for(self, workspace_id: str) -> WorkspaceStoragePaths:
                return WorkspaceStoragePaths(
                    workspace_id=workspace_id,
                    root="/tmp",
                    lock_path="/tmp/.lock",
                    generation_path="/tmp/.gen",
                )

        semantic = MagicMock()
        row = ReplacementTransition(
            id=1,
            workspace_id="ws",
            old_memory_id=1,
            old_content="old",
            new_content="new",
            archive_id=1,
            reason="test",
            confidence=1.0,
            status=TransitionStatus.PENDING,
            kind=TransitionKind.ADD,
        )
        semantic.memory_store.list_pending_transitions.return_value = [row]
        semantic.memory_store.has_later_intent.return_value = False
        semantic.get_id_by_content.return_value = 1
        semantic.contains_id.return_value = True
        count = reconcile_pending_replacements(
            semantic_engine=semantic,
            tantivy_engine=None,
            write_lock=threading.Lock(),
            lock=threading.RLock(),
            logger=MagicMock(),
            coordinator=_Coordinator(),
            workspace_id="ws",
        )
        assert acquired == ["lease"]
        assert count >= 1
        assert _Coordinator.published == [1]

    def test_reconcile_replace_then_earlier_add_does_not_reinsert_old_text(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
            added = store.begin_add_intents("proj", ["I live in NYC"])[0]
            replaced = store.begin_replacement_transition(
                old_memory_id=11,
                workspace_id="proj",
                old_content="I live in NYC",
                new_content="I moved to Boston",
                reason="updated",
                confidence=0.9,
            )
            assert added.id < replaced.id
            live: dict[str, int] = {}

            semantic = MagicMock()
            semantic.memory_store = store
            semantic.embedder.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
            semantic.index = set()

            def get_id(_workspace_id: str, content: str) -> int | None:
                return live.get(content)

            def add(_workspace_id: str, content: str, infer: bool = False) -> None:
                live[content] = 100 + len(live)
                semantic.index.add(live[content])

            def add_batch(
                workspace_id: str,
                contents: list[str],
                infer: bool = False,
                vectors: list[list[float]] | None = None,
            ) -> list[str]:
                _ = (workspace_id, infer, vectors)
                for content in contents:
                    add(workspace_id, content)
                return list(contents)

            def delete(*, memory_id: str) -> None:
                target = int(memory_id)
                for content, mem_id in list(live.items()):
                    if mem_id == target:
                        del live[content]
                        semantic.index.discard(target)

            semantic.get_id_by_content.side_effect = get_id
            semantic.add.side_effect = add
            semantic.add_batch.side_effect = add_batch
            semantic.delete.side_effect = delete
            semantic.contains_id.side_effect = lambda memory_id: (
                memory_id in semantic.index
            )

            count = reconcile_pending_replacements(
                semantic_engine=semantic,
                tantivy_engine=None,
                write_lock=threading.Lock(),
                lock=threading.RLock(),
                logger=MagicMock(),
            )
            assert count >= 1
            assert "I live in NYC" not in live
            assert "I moved to Boston" in live
            pending_ids = {row.id for row in store.list_pending_transitions()}
            assert added.id not in pending_ids
            assert replaced.id not in pending_ids
            store.close()
