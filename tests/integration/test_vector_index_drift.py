"""Real-engine acceptance tests for refusing a drifted vector index.

A workspace is *drifted* when the memory ids in SQLite and the live keys of
``vectors.usearch`` disagree (a row without a vector, a vector without a row,
or both with equal totals). These tests use a real ``USearchEngine``,
``MemoryStore`` and Tantivy index through real ``MemoryManager`` objects; no
engine is mocked. Scenarios:

* DRIFT-1 missing vector, DRIFT-2 orphan vector, DRIFT-3 equal-count swapped
  ids and restored older index: refused with ``InitializationError`` carrying
  the restore-or-rebuild sentence, through manager construction (eager),
  every public operation (lazy), the ``WorkspaceRegistry`` and the MCP tools.
* DRIFT-4 ``get_all`` keeps reading SQLite.
* Pending add/delete/replace rows left by a crash still reopen and converge,
  and a pending row can never mask unrelated drift.
* A refusal writes nothing: the vector file and the database stay identical.

Guards that pass on the current tree are marked in their docstrings; they pin
behaviour the drift check must not break. Everything else is red until the
check exists.
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
from pathlib import Path
import shutil
import sqlite3
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from usearch.index import Index

from reflectlog.application.config.settings import Config
from reflectlog.application.memory import replacement_recovery
from reflectlog.application.memory.add_phases import (
    Phase2Result,
    ReplacementInfo,
    SmartReplacementPhase,
)
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.memory.workspace_registry import WorkspaceRegistry
from reflectlog.application.tools.add_memories import AddMemoriesTool
from reflectlog.application.tools.delete_memories import DeleteMemoriesTool
from reflectlog.application.tools.search_memories import SearchMemoriesTool
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.exceptions import InitializationError, SearchError, StorageError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.storage_coordination import IStorageCoordinator
from reflectlog.core.types import ISemanticSearchEngine, ReplacementTransition
from reflectlog.infrastructure.index_integrity import INDEX_DRIFT_OPERATOR_ACTION
from reflectlog.infrastructure.memory_store import (
    MemoryStore,
    read_memory_store_snapshot,
)
from reflectlog.infrastructure.tantivy_engine import TantivyEngine
from reflectlog.infrastructure.usearch_engine import USearchEngine
from tests.integration.test_memory_manager_usearch import (
    MockEmbedder,
    create_memory_manager,
    create_usearch_config,
)
from tests.integration.test_replacement_recovery import _abandon_without_persist

pytestmark = pytest.mark.integration

CONTENTS: dict[str, str] = {
    "alpha": "Deploy the billing service with canary rollouts",
    "beta": "Prefer pytest fixtures over unittest setUp",
    "gamma": "Rotate the staging database credentials monthly",
    "delta": "Document every public API with examples",
}
FRESH = "Pin transitive dependencies before every release"
DIMS = 128


class Drift(StrEnum):
    MISSING_VECTOR = "missing_vector"
    ORPHAN_VECTOR = "orphan_vector"
    SWAPPED_IDS = "swapped_ids"
    OLDER_INDEX = "older_index"


class Operation(StrEnum):
    ADD = "add"
    ADD_ASYNC = "add_async"
    SEARCH = "search"
    REMOVE = "remove"


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``indexes/<workspace>`` (relative to the cwd) inside tmp_path."""
    monkeypatch.chdir(tmp_path)


@dataclass(frozen=True)
class Workspace:
    config: Config
    ids: Mapping[str, int]

    @property
    def workspace_id(self) -> str:
        return self.config.workspace_id

    @property
    def directory(self) -> Path:
        return Path.cwd() / "indexes" / self.workspace_id.lower() / "usearch"

    @property
    def index_path(self) -> Path:
        return self.directory / "vectors.usearch"

    @property
    def db_path(self) -> Path:
        return self.directory / "memories.db"


def _config(tmp_path: Path) -> Config:
    return replace(create_usearch_config(str(tmp_path)), enable_smart_replace=False)


def _lazy(config: Config) -> Config:
    return replace(config, eager_initialization=False)


def _new_logger() -> MagicMock:
    return MagicMock(spec=StructuredLogger)


def _open(config: Config, logger: MagicMock) -> MemoryManager:
    """Construct a real manager with a mock embedder and a caller-owned logger."""
    with patch(
        "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
        return_value=MockEmbedder(dims=DIMS),
    ):
        return MemoryManager(config, cast(IStructuredLogger, logger))


def _close(manager: MemoryManager) -> None:
    manager.close()


def _checkpoint(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()


def _populate(config: Config, names: tuple[str, ...]) -> dict[str, int]:
    manager, _logger = create_memory_manager(config)
    try:
        assert manager.add_memories([CONTENTS[name] for name in names]) == len(names)
        ids: dict[str, int] = {}
        for name in names:
            row_id = manager.get_id_by_content(CONTENTS[name])
            assert row_id is not None
            ids[name] = row_id
    finally:
        _close(manager)
    return ids


def _build(tmp_path: Path, names: tuple[str, ...] = tuple(CONTENTS)) -> Workspace:
    config = _config(tmp_path)
    ids = _populate(config, names)
    workspace = Workspace(config, ids)
    _checkpoint(workspace.db_path)
    return workspace


def _load_index(path: Path) -> Index:
    index = Index.restore(str(path))
    assert index is not None
    return index


def _vector_keys(path: Path) -> frozenset[int]:
    keys: list[int] = list(_load_index(path).keys)
    return frozenset(keys)


def _sqlite_ids(workspace: Workspace) -> frozenset[int]:
    snapshot = read_memory_store_snapshot(
        str(workspace.db_path), workspace.workspace_id
    )
    return frozenset(snapshot.contents_by_id)


def _difference(workspace: Workspace) -> tuple[frozenset[int], frozenset[int]]:
    """Return (rows without a vector, vectors without a row) from the files."""
    rows = _sqlite_ids(workspace)
    keys = _vector_keys(workspace.index_path)
    return rows - keys, keys - rows


def _drop_vector(workspace: Workspace, row_id: int) -> None:
    index = _load_index(workspace.index_path)
    index.remove(row_id)
    index.save(str(workspace.index_path))


def _add_vector(workspace: Workspace, key: int) -> None:
    index = _load_index(workspace.index_path)
    index.add(key, np.random.default_rng(key).random(DIMS).astype(np.float32))
    index.save(str(workspace.index_path))


def _delete_row(workspace: Workspace, row_id: int) -> None:
    connection = sqlite3.connect(workspace.db_path)
    try:
        _ = connection.execute("DELETE FROM memories WHERE id = ?", (row_id,))
        connection.commit()
    finally:
        connection.close()
    _checkpoint(workspace.db_path)


def _journal(workspace: Workspace, record: Callable[[MemoryStore], object]) -> None:
    store = MemoryStore(db_path=str(workspace.db_path))
    try:
        _ = record(store)
    finally:
        store.close()
    _checkpoint(workspace.db_path)


def _drifted(tmp_path: Path, drift: Drift) -> Workspace:
    match drift:
        case Drift.MISSING_VECTOR:
            workspace = _build(tmp_path)
            _drop_vector(workspace, workspace.ids["beta"])
        case Drift.ORPHAN_VECTOR:
            workspace = _build(tmp_path)
            _delete_row(workspace, workspace.ids["beta"])
        case Drift.SWAPPED_IDS:
            workspace = _build(tmp_path)
            _drop_vector(workspace, workspace.ids["beta"])
            _add_vector(workspace, max(workspace.ids.values()) + 10)
        case Drift.OLDER_INDEX:
            config = _config(tmp_path)
            first = ("alpha", "beta")
            ids = _populate(config, first)
            workspace = Workspace(config, ids)
            older = workspace.directory / "older.usearch"
            _ = shutil.copy2(workspace.index_path, older)
            ids.update(_populate(config, ("gamma", "delta")))
            _ = shutil.copy2(older, workspace.index_path)
            _checkpoint(workspace.db_path)
    missing, orphans = _difference(workspace)
    assert missing or orphans, "fixture must produce drift"
    if drift is Drift.SWAPPED_IDS:
        assert len(_sqlite_ids(workspace)) == len(_vector_keys(workspace.index_path))
    return workspace


@dataclass(frozen=True)
class FileState:
    """Bytes and mtimes that a refusal must leave untouched.

    ``-shm`` is excluded on purpose: it is SQLite's shared wal-index and any
    reader updates it. The WAL is compared by content with absent == empty.
    """

    vector: tuple[str, int]
    database: tuple[str, int]
    wal_digest: str


def _digest(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def _file_state(workspace: Workspace) -> FileState:
    wal = Path(f"{workspace.db_path}-wal")
    wal_bytes = wal.read_bytes() if wal.exists() else b""
    return FileState(
        vector=_digest(workspace.index_path),
        database=_digest(workspace.db_path),
        wal_digest=hashlib.sha256(wal_bytes).hexdigest(),
    )


def _id_list(ids: frozenset[int]) -> str:
    return ", ".join(str(item) for item in sorted(ids))


def _assert_refusal_text(text: str, workspace: Workspace) -> None:
    """The operator sentence, the exact id diagnostics, and no memory text."""
    missing, orphans = _difference(workspace)
    assert INDEX_DRIFT_OPERATOR_ACTION in text
    if missing:
        assert (
            f"{len(missing)} rows without a vector (ids: {_id_list(missing)})" in text
        )
    if orphans:
        assert (
            f"{len(orphans)} vectors without a row (ids: {_id_list(orphans)})" in text
        )
    for content in CONTENTS.values():
        assert content not in text


def _assert_no_memory_text_logged(logger: MagicMock) -> None:
    logged = " ".join(repr(call) for call in logger.mock_calls)
    for content in (*CONTENTS.values(), FRESH):
        assert content not in logged


def _construct_expecting_refusal(
    workspace: Workspace, logger: MagicMock
) -> InitializationError:
    try:
        manager = _open(workspace.config, logger)
    except InitializationError as error:
        return error
    _close(manager)
    pytest.fail("a workspace whose ids differ was opened without any error")


async def _invoke(
    manager: MemoryManager, workspace: Workspace, operation: Operation
) -> object:
    match operation:
        case Operation.ADD:
            return manager.add_memories([FRESH])
        case Operation.ADD_ASYNC:
            return await manager.add_memories_async([FRESH])
        case Operation.SEARCH:
            return await manager.search(CONTENTS["alpha"])
        case Operation.REMOVE:
            return manager.delete_memories([CONTENTS["alpha"]])


REFUSAL_TYPES: dict[Operation, tuple[type[Exception], ...]] = {
    operation: (InitializationError,) for operation in Operation
}


class TestRefusalAtConstruction:
    """Eager initialization opens the index while the manager is built."""

    @pytest.mark.parametrize("drift", list(Drift))
    def test_manager_construction_refuses_drift(
        self, tmp_path: Path, drift: Drift
    ) -> None:
        workspace = _drifted(tmp_path, drift)
        before = _file_state(workspace)
        logger = _new_logger()

        error = _construct_expecting_refusal(workspace, logger)

        _assert_refusal_text(str(error), workspace)
        assert _file_state(workspace) == before
        _assert_no_memory_text_logged(logger)

    @pytest.mark.parametrize("drift", [Drift.MISSING_VECTOR, Drift.SWAPPED_IDS])
    async def test_registry_acquire_carries_the_operator_text(
        self, tmp_path: Path, drift: Drift
    ) -> None:
        workspace = _drifted(tmp_path, drift)
        registry = WorkspaceRegistry(
            workspace.config,
            manager_factory=lambda config: create_memory_manager(config)[0],
        )
        try:
            with pytest.raises(InitializationError) as caught:
                async with registry.acquire(workspace.workspace_id):
                    pass
        finally:
            await registry.close()

        _assert_refusal_text(str(caught.value), workspace)


class TestRefusalOfEveryOperationWhenLazy:
    """With eager initialization off the first operation opens the index."""

    @pytest.mark.parametrize("operation", list(Operation))
    @pytest.mark.parametrize("drift", list(Drift))
    async def test_operation_refuses_drift(
        self, tmp_path: Path, drift: Drift, operation: Operation
    ) -> None:
        workspace = _drifted(tmp_path, drift)
        before = _file_state(workspace)
        logger = _new_logger()
        manager = _open(_lazy(workspace.config), logger)
        try:
            if operation is Operation.SEARCH:
                tantivy = manager._tantivy_engine
                assert isinstance(tantivy, TantivyEngine)
                assert CONTENTS["alpha"] in tantivy.find_by_exact_match(
                    workspace.workspace_id, CONTENTS["alpha"]
                ), "live full-text hits would let a swallowed error degrade search"
            with pytest.raises(REFUSAL_TYPES[operation]) as caught:
                _ = await _invoke(manager, workspace, operation)
            _assert_refusal_text(str(caught.value), workspace)
        finally:
            _close(manager)

        assert _file_state(workspace) == before
        _assert_no_memory_text_logged(logger)

    @pytest.mark.parametrize("drift", [Drift.MISSING_VECTOR, Drift.SWAPPED_IDS])
    @pytest.mark.parametrize(
        "tool_name", ["add_memories", "search_memories", "delete_memories"]
    )
    async def test_tool_wrapped_error_carries_the_operator_text(
        self, tmp_path: Path, drift: Drift, tool_name: str
    ) -> None:
        workspace = _drifted(tmp_path, drift)
        logger = _new_logger()
        config = _lazy(workspace.config)
        manager = _open(config, logger)
        try:
            call: Awaitable[object]
            expected: tuple[type[Exception], ...]
            match tool_name:
                case "add_memories":
                    call = AddMemoriesTool(config, manager, logger).get_handler()(
                        [FRESH]
                    )
                    expected = (StorageError,)
                case "search_memories":
                    call = SearchMemoriesTool(config, manager, logger).get_handler()(
                        CONTENTS["alpha"]
                    )
                    expected = (SearchError,)
                case _:
                    rows, _ = manager.get_page_records_with_total()
                    alpha_id = next(
                        row.id for row in rows if row.content == CONTENTS["alpha"]
                    )
                    call = DeleteMemoriesTool(config, manager, logger).get_handler()(
                        [alpha_id]
                    )
                    expected = (StorageError,)
            with pytest.raises(expected) as caught:
                _ = await call
            assert str(caught.value).startswith("Failed to ")
            _assert_refusal_text(str(caught.value), workspace)
        finally:
            _close(manager)

    @pytest.mark.parametrize("drift", list(Drift))
    def test_get_all_keeps_reading_sqlite(self, tmp_path: Path, drift: Drift) -> None:
        """DRIFT-4 guard: passes today; the drift check must not break it."""
        workspace = _drifted(tmp_path, drift)
        manager = _open(_lazy(workspace.config), _new_logger())
        try:
            snapshot = read_memory_store_snapshot(
                str(workspace.db_path), workspace.workspace_id
            )
            stored = list(snapshot.contents_by_id.values())

            assert manager.get_all() == stored
            page, total = manager.get_page_with_total()
            assert page == stored
            assert total == len(stored)
        finally:
            _close(manager)


class TestConsistentWorkspacesStillOpen:
    """Guards that pass today: agreeing ids must never be refused."""

    def test_removed_then_reloaded_workspace_opens_and_keys_match_rows(
        self, tmp_path: Path
    ) -> None:
        workspace = _build(tmp_path)
        doomed = [CONTENTS["beta"], CONTENTS["delta"]]
        manager, _logger = create_memory_manager(workspace.config)
        try:
            assert sorted(manager.delete_memories(doomed)) == sorted(doomed)
        finally:
            _close(manager)
        remaining = {workspace.ids["alpha"], workspace.ids["gamma"]}

        # Real on-disk save/reload: removed keys are absent from the file.
        assert _vector_keys(workspace.index_path) == remaining
        assert _sqlite_ids(workspace) == remaining

        reopened, _logger = create_memory_manager(workspace.config)
        try:
            engine = reopened._semantic_engine
            assert isinstance(engine, USearchEngine)
            assert {int(key) for key in engine.index.keys} == remaining
            assert sorted(reopened.get_all()) == sorted(
                [CONTENTS["alpha"], CONTENTS["gamma"]]
            )
        finally:
            _close(reopened)

    def test_open_and_close_of_a_consistent_workspace_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        """Validates the no-write baseline used by the refusal tests."""
        workspace = _build(tmp_path)
        before = _file_state(workspace)

        manager, _logger = create_memory_manager(workspace.config)
        try:
            assert len(manager.get_all()) == len(CONTENTS)
        finally:
            _close(manager)

        assert _file_state(workspace) == before


def _boom_commit(_engine: USearchEngine) -> None:
    raise RuntimeError("crash before usearch save")


class TestPendingJournalStatesStillConverge:
    """Crash states that restart recovery repairs must keep opening cleanly.

    Guards that pass today: after recovery, SQLite ids equal live vector keys.
    """

    @staticmethod
    def _assert_converged(workspace: Workspace, manager: MemoryManager) -> None:
        engine = manager._semantic_engine
        assert isinstance(engine, USearchEngine)
        live_keys = {int(key) for key in engine.index.keys}
        assert live_keys == set(_sqlite_ids(workspace))
        assert manager.pending_intent_count() == 0

    def test_pending_add_with_unsaved_vector_converges(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        first, _logger = create_memory_manager(config)
        try:
            assert first.add_memories([CONTENTS["alpha"]]) == 1
            with patch.object(USearchEngine, "commit", _boom_commit):
                with pytest.raises((StorageError, RuntimeError)):
                    _ = first.add_memories([CONTENTS["beta"]])
        finally:
            _abandon_without_persist(first)

        second, _logger = create_memory_manager(config)
        try:
            workspace = Workspace(config, {})
            assert sorted(second.get_all()) == sorted(
                [CONTENTS["alpha"], CONTENTS["beta"]]
            )
            self._assert_converged(workspace, second)
        finally:
            _close(second)

    def test_pending_delete_with_unsaved_removal_converges(
        self, tmp_path: Path
    ) -> None:
        config = _config(tmp_path)
        first, _logger = create_memory_manager(config)
        try:
            assert first.add_memories([CONTENTS["alpha"], CONTENTS["beta"]]) == 2
            with patch.object(USearchEngine, "commit", _boom_commit):
                with pytest.raises((StorageError, RuntimeError)):
                    _ = first.delete_by_memory(CONTENTS["beta"])
        finally:
            _abandon_without_persist(first)

        second, _logger = create_memory_manager(config)
        try:
            workspace = Workspace(config, {})
            assert second.get_all() == [CONTENTS["alpha"]]
            tantivy = second._tantivy_engine
            assert isinstance(tantivy, TantivyEngine)
            assert not tantivy.find_by_exact_match(
                second.workspace_id, CONTENTS["beta"]
            )
            self._assert_converged(workspace, second)
        finally:
            _close(second)

    def test_pending_add_whose_vector_is_missing_converges(
        self, tmp_path: Path
    ) -> None:
        workspace = _build(tmp_path)
        _drop_vector(workspace, workspace.ids["delta"])
        _journal(
            workspace,
            lambda store: store.begin_add_intents(
                workspace.workspace_id, [CONTENTS["delta"]]
            ),
        )

        manager, _logger = create_memory_manager(workspace.config)
        try:
            assert sorted(manager.get_all()) == sorted(CONTENTS.values())
            self._assert_converged(workspace, manager)
        finally:
            _close(manager)

    def test_pending_delete_whose_vector_is_still_present_converges(
        self, tmp_path: Path
    ) -> None:
        """Recorded id is an orphan: the row is gone, the vector is not."""
        workspace = _build(tmp_path)
        beta_id = workspace.ids["beta"]
        _journal(
            workspace,
            lambda store: store.begin_delete_intents(
                workspace.workspace_id, [(beta_id, CONTENTS["beta"])]
            ),
        )
        _delete_row(workspace, beta_id)
        assert _difference(workspace) == (frozenset(), frozenset({beta_id}))

        manager, _logger = create_memory_manager(workspace.config)
        try:
            assert CONTENTS["beta"] not in manager.get_all()
            self._assert_converged(workspace, manager)
        finally:
            _close(manager)

    def test_pending_delete_whose_row_is_still_present_converges(
        self, tmp_path: Path
    ) -> None:
        """A durable removal lost with the WAL: row present, vector gone."""
        workspace = _build(tmp_path)
        beta_id = workspace.ids["beta"]
        _journal(
            workspace,
            lambda store: store.begin_delete_intents(
                workspace.workspace_id, [(beta_id, CONTENTS["beta"])]
            ),
        )
        _drop_vector(workspace, beta_id)
        assert _difference(workspace) == (frozenset({beta_id}), frozenset())

        manager, _logger = create_memory_manager(workspace.config)
        try:
            assert CONTENTS["beta"] not in manager.get_all()
            self._assert_converged(workspace, manager)
        finally:
            _close(manager)

    def test_pending_delete_of_the_last_memory_keeps_the_existing_refusal(
        self, tmp_path: Path
    ) -> None:
        """Characterization: empty SQLite plus populated HNSW stays refused.

        Deleting the only memory and crashing before the index is saved leaves
        no rows but one vector. The pre-existing hard rule (no HNSW load when
        SQLite is empty) refuses it, even though a pending DELETE explains the
        difference. The drift check must not weaken that rule.
        """
        config = _config(tmp_path)
        first, _logger = create_memory_manager(config)
        try:
            assert first.add_memories([CONTENTS["alpha"]]) == 1
            with patch.object(USearchEngine, "commit", _boom_commit):
                with pytest.raises((StorageError, RuntimeError)):
                    _ = first.delete_by_memory(CONTENTS["alpha"])
        finally:
            _abandon_without_persist(first)
        workspace = Workspace(config, {})
        _checkpoint(workspace.db_path)
        before = _file_state(workspace)

        with pytest.raises(
            InitializationError,
            match="Refusing to load HNSW without a readable memory store",
        ):
            _ = create_memory_manager(config)

        assert _file_state(workspace) == before


class TestPendingJournalCannotMaskUnrelatedDrift:
    """A non-empty journal accounts only for its own identities."""

    def test_pending_add_does_not_excuse_an_unrelated_missing_vector(
        self, tmp_path: Path
    ) -> None:
        workspace = _build(tmp_path)
        excused = workspace.ids["delta"]
        unrelated = workspace.ids["beta"]
        _drop_vector(workspace, excused)
        _drop_vector(workspace, unrelated)
        _journal(
            workspace,
            lambda store: store.begin_add_intents(
                workspace.workspace_id, [CONTENTS["delta"]]
            ),
        )
        before = _file_state(workspace)
        logger = _new_logger()

        error = _construct_expecting_refusal(workspace, logger)

        text = str(error)
        assert INDEX_DRIFT_OPERATOR_ACTION in text
        assert f"1 rows without a vector (ids: {unrelated})" in text
        assert str(excused) not in text.split("ids:")[1]
        assert _file_state(workspace) == before
        _assert_no_memory_text_logged(logger)

    def test_pending_delete_does_not_excuse_an_unrelated_orphan_vector(
        self, tmp_path: Path
    ) -> None:
        workspace = _build(tmp_path)
        recorded, unrelated = 77, 88
        _add_vector(workspace, recorded)
        _add_vector(workspace, unrelated)
        _journal(
            workspace,
            lambda store: store.begin_delete_intents(
                workspace.workspace_id, [(recorded, "a memory that was deleted")]
            ),
        )
        before = _file_state(workspace)
        logger = _new_logger()

        error = _construct_expecting_refusal(workspace, logger)

        text = str(error)
        assert INDEX_DRIFT_OPERATOR_ACTION in text
        assert f"1 vectors without a row (ids: {unrelated})" in text
        assert _file_state(workspace) == before
        _assert_no_memory_text_logged(logger)

    def test_pending_delete_of_the_recorded_orphan_alone_still_converges(
        self, tmp_path: Path
    ) -> None:
        """Guard that passes today: the recorded id is the only difference."""
        workspace = _build(tmp_path)
        recorded = 77
        _add_vector(workspace, recorded)
        _journal(
            workspace,
            lambda store: store.begin_delete_intents(
                workspace.workspace_id, [(recorded, "a memory that was deleted")]
            ),
        )

        manager, _logger = create_memory_manager(workspace.config)
        try:
            assert sorted(manager.get_all()) == sorted(CONTENTS.values())
        finally:
            _close(manager)

        assert _difference(workspace) == (frozenset(), frozenset())


class TestPreflightPlacement:
    """Where the index is opened and verified relative to reads and writes."""

    @pytest.mark.parametrize("drift", list(Drift))
    def test_sqlite_only_reads_leave_the_index_unloaded(
        self, tmp_path: Path, drift: Drift
    ) -> None:
        workspace = _drifted(tmp_path, drift)
        before = _file_state(workspace)
        manager = _open(_lazy(workspace.config), _new_logger())
        try:
            engine = manager._semantic_engine
            assert isinstance(engine, USearchEngine)

            assert len(manager.get_all()) == manager.count()
            _ = manager.get_page_with_total()
            _ = manager.search_for_removal(CONTENTS["alpha"])

            assert engine._index is None
        finally:
            _close(manager)
        assert _file_state(workspace) == before

    async def test_async_add_with_a_replacement_is_refused_before_the_transition(
        self, tmp_path: Path
    ) -> None:
        workspace = _drifted(tmp_path, Drift.SWAPPED_IDS)
        before = _file_state(workspace)
        replacement = ReplacementInfo(
            old_memory=CONTENTS["alpha"],
            new_memory=FRESH,
            confidence=0.95,
            reason="updated",
        )
        manager = _open(_lazy(workspace.config), _new_logger())
        try:
            with patch.object(
                SmartReplacementPhase,
                "execute",
                new=AsyncMock(
                    return_value=Phase2Result({FRESH: [replacement]}, 1, 0.0)
                ),
            ):
                with pytest.raises(InitializationError) as caught:
                    _ = await manager.add_memories_async([FRESH])
            _assert_refusal_text(str(caught.value), workspace)
        finally:
            _close(manager)

        assert _file_state(workspace) == before

    @pytest.mark.parametrize("operation", list(Operation))
    async def test_drift_published_after_an_eager_open_is_refused_before_any_write(
        self, tmp_path: Path, operation: Operation
    ) -> None:
        workspace = _build(tmp_path)
        logger = _new_logger()
        manager = _open(workspace.config, logger)
        try:
            _drop_vector(workspace, workspace.ids["beta"])
            before = _file_state(workspace)

            with pytest.raises(InitializationError) as caught:
                _ = await _invoke(manager, workspace, operation)

            _assert_refusal_text(str(caught.value), workspace)
            assert _file_state(workspace) == before
        finally:
            _close(manager)
        _assert_no_memory_text_logged(logger)


class TestRefusedSearchIsCountedOnce:
    """A refusal at preflight never reaches the pipeline, so it is counted there."""

    async def test_lazy_search_refusal_counts_one_semantic_failure(
        self, tmp_path: Path
    ) -> None:
        workspace = _drifted(tmp_path, Drift.OLDER_INDEX)
        manager = _open(_lazy(workspace.config), _new_logger())
        try:
            tantivy = manager._tantivy_engine
            assert isinstance(tantivy, TantivyEngine)
            assert CONTENTS["alpha"] in tantivy.find_by_exact_match(
                workspace.workspace_id, CONTENTS["alpha"]
            ), "live full-text hits would let a swallowed error degrade search"

            with pytest.raises(InitializationError) as caught:
                _ = await manager.search(CONTENTS["alpha"])

            _assert_refusal_text(str(caught.value), workspace)
            failures = manager.search_failure_snapshot()
            assert failures.semantic.count == 1
            assert failures.semantic.exception_type == "InitializationError"
            assert failures.tantivy.count == 0
            assert failures.cross_encoder.count == 0
            assert failures.openrouter_reranker.count == 0
        finally:
            _close(manager)


def _assert_search_failures(manager: MemoryManager, *, semantic: int) -> None:
    """Only the semantic counter can move for a vector-index refusal."""
    failures = manager.search_failure_snapshot()
    assert failures.semantic.count == semantic
    if semantic:
        assert failures.semantic.exception_type == "InitializationError"
        assert failures.semantic.last_failure_at is not None
    else:
        assert failures.semantic.exception_type is None
    assert failures.tantivy.count == 0
    assert failures.cross_encoder.count == 0
    assert failures.openrouter_reranker.count == 0


def _lose_an_unrelated_row_and_journal_a_converged_add(workspace: Workspace) -> None:
    """Crash-like state an open manager cannot see until it reconciles.

    A pending ADD for a memory that is already complete makes recovery run (and
    succeed), while an unrelated row vanished from SQLite behind the open
    manager's in-memory index: after recovery the ids still disagree and the
    pending row accounts for none of the difference.
    """
    _delete_row(workspace, workspace.ids["beta"])
    _journal(
        workspace,
        lambda store: store.begin_add_intents(
            workspace.workspace_id, [CONTENTS["delta"]]
        ),
    )


class TestSearchRefusalsRaisedBeforeThePipelineAreCountedOnce:
    """Refresh and recovery refusals escape ``search`` before the pipeline runs."""

    async def test_a_drifted_file_published_after_an_eager_open_counts_once(
        self, tmp_path: Path
    ) -> None:
        workspace = _build(tmp_path)
        manager = _open(workspace.config, _new_logger())
        try:
            _drop_vector(workspace, workspace.ids["beta"])
            before = _file_state(workspace)

            with pytest.raises(InitializationError) as caught:
                _ = await manager.search(CONTENTS["alpha"])

            _assert_refusal_text(str(caught.value), workspace)
            _assert_search_failures(manager, semantic=1)
            assert _file_state(workspace) == before

            # Every retry refuses again and is counted once per call.
            with pytest.raises(InitializationError):
                _ = await manager.search(CONTENTS["alpha"])
            _assert_search_failures(manager, semantic=2)
        finally:
            _close(manager)

    async def test_a_refusal_during_recovery_inside_search_counts_once(
        self, tmp_path: Path
    ) -> None:
        workspace = _build(tmp_path)
        manager = _open(workspace.config, _new_logger())
        try:
            _lose_an_unrelated_row_and_journal_a_converged_add(workspace)

            with pytest.raises(InitializationError) as caught:
                _ = await manager.search(CONTENTS["alpha"])

            assert INDEX_DRIFT_OPERATOR_ACTION in str(caught.value)
            assert f"1 vectors without a row (ids: {workspace.ids['beta']})" in str(
                caught.value
            )
            _assert_search_failures(manager, semantic=1)
        finally:
            _close(manager)

    async def test_the_pipeline_still_counts_its_own_failures_once(
        self, tmp_path: Path
    ) -> None:
        """No double count: a refusal that reaches the pipeline is its own."""
        workspace = _drifted(tmp_path, Drift.OLDER_INDEX)
        manager = _open(_lazy(workspace.config), _new_logger())
        try:
            with pytest.raises(InitializationError):
                _ = await manager.search(CONTENTS["alpha"])
            _assert_search_failures(manager, semantic=1)
        finally:
            _close(manager)


class TestAddAndRemoveRefusalsDoNotTouchTheSearchCounters:
    """Search counters describe search only, whichever operation was refused."""

    @pytest.mark.parametrize(
        "operation", [Operation.ADD, Operation.ADD_ASYNC, Operation.REMOVE]
    )
    async def test_a_lazy_preflight_refusal(
        self, tmp_path: Path, operation: Operation
    ) -> None:
        workspace = _drifted(tmp_path, Drift.MISSING_VECTOR)
        manager = _open(_lazy(workspace.config), _new_logger())
        try:
            with pytest.raises(InitializationError):
                _ = await _invoke(manager, workspace, operation)
            _assert_search_failures(manager, semantic=0)
        finally:
            _close(manager)

    @pytest.mark.parametrize(
        "operation", [Operation.ADD, Operation.ADD_ASYNC, Operation.REMOVE]
    )
    async def test_a_reload_refusal_after_an_eager_open(
        self, tmp_path: Path, operation: Operation
    ) -> None:
        workspace = _build(tmp_path)
        manager = _open(workspace.config, _new_logger())
        try:
            _drop_vector(workspace, workspace.ids["beta"])

            with pytest.raises(InitializationError):
                _ = await _invoke(manager, workspace, operation)

            _assert_search_failures(manager, semantic=0)
        finally:
            _close(manager)

    async def test_a_recovery_refusal_inside_add(self, tmp_path: Path) -> None:
        workspace = _build(tmp_path)
        manager = _open(workspace.config, _new_logger())
        try:
            _lose_an_unrelated_row_and_journal_a_converged_add(workspace)

            with pytest.raises(InitializationError):
                _ = manager.add_memories([FRESH])

            _assert_search_failures(manager, semantic=0)
        finally:
            _close(manager)


class TestRecoveryReverification:
    """Recovery that leaves the ids inconsistent must not open the workspace."""

    def test_a_recovery_that_leaves_drift_is_refused(self, tmp_path: Path) -> None:
        workspace = _build(tmp_path)
        _drop_vector(workspace, workspace.ids["delta"])
        _journal(
            workspace,
            lambda store: store.begin_add_intents(
                workspace.workspace_id, [CONTENTS["delta"]]
            ),
        )
        beta = workspace.ids["beta"]
        original = replacement_recovery.apply_pending_transition

        def converge_then_lose_a_vector(
            transition: ReplacementTransition,
            *,
            semantic_engine: ISemanticSearchEngine,
            tantivy_engine: TantivyEngine | None,
            logger: IStructuredLogger,
            precomputed_vectors: dict[str, list[float]] | None = None,
            coordinator: IStorageCoordinator | None = None,
            orchestration_hook: Callable[[str], None] | None = None,
        ) -> bool:
            """A faulty recovery: it converges, then drops an unrelated vector."""
            converged = original(
                transition,
                semantic_engine=semantic_engine,
                tantivy_engine=tantivy_engine,
                logger=logger,
                precomputed_vectors=precomputed_vectors,
                coordinator=coordinator,
                orchestration_hook=orchestration_hook,
            )
            assert isinstance(semantic_engine, USearchEngine)
            semantic_engine.index.remove(beta)
            return converged

        logger = _new_logger()
        with patch.object(
            replacement_recovery,
            "apply_pending_transition",
            converge_then_lose_a_vector,
        ):
            error = _construct_expecting_refusal(workspace, logger)

        assert INDEX_DRIFT_OPERATOR_ACTION in str(error)
        assert f"1 rows without a vector (ids: {beta})" in str(error)
        _assert_no_memory_text_logged(logger)
