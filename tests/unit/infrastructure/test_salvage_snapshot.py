"""Unit tests for the read-only offline content-salvage snapshot."""

import base64
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3

import pytest

from reflectlog.core.exceptions import (
    SalvageDamagedDatabaseError,
    SalvageError,
    SalvageInputError,
    SalvageMalformedRecordError,
    SalvageUnsupportedSchemaError,
)
from reflectlog.core.salvage import (
    ISalvageSnapshotReader,
    JournalStatus,
    RawValue,
    RawValueKind,
    SalvageSnapshot,
)
from reflectlog.core.types import ReplacementTransitionRequest
from reflectlog.infrastructure.memory_store import MemoryStore
from reflectlog.infrastructure.salvage_snapshot import SqliteSalvageSnapshotReader

WS = "salvage-ws"
OTHER = "other-ws"
SECRET = "SECRET-MEMORY-TEXT"

MEMORIES_DDL = (
    "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "workspace_id TEXT NOT NULL, content TEXT NOT NULL, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
)
JOURNAL_DDL = (
    "CREATE TABLE replacement_transitions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "workspace_id TEXT NOT NULL, old_memory_id INTEGER NOT NULL, "
    "old_content TEXT NOT NULL, new_content TEXT NOT NULL, "
    "archive_id INTEGER NOT NULL, reason TEXT NOT NULL, "
    "confidence REAL NOT NULL, status TEXT NOT NULL{kind})"
)
KIND_COLUMN = ", kind TEXT NOT NULL DEFAULT 'replace'"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "memories.db"


def _read(path: Path, workspace_id: str = WS) -> SalvageSnapshot:
    return SqliteSalvageSnapshotReader().read_snapshot(str(path), workspace_id)


def _execute(path: Path, *statements: str | tuple[str, tuple[object, ...]]) -> None:
    connection = sqlite3.connect(path)
    try:
        for statement in statements:
            if isinstance(statement, str):
                _ = connection.execute(statement)
            else:
                _ = connection.execute(*statement)
        connection.commit()
    finally:
        connection.close()


def _plain_db(path: Path, *, kind: bool = True, journal: bool = True) -> None:
    statements = [MEMORIES_DDL]
    if journal:
        statements.append(JOURNAL_DDL.format(kind=KIND_COLUMN if kind else ""))
    _execute(path, *statements)
    connection = sqlite3.connect(path)
    try:
        _ = connection.execute("PRAGMA journal_mode=DELETE")
    finally:
        connection.close()


def _memory(
    path: Path, workspace: str, content: str, created: str | None = "t"
) -> None:
    _execute(
        path,
        (
            "INSERT INTO memories (workspace_id, content, created_at) VALUES (?, ?, ?)",
            (workspace, content, created),
        ),
    )


def _journal(
    path: Path,
    workspace: str,
    status: str,
    kind: str | None = "add",
    new_content: str = "intent",
) -> None:
    columns = "workspace_id, old_memory_id, old_content, new_content, archive_id, "
    columns += "reason, confidence, status"
    values: tuple[object, ...] = (workspace, 0, "", new_content, 0, "add", 1.0, status)
    if kind is not None:
        columns += ", kind"
        values = (*values, kind)
    marks = ", ".join("?" * len(values))
    _execute(
        path,
        (f"INSERT INTO replacement_transitions ({columns}) VALUES ({marks})", values),
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reader_satisfies_the_protocol() -> None:
    assert isinstance(SqliteSalvageSnapshotReader(), ISalvageSnapshotReader)


class TestWalVisibility:
    def test_committed_but_uncheckpointed_wal_rows_are_exported(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        crashed = tmp_path / "crashed"
        crashed.mkdir()
        store = MemoryStore(db_path=str(db_path))
        try:
            row_id = store.insert(WS, "committed only in the wal")
            wal = Path(f"{db_path}-wal")
            assert wal.stat().st_size > 0
            # -shm is deliberately not copied: recovery needs only db + wal.
            _ = shutil.copy2(db_path, crashed / "memories.db")
            _ = shutil.copy2(wal, crashed / "memories.db-wal")
            # The row exists only in the WAL: the main file alone lacks it.
            assert b"committed only in the wal" not in db_path.read_bytes()
        finally:
            store.close()

        snapshot = _read(crashed / "memories.db")

        assert [(r.source_id, r.content) for r in snapshot.records] == [
            (row_id, "committed only in the wal")
        ]


class TestActiveRecords:
    def test_only_requested_workspace_is_exported_sorted_by_id(
        self, db_path: Path
    ) -> None:
        _plain_db(db_path)
        _memory(db_path, WS, "b", "2026-01-02")
        _memory(db_path, OTHER, f"{SECRET} foreign")
        _memory(db_path, WS, "a", None)

        snapshot = _read(db_path)

        assert [(r.source_id, r.content, r.created_at) for r in snapshot.records] == [
            (1, "b", "2026-01-02"),
            (3, "a", None),
        ]
        assert snapshot.workspace_id == WS
        assert snapshot.foreign_active_count == 1
        assert SECRET not in repr(snapshot.records)

    def test_unknown_workspace_has_no_records(self, db_path: Path) -> None:
        _plain_db(db_path)
        _memory(db_path, WS, "a")

        snapshot = _read(db_path, "nobody")

        assert snapshot.records == ()
        assert snapshot.foreign_active_count == 1


class TestJournalEvidence:
    def test_pending_intents_stay_evidence_and_never_become_records(
        self, db_path: Path
    ) -> None:
        store = MemoryStore(db_path=str(db_path))
        try:
            kept = store.insert(WS, "kept")
            _ = store.begin_add_intents(WS, [f"{SECRET} pending add"])
            _ = store.begin_delete_intents(WS, [(kept, "kept")])
            _ = store.begin_replacement_transitions(
                [
                    ReplacementTransitionRequest(
                        old_memory_id=kept,
                        workspace_id=WS,
                        old_content="kept",
                        new_content=f"{SECRET} replacement",
                        reason="update",
                        confidence=0.9,
                    )
                ]
            )
        finally:
            store.close()

        snapshot = _read(db_path)

        assert [r.content for r in snapshot.records] == ["kept"]
        assert [row.kind for row in snapshot.pending] == ["add", "delete", "replace"]
        assert {row.status for row in snapshot.pending} == {"pending"}
        assert snapshot.unknown == ()
        assert snapshot.foreign == ()
        assert snapshot.schema.journal_status is JournalStatus.SUPPORTED
        names = [name for name, _ in snapshot.pending[0].columns]
        assert names[:3] == ["id", "workspace_id", "old_memory_id"]
        assert "kind" in names

    def test_completed_history_is_counted_not_listed(self, db_path: Path) -> None:
        _plain_db(db_path)
        _journal(db_path, WS, "completed", "replace")
        _journal(db_path, WS, "completed", "add")

        snapshot = _read(db_path)

        assert snapshot.completed_history_count == 2
        assert snapshot.pending == snapshot.unknown == snapshot.foreign == ()

    @pytest.mark.parametrize(
        ("status", "kind"),
        [("weird", "add"), ("pending", "mystery"), ("completed", "mystery")],
    )
    def test_unrecognized_status_or_kind_is_unknown_evidence(
        self, db_path: Path, status: str, kind: str
    ) -> None:
        _plain_db(db_path)
        _journal(db_path, WS, status, kind)

        snapshot = _read(db_path)

        assert len(snapshot.unknown) == 1
        assert snapshot.pending == ()
        assert snapshot.completed_history_count == 0

    def test_foreign_workspace_rows_of_any_status_are_foreign_evidence(
        self, db_path: Path
    ) -> None:
        _plain_db(db_path)
        _journal(db_path, OTHER, "pending", "add")
        _journal(db_path, OTHER, "completed", "add")
        _journal(db_path, OTHER, "weird", "nope")

        snapshot = _read(db_path)

        assert len(snapshot.foreign) == 3
        assert snapshot.pending == snapshot.unknown == ()
        assert snapshot.completed_history_count == 0

    def test_legacy_journal_without_kind_is_not_defaulted_to_replace(
        self, db_path: Path
    ) -> None:
        _plain_db(db_path, kind=False)
        _journal(db_path, WS, "pending", None)
        _journal(db_path, WS, "completed", None)

        snapshot = _read(db_path)

        assert snapshot.schema.journal_status is JournalStatus.LEGACY_WITHOUT_KIND
        assert snapshot.pending == ()
        assert len(snapshot.unknown) == 1
        assert snapshot.unknown[0].kind is None
        assert snapshot.completed_history_count == 1

    def test_missing_journal_table_is_reported_missing(self, db_path: Path) -> None:
        _plain_db(db_path, journal=False)
        _memory(db_path, WS, "kept")

        snapshot = _read(db_path)

        assert snapshot.schema.journal_status is JournalStatus.MISSING
        assert [r.content for r in snapshot.records] == ["kept"]
        assert snapshot.pending == snapshot.unknown == snapshot.foreign == ()

    def test_journal_without_required_columns_is_unsupported_and_kept_raw(
        self, db_path: Path
    ) -> None:
        _execute(
            db_path,
            MEMORIES_DDL,
            "CREATE TABLE replacement_transitions (id INTEGER PRIMARY KEY, note TEXT)",
            "INSERT INTO replacement_transitions (note) VALUES ('raw note')",
        )

        snapshot = _read(db_path)

        assert snapshot.schema.journal_status is JournalStatus.UNSUPPORTED
        assert len(snapshot.unknown) == 1
        assert dict(snapshot.unknown[0].columns)["note"] == RawValue(
            RawValueKind.TEXT, "raw note"
        )
        assert snapshot.unknown[0].workspace_id is None

    def test_extra_journal_columns_and_values_are_preserved_losslessly(
        self, db_path: Path
    ) -> None:
        _plain_db(db_path)
        _execute(
            db_path,
            "ALTER TABLE replacement_transitions ADD COLUMN extra",
            (
                "INSERT INTO replacement_transitions (workspace_id, old_memory_id, "
                "old_content, new_content, archive_id, reason, confidence, status, "
                "kind, extra) VALUES (?, 7, 'o', ?, 3, 'r', 0.25, 'pending', 'add', ?)",
                (WS, "ok", b"\x00\xffblob"),
            ),
            (
                "INSERT INTO replacement_transitions (workspace_id, old_memory_id, "
                "old_content, new_content, archive_id, reason, confidence, status, "
                "kind, extra) VALUES (?, 7, 'o', CAST(? AS TEXT), 3, 'r', 0.25, "
                "'pending', 'add', NULL)",
                (WS, b"bad\xff"),
            ),
        )

        snapshot = _read(db_path)

        first, second = (dict(row.columns) for row in snapshot.pending)
        assert first["old_memory_id"] == RawValue(RawValueKind.INTEGER, 7)
        assert first["confidence"] == RawValue(RawValueKind.REAL, "0.25")
        assert first["new_content"] == RawValue(RawValueKind.TEXT, "ok")
        assert first["extra"] == RawValue(
            RawValueKind.BLOB, base64.b64encode(b"\x00\xffblob").decode()
        )
        assert second["extra"] == RawValue(RawValueKind.NULL)
        assert second["new_content"].kind is RawValueKind.TEXT_BYTES
        assert base64.b64decode(str(second["new_content"].value)) == b"bad\xff"

    def test_pending_row_with_non_text_workspace_is_foreign(
        self, db_path: Path
    ) -> None:
        _plain_db(db_path)
        _execute(
            db_path,
            "INSERT INTO replacement_transitions (workspace_id, old_memory_id, "
            "old_content, new_content, archive_id, reason, confidence, status, kind) "
            "VALUES (x'ff', 0, '', 'n', 0, 'add', 1.0, 'pending', 'add')",
        )

        snapshot = _read(db_path)

        assert len(snapshot.foreign) == 1
        assert snapshot.foreign[0].workspace_id is None


class TestMalformedRecords:
    def _expect(self, db_path: Path, reason: str) -> SalvageMalformedRecordError:
        with pytest.raises(SalvageMalformedRecordError, match=reason) as caught:
            _ = _read(db_path)
        assert SECRET not in str(caught.value)
        return caught.value

    def test_blob_content(self, db_path: Path) -> None:
        _plain_db(db_path)
        _memory(db_path, WS, SECRET)
        _execute(db_path, "UPDATE memories SET content = x'4142'")

        _ = self._expect(db_path, r"memories row id 1 content is not TEXT")

    def test_invalid_utf8_content(self, db_path: Path) -> None:
        _plain_db(db_path)
        _execute(
            db_path,
            (
                "INSERT INTO memories (workspace_id, content) "
                "VALUES (?, CAST(? AS TEXT))",
                (WS, SECRET.encode() + b"\xff"),
            ),
        )

        _ = self._expect(db_path, r"memories row id 1 content is not valid UTF-8")

    def test_empty_content(self, db_path: Path) -> None:
        _plain_db(db_path)
        _memory(db_path, WS, "")

        _ = self._expect(db_path, r"memories row id 1 content is empty")

    def test_integer_created_at(self, db_path: Path) -> None:
        _plain_db(db_path)
        _memory(db_path, WS, "fine")
        _execute(db_path, "UPDATE memories SET created_at = 5")

        _ = self._expect(db_path, r"memories row id 1 has created_at")

    def test_duplicate_content_within_workspace(self, db_path: Path) -> None:
        _execute(
            db_path,
            "CREATE TABLE memories (id INTEGER PRIMARY KEY, workspace_id TEXT, "
            "content TEXT, created_at TEXT)",
            ("INSERT INTO memories VALUES (1, ?, ?, 't')", (WS, SECRET)),
            ("INSERT INTO memories VALUES (2, ?, ?, 't')", (WS, SECRET)),
        )

        _ = self._expect(
            db_path, r"memories row id 2 duplicates the content of memories row id 1"
        )

    def test_same_content_in_other_workspace_is_not_a_duplicate(
        self, db_path: Path
    ) -> None:
        _plain_db(db_path)
        _memory(db_path, WS, "same")
        _memory(db_path, OTHER, "same")

        assert [r.content for r in _read(db_path).records] == ["same"]

    def test_malformed_foreign_rows_are_only_counted(self, db_path: Path) -> None:
        _plain_db(db_path)
        _memory(db_path, OTHER, "")
        _execute(db_path, "UPDATE memories SET content = x'ff'")

        snapshot = _read(db_path)

        assert snapshot.records == ()
        assert snapshot.foreign_active_count == 1

    def test_non_integer_id(self, db_path: Path) -> None:
        _execute(
            db_path,
            "CREATE TABLE memories (id TEXT, workspace_id TEXT, content TEXT, "
            "created_at TEXT)",
            ("INSERT INTO memories VALUES ('abc', ?, 'x', 't')", (WS,)),
        )

        _ = self._expect(db_path, r"memories row #1 has a non-integer id")

    def test_duplicate_ids(self, db_path: Path) -> None:
        _execute(
            db_path,
            "CREATE TABLE memories (id INTEGER, workspace_id TEXT, content TEXT, "
            "created_at TEXT)",
            ("INSERT INTO memories VALUES (4, ?, 'x', 't')", (WS,)),
            ("INSERT INTO memories VALUES (4, ?, 'y', 't')", (WS,)),
        )

        _ = self._expect(db_path, r"memories row id 4 is duplicated")


class TestUnsupportedAndDamaged:
    def test_missing_memories_table(self, db_path: Path) -> None:
        _execute(db_path, "CREATE TABLE other (x)")

        with pytest.raises(SalvageUnsupportedSchemaError, match="table is absent"):
            _ = _read(db_path)

    def test_missing_memories_column(self, db_path: Path) -> None:
        _execute(db_path, "CREATE TABLE memories (id INTEGER, workspace_id TEXT)")

        with pytest.raises(
            SalvageUnsupportedSchemaError, match="missing columns: content, created_at"
        ):
            _ = _read(db_path)

    def test_empty_database_file_is_unsupported_and_stays_empty(
        self, db_path: Path
    ) -> None:
        db_path.write_bytes(b"")

        with pytest.raises(SalvageUnsupportedSchemaError):
            _ = _read(db_path)

        assert db_path.read_bytes() == b""
        assert os.listdir(db_path.parent) == ["memories.db"]

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SalvageInputError, match="missing"):
            _ = _read(tmp_path / "absent.db")

    def test_garbage_file_is_damaged(self, db_path: Path) -> None:
        db_path.write_bytes(b"this is not a sqlite database" * 100)

        with pytest.raises(SalvageDamagedDatabaseError, match="unreadable"):
            _ = _read(db_path)

    def test_failed_quick_check_is_damaged(self, db_path: Path) -> None:
        _plain_db(db_path)
        _memory(db_path, WS, SECRET)
        raw = bytearray(db_path.read_bytes())
        raw[36:40] = (3).to_bytes(4, "big")  # freelist count the file does not have
        db_path.write_bytes(raw)

        with pytest.raises(SalvageDamagedDatabaseError, match="integrity") as caught:
            _ = _read(db_path)
        assert SECRET not in str(caught.value)

    def test_all_failures_are_salvage_errors(self, tmp_path: Path) -> None:
        with pytest.raises(SalvageError):
            _ = _read(tmp_path / "absent.db")


class TestReadOnlyContract:
    def test_database_bytes_are_unchanged_and_no_sidecars_appear(
        self, db_path: Path
    ) -> None:
        _plain_db(db_path)
        _memory(db_path, WS, "a")
        before = (_digest(db_path), db_path.stat().st_mtime_ns)

        _ = _read(db_path)
        _ = _read(db_path, OTHER)

        assert (_digest(db_path), db_path.stat().st_mtime_ns) == before
        assert os.listdir(db_path.parent) == ["memories.db"]

    def test_unsupported_schema_is_not_created_in_place(self, db_path: Path) -> None:
        _execute(db_path, "CREATE TABLE unrelated (x)")
        before = _digest(db_path)

        with pytest.raises(SalvageUnsupportedSchemaError):
            _ = _read(db_path)

        assert _digest(db_path) == before
        assert _table_names(db_path) == ["unrelated"]


def _table_names(path: Path) -> list[str]:
    connection = sqlite3.connect(path)
    try:
        return [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
    finally:
        connection.close()
