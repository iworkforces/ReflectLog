"""Strictly read-only SQLite snapshot for offline content salvage.

Opens ONLY the disposable copy it is handed, ``mode=ro`` (never ``immutable``,
which would hide committed WAL frames). It never builds ``MemoryStore``, never
replays journal rows, and never touches vector or full-text engines. Every
cell is read as ``typeof`` plus raw bytes and decoded strictly in Python, so
invalid UTF-8 cannot raise inside ``sqlite3``. Error messages carry row ids,
column names, and reasons, never memory content.
"""

from __future__ import annotations

import base64
from contextlib import closing, contextmanager
from dataclasses import replace
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import TYPE_CHECKING

from reflectlog.core.enums import TransitionKind, TransitionStatus
from reflectlog.core.exceptions import (
    SalvageDamagedDatabaseError,
    SalvageInputError,
    SalvageMalformedRecordError,
    SalvageUnsupportedSchemaError,
)
from reflectlog.core.salvage import (
    ActiveRecord,
    JournalStatus,
    RawJournalRow,
    RawValue,
    RawValueKind,
    SalvageSnapshot,
    SchemaReport,
    TableSchema,
)
from reflectlog.infrastructure.salvage_paths import is_directory_within

if TYPE_CHECKING:
    from collections.abc import Generator

_MEMORIES_TABLE = "memories"
_JOURNAL_TABLE = "replacement_transitions"
_MEMORIES_REQUIRED = ("id", "workspace_id", "content", "created_at")
_JOURNAL_REQUIRED = ("id", "workspace_id", "status")
_KINDS = frozenset(kind.value for kind in TransitionKind)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _cell_sql(column: str) -> str:
    """Select ``typeof``, the numeric value, and the raw bytes of one column."""
    quoted = _quote(column)
    return (
        f"typeof({quoted}), "
        f"CASE WHEN typeof({quoted}) IN ('integer', 'real') THEN {quoted} END, "
        f"CASE WHEN typeof({quoted}) IN ('text', 'blob') "
        f"THEN CAST({quoted} AS BLOB) END"
    )


def _raw_value(storage: object, number: object, data: object) -> RawValue:
    """Rebuild one lossless raw value from its three selected parts."""
    if storage == "null":
        return RawValue(RawValueKind.NULL)
    if storage == "integer" and isinstance(number, int):
        return RawValue(RawValueKind.INTEGER, number)
    if storage == "real" and isinstance(number, float):
        return RawValue(RawValueKind.REAL, repr(number))
    if storage in ("text", "blob") and isinstance(data, bytes):
        if storage == "blob":
            return RawValue(RawValueKind.BLOB, base64.b64encode(data).decode("ascii"))
        try:
            return RawValue(RawValueKind.TEXT, data.decode("utf-8"))
        except UnicodeDecodeError:
            return RawValue(
                RawValueKind.TEXT_BYTES, base64.b64encode(data).decode("ascii")
            )
    raise SalvageDamagedDatabaseError("SQLite returned an unexpected value type")


def _row_values(row: tuple[object, ...], count: int, start: int = 0) -> list[RawValue]:
    return [
        _raw_value(row[start + 3 * i], row[start + 3 * i + 1], row[start + 3 * i + 2])
        for i in range(count)
    ]


class SqliteSalvageSnapshotReader:
    """``ISalvageSnapshotReader`` over a read-only connection to a copy."""

    @contextmanager
    def read_snapshot(
        self, database_path: str, workspace_id: str
    ) -> Generator[SalvageSnapshot]:
        if not os.path.isfile(database_path):
            raise SalvageInputError("SQLite database is missing")
        uri = f"{Path(database_path).absolute().as_uri()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True, isolation_level=None)) as db:
                _ = db.execute("PRAGMA cache_size = -2048")
                _ = db.execute("PRAGMA temp_store = FILE")
                _ = db.execute("BEGIN")
                if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise SalvageDamagedDatabaseError(
                        "SQLite database failed its integrity check"
                    )
                snapshot = _read(db, workspace_id)
                with closing(_stream_records(db, workspace_id)) as records:
                    yield replace(snapshot, records=records)
        except sqlite3.Error as e:
            raise SalvageDamagedDatabaseError(
                f"SQLite database is unreadable: {e}"
            ) from e


def _read(db: sqlite3.Connection, workspace_id: str) -> SalvageSnapshot:
    names = [
        str(row[0])
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND substr(name, 1, 7) != 'sqlite_' ORDER BY name"
        )
    ]
    tables = tuple(
        TableSchema(
            name,
            tuple(str(r[1]) for r in db.execute(f"PRAGMA table_info({_quote(name)})")),
        )
        for name in names
    )
    columns = {table.name: table.columns for table in tables}
    memory_columns = columns.get(_MEMORIES_TABLE, ())
    missing = [c for c in _MEMORIES_REQUIRED if c not in memory_columns]
    if _MEMORIES_TABLE not in columns or missing:
        detail = (
            "table is absent"
            if _MEMORIES_TABLE not in columns
            else ("missing columns: " + ", ".join(missing))
        )
        raise SalvageUnsupportedSchemaError(f"memories {detail}")
    record_count, foreign_active_count = _validate_records(db, workspace_id)
    journal_columns = columns.get(_JOURNAL_TABLE)
    journal_status = _journal_status(journal_columns)
    pending: list[RawJournalRow] = []
    unknown: list[RawJournalRow] = []
    foreign: list[RawJournalRow] = []
    history = 0
    if journal_columns is not None:
        for row in _read_journal(db, journal_columns):
            if journal_status is JournalStatus.UNSUPPORTED:
                unknown.append(row)
            elif row.workspace_id != workspace_id:
                foreign.append(row)
            elif _is_history(row, journal_status):
                history += 1
            elif row.status == TransitionStatus.PENDING and _kind_known(
                row, journal_status
            ):
                pending.append(row)
            else:
                unknown.append(row)
    return SalvageSnapshot(
        workspace_id=workspace_id,
        records=(),
        record_count=record_count,
        pending=tuple(pending),
        unknown=tuple(unknown),
        foreign=tuple(foreign),
        completed_history_count=history,
        foreign_active_count=foreign_active_count,
        schema=SchemaReport(tables, journal_status),
    )


def _journal_status(columns: tuple[str, ...] | None) -> JournalStatus:
    if columns is None:
        return JournalStatus.MISSING
    if any(required not in columns for required in _JOURNAL_REQUIRED):
        return JournalStatus.UNSUPPORTED
    if "kind" not in columns:
        return JournalStatus.LEGACY_WITHOUT_KIND
    return JournalStatus.SUPPORTED


def _kind_known(row: RawJournalRow, journal_status: JournalStatus) -> bool:
    # A legacy journal has no kind column: it is reported as absent and never
    # defaulted to "replace", but completed history stays countable.
    if journal_status is JournalStatus.LEGACY_WITHOUT_KIND:
        return False
    return row.kind in _KINDS


def _is_history(row: RawJournalRow, journal_status: JournalStatus) -> bool:
    if row.status != TransitionStatus.COMPLETED:
        return False
    return journal_status is JournalStatus.LEGACY_WITHOUT_KIND or _kind_known(
        row, journal_status
    )


def _validate_records(db: sqlite3.Connection, workspace_id: str) -> tuple[int, int]:
    source_parent = os.path.dirname(
        os.path.abspath(db.execute("PRAGMA database_list").fetchone()[2])
    )
    for variable in ("TMPDIR", "TEMP", "TMP", "SQLITE_TMPDIR"):
        candidate = os.environ.get(variable)
        if candidate and is_directory_within(candidate, source_parent):
            raise SalvageInputError(
                "the temporary directory is inside the database directory"
            )
    parent = os.path.realpath(tempfile.gettempdir())
    if is_directory_within(parent, source_parent):
        raise SalvageInputError(
            "the temporary directory is inside the database directory"
        )
    with (
        tempfile.TemporaryDirectory(
            prefix="reflectlog-salvage-validation-", dir=parent
        ) as scratch,
        closing(sqlite3.connect(os.path.join(scratch, "seen.db"))) as seen,
    ):
        _ = seen.execute("PRAGMA cache_size = -2048")
        _ = seen.execute("PRAGMA temp_store = FILE")
        _ = seen.execute("CREATE TABLE ids (id INTEGER PRIMARY KEY)")
        _ = seen.execute(
            "CREATE TABLE contents (content BLOB PRIMARY KEY, id INTEGER NOT NULL) WITHOUT ROWID"
        )
        return _validate_rows(db, seen, workspace_id)


def _validate_rows(
    db: sqlite3.Connection, seen: sqlite3.Connection, workspace_id: str
) -> tuple[int, int]:
    total = db.execute(f"SELECT COUNT(*) FROM {_MEMORIES_TABLE}").fetchone()
    cursor = db.execute(
        f"SELECT {_cell_sql('id')}, {_cell_sql('content')}, "
        f"{_cell_sql('created_at')} FROM {_MEMORIES_TABLE} WHERE workspace_id = ?",
        (workspace_id,),
    )
    count = 0
    for position, row in enumerate(cursor, start=1):
        id_value, content, created_at = _row_values(row, 3)
        if id_value.kind is not RawValueKind.INTEGER or not isinstance(
            id_value.value, int
        ):
            raise SalvageMalformedRecordError(
                f"memories row #{position} has a non-integer id"
            )
        source_id = id_value.value
        label = f"memories row id {source_id}"
        if (
            seen.execute("SELECT 1 FROM ids WHERE id = ?", (source_id,)).fetchone()
            is not None
        ):
            raise SalvageMalformedRecordError(f"{label} is duplicated")
        _ = seen.execute("INSERT INTO ids VALUES (?)", (source_id,))
        text = _active_content(content, label)
        if created_at.kind is not RawValueKind.NULL and not (
            created_at.kind is RawValueKind.TEXT and isinstance(created_at.value, str)
        ):
            raise SalvageMalformedRecordError(
                f"{label} has created_at that is neither TEXT nor NULL"
            )
        encoded = text.encode("utf-8")
        first = seen.execute(
            "SELECT id FROM contents WHERE content = ?", (encoded,)
        ).fetchone()
        if first is not None:
            raise SalvageMalformedRecordError(
                f"{label} duplicates the content of memories row id {first[0]}"
            )
        _ = seen.execute("INSERT INTO contents VALUES (?, ?)", (encoded, source_id))
        count += 1
    foreign_active = int(total[0]) - count
    return count, foreign_active


def _stream_records(
    db: sqlite3.Connection, workspace_id: str
) -> Generator[ActiveRecord]:
    with closing(
        db.execute(
            f"SELECT id, CAST(content AS BLOB), CAST(created_at AS BLOB) "
            f"FROM {_MEMORIES_TABLE} WHERE workspace_id = ? ORDER BY id COLLATE BINARY",
            (workspace_id,),
        )
    ) as cursor:
        for source_id, content, created_at in cursor:
            yield ActiveRecord(
                source_id,
                content.decode("utf-8"),
                None if created_at is None else created_at.decode("utf-8"),
            )


def _active_content(value: RawValue, label: str) -> str:
    if value.kind is RawValueKind.TEXT_BYTES:
        raise SalvageMalformedRecordError(f"{label} content is not valid UTF-8")
    if value.kind is not RawValueKind.TEXT or not isinstance(value.value, str):
        raise SalvageMalformedRecordError(f"{label} content is not TEXT")
    if value.value == "":
        raise SalvageMalformedRecordError(f"{label} content is empty")
    return value.value


def _read_journal(
    db: sqlite3.Connection, columns: tuple[str, ...]
) -> list[RawJournalRow]:
    if not columns:
        return []
    select = ", ".join(_cell_sql(column) for column in columns)
    rows: list[RawJournalRow] = []
    for row in db.execute(f"SELECT {select} FROM {_JOURNAL_TABLE} ORDER BY rowid"):
        cells = tuple(zip(columns, _row_values(row, len(columns)), strict=True))
        rows.append(
            RawJournalRow(
                columns=cells,
                workspace_id=_view(cells, "workspace_id"),
                status=_view(cells, "status"),
                kind=_view(cells, "kind"),
            )
        )
    return rows


def _view(cells: tuple[tuple[str, RawValue], ...], name: str) -> str | None:
    for column, value in cells:
        if column == name:
            return value.as_text()
    return None
