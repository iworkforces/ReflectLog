"""Helpers that fabricate archived workspace directories for salvage tests."""

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

from reflectlog.infrastructure.memory_store import MemoryStore

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
IDENTITY: Mapping[str, object] = {
    "version": 1,
    "workspace_id": WS,
    "provider": "wemm",
    "model": "tencent/WeMM-Embedding-2B",
    "dimensions": 2048,
}

type Row = tuple[str, str, str | None]
type JournalRow = tuple[str, str, str | None, str]


def make_archive(
    root: Path,
    *,
    memories: Sequence[Row] = (),
    journal: Sequence[JournalRow] = (),
    kind: bool = True,
    journal_table: bool = True,
    identity: Mapping[str, object] | bytes | None = None,
    generation: str | bytes | None = "3\n",
) -> Path:
    """Create ``root/usearch/memories.db`` (rollback-journal mode) plus sidecars.

    ``memories`` are ``(workspace, content, created_at)``; ``journal`` rows are
    ``(workspace, status, kind, new_content)``.
    """
    usearch = root / "usearch"
    usearch.mkdir(parents=True)
    connection = sqlite3.connect(usearch / "memories.db")
    try:
        _ = connection.execute("PRAGMA journal_mode=DELETE")
        _ = connection.execute(MEMORIES_DDL)
        for workspace, content, created in memories:
            _ = connection.execute(
                "INSERT INTO memories (workspace_id, content, created_at) "
                "VALUES (?, ?, ?)",
                (workspace, content, created),
            )
        if journal_table:
            _ = connection.execute(JOURNAL_DDL.format(kind=KIND_COLUMN if kind else ""))
        for workspace, status, row_kind, new_content in (
            journal if journal_table else ()
        ):
            columns = (
                "workspace_id, old_memory_id, old_content, new_content, "
                "archive_id, reason, confidence, status"
            )
            values: tuple[object, ...] = (workspace, 0, "", new_content, 0, "add", 1.0)
            values = (*values, status)
            if kind:
                columns += ", kind"
                values = (*values, row_kind)
            marks = ", ".join("?" * len(values))
            _ = connection.execute(
                f"INSERT INTO replacement_transitions ({columns}) VALUES ({marks})",
                values,
            )
        connection.commit()
    finally:
        connection.close()
    write_sidecars(root, identity=identity, generation=generation)
    return root


def write_sidecars(
    root: Path,
    *,
    identity: Mapping[str, object] | bytes | None = None,
    generation: str | bytes | None = "3\n",
) -> None:
    if identity is None:
        identity = IDENTITY
    if isinstance(identity, bytes):
        (root / ".reflectlog.embedding-identity.json").write_bytes(identity)
    else:
        (root / ".reflectlog.embedding-identity.json").write_text(
            json.dumps(identity), encoding="utf-8"
        )
    if isinstance(generation, bytes):
        (root / ".reflectlog.storage-generation").write_bytes(generation)
    elif generation is not None:
        (root / ".reflectlog.storage-generation").write_text(
            generation, encoding="utf-8"
        )


def make_wal_archive(root: Path, scratch: Path, content: str) -> tuple[Path, int]:
    """Archive whose only copy of ``content`` is a committed WAL frame.

    A live ``MemoryStore`` holds the row in its WAL while db and WAL are copied
    (like a crashed writer), so the main file alone lacks the row.
    """
    usearch = root / "usearch"
    usearch.mkdir(parents=True)
    live = scratch / "live" / "memories.db"
    live.parent.mkdir(parents=True)
    store = MemoryStore(db_path=str(live))
    try:
        row_id = store.insert(WS, content)
        _ = shutil.copy2(live, usearch / "memories.db")
        _ = shutil.copy2(f"{live}-wal", usearch / "memories.db-wal")
        shutil.copy2(f"{live}-shm", usearch / "memories.db-shm")
    finally:
        store.close()
    write_sidecars(root)
    return root, row_id


def make_clean_wal_format_archive(root: Path, scratch: Path, content: str) -> Path:
    """WAL-format database, cleanly closed, copied without any -wal/-shm file."""
    usearch = root / "usearch"
    usearch.mkdir(parents=True)
    live = scratch / "clean" / "memories.db"
    live.parent.mkdir(parents=True)
    store = MemoryStore(db_path=str(live))
    try:
        _ = store.insert(WS, content)
    finally:
        store.close()
    _ = shutil.copy2(live, usearch / "memories.db")
    write_sidecars(root)
    return root


def tree_digest(root: Path) -> list[tuple[str, str, int]]:
    """Return sorted (relative path, sha256-or-'dir', mtime_ns) of a tree."""
    entries: list[tuple[str, str, int]] = []
    for current, directories, files in os.walk(root):
        for name in [*directories, *files]:
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                entries.append((relative, "dir", path.stat().st_mtime_ns))
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                entries.append((relative, digest, path.stat().st_mtime_ns))
    return sorted(entries)


def as_object(value: object) -> dict[str, object]:
    """Narrow parsed JSON to an object, failing the test otherwise."""
    assert isinstance(value, dict)
    return value


def as_list(value: object) -> list[object]:
    """Narrow parsed JSON to an array, failing the test otherwise."""
    assert isinstance(value, list)
    return value
