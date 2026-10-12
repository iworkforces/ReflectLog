"""Offline content-salvage value types and boundary protocols.

Salvage reads an archived workspace copy without constructing any engine, so
these types describe what was *observed* in SQLite and nothing else. Journal
evidence is kept raw and separate from active records: unresolved intent never
becomes content.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

RECORDS_FILENAME = "memories.jsonl"
MANIFEST_FILENAME = "manifest.json"
INCOMPLETE_MARKER = "INCOMPLETE"


class RawValueKind(StrEnum):
    """Storage class of one observed SQLite value."""

    INTEGER = "integer"
    REAL = "real"
    TEXT = "text"
    TEXT_BYTES = "text_bytes"
    BLOB = "blob"
    NULL = "null"


@dataclass(frozen=True, slots=True)
class RawValue:
    """One SQLite value kept without interpretation.

    ``value`` holds the integer for ``INTEGER``, ``repr(float)`` for ``REAL``,
    the decoded string for ``TEXT``, standard base64 for ``BLOB`` and for
    ``TEXT_BYTES`` (TEXT whose bytes are not valid UTF-8), and ``None`` for
    ``NULL``.
    """

    kind: RawValueKind
    value: int | str | None = None

    def as_text(self) -> str | None:
        """Return the string when this is decodable TEXT, otherwise None."""
        if self.kind is RawValueKind.TEXT and isinstance(self.value, str):
            return self.value
        return None


@dataclass(frozen=True, slots=True)
class ActiveRecord:
    """One active memory row of the requested workspace."""

    source_id: int
    content: str
    created_at: str | None


@dataclass(frozen=True, slots=True)
class RawJournalRow:
    """One ``replacement_transitions`` row, columns in declared order.

    ``workspace_id``, ``status`` and ``kind`` are views of the raw cells: the
    string when the cell is decodable TEXT, otherwise ``None`` (column absent,
    NULL, or not text). They are never coerced to enums.
    """

    columns: tuple[tuple[str, RawValue], ...]
    workspace_id: str | None
    status: str | None
    kind: str | None


class JournalStatus(StrEnum):
    """How much of the replacement journal salvage could interpret."""

    SUPPORTED = "supported"
    LEGACY_WITHOUT_KIND = "legacy_without_kind"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class TableSchema:
    """Observed table name and its column names in declared order."""

    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SchemaReport:
    """Tables and columns observed, plus the journal interpretation."""

    tables: tuple[TableSchema, ...]
    journal_status: JournalStatus


@dataclass(frozen=True, slots=True)
class SalvageSnapshot:
    """Everything observed in one read-only transaction over a database copy.

    ``records`` are the requested workspace's ``memories`` rows sorted by id.
    ``pending``, ``unknown`` and ``foreign`` hold raw journal evidence in
    journal order and are never projected into ``records``.
    """

    workspace_id: str
    records: tuple[ActiveRecord, ...]
    pending: tuple[RawJournalRow, ...]
    unknown: tuple[RawJournalRow, ...]
    foreign: tuple[RawJournalRow, ...]
    completed_history_count: int
    foreign_active_count: int
    schema: SchemaReport


class SidecarStatus(StrEnum):
    """State of a workspace sidecar file in the archive."""

    PRESENT = "present"
    MISSING = "missing"
    UNREADABLE = "unreadable"
    MALFORMED = "malformed"


type SidecarScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class IdentitySidecarReport:
    """Raw embedding-identity sidecar observation; never used for validation.

    ``fields`` holds the well-formed scalar values of ``version``,
    ``workspace_id``, ``provider``, ``model`` and ``dimensions`` that were
    present, sorted by name.
    """

    status: SidecarStatus
    fields: tuple[tuple[str, SidecarScalar], ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationSidecarReport:
    """Raw storage-generation sidecar observation.

    ``raw`` is the stripped text when the file decoded, otherwise ``None``. A
    missing sidecar is reported as missing, never as generation zero.
    """

    status: SidecarStatus
    raw: str | None = None


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """Relative POSIX path, size, and sha256 of one inspected archive file."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SalvageBundle:
    """Fully rendered export payloads, built before any file is written."""

    records: bytes
    manifest: bytes


@runtime_checkable
class ISalvageSnapshotReader(Protocol):
    """Read a salvage snapshot from a disposable database copy."""

    def read_snapshot(self, database_path: str, workspace_id: str) -> SalvageSnapshot:
        """Read ``database_path`` strictly read-only in one transaction.

        Raises:
            SalvageError: If the database is missing, damaged, unsupported, or
                holds a malformed active record.
        """
        ...


@runtime_checkable
class IArchiveInspection(Protocol):
    """A disposable copy of an archive plus what was observed while copying."""

    @property
    def scratch_dir(self) -> str: ...

    @property
    def database_path(self) -> str: ...

    @property
    def fingerprints(self) -> tuple[SourceFingerprint, ...]: ...

    @property
    def identity(self) -> IdentitySidecarReport: ...

    @property
    def generation(self) -> GenerationSidecarReport: ...

    def verify_unchanged(self) -> None:
        """Re-fingerprint the original archive.

        Raises:
            SalvageArchiveChangedError: If any inspected file differs from the
                first fingerprint pass.
        """
        ...


@runtime_checkable
class IArchiveInspector(Protocol):
    """Isolate an archived workspace directory into a disposable copy."""

    def inspect(self, archive_dir: str) -> AbstractContextManager[IArchiveInspection]:
        """Copy the archive's SQLite files into scratch; remove scratch on exit.

        Raises:
            SalvageInputError: If the archive or its database is missing, or
                an inspected file is a symlink.
            SalvageArchiveChangedError: If the copy differs from the source.
        """
        ...


@runtime_checkable
class IBundlePublisher(Protocol):
    """Publish a rendered bundle into a new directory without overwriting."""

    def resolve_output(self, output: str, protected_dirs: Sequence[str]) -> str:
        """Return the absolute output path if it is safe to create.

        Raises:
            SalvageUnsafeOutputError: If the output exists (even as a dangling
                symlink), its parent is missing, or it lies inside a protected
                directory.
        """
        ...

    def publish(self, output: str, bundle: SalvageBundle) -> None:
        """Publish ``bundle`` at ``output``; a failed run never looks complete.

        Raises:
            SalvageUnsafeOutputError: If the output was created concurrently.
            SalvagePublicationError: If any publication step failed.
        """
        ...
