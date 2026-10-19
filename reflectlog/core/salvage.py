"""Offline content-salvage value types and boundary protocols.

Salvage reads an archived workspace copy without constructing any engine, so
these types describe what was *observed* in SQLite and nothing else. Journal
evidence is kept raw and separate from active records: unresolved intent never
becomes content.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
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
    records: Iterable[ActiveRecord]
    record_count: int
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
    """Single-pass records and a manifest finalized after stream exhaustion."""

    records: Iterable[bytes]
    finalize_manifest: Callable[[], bytes]


@runtime_checkable
class ISalvageSnapshotReader(Protocol):
    """Read a salvage snapshot from a disposable database copy."""

    def read_snapshot(
        self, database_path: str, workspace_id: str
    ) -> AbstractContextManager[SalvageSnapshot]:
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


AUTHENTICITY_NOT_VERIFIED = "not_verified"


@dataclass(frozen=True, slots=True)
class VerificationLimits:
    """Per-file input byte budgets, not a bound on total verifier memory."""

    max_manifest_bytes: int = 1024 * 1024
    max_records_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in (
            ("max_manifest_bytes", self.max_manifest_bytes),
            ("max_records_bytes", self.max_records_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_VERIFICATION_LIMITS: Final = VerificationLimits()


class UnresolvedReason(StrEnum):
    """Closed set of ``unresolved_reasons`` codes the exporter can write."""

    JOURNAL_PENDING_INTENT = "journal_pending_intent"
    JOURNAL_UNKNOWN_EVIDENCE = "journal_unknown_evidence"
    JOURNAL_FOREIGN_WORKSPACE_EVIDENCE = "journal_foreign_workspace_evidence"
    JOURNAL_MISSING = "journal_missing"
    JOURNAL_UNSUPPORTED = "journal_unsupported"


class VerificationStatus(StrEnum):
    """Outcome of verifying one bundle.

    ``READY`` and ``NOT_READY`` both mean the bundle is internally consistent;
    neither means it is authentic or that ``add_memories`` will accept every
    record. ``NOT_READY`` is a valid bundle whose manifest honestly reports
    unresolved journal evidence.
    """

    READY = "ready"
    NOT_READY = "not_ready"
    INVALID = "invalid"


class BundleEntry(StrEnum):
    """Part of a bundle an issue refers to."""

    BUNDLE = "bundle"
    MARKER = "marker"
    MANIFEST = "manifest"
    RECORDS = "records"


class VerificationIssueCode(StrEnum):
    """Closed vocabulary of verification failures; never carries content."""

    EXPECTED_WORKSPACE_INVALID = "expected_workspace_invalid"
    BUNDLE_MISSING = "bundle_missing"
    BUNDLE_SYMLINK = "bundle_symlink"
    BUNDLE_NOT_DIRECTORY = "bundle_not_directory"
    BUNDLE_UNREADABLE = "bundle_unreadable"
    INCOMPLETE_MARKER_PRESENT = "incomplete_marker_present"
    FILE_MISSING = "file_missing"
    FILE_SYMLINK = "file_symlink"
    FILE_NOT_REGULAR = "file_not_regular"
    FILE_UNREADABLE = "file_unreadable"
    FILE_TOO_LARGE = "file_too_large"
    CHANGED_DURING_READ = "changed_during_read"
    UTF8_INVALID = "utf8_invalid"
    BOM_PRESENT = "bom_present"
    JSON_INVALID = "json_invalid"
    DUPLICATE_KEY = "duplicate_key"
    NON_FINITE_NUMBER = "non_finite_number"
    NOT_OBJECT = "not_object"
    MANIFEST_SCHEMA_INVALID = "manifest_schema_invalid"
    FORMAT_UNSUPPORTED = "format_unsupported"
    VERSION_UNSUPPORTED = "version_unsupported"
    WORKSPACE_MISMATCH = "workspace_mismatch"
    RECORDS_PATH_INVALID = "records_path_invalid"
    UNRESOLVED_REASON_INVALID = "unresolved_reason_invalid"
    READINESS_INCONSISTENT = "readiness_inconsistent"
    DIGEST_MISMATCH = "digest_mismatch"
    COUNT_MISMATCH = "count_mismatch"
    MISSING_FINAL_NEWLINE = "missing_final_newline"
    RECORD_FIELDS_INVALID = "record_fields_invalid"
    RECORD_ID_INVALID = "record_id_invalid"
    RECORD_CONTENT_INVALID = "record_content_invalid"
    RECORD_TIMESTAMP_INVALID = "record_timestamp_invalid"
    RECORD_ID_DUPLICATE = "record_id_duplicate"
    RECORD_ID_OUT_OF_ORDER = "record_id_out_of_order"
    RECORD_CONTENT_DUPLICATE = "record_content_duplicate"


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    """One content-free finding: a code, the bundle part, and a 1-based line."""

    code: VerificationIssueCode
    entry: BundleEntry
    line: int | None = None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """What verification observed about one bundle.

    A ``READY`` or ``NOT_READY`` status means the bundle's own bytes agree with
    its own manifest. It is not authenticity: anyone able to write the bundle
    can write a matching digest. It is also no promise that ``add_memories``
    accepts the content, because records are checked for format only and are
    never run through add validation.

    ``workspace_id`` is the canonical expected workspace (``None`` when the
    requested id was invalid). ``record_count`` is the number of records parsed
    from the captured bytes when every record parsed cleanly, otherwise
    ``None``. ``rebuild_ready`` and ``unresolved_reasons`` are taken from the
    manifest only when it parsed and used closed-set reason codes.
    """

    status: VerificationStatus
    workspace_id: str | None
    record_count: int | None
    rebuild_ready: bool | None
    unresolved_reasons: tuple[str, ...]
    issues: tuple[VerificationIssue, ...]


@dataclass(frozen=True, slots=True)
class CapturedFile:
    """The exact bytes of one bundle file, read once from one descriptor."""

    data: bytes


@dataclass(frozen=True, slots=True)
class BundleCapture:
    """Result of reading a bundle directory without modifying it.

    ``issues`` hold filesystem-level findings. A file that was missing,
    unsafe, unreadable, oversized, or observed to change is ``None`` and has an issue.
    """

    manifest: CapturedFile | None
    records: CapturedFile | None
    issues: tuple[VerificationIssue, ...]


@runtime_checkable
class IBundleReader(Protocol):
    """Capture the required files of an exported bundle, read-only."""

    def capture(self, bundle_dir: str) -> BundleCapture:
        """Read ``manifest.json`` and ``memories.jsonl`` from ``bundle_dir``.

        Never writes, repairs, or follows symlinks. Detects observed changes
        during the read only; it is not a guarantee against every race.
        """
        ...
