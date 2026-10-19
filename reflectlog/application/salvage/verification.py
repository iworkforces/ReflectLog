"""Offline, read-only verification of a content-salvage bundle.

Validates what ``ContentSalvageService`` exports (format
``reflectlog.content-salvage`` version 1) using only the bytes captured by an
``IBundleReader``. The same captured bytes feed both the digest and the parser.

What a passing result means: the manifest is well-formed and names the
expected workspace, ``memories.jsonl`` hashes to the manifest's SHA-256 and has
the manifest's record count, every line is a well-formed record, and the
readiness fields agree with the journal evidence the manifest itself carries.
What it does *not* mean: the digest is not authenticity, since anyone able to
write the bundle can write a matching digest, and records are checked for
format only, so it is not a promise that ``add_memories`` accepts them.

Issue policy: filesystem findings come from the reader. The manifest reports
its first structural failure (parse, then format/version, then schema), after
which independent semantic checks (workspace, record path, readiness) run only
when the schema held. ``memories.jsonl`` is always examined on its own bytes:
file-level findings (BOM, missing final newline) are reported together with the
first bad record line, and later lines are not examined. The digest and count
are independent of record validity, so a corrupt digest never hides a
structural defect and a structural defect never hides a bad digest. A bundle
holding ``INCOMPLETE`` or an unsafe root is not read further. Issues carry
closed-vocabulary codes and line numbers, never content.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
import hashlib
import json
import math
from typing import TYPE_CHECKING, Final, TypeGuard

from reflectlog.core.exceptions import ValidationError
from reflectlog.core.salvage import (
    AUTHENTICITY_NOT_VERIFIED,
    DEFAULT_VERIFICATION_LIMITS,
    RECORDS_FILENAME,
    BundleEntry,
    JournalStatus,
    RawValueKind,
    SidecarStatus,
    UnresolvedReason,
    VerificationIssue,
    VerificationIssueCode,
    VerificationLimits,
    VerificationResult,
    VerificationStatus,
)
from reflectlog.utility.security import validate_workspace_id

if TYPE_CHECKING:
    from typing import Self

    from reflectlog.core.salvage import IBundleReader

SUPPORTED_FORMAT_NAME: Final = "reflectlog.content-salvage"
SUPPORTED_FORMAT_VERSION: Final = 1

_BOM: Final = codecs.BOM_UTF8
_CODES: Final = VerificationIssueCode
_MANIFEST_KEYS: Final = frozenset(
    {
        "foreign_active_count",
        "format",
        "format_version",
        "generation",
        "history",
        "identity",
        "journal",
        "not_preserved",
        "rebuild_ready",
        "records",
        "reflectlog_version",
        "schema",
        "source_fingerprints",
        "unresolved_reasons",
        "workspace_id",
    }
)
_RECORD_KEYS: Final = frozenset({"content", "source_created_at", "source_id"})
_IDENTITY_FIELDS: Final = frozenset(
    {"dimensions", "model", "provider", "version", "workspace_id"}
)
_JOURNAL_BUCKETS: Final = frozenset({"foreign_workspace", "pending", "unknown"})
_SIDECAR_STATUSES: Final = frozenset(item.value for item in SidecarStatus)
_JOURNAL_STATUSES: Final = frozenset(item.value for item in JournalStatus)
_RAW_KINDS: Final = frozenset(item.value for item in RawValueKind)
_REASONS: Final = frozenset(item.value for item in UnresolvedReason)


class _DuplicateKeyError(Exception):
    """A JSON object repeated a key."""


class _NonFiniteError(Exception):
    """A JSON document used NaN, Infinity, or -Infinity."""


class _LoadError(Exception):
    """A document could not be loaded; carries the closed issue code."""

    def __init__(self, code: VerificationIssueCode) -> None:
        super().__init__(code.value)
        self.code = code


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_constant(_name: str) -> object:
    raise _NonFiniteError


def _load_json(data: bytes) -> object:
    """Strictly decode ``data`` as UTF-8 JSON, or raise ``_LoadError``."""
    if data.startswith(_BOM):
        raise _LoadError(_CODES.BOM_PRESENT)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _LoadError(_CODES.UTF8_INVALID) from e
    try:
        parsed: object = json.loads(
            text, object_pairs_hook=_object_pairs, parse_constant=_reject_constant
        )
    except _DuplicateKeyError as e:
        raise _LoadError(_CODES.DUPLICATE_KEY) from e
    except _NonFiniteError as e:
        raise _LoadError(_CODES.NON_FINITE_NUMBER) from e
    except (ValueError, RecursionError) as e:
        # ValueError also covers JSONDecodeError and Python's huge-int limit.
        raise _LoadError(_CODES.JSON_INVALID) from e
    return parsed


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    """Narrow an object produced by the JSON decoder."""
    return isinstance(value, dict)


def _is_list(value: object) -> TypeGuard[list[object]]:
    """Narrow a list produced by the JSON decoder."""
    return isinstance(value, list)


def _object(value: object, keys: frozenset[str]) -> dict[str, object] | None:
    if _is_object(value):
        typed: dict[str, object] = value
        if typed.keys() == keys:
            return typed
    return None


def _strings(value: object) -> list[str] | None:
    if not _is_list(value):
        return None
    values: list[object] = value
    items: list[str] = []
    for item in values:
        text = _str(item)
        if text is None:
            return None
        items.append(text)
    return items


def _encodable(text: str) -> bool:
    """True unless ``text`` holds lone surrogates that no UTF-8 file can carry."""
    try:
        _ = text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class _Manifest:
    """The manifest fields verification needs, after the closed schema held."""

    workspace_id: str
    rebuild_ready: bool
    reasons: tuple[str, ...]
    journal_status: str
    pending: int
    unknown: int
    foreign: int
    records_file: str
    records_count: int
    records_sha256: str


def _scalar_ok(value: object) -> bool:
    if value is None or isinstance(value, str | int | bool):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _identity_ok(value: object) -> bool:
    item = _object(value, frozenset({"fields", "status"}))
    if item is None or _str(item["status"]) not in _SIDECAR_STATUSES:
        return False
    fields = item["fields"]
    if not _is_object(fields):
        return False
    typed: dict[str, object] = fields
    return set(typed) <= _IDENTITY_FIELDS and all(
        _scalar_ok(entry) for entry in typed.values()
    )


def _generation_ok(value: object) -> bool:
    item = _object(value, frozenset({"raw", "status"}))
    return (
        item is not None
        and _str(item["status"]) in _SIDECAR_STATUSES
        and (item["raw"] is None or _str(item["raw"]) is not None)
    )


def _fingerprints_ok(value: object) -> bool:
    if not _is_list(value):
        return False
    values: list[object] = value
    for entry in values:
        item = _object(entry, frozenset({"path", "sha256", "size"}))
        if (
            item is None
            or _str(item["path"]) is None
            or _str(item["sha256"]) is None
            or _int(item["size"]) is None
        ):
            return False
    return True


def _schema_ok(value: object) -> str | None:
    """Return the journal status when the ``schema`` section is well-formed."""
    item = _object(value, frozenset({"journal_status", "tables"}))
    if item is None:
        return None
    status = _str(item["journal_status"])
    tables = item["tables"]
    if status not in _JOURNAL_STATUSES or not _is_list(tables):
        return None
    typed_tables: list[object] = tables
    for table in typed_tables:
        entry = _object(table, frozenset({"columns", "name"}))
        if (
            entry is None
            or _str(entry["name"]) is None
            or _strings(entry["columns"]) is None
        ):
            return None
    return status


def _cell_ok(value: object) -> bool:
    cell = _object(value, frozenset({"name", "type", "value"}))
    if cell is None or _str(cell["name"]) is None:
        return False
    kind = _str(cell["type"])
    if kind not in _RAW_KINDS:
        return False
    if kind == RawValueKind.NULL.value:
        return cell["value"] is None
    if kind == RawValueKind.INTEGER.value:
        return _int(cell["value"]) is not None
    return _str(cell["value"]) is not None


def _journal_rows(value: object) -> int | None:
    """Return the row count when every row and cell is well-formed."""
    if not _is_list(value):
        return None
    rows: list[object] = value
    for row in rows:
        item = _object(row, frozenset({"columns"}))
        if item is None:
            return None
        cells = item["columns"]
        if not _is_list(cells):
            return None
        typed_cells: list[object] = cells
        if not all(_cell_ok(cell) for cell in typed_cells):
            return None
    return len(rows)


def _journal_counts(value: object) -> tuple[int, int, int] | None:
    item = _object(value, _JOURNAL_BUCKETS)
    if item is None:
        return None
    pending = _journal_rows(item["pending"])
    unknown = _journal_rows(item["unknown"])
    foreign = _journal_rows(item["foreign_workspace"])
    if pending is None or unknown is None or foreign is None:
        return None
    return pending, unknown, foreign


def _section_ok(data: dict[str, object]) -> bool:
    history = _object(data["history"], frozenset({"completed_journal_rows"}))
    return (
        _str(data["reflectlog_version"]) is not None
        and _identity_ok(data["identity"])
        and _generation_ok(data["generation"])
        and _fingerprints_ok(data["source_fingerprints"])
        and history is not None
        and _int(history["completed_journal_rows"]) is not None
        and _int(data["foreign_active_count"]) is not None
        and _strings(data["not_preserved"]) is not None
    )


def _manifest_view(data: dict[str, object]) -> _Manifest | None:
    """Apply the closed version-1 schema; ``None`` when anything deviates."""
    if data.keys() != _MANIFEST_KEYS or not _section_ok(data):
        return None
    records = _object(data["records"], frozenset({"count", "file", "sha256"}))
    reasons = _strings(data["unresolved_reasons"])
    ready = data["rebuild_ready"]
    workspace = _str(data["workspace_id"])
    status = _schema_ok(data["schema"])
    counts = _journal_counts(data["journal"])
    if (
        records is None
        or reasons is None
        or not isinstance(ready, bool)
        or workspace is None
        or status is None
        or counts is None
    ):
        return None
    file = _str(records["file"])
    count = _int(records["count"])
    digest = _str(records["sha256"])
    if file is None or count is None or digest is None:
        return None
    return _Manifest(
        workspace, ready, tuple(reasons), status, *counts, file, count, digest
    )


def _derived_reasons(manifest: _Manifest) -> tuple[str, ...] | None:
    """Reasons implied by the manifest's own evidence; ``None`` if contradictory."""
    status = manifest.journal_status
    if status == JournalStatus.MISSING.value and (
        manifest.pending or manifest.unknown or manifest.foreign
    ):
        return None
    if status == JournalStatus.UNSUPPORTED.value and (
        manifest.pending or manifest.foreign
    ):
        return None
    reasons: set[str] = set()
    if manifest.pending:
        reasons.add(UnresolvedReason.JOURNAL_PENDING_INTENT.value)
    if manifest.unknown:
        reasons.add(UnresolvedReason.JOURNAL_UNKNOWN_EVIDENCE.value)
    if manifest.foreign:
        reasons.add(UnresolvedReason.JOURNAL_FOREIGN_WORKSPACE_EVIDENCE.value)
    if status == JournalStatus.MISSING.value:
        reasons.add(UnresolvedReason.JOURNAL_MISSING.value)
    if status == JournalStatus.UNSUPPORTED.value:
        reasons.add(UnresolvedReason.JOURNAL_UNSUPPORTED.value)
    return tuple(sorted(reasons))


def _canonical(workspace_id: str) -> str | None:
    try:
        return validate_workspace_id(workspace_id)
    except ValidationError:
        return None


@dataclass(frozen=True, slots=True)
class _ManifestOutcome:
    issues: tuple[VerificationIssue, ...]
    manifest: _Manifest | None


def _manifest_issue(code: VerificationIssueCode) -> VerificationIssue:
    return VerificationIssue(code, BundleEntry.MANIFEST)


def _check_manifest(data: bytes, expected: str | None) -> _ManifestOutcome:
    try:
        parsed = _load_json(data)
    except _LoadError as e:
        return _ManifestOutcome((_manifest_issue(e.code),), None)
    if not _is_object(parsed):
        return _ManifestOutcome((_manifest_issue(_CODES.NOT_OBJECT),), None)
    typed: dict[str, object] = parsed
    issues: list[VerificationIssue] = []
    if typed.get("format") != SUPPORTED_FORMAT_NAME:
        issues.append(_manifest_issue(_CODES.FORMAT_UNSUPPORTED))
    version = typed.get("format_version")
    if _int(version) != SUPPORTED_FORMAT_VERSION:
        issues.append(_manifest_issue(_CODES.VERSION_UNSUPPORTED))
    if issues:
        return _ManifestOutcome(tuple(issues), None)
    manifest = _manifest_view(typed)
    if manifest is None:
        return _ManifestOutcome(
            (_manifest_issue(_CODES.MANIFEST_SCHEMA_INVALID),), None
        )
    if expected is None or manifest.workspace_id != expected:
        issues.append(_manifest_issue(_CODES.WORKSPACE_MISMATCH))
    if manifest.records_file != RECORDS_FILENAME:
        issues.append(_manifest_issue(_CODES.RECORDS_PATH_INVALID))
    issues.extend(_readiness_issues(manifest))
    return _ManifestOutcome(tuple(issues), manifest)


def _readiness_issues(manifest: _Manifest) -> list[VerificationIssue]:
    reasons = manifest.reasons
    if any(item not in _REASONS for item in reasons) or reasons != tuple(
        sorted(set(reasons))
    ):
        return [_manifest_issue(_CODES.UNRESOLVED_REASON_INVALID)]
    derived = _derived_reasons(manifest)
    if derived != reasons or manifest.rebuild_ready != (not reasons):
        return [_manifest_issue(_CODES.READINESS_INCONSISTENT)]
    return []


@dataclass(frozen=True, slots=True)
class _RecordsOutcome:
    issues: tuple[VerificationIssue, ...]
    line_count: int
    parsed_count: int | None


def _record_issue(code: VerificationIssueCode, line: int | None) -> VerificationIssue:
    return VerificationIssue(code, BundleEntry.RECORDS, line)


def _check_record_line(
    raw: bytes,
    ids: set[int],
    contents: set[str],
    previous_id: int | None,
) -> VerificationIssueCode | tuple[int, str]:
    """Validate one line; return an issue code, or ``(source_id, content)``."""
    try:
        parsed = _load_json(raw)
    except _LoadError as e:
        return e.code
    if not _is_object(parsed):
        return _CODES.NOT_OBJECT
    typed: dict[str, object] = parsed
    if typed.keys() != _RECORD_KEYS:
        return _CODES.RECORD_FIELDS_INVALID
    source_id = _int(typed["source_id"])
    content = _str(typed["content"])
    created = typed["source_created_at"]
    if source_id is None:
        return _CODES.RECORD_ID_INVALID
    if content is None or content == "" or not _encodable(content):
        return _CODES.RECORD_CONTENT_INVALID
    if created is not None and (
        not isinstance(created, str) or not _encodable(created)
    ):
        return _CODES.RECORD_TIMESTAMP_INVALID
    if source_id in ids:
        return _CODES.RECORD_ID_DUPLICATE
    if previous_id is not None and source_id < previous_id:
        return _CODES.RECORD_ID_OUT_OF_ORDER
    if content in contents:
        return _CODES.RECORD_CONTENT_DUPLICATE
    return source_id, content


def _check_records(data: bytes) -> _RecordsOutcome:
    issues: list[VerificationIssue] = []
    parts = data.split(b"\n") if data else []
    tail = parts.pop() if parts else b""
    line_count = len(parts) + (1 if tail else 0)
    clean = True
    if data.startswith(_BOM):
        issues.append(_record_issue(_CODES.BOM_PRESENT, 1))
        clean = False
    else:
        ids: set[int] = set()
        contents: set[str] = set()
        previous: int | None = None
        for number, raw in enumerate(parts, start=1):
            outcome = _check_record_line(raw, ids, contents, previous)
            if isinstance(outcome, VerificationIssueCode):
                issues.append(_record_issue(outcome, number))
                clean = False
                break
            previous, text = outcome
            ids.add(previous)
            contents.add(text)
    if tail:
        issues.append(_record_issue(_CODES.MISSING_FINAL_NEWLINE, line_count))
        clean = False
    return _RecordsOutcome(tuple(issues), line_count, len(parts) if clean else None)


class ContentSalvageVerifier:
    """Verify an exported bundle offline and read-only; never raises on bad input."""

    def __init__(self, reader: IBundleReader) -> None:
        self._reader = reader

    @classmethod
    def from_defaults(
        cls, limits: VerificationLimits = DEFAULT_VERIFICATION_LIMITS
    ) -> Self:
        """Wire the read-only reader with validated per-file input byte budgets."""
        from reflectlog.infrastructure.salvage_bundle_reader import (
            FileSystemBundleReader,
        )

        return cls(FileSystemBundleReader(limits=limits))

    def verify(self, *, bundle: str, workspace_id: str) -> VerificationResult:
        """Verify ``bundle`` against the operator-supplied ``workspace_id``.

        ``workspace_id`` is canonicalized like every other workspace id; the
        manifest's id must equal that canonical value exactly. A ``READY`` or
        ``NOT_READY`` result is internal consistency only: not authenticity,
        and no promise that ``add_memories`` accepts the records.
        """
        issues: list[VerificationIssue] = []
        expected = _canonical(workspace_id)
        if expected is None:
            issues.append(
                VerificationIssue(_CODES.EXPECTED_WORKSPACE_INVALID, BundleEntry.BUNDLE)
            )
        captured = self._reader.capture(bundle)
        issues.extend(captured.issues)
        # Bytes read while the bundle moved describe no single state of it, so
        # no count or readiness claim is derived from them.
        stable = all(
            item.code is not _CODES.CHANGED_DURING_READ for item in captured.issues
        )

        manifest: _Manifest | None = None
        if captured.manifest is not None:
            outcome = _check_manifest(captured.manifest.data, expected)
            issues.extend(outcome.issues)
            manifest = outcome.manifest

        record_count: int | None = None
        if captured.records is not None:
            data = captured.records.data
            checked = _check_records(data)
            issues.extend(checked.issues)
            record_count = checked.parsed_count if stable else None
            if manifest is not None:
                if hashlib.sha256(data).hexdigest() != manifest.records_sha256:
                    issues.append(_record_issue(_CODES.DIGEST_MISMATCH, None))
                if checked.line_count != manifest.records_count:
                    issues.append(_record_issue(_CODES.COUNT_MISMATCH, None))

        ready: bool | None = None
        reasons: tuple[str, ...] = ()
        if (
            stable
            and manifest is not None
            and all(item in _REASONS for item in manifest.reasons)
        ):
            ready = manifest.rebuild_ready
            reasons = manifest.reasons
        return VerificationResult(
            status=_status(issues, ready),
            workspace_id=expected,
            record_count=record_count,
            rebuild_ready=ready,
            unresolved_reasons=reasons,
            issues=tuple(issues),
        )


def _status(
    issues: list[VerificationIssue], rebuild_ready: bool | None
) -> VerificationStatus:
    if issues or rebuild_ready is None:
        return VerificationStatus.INVALID
    return VerificationStatus.READY if rebuild_ready else VerificationStatus.NOT_READY


def result_payload(result: VerificationResult) -> dict[str, object]:
    """Render ``result`` as content-free, JSON-serialisable data.

    Holds closed-vocabulary codes, counts, the canonical expected workspace id
    and line numbers only: no record content, timestamps, manifest string
    values, offending values, absolute paths, or exception text.
    ``"authenticity": "not_verified"`` is always present because a matching
    digest proves consistency, not who wrote the bundle.
    """
    return {
        "status": result.status.value,
        "workspace_id": result.workspace_id,
        "record_count": result.record_count,
        "rebuild_ready": result.rebuild_ready,
        "unresolved_reasons": list(result.unresolved_reasons),
        "errors": [
            {"code": issue.code.value, "entry": issue.entry.value, "line": issue.line}
            for issue in result.issues
        ],
        "authenticity": AUTHENTICITY_NOT_VERIFIED,
    }
