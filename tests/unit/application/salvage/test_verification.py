"""Semantic verification of real content-salvage exports and corruptions."""

from __future__ import annotations

import codecs
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING

import pytest

from reflectlog.application.salvage.service import ContentSalvageService
from reflectlog.application.salvage.verification import (
    ContentSalvageVerifier,
    result_payload,
)
from reflectlog.core.salvage import (
    BundleEntry,
)
from reflectlog.core.salvage import (
    VerificationIssueCode as Code,
)
from reflectlog.core.salvage import (
    VerificationStatus as Status,
)
from tests.unit.salvage_support import (
    OTHER,
    SECRET,
    WS,
    as_list,
    as_object,
    make_archive,
)
from tests.unit.salvage_verify_support import (
    export_bundle,
    load_manifest,
    record_line,
    store_manifest,
    write_records,
)

if TYPE_CHECKING:
    from reflectlog.core.salvage import VerificationResult
    from tests.unit.salvage_support import JournalRow

type Issue = tuple[Code, BundleEntry, int | None]


def _verify(bundle: Path, expected: str = WS) -> VerificationResult:
    return ContentSalvageVerifier.from_defaults().verify(
        bundle=str(bundle), workspace_id=expected
    )


def _assert_result(
    result: VerificationResult,
    *,
    status: Status,
    issues: tuple[Issue, ...] = (),
    count: int | None = 1,
    ready: bool | None = True,
    reasons: tuple[str, ...] = (),
    workspace: str | None = WS,
) -> None:
    assert result.status is status
    assert tuple((i.code, i.entry, i.line) for i in result.issues) == issues
    assert result.record_count == count
    assert result.rebuild_ready is ready
    assert result.unresolved_reasons == reasons
    assert result.workspace_id == workspace


def _assert_payload(result: VerificationResult, bundle: Path) -> None:
    payload = result_payload(result)
    assert set(payload) == {
        "status",
        "workspace_id",
        "record_count",
        "rebuild_ready",
        "unresolved_reasons",
        "errors",
        "authenticity",
    }
    assert payload["authenticity"] == "not_verified"
    assert payload["status"] == result.status.value
    assert payload["workspace_id"] == result.workspace_id
    assert payload["record_count"] == result.record_count
    assert payload["rebuild_ready"] == result.rebuild_ready
    assert payload["unresolved_reasons"] == list(result.unresolved_reasons)
    errors = as_list(payload["errors"])
    assert errors == [
        {"code": i.code.value, "entry": i.entry.value, "line": i.line}
        for i in result.issues
    ]
    assert all(set(as_object(error)) == {"code", "entry", "line"} for error in errors)
    serialized = json.dumps(payload, ensure_ascii=False)
    for private in (
        SECRET,
        str(bundle.absolute()),
        "reflectlog.content-salvage",
        "tencent/WeMM-Embedding-2B",
        "yesterday",
        OTHER,
        "SALVAGE-WS",
    ):
        assert private not in serialized
    for raw in (bundle / "memories.jsonl").read_bytes().splitlines():
        try:
            parsed: object = json.loads(raw)
        except UnicodeDecodeError, ValueError:
            continue
        if isinstance(parsed, dict):
            content = as_object(parsed).get("content")
            if isinstance(content, str):
                assert json.dumps(content, ensure_ascii=False) not in serialized


def _bundle(tmp_path: Path) -> Path:
    bundle = export_bundle(
        tmp_path, memories=((WS, SECRET, "yesterday"),), name=f"{SECRET}-bundle"
    )
    manifest = load_manifest(bundle)
    manifest["reflectlog_version"] = f"{SECRET}-version"
    store_manifest(bundle, manifest)
    return bundle


@pytest.mark.parametrize("expected", [WS, WS.upper(), f" \t{WS.upper()}\n"])
def test_export_is_ready_with_canonical_expected_workspace(
    tmp_path: Path, expected: str
) -> None:
    bundle = _bundle(tmp_path)

    result = _verify(bundle, expected)

    _assert_result(result, status=Status.READY)
    _assert_payload(result, bundle)


def test_empty_export_is_ready(tmp_path: Path) -> None:
    bundle = export_bundle(tmp_path, memories=(), name=f"{SECRET}-bundle")

    result = _verify(bundle)

    _assert_result(result, status=Status.READY, count=0)
    assert (bundle / "memories.jsonl").read_bytes() == b""
    _assert_payload(result, bundle)


def test_exported_content_is_not_subject_to_add_validation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    records = (
        (-5, " \t\n", "yesterday"),
        (0, " leading and trailing ", ""),
        (1, "nul\x00text", None),
        (2, "emoji 🪞", "yesterday"),
        (3, "e\u0301", ""),
        (4, "é", None),
        (2**70, "x" * (2 * 1024 * 1024), None),
    )
    write_records(bundle, b"".join(record_line(*row) for row in records))

    result = _verify(bundle)

    _assert_result(result, status=Status.READY, count=len(records))
    _assert_payload(result, bundle)


@pytest.mark.parametrize(
    ("journal", "journal_table", "reasons"),
    [
        (((WS, "pending", "add", SECRET),), True, ("journal_pending_intent",)),
        (((WS, "weird", "add", SECRET),), True, ("journal_unknown_evidence",)),
        (
            ((OTHER, "completed", "add", SECRET),),
            True,
            ("journal_foreign_workspace_evidence",),
        ),
        ((), False, ("journal_missing",)),
        (
            (
                (WS, "pending", "add", SECRET),
                (WS, "weird", "add", SECRET),
                (OTHER, "completed", "add", SECRET),
            ),
            True,
            (
                "journal_foreign_workspace_evidence",
                "journal_pending_intent",
                "journal_unknown_evidence",
            ),
        ),
    ],
)
def test_exporter_unresolved_evidence_is_valid_but_not_ready(
    tmp_path: Path,
    journal: tuple[JournalRow, ...],
    journal_table: bool,
    reasons: tuple[str, ...],
) -> None:
    bundle = export_bundle(
        tmp_path,
        memories=((WS, SECRET, None),),
        journal=journal,
        journal_table=journal_table,
        name=f"{SECRET}-bundle",
    )

    result = _verify(bundle)

    _assert_result(result, status=Status.NOT_READY, ready=False, reasons=reasons)
    _assert_payload(result, bundle)


@pytest.mark.parametrize("with_evidence", [False, True])
def test_real_exporter_unsupported_journal_is_not_ready(
    tmp_path: Path, with_evidence: bool
) -> None:
    archive = make_archive(
        tmp_path / "archive", memories=((WS, SECRET, None),), journal_table=False
    )
    with closing(sqlite3.connect(archive / "usearch" / "memories.db")) as connection:
        _ = connection.execute("CREATE TABLE replacement_transitions (note TEXT)")
        if with_evidence:
            _ = connection.execute(
                "INSERT INTO replacement_transitions VALUES (?)", (SECRET,)
            )
        connection.commit()
    bundle = tmp_path / f"{SECRET}-bundle"
    _ = ContentSalvageService.from_defaults().salvage(
        archive=str(archive), workspace_id=WS, output=str(bundle)
    )

    result = _verify(bundle)

    reasons = (
        ("journal_unknown_evidence", "journal_unsupported")
        if with_evidence
        else ("journal_unsupported",)
    )
    _assert_result(result, status=Status.NOT_READY, ready=False, reasons=reasons)
    _assert_payload(result, bundle)


@pytest.mark.parametrize(
    ("bad", "code"),
    [
        (
            b'{"content":"bad","content":"again","source_id":2,"source_created_at":null}\n',
            Code.DUPLICATE_KEY,
        ),
        (record_line(1, "different"), Code.RECORD_ID_DUPLICATE),
        (record_line(2, SECRET), Code.RECORD_CONTENT_DUPLICATE),
        (record_line(0, "different"), Code.RECORD_ID_OUT_OF_ORDER),
        *[
            (record_line(value, "different"), Code.RECORD_ID_INVALID)
            for value in (True, 1.0, "2", None)
        ],
        (b'{"content":"different","source_id":2}\n', Code.RECORD_FIELDS_INVALID),
        (
            b'{"content":"different","source_id":2,"source_created_at":null,"extra":1}\n',
            Code.RECORD_FIELDS_INVALID,
        ),
        *[
            (record_line(2, value), Code.RECORD_CONTENT_INVALID)
            for value in (None, "", 42)
        ],
        (record_line(2, "different", 123), Code.RECORD_TIMESTAMP_INVALID),
        (b"\n", Code.JSON_INVALID),
        (b"\xff\n", Code.UTF8_INVALID),
        (b"[]\n", Code.NOT_OBJECT),
        *[
            (
                b'{"content":"different","source_id":'
                + token
                + b',"source_created_at":null}\n',
                Code.NON_FINITE_NUMBER,
            )
            for token in (b"NaN", b"Infinity", b"-Infinity")
        ],
        (
            b'{"content":"\\ud800","source_id":2,"source_created_at":null}\n',
            Code.RECORD_CONTENT_INVALID,
        ),
        (record_line(2, "different").rstrip(b"\n") + b"garbage\n", Code.JSON_INVALID),
    ],
)
@pytest.mark.parametrize("independent_failures", [False, True])
def test_first_invalid_record_does_not_hide_digest_or_count(
    tmp_path: Path, bad: bytes, code: Code, independent_failures: bool
) -> None:
    bundle = _bundle(tmp_path)
    write_records(bundle, record_line(1, SECRET) + bad + b"also invalid\n")
    if independent_failures:
        manifest = load_manifest(bundle)
        records = as_object(manifest["records"])
        records.update(sha256="0" * 64, count=4)
        store_manifest(bundle, manifest)

    result = _verify(bundle)

    issues: tuple[Issue, ...] = ((code, BundleEntry.RECORDS, 2),)
    if independent_failures:
        issues += (
            (Code.DIGEST_MISMATCH, BundleEntry.RECORDS, None),
            (Code.COUNT_MISMATCH, BundleEntry.RECORDS, None),
        )
    _assert_result(result, status=Status.INVALID, count=None, issues=issues)
    _assert_payload(result, bundle)


@pytest.mark.parametrize("resync", [False, True])
@pytest.mark.parametrize("cut", [1, 20, 80])
def test_truncation_reports_unterminated_line_and_independent_integrity(
    tmp_path: Path, resync: bool, cut: int
) -> None:
    bundle = _bundle(tmp_path)
    original = record_line(1, SECRET) + record_line(2, "private-second-record")
    write_records(bundle, original)
    write_records(bundle, original[:-cut], resync=resync)

    result = _verify(bundle)

    line = 1 if cut == 80 else 2
    issues: tuple[Issue, ...] = (
        (Code.MISSING_FINAL_NEWLINE, BundleEntry.RECORDS, line),
    )
    if not resync:
        issues += ((Code.DIGEST_MISMATCH, BundleEntry.RECORDS, None),)
        if line == 1:
            issues += ((Code.COUNT_MISMATCH, BundleEntry.RECORDS, None),)
    _assert_result(result, status=Status.INVALID, count=None, issues=issues)
    _assert_payload(result, bundle)


def test_records_bom_is_invalid(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    write_records(bundle, codecs.BOM_UTF8 + record_line(1, SECRET))

    result = _verify(bundle)

    _assert_result(
        result,
        status=Status.INVALID,
        count=None,
        issues=((Code.BOM_PRESENT, BundleEntry.RECORDS, 1),),
    )
    _assert_payload(result, bundle)


@pytest.mark.parametrize("mutation", ["byte", "sha256", "count-low", "count-high"])
def test_integrity_mismatch_is_independent_of_valid_records(
    tmp_path: Path, mutation: str
) -> None:
    bundle = _bundle(tmp_path)
    manifest = load_manifest(bundle)
    records = as_object(manifest["records"])
    if mutation == "byte":
        write_records(
            bundle,
            record_line(1, SECRET.replace("S", "s", 1), "yesterday"),
            resync=False,
        )
    elif mutation == "sha256":
        records["sha256"] = "a" * 64
    else:
        records["count"] = 0 if mutation == "count-low" else 2
    store_manifest(bundle, manifest)

    result = _verify(bundle)

    code = Code.COUNT_MISMATCH if mutation.startswith("count") else Code.DIGEST_MISMATCH
    _assert_result(
        result, status=Status.INVALID, issues=((code, BundleEntry.RECORDS, None),)
    )
    _assert_payload(result, bundle)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b"not JSON", Code.JSON_INVALID),
        (b"[]", Code.NOT_OBJECT),
        (codecs.BOM_UTF8 + b"{}", Code.BOM_PRESENT),
        (b"\xff", Code.UTF8_INVALID),
        (b'{"format":1,"format":2}', Code.DUPLICATE_KEY),
    ],
)
def test_invalid_manifest_encoding_or_json(
    tmp_path: Path, raw: bytes, code: Code
) -> None:
    bundle = _bundle(tmp_path)
    _ = (bundle / "manifest.json").write_bytes(raw)

    result = _verify(bundle)

    _assert_result(
        result,
        status=Status.INVALID,
        ready=None,
        issues=((code, BundleEntry.MANIFEST, None),),
    )
    _assert_payload(result, bundle)


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("format",), f"{SECRET}-unsupported", Code.FORMAT_UNSUPPORTED),
        *[
            (("format_version",), version, Code.VERSION_UNSUPPORTED)
            for version in (2, "1", True, 1.0)
        ],
        (("extra",), SECRET, Code.MANIFEST_SCHEMA_INVALID),
        (("records", "count"), "1", Code.MANIFEST_SCHEMA_INVALID),
        (("identity", "status"), {}, Code.MANIFEST_SCHEMA_INVALID),
        (("generation", "status"), [], Code.MANIFEST_SCHEMA_INVALID),
        (
            ("journal", "pending"),
            [{"columns": [{"name": "id", "type": {}, "value": 1}]}],
            Code.MANIFEST_SCHEMA_INVALID,
        ),
        (("journal", "pending"), {}, Code.MANIFEST_SCHEMA_INVALID),
        (
            ("journal", "pending"),
            [{"columns": [{"name": "id", "type": "integer", "value": "1"}]}],
            Code.MANIFEST_SCHEMA_INVALID,
        ),
        (("workspace_id",), OTHER, Code.WORKSPACE_MISMATCH),
        (("workspace_id",), "SALVAGE-WS", Code.WORKSPACE_MISMATCH),
        *[
            (("records", "file"), value, Code.RECORDS_PATH_INVALID)
            for value in (
                "../memories.jsonl",
                "/abs/memories.jsonl",
                "sub/memories.jsonl",
                "memories.jsonl ",
                "MEMORIES.JSONL",
                "",
            )
        ],
        (("rebuild_ready",), False, Code.READINESS_INCONSISTENT),
        (("unresolved_reasons",), ["journal_missing"], Code.READINESS_INCONSISTENT),
        (
            ("unresolved_reasons",),
            [f"{SECRET}-unknown"],
            Code.UNRESOLVED_REASON_INVALID,
        ),
        (
            ("unresolved_reasons",),
            ["journal_unknown_evidence", "journal_pending_intent"],
            Code.UNRESOLVED_REASON_INVALID,
        ),
        (
            ("unresolved_reasons",),
            ["journal_missing", "journal_missing"],
            Code.UNRESOLVED_REASON_INVALID,
        ),
    ],
)
def test_manifest_semantic_and_schema_failures(
    tmp_path: Path, path: tuple[str, ...], value: object, code: Code
) -> None:
    bundle = _bundle(tmp_path)
    manifest = load_manifest(bundle)
    target = manifest
    for key in path[:-1]:
        target = as_object(target[key])
    target[path[-1]] = value
    store_manifest(bundle, manifest)

    result = _verify(bundle)

    schema_failed = code in {
        Code.MANIFEST_SCHEMA_INVALID,
        Code.FORMAT_UNSUPPORTED,
        Code.VERSION_UNSUPPORTED,
    }
    unknown_reason = path == ("unresolved_reasons",) and value == [f"{SECRET}-unknown"]
    reasons = tuple(as_list(value)) if path == ("unresolved_reasons",) else ()
    expected_reasons = tuple(str(reason) for reason in reasons)
    _assert_result(
        result,
        status=Status.INVALID,
        ready=None if schema_failed or unknown_reason else value is not False,
        reasons=() if unknown_reason else expected_reasons,
        issues=((code, BundleEntry.MANIFEST, None),),
    )
    _assert_payload(result, bundle)


def test_missing_manifest_field_is_invalid(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = load_manifest(bundle)
    del manifest["history"]
    store_manifest(bundle, manifest)

    result = _verify(bundle)

    _assert_result(
        result,
        status=Status.INVALID,
        ready=None,
        issues=((Code.MANIFEST_SCHEMA_INVALID, BundleEntry.MANIFEST, None),),
    )
    _assert_payload(result, bundle)


@pytest.mark.parametrize("expected", ["../x", "", "a b"])
def test_invalid_expected_workspace_has_no_canonical_id(
    tmp_path: Path, expected: str
) -> None:
    bundle = _bundle(tmp_path)

    result = _verify(bundle, expected)

    _assert_result(
        result,
        status=Status.INVALID,
        workspace=None,
        issues=(
            (Code.EXPECTED_WORKSPACE_INVALID, BundleEntry.BUNDLE, None),
            (Code.WORKSPACE_MISMATCH, BundleEntry.MANIFEST, None),
        ),
    )
    _assert_payload(result, bundle)


@pytest.mark.parametrize(
    "contradiction", ["pending-without-reasons", "missing-with-rows"]
)
def test_readiness_must_match_journal_evidence(
    tmp_path: Path, contradiction: str
) -> None:
    bundle = export_bundle(
        tmp_path,
        memories=((WS, SECRET, None),),
        journal=((WS, "pending", "add", SECRET),),
        name=f"{SECRET}-bundle",
    )
    manifest = load_manifest(bundle)
    if contradiction == "pending-without-reasons":
        manifest.update(unresolved_reasons=[], rebuild_ready=True)
        reasons: tuple[str, ...] = ()
    else:
        as_object(manifest["schema"])["journal_status"] = "missing"
        manifest["unresolved_reasons"] = ["journal_missing"]
        reasons = ("journal_missing",)
    store_manifest(bundle, manifest)

    result = _verify(bundle)

    _assert_result(
        result,
        status=Status.INVALID,
        ready=contradiction == "pending-without-reasons",
        reasons=reasons,
        issues=((Code.READINESS_INCONSISTENT, BundleEntry.MANIFEST, None),),
    )
    _assert_payload(result, bundle)
