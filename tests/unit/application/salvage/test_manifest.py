"""Unit tests for deterministic salvage records and manifest rendering."""

import hashlib
import json

import pytest

from reflectlog.application.salvage.manifest import (
    NOT_PRESERVED,
    build_bundle,
    render_records,
    unresolved_reasons,
)
from reflectlog.core.salvage import (
    ActiveRecord,
    GenerationSidecarReport,
    IdentitySidecarReport,
    JournalStatus,
    RawJournalRow,
    RawValue,
    RawValueKind,
    SalvageSnapshot,
    SchemaReport,
    SidecarStatus,
    SourceFingerprint,
    TableSchema,
)
from tests.unit.salvage_support import as_list, as_object


def _row(workspace: str, status: str, kind: str | None) -> RawJournalRow:
    return RawJournalRow(
        columns=(
            ("id", RawValue(RawValueKind.INTEGER, 1)),
            ("workspace_id", RawValue(RawValueKind.TEXT, workspace)),
            ("confidence", RawValue(RawValueKind.REAL, "0.5")),
            ("blob", RawValue(RawValueKind.BLOB, "AP8=")),
            ("none", RawValue(RawValueKind.NULL)),
        ),
        workspace_id=workspace,
        status=status,
        kind=kind,
    )


def _snapshot(
    *,
    records: tuple[ActiveRecord, ...] = (),
    pending: tuple[RawJournalRow, ...] = (),
    unknown: tuple[RawJournalRow, ...] = (),
    foreign: tuple[RawJournalRow, ...] = (),
    journal: JournalStatus = JournalStatus.SUPPORTED,
) -> SalvageSnapshot:
    return SalvageSnapshot(
        workspace_id="ws",
        records=records,
        record_count=len(records),
        pending=pending,
        unknown=unknown,
        foreign=foreign,
        completed_history_count=4,
        foreign_active_count=2,
        schema=SchemaReport((TableSchema("memories", ("id",)),), journal),
    )


def _bundle(snapshot: SalvageSnapshot) -> dict[str, object]:
    bundle = build_bundle(
        tool_version="9.9.9",
        snapshot=snapshot,
        fingerprints=(
            SourceFingerprint("usearch/memories.db", 10, "b" * 64),
            SourceFingerprint(".reflectlog.storage-generation", 2, "a" * 64),
        ),
        identity=IdentitySidecarReport(
            SidecarStatus.PRESENT, (("model", "m"), ("dimensions", 8))
        ),
        generation=GenerationSidecarReport(SidecarStatus.PRESENT, "3"),
    )
    _ = b"".join(bundle.records)
    parsed: dict[str, object] = json.loads(bundle.finalize_manifest())
    return parsed


class TestRenderRecords:
    def test_ordered_lines_are_compact_utf8_with_sorted_keys(self) -> None:
        rendered = b"".join(
            render_records(
                [
                    ActiveRecord(2, 'quote " and \\ and \n newline', "2026-01-01"),
                    ActiveRecord(7, "déjà vu \u2028 \u00e9", None),
                ]
            )
        )

        lines = rendered.decode("utf-8").split("\n")
        assert lines[-1] == ""
        assert lines[0] == (
            '{"content":"quote \\" and \\\\ and \\n newline",'
            '"source_created_at":"2026-01-01","source_id":2}'
        )
        assert lines[1] == (
            '{"content":"déjà vu \u2028 é","source_created_at":null,"source_id":7}'
        )
        assert "\\u" not in lines[1]

    def test_empty_records_render_empty_bytes(self) -> None:
        assert b"".join(render_records([])) == b""


class TestReasons:
    def test_resolved_snapshot_has_no_reasons(self) -> None:
        assert unresolved_reasons(_snapshot()) == ()

    def test_each_kind_of_evidence_has_a_stable_code(self) -> None:
        row = _row("ws", "pending", "add")
        assert unresolved_reasons(_snapshot(pending=(row,))) == (
            "journal_pending_intent",
        )
        assert unresolved_reasons(_snapshot(unknown=(row,))) == (
            "journal_unknown_evidence",
        )
        assert unresolved_reasons(_snapshot(foreign=(row,))) == (
            "journal_foreign_workspace_evidence",
        )
        assert unresolved_reasons(_snapshot(journal=JournalStatus.MISSING)) == (
            "journal_missing",
        )
        assert unresolved_reasons(_snapshot(journal=JournalStatus.UNSUPPORTED)) == (
            "journal_unsupported",
        )

    def test_legacy_journal_with_only_history_is_not_an_unresolved_reason(
        self,
    ) -> None:
        assert (
            unresolved_reasons(_snapshot(journal=JournalStatus.LEGACY_WITHOUT_KIND))
            == ()
        )

    def test_reasons_are_sorted_and_combined(self) -> None:
        row = _row("ws", "pending", "add")
        reasons = unresolved_reasons(
            _snapshot(pending=(row,), foreign=(row,), unknown=(row,))
        )
        assert list(reasons) == sorted(reasons)
        assert len(reasons) == 3


class TestManifest:
    def test_format_one_golden_bytes_and_deferred_finalization(self) -> None:
        bundle = build_bundle(
            tool_version="golden",
            snapshot=_snapshot(
                records=(
                    ActiveRecord(-2, 'é\0"\\\n\u2028', "t\0"),
                    ActiveRecord(9, "last", None),
                )
            ),
            fingerprints=(),
            identity=IdentitySidecarReport(SidecarStatus.MISSING),
            generation=GenerationSidecarReport(SidecarStatus.MISSING),
        )
        with pytest.raises(RuntimeError, match="exhausted"):
            _ = bundle.finalize_manifest()
        expected = (
            '{"content":"é\\u0000\\"\\\\\\n\u2028","source_created_at":"t\\u0000","source_id":-2}\n'
            '{"content":"last","source_created_at":null,"source_id":9}\n'
        ).encode("utf-8")
        assert b"".join(bundle.records) == expected
        manifest = bundle.finalize_manifest()
        expected_manifest = {
            "format": "reflectlog.content-salvage",
            "format_version": 1,
            "reflectlog_version": "golden",
            "workspace_id": "ws",
            "rebuild_ready": True,
            "unresolved_reasons": [],
            "identity": {"status": "missing", "fields": {}},
            "generation": {"status": "missing", "raw": None},
            "source_fingerprints": [],
            "schema": {
                "journal_status": "supported",
                "tables": [{"name": "memories", "columns": ["id"]}],
            },
            "records": {
                "file": "memories.jsonl",
                "count": 2,
                "sha256": hashlib.sha256(expected).hexdigest(),
            },
            "journal": {"pending": [], "unknown": [], "foreign_workspace": []},
            "history": {"completed_journal_rows": 4},
            "foreign_active_count": 2,
            "not_preserved": [
                "embedding identity: provider, model, and dimensions are recorded as observed but the rebuilt workspace gets its own",
                "memory ids: rebuilt memories receive new ids",
                "created_at timestamps: rebuilt memories are stamped when re-added",
                "replacement and archive history: archived memories and completed journal rows are not exported",
                "vectors: embeddings and the full-text index are rebuilt from content",
            ],
        }
        assert manifest == (
            json.dumps(expected_manifest, sort_keys=True, indent=2, ensure_ascii=False)
            + "\n"
        ).encode("utf-8")

    def test_manifest_is_sorted_indented_json_with_trailing_newline(self) -> None:
        bundle = build_bundle(
            tool_version="1",
            snapshot=_snapshot(),
            fingerprints=(),
            identity=IdentitySidecarReport(SidecarStatus.MISSING),
            generation=GenerationSidecarReport(SidecarStatus.MISSING),
        )

        _ = b"".join(bundle.records)
        text = bundle.finalize_manifest().decode("utf-8")
        assert text.endswith("}\n")
        assert (
            text
            == json.dumps(
                json.loads(text), sort_keys=True, indent=2, ensure_ascii=False
            )
            + "\n"
        )

    def test_top_level_facts(self) -> None:
        records = (ActiveRecord(1, "a", "t"),)
        manifest = _bundle(_snapshot(records=records))

        assert manifest["format"] == "reflectlog.content-salvage"
        assert manifest["format_version"] == 1
        assert manifest["reflectlog_version"] == "9.9.9"
        assert manifest["workspace_id"] == "ws"
        assert manifest["rebuild_ready"] is True
        assert manifest["unresolved_reasons"] == []
        assert manifest["identity"] == {
            "status": "present",
            "fields": {"dimensions": 8, "model": "m"},
        }
        assert manifest["generation"] == {"status": "present", "raw": "3"}
        assert manifest["history"] == {"completed_journal_rows": 4}
        assert manifest["foreign_active_count"] == 2
        assert manifest["schema"] == {
            "journal_status": "supported",
            "tables": [{"name": "memories", "columns": ["id"]}],
        }

    def test_fingerprints_are_sorted_by_path(self) -> None:
        manifest = _bundle(_snapshot())

        fingerprints = as_list(manifest["source_fingerprints"])
        assert [as_object(item)["path"] for item in fingerprints] == [
            ".reflectlog.storage-generation",
            "usearch/memories.db",
        ]

    def test_records_entry_matches_the_jsonl_bytes(self) -> None:
        records = (ActiveRecord(1, "a", None), ActiveRecord(2, "b", None))
        bundle = build_bundle(
            tool_version="1",
            snapshot=_snapshot(records=records),
            fingerprints=(),
            identity=IdentitySidecarReport(SidecarStatus.MISSING),
            generation=GenerationSidecarReport(SidecarStatus.MISSING),
        )

        payload = b"".join(bundle.records)
        entry = json.loads(bundle.finalize_manifest())["records"]
        assert entry == {
            "file": "memories.jsonl",
            "count": 2,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def test_raw_journal_evidence_is_lossless_and_separate(self) -> None:
        pending = _row("ws", "pending", "add")
        unknown = _row("ws", "weird", None)
        foreign = _row("other", "completed", "add")
        manifest = _bundle(
            _snapshot(pending=(pending,), unknown=(unknown,), foreign=(foreign,))
        )

        journal = as_object(manifest["journal"])
        assert set(journal) == {"pending", "unknown", "foreign_workspace"}
        first_pending = as_object(as_list(journal["pending"])[0])
        assert first_pending["columns"] == [
            {"name": "id", "type": "integer", "value": 1},
            {"name": "workspace_id", "type": "text", "value": "ws"},
            {"name": "confidence", "type": "real", "value": "0.5"},
            {"name": "blob", "type": "blob", "value": "AP8="},
            {"name": "none", "type": "null", "value": None},
        ]
        first_foreign = as_object(as_list(journal["foreign_workspace"])[0])
        assert as_object(as_list(first_foreign["columns"])[1])["value"] == "other"
        assert manifest["rebuild_ready"] is False
        assert manifest["unresolved_reasons"] == [
            "journal_foreign_workspace_evidence",
            "journal_pending_intent",
            "journal_unknown_evidence",
        ]

    @pytest.mark.parametrize(
        "concept", ["identity", "ids", "created_at", "history", "vectors"]
    )
    def test_not_preserved_names_every_lost_concept(self, concept: str) -> None:
        manifest = _bundle(_snapshot())

        assert manifest["not_preserved"] == list(NOT_PRESERVED)
        assert any(concept in line for line in NOT_PRESERVED)

    def test_rendering_is_deterministic(self) -> None:
        first = _bundle(_snapshot())
        second = _bundle(_snapshot())
        assert first == second
