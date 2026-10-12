"""Unit tests for the offline content-salvage application service."""

import ast
from collections.abc import Generator
import contextlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

import pytest

from reflectlog.application.memory import replacement_recovery
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.memory.workspace_registry import WorkspaceRegistry
from reflectlog.application.salvage.service import (
    ContentSalvageService,
    SalvageResult,
)
from reflectlog.core.exceptions import (
    SalvageArchiveChangedError,
    SalvageError,
    SalvageInputError,
    SalvageMalformedRecordError,
    SalvagePublicationError,
    SalvageUnsafeOutputError,
    SalvageUnsupportedSchemaError,
)
from reflectlog.core.salvage import IArchiveInspection, SalvageSnapshot
from reflectlog.infrastructure.memory_store import MemoryStore
from reflectlog.infrastructure.salvage_archive import ArchiveInspector
from reflectlog.infrastructure.salvage_publication import (
    FileSystemBundlePublisher,
    OsPublicationFileSystem,
)
from reflectlog.infrastructure.salvage_snapshot import SqliteSalvageSnapshotReader
from reflectlog.infrastructure.tantivy_engine import TantivyEngine
from reflectlog.infrastructure.usearch_engine import USearchEngine
from tests.unit.salvage_support import (
    OTHER,
    SECRET,
    WS,
    as_list,
    as_object,
    make_archive,
    make_wal_archive,
    tree_digest,
)

REPO = Path(__file__).resolve().parents[4]


def _salvage(archive: Path, output: Path, workspace: str = WS) -> SalvageResult:
    return ContentSalvageService.from_defaults().salvage(
        archive=str(archive), workspace_id=workspace, output=str(output)
    )


def _lines(output: Path) -> list[dict[str, object]]:
    text = (output / "memories.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _manifest(output: Path) -> dict[str, object]:
    manifest: dict[str, object] = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    return manifest


class RecordingInspector:
    """Wraps the real inspector and remembers each scratch directory."""

    def __init__(self) -> None:
        self.scratch_dirs: list[str] = []
        self.calls = 0
        self._inner = ArchiveInspector()

    @contextlib.contextmanager
    def inspect(self, archive_dir: str) -> Generator[IArchiveInspection]:
        self.calls += 1
        with self._inner.inspect(archive_dir) as inspection:
            self.scratch_dirs.append(inspection.scratch_dir)
            yield inspection


class TestExport:
    def test_wal_only_committed_row_is_exported_for_the_requested_workspace(
        self, tmp_path: Path
    ) -> None:
        archive, row_id = make_wal_archive(tmp_path / "archive", tmp_path, "wal row")
        output = tmp_path / "export"

        result = _salvage(archive, output)

        (line,) = _lines(output)
        assert (line["source_id"], line["content"]) == (row_id, "wal row")
        assert isinstance(line["source_created_at"], str)
        assert result == SalvageResult(
            output=os.path.join(os.path.realpath(tmp_path), "export"),
            record_count=1,
            rebuild_ready=True,
            unresolved_reasons=(),
        )
        assert sorted(os.listdir(output)) == ["manifest.json", "memories.jsonl"]

    def test_mixed_workspaces_export_only_the_requested_one_sorted(
        self, tmp_path: Path
    ) -> None:
        archive = make_archive(
            tmp_path / "archive",
            memories=[
                (WS, "second", "t2"),
                (OTHER, f"{SECRET} foreign", "t3"),
                (WS, "third", None),
            ],
        )
        output = tmp_path / "export"

        _ = _salvage(archive, output)

        assert [(r["source_id"], r["content"]) for r in _lines(output)] == [
            (1, "second"),
            (3, "third"),
        ]
        manifest = _manifest(output)
        assert manifest["foreign_active_count"] == 1
        assert SECRET not in (output / "manifest.json").read_text(encoding="utf-8")
        assert SECRET not in (output / "memories.jsonl").read_text(encoding="utf-8")

    def test_workspace_id_is_canonicalised_like_the_server_does(
        self, tmp_path: Path
    ) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "kept", None)])

        result = _salvage(archive, tmp_path / "export", workspace=f"  {WS.upper()} ")

        assert result.record_count == 1
        assert _manifest(tmp_path / "export")["workspace_id"] == WS

    def test_manifest_records_identity_generation_and_fingerprints_without_paths(
        self, tmp_path: Path
    ) -> None:
        archive, _ = make_wal_archive(tmp_path / "archive", tmp_path, "wal row")
        output = tmp_path / "export"

        _ = _salvage(archive, output)

        text = (output / "manifest.json").read_text(encoding="utf-8")
        manifest = _manifest(output)
        assert manifest["identity"] == {
            "status": "present",
            "fields": {
                "dimensions": 2048,
                "model": "tencent/WeMM-Embedding-2B",
                "provider": "wemm",
                "version": 1,
                "workspace_id": WS,
            },
        }
        assert manifest["generation"] == {"status": "present", "raw": "3"}
        paths = [
            as_object(item)["path"] for item in as_list(manifest["source_fingerprints"])
        ]
        assert paths == [
            ".reflectlog.embedding-identity.json",
            ".reflectlog.storage-generation",
            "usearch/memories.db",
            "usearch/memories.db-shm",
            "usearch/memories.db-wal",
        ]
        assert str(tmp_path) not in text
        assert "reflectlog-salvage-" not in text
        assert "os.path" not in text

    def test_missing_sidecars_are_reported_missing_not_defaulted(
        self, tmp_path: Path
    ) -> None:
        archive = make_archive(
            tmp_path / "archive", memories=[(WS, "a", None)], generation=None
        )
        (archive / ".reflectlog.embedding-identity.json").unlink()

        _ = _salvage(archive, tmp_path / "export")

        manifest = _manifest(tmp_path / "export")
        assert manifest["identity"] == {"status": "missing", "fields": {}}
        assert manifest["generation"] == {"status": "missing", "raw": None}

    def test_manifest_lists_what_content_salvage_cannot_preserve(
        self, tmp_path: Path
    ) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "a", None)])

        _ = _salvage(archive, tmp_path / "export")

        lost = " ".join(
            str(line)
            for line in as_list(_manifest(tmp_path / "export")["not_preserved"])
        )
        for concept in ("memory ids", "created_at", "vectors", "history", "identity"):
            assert concept in lost

    def test_empty_workspace_is_a_successful_empty_export(self, tmp_path: Path) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(OTHER, "x", None)])

        result = _salvage(archive, tmp_path / "export")

        assert result.record_count == 0
        assert (tmp_path / "export" / "memories.jsonl").read_bytes() == b""


class TestRebuildReadiness:
    def test_pending_intent_is_evidence_not_content(self, tmp_path: Path) -> None:
        archive = make_archive(
            tmp_path / "archive",
            memories=[(WS, "kept", None)],
            journal=[(WS, "pending", "add", f"{SECRET} pending add")],
        )
        output = tmp_path / "export"

        result = _salvage(archive, output)

        assert result.rebuild_ready is False
        assert result.unresolved_reasons == ("journal_pending_intent",)
        assert [r["content"] for r in _lines(output)] == ["kept"]
        assert SECRET not in (output / "memories.jsonl").read_text(encoding="utf-8")
        manifest = _manifest(output)
        assert manifest["rebuild_ready"] is False
        journal = as_object(manifest["journal"])
        assert len(as_list(journal["pending"])) == 1
        assert journal["unknown"] == [] == journal["foreign_workspace"]
        first = as_object(as_list(journal["pending"])[0])
        values = [as_object(cell)["value"] for cell in as_list(first["columns"])]
        assert f"{SECRET} pending add" in values

    @pytest.mark.parametrize(
        ("journal", "kind", "journal_table", "reasons"),
        [
            ([(WS, "weird", "add", "n")], True, True, ("journal_unknown_evidence",)),
            (
                [(OTHER, "completed", "add", "n")],
                True,
                True,
                ("journal_foreign_workspace_evidence",),
            ),
            ([], True, False, ("journal_missing",)),
            ([(WS, "pending", None, "n")], False, True, ("journal_unknown_evidence",)),
        ],
        ids=["unknown-status", "foreign-completed", "no-journal", "legacy-pending"],
    )
    def test_unresolved_evidence_exports_but_is_not_rebuild_ready(
        self,
        tmp_path: Path,
        journal: list[tuple[str, str, str | None, str]],
        kind: bool,
        journal_table: bool,
        reasons: tuple[str, ...],
    ) -> None:
        archive = make_archive(
            tmp_path / "archive",
            memories=[(WS, "kept", None)],
            journal=journal,
            kind=kind,
            journal_table=journal_table,
        )

        result = _salvage(archive, tmp_path / "export")

        assert result.record_count == 1
        assert result.rebuild_ready is False
        assert result.unresolved_reasons == reasons

    def test_completed_history_alone_is_rebuild_ready(self, tmp_path: Path) -> None:
        archive = make_archive(
            tmp_path / "archive",
            memories=[(WS, "kept", None)],
            journal=[(WS, "completed", "replace", "old")],
        )

        result = _salvage(archive, tmp_path / "export")

        assert result.rebuild_ready is True
        assert _manifest(tmp_path / "export")["history"] == {
            "completed_journal_rows": 1
        }

    def test_unsupported_journal_schema_is_not_rebuild_ready(
        self, tmp_path: Path
    ) -> None:
        archive = make_archive(
            tmp_path / "archive", memories=[(WS, "kept", None)], journal_table=False
        )
        connection = sqlite3.connect(archive / "usearch" / "memories.db")
        try:
            _ = connection.execute("CREATE TABLE replacement_transitions (note TEXT)")
            connection.commit()
        finally:
            connection.close()

        result = _salvage(archive, tmp_path / "export")

        assert result.unresolved_reasons == ("journal_unsupported",)


class TestDeterminism:
    def test_repeat_runs_and_a_moved_copy_are_byte_identical(
        self, tmp_path: Path
    ) -> None:
        archive, _ = make_wal_archive(tmp_path / "archive", tmp_path, "wal row")
        moved = tmp_path / "elsewhere" / "renamed-archive"
        _ = shutil.copytree(archive, moved)

        _ = _salvage(archive, tmp_path / "one")
        _ = _salvage(archive, tmp_path / "two")
        _ = _salvage(moved, tmp_path / "three")

        for name in ("memories.jsonl", "manifest.json"):
            assert (tmp_path / "one" / name).read_bytes() == (
                tmp_path / "two" / name
            ).read_bytes()
            assert (tmp_path / "one" / name).read_bytes() == (
                tmp_path / "three" / name
            ).read_bytes()


class TestArchiveAndScratchHygiene:
    def test_archive_tree_is_identical_after_success(self, tmp_path: Path) -> None:
        archive, _ = make_wal_archive(tmp_path / "archive", tmp_path, "wal row")
        before = tree_digest(archive)

        _ = _salvage(archive, tmp_path / "export")

        assert tree_digest(archive) == before

    def test_archive_tree_is_identical_after_failure(self, tmp_path: Path) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "", None)])
        before = tree_digest(archive)

        with pytest.raises(SalvageMalformedRecordError):
            _ = _salvage(archive, tmp_path / "export")

        assert tree_digest(archive) == before

    def test_scratch_is_removed_on_success_and_on_failure(self, tmp_path: Path) -> None:
        good = make_archive(tmp_path / "good", memories=[(WS, "a", None)])
        bad = make_archive(tmp_path / "bad", memories=[(WS, "", None)])
        inspector = RecordingInspector()
        service = ContentSalvageService(
            inspector, SqliteSalvageSnapshotReader(), FileSystemBundlePublisher()
        )

        _ = service.salvage(
            archive=str(good), workspace_id=WS, output=str(tmp_path / "o1")
        )
        with pytest.raises(SalvageMalformedRecordError):
            _ = service.salvage(
                archive=str(bad), workspace_id=WS, output=str(tmp_path / "o2")
            )

        assert len(inspector.scratch_dirs) == 2
        assert not any(os.path.exists(path) for path in inspector.scratch_dirs)


class TestFailures:
    def test_missing_database(self, tmp_path: Path) -> None:
        (tmp_path / "archive" / "usearch").mkdir(parents=True)
        with pytest.raises(SalvageInputError, match="SQLite database is missing"):
            _ = _salvage(tmp_path / "archive", tmp_path / "export")
        assert not (tmp_path / "export").exists()

    def test_unsupported_schema(self, tmp_path: Path) -> None:
        archive = make_archive(tmp_path / "archive")
        connection = sqlite3.connect(archive / "usearch" / "memories.db")
        try:
            _ = connection.execute("DROP TABLE memories")
            connection.commit()
        finally:
            connection.close()
        with pytest.raises(SalvageUnsupportedSchemaError):
            _ = _salvage(archive, tmp_path / "export")
        assert not (tmp_path / "export").exists()

    def test_malformed_content_publishes_nothing(self, tmp_path: Path) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, SECRET, None)])
        connection = sqlite3.connect(archive / "usearch" / "memories.db")
        try:
            _ = connection.execute("UPDATE memories SET content = x'ff'")
            connection.commit()
        finally:
            connection.close()
        with pytest.raises(SalvageMalformedRecordError) as caught:
            _ = _salvage(archive, tmp_path / "export")
        assert SECRET not in str(caught.value)
        assert not (tmp_path / "export").exists()

    @pytest.mark.parametrize("workspace", ["", "..", "bad/ws", "x" * 65])
    def test_invalid_workspace_id(self, tmp_path: Path, workspace: str) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "a", None)])
        with pytest.raises(SalvageInputError, match="invalid workspace id"):
            _ = _salvage(archive, tmp_path / "export", workspace=workspace)

    def test_all_failures_share_one_base_class(self, tmp_path: Path) -> None:
        with pytest.raises(SalvageError):
            _ = _salvage(tmp_path / "absent", tmp_path / "export")


class TestUnsafeOutput:
    def test_existing_outputs_are_untouched_and_nothing_is_copied(
        self, tmp_path: Path
    ) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "a", None)])
        existing_dir = tmp_path / "dir"
        existing_dir.mkdir()
        (existing_dir / "keep").write_text("earlier")
        existing_file = tmp_path / "file"
        existing_file.write_text("earlier")
        dangling = tmp_path / "dangling"
        dangling.symlink_to(tmp_path / "nowhere")
        inspector = RecordingInspector()
        service = ContentSalvageService(
            inspector, SqliteSalvageSnapshotReader(), FileSystemBundlePublisher()
        )

        for target in (existing_dir, existing_file, dangling):
            with pytest.raises(SalvageUnsafeOutputError, match="already exists"):
                _ = service.salvage(
                    archive=str(archive), workspace_id=WS, output=str(target)
                )

        assert inspector.calls == 0
        assert os.listdir(existing_dir) == ["keep"]
        assert existing_file.read_text() == "earlier"
        assert dangling.is_symlink() and not dangling.exists()

    def test_output_inside_the_archive_is_rejected_without_creating_it(
        self, tmp_path: Path
    ) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "a", None)])
        before = tree_digest(archive)

        for target in (archive / "export", archive / "usearch" / "export"):
            with pytest.raises(SalvageUnsafeOutputError, match="inside"):
                _ = _salvage(archive, target)

        assert tree_digest(archive) == before

    def test_missing_parent(self, tmp_path: Path) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "a", None)])
        with pytest.raises(SalvageUnsafeOutputError, match="parent"):
            _ = _salvage(archive, tmp_path / "no" / "export")


class TestChangeDuringSalvage:
    def test_archive_change_between_fingerprint_passes_fails_without_output(
        self, tmp_path: Path
    ) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "a", None)])

        class ChangingReader(SqliteSalvageSnapshotReader):
            def read_snapshot(
                self, database_path: str, workspace_id: str
            ) -> SalvageSnapshot:
                snapshot = super().read_snapshot(database_path, workspace_id)
                with open(archive / "usearch" / "memories.db", "ab") as handle:
                    _ = handle.write(b"\0")
                return snapshot

        inspector = RecordingInspector()
        service = ContentSalvageService(
            inspector, ChangingReader(), FileSystemBundlePublisher()
        )

        with pytest.raises(SalvageArchiveChangedError):
            _ = service.salvage(
                archive=str(archive), workspace_id=WS, output=str(tmp_path / "export")
            )

        assert not (tmp_path / "export").exists()
        assert not os.path.exists(inspector.scratch_dirs[0])

    def test_publication_failure_is_reported_and_leaves_no_export(
        self, tmp_path: Path
    ) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "a", None)])

        class NoSpace(OsPublicationFileSystem):
            def write_all(self, descriptor: int, payload: bytes) -> None:
                if payload != b"" and b"format" in payload:
                    raise OSError(28, "No space left on device")
                super().write_all(descriptor, payload)

        service = ContentSalvageService(
            ArchiveInspector(),
            SqliteSalvageSnapshotReader(),
            FileSystemBundlePublisher(NoSpace()),
        )

        with pytest.raises(SalvagePublicationError, match="No space left"):
            _ = service.salvage(
                archive=str(archive), workspace_id=WS, output=str(tmp_path / "export")
            )

        assert not (tmp_path / "export").exists()


class TestNothingHeavyIsUsed:
    def test_no_manager_store_engine_registry_or_replay_is_touched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive, _ = make_wal_archive(tmp_path / "archive", tmp_path, "wal row")

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("salvage must not construct or replay engines")

        for cls in (MemoryManager, WorkspaceRegistry, MemoryStore):
            monkeypatch.setattr(cls, "__init__", forbidden)
        for engine in (USearchEngine, TantivyEngine):
            monkeypatch.setattr(engine, "__init__", forbidden)
        monkeypatch.setattr(
            replacement_recovery, "reconcile_pending_replacements", forbidden
        )
        monkeypatch.setattr(replacement_recovery, "apply_pending_transition", forbidden)

        result = _salvage(archive, tmp_path / "export")

        assert result.record_count == 1

    @pytest.mark.parametrize(
        "relative",
        [
            "reflectlog/core/salvage.py",
            "reflectlog/application/salvage/__init__.py",
            "reflectlog/application/salvage/service.py",
            "reflectlog/application/salvage/manifest.py",
            "reflectlog/infrastructure/salvage_snapshot.py",
            "reflectlog/infrastructure/salvage_archive.py",
            "reflectlog/infrastructure/salvage_publication.py",
        ],
    )
    def test_salvage_modules_import_no_engine_model_or_manager_code(
        self, relative: str
    ) -> None:
        forbidden = (
            "usearch",
            "tantivy",
            "torch",
            "sentence_transformers",
            "numba",
            "ranx",
            "fastmcp",
            "reflectlog.application.memory",
            "reflectlog.application.mcp_server",
            "reflectlog.application.config",
            "reflectlog.application.tools",
            "reflectlog.infrastructure.memory_store",
            "reflectlog.infrastructure.usearch_engine",
            "reflectlog.infrastructure.tantivy_engine",
            "reflectlog.infrastructure.embeddings",
            "reflectlog.infrastructure.embedding_identity",
            "reflectlog.infrastructure.storage_coordinator",
            "reflectlog.utility.scoring",
        )
        tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
                imported.extend(f"{node.module}.{alias.name}" for alias in node.names)

        offenders = [
            name
            for name in imported
            if any(name == bad or name.startswith(f"{bad}.") for bad in forbidden)
        ]
        assert offenders == []
