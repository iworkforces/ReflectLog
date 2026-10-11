"""Unit tests for archive isolation used by offline content salvage."""

import os
from pathlib import Path
import shutil
import stat
import sys

import pytest

from reflectlog.core.exceptions import (
    SalvageArchiveChangedError,
    SalvageInputError,
)
from reflectlog.core.salvage import IArchiveInspector, SidecarStatus
from reflectlog.infrastructure.salvage_archive import ArchiveInspector
from reflectlog.infrastructure.salvage_snapshot import SqliteSalvageSnapshotReader
from tests.unit.salvage_support import (
    IDENTITY,
    OTHER,
    SECRET,
    WS,
    make_archive,
    make_clean_wal_format_archive,
    make_wal_archive,
    tree_digest,
    write_sidecars,
)

posix_only = pytest.mark.skipif(
    sys.platform == "win32" or (sys.platform != "win32" and os.geteuid() == 0),
    reason="POSIX permissions are not enforced here",
)


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    return make_archive(
        tmp_path / "archive", memories=[(WS, "kept", "t1"), (OTHER, "foreign", None)]
    )


def test_inspector_satisfies_the_protocol() -> None:
    assert isinstance(ArchiveInspector(), IArchiveInspector)


class TestCopy:
    def test_copy_is_readable_and_scratch_is_removed_after_success(
        self, archive: Path
    ) -> None:
        with ArchiveInspector().inspect(str(archive)) as inspection:
            scratch = Path(inspection.scratch_dir)
            assert scratch.name.startswith("reflectlog-salvage-")
            assert Path(inspection.database_path).parent == scratch
            snapshot = SqliteSalvageSnapshotReader().read_snapshot(
                inspection.database_path, WS
            )
            assert [r.content for r in snapshot.records] == ["kept"]
        assert not scratch.exists()

    def test_scratch_is_removed_when_the_body_fails(self, archive: Path) -> None:
        scratch: Path | None = None
        with pytest.raises(RuntimeError, match="boom"):
            with ArchiveInspector().inspect(str(archive)) as inspection:
                scratch = Path(inspection.scratch_dir)
                raise RuntimeError("boom")
        assert scratch is not None
        assert not scratch.exists()

    def test_scratch_is_removed_when_the_copy_fails(
        self, archive: Path, tmp_path: Path
    ) -> None:
        seen: list[str] = []

        def corrupt(source: str, destination: str) -> None:
            seen.append(os.path.dirname(destination))
            _ = shutil.copyfile(source, destination)
            with open(destination, "ab") as handle:
                _ = handle.write(b"x")

        with pytest.raises(SalvageArchiveChangedError, match="does not match"):
            with ArchiveInspector(corrupt).inspect(str(archive)):
                pass
        assert not Path(seen[0]).exists()

    def test_copy_error_is_an_input_error(self, archive: Path) -> None:
        def fail(source: str, destination: str) -> None:
            raise PermissionError(13, "Permission denied")

        with pytest.raises(SalvageInputError, match="cannot copy archive files"):
            with ArchiveInspector(fail).inspect(str(archive)):
                pass

    def test_committed_wal_rows_survive_the_copy(self, tmp_path: Path) -> None:
        wal_archive, row_id = make_wal_archive(
            tmp_path / "archive", tmp_path, "wal only"
        )

        with ArchiveInspector().inspect(str(wal_archive)) as inspection:
            snapshot = SqliteSalvageSnapshotReader().read_snapshot(
                inspection.database_path, WS
            )
            paths = [item.path for item in inspection.fingerprints]

        assert [(r.source_id, r.content) for r in snapshot.records] == [
            (row_id, "wal only")
        ]
        assert "usearch/memories.db-wal" in paths
        assert "usearch/memories.db-shm" in paths

    def test_wal_format_database_without_wal_file_gets_an_empty_scratch_wal(
        self, tmp_path: Path
    ) -> None:
        clean = make_clean_wal_format_archive(tmp_path / "archive", tmp_path, "clean")

        with ArchiveInspector().inspect(str(clean)) as inspection:
            snapshot = SqliteSalvageSnapshotReader().read_snapshot(
                inspection.database_path, WS
            )
            wal_in_archive = [
                item.path
                for item in inspection.fingerprints
                if item.path.endswith("-wal")
            ]

        assert [r.content for r in snapshot.records] == ["clean"]
        assert wal_in_archive == []
        assert not (clean / "usearch" / "memories.db-wal").exists()


class TestFingerprintsAndChangeDetection:
    def test_fingerprints_are_relative_sorted_and_cover_inspected_files(
        self, archive: Path
    ) -> None:
        with ArchiveInspector().inspect(str(archive)) as inspection:
            fingerprints = inspection.fingerprints

        paths = [item.path for item in fingerprints]
        assert paths == [
            ".reflectlog.embedding-identity.json",
            ".reflectlog.storage-generation",
            "usearch/memories.db",
        ]
        assert all(len(item.sha256) == 64 for item in fingerprints)
        database = archive / "usearch" / "memories.db"
        assert fingerprints[2].size == database.stat().st_size

    def test_unchanged_archive_verifies(self, archive: Path) -> None:
        with ArchiveInspector().inspect(str(archive)) as inspection:
            inspection.verify_unchanged()
            inspection.verify_unchanged()

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda root: (root / "usearch" / "memories.db").write_bytes(b"changed"),
            lambda root: (root / "usearch" / "memories.db-wal").write_bytes(b"new"),
            lambda root: (root / "usearch" / "memories.db-shm").write_bytes(b"new"),
            lambda root: (root / ".reflectlog.storage-generation").write_text("9"),
            lambda root: (root / ".reflectlog.embedding-identity.json").unlink(),
        ],
        ids=["db", "wal-appears", "shm-appears", "generation", "identity-removed"],
    )
    def test_any_change_between_fingerprint_passes_is_detected(
        self, archive: Path, mutate
    ) -> None:
        with ArchiveInspector().inspect(str(archive)) as inspection:
            mutate(archive)
            with pytest.raises(SalvageArchiveChangedError, match="archive changed"):
                inspection.verify_unchanged()

    def test_wal_removal_is_detected(self, tmp_path: Path) -> None:
        wal_archive, _ = make_wal_archive(tmp_path / "archive", tmp_path, "x")
        with ArchiveInspector().inspect(str(wal_archive)) as inspection:
            (wal_archive / "usearch" / "memories.db-wal").unlink()
            with pytest.raises(SalvageArchiveChangedError):
                inspection.verify_unchanged()

    def test_archive_disappearing_is_reported_as_input_error(
        self, archive: Path
    ) -> None:
        with ArchiveInspector().inspect(str(archive)) as inspection:
            shutil.rmtree(archive)
            with pytest.raises(SalvageInputError, match="does not exist"):
                inspection.verify_unchanged()


class TestArchiveIsNeverModified:
    def test_tree_is_identical_after_inspection_and_read(self, tmp_path: Path) -> None:
        wal_archive, _ = make_wal_archive(tmp_path / "archive", tmp_path, "x")
        before = tree_digest(wal_archive)

        with ArchiveInspector().inspect(str(wal_archive)) as inspection:
            _ = SqliteSalvageSnapshotReader().read_snapshot(
                inspection.database_path, WS
            )
            inspection.verify_unchanged()

        assert tree_digest(wal_archive) == before

    @posix_only
    def test_a_read_only_archive_is_enough(self, archive: Path) -> None:
        paths = [archive, archive / "usearch", *archive.rglob("*")]
        before = tree_digest(archive)
        for path in paths:
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        try:
            with ArchiveInspector().inspect(str(archive)) as inspection:
                snapshot = SqliteSalvageSnapshotReader().read_snapshot(
                    inspection.database_path, WS
                )
                inspection.verify_unchanged()
            assert [r.content for r in snapshot.records] == ["kept"]
            assert tree_digest(archive) == before
        finally:
            for path in paths:
                os.chmod(path, 0o755 if path.is_dir() else 0o644)


class TestRejectedInput:
    def test_missing_archive(self, tmp_path: Path) -> None:
        with pytest.raises(SalvageInputError, match="does not exist"):
            with ArchiveInspector().inspect(str(tmp_path / "nope")):
                pass

    def test_archive_that_is_a_file(self, tmp_path: Path) -> None:
        (tmp_path / "file").write_text("x")
        with pytest.raises(SalvageInputError, match="not a directory"):
            with ArchiveInspector().inspect(str(tmp_path / "file")):
                pass

    def test_missing_database(self, tmp_path: Path) -> None:
        (tmp_path / "archive" / "usearch").mkdir(parents=True)
        with pytest.raises(SalvageInputError, match="SQLite database is missing"):
            with ArchiveInspector().inspect(str(tmp_path / "archive")):
                pass

    def test_wal_without_database_is_missing_database(self, tmp_path: Path) -> None:
        (tmp_path / "archive" / "usearch").mkdir(parents=True)
        (tmp_path / "archive" / "usearch" / "memories.db-wal").write_bytes(b"w")
        with pytest.raises(SalvageInputError, match="SQLite database is missing"):
            with ArchiveInspector().inspect(str(tmp_path / "archive")):
                pass

    def test_symlinked_archive_root(self, archive: Path, tmp_path: Path) -> None:
        link = tmp_path / "link"
        link.symlink_to(archive, target_is_directory=True)
        with pytest.raises(SalvageInputError, match="archive directory is a symlink"):
            with ArchiveInspector().inspect(str(link)):
                pass

    def test_symlinked_usearch_directory(self, archive: Path, tmp_path: Path) -> None:
        elsewhere = tmp_path / "elsewhere"
        _ = (archive / "usearch").rename(elsewhere)
        (archive / "usearch").symlink_to(elsewhere, target_is_directory=True)
        with pytest.raises(SalvageInputError, match="usearch is a symlink"):
            with ArchiveInspector().inspect(str(archive)):
                pass

    @pytest.mark.parametrize(
        "name",
        [
            "usearch/memories.db",
            "usearch/memories.db-wal",
            "usearch/memories.db-shm",
            ".reflectlog.embedding-identity.json",
            ".reflectlog.storage-generation",
        ],
    )
    def test_symlinked_inspected_file(
        self, archive: Path, tmp_path: Path, name: str
    ) -> None:
        target = tmp_path / "target"
        target.write_bytes(b"x")
        link = archive / name
        link.unlink(missing_ok=True)
        link.symlink_to(target)
        with pytest.raises(SalvageInputError, match="is a symlink"):
            with ArchiveInspector().inspect(str(archive)):
                pass

    def test_database_that_is_a_directory(self, archive: Path) -> None:
        (archive / "usearch" / "memories.db").unlink()
        (archive / "usearch" / "memories.db").mkdir()
        with pytest.raises(SalvageInputError, match="not a regular file"):
            with ArchiveInspector().inspect(str(archive)):
                pass

    @posix_only
    def test_unreadable_database(self, archive: Path) -> None:
        database = archive / "usearch" / "memories.db"
        database.chmod(0)
        try:
            with pytest.raises(SalvageInputError, match="cannot read"):
                with ArchiveInspector().inspect(str(archive)):
                    pass
        finally:
            database.chmod(stat.S_IRUSR | stat.S_IWUSR)


class TestSidecars:
    def test_identity_is_reported_raw_without_validation(self, archive: Path) -> None:
        write_sidecars(
            archive,
            identity={**IDENTITY, "provider": "somebody-else", "dimensions": 7},
        )

        with ArchiveInspector().inspect(str(archive)) as inspection:
            report = inspection.identity

        assert report.status is SidecarStatus.PRESENT
        assert dict(report.fields) == {
            "dimensions": 7,
            "model": "tencent/WeMM-Embedding-2B",
            "provider": "somebody-else",
            "version": 1,
            "workspace_id": WS,
        }

    def test_identity_missing(self, archive: Path) -> None:
        (archive / ".reflectlog.embedding-identity.json").unlink()
        with ArchiveInspector().inspect(str(archive)) as inspection:
            assert inspection.identity.status is SidecarStatus.MISSING
            assert inspection.identity.fields == ()

    @pytest.mark.parametrize(
        "payload",
        [
            b"not json",
            b"[1, 2]",
            b'{"provider": {"nested": true}}',
            b'{"dimensions": NaN}',
            b"\xff\xfe",
            b'{"x": "' + b"a" * 70000 + b'"}',
        ],
        ids=["text", "array", "nested", "nan", "bytes", "oversized"],
    )
    def test_identity_malformed(self, archive: Path, payload: bytes) -> None:
        write_sidecars(archive, identity=payload)
        with ArchiveInspector().inspect(str(archive)) as inspection:
            assert inspection.identity.status is SidecarStatus.MALFORMED
            assert inspection.identity.fields == ()

    def test_identity_with_null_and_extra_fields(self, archive: Path) -> None:
        write_sidecars(archive, identity={"provider": None, "extra": [1]})
        with ArchiveInspector().inspect(str(archive)) as inspection:
            assert inspection.identity.status is SidecarStatus.PRESENT
            assert dict(inspection.identity.fields) == {"provider": None}

    @posix_only
    def test_identity_unreadable(self, archive: Path) -> None:
        sidecar = archive / ".reflectlog.embedding-identity.json"
        sidecar.chmod(0)
        try:
            with ArchiveInspector().inspect(str(archive)) as inspection:
                assert inspection.identity.status is SidecarStatus.UNREADABLE
                paths = [item.path for item in inspection.fingerprints]
        finally:
            sidecar.chmod(stat.S_IRUSR | stat.S_IWUSR)
        assert ".reflectlog.embedding-identity.json" not in paths

    def test_generation_present_is_raw_text(self, archive: Path) -> None:
        write_sidecars(archive, generation="  12\n")
        with ArchiveInspector().inspect(str(archive)) as inspection:
            assert inspection.generation.status is SidecarStatus.PRESENT
            assert inspection.generation.raw == "12"

    def test_missing_generation_is_not_defaulted_to_zero(self, archive: Path) -> None:
        (archive / ".reflectlog.storage-generation").unlink()
        with ArchiveInspector().inspect(str(archive)) as inspection:
            assert inspection.generation.status is SidecarStatus.MISSING
            assert inspection.generation.raw is None

    @pytest.mark.parametrize(
        ("payload", "raw"),
        [
            (b"abc", "abc"),
            (b"", ""),
            (b"-1", "-1"),
            (b"\xd9\xa1", "\u0661"),
            (b"\xff", None),
        ],
        ids=["text", "empty", "negative", "unicode-digit", "bytes"],
    )
    def test_generation_malformed(
        self, archive: Path, payload: bytes, raw: str | None
    ) -> None:
        write_sidecars(archive, generation=payload)
        with ArchiveInspector().inspect(str(archive)) as inspection:
            assert inspection.generation.status is SidecarStatus.MALFORMED
            assert inspection.generation.raw == raw

    def test_oversized_generation_is_malformed_but_fingerprinted(
        self, archive: Path
    ) -> None:
        write_sidecars(archive, generation=b"1" * 70000)
        with ArchiveInspector().inspect(str(archive)) as inspection:
            assert inspection.generation.status is SidecarStatus.MALFORMED
            assert ".reflectlog.storage-generation" in [
                item.path for item in inspection.fingerprints
            ]


def test_memory_content_never_appears_in_input_errors(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "archive", memories=[(WS, SECRET, None)])
    (archive / "usearch" / "memories.db").unlink()
    with pytest.raises(SalvageInputError) as caught:
        with ArchiveInspector().inspect(str(archive)):
            pass
    assert SECRET not in str(caught.value)
