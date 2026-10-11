"""Unit tests for crash-visible publication of an offline-salvage bundle."""

import os
from pathlib import Path
import stat
import sys

import pytest

from reflectlog.core.exceptions import (
    SalvagePublicationError,
    SalvageUnsafeOutputError,
)
from reflectlog.core.salvage import IBundlePublisher, SalvageBundle
from reflectlog.infrastructure.salvage_publication import (
    FileSystemBundlePublisher,
    OsPublicationFileSystem,
)

BUNDLE = SalvageBundle(records=b'{"a":1}\n', manifest=b"{}\n")
posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX permission bits are not enforced"
)


class FaultyFileSystem(OsPublicationFileSystem):
    """Fails the Nth call (default first) of one operation, optionally by name.

    ``persistent`` keeps failing every matching call from then on, which is how
    a cleanup that cannot succeed is simulated.
    """

    def __init__(
        self,
        operation: str,
        *,
        basename: str | None = None,
        persistent: bool = False,
        error: BaseException | None = None,
        fsync_fails: bool = False,
    ) -> None:
        super().__init__(self._failing_fsync if fsync_fails else os.fsync)
        self.operation = operation
        self.basename = basename
        self.persistent = persistent
        self.error: BaseException = error or OSError(5, "injected fault")
        self.fired = False
        self.descriptors: dict[int, str] = {}

    def _failing_fsync(self, descriptor: int) -> None:
        raise OSError(5, "injected fsync fault")

    def _maybe_fail(self, operation: str, path: str | None) -> None:
        if operation != self.operation:
            return
        if self.basename is not None and (
            path is None or os.path.basename(path) != self.basename
        ):
            return
        if self.fired and not self.persistent:
            return
        self.fired = True
        raise self.error

    def make_directory(self, path: str) -> None:
        self._maybe_fail("make_directory", path)
        super().make_directory(path)

    def create_file(self, path: str) -> int:
        self._maybe_fail("create_file", path)
        descriptor = super().create_file(path)
        self.descriptors[descriptor] = path
        return descriptor

    def write_all(self, descriptor: int, payload: bytes) -> None:
        self._maybe_fail("write_all", self.descriptors.get(descriptor))
        super().write_all(descriptor, payload)

    def sync_file(self, descriptor: int) -> None:
        self._maybe_fail("sync_file", self.descriptors.get(descriptor))
        super().sync_file(descriptor)

    def close_file(self, descriptor: int) -> None:
        path = self.descriptors.pop(descriptor, None)
        super().close_file(descriptor)
        self._maybe_fail("close_file", path)

    def sync_directory(self, path: str) -> None:
        self._maybe_fail("sync_directory", path)
        super().sync_directory(path)

    def remove_file(self, path: str) -> None:
        self._maybe_fail("remove_file", path)
        super().remove_file(path)

    def remove_directory(self, path: str) -> None:
        self._maybe_fail("remove_directory", path)
        super().remove_directory(path)


def _publish(publisher: FileSystemBundlePublisher, output: Path) -> None:
    publisher.publish(str(output), BUNDLE)


def test_publisher_satisfies_the_protocol() -> None:
    assert isinstance(FileSystemBundlePublisher(), IBundlePublisher)


class TestSuccessfulPublication:
    def test_complete_bundle_has_manifest_and_no_marker(self, tmp_path: Path) -> None:
        output = tmp_path / "export"

        _publish(FileSystemBundlePublisher(), output)

        assert sorted(os.listdir(output)) == ["manifest.json", "memories.jsonl"]
        assert (output / "memories.jsonl").read_bytes() == BUNDLE.records
        assert (output / "manifest.json").read_bytes() == BUNDLE.manifest

    @posix_only
    def test_directory_is_0700_and_files_are_0600(self, tmp_path: Path) -> None:
        output = tmp_path / "export"
        previous = os.umask(0)
        try:
            _publish(FileSystemBundlePublisher(), output)
        finally:
            _ = os.umask(previous)

        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        for name in ("manifest.json", "memories.jsonl"):
            assert stat.S_IMODE((output / name).stat().st_mode) == 0o600

    def test_marker_is_present_while_files_are_being_written(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "export"
        observed: list[list[str]] = []

        class Observing(OsPublicationFileSystem):
            def sync_directory(self, path: str) -> None:
                observed.append(sorted(os.listdir(path)))
                super().sync_directory(path)

        _publish(FileSystemBundlePublisher(Observing()), output)

        assert observed == [
            ["INCOMPLETE", "manifest.json", "memories.jsonl"],
            ["manifest.json", "memories.jsonl"],
        ]

    def test_short_writes_are_completed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_write = os.write

        def short_write(descriptor: int, data: bytes | memoryview) -> int:
            return real_write(descriptor, bytes(data)[:3])

        monkeypatch.setattr(os, "write", short_write)
        output = tmp_path / "export"

        _publish(FileSystemBundlePublisher(), output)

        assert (output / "memories.jsonl").read_bytes() == BUNDLE.records

    def test_directory_sync_is_skipped_when_it_cannot_be_opened(
        self, tmp_path: Path
    ) -> None:
        OsPublicationFileSystem().sync_directory(str(tmp_path / "absent"))


class TestNeverOverwrites:
    def test_existing_directory_is_untouched(self, tmp_path: Path) -> None:
        output = tmp_path / "export"
        output.mkdir()
        (output / "keep").write_text("earlier")

        with pytest.raises(SalvageUnsafeOutputError, match="already exists"):
            _publish(FileSystemBundlePublisher(), output)

        assert os.listdir(output) == ["keep"]
        assert (output / "keep").read_text() == "earlier"

    def test_earlier_export_is_untouched_by_a_second_run(self, tmp_path: Path) -> None:
        output = tmp_path / "export"
        _publish(FileSystemBundlePublisher(), output)
        first = {name: (output / name).read_bytes() for name in os.listdir(output)}

        with pytest.raises(SalvageUnsafeOutputError):
            _publish(FileSystemBundlePublisher(), output)

        assert {name: (output / name).read_bytes() for name in os.listdir(output)} == (
            first
        )

    def test_empty_racing_directory_is_not_adopted(self, tmp_path: Path) -> None:
        output = tmp_path / "export"
        output.mkdir()

        with pytest.raises(SalvageUnsafeOutputError):
            _publish(FileSystemBundlePublisher(), output)

        assert os.listdir(output) == []

    def test_existing_file_is_untouched(self, tmp_path: Path) -> None:
        output = tmp_path / "export"
        output.write_text("earlier")

        with pytest.raises(SalvageUnsafeOutputError):
            _publish(FileSystemBundlePublisher(), output)

        assert output.read_text() == "earlier"


class TestFaultsAtEveryStep:
    @pytest.mark.parametrize(
        ("operation", "basename"),
        [
            ("make_directory", None),
            ("create_file", "INCOMPLETE"),
            ("write_all", "INCOMPLETE"),
            ("sync_file", "INCOMPLETE"),
            ("close_file", "INCOMPLETE"),
            ("create_file", "memories.jsonl"),
            ("write_all", "memories.jsonl"),
            ("sync_file", "memories.jsonl"),
            ("close_file", "memories.jsonl"),
            ("create_file", "manifest.json"),
            ("write_all", "manifest.json"),
            ("sync_file", "manifest.json"),
            ("close_file", "manifest.json"),
            ("sync_directory", None),
            ("remove_file", "INCOMPLETE"),
        ],
    )
    def test_failure_leaves_no_completed_looking_output(
        self, tmp_path: Path, operation: str, basename: str | None
    ) -> None:
        output = tmp_path / "export"
        publisher = FileSystemBundlePublisher(
            FaultyFileSystem(operation, basename=basename)
        )

        with pytest.raises(SalvagePublicationError, match="injected fault|create"):
            _publish(publisher, output)

        assert not output.exists()

    def test_failure_of_the_final_directory_sync_rolls_everything_back(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "export"
        calls: list[str] = []

        class SecondSyncFails(OsPublicationFileSystem):
            def sync_directory(self, path: str) -> None:
                calls.append(path)
                if len(calls) == 2:
                    raise OSError(5, "second sync")
                super().sync_directory(path)

        with pytest.raises(SalvagePublicationError, match="rolled back"):
            _publish(FileSystemBundlePublisher(SecondSyncFails()), output)

        assert not output.exists()

    def test_real_file_fsync_failure_is_an_error(self, tmp_path: Path) -> None:
        output = tmp_path / "export"
        publisher = FileSystemBundlePublisher(
            FaultyFileSystem("unused", fsync_fails=True)
        )

        with pytest.raises(SalvagePublicationError, match="injected fsync fault"):
            _publish(publisher, output)

        assert not output.exists()

    def test_directory_creation_failure_creates_nothing(self, tmp_path: Path) -> None:
        output = tmp_path / "missing-parent" / "export"

        with pytest.raises(SalvagePublicationError, match="cannot create"):
            _publish(FileSystemBundlePublisher(), output)

        assert not output.parent.exists()

    def test_earlier_export_is_untouched_when_a_later_run_fails(
        self, tmp_path: Path
    ) -> None:
        first = tmp_path / "first"
        _publish(FileSystemBundlePublisher(), first)
        before = sorted(os.listdir(first))

        with pytest.raises(SalvagePublicationError):
            _publish(
                FileSystemBundlePublisher(
                    FaultyFileSystem("write_all", basename="manifest.json")
                ),
                tmp_path / "second",
            )

        assert sorted(os.listdir(first)) == before
        assert not (tmp_path / "second").exists()

    def test_interrupt_rolls_back_and_propagates(self, tmp_path: Path) -> None:
        output = tmp_path / "export"
        publisher = FileSystemBundlePublisher(
            FaultyFileSystem(
                "write_all", basename="manifest.json", error=KeyboardInterrupt()
            )
        )

        with pytest.raises(KeyboardInterrupt):
            _publish(publisher, output)

        assert not output.exists()


class TestFailedCleanupStaysVisible:
    @pytest.mark.parametrize("interrupt", [False, True], ids=["oserror", "interrupt"])
    @pytest.mark.parametrize("restore_fails", [False, True], ids=["marked", "unmarked"])
    def test_post_unlink_failure_reports_incomplete_cleanup(
        self, tmp_path: Path, interrupt: bool, restore_fails: bool
    ) -> None:
        output = tmp_path / "export"
        failure = KeyboardInterrupt() if interrupt else OSError(5, "second sync")

        class StubbornFinalSync(FaultyFileSystem):
            def __init__(self) -> None:
                super().__init__(
                    "remove_file", basename="manifest.json", persistent=True
                )
                self.syncs = 0
                self.markers = 0

            def sync_directory(self, path: str) -> None:
                self.syncs += 1
                if self.syncs == 2:
                    raise failure
                super().sync_directory(path)

            def create_file(self, path: str) -> int:
                if os.path.basename(path) == "INCOMPLETE":
                    self.markers += 1
                    if restore_fails and self.markers == 2:
                        raise OSError(5, "cannot restore marker")
                return super().create_file(path)

        with pytest.raises((SalvagePublicationError, KeyboardInterrupt)) as caught:
            _publish(FileSystemBundlePublisher(StubbornFinalSync()), output)

        assert isinstance(caught.value, SalvagePublicationError)
        assert (output / "manifest.json").exists()
        assert (output / "memories.jsonl").exists()
        assert "rolled back" not in str(caught.value)
        assert "cleanup was incomplete" in str(caught.value)
        assert caught.value.__cause__ is failure
        if restore_fails:
            assert "deleted before use" in str(caught.value)
        else:
            assert (
                (output / "INCOMPLETE").read_bytes().startswith(b"reflectlog salvage")
            )
            assert "marked INCOMPLETE" in str(caught.value)

    @pytest.mark.parametrize(
        ("fail_step", "fail_removal"),
        [
            (("write_all", "manifest.json"), "memories.jsonl"),
            (("write_all", "manifest.json"), "manifest.json"),
            (("sync_directory", None), "memories.jsonl"),
        ],
    )
    def test_marker_survives_a_removal_that_fails(
        self,
        tmp_path: Path,
        fail_step: tuple[str, str | None],
        fail_removal: str,
    ) -> None:
        output = tmp_path / "export"

        class Stubborn(OsPublicationFileSystem):
            def __init__(self) -> None:
                super().__init__()
                self.descriptors: dict[int, str] = {}

            def create_file(self, path: str) -> int:
                descriptor = super().create_file(path)
                self.descriptors[descriptor] = path
                return descriptor

            def write_all(self, descriptor: int, payload: bytes) -> None:
                if fail_step[0] == "write_all" and (
                    os.path.basename(self.descriptors[descriptor]) == fail_step[1]
                ):
                    raise OSError(5, "write fault")
                super().write_all(descriptor, payload)

            def sync_directory(self, path: str) -> None:
                if fail_step[0] == "sync_directory":
                    raise OSError(5, "sync fault")
                super().sync_directory(path)

            def remove_file(self, path: str) -> None:
                if os.path.basename(path) == fail_removal:
                    raise OSError(5, "cannot remove")
                super().remove_file(path)

        with pytest.raises(SalvagePublicationError):
            _publish(FileSystemBundlePublisher(Stubborn()), output)

        assert (output / "INCOMPLETE").exists()

    def test_marker_survives_when_its_own_removal_keeps_failing(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "export"
        publisher = FileSystemBundlePublisher(
            FaultyFileSystem("remove_file", basename="INCOMPLETE", persistent=True)
        )

        with pytest.raises(SalvagePublicationError):
            _publish(publisher, output)

        assert (output / "INCOMPLETE").exists()
        assert not (output / "manifest.json").exists()

    def test_a_directory_that_cannot_be_removed_holds_no_files(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "export"

        class Stuck(OsPublicationFileSystem):
            def sync_directory(self, path: str) -> None:
                raise OSError(5, "sync fault")

            def remove_directory(self, path: str) -> None:
                raise OSError(16, "busy")

        with pytest.raises(SalvagePublicationError):
            _publish(FileSystemBundlePublisher(Stuck()), output)

        assert os.listdir(output) == []

    def test_missing_file_during_cleanup_is_skipped(self, tmp_path: Path) -> None:
        output = tmp_path / "export"

        class VanishingFiles(OsPublicationFileSystem):
            def sync_directory(self, path: str) -> None:
                os.unlink(os.path.join(path, "memories.jsonl"))
                raise OSError(5, "sync fault")

        with pytest.raises(SalvagePublicationError):
            _publish(FileSystemBundlePublisher(VanishingFiles()), output)

        assert not output.exists()


class TestResolveOutput:
    def test_fresh_path_in_existing_parent_is_accepted(self, tmp_path: Path) -> None:
        resolved = FileSystemBundlePublisher().resolve_output(
            str(tmp_path / "export"), ()
        )

        assert resolved == os.path.join(os.path.realpath(tmp_path), "export")

    def test_existing_directory_file_and_dangling_symlink_are_rejected(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "dir").mkdir()
        (tmp_path / "file").write_text("x")
        (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
        for name in ("dir", "file", "dangling"):
            with pytest.raises(SalvageUnsafeOutputError, match="already exists"):
                _ = FileSystemBundlePublisher().resolve_output(str(tmp_path / name), ())

    def test_missing_parent_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SalvageUnsafeOutputError, match="parent"):
            _ = FileSystemBundlePublisher().resolve_output(
                str(tmp_path / "no" / "export"), ()
            )

    def test_parent_that_is_a_file_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "file").write_text("x")
        with pytest.raises(SalvageUnsafeOutputError, match="parent"):
            _ = FileSystemBundlePublisher().resolve_output(
                str(tmp_path / "file" / "export"), ()
            )

    def test_path_without_a_final_component_is_rejected(self) -> None:
        root = os.path.abspath(os.sep)
        with pytest.raises(SalvageUnsafeOutputError, match="final component"):
            _ = FileSystemBundlePublisher().resolve_output(root, ())

    def test_output_inside_a_protected_directory_is_rejected(
        self, tmp_path: Path
    ) -> None:
        archive = tmp_path / "archive"
        (archive / "nested").mkdir(parents=True)
        publisher = FileSystemBundlePublisher()
        for parent in (archive, archive / "nested"):
            with pytest.raises(SalvageUnsafeOutputError, match="inside"):
                _ = publisher.resolve_output(str(parent / "export"), (str(archive),))

    def test_output_inside_scratch_is_rejected(self, tmp_path: Path) -> None:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        with pytest.raises(SalvageUnsafeOutputError, match="inside"):
            _ = FileSystemBundlePublisher().resolve_output(
                str(scratch / "export"), (str(tmp_path / "archive"), str(scratch))
            )

    def test_symlinked_parent_into_the_archive_is_rejected(
        self, tmp_path: Path
    ) -> None:
        archive = tmp_path / "archive"
        archive.mkdir()
        (tmp_path / "alias").symlink_to(archive, target_is_directory=True)
        with pytest.raises(SalvageUnsafeOutputError, match="inside"):
            _ = FileSystemBundlePublisher().resolve_output(
                str(tmp_path / "alias" / "export"), (str(archive),)
            )

    def test_sibling_of_the_archive_is_accepted(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive"
        archive.mkdir()
        resolved = FileSystemBundlePublisher().resolve_output(
            str(tmp_path / "archive-export"), (str(archive),)
        )
        assert resolved.endswith("archive-export")

    def test_nonexistent_protected_directory_is_ignored(self, tmp_path: Path) -> None:
        resolved = FileSystemBundlePublisher().resolve_output(
            str(tmp_path / "export"), (str(tmp_path / "ghost"),)
        )
        assert resolved.endswith("export")
