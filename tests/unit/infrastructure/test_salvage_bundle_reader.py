"""Filesystem-only, read-only capture of salvage bundles."""

import ast
import os
from pathlib import Path
import sys
import tempfile
from threading import Thread
from typing import Literal

import pytest

from reflectlog.core.salvage import (
    INCOMPLETE_MARKER,
    MANIFEST_FILENAME,
    RECORDS_FILENAME,
    BundleCapture,
    BundleEntry,
    CapturedFile,
    VerificationIssue,
)
from reflectlog.core.salvage import (
    VerificationIssueCode as Code,
)
from reflectlog.infrastructure import salvage_bundle_reader as reader_module
from reflectlog.infrastructure.salvage_bundle_reader import FileSystemBundleReader
from tests.unit.salvage_support import tree_digest

MANIFEST = b'{"records":{"path":"memories.jsonl"}}\n'
RECORDS = b'{"content":"exact bytes"}\r\n\x00\xff'
REQUIRED = [
    (MANIFEST_FILENAME, BundleEntry.MANIFEST),
    (RECORDS_FILENAME, BundleEntry.RECORDS),
]


def _bundle(root: Path, records: bytes = RECORDS) -> Path:
    root.mkdir()
    _ = (root / MANIFEST_FILENAME).write_bytes(MANIFEST)
    _ = (root / RECORDS_FILENAME).write_bytes(records)
    return root


def _symlink(path: Path, target: Path, *, directory: bool = False) -> None:
    try:
        path.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")


def _fifo(path: Path) -> None:
    if "mkfifo" not in vars(os):
        pytest.skip("os.mkfifo unavailable")
    try:
        os.mkfifo(path)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"FIFO creation unavailable: {error}")


def _issues(result: BundleCapture) -> set[tuple[Code, BundleEntry]]:
    assert all(issue.line is None for issue in result.issues)
    pairs = {(issue.code, issue.entry) for issue in result.issues}
    assert len(pairs) == len(result.issues)
    return pairs


@pytest.mark.parametrize("records", [RECORDS, b""])
@pytest.mark.parametrize("trailing_separator", [False, True])
def test_capture_returns_exact_bytes(
    tmp_path: Path, records: bytes, trailing_separator: bool
) -> None:
    bundle = _bundle(tmp_path / "bundle", records)

    result = FileSystemBundleReader().capture(
        str(bundle) + (os.sep if trailing_separator else "")
    )

    assert result == BundleCapture(CapturedFile(MANIFEST), CapturedFile(records), ())


@pytest.mark.parametrize("kind", ["missing", "file", "symlink", "dangling"])
@pytest.mark.parametrize("trailing_separator", [False, True])
def test_unsafe_roots_are_rejected_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, trailing_separator: bool
) -> None:
    root = tmp_path / "root"
    expected = Code.BUNDLE_MISSING
    if kind == "file":
        _ = root.write_bytes(b"not a directory")
        expected = Code.BUNDLE_NOT_DIRECTORY
    elif kind in ("symlink", "dangling"):
        target = tmp_path / "target"
        if kind == "symlink":
            _ = _bundle(target)
        _symlink(root, target, directory=True)
        expected = Code.BUNDLE_SYMLINK
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", _forbidden)

        result = FileSystemBundleReader().capture(
            str(root) + (os.sep if trailing_separator else "")
        )

    assert result == BundleCapture(
        None, None, (VerificationIssue(expected, BundleEntry.BUNDLE),)
    )


@pytest.mark.parametrize("kind", ["file", "empty", "directory", "symlink", "dangling"])
def test_every_incomplete_entry_prevents_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    marker = bundle / INCOMPLETE_MARKER
    if kind == "directory":
        marker.mkdir()
    elif kind in ("symlink", "dangling"):
        _symlink(
            marker, bundle / (MANIFEST_FILENAME if kind == "symlink" else "absent")
        )
    else:
        _ = marker.write_bytes(b"incomplete" if kind == "file" else b"")
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", _forbidden)

        result = FileSystemBundleReader().capture(str(bundle))

    assert result == BundleCapture(
        None,
        None,
        (VerificationIssue(Code.INCOMPLETE_MARKER_PRESENT, BundleEntry.MARKER),),
    )


@pytest.mark.parametrize(("filename", "entry"), REQUIRED)
@pytest.mark.parametrize("kind", ["missing", "symlink", "dangling", "directory"])
def test_required_entry_errors_preserve_the_other_file(
    tmp_path: Path, filename: str, entry: BundleEntry, kind: str
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    path = bundle / filename
    path.unlink()
    expected = Code.FILE_MISSING
    if kind == "directory":
        path.mkdir()
        expected = Code.FILE_NOT_REGULAR
    elif kind in ("symlink", "dangling"):
        target = tmp_path / "target"
        if kind == "symlink":
            _ = target.write_bytes(b"external bytes")
        _symlink(path, target)
        expected = Code.FILE_SYMLINK

    result = FileSystemBundleReader().capture(str(bundle))

    assert _issues(result) == {(expected, entry)}
    assert result.manifest == (
        None if entry == BundleEntry.MANIFEST else CapturedFile(MANIFEST)
    )
    assert result.records == (
        None if entry == BundleEntry.RECORDS else CapturedFile(RECORDS)
    )


@pytest.mark.parametrize(("filename", "entry"), REQUIRED)
def test_fifo_is_rejected_without_blocking(
    tmp_path: Path, filename: str, entry: BundleEntry
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    path = bundle / filename
    path.unlink()
    _fifo(path)
    results: list[BundleCapture] = []

    def capture() -> None:
        results.append(FileSystemBundleReader().capture(str(bundle)))

    worker = Thread(target=capture, daemon=True)
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive(), "reader blocked on a FIFO"
    assert len(results) == 1
    assert _issues(results[0]) == {(Code.FILE_NOT_REGULAR, entry)}
    assert results[0].manifest == (
        None if entry == BundleEntry.MANIFEST else CapturedFile(MANIFEST)
    )
    assert results[0].records == (
        None if entry == BundleEntry.RECORDS else CapturedFile(RECORDS)
    )


@pytest.mark.parametrize(("filename", "entry"), REQUIRED)
def test_unreadable_required_file(
    tmp_path: Path, filename: str, entry: BundleEntry
) -> None:
    if sys.platform == "win32":
        pytest.skip("Windows does not enforce POSIX chmod read bits")
    bundle = _bundle(tmp_path / "bundle")
    path = bundle / filename
    previous_mode = path.stat().st_mode
    try:
        try:
            path.chmod(0)
        except OSError as error:
            pytest.skip(f"permission changes unavailable: {error}")
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except PermissionError:
            pass
        else:
            os.close(descriptor)
            pytest.skip(
                "chmod(0) does not deny reads with current privileges (including root)"
            )

        result = FileSystemBundleReader().capture(str(bundle))

        assert _issues(result) == {(Code.FILE_UNREADABLE, entry)}
        assert result.manifest == (
            None if entry == BundleEntry.MANIFEST else CapturedFile(MANIFEST)
        )
        assert result.records == (
            None if entry == BundleEntry.RECORDS else CapturedFile(RECORDS)
        )
    finally:
        path.chmod(previous_mode)


@pytest.mark.parametrize(
    "change", ["append", "rewrite", "replace", "delete", "marker", "root"]
)
def test_observed_changes_discard_affected_capture(
    tmp_path: Path,
    change: Literal["append", "rewrite", "replace", "delete", "marker", "root"],
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    records = bundle / RECORDS_FILENAME
    original = records.stat()
    replacement = tmp_path / "replacement"
    _ = replacement.write_bytes(b"x" * len(RECORDS))
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    if change in ("replace", "root") and original.st_ino == 0:
        pytest.skip("filesystem does not expose inode identity")
    calls: list[bool] = []

    def mutate() -> None:
        calls.append(True)
        match change:
            case "append":
                with records.open("ab") as handle:
                    _ = handle.write(b"extra")
            case "rewrite":
                _ = records.write_bytes(b"x" * len(RECORDS))
                os.utime(
                    records,
                    ns=(original.st_atime_ns, original.st_mtime_ns + 2_000_000_000),
                )
                assert records.stat().st_mtime_ns != original.st_mtime_ns
            case "replace":
                os.replace(replacement, records)
                assert records.stat().st_ino != original.st_ino
            case "delete":
                records.unlink()
            case "marker":
                _ = (bundle / INCOMPLETE_MARKER).write_bytes(b"")
            case "root":
                bundle.rename(tmp_path / "old-bundle")
                _ = _bundle(bundle)

    result = FileSystemBundleReader(after_capture=mutate).capture(str(bundle))

    expected_entry = {"marker": BundleEntry.MARKER, "root": BundleEntry.BUNDLE}.get(
        change, BundleEntry.RECORDS
    )
    assert _issues(result) == {(Code.CHANGED_DURING_READ, expected_entry)}
    assert result.records is None
    assert result.manifest == (
        None if change in ("marker", "root") else CapturedFile(MANIFEST)
    )
    assert calls == [True]


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("capture must not mutate or create scratch")


def test_directory_is_rejected_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    records = bundle / RECORDS_FILENAME
    records.unlink()
    records.mkdir()
    real_open = os.open

    def guarded_open(path: str, flags: int) -> int:
        assert path != str(records), "nonregular entries must never be opened"
        return real_open(path, flags)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", guarded_open)
        result = FileSystemBundleReader().capture(str(bundle))

    assert _issues(result) == {(Code.FILE_NOT_REGULAR, BundleEntry.RECORDS)}
    assert result.records is None


def test_capture_does_not_reread_paths_after_descriptor_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path / "bundle")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", _forbidden)
        patch.setattr(Path, "read_text", _forbidden)
        result = FileSystemBundleReader().capture(str(bundle))

    assert _issues(result) == set()
    assert result.manifest is not None
    assert result.records is not None


def test_incomplete_descriptor_read_is_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    real_read = os.read

    def shortened_read(descriptor: int, size: int) -> bytes:
        return real_read(descriptor, size)[1:]

    with monkeypatch.context() as patch:
        patch.setattr(os, "read", shortened_read)
        result = FileSystemBundleReader().capture(str(bundle))

    assert _issues(result) == {
        (Code.CHANGED_DURING_READ, entry) for _, entry in REQUIRED
    }
    assert result.manifest is None
    assert result.records is None


def test_replacement_between_lstat_and_open_is_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    records = bundle / RECORDS_FILENAME
    replacement = tmp_path / "replacement"
    _ = replacement.write_bytes(b"replacement")
    if records.stat().st_ino == 0 or replacement.stat().st_ino == 0:
        pytest.skip("filesystem does not expose inode identity")
    assert records.stat().st_ino != replacement.stat().st_ino
    real_open = os.open

    def replaced_open(path: str, flags: int) -> int:
        if path == str(records):
            os.replace(replacement, records)
        return real_open(path, flags)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", replaced_open)
        result = FileSystemBundleReader().capture(str(bundle))

    assert _issues(result) == {(Code.CHANGED_DURING_READ, BundleEntry.RECORDS)}
    assert result.records is None


def test_nonregular_open_descriptor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    directory_mode = bundle.stat().st_mode
    real_fstat = os.fstat

    def nonregular_fstat(descriptor: int) -> os.stat_result:
        fields = list(real_fstat(descriptor))
        fields[0] = directory_mode
        return os.stat_result(fields)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fstat", nonregular_fstat)
        result = FileSystemBundleReader().capture(str(bundle))

    assert _issues(result) == {(Code.FILE_NOT_REGULAR, entry) for _, entry in REQUIRED}
    assert result.manifest is None
    assert result.records is None


def test_capture_accepts_unavailable_initial_inode_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    real_lstat = os.lstat
    seen: set[str] = set()

    def initial_inode_unavailable(path: str) -> os.stat_result:
        info = real_lstat(path)
        if path in seen:
            return info
        seen.add(path)
        fields = list(info)
        fields[1] = 0
        return os.stat_result(fields)

    with monkeypatch.context() as patch:
        patch.setattr(os, "lstat", initial_inode_unavailable)
        result = FileSystemBundleReader().capture(str(bundle))

    assert _issues(result) == set()
    assert result.manifest is not None
    assert result.records is not None


@pytest.mark.parametrize("bad", [False, True])
def test_capture_is_read_only_and_scratch_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: bool
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    if bad:
        (bundle / RECORDS_FILENAME).unlink()
    before = tree_digest(bundle)
    parent_before = sorted(tmp_path.iterdir())
    root_mtime = bundle.stat().st_mtime_ns
    opened: list[str] = []
    real_open = os.open

    def read_only_open(path: str, flags: int) -> int:
        assert (
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            == 0
        )
        assert flags & (os.O_WRONLY | os.O_RDWR) == os.O_RDONLY
        if sys.platform != "win32":
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_NONBLOCK
        opened.append(path)
        return real_open(path, flags)

    with monkeypatch.context() as patch:
        for name in (
            "mkstemp",
            "mkdtemp",
            "NamedTemporaryFile",
            "TemporaryDirectory",
            "SpooledTemporaryFile",
        ):
            patch.setattr(tempfile, name, _forbidden)
        for name in (
            "mkdir",
            "makedirs",
            "rename",
            "replace",
            "remove",
            "unlink",
            "rmdir",
            "utime",
            "chmod",
            "truncate",
            "ftruncate",
            "fsync",
            "write",
        ):
            patch.setattr(os, name, _forbidden)
        patch.setattr(os, "open", read_only_open)

        result = FileSystemBundleReader().capture(str(bundle))

    assert tree_digest(bundle) == before
    assert sorted(tmp_path.iterdir()) == parent_before
    assert bundle.stat().st_mtime_ns == root_mtime
    assert opened == [str(bundle / MANIFEST_FILENAME)] + (
        [] if bad else [str(bundle / RECORDS_FILENAME)]
    )
    assert _issues(result) == (
        {(Code.FILE_MISSING, BundleEntry.RECORDS)} if bad else set()
    )


@pytest.mark.parametrize("length", [22, 24])
def test_multiple_chunks_are_byte_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, length: int
) -> None:
    payload = bytes(range(length))
    bundle = _bundle(tmp_path / "bundle", payload)
    monkeypatch.setattr(reader_module, "_CHUNK", 3)

    result = FileSystemBundleReader().capture(str(bundle))

    assert result == BundleCapture(CapturedFile(MANIFEST), CapturedFile(payload), ())


def test_reader_imports_no_mutation_database_engine_or_application_modules() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "reflectlog/infrastructure/salvage_bundle_reader.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
            imported.extend(f"{node.module}.{alias.name}" for alias in node.names)

    forbidden = (
        "sqlite3",
        "tempfile",
        "shutil",
        "subprocess",
        "socket",
        "usearch",
        "tantivy",
        "numba",
        "reflectlog.application",
        "reflectlog.infrastructure.memory_store",
    )
    assert [
        name
        for name in imported
        if any(name == bad or name.startswith(f"{bad}.") for bad in forbidden)
    ] == []
