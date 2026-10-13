"""Read-only capture of an exported content-salvage bundle.

This module only observes. It never writes, creates, renames, deletes, syncs,
or repairs anything, opens no SQLite database or archive, and creates no
scratch files. Symlinks are never followed: the bundle root, ``manifest.json``
and ``memories.jsonl`` are examined with ``lstat`` first, so a FIFO or device
is rejected before any ``open`` could block on it, and each file is opened
with ``O_NOFOLLOW`` and ``O_NONBLOCK`` where the platform has them.

Each required file is read exactly once into memory from a single descriptor;
callers hash and parse those same bytes and never re-read the path.

Change detection covers *observed* changes only: the descriptor's identity,
size and ``mtime_ns`` before and after the read, a fresh ``lstat`` of the path
afterwards, the root's identity, and the absence of ``INCOMPLETE``. A writer
that rewrites bytes in place without changing size or ``mtime_ns`` between the
checks, or a change after the final check, is not detected. This is not a
universal race guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import os
import stat
import sys
from typing import TYPE_CHECKING, Final

from reflectlog.core.salvage import (
    INCOMPLETE_MARKER,
    MANIFEST_FILENAME,
    RECORDS_FILENAME,
    BundleCapture,
    BundleEntry,
    CapturedFile,
    VerificationIssue,
    VerificationIssueCode,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_O_BINARY: Final = os.O_BINARY if sys.platform == "win32" else 0
_O_NOFOLLOW: Final = 0 if sys.platform == "win32" else os.O_NOFOLLOW
_O_NONBLOCK: Final = 0 if sys.platform == "win32" else os.O_NONBLOCK
_OPEN_FLAGS: Final = os.O_RDONLY | _O_BINARY | _O_NOFOLLOW | _O_NONBLOCK
_CHUNK: Final = 1024 * 1024


@dataclass(frozen=True, slots=True)
class _Stamp:
    """Identity and mutable attributes of one file-system object."""

    kind: int
    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def of(cls, info: os.stat_result) -> _Stamp:
        return cls(
            stat.S_IFMT(info.st_mode),
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
        )

    def same_object(self, other: _Stamp) -> bool:
        """Same type and, where the platform reports one, same device/inode."""
        if self.kind != other.kind:
            return False
        if self.inode == 0 or other.inode == 0:
            return True
        return (self.device, self.inode) == (other.device, other.inode)


@dataclass(frozen=True, slots=True)
class _Read:
    """Bytes of one file plus the stamps taken around reading them."""

    data: bytes
    opened: _Stamp
    closed: _Stamp


def _strip_trailing_separators(path: str) -> str:
    """Drop trailing separators so a symlinked root cannot hide behind ``/``."""
    separators = tuple(s for s in (os.sep, os.altsep) if s)
    while len(path) > 1 and path.endswith(separators):
        path = path[:-1]
    return path


class FileSystemBundleReader:
    """``IBundleReader`` that reads a bundle directory without modifying it.

    Args:
        after_capture: Called once after every required file was read and
            before the change re-checks; a seam so tests can mutate the bundle
            at the exact point a concurrent writer would be observed.
    """

    def __init__(self, after_capture: Callable[[], None] | None = None) -> None:
        self._after_capture = after_capture

    def capture(self, bundle_dir: str) -> BundleCapture:
        root_path = _strip_trailing_separators(bundle_dir)
        issues: list[VerificationIssue] = []
        root = self._examine_root(root_path, issues)
        if root is None:
            return BundleCapture(None, None, tuple(issues))
        marker_path = os.path.join(root_path, INCOMPLETE_MARKER)
        marker_issue = self._marker_issue(marker_path)
        if marker_issue is not None:
            issues.append(VerificationIssue(marker_issue, BundleEntry.MARKER))
            return BundleCapture(None, None, tuple(issues))

        manifest = self._read(
            os.path.join(root_path, MANIFEST_FILENAME), BundleEntry.MANIFEST, issues
        )
        records = self._read(
            os.path.join(root_path, RECORDS_FILENAME), BundleEntry.RECORDS, issues
        )
        if self._after_capture is not None:
            self._after_capture()

        if self._root_changed(root_path, root, marker_path, issues):
            return BundleCapture(None, None, tuple(issues))
        return BundleCapture(
            self._confirm(
                manifest,
                os.path.join(root_path, MANIFEST_FILENAME),
                BundleEntry.MANIFEST,
                issues,
            ),
            self._confirm(
                records,
                os.path.join(root_path, RECORDS_FILENAME),
                BundleEntry.RECORDS,
                issues,
            ),
            tuple(issues),
        )

    def _examine_root(
        self, root_path: str, issues: list[VerificationIssue]
    ) -> _Stamp | None:
        try:
            info = os.lstat(root_path)
        except FileNotFoundError, NotADirectoryError:
            code = VerificationIssueCode.BUNDLE_MISSING
        except OSError:
            code = VerificationIssueCode.BUNDLE_UNREADABLE
        else:
            if stat.S_ISLNK(info.st_mode):
                code = VerificationIssueCode.BUNDLE_SYMLINK
            elif not stat.S_ISDIR(info.st_mode):
                code = VerificationIssueCode.BUNDLE_NOT_DIRECTORY
            else:
                return _Stamp.of(info)
        issues.append(VerificationIssue(code, BundleEntry.BUNDLE))
        return None

    def _marker_issue(self, marker_path: str) -> VerificationIssueCode | None:
        """Any entry named ``INCOMPLETE`` counts, even a dangling symlink."""
        try:
            _ = os.lstat(marker_path)
        except FileNotFoundError:
            return None
        except OSError:
            return VerificationIssueCode.FILE_UNREADABLE
        return VerificationIssueCode.INCOMPLETE_MARKER_PRESENT

    def _read(
        self, path: str, entry: BundleEntry, issues: list[VerificationIssue]
    ) -> _Read | None:
        outcome = _read_once(path)
        if isinstance(outcome, VerificationIssueCode):
            issues.append(VerificationIssue(outcome, entry))
            return None
        return outcome

    def _root_changed(
        self,
        root_path: str,
        root: _Stamp,
        marker_path: str,
        issues: list[VerificationIssue],
    ) -> bool:
        try:
            current = _Stamp.of(os.lstat(root_path))
        except OSError:
            current = None
        if current is None or not current.same_object(root):
            issues.append(
                VerificationIssue(
                    VerificationIssueCode.CHANGED_DURING_READ, BundleEntry.BUNDLE
                )
            )
            return True
        if self._marker_issue(marker_path) is not None:
            issues.append(
                VerificationIssue(
                    VerificationIssueCode.CHANGED_DURING_READ, BundleEntry.MARKER
                )
            )
            return True
        return False

    def _confirm(
        self,
        captured: _Read | None,
        path: str,
        entry: BundleEntry,
        issues: list[VerificationIssue],
    ) -> CapturedFile | None:
        """Keep the bytes only if the path still names the file that was read."""
        if captured is None:
            return None
        try:
            current = _Stamp.of(os.lstat(path))
        except OSError:
            current = None
        if (
            current is None
            or current != captured.closed
            or not current.same_object(captured.opened)
        ):
            issues.append(
                VerificationIssue(VerificationIssueCode.CHANGED_DURING_READ, entry)
            )
            return None
        return CapturedFile(captured.data)


def _read_once(path: str) -> _Read | VerificationIssueCode:
    """Read one regular file fully from one descriptor, or say why not."""
    try:
        listed = os.lstat(path)
    except FileNotFoundError:
        return VerificationIssueCode.FILE_MISSING
    except OSError:
        return VerificationIssueCode.FILE_UNREADABLE
    if stat.S_ISLNK(listed.st_mode):
        return VerificationIssueCode.FILE_SYMLINK
    if not stat.S_ISREG(listed.st_mode):
        return VerificationIssueCode.FILE_NOT_REGULAR
    try:
        descriptor = os.open(path, _OPEN_FLAGS)
    except FileNotFoundError:
        return VerificationIssueCode.CHANGED_DURING_READ
    except OSError as e:
        if e.errno == errno.ELOOP:
            return VerificationIssueCode.FILE_SYMLINK
        return VerificationIssueCode.FILE_UNREADABLE
    try:
        opened = _Stamp.of(os.fstat(descriptor))
        if opened.kind != stat.S_IFREG:
            return VerificationIssueCode.FILE_NOT_REGULAR
        if not opened.same_object(_Stamp.of(listed)):
            return VerificationIssueCode.CHANGED_DURING_READ
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _CHUNK):
            chunks.append(chunk)
        closed = _Stamp.of(os.fstat(descriptor))
    except OSError:
        return VerificationIssueCode.FILE_UNREADABLE
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if closed != opened or len(data) != closed.size:
        return VerificationIssueCode.CHANGED_DURING_READ
    return _Read(data, opened, closed)
