"""Crash-visible publication of an offline-salvage export bundle.

No rename-based publish: a check-then-rename can replace an empty directory
that a racing run just created. Instead the output directory is created
exclusively, an ``INCOMPLETE`` marker is written first and removed last, and
every file is created ``O_EXCL`` with restricted permissions. An export is
complete only when ``manifest.json`` exists and ``INCOMPLETE`` does not.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Final, Protocol

from reflectlog.core.exceptions import SalvagePublicationError, SalvageUnsafeOutputError
from reflectlog.core.salvage import (
    INCOMPLETE_MARKER,
    MANIFEST_FILENAME,
    RECORDS_FILENAME,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from reflectlog.core.salvage import SalvageBundle

_DIRECTORY_MODE: Final = 0o700
_FILE_MODE: Final = 0o600
_O_BINARY: Final = os.O_BINARY if sys.platform == "win32" else 0
_MARKER_TEXT: Final = b"reflectlog salvage export is incomplete\n"


class IPublicationFileSystem(Protocol):
    """Operating-system operations used by publication, injectable for faults."""

    def make_directory(self, path: str) -> None: ...

    def create_file(self, path: str) -> int: ...

    def write_all(self, descriptor: int, payload: bytes) -> None: ...

    def sync_file(self, descriptor: int) -> None: ...

    def close_file(self, descriptor: int) -> None: ...

    def sync_directory(self, path: str) -> None: ...

    def remove_file(self, path: str) -> None: ...

    def remove_directory(self, path: str) -> None: ...


class OsPublicationFileSystem:
    """``IPublicationFileSystem`` backed by ``os``.

    Args:
        fsync: File-descriptor sync; replaceable so tests can fault durability.
    """

    def __init__(self, fsync: Callable[[int], None] = os.fsync) -> None:
        self._fsync = fsync

    def make_directory(self, path: str) -> None:
        os.mkdir(path, _DIRECTORY_MODE)

    def create_file(self, path: str) -> int:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | _O_BINARY
        return os.open(path, flags, _FILE_MODE)

    def write_all(self, descriptor: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]

    def sync_file(self, descriptor: int) -> None:
        self._fsync(descriptor)

    def close_file(self, descriptor: int) -> None:
        os.close(descriptor)

    def sync_directory(self, path: str) -> None:
        # Same precedent as storage_coordinator._fsync_directory: platforms that
        # cannot open a directory (Windows) skip it; a failed fsync is an error.
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            self._fsync(descriptor)
        finally:
            os.close(descriptor)

    def remove_file(self, path: str) -> None:
        os.unlink(path)

    def remove_directory(self, path: str) -> None:
        os.rmdir(path)


class FileSystemBundlePublisher:
    """``IBundlePublisher`` that never overwrites and never looks complete early."""

    def __init__(self, filesystem: IPublicationFileSystem | None = None) -> None:
        self._fs: IPublicationFileSystem = (
            filesystem if filesystem is not None else OsPublicationFileSystem()
        )

    def resolve_output(self, output: str, protected_dirs: Sequence[str]) -> str:
        absolute = os.path.abspath(output)
        parent, name = os.path.split(absolute)
        if not name:
            raise SalvageUnsafeOutputError("output path has no final component")
        if os.path.lexists(absolute):
            raise SalvageUnsafeOutputError("output path already exists")
        if not os.path.isdir(parent):
            raise SalvageUnsafeOutputError("output parent directory does not exist")
        real_parent = os.path.realpath(parent)
        for protected in protected_dirs:
            if _is_within(real_parent, protected):
                raise SalvageUnsafeOutputError(
                    "output path is inside the archive or the scratch directory"
                )
        return os.path.join(real_parent, name)

    def publish(self, output: str, bundle: SalvageBundle) -> None:
        try:
            self._fs.make_directory(output)
        except FileExistsError as e:
            raise SalvageUnsafeOutputError("output path already exists") from e
        except OSError as e:
            raise SalvagePublicationError(
                f"cannot create the output directory: {e.strerror}"
            ) from e
        marker = os.path.join(output, INCOMPLETE_MARKER)
        created: list[str] = []
        try:
            self._write(marker, _MARKER_TEXT, created)
            self._write(os.path.join(output, RECORDS_FILENAME), bundle.records, created)
            self._write(
                os.path.join(output, MANIFEST_FILENAME), bundle.manifest, created
            )
            self._fs.sync_directory(output)
            self._fs.remove_file(marker)
            created.remove(marker)
            self._fs.sync_directory(output)
        except OSError as e:
            self._abandon(output, created)
            raise SalvagePublicationError(
                f"publication failed and was rolled back: {e.strerror or e}"
            ) from e
        except BaseException:
            self._abandon(output, created)
            raise

    def _write(self, path: str, payload: bytes, created: list[str]) -> None:
        descriptor = self._fs.create_file(path)
        created.append(path)
        try:
            self._fs.write_all(descriptor, payload)
            self._fs.sync_file(descriptor)
        finally:
            self._fs.close_file(descriptor)

    def _abandon(self, output: str, created: list[str]) -> None:
        """Remove only what this run created; the marker goes last.

        Stops at the first failure so ``INCOMPLETE`` outlives every file that
        could not be removed.
        """
        for path in reversed(created):
            try:
                self._fs.remove_file(path)
            except FileNotFoundError:
                continue
            except OSError:
                return
        try:
            self._fs.remove_directory(output)
        except OSError:
            return


def _is_within(directory: str, protected: str) -> bool:
    """Return True if ``directory`` is ``protected`` or lies below it."""
    current = directory
    while True:
        try:
            if os.path.samefile(current, protected):
                return True
        except OSError:
            pass
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent
