"""Archive isolation for offline content salvage.

Inspects one archived workspace directory without ever opening an archive file
for writing or creating a file inside it: SQLite files are fingerprinted,
copied into a private scratch directory, and the copy is verified. The
original is re-fingerprinted afterwards so a change during salvage is caught.
The identity and generation sidecars are read raw and never validated, and a
missing generation is reported as missing, never as zero.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import os
import shutil
import stat
import tempfile
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from reflectlog.core.exceptions import (
    SalvageArchiveChangedError,
    SalvageInputError,
)
from reflectlog.core.salvage import (
    GenerationSidecarReport,
    IdentitySidecarReport,
    SidecarScalar,
    SidecarStatus,
    SourceFingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

# Mirrors infrastructure.embedding_identity / storage_coordinator names without
# importing them: those modules pull in locking and configuration machinery.
DATABASE_PATH: Final = "usearch/memories.db"
WAL_PATH: Final = "usearch/memories.db-wal"
SHM_PATH: Final = "usearch/memories.db-shm"
IDENTITY_PATH: Final = ".reflectlog.embedding-identity.json"
GENERATION_PATH: Final = ".reflectlog.storage-generation"

_INSPECTED: Final = (DATABASE_PATH, WAL_PATH, SHM_PATH, IDENTITY_PATH, GENERATION_PATH)
_SIDECARS: Final = (IDENTITY_PATH, GENERATION_PATH)
_IDENTITY_FIELDS: Final = ("dimensions", "model", "provider", "version", "workspace_id")
_SIDECAR_LIMIT: Final = 64 * 1024
_CHUNK: Final = 1024 * 1024
_SCRATCH_DATABASE: Final = "memories.db"
_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class _Observation:
    """Fingerprints and raw sidecars from one pass over the archive."""

    fingerprints: tuple[SourceFingerprint, ...]
    identity: IdentitySidecarReport
    generation: GenerationSidecarReport


class ArchiveInspection:
    """Disposable copy of an archive plus the observations made while copying."""

    __slots__ = (
        "_observation",
        "_verify",
        "database_path",
        "scratch_dir",
    )

    def __init__(
        self,
        scratch_dir: str,
        database_path: str,
        observation: _Observation,
        verify: Callable[[_Observation], None],
    ) -> None:
        self.scratch_dir = scratch_dir
        self.database_path = database_path
        self._observation = observation
        self._verify = verify

    @property
    def fingerprints(self) -> tuple[SourceFingerprint, ...]:
        return self._observation.fingerprints

    @property
    def identity(self) -> IdentitySidecarReport:
        return self._observation.identity

    @property
    def generation(self) -> GenerationSidecarReport:
        return self._observation.generation

    def verify_unchanged(self) -> None:
        self._verify(self._observation)


class ArchiveInspector:
    """``IArchiveInspector`` that copies an archive's SQLite files to scratch.

    Args:
        copy_file: Copies ``(source, destination)``; replaceable so tests can
            fault the copy step. Defaults to ``shutil.copyfile``.
    """

    def __init__(self, copy_file: Callable[[str, str], object] | None = None) -> None:
        self._copy_file: Callable[[str, str], object] = (
            copy_file if copy_file is not None else shutil.copyfile
        )

    @contextmanager
    def inspect(self, archive_dir: str) -> Generator[ArchiveInspection]:
        archive = os.path.abspath(archive_dir)
        observation = _observe(archive)
        scratch = tempfile.mkdtemp(prefix="reflectlog-salvage-")
        try:
            database_copy = os.path.join(scratch, _SCRATCH_DATABASE)
            self._copy_into_scratch(archive, database_copy, observation)

            def verify(first: _Observation) -> None:
                if _observe(archive) != first:
                    raise SalvageArchiveChangedError(
                        "archive changed while it was being salvaged"
                    )

            yield ArchiveInspection(scratch, database_copy, observation, verify)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _copy_into_scratch(
        self, archive: str, database_copy: str, observation: _Observation
    ) -> None:
        by_path = {item.path: item for item in observation.fingerprints}
        wal_copy = f"{database_copy}-wal"
        try:
            _ = self._copy_file(_absolute(archive, DATABASE_PATH), database_copy)
            if WAL_PATH in by_path:
                _ = self._copy_file(_absolute(archive, WAL_PATH), wal_copy)
            else:
                # A WAL-format database without a -wal file cannot be opened
                # mode=ro (SQLite may not create the WAL), so scratch gets an
                # empty one. The archive itself is never touched. The -shm file
                # is not copied: SQLite rebuilds it from the WAL.
                with open(wal_copy, "xb"):
                    pass
        except OSError as e:
            raise SalvageInputError(f"cannot copy archive files: {e.strerror}") from e
        for name, target in ((DATABASE_PATH, database_copy), (WAL_PATH, wal_copy)):
            expected = by_path.get(name)
            if expected is None:
                continue
            if _hash_file(target)[:2] != (expected.size, expected.sha256):
                raise SalvageArchiveChangedError(
                    f"copy of {name} does not match the archive; "
                    "the archive may have changed during the copy"
                )


def _absolute(archive: str, relative: str) -> str:
    return os.path.join(archive, *relative.split("/"))


def _hash_file(path: str, keep: int = 0) -> tuple[int, str, bytes | None]:
    """Return ``(size, sha256, content)``; content only when size <= ``keep``."""
    digest = hashlib.sha256()
    size = 0
    kept = bytearray()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
            if size <= keep:
                kept.extend(chunk)
    return size, digest.hexdigest(), bytes(kept) if size <= keep else None


def _file_state(archive: str, relative: str) -> str | None:
    """Return the path if it is a regular file, None if absent; reject others."""
    path = _absolute(archive, relative)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise SalvageInputError(f"cannot inspect {relative}: {e.strerror}") from e
    if stat.S_ISLNK(info.st_mode):
        raise SalvageInputError(f"{relative} is a symlink; refusing to follow it")
    if not stat.S_ISREG(info.st_mode):
        raise SalvageInputError(f"{relative} is not a regular file")
    return path


def _observe(archive: str) -> _Observation:
    try:
        root = os.lstat(archive)
    except FileNotFoundError as e:
        raise SalvageInputError("archive directory does not exist") from e
    except OSError as e:
        raise SalvageInputError(f"cannot inspect archive: {e.strerror}") from e
    if stat.S_ISLNK(root.st_mode):
        raise SalvageInputError("archive directory is a symlink")
    if not stat.S_ISDIR(root.st_mode):
        raise SalvageInputError("archive path is not a directory")
    try:
        subdirectory = os.lstat(os.path.join(archive, "usearch"))
    except FileNotFoundError:
        subdirectory = None
    if subdirectory is not None and stat.S_ISLNK(subdirectory.st_mode):
        raise SalvageInputError("usearch is a symlink; refusing to follow it")

    fingerprints: list[SourceFingerprint] = []
    sidecars: dict[str, tuple[SidecarStatus, bytes | None]] = {}
    for relative in sorted(_INSPECTED):
        path = _file_state(archive, relative)
        is_sidecar = relative in _SIDECARS
        if path is None:
            if relative == DATABASE_PATH:
                raise SalvageInputError(f"SQLite database is missing: {relative}")
            sidecars[relative] = (SidecarStatus.MISSING, None)
            continue
        try:
            size, sha256, content = _hash_file(
                path, _SIDECAR_LIMIT if is_sidecar else 0
            )
        except OSError as e:
            if not is_sidecar:
                raise SalvageInputError(f"cannot read {relative}: {e.strerror}") from e
            sidecars[relative] = (SidecarStatus.UNREADABLE, None)
            continue
        fingerprints.append(SourceFingerprint(relative, size, sha256))
        # Oversized sidecars keep their fingerprint but are reported malformed.
        state = (
            SidecarStatus.PRESENT if content is not None else SidecarStatus.MALFORMED
        )
        sidecars[relative] = (state, content)
    return _Observation(
        tuple(fingerprints),
        _identity_report(*sidecars[IDENTITY_PATH]),
        _generation_report(*sidecars[GENERATION_PATH]),
    )


def _identity_report(
    status: SidecarStatus, data: bytes | None
) -> IdentitySidecarReport:
    if status is not SidecarStatus.PRESENT or data is None:
        return IdentitySidecarReport(status)
    try:
        parsed = _OBJECT_ADAPTER.validate_json(data)
    except ValueError:
        return IdentitySidecarReport(SidecarStatus.MALFORMED)
    fields: list[tuple[str, SidecarScalar]] = []
    for name in _IDENTITY_FIELDS:
        if name not in parsed:
            continue
        value = parsed[name]
        if (
            value is None
            or isinstance(value, str | int | bool)
            or (isinstance(value, float) and math.isfinite(value))
        ):
            fields.append((name, value))
        else:
            return IdentitySidecarReport(SidecarStatus.MALFORMED)
    return IdentitySidecarReport(SidecarStatus.PRESENT, tuple(fields))


def _generation_report(
    status: SidecarStatus, data: bytes | None
) -> GenerationSidecarReport:
    if status is not SidecarStatus.PRESENT or data is None:
        return GenerationSidecarReport(status)
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return GenerationSidecarReport(SidecarStatus.MALFORMED)
    if text.isascii() and text.isdigit():
        return GenerationSidecarReport(SidecarStatus.PRESENT, text)
    return GenerationSidecarReport(SidecarStatus.MALFORMED, text)
