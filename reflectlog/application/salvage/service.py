"""Application service that salvages readable memory content from an archive.

Orchestrates through core protocols only: it never constructs
``MemoryManager``, a workspace registry, engines, embeddings, or models, and
loads no configuration. Infrastructure implementations are wired by
``from_defaults``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reflectlog.application.salvage.manifest import build_bundle, unresolved_reasons
from reflectlog.core.exceptions import SalvageInputError, ValidationError
from reflectlog.core.salvage import SalvageBundle
from reflectlog.utility.security import validate_workspace_id
from reflectlog.version import __version__

if TYPE_CHECKING:
    from typing import Self

    from reflectlog.core.salvage import (
        IArchiveInspector,
        IBundlePublisher,
        ISalvageSnapshotReader,
    )


@dataclass(frozen=True, slots=True)
class SalvageResult:
    """Summary of one successful export."""

    output: str
    record_count: int
    rebuild_ready: bool
    unresolved_reasons: tuple[str, ...]


class ContentSalvageService:
    """Export a workspace's active content from a quiescent archive copy."""

    def __init__(
        self,
        inspector: IArchiveInspector,
        reader: ISalvageSnapshotReader,
        publisher: IBundlePublisher,
        *,
        tool_version: str = __version__,
    ) -> None:
        self._inspector = inspector
        self._reader = reader
        self._publisher = publisher
        self._tool_version = tool_version

    @classmethod
    def from_defaults(cls) -> Self:
        """Wire the filesystem, SQLite, and publication implementations."""
        from reflectlog.infrastructure.salvage_archive import ArchiveInspector
        from reflectlog.infrastructure.salvage_publication import (
            FileSystemBundlePublisher,
        )
        from reflectlog.infrastructure.salvage_snapshot import (
            SqliteSalvageSnapshotReader,
        )

        return cls(
            ArchiveInspector(),
            SqliteSalvageSnapshotReader(),
            FileSystemBundlePublisher(),
        )

    def salvage(self, *, archive: str, workspace_id: str, output: str) -> SalvageResult:
        """Export ``workspace_id`` from ``archive`` into the new ``output`` dir.

        Raises:
            SalvageError: On any input, schema, record, archive-change,
                output-safety, or publication failure.
        """
        try:
            requested = validate_workspace_id(workspace_id)
        except ValidationError as e:
            raise SalvageInputError(f"invalid workspace id: {e}") from e
        # Fail fast on an unsafe output before anything is copied.
        _ = self._publisher.resolve_output(output, (archive,))
        with self._inspector.inspect(archive) as inspection:
            target = self._publisher.resolve_output(
                output, (archive, inspection.scratch_dir)
            )
            with self._reader.read_snapshot(
                inspection.database_path, requested
            ) as snapshot:
                inspection.verify_unchanged()
                bundle = build_bundle(
                    tool_version=self._tool_version,
                    snapshot=snapshot,
                    fingerprints=inspection.fingerprints,
                    identity=inspection.identity,
                    generation=inspection.generation,
                )

                def finalize_manifest() -> bytes:
                    inspection.verify_unchanged()
                    return bundle.finalize_manifest()

                self._publisher.publish(
                    target, SalvageBundle(bundle.records, finalize_manifest)
                )
        reasons = unresolved_reasons(snapshot)
        return SalvageResult(
            output=target,
            record_count=snapshot.record_count,
            rebuild_ready=not reasons,
            unresolved_reasons=reasons,
        )
