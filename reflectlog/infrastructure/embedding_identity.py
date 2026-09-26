from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter, ValidationError

from reflectlog.core.exceptions import InitializationError
from reflectlog.core.storage_coordination import LeaseMode, WorkspaceStoragePaths
from reflectlog.infrastructure.storage_coordinator import (
    GENERATION_NAME,
    LOCK_NAME,
    PortalockerStorageCoordinator,
)
from reflectlog.utility.security import validate_workspace_id

if TYPE_CHECKING:
    from reflectlog.core.storage_coordination import IStorageCoordinator
    from reflectlog.infrastructure.usearch_engine import USearchConfig

IDENTITY_NAME = ".reflectlog.embedding-identity.json"
_RECOVERY: Final = (
    "Preserve and export existing workspace data, then rebuild the workspace "
    "offline with the intended embedding provider, model, and dimensions."
)
_TABLES = frozenset(
    {"memories", "archived_memories", "replacement_transitions", "sqlite_sequence"}
)
_INDEXES = frozenset(
    {
        "idx_workspace_id",
        "idx_dedup",
        "idx_archived_workspace_id",
        "idx_archived_at",
        "idx_archived_original_replaced",
        "idx_transition_old_replace",
        "idx_transition_old_delete",
        "idx_pending_add",
        "idx_transition_pending",
        "idx_transition_identity",
        "idx_transition_old_memory",
    }
)
_COLUMNS = {
    "memories": {"id", "workspace_id", "content", "created_at"},
    "archived_memories": {
        "id",
        "original_id",
        "workspace_id",
        "content",
        "replaced_by",
        "reason",
        "confidence",
        "archived_at",
    },
    "replacement_transitions": {
        "id",
        "workspace_id",
        "old_memory_id",
        "old_content",
        "new_content",
        "archive_id",
        "reason",
        "confidence",
        "status",
        "created_at",
        "updated_at",
        "kind",
    },
    "sqlite_sequence": {"name", "seq"},
}


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    version: int
    workspace_id: str
    provider: str
    model: str
    dimensions: int

    @classmethod
    def from_config(cls, config: USearchConfig) -> EmbeddingIdentity:
        workspace_id = validate_workspace_id(config.workspace_id).lower()
        if not config.embedding_model or config.embedding_dims <= 0:
            raise InitializationError(
                "Embedding model and dimensions must be specified"
            )
        return cls(
            1,
            workspace_id,
            config.embedder_provider.value,
            config.embedding_model,
            config.embedding_dims,
        )


class _DirectCoordinator(PortalockerStorageCoordinator):
    def __init__(self, root: Path, workspace_id: str) -> None:
        super().__init__(str(root.parent))
        self._root = root
        self._workspace_id = workspace_id

    def paths_for(self, workspace_id: str) -> WorkspaceStoragePaths:
        if validate_workspace_id(workspace_id).lower() != self._workspace_id:
            raise InitializationError(
                "Coordinator workspace does not match embedding identity"
            )
        root = str(self._root)
        return WorkspaceStoragePaths(
            workspace_id=self._workspace_id,
            root=root,
            lock_path=os.path.join(root, LOCK_NAME),
            generation_path=os.path.join(root, GENERATION_NAME),
        )


def _workspace_root(config: USearchConfig) -> Path:
    index_dir = Path(os.path.abspath(config.index_path)).parent
    db_dir = Path(os.path.abspath(config.db_path)).parent
    if index_dir != db_dir:
        raise InitializationError("Index and SQLite must share a workspace directory")
    return index_dir.parent if index_dir.name == "usearch" else index_dir


def _coordinator_for(
    config: USearchConfig, coordinator: IStorageCoordinator | None
) -> tuple[Path, IStorageCoordinator]:
    root = _workspace_root(config)
    if (
        root.is_symlink()
        or Path(config.db_path).parent.is_symlink()
        or Path(config.index_path).parent.is_symlink()
    ):
        raise InitializationError("Embedding workspace paths must not be symlinks")
    identity = EmbeddingIdentity.from_config(config)
    active = (
        coordinator
        if coordinator is not None
        else _DirectCoordinator(root, identity.workspace_id)
    )
    paths = active.paths_for(identity.workspace_id)
    if (
        Path(os.path.abspath(paths.root)) != root
        or Path(os.path.abspath(paths.lock_path)) != root / LOCK_NAME
        or Path(os.path.abspath(paths.generation_path)) != root / GENERATION_NAME
    ):
        raise InitializationError(
            "Storage coordinator root does not match embedding workspace"
        )
    return root, active


def _read_identity(path: Path, expected: EmbeddingIdentity) -> bool:
    if path.is_symlink():
        raise InitializationError(
            f"Embedding identity sidecar is a symlink. {_RECOVERY}"
        )
    try:
        with path.open(encoding="utf-8") as handle:
            stored = TypeAdapter(dict[str, str | int]).validate_json(handle.read())
    except FileNotFoundError:
        return False
    except (OSError, ValueError, UnicodeError, ValidationError) as exc:
        raise InitializationError(
            f"Embedding identity sidecar is unreadable. {_RECOVERY}"
        ) from exc
    if (
        set(stored) != {"version", "workspace_id", "provider", "model", "dimensions"}
        or type(stored["version"]) is not int
        or type(stored["dimensions"]) is not int
        or type(stored["workspace_id"]) is not str
        or type(stored["provider"]) is not str
        or type(stored["model"]) is not str
    ):
        raise InitializationError(
            f"Embedding identity sidecar has an unknown format. {_RECOVERY}"
        )
    if stored != {
        "version": expected.version,
        "workspace_id": expected.workspace_id,
        "provider": expected.provider,
        "model": expected.model,
        "dimensions": expected.dimensions,
    }:
        raise InitializationError(
            f"Embedding identity does not match persisted workspace. {_RECOVERY}"
        )
    return True


def _sqlite_is_empty(path: Path) -> bool:
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                return False
            schema = connection.execute(
                "SELECT type, name FROM sqlite_master"
            ).fetchall()
            for kind, name in schema:
                if kind == "table":
                    if name not in _TABLES:
                        return False
                    columns = {
                        row[1]
                        for row in connection.execute(f'PRAGMA table_info("{name}")')
                    }
                    if name == "replacement_transitions":
                        if columns not in (_COLUMNS[name], _COLUMNS[name] - {"kind"}):
                            return False
                    elif columns != _COLUMNS[name]:
                        return False
                    if connection.execute(f'SELECT 1 FROM "{name}" LIMIT 1').fetchone():
                        return False
                elif kind != "index" or name not in _INDEXES:
                    return False
            return True
        finally:
            connection.close()
    except sqlite3.Error, OSError:
        return False


def _legacy_is_empty(root: Path, config: USearchConfig) -> bool:
    db = Path(os.path.abspath(config.db_path))
    index = Path(os.path.abspath(config.index_path))
    if root.is_symlink() or db.parent.is_symlink() or index.parent.is_symlink():
        return False
    try:
        if not root.exists():
            return True

        def fail_walk(error: OSError) -> None:
            raise error

        for directory, dirs, files in os.walk(
            root, followlinks=False, onerror=fail_walk
        ):
            for name in dirs:
                path = Path(directory) / name
                if path.is_symlink() or path != index.parent or path.parent != root:
                    return False
            for name in files:
                path = Path(directory) / name
                if path.is_symlink() or not path.is_file():
                    return False
                if path == db:
                    if not _sqlite_is_empty(path):
                        return False
                elif path == root / LOCK_NAME:
                    continue
                elif path == root / GENERATION_NAME:
                    if path.read_text(encoding="utf-8").strip() != "0":
                        return False
                else:
                    return False
        return True
    except OSError, UnicodeError:
        return False


def _external_tantivy_is_empty(root: Path, index_path: str) -> bool:
    path = Path(os.path.abspath(index_path))
    if path.is_relative_to(root):
        return True
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            return False
        with os.scandir(path) as entries:
            return next(entries, None) is None
    except FileNotFoundError:
        return True
    except OSError:
        return False


def preflight_embedding_identity(
    config: USearchConfig,
    coordinator: IStorageCoordinator | None = None,
    *,
    tantivy_index_path: str | None = None,
) -> bool:
    """Check persisted identity without creating storage; return True if published."""
    root, _ = _coordinator_for(config, coordinator)
    identity = EmbeddingIdentity.from_config(config)
    if _read_identity(root / IDENTITY_NAME, identity):
        return True
    if not _legacy_is_empty(root, config) or (
        tantivy_index_path is not None
        and not _external_tantivy_is_empty(root, tantivy_index_path)
    ):
        raise InitializationError(
            f"Legacy workspace has storage without embedding identity. {_RECOVERY}"
        )
    return False


def ensure_embedding_identity(
    config: USearchConfig,
    coordinator: IStorageCoordinator | None = None,
    *,
    tantivy_index_path: str | None = None,
) -> IStorageCoordinator:
    """Preflight and publish identity under an exclusive workspace lease."""
    root, active = _coordinator_for(config, coordinator)
    try:
        _ = preflight_embedding_identity(
            config, active, tantivy_index_path=tantivy_index_path
        )
    except InitializationError:
        if not any(root.glob(f"{IDENTITY_NAME}.*.tmp")):
            _ = preflight_embedding_identity(
                config, active, tantivy_index_path=tantivy_index_path
            )
    workspace_id = validate_workspace_id(config.workspace_id).lower()
    if active.is_held(workspace_id) and not active.is_held(
        workspace_id, LeaseMode.EXCLUSIVE
    ):
        raise InitializationError("Cannot open embedding identity under a shared lease")

    def publish() -> None:
        if not preflight_embedding_identity(
            config, active, tantivy_index_path=tantivy_index_path
        ):
            identity = EmbeddingIdentity.from_config(config)
            path = root / IDENTITY_NAME
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f"{IDENTITY_NAME}.", suffix=".tmp", dir=root
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "version": identity.version,
                            "workspace_id": identity.workspace_id,
                            "provider": identity.provider,
                            "model": identity.model,
                            "dimensions": identity.dimensions,
                        },
                        handle,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
        if os.name != "nt":
            directory = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    if active.is_held(workspace_id, LeaseMode.EXCLUSIVE):
        publish()
    else:
        with active.acquire(workspace_id, LeaseMode.EXCLUSIVE):
            publish()
    return active
