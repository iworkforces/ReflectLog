"""Read-only discovery of canonical workspace directories with stored identities."""

import errno
import os
from pathlib import Path
import stat

from reflectlog.core.exceptions import ValidationError
from reflectlog.utility.security import validate_workspace_id


def list_stored_workspaces(storage_path: str) -> list[str]:
    """List stored workspace ids without opening databases, engines, or models.

    A stored workspace has a regular usearch/memories.db marker. Symlinks at
    the root, workspace, usearch directory, or marker are never followed.
    """
    root = Path(storage_path)
    try:
        root_mode = root.stat(follow_symlinks=False).st_mode
    except FileNotFoundError:
        return []
    if not stat.S_ISDIR(root_mode):
        raise NotADirectoryError(errno.ENOTDIR, "Invalid workspace storage root", root)

    workspaces: list[str] = []
    with os.scandir(root) as entries:
        for entry in entries:
            try:
                canonical = validate_workspace_id(entry.name)
            except ValidationError:
                continue
            if canonical != entry.name or not entry.is_dir(follow_symlinks=False):
                continue
            usearch = Path(entry.path) / "usearch"
            try:
                if not stat.S_ISDIR(usearch.stat(follow_symlinks=False).st_mode):
                    continue
                marker_mode = (
                    (usearch / "memories.db").stat(follow_symlinks=False).st_mode
                )
            except FileNotFoundError:
                continue
            if stat.S_ISREG(marker_mode):
                workspaces.append(canonical)
    return sorted(workspaces)
