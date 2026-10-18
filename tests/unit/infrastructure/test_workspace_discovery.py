import os
from pathlib import Path
from unittest.mock import patch

import pytest

from reflectlog.infrastructure.workspace_discovery import list_stored_workspaces


def test_discovers_only_canonical_stored_direct_children(tmp_path: Path) -> None:
    for name in ("zebra", "alpha-2", "_internal", "0", "a.b"):
        usearch = tmp_path / name / "usearch"
        usearch.mkdir(parents=True)
        (usearch / "memories.db").touch()
    for name in ("UPPER", " padded ", "a..b", "with space", "caf\u00e9", "x" * 65):
        usearch = tmp_path / name / "usearch"
        usearch.mkdir(parents=True)
        (usearch / "memories.db").touch()
    (tmp_path / "empty").mkdir()
    (tmp_path / "locked").mkdir()
    (tmp_path / "locked" / ".lock").touch()
    (tmp_path / "file").touch()
    (tmp_path / "vectors" / "usearch").mkdir(parents=True)
    (tmp_path / "vectors" / "usearch" / "vectors.usearch").touch()
    (tmp_path / "folder-marker" / "usearch" / "memories.db").mkdir(parents=True)
    (tmp_path / "file-usearch").mkdir()
    (tmp_path / "file-usearch" / "usearch").touch()
    (tmp_path / "nested" / "child" / "usearch").mkdir(parents=True)
    (tmp_path / "nested" / "child" / "usearch" / "memories.db").touch()

    result = list_stored_workspaces(str(tmp_path))

    assert result == ["0", "_internal", "a.b", "alpha-2", "zebra"]


@pytest.mark.parametrize("exists", [False, True])
def test_empty_or_missing_root_returns_empty_without_writes(
    tmp_path: Path, exists: bool
) -> None:
    root = tmp_path / "indexes"
    if exists:
        root.mkdir()

    result = list_stored_workspaces(str(root))

    assert result == []
    assert root.exists() is exists
    assert list(tmp_path.rglob("*")) == ([root] if exists else [])


def test_symlinked_workspaces_and_markers_are_excluded(tmp_path: Path) -> None:
    root = tmp_path / "indexes"
    root.mkdir()
    outside = tmp_path / "outside" / "usearch"
    outside.mkdir(parents=True)
    marker = outside / "memories.db"
    marker.touch()
    (root / "workspace-link").symlink_to(outside.parent, target_is_directory=True)
    (root / "dangling").symlink_to(tmp_path / "absent", target_is_directory=True)
    (root / "engine-link").mkdir()
    (root / "engine-link" / "usearch").symlink_to(outside, target_is_directory=True)
    (root / "marker-link" / "usearch").mkdir(parents=True)
    (root / "marker-link" / "usearch" / "memories.db").symlink_to(marker)

    result = list_stored_workspaces(str(root))

    assert result == []


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling-symlink"])
def test_invalid_storage_root_is_an_error(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "indexes"
    if kind == "file":
        root.touch()
    else:
        target = tmp_path / "target"
        if kind == "symlink":
            target.mkdir()
        root.symlink_to(target, target_is_directory=True)

    with pytest.raises(NotADirectoryError):
        list_stored_workspaces(str(root))


def test_scan_io_error_is_not_an_empty_listing(tmp_path: Path) -> None:
    with patch(
        "reflectlog.infrastructure.workspace_discovery.os.scandir",
        side_effect=PermissionError("scan denied"),
    ):
        with pytest.raises(PermissionError, match="scan denied"):
            list_stored_workspaces(str(tmp_path))


def test_marker_io_error_is_not_silently_skipped(tmp_path: Path) -> None:
    usearch = tmp_path / "stored" / "usearch"
    usearch.mkdir(parents=True)
    marker = usearch / "memories.db"
    marker.touch()
    original_stat = Path.stat

    def stat_path(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == marker:
            raise OSError("marker unavailable")
        return original_stat(path, follow_symlinks=follow_symlinks)

    with patch.object(Path, "stat", stat_path):
        with pytest.raises(OSError, match="marker unavailable"):
            list_stored_workspaces(str(tmp_path))
