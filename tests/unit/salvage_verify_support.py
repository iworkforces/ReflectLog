"""Helpers that build and deliberately corrupt exported salvage bundles.

Bundles come from the real exporter so verifier tests follow the shipped
format. Corruption helpers rewrite one file at a time and keep the manifest's
digest and count in step unless the caller asks otherwise, so a test can
isolate a single defect.
"""

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

from reflectlog.application.salvage.service import ContentSalvageService
from tests.unit.salvage_support import (
    WS,
    JournalRow,
    Row,
    as_object,
    make_archive,
)

DEFAULT_MEMORIES: Sequence[Row] = (
    (WS, "first memory", "2024-01-01 00:00:00"),
    (WS, "second memory", None),
)


def export_bundle(
    base: Path,
    *,
    memories: Sequence[Row] = DEFAULT_MEMORIES,
    journal: Sequence[JournalRow] = (),
    kind: bool = True,
    journal_table: bool = True,
    workspace: str = WS,
    name: str = "bundle",
) -> Path:
    """Export a real bundle from a fabricated archive; return the bundle dir."""
    archive = make_archive(
        base / f"{name}-archive",
        memories=memories,
        journal=journal,
        kind=kind,
        journal_table=journal_table,
    )
    output = base / name
    _ = ContentSalvageService.from_defaults().salvage(
        archive=str(archive), workspace_id=workspace, output=str(output)
    )
    return output


def load_manifest(bundle: Path) -> dict[str, object]:
    return as_object(json.loads((bundle / "manifest.json").read_text(encoding="utf-8")))


def store_manifest(bundle: Path, manifest: Mapping[str, object]) -> None:
    """Write ``manifest`` in the exporter's own layout."""
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    (bundle / "manifest.json").write_text(text + "\n", encoding="utf-8")


def record_line(source_id: object, content: object, created: object = None) -> bytes:
    """One record line exactly as the exporter renders it."""
    return (
        json.dumps(
            {
                "content": content,
                "source_created_at": created,
                "source_id": source_id,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def line_count(data: bytes) -> int:
    """Number of lines the verifier counts, counting an unterminated tail."""
    return data.count(b"\n") + (0 if not data or data.endswith(b"\n") else 1)


def resync_manifest(bundle: Path) -> None:
    """Point the manifest's digest and count at the bytes now on disk."""
    data = (bundle / "memories.jsonl").read_bytes()
    manifest = load_manifest(bundle)
    records = as_object(manifest["records"])
    records["sha256"] = hashlib.sha256(data).hexdigest()
    records["count"] = line_count(data)
    store_manifest(bundle, manifest)


def write_records(bundle: Path, data: bytes, *, resync: bool = True) -> None:
    """Replace ``memories.jsonl``; keep digest and count consistent by default."""
    (bundle / "memories.jsonl").write_bytes(data)
    if resync:
        resync_manifest(bundle)
