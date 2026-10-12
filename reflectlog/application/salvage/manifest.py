"""Deterministic rendering of the salvage records file and manifest.

Output depends only on the archive bytes, the requested workspace, and the
ReflectLog version: no wall-clock time, absolute paths, or scratch names.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Final

from reflectlog.core.salvage import (
    RECORDS_FILENAME,
    JournalStatus,
    SalvageBundle,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reflectlog.core.salvage import (
        ActiveRecord,
        GenerationSidecarReport,
        IdentitySidecarReport,
        RawJournalRow,
        RawValue,
        SalvageSnapshot,
        SourceFingerprint,
    )

FORMAT_NAME: Final = "reflectlog.content-salvage"
FORMAT_VERSION: Final = 1
NOT_PRESERVED: Final = (
    "embedding identity: provider, model, and dimensions are recorded as observed "
    "but the rebuilt workspace gets its own",
    "memory ids: rebuilt memories receive new ids",
    "created_at timestamps: rebuilt memories are stamped when re-added",
    "replacement and archive history: archived memories and completed journal rows "
    "are not exported",
    "vectors: embeddings and the full-text index are rebuilt from content",
)


def render_records(records: Sequence[ActiveRecord]) -> bytes:
    """Render one JSON line per record, sorted by source id."""
    lines = [
        json.dumps(
            {
                "content": record.content,
                "source_created_at": record.created_at,
                "source_id": record.source_id,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in sorted(records, key=lambda item: item.source_id)
    ]
    return "".join(lines).encode("utf-8")


def unresolved_reasons(snapshot: SalvageSnapshot) -> tuple[str, ...]:
    """Return sorted stable codes for everything salvage could not interpret."""
    reasons: set[str] = set()
    if snapshot.pending:
        reasons.add("journal_pending_intent")
    if snapshot.unknown:
        reasons.add("journal_unknown_evidence")
    if snapshot.foreign:
        reasons.add("journal_foreign_workspace_evidence")
    if snapshot.schema.journal_status is JournalStatus.MISSING:
        reasons.add("journal_missing")
    if snapshot.schema.journal_status is JournalStatus.UNSUPPORTED:
        reasons.add("journal_unsupported")
    return tuple(sorted(reasons))


def _raw_value(value: RawValue) -> dict[str, object]:
    return {"type": value.kind.value, "value": value.value}


def _journal_rows(rows: Sequence[RawJournalRow]) -> list[dict[str, object]]:
    return [
        {
            "columns": [
                {"name": name, **_raw_value(value)} for name, value in row.columns
            ]
        }
        for row in rows
    ]


def build_bundle(
    *,
    tool_version: str,
    snapshot: SalvageSnapshot,
    fingerprints: Sequence[SourceFingerprint],
    identity: IdentitySidecarReport,
    generation: GenerationSidecarReport,
) -> SalvageBundle:
    """Render ``memories.jsonl`` and ``manifest.json`` bytes in memory."""
    records = render_records(snapshot.records)
    reasons = unresolved_reasons(snapshot)
    manifest: dict[str, object] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "reflectlog_version": tool_version,
        "workspace_id": snapshot.workspace_id,
        "rebuild_ready": not reasons,
        "unresolved_reasons": list(reasons),
        "identity": {
            "status": identity.status.value,
            "fields": dict(identity.fields),
        },
        "generation": {"status": generation.status.value, "raw": generation.raw},
        "source_fingerprints": [
            {"path": item.path, "size": item.size, "sha256": item.sha256}
            for item in sorted(fingerprints, key=lambda item: item.path)
        ],
        "schema": {
            "journal_status": snapshot.schema.journal_status.value,
            "tables": [
                {"name": table.name, "columns": list(table.columns)}
                for table in snapshot.schema.tables
            ],
        },
        "records": {
            "file": RECORDS_FILENAME,
            "count": len(snapshot.records),
            "sha256": hashlib.sha256(records).hexdigest(),
        },
        "journal": {
            "pending": _journal_rows(snapshot.pending),
            "unknown": _journal_rows(snapshot.unknown),
            "foreign_workspace": _journal_rows(snapshot.foreign),
        },
        "history": {"completed_journal_rows": snapshot.completed_history_count},
        "foreign_active_count": snapshot.foreign_active_count,
        "not_preserved": list(NOT_PRESERVED),
    }
    rendered = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False)
    return SalvageBundle(records=records, manifest=(rendered + "\n").encode("utf-8"))
