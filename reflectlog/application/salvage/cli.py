"""``reflectlog salvage`` command: argument handoff to the salvage service."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from reflectlog.application.salvage.service import ContentSalvageService
from reflectlog.core.exceptions import SalvageError

if TYPE_CHECKING:
    import argparse


def run(args: argparse.Namespace) -> int:
    """Run a salvage and return the process exit code.

    Prints one JSON summary line to stdout on success. A ``SalvageError`` is
    reported as a single ``reflectlog salvage: <message>`` line on stderr and
    exit code 1; messages never contain memory content. Unexpected exceptions
    propagate.
    """
    service = ContentSalvageService.from_defaults()
    try:
        result = service.salvage(
            archive=str(args.archive),
            workspace_id=str(args.workspace_id),
            output=str(args.output),
        )
    except SalvageError as e:
        print(f"reflectlog salvage: {e}", file=sys.stderr)
        return 1
    summary = {
        "output": result.output,
        "record_count": result.record_count,
        "rebuild_ready": result.rebuild_ready,
        "unresolved_reasons": list(result.unresolved_reasons),
    }
    print(json.dumps(summary, sort_keys=True), file=sys.stdout)
    return 0
