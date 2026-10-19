"""Content-free command output for offline salvage-bundle verification."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, assert_never

from reflectlog.application.salvage.verification import (
    ContentSalvageVerifier,
    result_payload,
)
from reflectlog.core.salvage import (
    VerificationIssueCode,
    VerificationLimits,
    VerificationStatus,
)

if TYPE_CHECKING:
    import argparse


def run(args: argparse.Namespace) -> int:
    """Verify with the parsed byte budgets and emit only the content-free result."""
    limits = VerificationLimits(
        max_manifest_bytes=args.max_manifest_bytes,
        max_records_bytes=args.max_records_bytes,
    )
    result = ContentSalvageVerifier.from_defaults(limits).verify(
        bundle=str(args.bundle), workspace_id=str(args.workspace_id)
    )
    if any(
        issue.code == VerificationIssueCode.EXPECTED_WORKSPACE_INVALID
        for issue in result.issues
    ):
        print(
            "reflectlog salvage-verify: --workspace-id is not a valid workspace id",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result_payload(result), sort_keys=True), file=sys.stdout)
    match result.status:
        case VerificationStatus.READY:
            return 0
        case VerificationStatus.NOT_READY | VerificationStatus.INVALID:
            return 1
    assert_never(result.status)
