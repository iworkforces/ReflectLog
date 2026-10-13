#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14.8"
# dependencies = []
# ///
"""Lock remediation performance contracts and record a bounded measurement."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UV_RUN = ("uv", "run", "--frozen", "--no-sync")
HARD_GATES = [
    "tests/unit/infrastructure/test_cached_embeddings.py",
    "tests/unit/application/memory/test_search_strategies.py",
]


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--records", type=int, default=1000)
    _ = parser.add_argument("--repetitions", type=int, default=1)
    _ = parser.add_argument("--output", type=Path, required=True)
    _ = parser.add_argument("--inject-regression", type=str, default="")
    args = parser.parse_args()

    if args.inject_regression == "timestamp-n-plus-one":
        payload = {
            "status": "failed",
            "gate": "timestamp-n-plus-one",
            "records": args.records,
            "generated_at": _now(),
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 1

    command = [
        *UV_RUN,
        "pytest",
        "-q",
        *HARD_GATES,
        "-k",
        "single_flight or records_by_contents or timestamp",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    payload: dict[str, Any] = {
        "status": "ok" if completed.returncode == 0 else "failed",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "records": args.records,
        "repetitions": args.repetitions,
        "hard_gates_exit": completed.returncode,
        "generated_at": _now(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if completed.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
