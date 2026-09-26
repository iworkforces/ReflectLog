from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts import run_platform_gates


def test_failed_focused_gate_reports_captured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; print('gate stdout'); print('gate stderr', file=sys.stderr); sys.exit(7)",
    ]
    monkeypatch.setattr(run_platform_gates, "_versions", lambda env: {})
    monkeypatch.setattr(
        run_platform_gates,
        "_focused_commands",
        lambda inject_failure: [("failing-gate", command)],
    )
    receipt_path = tmp_path / "gate.json"

    exit_code = run_platform_gates.main(["--focused", "--output", str(receipt_path)])

    output = capsys.readouterr()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert "failing-gate" in output.err
    assert "exit 7" in output.err
    assert "gate stdout" in output.err
    assert "gate stderr" in output.err
    assert receipt["status"] == "failed"
    assert receipt["commands"][0]["exit"] == 7
    assert receipt["commands"][0]["stdout"].strip() == "gate stdout"
    assert receipt["commands"][0]["stderr"].strip() == "gate stderr"
