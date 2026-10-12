"""Unit tests for the ``reflectlog salvage`` subcommand."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from reflectlog.application.salvage.cli import run
from reflectlog.application.salvage.service import ContentSalvageService
from reflectlog.server import main, parse_args
from tests.unit.salvage_support import (
    OTHER,
    SECRET,
    WS,
    make_archive,
    make_wal_archive,
    tree_digest,
)

REPO = Path(__file__).resolve().parents[2]
FORBIDDEN_MODULES = (
    "usearch",
    "tantivy",
    "torch",
    "sentence_transformers",
    "numba",
    "ranx",
    "fastmcp",
    "reflectlog.application.mcp_server",
    "reflectlog.application.memory.manager",
    "reflectlog.application.memory.replacement_recovery",
    "reflectlog.application.memory.workspace_registry",
    "reflectlog.infrastructure.usearch_engine",
    "reflectlog.infrastructure.tantivy_engine",
    "reflectlog.infrastructure.memory_store",
    "reflectlog.infrastructure.embeddings",
    "reflectlog.utility.scoring",
)
HARNESS = """
import json, sys
archive, workspace, output = sys.argv[1:4]
sys.argv = [
    "reflectlog", "salvage",
    "--archive", archive, "--workspace-id", workspace, "--output", output,
]
from reflectlog.server import main
code = "no-exit"
try:
    main()
except SystemExit as exc:
    code = exc.code
print("EXIT=" + repr(code))
print("MODULES=" + json.dumps(sorted(sys.modules)))
"""


def _args(archive: Path, output: Path, workspace: str = WS) -> argparse.Namespace:
    return argparse.Namespace(
        command="salvage",
        archive=str(archive),
        workspace_id=workspace,
        output=str(output),
    )


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "--frozen", "--no-sync", "reflectlog", *args],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )


class TestParsing:
    def test_subcommand_requires_all_three_options(self) -> None:
        for missing in ("--archive", "--workspace-id", "--output"):
            argv = ["reflectlog", "salvage"]
            for option in ("--archive", "--workspace-id", "--output"):
                if option != missing:
                    argv += [option, "x"]
            with patch("sys.argv", argv), pytest.raises(SystemExit) as exc:
                _ = parse_args()
            assert exc.value.code == 2

    def test_subcommand_values_are_parsed(self) -> None:
        argv = [
            "reflectlog",
            "salvage",
            "--archive",
            "a",
            "--workspace-id",
            "w",
            "--output",
            "o",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        assert (args.command, args.archive, args.workspace_id, args.output) == (
            "salvage",
            "a",
            "w",
            "o",
        )

    def test_no_subcommand_keeps_the_server_flags(self) -> None:
        with patch("sys.argv", ["reflectlog", "--port", "9999"]):
            args = parse_args()
        assert args.command is None
        assert args.port == 9999
        assert args.transport is None

    def test_server_flags_are_rejected_after_the_subcommand(self) -> None:
        with patch("sys.argv", ["reflectlog", "salvage", "--port", "1"]):
            with pytest.raises(SystemExit) as exc:
                _ = parse_args()
        assert exc.value.code == 2


class TestMainDispatch:
    def test_salvage_exits_with_the_command_status_before_any_server_work(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        argv = ["reflectlog", "salvage", "--archive", "a", "--workspace-id", "w"]
        argv += ["--output", "o"]
        with (
            patch("sys.argv", argv),
            patch("reflectlog.application.salvage.cli.run", return_value=7) as fake,
            patch("reflectlog.server._apply_cli_env_vars") as env,
            patch("reflectlog.server._start_server") as start,
            patch("reflectlog.server._run_numba_warmup") as warmup,
            pytest.raises(SystemExit) as exc,
        ):
            main()

        assert exc.value.code == 7
        fake.assert_called_once()
        env.assert_not_called()
        start.assert_not_called()
        warmup.assert_not_called()
        assert "MCP_TRANSPORT" not in os.environ
        assert capsys.readouterr().out == ""


class TestRun:
    def test_success_prints_exactly_one_json_summary_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        archive, _ = make_wal_archive(tmp_path / "archive", tmp_path, "wal row")

        code = run(_args(archive, tmp_path / "export"))

        captured = capsys.readouterr()
        assert code == 0
        assert captured.err == ""
        assert captured.out.count("\n") == 1
        assert json.loads(captured.out) == {
            "output": os.path.join(os.path.realpath(tmp_path), "export"),
            "record_count": 1,
            "rebuild_ready": True,
            "unresolved_reasons": [],
        }
        assert "wal row" not in captured.out

    def test_unresolved_evidence_still_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        archive = make_archive(
            tmp_path / "archive",
            memories=[(WS, "kept", None)],
            journal=[(WS, "pending", "add", f"{SECRET} intent")],
        )

        code = run(_args(archive, tmp_path / "export"))

        summary = json.loads(capsys.readouterr().out)
        assert code == 0
        assert summary["rebuild_ready"] is False
        assert summary["unresolved_reasons"] == ["journal_pending_intent"]

    @pytest.mark.parametrize(
        "case", ["missing-db", "existing-output", "inside-archive"]
    )
    def test_failures_print_one_stderr_line_and_exit_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
    ) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, SECRET, None)])
        output = tmp_path / "export"
        if case == "missing-db":
            (archive / "usearch" / "memories.db").unlink()
        elif case == "existing-output":
            output.mkdir()
        else:
            output = archive / "export"
        before = tree_digest(archive)

        code = run(_args(archive, output))

        captured = capsys.readouterr()
        assert code == 1
        assert captured.out == ""
        assert captured.err.startswith("reflectlog salvage: ")
        assert captured.err.count("\n") == 1
        assert "Traceback" not in captured.err
        assert SECRET not in captured.err
        assert tree_digest(archive) == before

    def test_invalid_workspace_is_a_clean_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "a", None)])

        code = run(_args(archive, tmp_path / "export", workspace="bad/ws"))

        assert code == 1
        assert "invalid workspace id" in capsys.readouterr().err

    def test_unexpected_exceptions_propagate(self, tmp_path: Path) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "a", None)])
        with patch.object(
            ContentSalvageService, "salvage", side_effect=RuntimeError("bug")
        ):
            with pytest.raises(RuntimeError, match="bug"):
                _ = run(_args(archive, tmp_path / "export"))


class TestSubprocess:
    def test_installed_command_exports_wal_only_content_without_heavy_imports(
        self, tmp_path: Path
    ) -> None:
        archive, row_id = make_wal_archive(tmp_path / "archive", tmp_path, "wal row")
        before = tree_digest(archive)
        output = tmp_path / "export"

        completed = subprocess.run(
            [sys.executable, "-c", HARNESS, str(archive), WS, str(output)],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        lines = completed.stdout.splitlines()
        assert json.loads(lines[0])["record_count"] == 1
        assert lines[1] == "EXIT=0"
        modules = set(json.loads(lines[2].removeprefix("MODULES=")))
        assert "reflectlog.application.salvage.cli" in modules
        offenders = sorted(
            name
            for name in modules
            if any(
                name == bad or name.startswith(f"{bad}.") for bad in FORBIDDEN_MODULES
            )
        )
        assert offenders == []
        rows = [
            json.loads(line)
            for line in (output / "memories.jsonl").read_text("utf-8").splitlines()
        ]
        assert [(row["source_id"], row["content"]) for row in rows] == [
            (row_id, "wal row")
        ]
        assert tree_digest(archive) == before

    def test_reflectlog_salvage_command_line_succeeds(self, tmp_path: Path) -> None:
        archive = make_archive(
            tmp_path / "archive",
            memories=[(WS, "kept", "t"), (OTHER, "foreign", None)],
        )
        output = tmp_path / "export"

        completed = _cli(
            "salvage",
            "--archive",
            str(archive),
            "--workspace-id",
            WS,
            "--output",
            str(output),
        )

        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["record_count"] == 1
        assert completed.stderr == ""
        assert sorted(os.listdir(output)) == ["manifest.json", "memories.jsonl"]

    @pytest.mark.parametrize(
        "case", ["missing-db", "existing-output", "inside-archive"]
    )
    def test_reflectlog_salvage_command_line_failures_exit_one(
        self, tmp_path: Path, case: str
    ) -> None:
        archive = make_archive(tmp_path / "archive", memories=[(WS, "kept", None)])
        output = tmp_path / "export"
        if case == "missing-db":
            (archive / "usearch" / "memories.db").unlink()
        elif case == "existing-output":
            output.mkdir()
        else:
            output = archive / "export"

        completed = _cli(
            "salvage",
            "--archive",
            str(archive),
            "--workspace-id",
            WS,
            "--output",
            str(output),
        )

        assert completed.returncode == 1
        assert completed.stdout == ""
        assert completed.stderr.startswith("reflectlog salvage: ")
        assert "Traceback" not in completed.stderr

    def test_missing_required_option_is_a_usage_error(self, tmp_path: Path) -> None:
        completed = _cli("salvage", "--archive", str(tmp_path))
        assert completed.returncode == 2
        assert "--workspace-id" in completed.stderr
