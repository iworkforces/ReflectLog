"""Unit tests for the ``reflectlog salvage`` subcommand."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
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
    @pytest.mark.parametrize(
        "stage", ["iteration", "encoding", "write", "finalization"]
    )
    def test_streaming_failures_roll_back_through_offline_dispatch(
        self, tmp_path: Path, stage: str
    ) -> None:
        archive = make_archive(
            tmp_path / "archive", memories=[(WS, SECRET, None), (WS, "second", None)]
        )
        before = tree_digest(archive)
        output = tmp_path / "export"
        injection = r"""
import contextlib, os, sqlite3, tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from reflectlog.application.salvage import service
from reflectlog.infrastructure import salvage_snapshot, salvage_publication
connections, scratch_dirs, descriptors = [], [], {}
written = 0
original_connect = sqlite3.connect
original_mkdtemp = tempfile.mkdtemp
original_stream = salvage_snapshot._stream_records
original_bundle = service.build_bundle
original_create = salvage_publication.OsPublicationFileSystem.create_file
original_write = salvage_publication.OsPublicationFileSystem.write_all
original_close = salvage_publication.OsPublicationFileSystem.close_file
def connect(*args, **kwargs):
    connection = original_connect(*args, **kwargs)
    connections.append(connection)
    return connection
def mkdtemp(*args, **kwargs):
    directory = original_mkdtemp(*args, **kwargs)
    scratch_dirs.append(directory)
    return directory
def stream(database, workspace_id):
    with contextlib.closing(original_stream(database, workspace_id)) as records:
        for position, record in enumerate(records):
            if position == 1 and stage == "iteration":
                assert written == 1
                raise RuntimeError("injected iteration failure")
            if position == 1 and stage == "encoding":
                assert written == 1
                record = replace(record, content="\ud800")
            yield record
def build_bundle(**kwargs):
    bundle = original_bundle(**kwargs)
    def finalize():
        assert written == 2
        bundle.finalize_manifest()
        raise RuntimeError("injected finalization failure")
    return replace(bundle, finalize_manifest=finalize) if stage == "finalization" else bundle
def create(filesystem, path):
    descriptor = original_create(filesystem, path)
    descriptors[descriptor] = Path(path).name
    return descriptor
def write(filesystem, descriptor, payload):
    global written
    if descriptors[descriptor] == "memories.jsonl" and stage == "write" and written == 1:
        raise OSError(28, "injected write failure")
    original_write(filesystem, descriptor, payload)
    if descriptors[descriptor] == "memories.jsonl":
        written += 1
def close(filesystem, descriptor):
    original_close(filesystem, descriptor)
    del descriptors[descriptor]
with contextlib.ExitStack() as patches:
    patches.enter_context(patch.object(sqlite3, "connect", connect))
    patches.enter_context(patch.object(tempfile, "mkdtemp", mkdtemp))
    patches.enter_context(patch.object(salvage_snapshot, "_stream_records", stream))
    patches.enter_context(patch.object(service, "build_bundle", build_bundle))
    patches.enter_context(patch.object(salvage_publication.OsPublicationFileSystem, "create_file", create))
    patches.enter_context(patch.object(salvage_publication.OsPublicationFileSystem, "write_all", write))
    patches.enter_context(patch.object(salvage_publication.OsPublicationFileSystem, "close_file", close))
    code = "no-exit"
    try:
        main()
    except SystemExit as exc:
        code = exc.code
    finally:
        closed = []
        for connection in connections:
            try:
                connection.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                closed.append(True)
            else:
                closed.append(False)
        print("RECEIPT=" + json.dumps({"exit": code, "written": written,
            "connections_closed": closed, "scratch_removed": [not os.path.exists(path) for path in scratch_dirs],
            "descriptors_closed": not descriptors}))
        print("MODULES=" + json.dumps(sorted(sys.modules)))
"""
        harness = (
            "import sys\nstage = sys.argv[4]\n"
            + HARNESS.split('code = "no-exit"')[0]
            + injection
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if not any(
                secret in key for secret in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
            )
        }

        completed = subprocess.run(
            [sys.executable, "-c", harness, str(archive), WS, str(output), stage],
            cwd=REPO,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        lines = completed.stdout.splitlines()
        assert len(lines) == 2, completed.stderr
        receipt = json.loads(lines[0].removeprefix("RECEIPT="))
        assert receipt == {
            "exit": 1 if stage == "write" else "no-exit",
            "written": 2 if stage == "finalization" else 1,
            "connections_closed": [True, True],
            "scratch_removed": [True, True],
            "descriptors_closed": True,
        }
        assert not output.exists()
        assert tree_digest(archive) == before
        modules = json.loads(lines[1].removeprefix("MODULES="))
        assert not [
            name
            for name in modules
            if any(
                name == bad or name.startswith(f"{bad}.") for bad in FORBIDDEN_MODULES
            )
            or name.startswith("reflectlog.application.config")
        ]
        assert SECRET not in completed.stdout + completed.stderr
        if stage == "write":
            assert completed.returncode == 0
            assert completed.stderr.startswith("reflectlog salvage: ")
            assert completed.stderr.count("\n") == 1
            assert "rolled back" in completed.stderr
        else:
            assert completed.returncode == 1
            assert "Traceback" in completed.stderr
            expected = "UnicodeEncodeError" if stage == "encoding" else "RuntimeError"
            assert expected in completed.stderr
            assert "reflectlog salvage: " not in completed.stderr

    def test_growing_fixed_size_payload_has_bounded_offline_cli_memory(
        self, tmp_path: Path
    ) -> None:
        peaks: list[int] = []
        environment = {
            key: value
            for key, value in os.environ.items()
            if not any(
                secret in key for secret in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
            )
        }
        harness = (
            HARNESS.replace(
                'code = "no-exit"',
                'import tracemalloc\nfrom reflectlog.application.salvage.service import ContentSalvageService\nContentSalvageService.from_defaults()\ntracemalloc.start()\ncode = "no-exit"',
            )
            + '\nprint("PEAK=" + str(tracemalloc.get_traced_memory()[1]))\n'
        )
        for count in (32, 256, 2048):
            archive = make_archive(tmp_path / f"archive-{count}", memories=[])
            connection = sqlite3.connect(archive / "usearch" / "memories.db")
            try:
                _ = connection.executemany(
                    "INSERT INTO memories (workspace_id, content, created_at) VALUES (?, ?, NULL)",
                    ((WS, f"{index:08d}" + "x" * (8192 - 8)) for index in range(count)),
                )
                connection.commit()
            finally:
                connection.close()
            output = tmp_path / f"export-{count}"
            completed = subprocess.run(
                [sys.executable, "-c", harness, str(archive), WS, str(output)],
                cwd=REPO,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, completed.stderr
            lines = completed.stdout.splitlines()
            assert json.loads(lines[0])["record_count"] == count
            assert lines[1] == "EXIT=0"
            modules = json.loads(lines[2].removeprefix("MODULES="))
            assert not [
                name
                for name in modules
                if any(
                    name == bad or name.startswith(f"{bad}.")
                    for bad in FORBIDDEN_MODULES
                )
            ]
            peaks.append(int(lines[3].removeprefix("PEAK=")))
            assert (output / "memories.jsonl").stat().st_size > count * 8192
        assert max(peaks) - min(peaks) < 2 * 1024 * 1024, peaks
        assert max(peaks) < 8 * 1024 * 1024, peaks

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
