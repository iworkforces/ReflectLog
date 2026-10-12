"""Tests of the real offline verification command and its thin runner."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from reflectlog.application.salvage.verification import ContentSalvageVerifier
from reflectlog.application.salvage.verify_cli import run
from reflectlog.server import parse_args
from tests.unit.salvage_support import SECRET, WS, as_object, tree_digest
from tests.unit.salvage_verify_support import (
    export_bundle,
    load_manifest,
    record_line,
    store_manifest,
    write_records,
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
    "sqlite3",
    "reflectlog.application.mcp_server",
    "reflectlog.application.memory.manager",
    "reflectlog.application.memory.replacement_recovery",
    "reflectlog.application.memory.workspace_registry",
    "reflectlog.infrastructure.usearch_engine",
    "reflectlog.infrastructure.tantivy_engine",
    "reflectlog.infrastructure.memory_store",
    "reflectlog.infrastructure.embeddings",
    "reflectlog.utility.scoring",
    "reflectlog.application.config",
    "reflectlog.application.salvage.service",
)
HARNESS = """
import json, sys
bundle, workspace = sys.argv[1:3]
sys.argv = ["reflectlog", "salvage-verify", "--bundle", bundle, "--workspace-id", workspace]
from reflectlog.server import main
code = "no-exit"
try:
    main()
except SystemExit as exc:
    code = exc.code
print("EXIT=" + repr(code))
print("MODULES=" + json.dumps(sorted(sys.modules)))
"""


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "--frozen", "--no-sync", "reflectlog", "salvage-verify", *args],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )


def _verify(bundle: Path, workspace: str = WS) -> subprocess.CompletedProcess[str]:
    return _cli("--bundle", str(bundle), "--workspace-id", workspace)


@pytest.mark.parametrize("empty", [False, True])
def test_ready_bundle_when_exported(tmp_path: Path, empty: bool) -> None:
    bundle = export_bundle(tmp_path, memories=() if empty else [(WS, "kept", None)])
    completed = _verify(bundle)
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.count("\n") == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ready"
    assert payload["record_count"] == (0 if empty else 1)
    assert payload["authenticity"] == "not_verified"


def test_not_ready_when_journal_pending(tmp_path: Path) -> None:
    bundle = export_bundle(tmp_path, journal=[(WS, "pending", "add", "new text")])
    completed = _verify(bundle)
    assert completed.returncode == 1
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["status"] == "not_ready"
    assert payload["unresolved_reasons"] == ["journal_pending_intent"]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("digest", "digest_mismatch"),
        ("count", "count_mismatch"),
        ("truncated", "missing_final_newline"),
        ("duplicate", "record_id_duplicate"),
        ("workspace", "workspace_mismatch"),
        ("missing", "bundle_missing"),
        ("incomplete", "incomplete_marker_present"),
        ("symlink", "bundle_symlink"),
    ],
)
def test_invalid_bundle_when_corrupted(
    tmp_path: Path, case: str, expected: str
) -> None:
    bundle = export_bundle(tmp_path)
    workspace = WS
    if case == "digest":
        write_records(bundle, record_line(1, "changed"), resync=False)
    elif case == "count":
        manifest = load_manifest(bundle)
        as_object(manifest["records"])["count"] = 99
        store_manifest(bundle, manifest)
    elif case == "truncated":
        write_records(bundle, (bundle / "memories.jsonl").read_bytes()[:-1])
    elif case == "duplicate":
        write_records(bundle, record_line(1, "one") + record_line(1, "two"))
    elif case == "workspace":
        workspace = "other-ws"
    elif case == "missing":
        bundle = tmp_path / "missing"
    elif case == "incomplete":
        (bundle / "INCOMPLETE").touch()
    else:
        link = tmp_path / "link"
        try:
            link.symlink_to(bundle, target_is_directory=True)
        except OSError, NotImplementedError:
            pytest.skip("Directory symlinks unavailable")
        bundle = link
    completed = _verify(bundle, workspace)
    assert completed.returncode == 1, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.count("\n") == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "invalid"
    assert expected in [issue["code"] for issue in payload["errors"]]


def test_workspace_when_canonicalized(tmp_path: Path) -> None:
    completed = _verify(export_bundle(tmp_path), f"  {WS.upper()}  ")
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["workspace_id"] == WS


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--bundle", "x"],
        ["--workspace-id", WS],
        ["--bundle", "x", "--workspace-id", WS, "--unknown"],
    ],
)
def test_usage_when_flags_invalid(args: list[str]) -> None:
    completed = _cli(*args)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "usage:" in completed.stderr


@pytest.mark.parametrize("workspace", ["../x", ""])
def test_usage_when_workspace_invalid(tmp_path: Path, workspace: str) -> None:
    completed = _verify(export_bundle(tmp_path), workspace)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == (
        "reflectlog salvage-verify: --workspace-id is not a valid workspace id\n"
    )


def test_dispatch_when_server_environment_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = export_bundle(tmp_path)
    monkeypatch.setenv("MCP_PORT", "not-a-port")
    monkeypatch.setenv("MCP_TRANSPORT", "bogus")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "bogus")
    monkeypatch.delenv("REFLECTLOG_MCP_AUTH_TOKEN", raising=False)
    completed = _verify(bundle)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "ready"
    assert completed.stderr == ""


def test_imports_when_dispatched(tmp_path: Path) -> None:
    bundle = export_bundle(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", HARNESS, str(bundle), WS],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert json.loads(lines[0])["status"] == "ready"
    assert lines[1] == "EXIT=0"
    modules = set(json.loads(lines[2].removeprefix("MODULES=")))
    assert "reflectlog.application.salvage.verify_cli" in modules
    assert (
        sorted(
            name
            for name in modules
            if any(
                name == bad or name.startswith(f"{bad}.") for bad in FORBIDDEN_MODULES
            )
        )
        == []
    )


def test_read_only_when_run_outside_repo(tmp_path: Path) -> None:
    bundle = export_bundle(tmp_path)
    cwd, scratch = tmp_path / "cwd", tmp_path / "scratch"
    cwd.mkdir()
    scratch.mkdir()
    before = tree_digest(bundle)
    siblings = sorted(tmp_path.iterdir())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(REPO),
            "--frozen",
            "--no-sync",
            "python",
            "-c",
            "import os, subprocess, sys; "
            "env = dict(os.environ, TMPDIR=sys.argv[1], TEMP=sys.argv[1], TMP=sys.argv[1]); "
            "sys.exit(subprocess.run(['reflectlog', *sys.argv[2:]], env=env).returncode)",
            str(scratch),
            "salvage-verify",
            "--bundle",
            str(bundle),
            "--workspace-id",
            WS,
        ],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert tree_digest(bundle) == before
    assert sorted(tmp_path.iterdir()) == siblings
    assert list(cwd.iterdir()) == []
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("state", ["ready", "not_ready", "invalid"])
def test_content_free_when_secret_in_content_and_path(
    tmp_path: Path, state: str
) -> None:
    bundle = export_bundle(
        tmp_path,
        name=SECRET,
        memories=[(WS, SECRET, None)],
        journal=[(WS, "pending", "add", SECRET)] if state == "not_ready" else (),
    )
    if state == "invalid":
        write_records(bundle, record_line(1, SECRET + "changed"), resync=False)
    completed = _verify(bundle)
    assert completed.returncode == (0 if state == "ready" else 1)
    assert json.loads(completed.stdout)["status"] == state
    assert SECRET not in completed.stdout + completed.stderr
    assert str(bundle) not in completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("state", "code"), [("ready", 0), ("not_ready", 1), ("invalid", 1), ("usage", 2)]
)
def test_runner_when_result_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], state: str, code: int
) -> None:
    bundle = export_bundle(
        tmp_path,
        journal=[(WS, "pending", "add", "new text")] if state == "not_ready" else (),
    )
    if state == "invalid":
        (bundle / "INCOMPLETE").touch()
    args = argparse.Namespace(
        bundle=str(bundle), workspace_id="../x" if state == "usage" else WS
    )
    assert run(args) == code
    captured = capsys.readouterr()
    if state == "usage":
        assert captured.out == ""
        assert (
            captured.err
            == "reflectlog salvage-verify: --workspace-id is not a valid workspace id\n"
        )
    else:
        assert captured.err == ""
        assert captured.out.count("\n") == 1
        assert json.loads(captured.out)["status"] == state


def test_runner_when_unexpected_exception() -> None:
    with patch.object(
        ContentSalvageVerifier, "verify", side_effect=RuntimeError("bug")
    ):
        with pytest.raises(RuntimeError, match="bug"):
            _ = run(argparse.Namespace(bundle="x", workspace_id=WS))


def test_dispatch_does_not_load_server_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from reflectlog.server import main

    bundle = export_bundle(tmp_path)
    argv = [
        "reflectlog",
        "salvage-verify",
        "--bundle",
        str(bundle),
        "--workspace-id",
        WS,
    ]
    with (
        patch("sys.argv", argv),
        patch(
            "reflectlog.server._apply_cli_env_vars",
            side_effect=AssertionError("offline verification must not load server env"),
        ),
        pytest.raises(SystemExit) as exc,
    ):
        main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["status"] == "ready"


@pytest.mark.parametrize("command", ["salvage", "salvage-verify"])
def test_parser_when_command_selected(command: str) -> None:
    options = (
        ["--archive", "a", "--output", "o"]
        if command == "salvage"
        else ["--bundle", "b"]
    )
    with patch("sys.argv", ["reflectlog", command, *options, "--workspace-id", WS]):
        args = parse_args()
    assert args.command == command
    assert args.workspace_id == WS
    if command == "salvage-verify":
        assert args.bundle == "b"
    else:
        assert (args.archive, args.output) == ("a", "o")


def test_help_when_requested(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("sys.argv", ["reflectlog", "--help"]), pytest.raises(SystemExit) as exc:
        _ = parse_args()
    assert exc.value.code == 0
    assert "{salvage,salvage-verify}" in capsys.readouterr().out
