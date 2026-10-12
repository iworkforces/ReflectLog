"""Reproducible real-surface acceptance against pinned Hermes and ReflectLog."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from typing import cast

COMMIT = "818c13be1dc4fd28987e1e881a9408224afd4535"


def object_json(text: str) -> dict[str, object]:
    value: object = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON evidence object")
    return cast("dict[str, object]", value)


def env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.removeprefix("export ").split("=", 1)
                result[key.strip()] = value.strip().strip("\"'")
    return result


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        self.repo = Path(cast("str", args.repo)).resolve()
        self.root = Path(tempfile.mkdtemp(prefix="reflectlog-hermes-qa-"))
        self.source = self.root / "hermes-src"
        self.snapshot = self.root / "snapshot"
        self.python = self.root / "venv/bin/python"
        self.cli = self.root / "venv/bin/hermes"
        self.checkout = (
            Path(cast("str", args.hermes_checkout)) if args.hermes_checkout else None
        )
        self.keep = bool(args.keep)
        self.settings = env_file(Path(cast("str", args.env_file)))
        self.token = secrets.token_urlsafe(24)
        self.base = {
            **os.environ,
            "GIT_MASTER": "1",
            "HERMES_HOME": str(self.root / "setup-home"),
        }
        for key in ("REFLECTLOG_URL", "REFLECTLOG_MCP_AUTH_TOKEN"):
            _ = self.base.pop(key, None)
        self.processes: list[subprocess.Popen[bytes]] = []
        self.commands: list[subprocess.Popen[str]] = []
        self.ports: list[int] = []
        self.worktree = False
        self.real_home_existed = (Path.home() / ".hermes").exists()

    def redact(self, text: str) -> str:
        inherited = [
            value
            for key, value in os.environ.items()
            if any(word in key for word in ("TOKEN", "KEY", "SECRET", "PASSWORD"))
        ]
        for value in [self.token, *self.settings.values(), *inherited]:
            if len(value) >= 8:
                text = text.replace(value, "[REDACTED]")
        return text

    def command(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        timeout: float = 240,
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.Popen(
            command,
            env=self.base if env is None else env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.commands.append(process)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(
                command, process.returncode, stdout, stderr
            )
        finally:
            self.stop(process)

    def required(self, command: list[str]) -> None:
        result = self.command(command)
        if result.returncode:
            raise RuntimeError(self.redact(result.stderr[-2000:]))

    def setup(self) -> None:
        if self.checkout:
            self.required(
                [
                    "git",
                    "-C",
                    str(self.checkout),
                    "worktree",
                    "add",
                    "--detach",
                    str(self.source),
                    COMMIT,
                ]
            )
            self.worktree = True
        else:
            self.required(["git", "init", str(self.source)])
            self.required(
                [
                    "git",
                    "-C",
                    str(self.source),
                    "fetch",
                    "--depth",
                    "1",
                    "https://github.com/NousResearch/hermes-agent.git",
                    COMMIT,
                ]
            )
            self.required(
                ["git", "-C", str(self.source), "checkout", "--detach", "FETCH_HEAD"]
            )
        self.required(["uv", "venv", "--python", "3.14", str(self.root / "venv")])
        self.required(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(self.python),
                "-e",
                str(self.source),
            ]
        )
        tracked = self.command(
            ["git", "-C", str(self.repo), "ls-files", "integrations/hermes"]
        )
        if tracked.returncode:
            raise RuntimeError("Cannot enumerate tracked plugin files")
        for name in tracked.stdout.splitlines():
            target = self.snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copy2(self.repo / name, target)
        self.required(["git", "init", str(self.snapshot)])
        self.required(["git", "-C", str(self.snapshot), "add", "."])
        self.required(
            [
                "git",
                "-C",
                str(self.snapshot),
                "-c",
                "user.name=qa",
                "-c",
                "user.email=qa@example.invalid",
                "commit",
                "-m",
                "QA working-tree snapshot",
            ]
        )

    def helper(
        self, phase: str, env: dict[str, str], timeout: float = 150
    ) -> dict[str, object]:
        result = self.command(
            [str(self.python), str(Path(__file__).with_name("scenarios.py")), phase],
            env,
            timeout,
        )
        markers = [
            line.removeprefix("QA_RESULT=")
            for line in result.stdout.splitlines()
            if line.startswith("QA_RESULT=")
        ]
        if not markers:
            raise RuntimeError(
                "Hermes driver failed: " + self.redact(result.stderr[-2000:])
            )
        return object_json(markers[-1])

    def start(
        self, storage: Path, port: int, env: dict[str, str]
    ) -> subprocess.Popen[bytes]:
        server_env = {
            **self.base,
            **self.settings,
            "REFLECTLOG_MCP_AUTH_TOKEN": self.token,
            "MCP_HOST": "127.0.0.1",
            "MCP_PORT": str(port),
            "MCP_TRANSPORT": "http",
        }
        with (storage / "server.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    str(self.repo / ".venv/bin/reflectlog"),
                    "--transport",
                    "http",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--path",
                    "/mcp",
                ],
                cwd=storage,
                env=server_env,
                stdout=log,
                stderr=log,
            )
        self.processes.append(process)
        end = time.monotonic() + 180
        while time.monotonic() < end and process.poll() is None:
            probe = self.helper("probe", env, 15)
            if not probe["reason"]:
                return process
            time.sleep(0.3)
        raise RuntimeError("ReflectLog readiness deadline exceeded")

    @staticmethod
    def stop(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.terminate()
        try:
            _ = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            _ = process.wait()

    def scenario(self, name: str) -> dict[str, object]:
        start = time.monotonic()
        assertions: list[object] = []
        verdict = "PASS"
        reason = ""
        directory = self.root / name
        home = directory / "home"
        storage = directory / "storage"
        home.mkdir(parents=True)
        storage.mkdir()
        env = {
            **self.base,
            "HERMES_HOME": str(home),
            "QA_TOKEN": self.token,
            "REFLECTLOG_MCP_AUTH_TOKEN": self.token,
        }
        process: subprocess.Popen[bytes] | None = None
        try:
            if name == "S1":
                plugin = self.snapshot / "integrations/hermes"
                for args in (
                    ["plugins", "validate", str(plugin), "--json"],
                    ["plugins", "doctor", "--ci", str(plugin)],
                    [
                        "plugins",
                        "install",
                        "file://" + str(self.snapshot) + "#integrations/hermes",
                        "--enable",
                        "--yes-deps",
                    ],
                ):
                    result = self.command([str(self.cli), *args], env)
                    ok = result.returncode == 0
                    if "validate" in args:
                        ok = ok and object_json(result.stdout)["ok"] is True
                    assertions.append(
                        {
                            "name": " ".join(args[:2]),
                            "ok": ok,
                            "detail": self.redact(result.stdout[-1800:]),
                        }
                    )
                    if not ok:
                        raise AssertionError("Hermes CLI " + " ".join(args[:2]))
                installed = (home / "plugins/reflectlog/plugin.yaml").exists()
                assertions.append(
                    {"name": "installed plugin manifest", "ok": installed, "detail": ""}
                )
                assert installed, "installed plugin manifest"
                config = self.helper("config", env)
                assertions.extend(cast("list[object]", config["assertions"]))
                _ = env.pop("REFLECTLOG_MCP_AUTH_TOKEN")
                _ = (home / ".env").write_text(
                    "REFLECTLOG_MCP_AUTH_TOKEN=" + self.token + "\n"
                )
                with_token = self.command([str(self.cli), "memory", "status"], env)
                (home / ".env").unlink()
                without = self.command([str(self.cli), "memory", "status"], env)
                ok = (
                    re.search(r"Provider:\s+reflectlog\b", with_token.stdout)
                    is not None
                    and re.search(r"Status:\s+available", with_token.stdout) is not None
                    and re.search(r"Status:\s+not available", without.stdout)
                    is not None
                    and "REFLECTLOG_MCP_AUTH_TOKEN" in without.stdout
                )
                assertions.append(
                    {
                        "name": "secret-backed status and missing-token status",
                        "ok": ok,
                        "detail": self.redact(with_token.stdout + without.stdout),
                    }
                )
                assert not config["reason"], config["reason"]
                assert ok, "availability status"
            else:
                # S1 verifies CLI cloning; copy its same tracked snapshot for runtime
                # cases to avoid repeatedly reinstalling an identical dependency set.
                _ = shutil.copytree(
                    self.snapshot / "integrations/hermes", home / "plugins/reflectlog"
                )
                _ = (home / "config.yaml").write_text(
                    "memory:\n  provider: reflectlog\n"
                    "plugins:\n  enabled:\n    - reflectlog\n"
                )
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                self.ports.append(port)
                env.update(QA_URL=f"http://127.0.0.1:{port}/mcp", QA_PORT=str(port))
                _ = (home / "reflectlog.json").write_text(
                    json.dumps({"url": env["QA_URL"]})
                )
                process = self.start(storage, port, env)
                env["QA_SERVER_PID"] = str(process.pid)
                phase = "seed" if name == "S6" else name
                evidence = self.helper(phase, env)
                assertions.extend(cast("list[object]", evidence["assertions"]))
                assert not evidence["reason"], evidence["reason"]
                if name == "S4":
                    self.stop(process)
                    process = self.start(storage, port, env)
                    evidence = self.helper("recovery", env)
                    assertions.extend(cast("list[object]", evidence["assertions"]))
                    assert not evidence["reason"], evidence["reason"]
                if name == "S6":
                    key = self.settings.get("OPENROUTER_API_KEY", "")
                    if not key:
                        verdict, reason = "SKIP", "OPENROUTER_API_KEY missing"
                    else:
                        model = self.settings.get(
                            "LLM_MODEL", "google/gemini-2.5-flash-lite"
                        )
                        _ = (home / ".env").write_text(
                            "OPENROUTER_API_KEY="
                            + key
                            + "\nREFLECTLOG_MCP_AUTH_TOKEN="
                            + self.token
                            + "\n"
                        )
                        _ = (home / "config.yaml").write_text(
                            "memory:\n  provider: reflectlog\nmodel:\n  default: "
                            + json.dumps(model)
                            + "\n  provider: openrouter\n"
                        )
                        help_result = self.command(
                            [str(self.cli), "chat", "--help"], env
                        )
                        assert (
                            "--query" in help_result.stdout
                            or "-q" in help_result.stdout
                        )
                        chat = self.command(
                            [
                                str(self.cli),
                                "chat",
                                "-q",
                                "Which widgets does project Zephyr-7741 use? "
                                "Answer with the widget color.",
                            ],
                            env,
                        )
                        output = chat.stdout + chat.stderr
                        unusable = any(
                            word in output.lower()
                            for word in (
                                "invalid api key",
                                "authentication failed",
                                "no endpoints found",
                                "model not found",
                                "401 unauthorized",
                                "402 payment",
                            )
                        )
                        if unusable:
                            verdict, reason = (
                                "SKIP",
                                "Hermes reports unusable LLM credentials/model",
                            )
                        else:
                            ok = (
                                chat.returncode == 0
                                and "cobalt" in output.lower()
                                and "reflectlog" in output.lower()
                            )
                            assertions.append(
                                {
                                    "name": "real chat seeded answer/provider evidence",
                                    "ok": ok,
                                    "detail": self.redact(output[-3500:]),
                                }
                            )
                            assert ok, "real chat seeded answer/provider indicator"
        except Exception as error:
            verdict, reason = "FAIL", self.redact(f"{type(error).__name__}: {error}")
            log = storage / "server.log"
            if log.exists():
                print(
                    json.dumps(
                        {
                            "server_log_tail": self.redact(
                                "\n".join(
                                    log.read_text(errors="replace").splitlines()[-40:]
                                )
                            )
                        }
                    )
                )
        finally:
            if process:
                self.stop(process)
        return {
            "scenario": name,
            "verdict": verdict,
            "elapsed_s": round(time.monotonic() - start, 3),
            "hermes_commit": COMMIT,
            "backend": {
                "embedder": self.settings.get("EMBEDDER_PROVIDER", "openai"),
                "reranker": self.settings.get("RERANKER_ENGINE", "cross_encoder"),
            },
            "assertions": assertions,
            "reason": reason,
        }

    def cleanup(self) -> dict[str, object]:
        for process in self.commands:
            self.stop(process)
        for process in self.processes:
            self.stop(process)
        if self.worktree and self.checkout:
            self.required(
                [
                    "git",
                    "-C",
                    str(self.checkout),
                    "worktree",
                    "remove",
                    "--force",
                    str(self.source),
                ]
            )
            self.required(["git", "-C", str(self.checkout), "worktree", "prune"])
        free = True
        for port in self.ports:
            with socket.socket() as sock:
                free = free and sock.connect_ex(("127.0.0.1", port)) != 0
        if not self.keep:
            shutil.rmtree(self.root)
        home_safe = self.real_home_existed or not (Path.home() / ".hermes").exists()
        receipt: dict[str, object] = {
            "cleanup": {
                "pids_reaped": [
                    p.pid for p in self.processes if p.returncode is not None
                ],
                "ports_free": free,
                "worktree_removed": not self.worktree or not self.source.exists(),
                "temp_dirs_deleted": not self.root.exists(),
                "real_hermes_home_untouched": home_safe,
            }
        }
        if self.keep:
            receipt["kept_directory"] = str(self.root)
        print(json.dumps(receipt))
        assert free and home_safe, "Resource cleanup failed"
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--scenario", action="append", choices=[f"S{i}" for i in range(1, 7)]
    )
    _ = parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    _ = parser.add_argument("--env-file")
    _ = parser.add_argument("--hermes-checkout")
    _ = parser.add_argument("--report")
    _ = parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    args.env_file = args.env_file or str(Path(cast("str", args.repo)) / ".env")
    run = Run(args)
    reports: list[dict[str, object]] = []
    try:
        run.setup()
        for name in cast("list[str]", args.scenario or [f"S{i}" for i in range(1, 7)]):
            evidence = run.scenario(name)
            reports.append(evidence)
            print(json.dumps(evidence), flush=True)
    finally:
        receipt = run.cleanup()
        if args.report:
            _ = Path(cast("str", args.report)).write_text(
                json.dumps({"scenarios": reports, **receipt}, indent=2) + "\n"
            )
    return int(any(item["verdict"] == "FAIL" for item in reports))


if __name__ == "__main__":
    raise SystemExit(main())
