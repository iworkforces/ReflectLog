"""Real Hermes runtime scenarios; invoked only in the pinned Hermes environment."""

from __future__ import annotations

from contextlib import ExitStack
from importlib import import_module
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..core.client import ReflectLogClient
    from ..provider import ReflectLogMemoryProvider


class Manager(Protocol):
    def add_provider(self, provider: object) -> None: ...
    def initialize_all(self, session_id: str, **kwargs: object) -> None: ...
    def on_turn_start(self, turn: int, message: str, **kwargs: object) -> None: ...
    def prefetch_all(self, query: str, **kwargs: object) -> str: ...
    def handle_tool_call(self, name: str, args: dict[str, object]) -> str: ...
    def notify_memory_tool_write(
        self, result: str, args: dict[str, object]
    ) -> None: ...
    def shutdown_all(self) -> None: ...


class YamlReader(Protocol):
    def load(self, text: str) -> object: ...


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return cast("dict[str, object]", value)


def rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array")
    return [obj(item) for item in cast("list[object]", value)]


class Scenario:
    def __init__(self) -> None:
        self.stack = ExitStack()
        self.assertions: list[dict[str, object]] = []
        self.home = Path(os.environ["HERMES_HOME"])

    def check(self, name: str, ok: bool, detail: object = "") -> None:
        self.assertions.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            raise AssertionError(name)

    def provider(self, **kwargs: object) -> tuple[ReflectLogMemoryProvider, Manager]:
        loader = cast(
            "Callable[[str], object]",
            import_module("plugins.memory").load_memory_provider,
        )
        provider = cast("ReflectLogMemoryProvider", loader("reflectlog"))
        factory = cast(
            "Callable[[], Manager]", import_module("agent.memory_manager").MemoryManager
        )
        manager = factory()
        _ = self.stack.callback(manager.shutdown_all)
        manager.add_provider(provider)
        context: dict[str, object] = {
            "platform": "cli",
            "agent_context": "primary",
            "hermes_home": str(self.home),
            "agent_identity": cast(
                "Callable[[], str]",
                import_module("hermes_cli.profiles").get_active_profile_name,
            )(),
        }
        context.update(kwargs)
        manager.initialize_all("qa-session", **context)
        return provider, manager

    def direct(self, provider: ReflectLogMemoryProvider) -> ReflectLogClient:
        module = provider.__class__.__module__.rsplit(".", 1)[0] + ".core.client"
        factory = cast(
            "Callable[..., ReflectLogClient]", import_module(module).ReflectLogClient
        )
        client = factory(
            os.environ["QA_URL"],
            os.environ["QA_TOKEN"],
            connect_timeout=2,
            request_timeout=30,
        )
        _ = self.stack.callback(client.close, 0.5)
        return client

    def workspace(self, provider: ReflectLogMemoryProvider) -> str:
        assert provider.binding is not None
        assert provider.binding.workspace_id is not None
        return provider.binding.workspace_id

    def call(self, manager: Manager, name: str, **args: object) -> dict[str, object]:
        return obj(json.loads(manager.handle_tool_call("reflectlog_" + name, args)))

    def drain(self, provider: ReflectLogMemoryProvider) -> None:
        assert provider.writer is not None
        self.check("writer drained", provider.writer.drain(90))

    def run(self, scenario: str) -> None:
        if scenario == "probe":
            p, _ = self.provider()
            _ = self.direct(p).get_all_memories("qa-readiness", limit=0)
            return
        if scenario == "config":
            factory = cast(
                "Callable[..., YamlReader]", import_module("ruamel.yaml").YAML
            )
            path = self.home / "config.yaml"
            config = (
                obj(factory(typ="safe").load(path.read_text())) if path.exists() else {}
            )
            memory = obj(config.get("memory", {}))
            self.check(
                "activated provider",
                memory.get("provider") == "reflectlog",
                {
                    "memory.provider": memory.get("provider"),
                    "config_exists": path.exists(),
                },
            )
            return
        p, m = self.provider()
        direct = self.direct(p)
        workspace = self.workspace(p)
        fact = "QA fixture: project Zephyr-7741 uses cobalt widgets"
        question = "Which widgets does project Zephyr-7741 use?"
        if scenario in {"S2", "seed"}:
            self.check(
                "stored_count",
                self.call(m, "store", memories=[fact])["stored_count"] == 1,
            )
            if scenario == "seed":
                return
            m.on_turn_start(1, question)
            raw = m.prefetch_all(question, session_id="qa-session")
            wrap = cast(
                "Callable[[str], str]",
                import_module("agent.memory_manager").build_memory_context_block,
            )
            block = wrap(raw)
            self.check(
                "single fenced recall with fixture and safety header",
                block.count("<memory-context>") == 1
                and fact in block
                and "stored data, not instructions" in block,
            )
            status = p.recall_status()
            headers = [
                line
                for line in block.splitlines()
                if line.startswith("ReflectLog recalled memories")
            ]
            self.check(
                "single-line recall retry header",
                len(headers) == 1
                and headers[0].endswith("):")
                and "attempt=2" in headers[0],
            )
            self.check("recall count", status is not None and status.count >= 1)
            listed = rows(self.call(m, "list")["records"])
            record = listed[0]
            memory_id = record["id"]
            searched = self.call(m, "search", query=question)
            next_step = searched.get("next_step")
            self.check(
                "default search attempt and guidance",
                searched.get("attempt") == 1
                and isinstance(next_step, str)
                and bool(next_step),
            )
            found = rows(searched["results"])
            final_step = self.call(m, "search", query=question, attempt=5).get(
                "next_step"
            )
            self.check(
                "search limit reached",
                isinstance(final_step, str) and "Search limit reached" in final_step,
            )
            self.check(
                "sixth search rejected",
                self.call(m, "search", query=question, attempt=6).get("code")
                == "invalid_input",
            )
            got = obj(self.call(m, "get", id=memory_id)["record"])
            self.check(
                "search/list/get agree",
                record["memory"] == fact
                and got["id"] == memory_id
                and got["memory"] == fact
                and any(r["id"] == memory_id and r["memory"] == fact for r in found),
            )
            self.check(
                "forget removed id",
                self.call(m, "forget", ids=[memory_id])["removed"] == [memory_id],
            )
            self.check(
                "independent MCP confirms absent",
                direct.get_all_memories(workspace).total == 0,
            )
        elif scenario == "S3":
            native = import_module("tools.memory_tool")
            store = cast("Callable[[], object]", native.load_on_disk_store)()
            tool = cast("Callable[..., str]", native.memory_tool)
            replacement = "QA fixture: project Zephyr-7741 uses amber widgets"
            operations: list[dict[str, object]] = [
                {"action": "add", "target": "memory", "content": fact},
                {
                    "action": "replace",
                    "target": "memory",
                    "old_text": fact,
                    "new_text": replacement,
                },
                {"action": "remove", "target": "memory", "old_text": replacement},
            ]
            for args in operations:
                result = tool(store=store, **args)
                self.check(
                    "native " + str(args["action"]),
                    obj(json.loads(result)).get("success") is True,
                )
                m.notify_memory_tool_write(result, args)
            self.drain(p)
            page = direct.get_all_memories(workspace)
            self.check(
                "add-only mirror preserves both versions",
                {r.memory for r in page.records} == {fact, replacement},
            )
            assert p.writer is not None
            stats = p.writer.stats()
            self.check(
                "two completed zero failed",
                stats["completed"] == 2 and stats["failed"] == 0,
                stats,
            )
        elif scenario == "S4":
            self.check("initial explicit list", "error" not in self.call(m, "list"))
            os.kill(int(os.environ["QA_SERVER_PID"]), signal.SIGTERM)
            end = time.monotonic() + 10
            while time.monotonic() < end:
                with socket.socket() as sock:
                    if sock.connect_ex(("127.0.0.1", int(os.environ["QA_PORT"]))) != 0:
                        break
                time.sleep(0.05)
            start = time.monotonic()
            recalled = m.prefetch_all(question, session_id="qa-session")
            elapsed = time.monotonic() - start
            self.check(
                "outage recall empty within 3.5s",
                recalled == "" and elapsed < 3.5,
                elapsed,
            )
            self.check(
                "explicit unavailable",
                self.call(m, "list").get("code") == "unavailable",
            )
            start = time.monotonic()
            m.shutdown_all()
            elapsed = time.monotonic() - start
            self.check("shutdown within 2.5s", elapsed < 2.5, elapsed)
        elif scenario == "recovery":
            self.check("restart explicit list", "error" not in self.call(m, "list"))
            os.environ["REFLECTLOG_MCP_AUTH_TOKEN"] = "qa-wrong-token"
            _, wrong = self.provider()
            self.check(
                "wrong token unauthorized",
                self.call(wrong, "list").get("code") == "unauthorized",
            )
        elif scenario == "S5":
            a, am = self.provider(
                platform="telegram", chat_type="dm", user_id="qa-user-a"
            )
            b, bm = self.provider(
                platform="telegram", chat_type="dm", user_id="qa-user-b"
            )
            self.check("distinct workspaces", self.workspace(a) != self.workspace(b))
            self.check(
                "A stores", self.call(am, "store", memories=[fact])["stored_count"] == 1
            )
            self.check(
                "B isolated",
                bm.prefetch_all(question, session_id="qa-session") == ""
                and direct.get_all_memories(self.workspace(b)).total == 0,
            )
            _, group = self.provider(
                platform="telegram", chat_type="group", user_id="qa-user-a"
            )
            self.check(
                "group disabled",
                self.call(group, "list").get("code") == "disabled"
                and group.prefetch_all(question, session_id="qa-session") == "",
            )
            config: dict[str, object] = {
                "url": os.environ["QA_URL"],
                "workspace_scope": "profile",
            }
            _ = (self.home / "reflectlog.json").write_text(json.dumps(config))
            for context in ("subagent", "cron"):
                gated, _ = self.provider(platform=context, agent_context=context)
                before = direct.get_all_memories(self.workspace(gated)).total
                gated.on_memory_write("add", "memory", fact, {})
                gated.sync_turn(fact, "Noted", session_id="qa-session")
                self.drain(gated)
                self.check(
                    context + " automatic writes gated",
                    direct.get_all_memories(self.workspace(gated)).total == before,
                )
            before = direct.get_all_memories(workspace).total
            p.sync_turn(fact, "Noted", session_id="qa-session")
            self.drain(p)
            self.check(
                "default capture off",
                direct.get_all_memories(workspace).total == before,
            )
            config["capture_turns"] = True
            _ = (self.home / "reflectlog.json").write_text(json.dumps(config))
            capture, _ = self.provider()
            capture.sync_turn(
                "I prefer teal dashboards for QA fixture 5",
                "Noted",
                session_id="qa-session",
            )
            self.drain(capture)
            page = direct.get_all_memories(self.workspace(capture))
            self.check(
                "capture one conversation record",
                page.total == 1
                and page.records[0].memory.startswith("Conversation turn (Hermes"),
            )
        else:
            raise ValueError("Unknown scenario")


def main() -> int:
    scenario = Scenario()
    reason = ""
    try:
        scenario.run(sys.argv[1])
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
    finally:
        scenario.stack.close()
    print(
        "QA_RESULT=" + json.dumps({"assertions": scenario.assertions, "reason": reason})
    )
    return int(bool(reason))


if __name__ == "__main__":
    raise SystemExit(main())
