from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock, patch

from fastmcp import Client
from mcp.types import Tool as MCPTool
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.mcp_server import AVAILABLE_TOOL_CLASSES, FastMCPServer
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.core.prompts import INSTRUCTIONS_HEADER, TOOL_ORDER, build_instructions

NAMES = [
    "add_memories",
    "search_memories",
    "get_all_memories",
    "get_memory_by_id",
    "delete_memories",
]


@pytest.mark.parametrize("names", [NAMES, NAMES[:2], NAMES[2:3], []])
def test_instructions_contain_only_selected_tools(names: list[str]) -> None:
    snippets = [(name, f"    • {name}()") for name in names]
    result = build_instructions(snippets)
    assert result.startswith(f"{INSTRUCTIONS_HEADER}\n")
    for name in NAMES:
        assert (f"{name}()" in result) == (name in names)
    if not names:
        assert "(No tools available)" in result


def test_instructions_order_tools_independently_of_input_order() -> None:
    result = build_instructions([(name, f"    • {name}()") for name in reversed(NAMES)])
    assert TOOL_ORDER == NAMES
    positions = [result.index(f"{name}()") for name in NAMES]
    assert positions == sorted(positions)


def test_unknown_tools_sort_alphabetically_after_known_tools() -> None:
    result = build_instructions(
        [
            ("zebra_tool", "zebra_tool()"),
            ("add_memories", "add_memories()"),
            ("alpha_tool", "alpha_tool()"),
        ]
    )
    assert (
        result.index("add_memories()")
        < result.index("alpha_tool()")
        < result.index("zebra_tool()")
    )


@pytest.mark.parametrize("name", NAMES)
def test_tool_snippet_documents_wire_signature(
    name: str, set_env_vars: dict[str, str]
) -> None:
    tool = AVAILABLE_TOOL_CLASSES[name](
        Config.from_environment(), MagicMock(spec=MemoryManager), MagicMock()
    )
    snippet = tool.get_instruction_snippet()
    assert snippet.startswith("    •")
    assert f"{name}(" in snippet
    assert "workspace_id: str" in snippet
    assert "-> dict" in snippet


@pytest.mark.parametrize(
    ("allowed", "expected"),
    [
        (None, NAMES),
        ("add_memories,search_memories", NAMES[:2]),
        ("get_all_memories", NAMES[2:3]),
        ("none", []),
        ("Add-Memories-Tool", NAMES[:1]),
    ],
)
async def test_dynamic_selection_and_required_workspace(
    monkeypatch: pytest.MonkeyPatch,
    set_env_vars: dict[str, str],
    allowed: str | None,
    expected: list[str],
) -> None:
    if allowed is None:
        monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    else:
        monkeypatch.setenv("ALLOWED_TOOLS", allowed)
    with patch(
        "reflectlog.application.mcp_server.create_logger", return_value=MagicMock()
    ):
        server = FastMCPServer(Config.from_environment())
    client_factory = cast("Callable[[object], Client]", Client)
    async with client_factory(server.mcp) as client:
        tools = cast("list[MCPTool]", await client.list_tools())
    assert list(server.registered_tools) == expected
    assert {tool.name for tool in tools} == set(expected)
    for tool in tools:
        assert "workspace_id" in tool.input_schema["required"]
