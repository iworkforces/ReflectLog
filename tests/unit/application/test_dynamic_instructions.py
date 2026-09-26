"""Tests for dynamic MCP instructions generation."""
# mypy: disable-error-code="misc,var-annotated"

from collections.abc import Callable
from typing import Protocol, cast, runtime_checkable
from unittest.mock import MagicMock, patch

from fastmcp import Client
import pytest
from pytest import MonkeyPatch

from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.core.prompts import (
    INSTRUCTIONS_HEADER,
    TOOL_ORDER,
    build_instructions,
)


@runtime_checkable
class _SchemaTool(Protocol):
    name: str
    inputSchema: dict[str, list[str]]


@pytest.mark.unit
class TestBuildInstructions:
    """Test suite for build_instructions() function."""

    def test_build_with_all_tools(self):
        """Test building instructions with all four tools."""
        snippets = [
            ("add", "    • add snippet"),
            ("get_all", "    • get_all snippet"),
            ("search", "    • search snippet"),
            ("remove", "    • remove snippet"),
        ]
        result = build_instructions(snippets)

        assert INSTRUCTIONS_HEADER in result
        for _, snippet in snippets:
            assert snippet in result

    def test_build_with_subset_of_tools(self):
        """Test building with only some tools enabled."""
        snippets = [
            ("add", "    • add snippet"),
            ("search", "    • search snippet"),
        ]
        result = build_instructions(snippets)

        assert "add snippet" in result
        assert "search snippet" in result
        assert "get_all snippet" not in result
        assert "remove snippet" not in result

    def test_build_with_single_tool(self):
        """Test building with a single tool."""
        snippets = [("get_all", "    • get_all() -> list[str]")]
        result = build_instructions(snippets)

        assert "get_all() -> list[str]" in result
        assert "add" not in result

    def test_build_with_empty_tools(self):
        """Test building with no tools."""
        result = build_instructions([])

        assert INSTRUCTIONS_HEADER in result
        assert "(No tools available)" in result

    def test_tool_ordering_preserved(self):
        """Test tools appear in predefined order regardless of input order."""
        # Input in reverse order
        snippets = [
            ("remove", "    • remove snippet"),
            ("add", "    • add snippet"),
            ("search", "    • search snippet"),
            ("get_all", "    • get_all snippet"),
        ]
        result = build_instructions(snippets)

        # Should be sorted according to TOOL_ORDER: add, get_all, search, remove
        add_pos = result.find("add snippet")
        get_all_pos = result.find("get_all snippet")
        search_pos = result.find("search snippet")
        remove_pos = result.find("remove snippet")

        assert add_pos < get_all_pos < search_pos < remove_pos

    def test_unknown_tools_appear_at_end_alphabetically(self):
        """Test unknown tools are sorted to end in alphabetical order."""
        snippets = [
            ("zebra_tool", "    • zebra snippet"),
            ("add", "    • add snippet"),
            ("alpha_tool", "    • alpha snippet"),
        ]
        result = build_instructions(snippets)

        add_pos = result.find("add snippet")
        alpha_pos = result.find("alpha snippet")
        zebra_pos = result.find("zebra snippet")

        # add should come first (in TOOL_ORDER), then alpha, then zebra
        assert add_pos < alpha_pos < zebra_pos

    def test_header_always_present(self):
        """Header should always be included."""
        result = build_instructions([("add", "    • add snippet")])

        assert "ReflectLog Server" in result
        assert "Available Tools:" in result

    def test_tool_order_constant(self):
        """Verify TOOL_ORDER contains expected tools in correct order."""
        assert TOOL_ORDER == ["add", "get_all", "search", "remove", "health_check"]


@pytest.mark.unit
class TestToolInstructionSnippets:
    """Tests for tool get_instruction_snippet() implementations."""

    def test_add_tool_snippet_format(
        self, mock_memory_class: MagicMock, set_env_vars: None
    ) -> None:
        """Test AddTool provides properly formatted snippet."""
        from reflectlog.application.tools.add import AddTool

        mock_memory_manager = cast(MemoryManager, MagicMock())
        mock_logger = MagicMock()
        mock_config = MagicMock()

        tool = AddTool(mock_config, mock_memory_manager, mock_logger)
        snippet = tool.get_instruction_snippet()

        assert "add(memories: list[str], dry_run: bool = False)" in snippet
        assert (
            "semantic embeddings" in snippet.lower() or "embeddings" in snippet.lower()
        )
        assert snippet.startswith("    •")

    def test_get_all_tool_snippet_format(
        self, mock_memory_class: MagicMock, set_env_vars: None
    ) -> None:
        """Test GetAllTool provides properly formatted snippet."""
        from reflectlog.application.tools.get_all import GetAllTool

        mock_memory_manager = cast(MemoryManager, MagicMock())
        mock_logger = MagicMock()
        mock_config = MagicMock()

        tool = GetAllTool(mock_config, mock_memory_manager, mock_logger)
        snippet = tool.get_instruction_snippet()

        assert "get_all(" in snippet
        assert "dict" in snippet
        assert snippet.startswith("    •")

    def test_search_tool_snippet_format(
        self, mock_memory_class: MagicMock, set_env_vars: None
    ) -> None:
        """Test SearchTool provides properly formatted snippet."""
        from reflectlog.application.tools.search import SearchTool

        mock_memory_manager = cast(MemoryManager, MagicMock())
        mock_logger = MagicMock()
        mock_config = MagicMock()

        tool = SearchTool(mock_config, mock_memory_manager, mock_logger)
        snippet = tool.get_instruction_snippet()

        assert "search(query: str)" in snippet
        assert "semantic" in snippet.lower()
        assert snippet.startswith("    •")

    def test_remove_tool_snippet_format(
        self, mock_memory_class: MagicMock, set_env_vars: None
    ) -> None:
        """Test RemoveTool provides properly formatted snippet."""
        from reflectlog.application.tools.remove import RemoveTool

        mock_memory_manager = cast(MemoryManager, MagicMock())
        mock_logger = MagicMock()
        mock_config = MagicMock()

        tool = RemoveTool(mock_config, mock_memory_manager, mock_logger)
        snippet = tool.get_instruction_snippet()

        assert "remove(memories: list[str])" in snippet
        assert "exact" in snippet.lower()
        assert snippet.startswith("    •")


@pytest.mark.unit
class TestDynamicInstructionsIntegration:
    """Test suite for dynamic MCP instructions with FastMCPServer."""

    def _build_server(
        self, monkeypatch: MonkeyPatch, allowed_value: str | None
    ) -> FastMCPServer:
        monkeypatch.setenv("WORKSPACE_ID", "test_project")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")

        if allowed_value is None:
            monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
        else:
            monkeypatch.setenv("ALLOWED_TOOLS", allowed_value)

        with (
            patch("reflectlog.application.mcp_server.MemoryManager"),
            patch(
                "reflectlog.application.mcp_server.create_logger"
            ) as mock_create_logger,
        ):
            mock_create_logger.return_value = MagicMock()

            from reflectlog.application.config.settings import Config
            from reflectlog.application.mcp_server import FastMCPServer

            server_config = Config.from_environment()
            server = FastMCPServer(server_config=server_config)

        return server

    async def test_all_tools_require_workspace_when_no_restriction(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        server = self._build_server(monkeypatch, None)

        client_factory = cast("Callable[[object], Client]", Client)
        async with client_factory(server.mcp) as client:
            tools = await client.list_tools()

        assert {tool.name for tool in tools} == {
            "add",
            "get_all",
            "search",
            "remove",
            "health_check",
        }
        for tool in tools:
            assert isinstance(tool, _SchemaTool)
            assert "workspace_id" in tool.inputSchema["required"]

    async def test_only_allowed_tools_require_workspace(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        server = self._build_server(monkeypatch, "add,search")

        client_factory = cast("Callable[[object], Client]", Client)
        async with client_factory(server.mcp) as client:
            tools = await client.list_tools()

        assert {tool.name for tool in tools} == {"add", "search"}
        for tool in tools:
            assert isinstance(tool, _SchemaTool)
            assert "workspace_id" in tool.inputSchema["required"]

    def test_single_tool_selection(self, monkeypatch: MonkeyPatch) -> None:
        server = self._build_server(monkeypatch, "get_all")

        assert set(server.registered_tools) == {"get_all"}

    def test_no_tools_registered_when_none_enabled(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        server = self._build_server(monkeypatch, "none")

        assert server.registered_tools == {}

    def test_tool_alias_selection(self, monkeypatch: MonkeyPatch) -> None:
        server = self._build_server(monkeypatch, "Add-Tool")

        assert set(server.registered_tools) == {"add"}


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test backward compatibility of MCP_INSTRUCTIONS constant."""

    def test_mcp_instructions_constant_includes_all_tools(self):
        """MCP_INSTRUCTIONS constant should include all tools (backward compatibility)."""
        from reflectlog.core.prompts import MCP_INSTRUCTIONS

        assert "add(memories: list[str], dry_run: bool = False)" in MCP_INSTRUCTIONS
        assert "get_all(" in MCP_INSTRUCTIONS
        assert "-> dict" in MCP_INSTRUCTIONS
        assert "search(query: str) -> list[str]" in MCP_INSTRUCTIONS
        assert "remove(memories: list[str])" in MCP_INSTRUCTIONS

    def test_mcp_instructions_has_header(self):
        """MCP_INSTRUCTIONS should have proper header."""
        from reflectlog.core.prompts import MCP_INSTRUCTIONS

        assert "ReflectLog Server" in MCP_INSTRUCTIONS
        assert "Available Tools:" in MCP_INSTRUCTIONS
