from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

from fastmcp import Client
from mcp.types import Tool as MCPTool
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.application.utils.validation import validate_memories
from reflectlog.core.enums import EmbedderProvider, RerankerEngine


@pytest.mark.integration
async def test_installed_surface_discovery_constraints_and_missing_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, set_env_vars: dict[str, str]
) -> None:
    monkeypatch.chdir(tmp_path)
    config = replace(
        Config.from_environment(),
        workspace_id="surface",
        allowed_tools=None,
        embedder_provider=EmbedderProvider.LANGCHAIN,
        embedding_model="qwen/qwen3-embedding-8b",
        embedding_dims=4,
        qwen_embedding_dims=4,
        embedding_cache_enabled=False,
        eager_initialization=False,
        enable_smart_replace=False,
        reranker_engine=RerankerEngine.NONE,
    )
    server = FastMCPServer(config)
    try:
        client_factory = cast("Callable[[object], Client]", Client)
        async with client_factory(server.mcp) as client:
            tools = await client.list_tools()
            expected = [
                "add_memories",
                "search_memories",
                "get_all_memories",
                "get_memory_by_id",
                "delete_memories",
            ]
            assert [tool.name for tool in tools] == expected
            schemas = {tool.name: cast(MCPTool, tool).input_schema for tool in tools}
            for schema in schemas.values():
                assert "workspace_id" in schema["required"]
            search = schemas["search_memories"]["properties"]
            assert search["query"]["minLength"] == 1
            assert search["query"]["maxLength"] == 1000
            limit = next(
                branch
                for branch in search["limit"]["anyOf"]
                if branch["type"] == "integer"
            )
            assert limit["minimum"] == 1
            assert limit["maximum"] == 50
            assert schemas["get_memory_by_id"]["properties"]["id"]["minimum"] == 1
            ids = schemas["delete_memories"]["properties"]["ids"]
            assert ids["items"]["minimum"] == 1
            assert ids["maxItems"] == 100
            add = schemas["add_memories"]["properties"]
            assert add["dry_run"]["default"] is False
            assert add["allow_replace"]["default"] is True
            result = await client.call_tool(
                "get_memory_by_id", {"id": 999, "workspace_id": "surface"}
            )
            assert result.structured_content == {"record": None}
    finally:
        await server.aclose()


def test_malformed_memories_rejected_by_validator() -> None:
    ok, error = validate_memories(["ok"], 1, 100)
    assert ok is True
    assert error is None
    ok_bad, error_bad = validate_memories([None], 1, 100)
    assert ok_bad is False
    assert error_bad is not None
