from collections.abc import Callable
from dataclasses import replace
import os
from pathlib import Path
from typing import cast
from unittest.mock import patch

from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import Tool as MCPTool
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.application.memory.workspace_registry import WorkspaceRegistry
from reflectlog.application.utils.security import SecretString


@pytest.mark.parametrize(
    ("arguments", "expected_page", "offset", "limit", "truncated"),
    [
        ({}, ["alpha", "beta"], 0, 2, True),
        ({"limit": None}, ["alpha", "beta"], 0, 2, True),
        ({"limit": 999}, ["alpha", "beta"], 0, 2, True),
        ({"limit": 1, "offset": 1}, ["beta"], 1, 1, True),
        ({"offset": 2}, ["zeta"], 2, 2, False),
        ({"limit": 0}, [], 0, 0, True),
        ({"limit": -1, "offset": -2}, [], 0, 0, True),
        ({"offset": -1}, ["alpha", "beta"], 0, 2, True),
        ({"offset": 9}, [], 9, 2, False),
    ],
)
async def test_mcp_workspace_page_uses_one_scan_without_allocating_managers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    set_env_vars: dict[str, str],
    arguments: dict[str, object],
    expected_page: list[str],
    offset: int,
    limit: int,
    truncated: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("zeta", "beta", "alpha"):
        usearch = tmp_path / "indexes" / name / "usearch"
        usearch.mkdir(parents=True)
        (usearch / "memories.db").write_bytes(b"metadata marker only")
    config = replace(
        Config.from_environment(),
        workspace_id="",
        get_all_limit=2,
        workspace_max_resident=1,
        allowed_tools=("get_all_workspaces",),
    )
    with (
        patch("reflectlog.application.mcp_server.MemoryManager") as factory,
        patch.object(WorkspaceRegistry, "acquire") as acquire,
        patch(
            "reflectlog.infrastructure.workspace_discovery.os.scandir",
            wraps=os.scandir,
        ) as scan,
    ):
        server = FastMCPServer(config)
        client_factory = cast("Callable[[object], Client]", Client)
        async with client_factory(server.mcp) as client:
            result = await client.call_tool(
                "get_all_workspaces",
                {"admin_password": set_env_vars["ADMIN_PASSWORD"], **arguments},
            )

        assert result.structured_content == {
            "workspaces": expected_page,
            "total": 3,
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
        }
        factory.assert_not_called()
        acquire.assert_not_called()
        assert sum(call.args == (Path("indexes"),) for call in scan.call_args_list) == 1


async def test_workspace_discovery_schema_requires_password_with_optional_pagination(
    set_env_vars: dict[str, str],
) -> None:
    server = FastMCPServer(
        replace(Config.from_environment(), allowed_tools=("get_all_workspaces",))
    )
    client_factory = cast("Callable[[object], Client]", Client)

    async with client_factory(server.mcp) as client:
        tools = cast("list[MCPTool]", await client.list_tools())

    assert [tool.name for tool in tools] == ["get_all_workspaces"]
    schema = tools[0].input_schema
    assert schema["required"] == ["admin_password"]
    assert set(schema["properties"]) == {"admin_password", "limit", "offset"}
    assert schema["properties"]["admin_password"]["type"] == "string"
    assert "default" not in schema["properties"]["admin_password"]
    assert schema["properties"]["limit"]["default"] is None
    assert {item["type"] for item in schema["properties"]["limit"]["anyOf"]} == {
        "integer",
        "null",
    }
    assert schema["properties"]["offset"]["type"] == "integer"
    assert schema["properties"]["offset"]["default"] == 0


async def test_workspace_discovery_with_missing_root_is_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, set_env_vars: dict[str, str]
) -> None:
    monkeypatch.chdir(tmp_path)
    server = FastMCPServer(replace(Config.from_environment(), allowed_tools=None))
    client_factory = cast("Callable[[object], Client]", Client)

    async with client_factory(server.mcp) as client:
        result = await client.call_tool(
            "get_all_workspaces",
            {"admin_password": set_env_vars["ADMIN_PASSWORD"], "limit": 0},
        )

    assert result.structured_content == {
        "workspaces": [],
        "total": 0,
        "offset": 0,
        "limit": 0,
        "truncated": False,
    }
    assert not (tmp_path / "indexes").exists()


async def test_workspace_discovery_io_failure_is_a_tool_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, set_env_vars: dict[str, str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "indexes").mkdir()
    server = FastMCPServer(
        replace(Config.from_environment(), allowed_tools=("get_all_workspaces",))
    )
    client_factory = cast("Callable[[object], Client]", Client)

    with patch(
        "reflectlog.infrastructure.workspace_discovery.os.scandir",
        side_effect=PermissionError("storage unavailable"),
    ):
        async with client_factory(server.mcp) as client:
            with pytest.raises(ToolError, match="storage unavailable"):
                await client.call_tool(
                    "get_all_workspaces",
                    {"admin_password": set_env_vars["ADMIN_PASSWORD"]},
                )


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"admin_password": "incorrect-secret"},
        {"admin_password": ""},
        {"admin_password": " test-admin-password "},
        {"admin_password": "\u5bc6\u78bc"},
    ],
)
async def test_workspace_discovery_rejects_unauthenticated_calls_before_discovery(
    set_env_vars: dict[str, str],
    arguments: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = FastMCPServer(
        replace(Config.from_environment(), allowed_tools=("get_all_workspaces",))
    )
    with patch("reflectlog.application.mcp_server.list_stored_workspaces") as discover:
        async with Client(server.mcp) as client:
            with pytest.raises(ToolError) as error:
                await client.call_tool("get_all_workspaces", arguments)

    discover.assert_not_called()
    assert set_env_vars["ADMIN_PASSWORD"] not in str(error.value)
    assert "incorrect-secret" not in str(error.value)
    assert set_env_vars["ADMIN_PASSWORD"] not in caplog.text
    assert "incorrect-secret" not in caplog.text
    if arguments:
        assert str(error.value) == "Unauthorized"


@pytest.mark.parametrize(
    "password",
    ["test-admin-password", "\u5bc6\u78bc-\u00e9-\U0001f512", "  exact password  "],
)
async def test_workspace_discovery_accepts_matching_password(
    set_env_vars: dict[str, str], password: str
) -> None:
    server = FastMCPServer(
        replace(Config.from_environment(), admin_password=SecretString(password))
    )
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "get_all_workspaces", {"admin_password": password}
        )

    assert result.structured_content == {
        "workspaces": [],
        "total": 0,
        "offset": 0,
        "limit": server.config.get_all_limit,
        "truncated": False,
    }
