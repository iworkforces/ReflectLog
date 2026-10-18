from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import Tool as MCPTool
import pytest

from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.application.memory.add_phases import (
    ReplacementInfo,
    SmartReplacementPhase,
)
from tests.integration.test_memory_manager_usearch import (
    MockEmbedder,
    create_usearch_config,
)


@pytest.mark.integration
async def test_successor_workflow_preserves_missing_and_live_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = create_usearch_config(str(tmp_path))
    workspace = config.workspace_id
    with patch(
        "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
        return_value=MockEmbedder(),
    ):
        server = FastMCPServer(config)
        try:
            await server._add_memories(
                ["old convention"], workspace, allow_replace=False
            )
            async with server._registry.acquire(workspace) as manager:
                original = manager.get_page_records_with_total()[0][0]
                await manager._storage_phase.execute(
                    ["new convention"],
                    {
                        "new convention": [
                            ReplacementInfo(
                                old_memory="old convention",
                                new_memory="new convention",
                                confidence=0.95,
                                reason="private reason",
                                similarity_score=0.9,
                            )
                        ]
                    },
                )
                successor = manager.get_page_records_with_total()[0][0]
            assert await server._get_memory_by_id(original.id, workspace) == {
                "record": None
            }
            assert await server._get_memory_by_id(original.id, workspace, True) == {
                "record": None,
                "successor": {"id": successor.id},
            }
            live = await server._get_memory_by_id(successor.id, workspace)
            assert await server._get_memory_by_id(successor.id, workspace, True) == live
            assert await server._get_memory_by_id(
                original.id, "foreign-workspace", True
            ) == {
                "record": None,
                "successor": None,
            }
            await server._delete_memories([successor.id], workspace)
            assert await server._get_memory_by_id(original.id, workspace, True) == {
                "record": None,
                "successor": None,
            }
            await server._add_memories(
                ["new convention"], workspace, allow_replace=False
            )
            async with server._registry.acquire(workspace) as manager:
                readded = manager.get_page_records_with_total()[0][0]
            assert readded.id != successor.id
            assert await server._get_memory_by_id(original.id, workspace, True) == {
                "record": None,
                "successor": {"id": readded.id},
            }
        finally:
            await server.aclose()


@pytest.mark.integration
async def test_records_workflow_real_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = replace(
        create_usearch_config(str(tmp_path)),
        enable_smart_replace=True,
        search_score_threshold=0.0,
    )
    workspace = config.workspace_id
    with patch(
        "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
        return_value=MockEmbedder(),
    ):
        server = FastMCPServer(config)
        try:
            with patch.object(
                SmartReplacementPhase,
                "execute",
                AsyncMock(side_effect=AssertionError("must not replace")),
            ) as replace_phase:
                added = await server._add_memories(
                    [
                        "Python astronomy notebook",
                        "Python gardening notebook",
                        "Python cooking notebook",
                    ],
                    workspace,
                    allow_replace=False,
                )
                assert added["stored_count"] == 3
                assert added["replaced_count"] == 0
                results = await server._search_memories("Python", workspace, limit=3)
                hits = results["results"]
                assert isinstance(hits, list)
                assert hits
                ids: list[int] = []
                for hit in hits:
                    assert isinstance(hit, dict)
                    assert isinstance(hit["id"], int)
                    assert isinstance(hit["score"], float)
                    ids.append(hit["id"])
                assert results["score_kind"] in {"rrf", "semantic", "fulltext"}
                assert results["limit"] == 3
                for hit, id in zip(hits, ids, strict=True):
                    resolved = await server._get_memory_by_id(id, workspace)
                    record = resolved["record"]
                    assert isinstance(record, dict)
                    assert record["id"] == id
                    assert record["memory"] == hit["memory"]
                missing_id = max(ids) + 1
                assert await server._get_memory_by_id(missing_id, workspace) == {
                    "record": None
                }
                assert await server._get_memory_by_id(ids[0], "foreign-workspace") == {
                    "record": None
                }
                empty_page = await server._get_all_memories(workspace, limit=0)
                assert empty_page["records"] == []
                assert empty_page["total"] == 3
                page = await server._get_all_memories(workspace, limit=2)
                assert page["total"] == 3
                assert page["truncated"] is True
                assert await server._delete_memories(
                    [ids[0], missing_id], workspace
                ) == {
                    "removed": [ids[0]],
                    "missing": [missing_id],
                }
                assert await server._get_memory_by_id(ids[0], workspace) == {
                    "record": None,
                }
                again = await server._add_memories(
                    ["Python astronomy notebook updated"],
                    workspace,
                    allow_replace=False,
                )
                assert again["stored_count"] == 1
                assert again["replaced_count"] == 0
                replace_phase.assert_not_awaited()
        finally:
            await server._registry.close()


@pytest.mark.integration
async def test_cursor_survives_mutation_eviction_and_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = replace(create_usearch_config(str(tmp_path)), workspace_max_resident=1)
    workspace = config.workspace_id
    with patch(
        "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
        return_value=MockEmbedder(),
    ):
        server = FastMCPServer(config)
        try:
            await server._add_memories(
                ["old", "retained", "last"], workspace, allow_replace=False
            )
            first = await server._get_all_memories(workspace, limit=2, after_id=0)
            cursor = first["next_after_id"]
            assert isinstance(cursor, int)
            first_rows = first["records"]
            assert isinstance(first_rows, list)
            await server._delete_memories([first_rows[0]["id"]], workspace)
            async with server._registry.acquire(workspace) as manager:
                await manager._storage_phase.execute(
                    ["replacement"],
                    {
                        "replacement": [
                            ReplacementInfo(
                                old_memory="retained",
                                new_memory="replacement",
                                confidence=0.95,
                                reason="test",
                                similarity_score=0.9,
                            )
                        ]
                    },
                )
            foreign = await server._get_all_memories("foreign-workspace", after_id=0)
            assert foreign["records"] == []
            assert foreign["next_after_id"] == 0
            resumed = await server._get_all_memories(
                workspace, limit=1, after_id=cursor
            )
            rows = resumed["records"]
            assert isinstance(rows, list)
            assert rows[0]["memory"] == "last"
            cursor = resumed["next_after_id"]
            assert isinstance(cursor, int)
        finally:
            await server.aclose()
        restarted = FastMCPServer(config)
        try:
            probe = await restarted._get_all_memories(
                workspace, limit=0, after_id=cursor
            )
            assert probe["records"] == []
            assert probe["total"] == 2
            assert probe["next_after_id"] == cursor
            assert probe["truncated"] is True
            tail = await restarted._get_all_memories(workspace, after_id=cursor)
            rows = tail["records"]
            assert isinstance(rows, list)
            assert [row["memory"] for row in rows] == ["replacement"]
            assert tail["truncated"] is False
        finally:
            await restarted.aclose()


@pytest.mark.integration
async def test_registered_cursor_schema_and_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = create_usearch_config(str(tmp_path))
    with patch(
        "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
        return_value=MockEmbedder(),
    ):
        server = FastMCPServer(config)
        try:
            await server._add_memories(
                ["one", "two"], config.workspace_id, allow_replace=False
            )
            async with Client(server.mcp) as client:
                tools = await client.list_tools()
                tool = next(tool for tool in tools if tool.name == "get_all_memories")
                assert isinstance(tool, MCPTool)
                schema = tool.input_schema
                assert "after_id" not in schema["required"]
                integer = next(
                    branch
                    for branch in schema["properties"]["after_id"]["anyOf"]
                    if branch["type"] == "integer"
                )
                assert integer["minimum"] == 0
                page = await client.call_tool(
                    "get_all_memories",
                    {"workspace_id": config.workspace_id, "after_id": 0, "limit": 1},
                )
                assert page.structured_content is not None
                cursor = page.structured_content["next_after_id"]
                tail = await client.call_tool(
                    "get_all_memories",
                    {"workspace_id": config.workspace_id, "after_id": cursor},
                )
                assert tail.structured_content is not None
                assert tail.structured_content["truncated"] is False
                records = tail.structured_content["records"]
                assert isinstance(records, list)
                assert len(records) == 1
                with pytest.raises(ToolError):
                    await client.call_tool(
                        "get_all_memories",
                        {
                            "workspace_id": config.workspace_id,
                            "after_id": 0,
                            "offset": 1,
                        },
                    )
        finally:
            await server.aclose()
