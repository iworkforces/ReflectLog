from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.application.memory.add_phases import SmartReplacementPhase
from tests.integration.test_memory_manager_usearch import (
    MockEmbedder,
    create_usearch_config,
)


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
