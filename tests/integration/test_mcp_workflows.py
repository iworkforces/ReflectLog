"""Complete MCP workflows against real storage and deterministic embeddings."""

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from reflectlog.application.mcp_server import FastMCPServer
from tests.integration.test_memory_manager_usearch import (
    MockEmbedder,
    create_usearch_config,
)


class WorkflowEmbedder(MockEmbedder):
    """Orthogonal fixture topics make matching independent of random vectors."""

    def embed_query(self, text: str) -> list[float]:
        topics = ("python", "java", "@", "$", "你好", "😀")
        vector = [0.0] * self.dims
        for index, topic in enumerate(topics, start=1):
            if topic in text.lower():
                vector[index] = 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector


@pytest.fixture
async def mcp_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[FastMCPServer]:
    monkeypatch.chdir(tmp_path)
    config = replace(
        create_usearch_config(str(tmp_path)),
        enable_smart_replace=False,
        enable_rrf_fusion=True,
        fusion_ranking_threshold=0.025,
        search_score_threshold=0.1,
    )
    with patch(
        "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
        return_value=WorkflowEmbedder(),
    ):
        server = FastMCPServer(config)
        try:
            yield server
        finally:
            await server.aclose()


async def _add(server: FastMCPServer, memories: list[str]) -> None:
    await server._add_memories(memories, workspace_id=server.config.workspace_id)


async def _get_all(server: FastMCPServer) -> list[str]:
    page = await server._get_all_memories(workspace_id=server.config.workspace_id)
    rows = page["records"]
    assert isinstance(rows, list)
    memories: list[str] = []
    for row in rows:
        assert isinstance(row, dict)
        assert isinstance(row["memory"], str)
        memories.append(row["memory"])
    return memories


async def _search(server: FastMCPServer, query: str) -> list[str]:
    result = await server._search_memories(
        query, workspace_id=server.config.workspace_id
    )
    rows = result["results"]
    assert isinstance(rows, list)
    memories: list[str] = []
    for row in rows:
        assert isinstance(row, dict)
        assert isinstance(row["memory"], str)
        memories.append(row["memory"])
    return memories


async def _remove(server: FastMCPServer, memories: list[str]) -> None:
    async with server._registry.acquire(server.config.workspace_id) as manager:
        rows, _ = manager.get_page_records_with_total()
    ids = [row.id for row in rows if row.content in memories]
    assert len(ids) == len(memories)
    result = await server._delete_memories(ids, workspace_id=server.config.workspace_id)
    assert result == {"removed": ids, "missing": []}


@pytest.mark.integration
class TestMCPWorkflows:
    async def test_add_then_get_all_workflow(self, mcp_server: FastMCPServer) -> None:
        memories = ["Memory 1", "Memory 2", "Memory 3"]
        await _add(mcp_server, memories)
        result = await _get_all(mcp_server)
        assert len(result) == 3
        assert result == memories

    async def test_add_then_search_workflow(self, mcp_server: FastMCPServer) -> None:
        await _add(mcp_server, ["Python tutorial", "Java guide", "Python examples"])
        result = await _search(mcp_server, "Python")
        assert len(result) == 2
        assert "Python tutorial" in result
        assert "Python examples" in result
        assert "Java guide" not in result

    async def test_add_remove_get_all_workflow(self, mcp_server: FastMCPServer) -> None:
        await _add(mcp_server, ["Memory 1", "Memory 2", "Memory 3"])
        assert len(await _get_all(mcp_server)) == 3
        await _remove(mcp_server, ["Memory 2"])
        remaining = await _get_all(mcp_server)
        assert len(remaining) == 2
        assert "Memory 1" in remaining
        assert "Memory 3" in remaining
        assert "Memory 2" not in remaining

    async def test_add_search_remove_search_workflow(
        self, mcp_server: FastMCPServer
    ) -> None:
        await _add(
            mcp_server,
            [
                "Python tutorial for beginners",
                "Advanced Python techniques",
                "Java programming guide",
                "Python data science",
            ],
        )
        assert len(await _search(mcp_server, "Python")) == 3
        await _remove(mcp_server, ["Python tutorial for beginners"])
        remaining = await _search(mcp_server, "Python")
        assert len(remaining) == 2
        assert "Advanced Python techniques" in remaining
        assert "Python data science" in remaining
        assert "Python tutorial for beginners" not in remaining

    async def test_multiple_add_operations(self, mcp_server: FastMCPServer) -> None:
        await _add(mcp_server, ["Memory 1", "Memory 2"])
        await _add(mcp_server, ["Memory 3"])
        await _add(mcp_server, ["Memory 4", "Memory 5"])
        assert len(await _get_all(mcp_server)) == 5

    async def test_empty_store_operations(self, mcp_server: FastMCPServer) -> None:
        assert await _get_all(mcp_server) == []
        assert await _search(mcp_server, "anything") == []
        assert await mcp_server._delete_memories(
            [1], mcp_server.config.workspace_id
        ) == {"removed": [], "missing": [1]}

    async def test_add_duplicate_memories(self, mcp_server: FastMCPServer) -> None:
        await _add(
            mcp_server, ["Duplicate memory", "Unique memory", "Duplicate memory"]
        )
        assert len(await _get_all(mcp_server)) == 2
        await _remove(mcp_server, ["Duplicate memory"])
        remaining = await _get_all(mcp_server)
        assert len(remaining) == 1
        assert "Unique memory" in remaining

    async def test_search_with_no_semantic_matches(
        self, mcp_server: FastMCPServer
    ) -> None:
        assert await _search(mcp_server, "nonexistent query") == []

    async def test_large_dataset_workflow(self, mcp_server: FastMCPServer) -> None:
        await _add(mcp_server, [f"Memory {index}" for index in range(100)])
        assert len(await _get_all(mcp_server)) == 100
        assert len(await _search(mcp_server, "Memory")) <= 5

    async def test_special_characters_workflow(self, mcp_server: FastMCPServer) -> None:
        await _add(
            mcp_server,
            [
                "Email: user@example.com",
                "Price: $100.00",
                "Math: 2 + 2 = 4",
                "Code: function() { return true; }",
            ],
        )
        email_results = await _search(mcp_server, "@")
        price_results = await _search(mcp_server, "$")
        assert len(email_results) == 1
        assert "user@example.com" in email_results[0]
        assert len(price_results) == 1
        assert "$100.00" in price_results[0]

    async def test_unicode_workflow(self, mcp_server: FastMCPServer) -> None:
        await _add(
            mcp_server,
            [
                "Hello in Chinese: 你好",
                "Hello in Japanese: こんにちは",
                "Hello in Arabic: مرحبا",
                "Emoji: 😀 🌍 🚀",
            ],
        )
        assert len(await _search(mcp_server, "你好")) == 1
        assert len(await _search(mcp_server, "😀")) == 1
