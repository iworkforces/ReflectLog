"""Live MCP client demonstration with real embeddings and persistent storage."""

import os

import anyio
from fastmcp import Client
import pytest

from tests.test_get_all_memories import (
    AddResponse,
    SearchResponse,
    all_records,
    parse_response,
)
from tests.test_real_server import MEMORIES

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="Requires live server and API credentials",
)


async def test_real_server_with_mcp_client() -> None:
    workspace = os.getenv("WORKSPACE_ID", "ReflectLog")
    async with Client("http://127.0.0.1:9103/mcp") as client:
        tools = await client.list_tools()
        print("Connected via MCP:", [tool.name for tool in tools])
        response = parse_response(
            await client.call_tool(
                "add_memories", {"workspace_id": workspace, "memories": MEMORIES}
            ),
            AddResponse,
        )
        print("Storage summary:", response.model_dump())
        records = await all_records(client, workspace)
        for record in records:
            print(record.id, record.memory, record.created_at)
        for query in ("Python", "JavaScript"):
            result = parse_response(
                await client.call_tool(
                    "search_memories", {"workspace_id": workspace, "query": query}
                ),
                SearchResponse,
            )
            print("Hybrid search:", query, result.score_kind)
            for hit in result.results:
                print(hit.id, hit.score, hit.memory)
        print(f"MCP workflow complete: {len(records)} persistent records")


if __name__ == "__main__":
    anyio.run(test_real_server_with_mcp_client)
