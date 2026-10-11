"""Live HTTP workflow; requires the configured embedding service and server."""

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

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="Requires live server and API credentials",
)

MEMORIES = [
    "Python tutorial for beginners",
    "Advanced Python programming",
    "JavaScript fundamentals",
    "Python data science guide",
    "Java enterprise patterns",
    "Python machine learning",
    "React frontend development",
    "Python web scraping",
    "Docker containerization",
    "Python automation scripts",
]


async def test_real_server_workflow() -> None:
    workspace = os.getenv("WORKSPACE_ID", "ReflectLog")
    async with Client("http://127.0.0.1:9103/mcp") as client:
        print("Available tools:", [tool.name for tool in await client.list_tools()])
        added = parse_response(
            await client.call_tool(
                "add_memories", {"workspace_id": workspace, "memories": MEMORIES}
            ),
            AddResponse,
        )
        print("Add summary:", added.model_dump())
        records = await all_records(client, workspace)
        print("Stored records:", [record.model_dump() for record in records])
        for query in ("Python", "JavaScript"):
            searched = parse_response(
                await client.call_tool(
                    "search_memories",
                    {"workspace_id": workspace, "query": query, "limit": 10},
                ),
                SearchResponse,
            )
            print(query, searched.model_dump())
        print("Real embeddings, persistent USearch storage and HTTP workflow complete")


if __name__ == "__main__":
    anyio.run(test_real_server_workflow)
