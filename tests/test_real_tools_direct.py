"""Exercise real tools through HTTP; start the server with API credentials first."""

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
    reason="Requires live USearch and embedding service",
)


async def test_real_tools_workflow() -> None:
    workspace = os.getenv("WORKSPACE_ID", "ReflectLog")
    print(
        "Real tool workflow: OpenRouter/OpenAI credentials depend on server configuration"
    )
    async with Client("http://127.0.0.1:9103/mcp") as client:
        print("Tools:", [tool.name for tool in await client.list_tools()])
        added = parse_response(
            await client.call_tool(
                "add_memories", {"workspace_id": workspace, "memories": MEMORIES}
            ),
            AddResponse,
        )
        print("Added/skipped/replaced:", added.model_dump())
        records = await all_records(client, workspace)
        print("USearch-backed records:", [record.model_dump() for record in records])
        for query in ("Python", "JavaScript"):
            response = parse_response(
                await client.call_tool(
                    "search_memories", {"workspace_id": workspace, "query": query}
                ),
                SearchResponse,
            )
            print(query, response.model_dump())
        print("Real-service add, listing and hybrid search complete")


if __name__ == "__main__":
    anyio.run(test_real_tools_workflow)
