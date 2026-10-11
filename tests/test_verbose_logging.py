import os

import anyio
from fastmcp import Client
import pytest

from tests.test_get_all_memories import (
    AddResponse,
    DeleteResponse,
    SearchResponse,
    all_records,
    parse_response,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1", reason="Requires live server"
)


async def test_verbose_logging() -> None:
    workspace = os.getenv("WORKSPACE_ID", "ReflectLog")
    memories = ["Test memory 1", "Test memory 2", "Test memory 3"]
    print(
        "Start the server with LOG_LEVEL=DEBUG; observe redacted operation metadata, not memory text"
    )
    async with Client("http://127.0.0.1:9103/mcp") as client:
        print(
            parse_response(
                await client.call_tool(
                    "add_memories",
                    {
                        "workspace_id": workspace,
                        "memories": memories,
                        "allow_replace": False,
                    },
                ),
                AddResponse,
            ).model_dump()
        )
        await anyio.sleep(1)
        records = await all_records(client, workspace)
        print("Listed", len(records), "records")
        await anyio.sleep(1)
        result = parse_response(
            await client.call_tool(
                "search_memories", {"workspace_id": workspace, "query": "Test"}
            ),
            SearchResponse,
        )
        print(
            "Search returned",
            len(result.results),
            "records with score kind",
            result.score_kind,
        )
        record = next(row for row in records if row.memory == memories[1])
        deleted = parse_response(
            await client.call_tool(
                "delete_memories", {"workspace_id": workspace, "ids": [record.id]}
            ),
            DeleteResponse,
        )
        print("Deletion summary:", deleted.model_dump())
    print(
        "Inspect invocation, counts, timing and completion metadata in the server logs"
    )


if __name__ == "__main__":
    anyio.run(test_verbose_logging)
