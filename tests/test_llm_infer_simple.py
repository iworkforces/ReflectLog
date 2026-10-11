import os
from uuid import uuid4

import anyio
from fastmcp import Client
import pytest

from tests.test_get_all_memories import (
    AddResponse,
    DeleteResponse,
    all_records,
    parse_response,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="Requires server with ENABLE_LLM_INFER=false",
)


async def test_infer_disabled() -> None:
    workspace = f"infer-disabled-{uuid4().hex}"
    memory = "Python tutorial for beginners"
    print(
        "Run the live server with ENABLE_LLM_INFER=false and configured embedding credentials"
    )
    async with Client("http://127.0.0.1:9103/mcp") as client:
        try:
            added = parse_response(
                await client.call_tool(
                    "add_memories", {"workspace_id": workspace, "memories": [memory]}
                ),
                AddResponse,
            )
            print("Add summary:", added.model_dump())
            records = await all_records(client, workspace)
            assert [record.memory for record in records] == [memory]
            print(
                "Inference disabled: stored content is unchanged",
                records[0].model_dump(),
            )
        finally:
            records = await all_records(client, workspace)
            ids = [record.id for record in records if record.memory == memory]
            if ids:
                deleted = parse_response(
                    await client.call_tool(
                        "delete_memories", {"workspace_id": workspace, "ids": ids}
                    ),
                    DeleteResponse,
                )
                assert deleted.removed == ids and not deleted.missing


async def main() -> None:
    await test_infer_disabled()


if __name__ == "__main__":
    anyio.run(main)
