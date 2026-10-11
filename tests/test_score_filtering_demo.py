import os
from uuid import uuid4

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
from tests.test_real_server_fastmcp import MEMORIES

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="Requires live embedding and reranking services",
)


async def test_score_filtering_demo() -> None:
    workspace = f"score-demo-{uuid4().hex}"
    print(
        "Score filtering demo: configure the live server's reranker and threshold first"
    )
    async with Client("http://127.0.0.1:9103/mcp") as client:
        try:
            added = parse_response(
                await client.call_tool(
                    "add_memories",
                    {
                        "workspace_id": workspace,
                        "memories": MEMORIES,
                        "allow_replace": False,
                    },
                ),
                AddResponse,
            )
            print("Add summary:", added.model_dump())
            for query in ("Python machine learning", "Python programming"):
                response = parse_response(
                    await client.call_tool(
                        "search_memories",
                        {"workspace_id": workspace, "query": query, "limit": 10},
                    ),
                    SearchResponse,
                )
                print(
                    query,
                    "score_kind:",
                    response.score_kind,
                    "returned:",
                    len(response.results),
                )
                for hit in response.results:
                    print(hit.id, hit.score, hit.memory)
            print(
                "Scores are interpreted by score_kind; no universal 0.8 cutoff applies to all pipelines"
            )
            print("Configured reranker filtering determines which final hits survive")
        finally:
            records = await all_records(client, workspace)
            ids = [record.id for record in records if record.memory in MEMORIES]
            if ids:
                deleted = parse_response(
                    await client.call_tool(
                        "delete_memories", {"workspace_id": workspace, "ids": ids}
                    ),
                    DeleteResponse,
                )
                assert set(deleted.removed) == set(ids) and not deleted.missing
                print("Cleaned demo records:", len(deleted.removed))


if __name__ == "__main__":
    anyio.run(test_score_filtering_demo)
