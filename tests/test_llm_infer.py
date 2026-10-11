import os
import subprocess
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
    reason="Requires live embedding/LLM credentials and starts a server",
)


async def infer_workflow(enabled: bool, port: int) -> None:
    workspace = f"TestLLMInfer-{uuid4().hex}"
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_LLM_INFER": str(enabled).lower(),
            "MCP_TRANSPORT": "http",
            "MCP_PORT": str(port),
            "LOG_LEVEL": "INFO",
            "WORKSPACE_ID": workspace,
        }
    )
    server = subprocess.Popen(
        [
            "./start-reflectlog-mcp-server.sh",
            "--workspace_id",
            workspace,
            "--transport",
            "http",
            "--port",
            str(port),
        ],
        env=env,
    )
    try:
        await anyio.sleep(3)
        async with Client(f"http://127.0.0.1:{port}/mcp") as client:
            memory = "Python tutorial"
            try:
                added = parse_response(
                    await client.call_tool(
                        "add_memories",
                        {"workspace_id": workspace, "memories": [memory]},
                    ),
                    AddResponse,
                )
                print("Add summary:", added.model_dump())
                records = await all_records(client, workspace)
                print("Stored records:", [record.model_dump() for record in records])
                if enabled:
                    print(
                        "Content transformed:",
                        any(record.memory != memory for record in records),
                    )
                else:
                    assert [record.memory for record in records] == [memory]
                    print("Inference disabled: content preserved exactly")
            finally:
                records = await all_records(client, workspace)
                ids = [record.id for record in records]
                if ids:
                    deleted = parse_response(
                        await client.call_tool(
                            "delete_memories", {"workspace_id": workspace, "ids": ids}
                        ),
                        DeleteResponse,
                    )
                    assert deleted.removed == ids and not deleted.missing
    finally:
        server.terminate()
        try:
            server.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=2)


async def test_with_infer_enabled() -> None:
    await infer_workflow(True, 9103)


async def test_with_infer_disabled() -> None:
    await infer_workflow(False, 9105)


async def main() -> None:
    await test_with_infer_disabled()
    print(
        "ENABLE_LLM_INFER=true requires an LLM API key and may incur costs; skipped for safety"
    )
    print("Run test_with_infer_enabled explicitly to compare the enabled behavior")


if __name__ == "__main__":
    anyio.run(main)
