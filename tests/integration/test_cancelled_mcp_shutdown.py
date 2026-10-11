import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
import threading
from typing import cast
from unittest.mock import patch

import anyio
from fastmcp import Client
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.application.utils.security import SecretString
from reflectlog.core.enums import EmbedderProvider, RerankerEngine
from reflectlog.infrastructure.embeddings.qwen3_embedding import LangchainQwenEmbeddings
from reflectlog.infrastructure.usearch_engine import USearchEngine


@pytest.fixture
def deterministic_embeddings() -> Iterator[None]:
    def vector(_embedder: LangchainQwenEmbeddings, _text: str) -> list[float]:
        return [1.0, *([0.0] * 127)]

    def vectors(
        embedder: LangchainQwenEmbeddings, texts: list[str]
    ) -> list[list[float]]:
        return [vector(embedder, text) for text in texts]

    async def async_vectors(
        embedder: LangchainQwenEmbeddings, texts: list[str]
    ) -> list[list[float]]:
        return vectors(embedder, texts)

    with (
        patch.object(LangchainQwenEmbeddings, "embed_documents", vectors),
        patch.object(LangchainQwenEmbeddings, "aembed_documents", async_vectors),
        patch.object(LangchainQwenEmbeddings, "embed_query", vector),
    ):
        yield


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", ["predispatch", "postwrite"])
async def test_cancelled_mcp_add_drains_before_close_and_reopens(
    tmp_path: Path, set_env_vars: None, deterministic_embeddings: None, barrier: str
) -> None:
    config = Config(
        workspace_id="",
        openrouter_api_key=SecretString("test-key"),
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="openai/text-embedding-3-small",
        embedding_dims=128,
        reranker_engine=RerankerEngine.NONE,
        enable_smart_replace=False,
        embedding_cache_enabled=False,
        eager_initialization=False,
        tantivy_index_path_template=str(tmp_path / "{workspace_id}" / "tantivy"),
    )
    server_tasks: list[asyncio.Task[object]] = []
    original_handler = FastMCPServer._add_memories
    original_add = USearchEngine.add_batch

    async def capture_handler(
        server: FastMCPServer,
        memories: list[str],
        workspace_id: str,
        dry_run: bool = False,
        allow_replace: bool = True,
    ) -> dict[str, object]:
        task = asyncio.current_task()
        assert task is not None
        server_tasks.append(task)
        return await original_handler(
            server, memories, workspace_id, dry_run, allow_replace
        )

    with patch.object(FastMCPServer, "_add_memories", capture_handler):
        server = FastMCPServer(config)
    client_factory = cast("Callable[[object], Client]", Client)
    workspace_id = "cancelled"
    memory = "cancelled worker durable outcome"
    entered = asyncio.Event()
    release_dispatch = asyncio.Event()
    write_entered = threading.Event()
    release_write = threading.Event()
    drain_waiting = asyncio.Event()
    ordering: list[str] = []
    invocation: asyncio.Task[object] | None = None
    closing: asyncio.Task[None] | None = None

    try:
        async with client_factory(server.mcp) as client:
            async with server._registry.acquire(workspace_id) as manager:
                pass

            original_acquire = server._registry._acquire_entry

            async def pause_before_dispatch(key: str) -> object:
                entry = await original_acquire(key)
                ordering.append("pin")
                entered.set()
                await release_dispatch.wait()
                return entry

            def pause_after_write(
                engine: USearchEngine,
                workspace_id: str,
                contents: list[str],
                infer: bool,
                vectors: list[list[float]] | None = None,
            ) -> list[str]:
                written = original_add(engine, workspace_id, contents, infer, vectors)
                ordering.append("write")
                write_entered.set()
                assert release_write.wait(timeout=10)
                return written

            registry_patch = (
                patch.object(server._registry, "_acquire_entry", pause_before_dispatch)
                if barrier == "predispatch"
                else patch.object(USearchEngine, "add_batch", pause_after_write)
            )
            original_close = manager.close
            close_count = 0

            def close_after_release() -> None:
                nonlocal close_count
                assert (
                    release_dispatch.is_set()
                    if barrier == "predispatch"
                    else release_write.is_set()
                )
                close_count += 1
                original_close()
                ordering.append("closed")

            with (
                registry_patch,
                patch.object(manager, "close", side_effect=close_after_release),
            ):
                try:
                    invocation = asyncio.create_task(
                        client.call_tool(
                            "add_memories",
                            {"workspace_id": workspace_id, "memories": [memory]},
                        )
                    )
                    if barrier == "predispatch":
                        await asyncio.wait_for(entered.wait(), 10)
                    else:
                        assert await asyncio.wait_for(
                            asyncio.to_thread(write_entered.wait, 10), 11
                        )
                    assert len(server_tasks) == 1
                    ordering.append("cancel")
                    _ = invocation.cancel()
                    _ = server_tasks[0].cancel()

                    drained = server._registry._drained
                    original_wait = type(drained).wait

                    async def observe_drain(event: anyio.Event) -> None:
                        if event is drained:
                            ordering.append("drain_wait")
                            drain_waiting.set()
                        await original_wait(event)

                    with patch.object(type(drained), "wait", observe_drain):
                        closing = asyncio.create_task(server.aclose())
                        await asyncio.wait_for(drain_waiting.wait(), 10)
                        assert not closing.done()
                        assert close_count == 0
                        ordering.append("release")
                        if barrier == "predispatch":
                            release_dispatch.set()
                        else:
                            release_write.set()
                        with pytest.raises(asyncio.CancelledError):
                            await asyncio.wait_for(invocation, 10)
                        await asyncio.wait_for(closing, 10)
                    assert close_count == 1
                finally:
                    release_dispatch.set()
                    release_write.set()
                    if invocation is not None:
                        _ = await asyncio.wait({invocation}, timeout=10)
                    if closing is not None:
                        _ = await asyncio.wait({closing}, timeout=10)
        reopened = FastMCPServer(config)
        try:
            async with client_factory(reopened.mcp) as client:
                page = await client.call_tool(
                    "get_all_memories", {"workspace_id": workspace_id}
                )
                result = await client.call_tool(
                    "search_memories", {"workspace_id": workspace_id, "query": memory}
                )
                expected = [memory] if barrier == "postwrite" else []
                assert page.structured_content is not None
                assert [
                    row["memory"] for row in page.structured_content["records"]
                ] == expected
                assert page.structured_content["total"] == len(expected)
                assert result.structured_content is not None
                assert [
                    row["memory"] for row in result.structured_content["results"]
                ] == expected
        finally:
            await asyncio.wait_for(reopened.aclose(), 10)
        assert ordering == (
            ["pin", "cancel", "drain_wait", "release", "closed"]
            if barrier == "predispatch"
            else ["write", "cancel", "drain_wait", "release", "closed"]
        )
    finally:
        release_dispatch.set()
        release_write.set()
        await asyncio.wait_for(server.aclose(), 10)
