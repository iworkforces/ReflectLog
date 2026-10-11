from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import anyio
from fastmcp import Client
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.application.utils.security import SecretString
from reflectlog.core.enums import EmbedderProvider, RerankerEngine
from reflectlog.infrastructure.embeddings.qwen3_embedding import LangchainQwenEmbeddings
from tests.load.locustfile import run_benchmark


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


@pytest.fixture
def server_config(tmp_path: Path, set_env_vars: None) -> Config:
    return Config(
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_measures_real_mcp_tools_without_setup_calls(
    server_config: Config, deterministic_embeddings: None
) -> None:
    workspace_id = "benchmark-smoke"
    unrelated = "unrelated pre-existing memory"
    seed_server = FastMCPServer(server_config)
    try:
        async with Client(seed_server.mcp) as client:
            await client.call_tool(
                "add_memories", {"workspace_id": workspace_id, "memories": [unrelated]}
            )
    finally:
        await seed_server.aclose()
    server = FastMCPServer(server_config)
    try:
        report = await run_benchmark(server.mcp, workspace_id, clients=2, iterations=2)
        reopened = FastMCPServer(server_config)
        try:
            async with Client(reopened.mcp) as client:
                page = await client.call_tool(
                    "get_all_memories", {"workspace_id": workspace_id}
                )
        finally:
            await reopened.aclose()
        assert report["workspace_id"] == workspace_id
        assert len(report["connection_ms"]) == 2
        assert len(report["warmup_ms"]) == 2
        assert report["measured_seconds"] > 0
        assert report["requests_per_second"] > 0
        assert report["cleanup_errors"] == 0
        assert set(report["tools"]) == {
            "add_memories",
            "search_memories",
            "get_all_memories",
            "get_memory_by_id",
            "delete_memories",
        }
        assert all(
            metric["count"] == (8 if name == "get_all_memories" else 4)
            and metric["errors"] == 0
            and metric["mean_ms"] > 0
            and 0 < metric["p50_ms"] <= metric["p95_ms"]
            and metric["p95_ms"] > 0
            for name, metric in report["tools"].items()
        )
        assert page.structured_content is not None
        assert page.structured_content["total"] == 1
        records = page.structured_content["records"]
        assert isinstance(records, list)
        assert len(records) == 1
        assert isinstance(records[0], dict)
        assert records[0]["memory"] == unrelated
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_keeps_clients_open_until_every_cleanup_finishes(
    server_config: Config, deterministic_embeddings: None
) -> None:
    fast_removed = anyio.Event()
    release_slow = anyio.Event()
    client_closed = anyio.Event()
    original_exit = Client.__aexit__
    original_remove = FastMCPServer._delete_memories

    async def remove(
        server: FastMCPServer, ids: list[int], workspace_id: str
    ) -> dict[str, object]:
        async with server._registry.acquire(workspace_id) as manager:
            rows, _ = manager.get_records(ids)
        memories = [row.content for row in rows]
        if not any(memory.endswith(" seed") for memory in memories):
            return await original_remove(server, ids, workspace_id)
        if "worker 0 seed" in memories[0]:
            fast_removed.set()
        else:
            await fast_removed.wait()
            await release_slow.wait()
        return await original_remove(server, ids, workspace_id)

    async def tracked_exit(
        client: Client,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        try:
            await original_exit(client, exc_type, exc_value, traceback)
        finally:
            client_closed.set()

    async def observe_early_exit() -> None:
        nonlocal premature_exit
        await fast_removed.wait()
        with anyio.move_on_after(0.2):
            await client_closed.wait()
        premature_exit = client_closed.is_set()
        release_slow.set()

    premature_exit = False
    report = None
    with patch.object(FastMCPServer, "_delete_memories", remove):
        server = FastMCPServer(server_config)
    try:
        with patch.object(Client, "__aexit__", tracked_exit):
            async with anyio.create_task_group() as group:
                group.start_soon(observe_early_exit)
                report = await run_benchmark(
                    server.mcp, "benchmark-lifecycle", clients=2, iterations=1
                )
        assert report is not None
        assert not premature_exit
        assert report["cleanup_errors"] == 0
        assert all(metric["errors"] == 0 for metric in report["tools"].values())
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_counts_mcp_tool_errors(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def failed_lookup(
        _server: FastMCPServer, id: int, workspace_id: str
    ) -> dict[str, object]:
        raise RuntimeError("lookup unavailable")

    with patch.object(FastMCPServer, "_get_memory_by_id", failed_lookup):
        server = FastMCPServer(server_config)
    try:
        report = await run_benchmark(
            server.mcp, "benchmark-errors", clients=1, iterations=1
        )
        assert report["tools"]["get_memory_by_id"]["count"] == 1
        assert report["tools"]["get_memory_by_id"]["errors"] == 1
        assert report["cleanup_errors"] == 0
        assert all(
            report["tools"][name]["errors"] == 0
            for name in (
                "add_memories",
                "search_memories",
                "get_all_memories",
                "delete_memories",
            )
        )
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_counts_timed_out_tool_and_cleans_up(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def slow_lookup(
        _server: FastMCPServer, id: int, workspace_id: str
    ) -> dict[str, object]:
        await anyio.sleep(1)
        return {"workspace_id": workspace_id}

    with patch.object(FastMCPServer, "_get_memory_by_id", slow_lookup):
        server = FastMCPServer(server_config)
    workspace_id = "benchmark-timeout"
    try:
        report = await run_benchmark(
            server.mcp, workspace_id, clients=1, iterations=1, timeout_seconds=0.2
        )
        reopened = FastMCPServer(server_config)
        try:
            async with Client(reopened.mcp) as client:
                page = await client.call_tool(
                    "get_all_memories", {"workspace_id": workspace_id}
                )
        finally:
            await reopened.aclose()
        assert report["tools"]["get_memory_by_id"]["errors"] == 1
        assert report["tools"]["get_memory_by_id"]["count"] == 1
        assert report["cleanup_errors"] == 0
        assert page.structured_content is not None
        assert page.structured_content["total"] == 0
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_reports_cleanup_tool_error(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def failed_remove(
        _server: FastMCPServer, ids: list[int], workspace_id: str
    ) -> dict[str, object]:
        raise RuntimeError("remove unavailable")

    with patch.object(FastMCPServer, "_delete_memories", failed_remove):
        server = FastMCPServer(server_config)
    try:
        report = await run_benchmark(
            server.mcp, "benchmark-cleanup-error", clients=1, iterations=1
        )
        assert report["cleanup_errors"] == 1
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_bounds_cleanup_tool_call(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def slow_remove(
        server: FastMCPServer, ids: list[int], workspace_id: str
    ) -> dict[str, object]:
        async with server._registry.acquire(workspace_id) as manager:
            rows, _ = manager.get_records(ids)
        if any(row.content.endswith(" seed") for row in rows):
            await anyio.sleep(1)
        return await original_delete(server, ids, workspace_id)

    original_delete = FastMCPServer._delete_memories
    with patch.object(FastMCPServer, "_delete_memories", slow_remove):
        server = FastMCPServer(server_config)
    try:
        report = await run_benchmark(
            server.mcp,
            "benchmark-slow-cleanup",
            clients=1,
            iterations=1,
            timeout_seconds=0.3,
        )
        assert report["cleanup_errors"] == 1
        assert report["measured_seconds"] < 0.3
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_cleans_seed_after_warmup_error(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def slow_search(
        _server: FastMCPServer, query: str, workspace_id: str, limit: int | None = None
    ) -> dict[str, object]:
        await anyio.sleep(1)
        return {"results": [], "score_kind": "semantic", "limit": limit or 5}

    with patch.object(FastMCPServer, "_search_memories", slow_search):
        server = FastMCPServer(server_config)
    workspace_id = "benchmark-warmup-timeout"
    try:
        with pytest.raises(ExceptionGroup) as failure:
            await run_benchmark(
                server.mcp,
                workspace_id,
                clients=1,
                iterations=1,
                timeout_seconds=0.3,
            )
        assert any(
            isinstance(error, TimeoutError) for error in failure.value.exceptions
        )
        reopened = FastMCPServer(server_config)
        try:
            async with Client(reopened.mcp) as client:
                page = await client.call_tool(
                    "get_all_memories", {"workspace_id": workspace_id}
                )
        finally:
            await reopened.aclose()
        assert page.structured_content is not None
        assert page.structured_content["total"] == 0
    finally:
        await server.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_benchmark_reports_cleanup_error_when_warmup_fails(
    server_config: Config, deterministic_embeddings: None
) -> None:
    async def failed_search(
        _server: FastMCPServer, query: str, workspace_id: str, limit: int | None = None
    ) -> dict[str, object]:
        raise RuntimeError("warmup search unavailable")

    async def failed_remove(
        _server: FastMCPServer, ids: list[int], workspace_id: str
    ) -> dict[str, object]:
        raise RuntimeError("remove unavailable")

    with (
        patch.object(FastMCPServer, "_search_memories", failed_search),
        patch.object(FastMCPServer, "_delete_memories", failed_remove),
    ):
        server = FastMCPServer(server_config)
    try:
        with pytest.raises(ExceptionGroup) as failure:
            await run_benchmark(
                server.mcp, "benchmark-warmup-cleanup-error", clients=2, iterations=1
            )
        assert failure.value.__notes__ == ["benchmark cleanup_errors=2"]
        assert any(
            isinstance(error, RuntimeError)
            and "MCP tool returned an error" in str(error)
            for error in failure.value.exceptions
        )
    finally:
        await server.aclose()
