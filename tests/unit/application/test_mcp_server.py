import asyncio
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from inspect import signature
import json
from threading import Event, get_ident
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import Tool as MCPTool
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.mcp_server import FastMCPServer, main
from reflectlog.application.memory.manager import (
    AddResult,
    MemoryManager,
    RecordSearchResult,
    ScoredMemoryRecord,
)
from reflectlog.core.enums import SearchScoreKind
from reflectlog.core.exceptions import InconsistentStateError, SearchError, StorageError
from reflectlog.infrastructure.memory_store import MemoryRecord

NAMES = [
    "add_memories",
    "search_memories",
    "get_all_memories",
    "get_memory_by_id",
    "delete_memories",
]
OLD_NAMES = {
    "add",
    "get_all",
    "search",
    "remove",
    "health_check",
    "search_records",
    "get_records",
    "list_records",
    "remove_records",
}

CAPACITY_MESSAGE = (
    "Resident workspace limit reached; no idle workspace could be released"
)
TOOL_ARGUMENTS: dict[str, dict[str, object]] = {
    "add_memories": {"memories": ["memory"]},
    "search_memories": {"query": "memory"},
    "get_all_memories": {},
    "get_memory_by_id": {"id": 1},
    "delete_memories": {"ids": [1]},
}


def capacity_manager(config: Config, _logger: object) -> MagicMock:
    manager = MagicMock(spec=MemoryManager)
    manager.config = config
    manager.add_memories_async = AsyncMock(return_value=AddResult(1, 0, 0, []))
    manager.search_records = AsyncMock(return_value=search_result(["memory"]))
    manager.get_page_records_with_total.return_value = (rows(["memory"]), 1)
    manager.get_records.return_value = (rows(["memory"]), [])
    manager.delete_records.return_value = ([1], [])
    return manager


@pytest.mark.parametrize("tool_name", NAMES)
async def test_default_resident_capacity_is_unlimited(
    set_env_vars, tool_name: str
) -> None:
    entered = anyio.Event()
    release = anyio.Event()
    active: set[str] = set()

    def create_manager(config: Config, logger: object) -> MagicMock:
        manager = capacity_manager(config, logger)

        async def search(query: str, *, limit: int) -> RecordSearchResult:
            if query == "hold":
                active.add(config.workspace_id)
                if len(active) == 12:
                    entered.set()
                await release.wait()
            return search_result(["memory"])

        manager.search_records = AsyncMock(side_effect=search)
        return manager

    with patch(
        "reflectlog.application.mcp_server.MemoryManager", side_effect=create_manager
    ) as factory:
        server = FastMCPServer(replace(Config.from_environment(), workspace_id=""))
        assert server.config.workspace_max_resident is None
        client_factory = cast("Callable[[object], Client]", Client)
        async with client_factory(server.mcp) as client:
            invocations = [
                asyncio.create_task(
                    client.call_tool(
                        "search_memories",
                        {"workspace_id": f"workspace-{index}", "query": "hold"},
                    )
                )
                for index in range(12)
            ]
            try:
                with anyio.fail_after(5):
                    await entered.wait()
                for index in range(12):
                    result = await client.call_tool(
                        tool_name,
                        {
                            "workspace_id": f"workspace-{index}",
                            **TOOL_ARGUMENTS[tool_name],
                        },
                    )
                    assert not result.is_error
                assert factory.call_count == 12
            finally:
                release.set()
                with anyio.fail_after(5):
                    await asyncio.gather(*invocations)


@pytest.mark.parametrize("tool_name", NAMES)
async def test_resident_capacity_rejects_active_then_evicts_idle(
    set_env_vars, tool_name: str
) -> None:
    entered = anyio.Event()
    release = anyio.Event()
    ordering: list[str] = []
    managers: dict[str, MagicMock] = {}

    def create_manager(config: Config, logger: object) -> MagicMock:
        ordering.append(f"create:{config.workspace_id}")
        manager = capacity_manager(config, logger)
        manager.close.side_effect = lambda: ordering.append(
            f"close:{config.workspace_id}"
        )
        if config.workspace_id == "private-alpha":

            async def search(_query: str, *, limit: int) -> RecordSearchResult:
                entered.set()
                await release.wait()
                return search_result(["memory"])

            manager.search_records = AsyncMock(side_effect=search)
        managers[config.workspace_id] = manager
        return manager

    with patch(
        "reflectlog.application.mcp_server.MemoryManager", side_effect=create_manager
    ) as factory:
        server = FastMCPServer(
            replace(
                Config.from_environment(), workspace_id="", workspace_max_resident=1
            )
        )
        client_factory = cast("Callable[[object], Client]", Client)
        async with client_factory(server.mcp) as client:
            invocation = asyncio.create_task(
                client.call_tool(
                    "search_memories",
                    {"workspace_id": "private-alpha", "query": "memory"},
                )
            )
            try:
                with anyio.fail_after(5):
                    await entered.wait()
                with pytest.raises(ToolError, match=CAPACITY_MESSAGE) as raised:
                    await client.call_tool(
                        tool_name,
                        {"workspace_id": "private-beta", **TOOL_ARGUMENTS[tool_name]},
                    )
                assert "private-alpha" not in str(raised.value)
                assert "private-beta" not in str(raised.value)
                assert factory.call_count == 1
                managers["private-alpha"].close.assert_not_called()
            finally:
                release.set()
                with anyio.fail_after(5):
                    await invocation
            result = await client.call_tool(
                tool_name,
                {"workspace_id": "private-beta", **TOOL_ARGUMENTS[tool_name]},
            )
            assert not result.is_error
            assert ordering == [
                "create:private-alpha",
                "close:private-alpha",
                "create:private-beta",
            ]
            assert factory.call_count == 2


async def test_resident_capacity_reuses_canonical_workspace(set_env_vars) -> None:
    with patch(
        "reflectlog.application.mcp_server.MemoryManager", side_effect=capacity_manager
    ) as factory:
        server = FastMCPServer(
            replace(
                Config.from_environment(), workspace_id="", workspace_max_resident=1
            )
        )
        client_factory = cast("Callable[[object], Client]", Client)
        async with client_factory(server.mcp) as client:
            for workspace in ("ReflectLog", "  REFLECTLOG \t", "reflectlog"):
                for name in NAMES:
                    result = await client.call_tool(
                        name, {"workspace_id": workspace, **TOOL_ARGUMENTS[name]}
                    )
                    assert not result.is_error
            assert factory.call_count == 1
            assert factory.call_args.args[0].workspace_id == "reflectlog"


async def test_resident_capacity_does_not_change_tool_schemas(set_env_vars) -> None:
    config = replace(Config.from_environment(), workspace_id="")
    schemas: list[dict[str, dict[str, object]]] = []
    client_factory = cast("Callable[[object], Client]", Client)
    for maximum in (None, 1):
        server = FastMCPServer(replace(config, workspace_max_resident=maximum))
        async with client_factory(server.mcp) as client:
            tools = cast("list[MCPTool]", await client.list_tools())
            assert [tool.name for tool in tools] == NAMES
            schemas.append({tool.name: tool.input_schema for tool in tools})
    assert schemas[0] == schemas[1]
    for schema in schemas[1].values():
        assert "capacity" not in json.dumps(schema).lower()
        assert "max_resident" not in json.dumps(schema)


class _TestServer(FastMCPServer):
    memory_manager: MagicMock


def rows(memories: list[str], workspace: str = "test_project") -> list[MemoryRecord]:
    return [MemoryRecord(i, workspace, text, "") for i, text in enumerate(memories, 1)]


def payload(memories: list[str]) -> list[dict[str, object]]:
    return [
        {"id": i, "memory": text, "created_at": None}
        for i, text in enumerate(memories, 1)
    ]


def search_result(memories: list[str]) -> RecordSearchResult:
    return RecordSearchResult(
        tuple(
            ScoredMemoryRecord(i, text, 0.5, None) for i, text in enumerate(memories, 1)
        ),
        SearchScoreKind.RRF if memories else SearchScoreKind.NONE,
    )


@pytest.fixture
def mcp_server(set_env_vars, mock_usearch_engine):
    with (
        patch("reflectlog.application.memory.manager.USearchEngine") as usearch,
        patch("reflectlog.application.memory.manager.TantivyEngine") as tantivy,
        patch(
            "reflectlog.application.memory.manager.LangchainQwenEmbeddings"
        ) as embedder,
        patch("reflectlog.application.memory.manager.CachedEmbeddings") as cached,
    ):
        usearch.return_value = mock_usearch_engine
        tantivy.return_value = MagicMock()
        tantivy.return_value.ensure_initialized.return_value = None
        tantivy.return_value.search.return_value = []
        tantivy.return_value.is_ready.return_value = False
        tantivy.return_value.find_by_exact_match.return_value = []
        tantivy.return_value.delete.return_value = True
        tantivy.return_value.delete_batch.side_effect = (
            lambda _workspace, contents, verify_exists=True: len(contents)
        )
        embedding = MagicMock()
        embedding.embed_documents.side_effect = lambda texts: [[0.1] * 4 for _ in texts]
        embedding.embed_query.side_effect = lambda _query: [0.1] * 4
        embedder.return_value = embedding
        cached.return_value = MagicMock()
        cached.return_value.embedder = embedding
        cached.return_value.embed_documents.side_effect = embedding.embed_documents
        cached.return_value.embed_query.side_effect = embedding.embed_query
        server = _TestServer(
            replace(Config.from_environment(), workspace_id="test_project")
        )

        async def manager() -> MemoryManager:
            async with server._registry.acquire(server.config.workspace_id) as active:
                return active

        server.memory_manager = cast("MagicMock", anyio.run(manager))
        server.memory_manager.memory = mock_usearch_engine
        for tool in server.registered_tools.values():
            tool.fn = partial(tool.fn, workspace_id=server.config.workspace_id)
        yield server
        anyio.run(server.aclose)


def test_initialization(mcp_server) -> None:
    assert mcp_server.mcp is not None
    assert mcp_server.memory_manager.memory is not None
    assert list(mcp_server.registered_tools) == NAMES
    assert not OLD_NAMES.intersection(mcp_server.registered_tools)


def test_environment_workspace_remains_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORKSPACE_ID", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")
    assert Config.from_environment().workspace_id == ""


@pytest.mark.parametrize("cancellation", ["task", "anyio"])
async def test_cancelled_add_keeps_manager_until_embedding_finishes(
    mcp_server, cancellation: str
):
    manager = mcp_server.memory_manager
    entered = Event()
    release = Event()
    finished = Event()
    close_waiting = anyio.Event()
    scope_ready = anyio.Event()
    scopes: list[anyio.CancelScope] = []

    def embed(texts: list[str]) -> list[list[float]]:
        entered.set()
        assert release.wait(5)
        finished.set()
        return [[0.1] * 4 for _ in texts]

    async def aembed(texts: list[str]) -> list[list[float]]:
        return await anyio.to_thread.run_sync(embed, texts)

    async def call_in_scope() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            scope_ready.set()
            await mcp_server.tool_fn("add_memories")(["cancellation regression"])

    with (
        patch.object(manager.memory.memory_store, "exists_many", return_value=set()),
        patch.object(
            manager._semantic_engine.embedder, "aembed_documents", side_effect=aembed
        ),
        patch.object(manager, "close", wraps=manager.close) as close,
    ):
        invocation: asyncio.Task[object] | None = None
        closing: asyncio.Task[None] | None = None
        try:
            invocation = asyncio.create_task(
                call_in_scope()
                if cancellation == "anyio"
                else mcp_server.tool_fn("add_memories")(["cancellation regression"])
            )
            with anyio.fail_after(4):
                assert await anyio.to_thread.run_sync(entered.wait, 3)
            if cancellation == "anyio":
                await scope_ready.wait()
                scopes[0].cancel()
            else:
                _ = invocation.cancel()
            drained = mcp_server._registry._drained
            wait_for_drain = type(drained).wait

            async def observe_drain(event: anyio.Event) -> None:
                if event is drained:
                    close_waiting.set()
                await wait_for_drain(event)

            with patch.object(type(drained), "wait", observe_drain):
                closing = asyncio.create_task(mcp_server.aclose())
                with anyio.fail_after(4):
                    await close_waiting.wait()
                prematurely_finished = invocation.done()
                prematurely_closed = close.called
                release.set()
                with anyio.fail_after(4):
                    if cancellation == "anyio":
                        await invocation
                        assert scopes[0].cancelled_caught
                    else:
                        with pytest.raises(asyncio.CancelledError):
                            await invocation
                    await closing
            assert finished.is_set()
            manager.memory.add_batch.assert_called_once()
            close.assert_called_once()
            assert not prematurely_finished
            assert not prematurely_closed
        finally:
            release.set()
            if invocation is not None:
                _ = await asyncio.wait({invocation}, timeout=4)
            if closing is not None:
                _ = await asyncio.wait({closing}, timeout=4)


@pytest.mark.parametrize(
    "memories",
    [
        ["Single memory"],
        ["First", "Second"],
        ["Message with\nnewlines"],
        ["Unicode: 你好"],
        ["!@#$%", "🌍", '"quotes"'],
        ["a"],
        ["x" * 30720],
    ],
)
async def test_add_valid_memories(mcp_server, memories: list[str]) -> None:
    result = await mcp_server.tool_fn("add_memories")(memories)
    assert result["stored_count"] == len(memories)
    assert result["skipped_count"] == 0
    assert result["dry_run"] is False
    mcp_server.memory_manager.memory.add_batch.assert_called_once()
    kwargs = mcp_server.memory_manager.memory.add_batch.call_args.kwargs
    assert kwargs["contents"] == memories
    assert kwargs["workspace_id"] == mcp_server.config.workspace_id


@pytest.mark.parametrize(
    "memories", [[123], [""], ["   "], ["x" * 30721], ["Valid", ""]]
)
async def test_add_validation_precedes_storage(mcp_server, memories) -> None:
    with pytest.raises(ValueError):
        await mcp_server.tool_fn("add_memories")(memories)
    mcp_server.memory_manager.memory.add_batch.assert_not_called()


async def test_add_empty_list_is_noop(mcp_server) -> None:
    assert await mcp_server.tool_fn("add_memories")([]) == {
        "stored_count": 0,
        "skipped_count": 0,
        "replaced_count": 0,
        "replacements": [],
        "dry_run": False,
    }
    mcp_server.memory_manager.memory.add_batch.assert_not_called()


async def test_add_storage_failure(mcp_server) -> None:
    mcp_server.memory_manager.memory.add_batch.side_effect = RuntimeError(
        "Storage failed"
    )
    with pytest.raises(StorageError, match="Failed to add memories"):
        await mcp_server.tool_fn("add_memories")(["Valid memory"])


async def test_add_skips_duplicates(mcp_server) -> None:
    mcp_server.memory_manager.add_memories_async = AsyncMock(
        return_value=AddResult(
            stored_count=0, skipped_count=1, replaced_count=0, replacements=[]
        )
    )
    result = await mcp_server.tool_fn("add_memories")(["Duplicate memory"])
    assert result["skipped_count"] == 1
    mcp_server.memory_manager.add_memories_async.assert_awaited_once()


@pytest.mark.parametrize(
    (
        "limit",
        "offset",
        "page",
        "total",
        "expected_limit",
        "expected_offset",
        "truncated",
    ),
    [
        (None, 0, [], 0, 1000, 0, False),
        (None, 0, ["first", "second"], 3, 1000, 0, True),
        (2, 1, ["second", "third"], 3, 2, 1, False),
        (0, 0, [], 3, 0, 0, True),
        (-3, -7, [], 3, 0, 0, True),
        (10, 20, [], 3, 10, 20, False),
        (2000, 0, ["first"], 1001, 1000, 0, True),
    ],
)
async def test_page_metadata_from_single_manager_read(
    mcp_server, limit, offset, page, total, expected_limit, expected_offset, truncated
):
    with (
        patch.object(
            mcp_server.memory_manager,
            "get_page_records_with_total",
            return_value=(rows(page), total),
        ) as read,
        patch.object(
            mcp_server.memory_manager,
            "get_all",
            side_effect=AssertionError("separate page"),
        ),
        patch.object(
            mcp_server.memory_manager,
            "count",
            side_effect=AssertionError("separate count"),
        ),
    ):
        result = await mcp_server.tool_fn("get_all_memories")(
            limit=limit, offset=offset
        )
    assert result == {
        "records": payload(page),
        "total": total,
        "offset": expected_offset,
        "limit": expected_limit,
        "truncated": truncated,
    }
    read.assert_called_once_with(limit=expected_limit, offset=expected_offset)


@pytest.mark.parametrize(
    "memories",
    [
        [],
        ["one"],
        ["one", "two", "three"],
        ["!@#$", "你好", "🌍"],
        [f"Memory {i}" for i in range(1000)],
    ],
)
async def test_get_all_preserves_content_and_copies_page(
    mcp_server, memories: list[str]
) -> None:
    records = rows(memories)
    with patch.object(
        mcp_server.memory_manager,
        "get_page_records_with_total",
        return_value=(records, len(records)),
    ) as read:
        first = await mcp_server.tool_fn("get_all_memories")()
        second = await mcp_server.tool_fn("get_all_memories")()
    assert first["records"] == payload(memories)
    assert second["records"] == payload(memories)
    assert first["records"] is not records
    assert first["records"] is not second["records"]
    assert read.call_count == 2


async def test_configured_page_cap(mcp_server) -> None:
    with (
        patch.object(
            mcp_server.memory_manager,
            "get_page_records_with_total",
            return_value=(rows(["one", "two"]), 3),
        ) as read,
        patch.object(mcp_server, "config", replace(mcp_server.config, get_all_limit=2)),
    ):
        result = await mcp_server.tool_fn("get_all_memories")(limit=50)
    assert result == {
        "records": payload(["one", "two"]),
        "total": 3,
        "offset": 0,
        "limit": 2,
        "truncated": True,
    }
    read.assert_called_once_with(limit=2, offset=0)


async def test_blocked_page_worker_keeps_event_loop_responsive(mcp_server) -> None:
    entered = Event()
    release = Event()
    completed = anyio.Event()
    worker_threads: list[int] = []
    results: list[dict[str, object]] = []
    event_loop_thread = get_ident()

    def read_page(*, limit: int, offset: int) -> tuple[list[MemoryRecord], int]:
        worker_threads.append(get_ident())
        entered.set()
        release.wait(5)
        return rows(["memory"]), 1

    async def call_tool() -> None:
        results.append(await mcp_server.tool_fn("get_all_memories")())
        completed.set()

    with patch.object(
        mcp_server.memory_manager, "get_page_records_with_total", side_effect=read_page
    ) as read:
        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(call_tool)
                with anyio.fail_after(3):
                    assert await anyio.to_thread.run_sync(entered.wait, 2)
                    await anyio.sleep(0)
                assert not completed.is_set()
                release.set()
        finally:
            release.set()
    assert worker_threads == [worker_threads[0]]
    assert worker_threads[0] != event_loop_thread
    assert results[0]["records"] == payload(["memory"])
    read.assert_called_once_with(limit=1000, offset=0)


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("Retrieval failed"),
        StorageError("read failed"),
        InconsistentStateError("split"),
    ],
)
async def test_page_errors_preserve_cause_or_inconsistent_identity(
    mcp_server, failure: Exception
) -> None:
    with patch.object(
        mcp_server.memory_manager, "get_page_records_with_total", side_effect=failure
    ) as read:
        with pytest.raises(
            InconsistentStateError
            if isinstance(failure, InconsistentStateError)
            else StorageError
        ) as raised:
            await mcp_server.tool_fn("get_all_memories")()
    assert (
        raised.value is failure
        if isinstance(failure, InconsistentStateError)
        else raised.value.__cause__ is failure
    )
    read.assert_called_once_with(limit=1000, offset=0)


@pytest.mark.parametrize(
    ("query", "memories"),
    [
        ("Hello", ["Hello, World!", "Hello"]),
        ("World", ["Hello, World!", "Goodbye", "World"]),
        ("programming", ["Python tutorial", "JavaScript guide", "Database design"]),
        ("Programming", ["Hello", "World", "Test"]),
        ("Python", []),
        ("@user", ["Memory to @user", "Another memory", "@user replied"]),
        ("你好", ["你好世界", "Hello World", "你好 friend"]),
        ("a", ["Apple", "Banana", "Cherry"]),
        ("test", []),
        ("test", ["This is a test", "testing", "no match"]),
        ("Python", ["Python tutorial", "Java guide", "Python examples"]),
        ("🌍", ["Hello 🌍", "World 🌎", "🌍 emoji"]),
    ],
)
async def test_search_preserves_semantic_matches_without_substring_filtering(
    mcp_server, query: str, memories: list[str]
) -> None:
    mcp_server.memory_manager.search_records = AsyncMock(
        return_value=search_result(memories)
    )
    result = await mcp_server.tool_fn("search_memories")(query)
    assert result["results"] == [
        {**record, "score": 0.5} for record in payload(memories)
    ]
    mcp_server.memory_manager.search_records.assert_awaited_once_with(
        query, limit=mcp_server.config.search_limit
    )


async def test_search_error_wrapping(mcp_server) -> None:
    failure = RuntimeError("Search failed")
    mcp_server.memory_manager.search_records = AsyncMock(side_effect=failure)
    with pytest.raises(SearchError, match="Failed to search memory records") as raised:
        await mcp_server.tool_fn("search_memories")("test")
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize(
    ("ids", "removed", "missing"),
    [
        ([1], [1], []),
        ([1, 2], [1, 2], []),
        ([], [], []),
        ([99], [], [99]),
        ([1, 1], [1], []),
        ([1, 99], [1], [99]),
    ],
)
async def test_delete_reports_removed_and_missing(
    mcp_server, ids, removed, missing
) -> None:
    with patch.object(
        mcp_server.memory_manager, "delete_records", return_value=(removed, missing)
    ) as deleted:
        result = await mcp_server.tool_fn("delete_memories")(ids)
    assert result == {"removed": removed, "missing": missing}
    deleted.assert_called_once_with(ids)


@pytest.mark.parametrize("ids", [[0], [-1], [1] * 101])
async def test_delete_rejects_invalid_ids_before_manager(mcp_server, ids) -> None:
    with patch.object(mcp_server.memory_manager, "delete_records") as deleted:
        with pytest.raises(ValueError):
            await mcp_server.tool_fn("delete_memories")(ids)
    deleted.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("Delete failed"), InconsistentStateError("USearch/Tantivy split")],
)
async def test_delete_error_wrapping(mcp_server, failure: Exception) -> None:
    with patch.object(mcp_server.memory_manager, "delete_records", side_effect=failure):
        with pytest.raises(
            InconsistentStateError
            if isinstance(failure, InconsistentStateError)
            else StorageError
        ) as raised:
            await mcp_server.tool_fn("delete_memories")([1])
    assert (
        raised.value is failure
        if isinstance(failure, InconsistentStateError)
        else raised.value.__cause__ is failure
    )


@pytest.mark.parametrize(
    ("allowed", "expected", "invalid"),
    [
        (None, NAMES, []),
        ("add_memories,get_all_memories", [NAMES[0], NAMES[2]], []),
        (
            "search-memories,SearchMemories,get_memory_by_id_tool",
            [NAMES[1], NAMES[3]],
            [],
        ),
        (
            "add_memories,unknown,delete_memories_tool,search_records,add",
            [NAMES[0], NAMES[4]],
            ["unknown", "search_records", "add"],
        ),
        (",".join(reversed(NAMES)), NAMES, []),
        ("delete_memories,search_memories", [NAMES[1], NAMES[4]], []),
        ("none", [], []),
    ],
)
def test_allowed_tools_filtering(
    monkeypatch, set_env_vars, allowed, expected, invalid
) -> None:
    if allowed is None:
        monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    else:
        monkeypatch.setenv("ALLOWED_TOOLS", allowed)
    with patch("reflectlog.application.mcp_server.create_logger") as logger_factory:
        logger = MagicMock()
        logger_factory.return_value = logger
        server = FastMCPServer(Config.from_environment())
    assert list(server.registered_tools) == expected
    if invalid:
        reported = logger.warning.call_args_list[0].kwargs["extra"]["invalid_tools"]
        assert set(reported) == set(invalid)
    elif expected:
        logger.warning.assert_not_called()
    else:
        assert "No MCP tools selected" in logger.warning.call_args.args[0]


async def test_allowed_tools_discovery_keeps_canonical_order(
    monkeypatch, set_env_vars
) -> None:
    monkeypatch.setenv("ALLOWED_TOOLS", "delete_memories,get_all_memories,add_memories")
    server = FastMCPServer(Config.from_environment())
    client_factory = cast("Callable[[object], Client]", Client)
    async with client_factory(server.mcp) as client:
        tools = cast("list[MCPTool]", await client.list_tools())
    assert [tool.name for tool in tools] == [NAMES[0], NAMES[2], NAMES[4]]


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("", None),
        ("   ", None),
        ("searchmemories", "search_memories"),
        ("getallmemories", "get_all_memories"),
        ("search-memories", "search_memories"),
        ("get_memory_by_id_tool", "get_memory_by_id"),
        ("SearchMemories", "search_memories"),
    ],
)
def test_canonicalize_tool_token(token: str, expected: str | None) -> None:
    assert FastMCPServer._canonicalize_tool_token(token, set(NAMES)) == expected


@pytest.mark.parametrize(
    "workspace", ["../other", "a/b", "/Volumes/Data/oss/oc/ReflectLog", " . ", "  "]
)
async def test_invalid_workspace_is_rejected_before_manager(
    set_env_vars, workspace: str
) -> None:
    with patch("reflectlog.application.mcp_server.MemoryManager") as factory:
        server = FastMCPServer(replace(Config.from_environment(), workspace_id=""))
        client_factory = cast("Callable[[object], Client]", Client)
        async with client_factory(server.mcp) as client:
            with pytest.raises(ToolError, match="Invalid WORKSPACE_ID"):
                await client.call_tool(
                    "get_all_memories", {"workspace_id": workspace, "limit": 0}
                )
        factory.assert_not_called()


async def test_workspace_normalization_required_schema_and_startup_metrics(
    set_env_vars,
) -> None:
    managers: list[MagicMock] = []

    def create_manager(config: Config, _logger: object) -> MagicMock:
        manager = MagicMock(spec=MemoryManager)
        manager.config = config
        manager.get_page_records_with_total.return_value = (
            rows([config.workspace_id], config.workspace_id),
            1,
        )
        managers.append(manager)
        return manager

    with patch(
        "reflectlog.application.mcp_server.MemoryManager", side_effect=create_manager
    ) as factory:
        server = FastMCPServer(replace(Config.from_environment(), workspace_id=""))
        server.set_startup_metrics({"total_startup": 0.25})
        assert factory.call_count == 0
        client_factory = cast("Callable[[object], Client]", Client)
        async with client_factory(server.mcp) as client:
            tools = cast("list[MCPTool]", await client.list_tools())
            assert [tool.name for tool in tools] == NAMES
            assert list(server.registered_tools) == NAMES
            for tool in tools:
                assert "workspace_id" in tool.input_schema["required"]
            with pytest.raises(Exception, match="workspace_id"):
                await client.call_tool("get_all_memories", {})
            assert factory.call_count == 0
            for workspace in ("ReflectLog", "  ReflectLog \t"):
                result = await client.call_tool(
                    "get_all_memories", {"workspace_id": workspace}
                )
                assert json.loads(result.content[0].text)["records"] == payload(
                    ["reflectlog"]
                )
            assert factory.call_count == 1
            assert managers[0].config.workspace_id == "reflectlog"
            assert managers[0].startup_metrics == {"total_startup": 0.25}
            assert server.startup_metrics == {"total_startup": 0.25}
            for workspace in ("alpha", "beta"):
                result = await client.call_tool(
                    "get_all_memories", {"workspace_id": workspace}
                )
                assert json.loads(result.content[0].text)["records"] == payload(
                    [workspace]
                )
            assert [manager.config.workspace_id for manager in managers] == [
                "reflectlog",
                "alpha",
                "beta",
            ]
        managers[0].close.assert_called_once()


async def test_concurrent_workspace_isolation_and_shutdown_waits(set_env_vars) -> None:
    managers: dict[str, MagicMock] = {}
    entered = {"alpha": anyio.Event(), "beta": anyio.Event()}
    release = anyio.Event()

    def create_manager(config: Config, _logger: object) -> MagicMock:
        manager = MagicMock(spec=MemoryManager)
        manager.config = config

        async def search(_query: str, *, limit: int) -> RecordSearchResult:
            entered[config.workspace_id].set()
            await release.wait()
            return search_result([config.workspace_id])

        manager.search_records = AsyncMock(side_effect=search)
        managers[config.workspace_id] = manager
        return manager

    with patch(
        "reflectlog.application.mcp_server.MemoryManager", side_effect=create_manager
    ):
        server = FastMCPServer(replace(Config.from_environment(), workspace_id=""))
        with pytest.raises(TypeError, match="workspace_id"):
            signature(server._search_memories).bind("query")
        results: dict[str, dict[str, object]] = {}

        async def search_in(workspace: str) -> None:
            results[workspace] = await server._search_memories(
                "query", workspace_id=workspace
            )

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(search_in, "alpha")
            tasks.start_soon(search_in, "beta")
            await entered["alpha"].wait()
            await entered["beta"].wait()
            tasks.start_soon(server.aclose)
            assert not managers["alpha"].close.called
            assert not managers["beta"].close.called
            release.set()
        for workspace in ("alpha", "beta"):
            assert results[workspace]["results"] == [
                {**payload([workspace])[0], "score": 0.5}
            ]
            managers[workspace].close.assert_called_once()


@pytest.mark.parametrize("failure", [None, RuntimeError("disk full")])
def test_close_persists_and_reports_errors(monkeypatch, set_env_vars, failure) -> None:
    manager = MagicMock(spec=MemoryManager)
    manager.close.side_effect = failure
    logger = MagicMock()
    with (
        patch("reflectlog.application.mcp_server.MemoryManager", return_value=manager),
        patch("reflectlog.application.mcp_server.create_logger", return_value=logger),
    ):
        server = FastMCPServer(Config.from_environment())

        async def open_workspace() -> None:
            async with server._registry.acquire("test_project"):
                pass

        anyio.run(open_workspace)
    if failure:
        with pytest.raises(
            ExceptionGroup, match="Workspace managers could not be closed"
        ) as raised:
            server.close()
        assert raised.value.exceptions[0] is failure
        logger.error.assert_called_once()
    else:
        server.close()
        logger.info.assert_any_call("Initiating graceful server shutdown...")
        logger.info.assert_any_call("Server shutdown complete - all data persisted")
    manager.close.assert_called_once()


@pytest.mark.parametrize(
    ("failure", "during_run"),
    [
        (RuntimeError("port in use"), False),
        (KeyboardInterrupt(), True),
        (ValueError("unexpected"), True),
        (KeyboardInterrupt(), False),
        (TypeError("bad init"), False),
    ],
)
def test_main_error_paths(
    set_env_vars, failure: BaseException, during_run: bool
) -> None:
    with (
        patch("reflectlog.application.mcp_server.FastMCPServer") as factory,
        patch("reflectlog.application.mcp_server.get_logger"),
    ):
        instance = MagicMock()
        if during_run:
            factory.return_value = instance
            instance.run.side_effect = failure
        else:
            factory.side_effect = failure
        if isinstance(failure, KeyboardInterrupt):
            main()
        else:
            with pytest.raises(type(failure)):
                main()
        if during_run:
            instance.close.assert_called_once()
