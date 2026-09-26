"""Integration tests for MCP server workflows."""

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypedDict, cast

import pytest

type SemanticSearchResult = tuple[str, float, str]
type TantivySearchResult = tuple[str, float]
type SearchResultFactory = Callable[[list[str]], list[SemanticSearchResult]]
type BatchCallback = Callable[..., list[str]]
type SearchCallback = Callable[..., list[SemanticSearchResult]]
type DeleteCallback = Callable[..., None]
type GetAllCallback = Callable[..., list[str]]
type MemoryIdCallback = Callable[[str, str], int | None]
type SemanticEngineCallback = Callable[..., list[SemanticSearchResult]]
type TantivyEngineCallback = Callable[..., list[TantivySearchResult]]


class SampleMemories(TypedDict):
    multiple: list[str]


class CallbackSlot[T](Protocol):
    side_effect: T


class GetAllSlot(Protocol):
    side_effect: GetAllCallback
    return_value: list[str]


class SearchSlot(Protocol):
    side_effect: SearchCallback
    return_value: list[SemanticSearchResult]


class WorkflowMemory(Protocol):
    add_batch: CallbackSlot[BatchCallback]
    get_all: GetAllSlot
    search: SearchSlot
    delete: CallbackSlot[DeleteCallback]
    get_id_by_memory: CallbackSlot[MemoryIdCallback]
    get_id_by_content: CallbackSlot[MemoryIdCallback]


class WorkflowSemanticEngine(Protocol):
    search: CallbackSlot[SemanticEngineCallback]


class WorkflowTantivyEngine(Protocol):
    search: CallbackSlot[TantivyEngineCallback]


class WorkflowConfig(Protocol):
    workspace_id: str
    enable_rrf_fusion: bool
    reranker_engine: str
    fusion_ranking_threshold: float


class WorkflowMemoryManager(Protocol):
    config: WorkflowConfig
    memory: WorkflowMemory
    _semantic_engine: WorkflowSemanticEngine
    _tantivy_engine: WorkflowTantivyEngine

    def configure_search_engines(
        self,
        semantic_search: SemanticEngineCallback,
        tantivy_search: TantivyEngineCallback,
    ) -> None:
        self._semantic_engine.search.side_effect = semantic_search
        tantivy: Any = self._tantivy_engine
        if tantivy is not None:
            tantivy.search.side_effect = tantivy_search


class WorkflowServer(Protocol):
    tools: list["WorkflowTool"]
    memory_manager: WorkflowMemoryManager
    config: WorkflowConfig


class WorkflowTool(Protocol):
    def get_name(self) -> str: ...

    def get_handler(self) -> Callable[..., Awaitable[list[str]]]: ...


def _tool_handler(
    server: WorkflowServer, name: str
) -> Callable[..., Awaitable[list[str]]]:
    for tool in server.tools:
        if tool.get_name() == name:
            return tool.get_handler()
    raise AssertionError(f"{name} tool not found")


async def _add(server: WorkflowServer, memories: list[str]) -> None:
    _ = await _tool_handler(server, "add")(
        memories, workspace_id=server.config.workspace_id
    )


async def _get_all(server: WorkflowServer) -> list[str]:
    result: object = await _tool_handler(server, "get_all")(
        workspace_id=server.config.workspace_id
    )
    if isinstance(result, dict):
        page = cast(dict[str, object], result).get("memories")
        if isinstance(page, list):
            return [str(item) for item in cast(list[object], page)]
        return []
    return result


async def _search(server: WorkflowServer, query: str) -> list[str]:
    return await _tool_handler(server, "search")(
        query, workspace_id=server.config.workspace_id
    )


async def _remove(server: WorkflowServer, memories: list[str]) -> None:
    _ = await _tool_handler(server, "remove")(
        memories, workspace_id=server.config.workspace_id
    )


def _add_to_store(stored_memories: list[str]) -> BatchCallback:
    def add_side_effect(
        *,
        workspace_id: str | None = None,
        memories: list[str] | None = None,
        contents: list[str] | None = None,
        infer: bool = True,
        **_kwargs: str | bool | list[list[float]] | None,
    ) -> list[str]:
        del workspace_id, infer
        batch = contents if contents is not None else memories
        stored_memories.extend(batch or [])
        return batch or []

    return add_side_effect


def _add_unique_to_store(stored_memories: list[str]) -> BatchCallback:
    def add_side_effect(
        *,
        workspace_id: str | None = None,
        memories: list[str] | None = None,
        contents: list[str] | None = None,
        infer: bool = True,
        **_kwargs: str | bool | list[list[float]] | None,
    ) -> list[str]:
        del workspace_id, infer
        batch = contents if contents is not None else memories
        for memory in batch or []:
            if memory not in stored_memories:
                stored_memories.append(memory)
        return batch or []

    return add_side_effect


def _all_memories(stored_memories: list[str]) -> GetAllCallback:
    def get_all_side_effect(**_kwargs: str | int | None) -> list[str]:
        return stored_memories.copy()

    return get_all_side_effect


def _matching_search(
    stored_memories: list[str],
    create_search_results: SearchResultFactory,
) -> SearchCallback:
    def search_side_effect(
        query: str, **_kwargs: str | int | None
    ) -> list[SemanticSearchResult]:
        matching = [
            memory for memory in stored_memories if query.lower() in memory.lower()
        ]
        return create_search_results(matching)

    return search_side_effect


def _exact_search(
    stored_memories: list[str],
    create_search_results: SearchResultFactory,
) -> SearchCallback:
    def search_side_effect(
        query: str, **_kwargs: str | int | None
    ) -> list[SemanticSearchResult]:
        return create_search_results(
            [memory for memory in stored_memories if memory == query]
        )

    return search_side_effect


def _delete_from_store(stored_memories: list[str]) -> DeleteCallback:
    def delete_side_effect(
        memory_id: str | int | None = None,
        workspace_id: str | None = None,
    ) -> None:
        del workspace_id
        if memory_id is None:
            return
        try:
            index = int(memory_id)
        except TypeError, ValueError:
            return
        if 0 <= index < len(stored_memories):
            _ = stored_memories.pop(index)

    return delete_side_effect


def _find_memory_id(stored_memories: list[str]) -> MemoryIdCallback:
    def find_memory_id(workspace_id: str, memory: str) -> int | None:
        del workspace_id
        return stored_memories.index(memory) if memory in stored_memories else None

    return find_memory_id


def _semantic_search(stored_memories: list[str]) -> SemanticEngineCallback:
    def semantic_search(
        query: str, **_kwargs: str | int | None
    ) -> list[SemanticSearchResult]:
        return [
            (memory, 0.9, "2026-08-22T00:00:00+00:00")
            for memory in stored_memories
            if query.lower() in memory.lower()
        ]

    return semantic_search


def _tantivy_search(stored_memories: list[str]) -> TantivyEngineCallback:
    def tantivy_search(
        query: str,
        *_args: str | int,
        **_kwargs: str | int | None,
    ) -> list[TantivySearchResult]:
        return [
            (memory, 0.8)
            for memory in stored_memories
            if query.lower() in memory.lower()
        ]

    return tantivy_search


@pytest.mark.integration
class TestMCPWorkflows:
    """Integration tests for complete MCP tool workflows."""

    @pytest.mark.asyncio
    async def test_add_then_get_all_workflow(
        self, mcp_server: WorkflowServer, sample_memories: SampleMemories
    ) -> None:
        memories = sample_memories["multiple"]
        stored_memories: list[str] = []
        mcp_server.memory_manager.memory.add_batch.side_effect = _add_to_store(
            stored_memories
        )
        mcp_server.memory_manager.memory.get_all.side_effect = _all_memories(
            stored_memories
        )

        await _add(mcp_server, memories)
        result = await _get_all(mcp_server)

        assert len(result) == 3
        assert result == memories

    @pytest.mark.asyncio
    async def test_add_then_search_workflow(
        self,
        mcp_server: WorkflowServer,
        create_search_results: SearchResultFactory,
    ) -> None:
        memories = ["Python tutorial", "Java guide", "Python examples"]
        stored_memories: list[str] = []
        mcp_server.memory_manager.memory.add_batch.side_effect = _add_to_store(
            stored_memories
        )
        mcp_server.memory_manager.memory.search.side_effect = _matching_search(
            stored_memories, create_search_results
        )

        await _add(mcp_server, memories)
        object.__setattr__(mcp_server.memory_manager.config, "enable_rrf_fusion", False)
        object.__setattr__(mcp_server.memory_manager.config, "reranker_engine", "none")
        result = await _search(mcp_server, "Python")

        assert len(result) == 2
        assert "Python tutorial" in result
        assert "Python examples" in result
        assert "Java guide" not in result

    @pytest.mark.asyncio
    async def test_add_remove_get_all_workflow(
        self,
        mcp_server: WorkflowServer,
        create_search_results: SearchResultFactory,
    ) -> None:
        initial_memories = ["Memory 1", "Memory 2", "Memory 3"]
        stored_memories: list[str] = []
        memory = mcp_server.memory_manager.memory
        memory.add_batch.side_effect = _add_to_store(stored_memories)
        memory.search.side_effect = _exact_search(
            stored_memories, create_search_results
        )
        memory.delete.side_effect = _delete_from_store(stored_memories)
        memory.get_all.side_effect = _all_memories(stored_memories)
        memory.get_id_by_memory.side_effect = _find_memory_id(stored_memories)
        memory.get_id_by_content.side_effect = _find_memory_id(stored_memories)

        await _add(mcp_server, initial_memories)
        assert len(await _get_all(mcp_server)) == 3
        await _remove(mcp_server, ["Memory 2"])
        remaining = await _get_all(mcp_server)

        assert len(remaining) == 2
        assert "Memory 1" in remaining
        assert "Memory 3" in remaining
        assert "Memory 2" not in remaining

    @pytest.mark.asyncio
    async def test_add_search_remove_search_workflow(
        self,
        mcp_server: WorkflowServer,
        create_search_results: SearchResultFactory,
    ) -> None:
        memories = [
            "Python tutorial for beginners",
            "Advanced Python techniques",
            "Java programming guide",
            "Python data science",
        ]
        stored_memories: list[str] = []
        memory = mcp_server.memory_manager.memory
        memory.add_batch.side_effect = _add_to_store(stored_memories)
        memory.search.side_effect = _matching_search(
            stored_memories, create_search_results
        )
        memory.delete.side_effect = _delete_from_store(stored_memories)
        memory.get_id_by_memory.side_effect = _find_memory_id(stored_memories)
        memory.get_id_by_content.side_effect = _find_memory_id(stored_memories)

        await _add(mcp_server, memories)
        object.__setattr__(mcp_server.memory_manager.config, "enable_rrf_fusion", False)
        object.__setattr__(mcp_server.memory_manager.config, "reranker_engine", "none")
        assert len(await _search(mcp_server, "Python")) == 3
        await _remove(mcp_server, ["Python tutorial for beginners"])
        remaining = await _search(mcp_server, "Python")

        assert len(remaining) == 2
        assert "Advanced Python techniques" in remaining
        assert "Python data science" in remaining
        assert "Python tutorial for beginners" not in remaining

    @pytest.mark.asyncio
    async def test_multiple_add_operations(self, mcp_server: WorkflowServer) -> None:
        stored_memories: list[str] = []
        mcp_server.memory_manager.memory.add_batch.side_effect = _add_to_store(
            stored_memories
        )
        mcp_server.memory_manager.memory.get_all.side_effect = _all_memories(
            stored_memories
        )

        await _add(mcp_server, ["Memory 1", "Memory 2"])
        await _add(mcp_server, ["Memory 3"])
        await _add(mcp_server, ["Memory 4", "Memory 5"])

        assert len(await _get_all(mcp_server)) == 5

    @pytest.mark.asyncio
    async def test_empty_store_operations(self, mcp_server: WorkflowServer) -> None:
        mcp_server.memory_manager.memory.get_all.return_value = []
        mcp_server.memory_manager.memory.search.return_value = []

        assert await _get_all(mcp_server) == []
        assert await _search(mcp_server, "anything") == []
        assert await _remove(mcp_server, ["non-existent"]) is None

    @pytest.mark.asyncio
    async def test_add_duplicate_memories(
        self,
        mcp_server: WorkflowServer,
        create_search_results: SearchResultFactory,
    ) -> None:
        duplicate_memory = "Duplicate memory"
        stored_memories: list[str] = []
        memory = mcp_server.memory_manager.memory
        memory.add_batch.side_effect = _add_unique_to_store(stored_memories)
        memory.search.side_effect = _exact_search(
            stored_memories, create_search_results
        )
        memory.delete.side_effect = _delete_from_store(stored_memories)
        memory.get_all.side_effect = _all_memories(stored_memories)
        memory.get_id_by_memory.side_effect = _find_memory_id(stored_memories)
        memory.get_id_by_content.side_effect = _find_memory_id(stored_memories)

        await _add(mcp_server, [duplicate_memory, "Unique memory", duplicate_memory])
        assert len(await _get_all(mcp_server)) == 2
        await _remove(mcp_server, [duplicate_memory])
        remaining = await _get_all(mcp_server)

        assert len(remaining) == 1
        assert "Unique memory" in remaining

    @pytest.mark.asyncio
    async def test_search_with_no_semantic_matches(
        self, mcp_server: WorkflowServer
    ) -> None:
        mcp_server.memory_manager.memory.search.return_value = []

        assert await _search(mcp_server, "nonexistent query") == []

    @pytest.mark.asyncio
    async def test_large_dataset_workflow(
        self,
        mcp_server: WorkflowServer,
        create_search_results: SearchResultFactory,
    ) -> None:
        large_dataset = [f"Memory {index}" for index in range(100)]
        stored_memories: list[str] = []

        def limited_search(
            query: str, **_kwargs: str | int | None
        ) -> list[SemanticSearchResult]:
            matching = [memory for memory in stored_memories if query in memory]
            return create_search_results(matching[:5])

        memory = mcp_server.memory_manager.memory
        memory.add_batch.side_effect = _add_to_store(stored_memories)
        memory.get_all.side_effect = _all_memories(stored_memories)
        memory.search.side_effect = limited_search

        await _add(mcp_server, large_dataset)
        assert len(await _get_all(mcp_server)) == 100
        assert len(await _search(mcp_server, "Memory")) <= 5

    @pytest.mark.asyncio
    async def test_special_characters_workflow(
        self,
        mcp_server: WorkflowServer,
        create_search_results: SearchResultFactory,
    ) -> None:
        special_memories = [
            "Email: user@example.com",
            "Price: $100.00",
            "Math: 2 + 2 = 4",
            "Code: function() { return true; }",
        ]
        stored_memories: list[str] = []
        memory_manager = mcp_server.memory_manager
        memory_manager.memory.add_batch.side_effect = _add_to_store(stored_memories)
        memory_manager.memory.search.side_effect = _matching_search(
            stored_memories, create_search_results
        )
        object.__setattr__(memory_manager.config, "reranker_engine", "none")
        object.__setattr__(memory_manager.config, "fusion_ranking_threshold", 0.0)
        patched: Any = memory_manager
        patched._semantic_engine = memory_manager.memory
        WorkflowMemoryManager.configure_search_engines(
            memory_manager,
            _semantic_search(stored_memories),
            _tantivy_search(stored_memories),
        )

        await _add(mcp_server, special_memories)
        email_results = await _search(mcp_server, "@")
        price_results = await _search(mcp_server, "$")

        assert len(email_results) == 1
        assert "user@example.com" in email_results[0]
        assert len(price_results) == 1
        assert "$100.00" in price_results[0]

    @pytest.mark.asyncio
    async def test_unicode_workflow(
        self,
        mcp_server: WorkflowServer,
        create_search_results: SearchResultFactory,
    ) -> None:
        unicode_memories = [
            "Hello in Chinese: 你好",
            "Hello in Japanese: こんにちは",
            "Hello in Arabic: مرحبا",
            "Emoji: 😀 🌍 🚀",
        ]
        stored_memories: list[str] = []
        memory_manager = mcp_server.memory_manager
        memory_manager.memory.add_batch.side_effect = _add_to_store(stored_memories)
        memory_manager.memory.search.side_effect = _matching_search(
            stored_memories, create_search_results
        )
        object.__setattr__(memory_manager.config, "reranker_engine", "none")
        object.__setattr__(memory_manager.config, "fusion_ranking_threshold", 0.0)
        patched: Any = memory_manager
        patched._semantic_engine = memory_manager.memory
        WorkflowMemoryManager.configure_search_engines(
            memory_manager,
            _semantic_search(stored_memories),
            _tantivy_search(stored_memories),
        )

        await _add(mcp_server, unicode_memories)

        assert len(await _search(mcp_server, "你好")) == 1
        assert len(await _search(mcp_server, "😀")) == 1
