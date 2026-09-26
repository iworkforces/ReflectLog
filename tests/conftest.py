"""Shared fixtures and configuration for all tests."""

from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import anyio
import pytest
from pytest import MonkeyPatch

from reflectlog.application.mcp_server import FastMCPServer


class _WorkflowServer(FastMCPServer):
    memory_manager: MagicMock


class MockMemorySearchResult:
    """Mock implementation of MemorySearchResult for testing."""

    def __init__(self, text: str, index: int | str):
        """Initialize mock search result.

        Args:
            text: The memory text content
            index: Unique identifier for the result
        """
        self._text = text
        self.index = index

    def __str__(self) -> str:
        """Return string representation."""
        return self._text

    def lower(self) -> str:
        """Return lowercase version of the text."""
        return self._text.lower()


@pytest.fixture
def mock_search_result() -> Callable[[str, int | str], MockMemorySearchResult]:
    """Factory fixture for creating mock search results.

    Returns:
        Callable that creates MockMemorySearchResult instances
    """

    def _create_result(text: str, index: int | str = 0) -> MockMemorySearchResult:
        return MockMemorySearchResult(text, index)

    return _create_result


@pytest.fixture
def mock_usearch_engine() -> MagicMock:
    """Mock USearchEngine instance for testing.

    Returns:
        MagicMock configured to behave like a USearchEngine instance
    """
    engine = MagicMock()
    engine.add = MagicMock(return_value=None)

    def add_batch(
        workspace_id: str,
        messages: list[str] | None = None,
        infer: bool = False,
        contents: list[str] | None = None,
        vectors: list[list[float]] | None = None,
    ) -> list[str] | None:
        del workspace_id, infer, vectors
        return contents if contents is not None else messages

    engine.add_batch = MagicMock(side_effect=add_batch)
    # USearchEngine.get_all returns List[str] directly
    engine.get_all = MagicMock(return_value=[])
    # USearchEngine.search returns List[Tuple[str, float]]
    engine.search = MagicMock(return_value=[])
    engine.delete = MagicMock(return_value=None)
    engine.commit = MagicMock(return_value=None)
    engine.ensure_initialized = MagicMock(return_value=None)
    # Default constructed engines are not warmed; a bare MagicMock is truthy.
    engine.is_ready = MagicMock(return_value=False)
    engine.get_id_by_content = MagicMock(return_value=None)
    engine.contains_id = MagicMock(return_value=None)
    engine.count = MagicMock(return_value=0)
    engine.embedder.embed_documents.side_effect = lambda texts: [
        [0.1] * 4 for _ in texts
    ]
    engine.embedder.embed_query.side_effect = lambda _query: [0.1] * 4
    engine.memory_store.begin_add_intents.return_value = []
    engine.memory_store.begin_delete_intents.return_value = []
    engine.memory_store.list_pending_transitions.return_value = []
    engine.memory_store.has_later_intent.return_value = False
    engine.memory_store.get.return_value = None
    engine.memory_store.exists_many.side_effect = lambda _workspace, contents: set(
        contents
    )
    engine.get_records_by_contents.return_value = []
    return engine


@pytest.fixture
def mock_memory_class(mock_usearch_engine: MagicMock) -> Generator[MagicMock]:
    """Mock USearchEngine class for patching imports.

    Yields:
        Patched USearchEngine class that returns mock_usearch_engine instance
    """
    with patch("reflectlog.application.memory.manager.USearchEngine") as mock_cls:
        mock_cls.return_value = mock_usearch_engine
        yield mock_cls


@pytest.fixture
def set_env_vars(monkeypatch: MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    """Set required environment variables for testing.

    Args:
        monkeypatch: pytest monkeypatch fixture
        tmp_path: Per-test workspace storage root

    Returns:
        Dictionary of set environment variables
    """
    monkeypatch.chdir(tmp_path)
    env_vars = {
        "WORKSPACE_ID": "test_project",
        "OPENROUTER_API_KEY": "test_key",
        "MCP_TRANSPORT": "stdio",
        "MCP_PORT": "9103",
        "MCP_HOST": "127.0.0.1",
        "MCP_PATH": "/mcp",
        "SEARCH_LIMIT": "5",
        "REMOVE_SEARCH_LIMIT": "5",
        "MAX_MEMORY_LENGTH": "30720",
        "MIN_MEMORY_LENGTH": "1",
        "LOG_LEVEL": "INFO",
        "DEDUPLICATE_MEMORIES": "true",
        "ADD_MAX_CONCURRENCY": "8",
        "RERANKER_ENGINE": "cross_encoder",
        # Disable embedding cache in tests to avoid issues with mocked embedders
        "EMBEDDING_CACHE_ENABLED": "false",
        # Disable eager initialization in tests to avoid issues with mocked engines
        "EAGER_INITIALIZATION": "false",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def unset_workspace_id(monkeypatch: MonkeyPatch) -> None:
    """Remove WORKSPACE_ID from environment for testing missing config.

    Args:
        monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.delenv("WORKSPACE_ID", raising=False)


@pytest.fixture
def mcp_server(
    set_env_vars: dict[str, str], mock_usearch_engine: MagicMock
) -> Generator["FastMCPServer"]:
    """Create FastMCPServer instance with mocked dependencies.

    Args:
        set_env_vars: Environment variables fixture
        mock_usearch_engine: Mocked USearchEngine instance

    Returns:
        FastMCPServer instance for testing
    """
    with (
        patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch_cls,
        patch(
            "reflectlog.application.memory.manager.TantivyEngine"
        ) as mock_tantivy_cls,
        patch(
            "reflectlog.application.memory.manager.LangchainQwenEmbeddings"
        ) as mock_embedder_cls,
        patch(
            "reflectlog.application.memory.manager.CachedEmbeddings"
        ) as mock_cached_embedder_cls,
    ):
        from reflectlog.application.config.settings import Config
        from reflectlog.application.memory.manager import MemoryManager

        # Configure USearchEngine mock
        mock_usearch_cls.return_value = mock_usearch_engine

        # Configure TantivyEngine mock (basic mock, not used in most tests)
        mock_tantivy = MagicMock()
        mock_tantivy.ensure_initialized = MagicMock(return_value=None)
        mock_tantivy.search = MagicMock(return_value=[])
        mock_tantivy.is_ready = MagicMock(return_value=False)
        mock_tantivy_cls.return_value = mock_tantivy

        # Configure embedder mock
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.side_effect = lambda texts: [
            [0.1] * 4 for _ in texts
        ]
        mock_embedder.embed_query.side_effect = lambda _query: [0.1] * 4
        mock_embedder_cls.return_value = mock_embedder

        # Configure CachedEmbeddings mock to return the base embedder mock
        # This prevents Pydantic validation errors in tests
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = mock_embedder
        mock_cached_embedder.embed_documents.side_effect = mock_embedder.embed_documents
        mock_cached_embedder.embed_query.side_effect = mock_embedder.embed_query
        mock_cached_embedder_cls.return_value = mock_cached_embedder
        mock_tantivy.find_by_exact_match.return_value = []
        mock_tantivy.delete.return_value = True
        mock_tantivy.delete_batch.side_effect = (
            lambda _workspace, contents, verify_exists=True: len(contents)
        )

        workspace_id = set_env_vars["WORKSPACE_ID"]
        server = _WorkflowServer(
            replace(Config.from_environment(), workspace_id=workspace_id)
        )

        async def acquire_manager() -> MemoryManager:
            async with server._registry.acquire(workspace_id) as manager:
                return manager

        server.memory_manager = cast("MagicMock", anyio.run(acquire_manager))
        server.memory_manager.memory = mock_usearch_engine
        semantic_engine = server.memory_manager._semantic_engine
        tantivy_engine = server.memory_manager._tantivy_engine
        try:
            yield server
        finally:
            server.memory_manager._semantic_engine = semantic_engine
            server.memory_manager._tantivy_engine = tantivy_engine
            anyio.run(server.aclose)


@pytest.fixture
def sample_memories() -> dict[str, list[str] | dict[str, str]]:
    """Provide sample memories for testing.

    Returns:
        Dictionary of sample memory lists for various test scenarios
    """
    return {
        "single": ["Hello, World!"],
        "multiple": [
            "First message",
            "Second message",
            "Third message",
        ],
        "with_special_chars": [
            "Message with special chars: !@#$%^&*()",
            "Unicode message: 你好世界 🌍",
            "Newlines\nand\ttabs",
        ],
        "duplicates": [
            "Duplicate message",
            "Unique message",
            "Duplicate message",
        ],
        "edge_cases": {
            "min_length": "a",  # 1 character
            "max_length": "x" * 30720,  # 30720 characters
            "whitespace": "   spaces   ",
        },
        "invalid": {
            "empty": "",
            "whitespace_only": "   ",
            "too_long": "x" * 30721,  # 30721 characters
        },
    }


@pytest.fixture
def mock_logger() -> MagicMock:
    """Mock logger for testing logging behavior.

    Returns:
        MagicMock configured as a logger
    """
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    return logger


@pytest.fixture(autouse=True)
def reset_env_after_test():
    """Automatically reset environment after each test.

    This fixture runs after every test to ensure clean state.
    """
    yield
    # Reset cached config singleton to avoid cross-test config mutation
    try:
        from reflectlog.application.config import settings as config_settings

        with config_settings._config_lock:
            config_settings._config = None
        # Remove any per-test overrides set on the lazy config proxy.
        try:
            config_settings.config.__dict__.clear()
        except AttributeError:
            pass
    except Exception:
        # Best-effort reset; tests should still proceed if this fails
        pass


@pytest.fixture
def create_search_results(
    mock_search_result: Callable[[str, int | str], MockMemorySearchResult],
) -> Callable[[list[str]], list[tuple[str, float, str]]]:
    """Factory for creating lists of search results in USearchEngine format.

    Args:
        mock_search_result: Factory fixture for single results (unused, kept for compatibility)

    Returns:
        Callable that creates lists of tuples (memory, score, created_at) for USearchEngine.search()
    """

    def _create_results(memories: list[str]) -> list[tuple[str, float, str]]:
        # USearchEngine.search returns List[Tuple[str, float, str]]
        # Default scores decrease from 0.9 for ranking
        results: list[tuple[str, float, str]] = []
        for idx, memory in enumerate(memories):
            score = 0.9 - (idx * 0.1)
            created_at = f"2024-01-{idx + 1:02d}T00:00:00"
            results.append((memory, score, created_at))
        return results

    return _create_results


@pytest.fixture
def create_search_response() -> Callable[
    [list[str], bool], dict[str, list[dict[str, str]]]
]:
    """Factory for creating search response dictionaries.

    Returns:
        Callable that creates search response format: {'results': [{'memory': '...', 'id': '...'}]}
    """

    def _create_response(
        memories: list[str], include_ids: bool = True
    ) -> dict[str, list[dict[str, str]]]:
        results: list[dict[str, str]] = []
        for idx, memory in enumerate(memories):
            result = {"memory": memory}
            if include_ids:
                result["id"] = f"id_{idx}"
            results.append(result)
        return {"results": results}

    return _create_response


@pytest.fixture
def get_tool_func(
    mcp_server: "FastMCPServer",
) -> Callable[[str], Callable[..., object]]:
    """Helper to get tool function by name.

    Args:
        mcp_server: FastMCPServer instance

    Returns:
        Callable that retrieves tool functions by name
    """

    def _get_tool(tool_name: str) -> Callable[..., object]:
        """Get tool function by name."""
        for tool in mcp_server.tools:
            if tool.get_name() == tool_name:
                return tool.get_handler()
        raise ValueError(f"Tool '{tool_name}' not found")

    return _get_tool


@pytest.fixture
def add_tool(
    get_tool_func: Callable[[str], Callable[..., object]],
) -> Callable[..., object]:
    """Get the add tool function."""
    return get_tool_func("add")


@pytest.fixture
def get_all_tool(
    get_tool_func: Callable[[str], Callable[..., object]],
) -> Callable[..., object]:
    """Get the get_all tool function."""
    return get_tool_func("get_all")


@pytest.fixture
def search_tool(
    get_tool_func: Callable[[str], Callable[..., object]],
) -> Callable[..., object]:
    """Get the search tool function."""
    return get_tool_func("search")


@pytest.fixture
def remove_tool(
    get_tool_func: Callable[[str], Callable[..., object]],
) -> Callable[..., object]:
    """Get the remove tool function."""
    return get_tool_func("remove")
