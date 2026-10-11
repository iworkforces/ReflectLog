"""Unit tests for error handling in mcp_server.py."""
# mypy: disable-error-code="misc,var-annotated,method-assign"

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reflectlog.core.enums import TransportMode
from reflectlog.core.exceptions import ConfigurationError, SearchError, StorageError


@pytest.fixture
def mock_usearch_engine():
    """Create a mock USearchEngine instance."""
    engine = MagicMock()
    engine.add = MagicMock(return_value=None)
    # USearchEngine.get_all returns List[str] directly
    engine.get_all = MagicMock(return_value=[])
    # USearchEngine.search returns List[Tuple[str, float]]
    engine.search = MagicMock(return_value=[])
    engine.delete = MagicMock(return_value=None)
    engine.commit = MagicMock(return_value=None)
    return engine


@pytest.fixture
def mock_cached_embedder():
    """Create a mock CachedEmbeddings instance."""
    embedder = MagicMock()
    embedder.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    embedder.aembed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    embedder.embed_documents = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    embedder.aembed_documents = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    return embedder


@pytest.mark.unit
class TestServerInitializationErrors:
    """Test error handling in server initialization."""

    @pytest.mark.asyncio
    async def test_init_without_workspace_id_raises_error(self, monkeypatch):
        from reflectlog.application.mcp_server import FastMCPServer

        monkeypatch.delenv("WORKSPACE_ID", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
        server = FastMCPServer()
        try:
            assert server.config.workspace_id == ""
            with pytest.raises(TypeError, match="workspace_id"):
                await server.registered_tools["get_all_memories"].fn()
        finally:
            await server.aclose()

    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    @pytest.mark.asyncio
    async def test_init_with_memory_config_error(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        set_env_vars: dict[str, str],
    ):
        # Configure CachedEmbeddings mock to return embedder mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        # Make USearchEngine raise an exception
        mock_usearch_engine_class.side_effect = Exception(
            "Memory initialization failed"
        )

        from reflectlog.application.mcp_server import FastMCPServer

        server = FastMCPServer()
        mock_usearch_engine_class.assert_not_called()
        try:
            with pytest.raises(Exception) as exc_info:
                await server.registered_tools["get_all_memories"].fn(
                    workspace_id=set_env_vars["WORKSPACE_ID"]
                )
        finally:
            await server.aclose()

        assert "Memory initialization failed" in str(exc_info.value)


@pytest.mark.unit
class TestAddMemoriesToolErrorHandling:
    """Test error handling in add_memories tool."""

    @pytest.mark.asyncio
    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    async def test_add_memory_storage_failure(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
        set_env_vars: dict[str, str],
    ):
        """Test add_memories tool handles memory storage failures."""
        # Configure CachedEmbeddings mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        # Configure mock to raise exception on add
        mock_usearch_engine.get_id_by_content.return_value = None
        mock_usearch_engine.embedder.embed_documents.side_effect = lambda texts: [
            [0.1] * 4 for _ in texts
        ]
        mock_usearch_engine.embedder.aembed_documents = AsyncMock(
            side_effect=lambda texts: [[0.1] * 4 for _ in texts]
        )
        mock_usearch_engine.add_batch.side_effect = Exception("Storage failure")
        mock_usearch_engine.add.side_effect = Exception("Storage failure")
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from reflectlog.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        add_memories_tool = mcp_server.registered_tools["add_memories"].fn

        with pytest.raises(StorageError) as exc_info:
            await add_memories_tool(
                ["Test message"], workspace_id=set_env_vars["WORKSPACE_ID"]
            )

        assert "Failed to add memories" in str(exc_info.value)
        assert "Storage failure" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    async def test_add_validation_error_with_non_string(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
        set_env_vars: dict[str, str],
    ):
        """Test add_memories tool validation error with non-string message."""
        # Configure CachedEmbeddings mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        mock_usearch_engine_class.return_value = mock_usearch_engine

        from reflectlog.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        add_memories_tool = mcp_server.registered_tools["add_memories"].fn

        with pytest.raises(ValueError) as exc_info:
            await add_memories_tool(
                [123, "Valid message"], workspace_id=set_env_vars["WORKSPACE_ID"]
            )

        assert "not a string" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    async def test_add_validation_error_with_empty_string(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
        set_env_vars: dict[str, str],
    ):
        """Test add_memories tool validation error with empty string."""
        # Configure CachedEmbeddings mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        mock_usearch_engine_class.return_value = mock_usearch_engine

        from reflectlog.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        add_memories_tool = mcp_server.registered_tools["add_memories"].fn

        with pytest.raises(ValueError) as exc_info:
            await add_memories_tool([""], workspace_id=set_env_vars["WORKSPACE_ID"])

        assert "contains only whitespace" in str(exc_info.value).lower()


@pytest.mark.unit
class TestGetAllMemoriesToolErrorHandling:
    """Test error handling in get_all_memories tool."""

    @pytest.mark.asyncio
    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    async def test_get_all_memory_retrieval_failure(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
        set_env_vars: dict[str, str],
    ):
        """Test get_all_memories tool handles retrieval failures."""
        # Configure CachedEmbeddings mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        # Configure mock to raise exception on page reads
        mock_usearch_engine.memory_store.get_page_records.side_effect = Exception(
            "Retrieval failure"
        )
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from reflectlog.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        get_all_memories_tool = mcp_server.registered_tools["get_all_memories"].fn

        with pytest.raises(StorageError) as exc_info:
            await get_all_memories_tool(workspace_id=set_env_vars["WORKSPACE_ID"])

        assert "Failed to list memory records" in str(exc_info.value)
        assert "Retrieval failure" in str(exc_info.value)


@pytest.mark.unit
class TestSearchMemoriesToolErrorHandling:
    """Test error handling in search_memories tool."""

    @pytest.mark.asyncio
    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    async def test_search_memory_failure(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
        set_env_vars: dict[str, str],
    ):
        """Test search_memories tool handles search failures."""
        # Configure CachedEmbeddings mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        # Configure mock to raise exception on search
        mock_usearch_engine.search.side_effect = Exception("Search failure")
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from reflectlog.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        search_memories_tool = mcp_server.registered_tools["search_memories"].fn

        with pytest.raises(SearchError, match="Failed to search memory records"):
            await search_memories_tool(
                "test query", workspace_id=set_env_vars["WORKSPACE_ID"]
            )


@pytest.mark.unit
class TestDeleteMemoriesToolErrorHandling:
    """Test error handling in delete_memories tool."""

    @pytest.mark.asyncio
    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    async def test_delete_with_empty_list(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
        set_env_vars: dict[str, str],
    ):
        """Test delete_memories tool with empty list (no-op)."""
        # Configure CachedEmbeddings mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        mock_usearch_engine_class.return_value = mock_usearch_engine

        from reflectlog.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        delete_memories_tool = mcp_server.registered_tools["delete_memories"].fn

        # Should not raise any errors
        await delete_memories_tool([], workspace_id=set_env_vars["WORKSPACE_ID"])

        # Verify delete was never called
        mock_usearch_engine.delete.assert_not_called()

    @pytest.mark.asyncio
    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    async def test_delete_memory_delete_failure(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
        set_env_vars: dict[str, str],
    ):
        """Test delete_memories tool handles delete failures."""
        # Configure CachedEmbeddings mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        mock_usearch_engine.get_id_by_memory.return_value = 1
        # Configure mock to raise exception on delete
        mock_usearch_engine.delete.side_effect = Exception("Delete failure")
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from reflectlog.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        delete_memories_tool = mcp_server.registered_tools["delete_memories"].fn

        with (
            patch(
                "reflectlog.application.memory.manager.MemoryManager.delete_records",
                side_effect=StorageError("Delete failure"),
            ),
            pytest.raises(StorageError) as exc_info,
        ):
            await delete_memories_tool([1], workspace_id=set_env_vars["WORKSPACE_ID"])

        assert "Failed to remove memory records" in str(exc_info.value)
        assert "Delete failure" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    async def test_delete_memory_not_found(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
        set_env_vars: dict[str, str],
    ):
        """Test delete_memories tool when the id is not found."""
        # Configure CachedEmbeddings mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        mock_usearch_engine.get_id_by_memory.return_value = None
        mock_usearch_engine.get_id_by_content.return_value = None
        mock_usearch_engine_class.return_value = mock_usearch_engine

        from reflectlog.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        delete_memories_tool = mcp_server.registered_tools["delete_memories"].fn

        # Should not raise any errors, just silently skip
        with patch(
            "reflectlog.application.memory.manager.MemoryManager.get_records",
            return_value=([], [99]),
        ):
            result = await delete_memories_tool(
                [99], workspace_id=set_env_vars["WORKSPACE_ID"]
            )
        assert result == {"removed": [], "missing": [99]}

        # Verify delete was never called
        mock_usearch_engine.delete.assert_not_called()


@pytest.mark.unit
class TestRunMethodCoverage:
    """Test coverage for run() method."""

    @patch.dict(
        os.environ,
        {"MCP_TRANSPORT": "http", "MCP_AUTH_TOKEN": "test-token"},
        clear=False,
    )
    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    def test_run_method_with_http_transport(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
        set_env_vars: dict[str, str],
    ):
        """Test run() method with HTTP transport."""
        # Configure CachedEmbeddings mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        mock_usearch_engine_class.return_value = mock_usearch_engine

        from reflectlog.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()

        # Mock the mcp.run() method to avoid actually starting the server
        mock_run = MagicMock()
        object.__setattr__(mcp_server.mcp, "run", mock_run)

        mcp_server.run()

        # Verify run was called
        mock_run.assert_called_once()

    @patch.dict(
        os.environ,
        {"MCP_TRANSPORT": "streamable-http", "MCP_AUTH_TOKEN": "test-token"},
        clear=False,
    )
    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    def test_run_method_with_streamable_http_transport(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
        set_env_vars: dict[str, str],
    ):
        """Test run() method with streamable HTTP transport."""
        # Configure CachedEmbeddings mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        mock_usearch_engine_class.return_value = mock_usearch_engine

        from reflectlog.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        mock_run = MagicMock()
        object.__setattr__(mcp_server.mcp, "run", mock_run)

        mcp_server.run()

        mock_run.assert_called_once_with(
            transport="streamable-http",
            port=mcp_server.config.port,
            host=mcp_server.config.host,
            path=mcp_server.config.path,
            uvicorn_config={"timeout_graceful_shutdown": None},
        )

    @patch.dict(
        os.environ,
        {
            "MCP_TRANSPORT": "streamable-http",
            "MCP_PORT": "8080",
            "MCP_HOST": "localhost",
            "MCP_PATH": "/custom",
            "MCP_AUTH_TOKEN": "test-token",
        },
        clear=False,
    )
    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    def test_run_method_with_custom_config(
        self,
        mock_cached_embedder_class: MagicMock,
        mock_usearch_engine_class: MagicMock,
        mock_embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
        set_env_vars: dict[str, str],
    ):
        """Test run() method with custom configuration."""
        # Configure CachedEmbeddings mock
        mock_cached_embedder = MagicMock()
        mock_cached_embedder.embedder = MagicMock()
        mock_cached_embedder_class.return_value = mock_cached_embedder

        mock_usearch_engine_class.return_value = mock_usearch_engine

        from reflectlog.application.mcp_server import FastMCPServer

        mcp_server = FastMCPServer()
        mock_run = MagicMock()
        object.__setattr__(mcp_server.mcp, "run", mock_run)

        mcp_server.run()

        mock_run.assert_called_once()


@pytest.mark.unit
class TestHttpAuthAndBind:
    @pytest.mark.parametrize("token", ["test-token", ""])
    def test_default_config_installs_auth_only_with_token(
        self, monkeypatch: pytest.MonkeyPatch, token: str
    ) -> None:
        from fastmcp import FastMCP

        from reflectlog.application.config.settings import Config
        from reflectlog.application.mcp_server import (
            FastMCPServer,
            _BearerTokenMiddleware,
        )
        from reflectlog.application.utils.security import SecretString

        monkeypatch.setenv("MCP_AUTH_TOKEN", token)
        with patch.object(FastMCP, "add_middleware", autospec=True) as add_middleware:
            server = FastMCPServer(
                Config(workspace_id="test", openrouter_api_key=SecretString("k"))
            )
            server._install_http_auth()
        if token:
            add_middleware.assert_called_once()
            assert isinstance(add_middleware.call_args.args[1], _BearerTokenMiddleware)
        else:
            add_middleware.assert_not_called()

    @pytest.mark.asyncio
    async def test_bearer_middleware_rejects_and_accepts(self) -> None:
        from fastmcp.exceptions import ToolError

        from reflectlog.application.mcp_server import _BearerTokenMiddleware

        middleware = _BearerTokenMiddleware("secret")
        with patch(
            "reflectlog.application.mcp_server.get_http_request",
            side_effect=RuntimeError("no request"),
        ):
            with pytest.raises(ToolError, match="Unauthorized"):
                await middleware.on_request(object(), lambda _ctx: "ok")

        request = MagicMock()
        request.headers.get.return_value = "Bearer wrong"
        with patch(
            "reflectlog.application.mcp_server.get_http_request",
            return_value=request,
        ):
            with pytest.raises(ToolError, match="Unauthorized"):
                await middleware.on_request(object(), lambda _ctx: "ok")

        request.headers.get.return_value = "Bearer secret"
        with patch(
            "reflectlog.application.mcp_server.get_http_request",
            return_value=request,
        ):
            result = await middleware.on_request(object(), lambda _ctx: "ok")
        assert result == "ok"

    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    def test_http_requires_auth_token(
        self,
        _cached: MagicMock,
        _usearch_cls: MagicMock,
        _embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
    ) -> None:
        from reflectlog.application.config.settings import Config
        from reflectlog.application.mcp_server import FastMCPServer
        from reflectlog.application.utils.security import SecretString

        _usearch_cls.return_value = mock_usearch_engine
        server = FastMCPServer(
            Config(
                workspace_id="test",
                openrouter_api_key=SecretString("k"),
                transport=TransportMode.HTTP,
                host="127.0.0.1",
            )
        )
        with patch.dict(os.environ, {"MCP_AUTH_TOKEN": ""}, clear=False):
            with pytest.raises(ConfigurationError, match="MCP_AUTH_TOKEN"):
                server.run()

    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    def test_http_refuses_bind_all_without_allow(
        self,
        _cached: MagicMock,
        _usearch_cls: MagicMock,
        _embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
    ) -> None:
        from reflectlog.application.config.settings import Config
        from reflectlog.application.mcp_server import FastMCPServer
        from reflectlog.application.utils.security import SecretString

        _usearch_cls.return_value = mock_usearch_engine
        server = FastMCPServer(
            Config(
                workspace_id="test",
                openrouter_api_key=SecretString("k"),
                transport=TransportMode.HTTP,
                host="0.0.0.0",
            )
        )
        with patch.dict(
            os.environ,
            {"MCP_AUTH_TOKEN": "token", "ALLOW_PUBLIC_BIND": "false"},
            clear=False,
        ):
            with pytest.raises(ConfigurationError, match="ALLOW_PUBLIC_BIND"):
                server.run()

    @patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings")
    @patch("reflectlog.application.memory.manager.USearchEngine")
    @patch("reflectlog.application.memory.manager.CachedEmbeddings")
    def test_http_refuses_ipv6_unspecified_without_allow(
        self,
        _cached: MagicMock,
        _usearch_cls: MagicMock,
        _embeddings: MagicMock,
        mock_usearch_engine: MagicMock,
    ) -> None:
        from reflectlog.application.config.settings import Config
        from reflectlog.application.mcp_server import FastMCPServer
        from reflectlog.application.utils.security import SecretString

        _usearch_cls.return_value = mock_usearch_engine
        server = FastMCPServer(
            Config(
                workspace_id="test",
                openrouter_api_key=SecretString("k"),
                transport=TransportMode.HTTP,
                host="::0",
            )
        )
        with patch.dict(
            os.environ,
            {"MCP_AUTH_TOKEN": "token", "ALLOW_PUBLIC_BIND": "false"},
            clear=False,
        ):
            with pytest.raises(ConfigurationError, match="ALLOW_PUBLIC_BIND"):
                server.run()
