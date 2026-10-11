"""ReflectLog Server - Refactored modular implementation."""

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
import hmac
from ipaddress import ip_address
import os
from typing import TYPE_CHECKING, Annotated, Any, cast

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.middleware import Middleware
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from reflectlog.application.config.settings import Config, get_config
from reflectlog.application.constants import MAX_RECORD_IDS, MAX_SEARCH_RECORDS_LIMIT
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.memory.workspace_registry import WorkspaceRegistry
from reflectlog.application.tools.add_memories import AddMemoriesTool
from reflectlog.application.tools.delete_memories import DeleteMemoriesTool
from reflectlog.application.tools.get_all_memories import GetAllMemoriesTool
from reflectlog.application.tools.get_memory_by_id import GetMemoryByIdTool
from reflectlog.application.tools.search_memories import SearchMemoriesTool
from reflectlog.core.enums import EmbedderProvider, TransportMode
from reflectlog.core.exceptions import ConfigurationError
from reflectlog.core.prompts import build_instructions

from .utils.logging import create_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from reflectlog.application.tools.base import BaseTool


class _BearerTokenMiddleware(Middleware):
    """Reject HTTP MCP requests that lack the configured bearer token."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token

    async def on_request(
        self, context: object, call_next: Callable[..., object]
    ) -> object:
        try:
            request = get_http_request()
        except Exception as exc:
            raise ToolError("Unauthorized") from exc
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {self._token}"
        if not hmac.compare_digest(auth.encode("utf-8"), expected.encode("utf-8")):
            raise ToolError("Unauthorized")
        import inspect

        result = call_next(context)
        if inspect.isawaitable(result):
            return await result
        return result


# Canonical registry of available MCP tool implementations.
AVAILABLE_TOOL_CLASSES: dict[str, type[BaseTool]] = {
    "add_memories": AddMemoriesTool,
    "search_memories": SearchMemoriesTool,
    "get_all_memories": GetAllMemoriesTool,
    "get_memory_by_id": GetMemoryByIdTool,
    "delete_memories": DeleteMemoriesTool,
}

TOOL_INSTRUCTIONS = {
    "add_memories": "    • add_memories(memories: list[str], workspace_id: str, dry_run: bool = False, allow_replace: bool = True) -> dict\n"
    "      Add memories with semantic embeddings. Empty lists are no-op.\n"
    "      Returns stored/skipped/replaced counts. dry_run previews replacements.",
    "get_all_memories": "    • get_all_memories(workspace_id: str, limit: int | None = None, offset: int = 0) -> dict\n"
    "      Page stored records. Default cap 1000. Returns records, "
    "total, offset, limit, truncated.",
    "search_memories": "    • search_memories(query: str, workspace_id: str, limit: int | None = None) -> dict\n      Search stable ids with final scores and score_kind. Limit 1..50.",
    "get_memory_by_id": "    • get_memory_by_id(id: int, workspace_id: str) -> dict\n      Return a record or null when missing in this workspace.",
    "delete_memories": "    • delete_memories(ids: list[int], workspace_id: str) -> dict\n      Report actually removed and missing ids.",
}


@dataclass
class _RegisteredTool:
    name: str
    fn: Callable[..., Awaitable[object]]

    def get_name(self) -> str:
        return self.name

    def get_handler(self) -> Callable[..., Awaitable[object]]:
        return self.fn

    def get_instruction_snippet(self) -> str:
        return TOOL_INSTRUCTIONS[self.name]


class FastMCPServer:
    """Orchestrator for the ReflectLog Server.

    This class coordinates the initialization and registration of all components:
    - Configuration management
    - Memory storage
    - MCP tools
    - Server transport
    """

    def __init__(self, server_config: Config | None = None) -> None:
        """Initialize the MCP server with all components.

        Args:
            server_config: Configuration instance (defaults to singleton).
        """
        super().__init__()
        self.config = server_config if server_config is not None else get_config()

        # Initialize structured logger
        self.logger = create_logger(
            __name__, self.config.workspace_id, self.config.log_level
        )

        # Log initialization
        self.logger.info("Initializing reflectlog MCP server")

        self.logger.info(
            f"transport={self.config.transport}, port={self.config.port}, "
            f"log_level={self.config.log_level}"
        )

        match self.config.embedder_provider:
            case EmbedderProvider.OPENAI:
                embedding_dimensions = self.config.embedding_dims
            case EmbedderProvider.LANGCHAIN:
                embedding_dimensions = self.config.qwen_embedding_dims
            case EmbedderProvider.WEMM:
                embedding_dimensions = self.config.wemm_embedding_dims
        self.logger.info(f"embedding_dims={embedding_dimensions}")

        self._startup_metrics: dict[str, float] | None = None
        self._registry = WorkspaceRegistry(
            self.config, manager_factory=self._create_manager
        )
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._serving_task: asyncio.Task[object] | None = None

        # Initialize tools BEFORE creating FastMCP (to build dynamic instructions)
        self._initialize_tools()

        # Build dynamic instructions based on registered tools
        instructions = self._build_dynamic_instructions()

        # Initialize FastMCP with dynamic instructions
        self.mcp = cast("Callable[..., FastMCP]", FastMCP)(
            name="reflectlog-mcp", instructions=instructions, lifespan=self._lifespan
        )
        self._http_auth_installed = False
        self._install_http_auth()

        # Register tools with FastMCP
        self._register_tools()

        self.logger.info("ReflectLog Server initialized successfully")

    def _initialize_tools(self) -> None:
        """Initialize all MCP tool instances."""
        available_names = list(AVAILABLE_TOOL_CLASSES.keys())

        selected_names, invalid_names = self._determine_tool_selection(available_names)

        if invalid_names:
            self.logger.warning(
                "Ignoring unknown tool identifiers from ALLOWED_TOOLS",
                extra={"invalid_tools": sorted(invalid_names)},
            )

        if not selected_names:
            self.logger.warning(
                "No MCP tools selected for registration. Server will start without tools.",
                extra={"available_tools": available_names},
            )

        handlers = self._handlers()
        self.tools: list[_RegisteredTool] = []
        for tool_name in selected_names:
            self.tools.append(_RegisteredTool(tool_name, handlers[tool_name]))

            self.logger.info(
                f"Initialized tool: {tool_name}",
                extra={"tool": tool_name},
            )

        if selected_names:
            self.logger.info(
                "Tool initialization complete",
                extra={"registered_tools": selected_names},
            )

    def _register_tools(self) -> None:
        """Register all tools with the FastMCP instance."""
        for tool in self.tools:
            _ = self.mcp.tool(tool.get_handler(), name=tool.get_name())

            self.logger.info(
                f"Registered tool: {tool.get_name()}", extra={"tool": tool.get_name()}
            )

        self.logger.info(
            f"Registered {len(self.tools)} tools with FastMCP",
            extra={"tool_count": len(self.tools)},
        )
        self.registered_tools = {tool.get_name(): tool for tool in self.tools}

    def _handlers(self) -> dict[str, Callable[..., Awaitable[object]]]:
        return {
            "add_memories": self._add_memories,
            "search_memories": self._search_memories,
            "get_all_memories": self._get_all_memories,
            "get_memory_by_id": self._get_memory_by_id,
            "delete_memories": self._delete_memories,
        }

    def tool_fn(self, name: str) -> Callable[..., object]:
        """Return the registered handler for an MCP tool name."""
        return self.registered_tools[name].fn

    def _create_manager(self, config: Config) -> MemoryManager:
        logger = create_logger(__name__, config.workspace_id, config.log_level)
        manager = MemoryManager(config, logger)
        manager.startup_metrics = self._startup_metrics
        return manager

    def _handler(self, name: str, manager: MemoryManager) -> Callable[..., Any]:
        tool = AVAILABLE_TOOL_CLASSES[name](
            replace(self.config, workspace_id=manager.config.workspace_id),
            manager,
            self.logger,
        )
        return tool.get_handler()

    @staticmethod
    async def _finish_tool[T](operation: Awaitable[T]) -> T:
        task = asyncio.ensure_future(operation)
        try:
            _ = await asyncio.wait({task})
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                while not task.done():
                    try:
                        _ = await asyncio.wait({task})
                    except asyncio.CancelledError:
                        continue
            _ = task.exception() if not task.cancelled() else None
            raise
        return task.result()

    async def _add_memories(
        self,
        memories: list[str],
        workspace_id: str,
        dry_run: bool = False,
        allow_replace: bool = True,
    ) -> dict[str, object]:
        async with self._registry.acquire(workspace_id) as manager:
            return await self._finish_tool(
                self._handler("add_memories", manager)(memories, dry_run, allow_replace)
            )

    async def _search_memories(
        self,
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=1000,
                description="Search query for semantic matching",
            ),
        ],
        workspace_id: str,
        limit: Annotated[int | None, Field(ge=1, le=MAX_SEARCH_RECORDS_LIMIT)] = None,
    ) -> dict[str, object]:
        async with self._registry.acquire(workspace_id) as manager:
            return await self._finish_tool(
                self._handler("search_memories", manager)(query, limit)
            )

    async def _get_all_memories(
        self, workspace_id: str, limit: int | None = None, offset: int = 0
    ) -> dict[str, object]:
        async with self._registry.acquire(workspace_id) as manager:
            return await self._finish_tool(
                self._handler("get_all_memories", manager)(limit, offset)
            )

    async def _get_memory_by_id(
        self,
        id: Annotated[int, Field(ge=1)],
        workspace_id: str,
    ) -> dict[str, object]:
        async with self._registry.acquire(workspace_id) as manager:
            return await self._finish_tool(
                self._handler("get_memory_by_id", manager)(id)
            )

    async def _delete_memories(
        self,
        ids: Annotated[
            list[Annotated[int, Field(ge=1)]], Field(max_length=MAX_RECORD_IDS)
        ],
        workspace_id: str,
    ) -> dict[str, object]:
        async with self._registry.acquire(workspace_id) as manager:
            return await self._finish_tool(
                self._handler("delete_memories", manager)(ids)
            )

    @asynccontextmanager
    async def _lifespan(self, _server: FastMCP) -> AsyncGenerator[None]:
        self._serving_task = asyncio.current_task()
        reaper = asyncio.create_task(self._registry.run_reaper())
        try:
            yield
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.aclose()
                finally:
                    _ = reaper.cancel()
                    cancelled = False
                    while not reaper.done():
                        try:
                            _ = await asyncio.wait({reaper})
                        except asyncio.CancelledError:
                            cancelled = True
                    with suppress(asyncio.CancelledError):
                        reaper.result()
                    self._serving_task = None
                    if cancelled:
                        raise asyncio.CancelledError

    def _build_dynamic_instructions(self) -> str:
        """Build MCP instructions dynamically from registered tools.

        Assembles instructions by collecting snippets from each initialized tool
        and passing them to the build_instructions() function.

        Returns:
            Complete MCP instructions string with only registered tools documented.
        """
        tool_snippets = [
            (tool.get_name(), tool.get_instruction_snippet()) for tool in self.tools
        ]

        instructions = (
            f"{build_instructions(tool_snippets)}\n\n"
            "Every tool requires workspace_id: str as an argument."
        )

        self.logger.info(
            f"Built dynamic instructions for {len(self.tools)} tool(s)",
            extra={
                "tool_count": len(self.tools),
                "tools": [tool.get_name() for tool in self.tools],
            },
        )

        return instructions

    def _determine_tool_selection(
        self, available_names: list[str]
    ) -> tuple[list[str], list[str]]:
        """Determine which tools should be initialized based on configuration.

        Args:
            available_names: List of canonical tool identifiers that ship with the server.

        Returns:
            Tuple of (selected tool names in canonical order, invalid tool names).
        """
        allowed = self.config.allowed_tools

        if allowed is None:
            return available_names, []

        available_set = set(available_names)
        selected: set[str] = set()
        invalid: list[str] = []

        for token in allowed:
            canonical = self._canonicalize_tool_token(token, available_set)
            if canonical:
                selected.add(canonical)
            else:
                invalid.append(token)

        return [name for name in available_names if name in selected], invalid

    @staticmethod
    def _canonicalize_tool_token(token: str, available_set: set[str]) -> str | None:
        """Map a token from configuration to a canonical tool name."""
        normalized = token.strip().lower().replace("-", "_")
        if not normalized:
            return None

        if normalized in available_set:
            return normalized

        collapsed = normalized.replace("_", "")
        for name in available_set:
            if name.replace("_", "") == collapsed:
                return name

        suffixes = ("_tool", "tool")
        for suffix in suffixes:
            if normalized.endswith(suffix):
                base = normalized[: -len(suffix)]
                return FastMCPServer._canonicalize_tool_token(base, available_set)

        return None

    @staticmethod
    def _is_unspecified_bind(host: str) -> bool:
        """Return True when ``host`` is a bind-all / unspecified address."""
        candidate = host.strip()
        if candidate.startswith("[") and candidate.endswith("]"):
            candidate = candidate[1:-1]
        try:
            return ip_address(candidate).is_unspecified
        except ValueError:
            return False

    def _install_http_auth(self) -> None:
        """Attach bearer middleware when a token is configured for HTTP."""
        if self._http_auth_installed or self.config.transport == TransportMode.STDIO:
            return
        auth_token = os.environ.get("MCP_AUTH_TOKEN", "").strip()
        if not auth_token:
            return
        _ = self.mcp.add_middleware(_BearerTokenMiddleware(auth_token))
        self._http_auth_installed = True

    def run(self) -> None:
        """Start the FastMCP server with configured transport.

        The transport configuration is determined by:
        1. Environment variables (MCP_TRANSPORT, MCP_PORT, MCP_HOST, MCP_PATH)
        2. Config defaults
        """
        transport = self.config.transport

        if transport == TransportMode.STDIO:
            self.logger.info("Running MCP server with stdio transport")
            self.mcp.run(transport=TransportMode.STDIO)
            return

        allow_public = os.environ.get("ALLOW_PUBLIC_BIND", "false").lower() == "true"
        if self._is_unspecified_bind(self.config.host) and not allow_public:
            raise ConfigurationError(
                f"Refusing to bind {self.config.host} without ALLOW_PUBLIC_BIND=true"
            )
        auth_token = os.environ.get("MCP_AUTH_TOKEN", "").strip()
        if not auth_token:
            raise ConfigurationError(
                "MCP_AUTH_TOKEN is required for non-stdio transports"
            )
        self._install_http_auth()
        self.logger.info(
            f"Running MCP server with {transport} transport",
            extra={
                "host": self.config.host,
                "port": self.config.port,
                "path": self.config.path,
            },
        )
        self.mcp.run(
            transport=transport,
            port=self.config.port,
            host=self.config.host,
            path=self.config.path,
            uvicorn_config={"timeout_graceful_shutdown": None},
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        task = self._close_task
        if task is None or task.done():
            task = asyncio.create_task(self._close_resources())
            self._close_task = task
        cancelled = False
        with anyio.CancelScope(shield=True):
            while not task.done():
                try:
                    _ = await asyncio.wait({task})
                except asyncio.CancelledError:
                    cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    @property
    def shutdown_started(self) -> bool:
        return self._close_task is not None

    def cancel_serving(self) -> bool:
        task = self._serving_task
        if task is None or task.done():
            return False
        _ = task.cancel()
        return True

    async def _close_resources(self) -> None:
        self.logger.info("Initiating graceful server shutdown...")
        try:
            from reflectlog.utility.http import HttpClientFactory

            errors: list[Exception] = []
            try:
                await self._registry.close()
            except Exception as error:
                errors.append(error)
            try:
                await HttpClientFactory.close_all()
            except Exception as error:
                errors.append(error)
            if len(errors) == 1:
                raise errors[0]
            if len(errors) > 1:
                raise ExceptionGroup("Server shutdown incomplete", errors)
            self._closed = True
            self.logger.info("Server shutdown complete - all data persisted")
        except Exception as e:
            self.logger.error(
                f"Error during server shutdown: {e}",
                extra={"error": str(e)},
            )
            raise

    def close(self) -> None:
        try:
            _ = anyio.get_current_task()
        except anyio.NoEventLoopError:
            pass
        else:
            raise RuntimeError("Use await aclose() inside an async event loop")
        anyio.run(self.aclose)

    def set_startup_metrics(self, metrics: dict[str, float]) -> None:
        """Store startup timing metrics on the memory manager.

        Called by server.py after initialization to record per-phase timing
        metadata retained on workspace managers.

        Args:
            metrics: Mapping of phase name to elapsed seconds.
        """
        self._startup_metrics = metrics

    @property
    def startup_metrics(self) -> dict[str, float] | None:
        return self._startup_metrics


def main() -> None:
    """Entry point for the ReflectLog server.

    This function:
    1. Loads configuration from environment
    2. Initializes the server
    3. Starts the transport
    4. Handles graceful shutdown on SIGINT/SIGTERM
    """
    # Create fallback logger once for exception handling
    fallback_logger = get_logger(__name__)
    server: FastMCPServer | None = None

    try:
        # Configuration is loaded automatically via the singleton
        server = FastMCPServer()
        server.run()
        server.close()
    except RuntimeError as e:
        fallback_logger.error(f"Failed to start server: {e}")
        if server is not None:
            server.close()
        raise
    except KeyboardInterrupt:
        fallback_logger.info("Server shutdown requested (Ctrl+C)")
        if server is not None:
            server.close()
    except Exception as e:
        fallback_logger.error(f"Unexpected error: {e}")
        if server is not None:
            server.close()
        raise


if __name__ == "__main__":
    main()
