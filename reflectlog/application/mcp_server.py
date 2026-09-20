"""ReflectLog Server - Refactored modular implementation."""

import hmac
from ipaddress import ip_address
import os
from typing import TYPE_CHECKING

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.middleware import Middleware
from fastmcp.utilities.logging import get_logger

from reflectlog.application.config.settings import Config, get_config
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.tools.add import AddTool
from reflectlog.application.tools.get_all import GetAllTool
from reflectlog.application.tools.health_check import HealthCheckTool
from reflectlog.application.tools.remove import RemoveTool
from reflectlog.application.tools.search import SearchTool
from reflectlog.core.enums import EmbedderProvider, TransportMode
from reflectlog.core.exceptions import ConfigurationError
from reflectlog.core.prompts import build_instructions

from .utils.logging import create_logger

if TYPE_CHECKING:
    from collections.abc import Callable

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
    "add": AddTool,
    "get_all": GetAllTool,
    "search": SearchTool,
    "remove": RemoveTool,
    "health_check": HealthCheckTool,
}


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
        init_msg = f"Initializing reflectlog MCP server [workspace_id={self.config.workspace_id}]"
        self.logger.info(init_msg)

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

        # Initialize memory manager
        self._memory_manager = MemoryManager(self.config, self.logger)

        # Initialize tools BEFORE creating FastMCP (to build dynamic instructions)
        self._initialize_tools()

        # Build dynamic instructions based on registered tools
        instructions = self._build_dynamic_instructions()

        # Initialize FastMCP with dynamic instructions
        self.mcp = FastMCP(name="reflectlog", instructions=instructions)
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

        # Initialize each permitted tool with dependencies
        self.tools: list[BaseTool] = []
        for tool_name in selected_names:
            tool_class = AVAILABLE_TOOL_CLASSES[tool_name]
            tool = tool_class(
                config=self.config,
                memory_manager=self._memory_manager,
                logger=self.logger,
            )
            self.tools.append(tool)

            self.logger.info(
                f"Initialized tool: {tool.get_name()}",
                extra={"tool": tool.get_name()},
            )

        if selected_names:
            self.logger.info(
                "Tool initialization complete",
                extra={"registered_tools": selected_names},
            )

    def _register_tools(self) -> None:
        """Register all tools with the FastMCP instance."""
        for tool in self.tools:
            # Get the handler function from the tool
            handler = tool.get_handler()

            # Register with FastMCP using the decorator
            _ = self.mcp.tool(handler)

            self.logger.info(
                f"Registered tool: {tool.get_name()}", extra={"tool": tool.get_name()}
            )

        self.logger.info(
            f"Registered {len(self.tools)} tools with FastMCP",
            extra={"tool_count": len(self.tools)},
        )
        from types import SimpleNamespace

        self.registered_tools = {
            tool.get_name(): SimpleNamespace(
                name=tool.get_name(), fn=tool.get_handler()
            )
            for tool in self.tools
        }

    def tool_fn(self, name: str) -> Callable[..., object]:
        """Return the registered handler for an MCP tool name."""
        for tool in self.tools:
            if tool.get_name() == name:
                return tool.get_handler()
        raise KeyError(name)

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

        instructions = build_instructions(tool_snippets)

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
            Tuple of (selected tool names, invalid tool names).
        """
        allowed = self.config.allowed_tools

        if allowed is None:
            return available_names, []

        available_set = set(available_names)
        selected: list[str] = []
        invalid: list[str] = []

        for token in allowed:
            canonical = self._canonicalize_tool_token(token, available_set)
            if canonical:
                if canonical not in selected:
                    selected.append(canonical)
            else:
                invalid.append(token)

        return selected, invalid

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
        )

    def close(self) -> None:
        """Gracefully shutdown the server and persist all data.

        This method ensures all data is properly saved before shutdown:
        1. Closes the MemoryManager (which persists USearch and Tantivy data)

        Should be called during graceful shutdown (e.g., on SIGINT/SIGTERM)
        to prevent data loss.
        """
        self.logger.info("Initiating graceful server shutdown...")

        try:
            self._memory_manager.close()
            from reflectlog.utility.http import HttpClientFactory

            HttpClientFactory.close_all_sync()
            self.logger.info("Server shutdown complete - all data persisted")
        except Exception as e:
            self.logger.error(
                f"Error during server shutdown: {e}",
                extra={"error": str(e)},
            )
            raise

    def set_startup_metrics(self, metrics: dict[str, float]) -> None:
        """Store startup timing metrics on the memory manager.

        Called by server.py after initialization to record per-phase timing
        data that the health_check tool exposes to callers.

        Args:
            metrics: Mapping of phase name to elapsed seconds.
        """
        self._memory_manager.startup_metrics = metrics

    @property
    def startup_metrics(self) -> dict[str, float] | None:
        return self._memory_manager.startup_metrics


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
    except RuntimeError as e:
        fallback_logger.error(f"Failed to start server: {e}")
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
