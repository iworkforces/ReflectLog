"""Health check tool implementation for ReflectLog Server."""

from typing import TYPE_CHECKING, Any, override

from reflectlog.core.enums import EngineReadiness, HealthStatus, ToolName

from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class HealthCheckTool(BaseTool):
    """Tool for health checking the server and its components."""

    @override
    def get_name(self) -> str:
        """Get the tool name."""
        return ToolName.HEALTH_CHECK

    @override
    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS."""
        return (
            "    • health_check() -> dict[str, Any]\n"
            "      Get server health status including component initialization state."
        )

    @override
    def get_handler(self) -> Callable[..., Awaitable[dict[str, Any]]]:
        """Get the async tool handler function."""

        async def health_check() -> dict[str, Any]:
            """Get server health status (async).

            Returns a health status dictionary with information about:
            - Overall server status
            - Workspace ID
            - Semantic engine initialization state
            - Tantivy full-text engine state
            - Configured reranker engine

            This tool is useful for monitoring and debugging server state.

            Returns:
                Dictionary containing health status information with keys:
                - status: Overall status string ("healthy", "degraded", or
                  "unhealthy"). Degraded means leftover replacement transitions
                  are still pending. This check is read-only; leftovers are
                  finished on startup or the next add persist.
                - workspace_id: The configured workspace identifier
                - semantic_engine: "initialized", "pending", or "not_initialized"
                - tantivy_engine: "initialized", "pending", or "disabled"
                - reranker_engine: The configured reranker type
                - hybrid_search_enabled: Whether full-text search is enabled
                - rrf_fusion_enabled: Whether RRF fusion is enabled
                - recency_boost_enabled: Whether recency boost is enabled
                - pending_replacement_transitions: Leftover replacement rows
                - startup_metrics: Optional dict with startup timing data (milliseconds)

            Examples:
                >>> health_check()
                {
                    "status": "healthy",
                    "workspace_id": "my-project",
                    "semantic_engine": "initialized",
                    "tantivy_engine": "initialized",
                    "reranker_engine": "cross_encoder",
                    "hybrid_search_enabled": True,
                    "rrf_fusion_enabled": True,
                    "recency_boost_enabled": True,
                    "startup_metrics": {
                        "numba_warmup": 250.5,
                        "server_initialization": 150.3,
                        "total_startup": 400.8
                    }
                }
            """
            engine_status = {
                "semantic_engine": EngineReadiness.UNKNOWN,
                "tantivy_engine": EngineReadiness.UNKNOWN,
            }
            try:
                self.log_invocation(ToolName.HEALTH_CHECK)
                engine_status = self.memory.search_engine_status()
                semantic_engine_status = engine_status["semantic_engine"]
                tantivy_engine_status = engine_status["tantivy_engine"]

                pending_count = self.memory.pending_intent_count()
                status = HealthStatus.HEALTHY
                if pending_count > 0:
                    status = HealthStatus.DEGRADED
                expected_up = {
                    EngineReadiness.PENDING,
                    EngineReadiness.NOT_INITIALIZED,
                }
                if semantic_engine_status in expected_up:
                    status = (
                        HealthStatus.UNHEALTHY
                        if self.config.eager_initialization
                        else HealthStatus.DEGRADED
                    )
                if tantivy_engine_status in expected_up:
                    status = (
                        HealthStatus.UNHEALTHY
                        if self.config.eager_initialization
                        else HealthStatus.DEGRADED
                    )
                health_status: dict[str, Any] = {
                    "status": status,
                    "workspace_id": self.config.workspace_id,
                    "semantic_engine": semantic_engine_status,
                    "tantivy_engine": tantivy_engine_status,
                    "reranker_engine": self.config.reranker_engine,
                    "hybrid_search_enabled": tantivy_engine_status
                    != EngineReadiness.DISABLED,
                    "rrf_fusion_enabled": self.config.enable_rrf_fusion,
                    "recency_boost_enabled": self.config.enable_recency_boost,
                    "pending_intent_count": pending_count,
                }

                # Add startup metrics if available
                if self.memory.startup_metrics is not None:
                    # Convert seconds to milliseconds for better readability
                    startup_metrics_ms = {
                        phase: duration * 1000
                        for phase, duration in self.memory.startup_metrics.items()
                    }
                    health_status["startup_metrics"] = startup_metrics_ms

                self.log_completion(
                    ToolName.HEALTH_CHECK, status=health_status["status"]
                )

                return health_status

            except Exception as e:
                self.log_error(ToolName.HEALTH_CHECK, e)
                # Return unhealthy status with diagnostic information
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "workspace_id": self.config.workspace_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    # Provide component states even during error
                    "diagnostics": {
                        **engine_status,
                        "reranker_engine": self.config.reranker_engine,
                        "hybrid_search_enabled": engine_status["tantivy_engine"]
                        != EngineReadiness.DISABLED,
                    },
                }

        return health_check
