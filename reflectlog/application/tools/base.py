"""Base class for MCP tools."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Never

from reflectlog.core.exceptions import InconsistentStateError, ReflectLogError

from ..constants import LOG_SEPARATOR_LENGTH

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from reflectlog.core.logging import IStructuredLogger

    from ..config.settings import Config
    from ..memory.manager import MemoryManager


class BaseTool(ABC):
    """Abstract base class for MCP tools."""

    def __init__(
        self,
        config: Config,
        memory_manager: MemoryManager,
        logger: IStructuredLogger | None,
    ) -> None:
        """Initialize tool with dependencies.

        Args:
            config: Application configuration.
            memory_manager: Memory management instance.
            logger: Structured logger instance.
        """
        if logger is None:
            raise ValueError("logger is required")

        self.config = config
        self.memory = memory_manager
        self.logger = logger

    @abstractmethod
    def get_name(self) -> str:
        """Get the tool name for registration.

        Returns:
            Tool name as string.
        """
        pass

    @abstractmethod
    def get_handler(self) -> Callable[..., Awaitable[object]]:
        """Get the tool handler function.

        Returns:
            Callable that implements the tool logic.
        """
        pass

    @abstractmethod
    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS.

        Each tool must provide a formatted documentation snippet that will be
        included in the dynamic MCP_INSTRUCTIONS when the tool is registered.

        Returns:
            Formatted string for this tool's documentation.
            Expected format:
                '    • signature\\n      description line 1\\n      description line 2'

        Example:
            '    • add_memories(memories: list[str], workspace_id: str)\\n      Add memories with semantic embeddings.'
        """
        pass

    def log_invocation(self, tool_name: str, **kwargs: Any) -> None:
        """Log tool invocation with context.

        Args:
            tool_name: Name of the tool being invoked.
            **kwargs: Additional context to log.
        """
        self.logger.info(
            f"Tool '{tool_name}' invoked", extra={"tool": tool_name, **kwargs}
        )

    def log_completion(self, tool_name: str, **kwargs: Any) -> None:
        """Log tool completion with results.

        Args:
            tool_name: Name of the tool that completed.
            **kwargs: Results or additional context to log.
        """
        self.logger.info(
            f"Tool '{tool_name}' completed successfully",
            extra={"tool": tool_name, **kwargs},
        )

    def log_error(self, tool_name: str, error: Exception, **kwargs: Any) -> None:
        """Log tool error with context.

        Args:
            tool_name: Name of the tool that failed.
            error: The exception that occurred.
            **kwargs: Additional error context.
        """
        self.logger.error(
            f"Tool '{tool_name}' failed: {error}",
            extra={"tool": tool_name, "error": str(error), **kwargs},
        )

    def _log_operation_header(
        self, operation: str, title: str, **detail_kwargs: Any
    ) -> float:
        """Log operation header with separator and title.

        Emits a visual separator line followed by a titled log entry.
        Returns the current timestamp for later duration calculation.

        Args:
            operation: Tool name (e.g. "add", "search").
            title: Operation title line (e.g. "ADD OPERATION: Storing 3 memory(ies)").
            **detail_kwargs: Extra fields merged into the log entry.

        Returns:
            Start time as float (from time.time()) for use with _log_operation_footer.
        """
        import time

        start_time = time.time()
        self.logger.info(
            "=" * LOG_SEPARATOR_LENGTH,
            extra={"tool": operation, "section": "header"},
        )
        self.logger.info(
            title,
            extra={"tool": operation, **detail_kwargs},
        )
        return start_time

    def _log_operation_footer(
        self, operation: str, start_time: float, **detail_kwargs: Any
    ) -> float:
        """Log operation footer with timing and separator.

        Emits a timing summary line and a visual separator to close the
        operation block started by _log_operation_header.

        Args:
            operation: Tool name.
            start_time: Timestamp from _log_operation_header return value.
            **detail_kwargs: Extra fields merged into the timing log entry.

        Returns:
            Duration in milliseconds.
        """
        import time

        duration_ms = (time.time() - start_time) * 1000
        self.logger.info(
            f"Completed in {duration_ms:.0f}ms",
            extra={"tool": operation, "duration_ms": duration_ms, **detail_kwargs},
        )
        self.logger.info(
            "=" * LOG_SEPARATOR_LENGTH,
            extra={"tool": operation, "section": "footer"},
        )
        return duration_ms

    def _raise_tool_error(
        self,
        operation: str,
        error: Exception,
        *,
        error_cls: type[ReflectLogError],
        message: str,
        **kwargs: Any,
    ) -> Never:
        """Log error and raise a wrapped exception with from-chaining.

        Args:
            operation: Tool name.
            error: The original exception.
            error_cls: The ReflectLogError subclass to raise.
            message: Human-readable prefix for the error message.
            **kwargs: Extra context passed to log_error.

        Raises:
            ReflectLogError: Always raised (typed as Never).
        """
        self.log_error(operation, error, **kwargs)
        if isinstance(error, InconsistentStateError):
            raise error
        raise error_cls(f"{message}: {error}") from error
