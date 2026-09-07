"""Add tool implementation for ReflectLog Server."""

from typing import TYPE_CHECKING, override

from reflectlog.core.enums import ToolName
from reflectlog.core.exceptions import StorageError

from ..utils.validation import validate_add_batch, validate_memories
from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class AddTool(BaseTool):
    """Tool for adding memories to memory storage."""

    @override
    def get_name(self) -> str:
        """Get the tool name."""
        return ToolName.ADD

    @override
    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS."""
        return (
            "    • add(memories: list[str], dry_run: bool = False) -> dict\n"
            "      Add memories with semantic embeddings. Empty lists are no-op.\n"
            "      Returns stored/skipped/replaced counts. dry_run previews replacements."
        )

    @override
    def get_handler(self) -> Callable[..., Awaitable[dict[str, object]]]:
        """Get the async tool handler function."""

        async def add(memories: list[str], dry_run: bool = False) -> dict[str, object]:
            """Add memories to the memory store with parallel processing (async).

            This tool stores one or more text memories in the memory store using
            semantic embeddings. Memories are processed concurrently for improved
            performance when adding multiple memories. Memories are stored with
            vector embeddings and will be available via get_all() or search().

            Note: Uses async add_memories_async() for concurrent processing of
            batch additions without blocking the event loop.

            Args:
                memories: List of memory strings to add to storage. Empty list
                    is treated as no-op. Each memory must be 1-30720 characters
                    and contain non-whitespace content.
                dry_run: If True, performs smart replacement detection without
                    actually storing memories. Useful for testing what changes
                    would occur. When enabled, returns AddResult without
                    modifying storage. Default is False (live mode).

            Returns:
                Dict with stored_count, skipped_count, replaced_count,
                replacements, and dry_run. Empty input is a no-op dict.

            Raises:
                ValueError: If any memory is invalid (empty, too long, whitespace-only).
                RuntimeError: If storage operation fails (not raised in dry_run mode).

            Examples:
                >>> add(["Hello", "World"])  # Processed in parallel
                >>> add(["Meeting notes: Discussed API design"])
                >>> add([])  # No-op, no error
                >>> add(["Test message"], dry_run=True)  # Check replacements without storing
            """
            # Handle empty list gracefully (no-op, no error)
            if not memories:
                self.log_invocation(ToolName.ADD, count=0)
                self.logger.info("Add called with empty list, nothing to store")
                return {
                    "stored_count": 0,
                    "skipped_count": 0,
                    "replaced_count": 0,
                    "replacements": [],
                    "dry_run": dry_run,
                }

            # Validate memories
            is_valid, error_msg = validate_memories(
                memories, self.config.min_memory_length, self.config.max_memory_length
            )
            if is_valid:
                is_valid, error_msg = validate_add_batch(
                    memories, self.config.max_add_batch, self.config.max_add_chars
                )

            if not is_valid:
                self.log_error(ToolName.ADD, ValueError(error_msg), count=len(memories))
                raise ValueError(f"Invalid memory: {error_msg}")

            # Log invocation and operation header
            mode_str = "DRY_RUN" if dry_run else "LIVE"
            self.log_invocation(
                ToolName.ADD, count=len(memories), mode=mode_str, dry_run=dry_run
            )

            total_chars = sum(len(m) for m in memories)
            start_time = self._log_operation_header(
                "add",
                f"ADD OPERATION: Storing {len(memories)} memory(ies) "
                f"({total_chars:,} total characters)",
                memory_count=len(memories),
                total_characters=total_chars,
            )

            self.logger.info(
                f"  {len(memories)} memory payload(s), {total_chars:,} characters",
                extra={
                    "tool": "add",
                    "memory_count": len(memories),
                    "total_characters": total_chars,
                },
            )

            # Store memories using async method for better concurrency
            try:
                result = await self.memory.add_memories_async(memories, dry_run=dry_run)

                stored_count = result.stored_count
                skipped_count = result.skipped_count
                replaced_count = result.replaced_count

                self.logger.info(
                    "─" * 50,
                    extra={"tool": "add", "section": "summary"},
                )

                # Build summary message
                summary_parts: list[str] = []
                if stored_count > 0:
                    summary_parts.append(f"{stored_count} stored")
                if replaced_count > 0:
                    summary_parts.append(f"{replaced_count} replaced")
                if skipped_count > 0:
                    summary_parts.append(f"{skipped_count} skipped (duplicates)")

                if summary_parts:
                    summary = ", ".join(summary_parts)
                    self.logger.info(
                        f"ADD SUMMARY: {summary}",
                        extra={
                            "tool": "add",
                            "stored_count": stored_count,
                            "replaced_count": replaced_count,
                            "skipped_count": skipped_count,
                        },
                    )
                else:
                    self.logger.info(
                        f"ADD COMPLETE: All {stored_count} memory(ies) stored successfully",
                        extra={
                            "tool": "add",
                            "stored_count": stored_count,
                        },
                    )

                if result.replacements:
                    self.logger.info(
                        f"  {len(result.replacements)} replacement(s)",
                        extra={
                            "tool": "add",
                            "action": "replacement",
                            "replacement_count": len(result.replacements),
                        },
                    )

                duration_ms = self._log_operation_footer("add", start_time)

                avg_time = duration_ms / len(memories) if memories else 0
                self.logger.info(
                    f"  {avg_time:.0f}ms/memory avg",
                    extra={"tool": "add", "avg_ms_per_memory": avg_time},
                )

                self.log_completion(
                    ToolName.ADD, requested=len(memories), stored=stored_count
                )
                return {
                    "stored_count": stored_count,
                    "skipped_count": skipped_count,
                    "replaced_count": replaced_count,
                    "replacements": [
                        {
                            "old_memory": replacement.old_memory,
                            "new_memory": replacement.new_memory,
                            "confidence": replacement.confidence,
                            "reason": replacement.reason,
                        }
                        for replacement in result.replacements
                    ],
                    "dry_run": dry_run,
                }

            except Exception as e:
                self._raise_tool_error(
                    "add",
                    e,
                    error_cls=StorageError,
                    message="Failed to add memories to memory store",
                    count=len(memories),
                )
                raise

        return add
