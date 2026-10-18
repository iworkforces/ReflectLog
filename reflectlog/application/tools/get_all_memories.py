from typing import TYPE_CHECKING, override

from asyncer import asyncify

from reflectlog.core.enums import ToolName
from reflectlog.core.exceptions import StorageError

from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class GetAllMemoriesTool(BaseTool):
    @override
    def get_name(self) -> str:
        return ToolName.GET_ALL_MEMORIES

    @override
    def get_instruction_snippet(self) -> str:
        return (
            "    • get_all_memories(workspace_id: str, limit: int | None = None, offset: int = 0, after_id: int | None = None) -> dict\n"
            "      Page stable-id records in id order with a total; limit=0 returns the total only. "
            "Use after_id=0 to start skip-free traversal, then next_after_id to continue. "
            "Cursor mode rejects positive offsets and retains the cursor on empty pages and probes. "
            "Continuation survives eviction/restart, but is not a frozen snapshot across writes; "
            "restoring older database contents invalidates continuation."
        )

    @override
    def get_handler(self) -> Callable[..., Awaitable[dict[str, object]]]:
        async def get_all_memories(
            limit: int | None = None, offset: int = 0, after_id: int | None = None
        ) -> dict[str, object]:
            if after_id is not None:
                if type(after_id) is not int or after_id < 0:
                    raise ValueError("after_id must be a nonnegative integer")
                if offset > 0:
                    raise ValueError(
                        "after_id cannot be combined with a positive offset"
                    )
            start = max(0, offset)
            page_size = min(
                self.config.get_all_limit,
                self.config.get_all_limit if limit is None else max(0, limit),
            )
            next_after_id = after_id
            try:
                self.log_invocation(self.get_name(), limit=page_size, offset=start)
                if after_id is None:
                    records, total = await asyncify(
                        self.memory.get_page_records_with_total
                    )(limit=page_size, offset=start)
                    truncated = start + len(records) < total
                else:
                    page = await asyncify(self.memory.get_cursor_page)(
                        limit=page_size, after_id=after_id
                    )
                    records, total = list(page.records), page.total
                    truncated = page.truncated
                    next_after_id = page.next_after_id
                # limit=0 is the connectivity probe; an empty page is expected.
                if page_size and truncated:
                    self.logger.warning(
                        "get_all_memories truncated",
                        extra={
                            "tool": self.get_name(),
                            "returned": len(records),
                            "total": total,
                            "offset": start,
                            "limit": page_size,
                        },
                    )
                self.log_completion(self.get_name(), count=len(records), total=total)
                response: dict[str, object] = {
                    "records": [
                        {
                            "id": row.id,
                            "memory": row.content,
                            "created_at": row.created_at or None,
                        }
                        for row in records
                    ],
                    "total": total,
                    "offset": start,
                    "limit": page_size,
                    "truncated": truncated,
                }
                if after_id is not None:
                    response["next_after_id"] = next_after_id
                return response
            except Exception as e:
                self._raise_tool_error(
                    self.get_name(),
                    e,
                    error_cls=StorageError,
                    message="Failed to list memory records",
                    limit=page_size,
                    offset=start,
                )

        return get_all_memories
