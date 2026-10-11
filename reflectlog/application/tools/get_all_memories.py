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
        return "    • get_all_memories(workspace_id: str, limit: int | None = None, offset: int = 0) -> dict\n      Page stable-id records in id order with a total; limit=0 returns the total only."

    @override
    def get_handler(self) -> Callable[..., Awaitable[dict[str, object]]]:
        async def get_all_memories(
            limit: int | None = None, offset: int = 0
        ) -> dict[str, object]:
            start = max(0, offset)
            page_size = min(
                self.config.get_all_limit,
                self.config.get_all_limit if limit is None else max(0, limit),
            )
            try:
                self.log_invocation(self.get_name(), limit=page_size, offset=start)
                records, total = await asyncify(
                    self.memory.get_page_records_with_total
                )(limit=page_size, offset=start)
                # limit=0 is the connectivity probe; an empty page is expected.
                if page_size and start + len(records) < total:
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
                return {
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
                    "truncated": start + len(records) < total,
                }
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
