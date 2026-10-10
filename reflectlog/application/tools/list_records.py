from typing import TYPE_CHECKING, override

from asyncer import asyncify

from reflectlog.core.enums import ToolName
from reflectlog.core.exceptions import StorageError

from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class ListRecordsTool(BaseTool):
    @override
    def get_name(self) -> str:
        return ToolName.LIST_RECORDS

    @override
    def get_instruction_snippet(self) -> str:
        return "    • list_records(limit=None, offset=0) -> dict\n      Page stable-id records in id order with a total; same cap as get_all."

    @override
    def get_handler(self) -> Callable[..., Awaitable[dict[str, object]]]:
        async def list_records(
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

        return list_records
