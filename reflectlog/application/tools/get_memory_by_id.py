from typing import TYPE_CHECKING, Annotated, override

from asyncer import asyncify
from pydantic import Field

from reflectlog.core.enums import ToolName
from reflectlog.core.exceptions import StorageError

from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class GetMemoryByIdTool(BaseTool):
    @override
    def get_name(self) -> str:
        return ToolName.GET_MEMORY_BY_ID

    @override
    def get_instruction_snippet(self) -> str:
        return (
            "    • get_memory_by_id(id: int, workspace_id: str, include_successor: bool = False) -> dict\n"
            "      Returns {record: record or null}. With include_successor, missing records also return "
            "successor: {id: int} or null for currently active replacement content in this workspace. "
            "Single hop only; a re-added matching string has its current id, not historical identity."
        )

    @override
    def get_handler(self) -> Callable[..., Awaitable[dict[str, object]]]:
        async def get_memory_by_id(
            id: Annotated[int, Field(ge=1)],
            include_successor: bool = False,
        ) -> dict[str, object]:
            if id < 1:
                raise ValueError("A positive record id is required")
            try:
                self.log_invocation(self.get_name(), record_id=id)
                if include_successor:
                    result = await asyncify(self.memory.get_record_with_successor)(id)
                    if result.record is None:
                        return {
                            "record": None,
                            "successor": {"id": result.successor_id}
                            if result.successor_id is not None
                            else None,
                        }
                    records = [result.record]
                else:
                    records, _ = await asyncify(self.memory.get_records)([id])
                return {
                    "record": {
                        "id": records[0].id,
                        "memory": records[0].content,
                        "created_at": records[0].created_at or None,
                    }
                    if records
                    else None,
                }
            except Exception as e:
                self._raise_tool_error(
                    self.get_name(),
                    e,
                    error_cls=StorageError,
                    message="Failed to retrieve memory records",
                    record_id=id,
                )

        return get_memory_by_id
