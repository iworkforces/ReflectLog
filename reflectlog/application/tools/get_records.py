from typing import TYPE_CHECKING, Annotated, override

from asyncer import asyncify
from pydantic import Field

from reflectlog.core.enums import ToolName
from reflectlog.core.exceptions import StorageError

from ..constants import MAX_RECORD_IDS
from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class GetRecordsTool(BaseTool):
    @override
    def get_name(self) -> str:
        return ToolName.GET_RECORDS

    @override
    def get_instruction_snippet(self) -> str:
        return "    • get_records(ids) -> dict\n      Resolve up to 100 positive ids in request order; report missing ids."

    @override
    def get_handler(self) -> Callable[..., Awaitable[dict[str, object]]]:
        async def get_records(
            ids: Annotated[
                list[Annotated[int, Field(ge=1)]], Field(max_length=MAX_RECORD_IDS)
            ],
        ) -> dict[str, object]:
            if len(ids) > MAX_RECORD_IDS or any(record_id < 1 for record_id in ids):
                raise ValueError("At most 100 positive record ids are allowed")
            try:
                self.log_invocation(self.get_name(), count=len(ids))
                records, missing = await asyncify(self.memory.get_records)(ids)
                return {
                    "records": [
                        {
                            "id": row.id,
                            "memory": row.content,
                            "created_at": row.created_at or None,
                        }
                        for row in records
                    ],
                    "missing": missing,
                }
            except Exception as e:
                self._raise_tool_error(
                    self.get_name(),
                    e,
                    error_cls=StorageError,
                    message="Failed to retrieve memory records",
                    count=len(ids),
                )

        return get_records
