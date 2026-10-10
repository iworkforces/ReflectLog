from typing import TYPE_CHECKING, Annotated, override

from asyncer import asyncify
from pydantic import Field

from reflectlog.core.enums import ToolName
from reflectlog.core.exceptions import StorageError

from ..constants import MAX_RECORD_IDS
from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class RemoveRecordsTool(BaseTool):
    @override
    def get_name(self) -> str:
        return ToolName.REMOVE_RECORDS

    @override
    def get_instruction_snippet(self) -> str:
        return "    • remove_records(ids) -> dict\n      Remove up to 100 positive ids; report actually removed and missing ids."

    @override
    def get_handler(self) -> Callable[..., Awaitable[dict[str, object]]]:
        async def remove_records(
            ids: Annotated[
                list[Annotated[int, Field(ge=1)]], Field(max_length=MAX_RECORD_IDS)
            ],
        ) -> dict[str, object]:
            if len(ids) > MAX_RECORD_IDS or any(record_id < 1 for record_id in ids):
                raise ValueError("At most 100 positive record ids are allowed")
            try:
                self.log_invocation(self.get_name(), count=len(ids))
                removed, missing = await asyncify(self.memory.delete_records)(ids)
                return {"removed": removed, "missing": missing}
            except Exception as e:
                self._raise_tool_error(
                    self.get_name(),
                    e,
                    error_cls=StorageError,
                    message="Failed to remove memory records",
                    count=len(ids),
                )

        return remove_records
