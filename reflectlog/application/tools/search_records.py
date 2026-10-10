from typing import TYPE_CHECKING, Annotated, override

from pydantic import Field

from reflectlog.core.enums import ToolName
from reflectlog.core.exceptions import SearchError

from ..constants import MAX_SEARCH_RECORDS_LIMIT
from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class SearchRecordsTool(BaseTool):
    @override
    def get_name(self) -> str:
        return ToolName.SEARCH_RECORDS

    @override
    def get_instruction_snippet(self) -> str:
        return "    • search_records(query, limit=None) -> dict\n      Search with stable ids, final scores and score_kind. Limit 1..50."

    @override
    def get_handler(self) -> Callable[..., Awaitable[dict[str, object]]]:
        async def search_records(
            query: Annotated[str, Field(min_length=1, max_length=1000)],
            limit: Annotated[
                int | None, Field(ge=1, le=MAX_SEARCH_RECORDS_LIMIT)
            ] = None,
        ) -> dict[str, object]:
            if not query.strip() or not 1 <= len(query) <= 1000:
                raise ValueError(
                    "Search query must be 1..1000 non-whitespace characters"
                )
            if limit is not None and not 1 <= limit <= MAX_SEARCH_RECORDS_LIMIT:
                raise ValueError("Search limit must be 1..50")
            page_size = self.config.search_limit if limit is None else limit
            try:
                self.log_invocation(
                    self.get_name(), query_length=len(query), limit=page_size
                )
                result = await self.memory.search_records(query, limit=page_size)
                return {
                    "results": [
                        {
                            "id": row.id,
                            "memory": row.content,
                            "score": row.score,
                            "created_at": row.created_at,
                        }
                        for row in result.records
                    ],
                    "score_kind": result.score_kind.value,
                    "limit": page_size,
                }
            except Exception as e:
                self._raise_tool_error(
                    self.get_name(),
                    e,
                    error_cls=SearchError,
                    message="Failed to search memory records",
                    query_length=len(query),
                )

        return search_records
