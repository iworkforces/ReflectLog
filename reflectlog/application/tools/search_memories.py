from typing import TYPE_CHECKING, Annotated, override

from pydantic import Field

from reflectlog.core.enums import ToolName
from reflectlog.core.exceptions import SearchError

from ..constants import MAX_SEARCH_RECORDS_LIMIT
from .base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class SearchMemoriesTool(BaseTool):
    @override
    def get_name(self) -> str:
        return ToolName.SEARCH_MEMORIES

    @override
    def get_instruction_snippet(self) -> str:
        return "    • search_memories(query: str, workspace_id: str, limit: int | None = None, explain: bool = False) -> dict\n      Search with stable ids, final scores and score_kind. Limit 1..50. explain=true adds per-record explanation and request-local stage_outcomes without changing ranking. Semantic/fulltext contributions have one-based ranks and raw scores; null means absent from that backend's filtered fusion input. pre_rerank_score and its kind describe the score before reranking; final_score_kind describes the returned score. Outcomes: disabled, failed, unavailable, empty, skipped_single_hit, applied, degraded. No explanation fields are returned by default."

    @override
    def get_handler(self) -> Callable[..., Awaitable[dict[str, object]]]:
        async def search_memories(
            query: Annotated[str, Field(min_length=1, max_length=1000)],
            limit: Annotated[
                int | None, Field(ge=1, le=MAX_SEARCH_RECORDS_LIMIT)
            ] = None,
            explain: bool = False,
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
                start_time = self._log_operation_header(
                    self.get_name(), "SEARCH OPERATION"
                )
                if self.config.log_search_results_verbose:
                    self.logger.info("Query length", extra={"query_length": len(query)})
                result = (
                    await self.memory.search_records(
                        query, limit=page_size, explain=True
                    )
                    if explain
                    else await self.memory.search_records(query, limit=page_size)
                )
                result_count = len(result.records)
                if self.config.log_search_results_verbose:
                    self.logger.info(
                        "FINAL RESULTS", extra={"result_count": result_count}
                    )
                    self.logger.info(
                        "Returning results"
                        if result_count
                        else "No matching memories found",
                        extra={"result_count": result_count},
                    )
                _ = self._log_operation_footer(self.get_name(), start_time)
                self.log_completion(
                    self.get_name(), query_length=len(query), result_count=result_count
                )
                payload: dict[str, object] = {
                    "results": [
                        {
                            "id": row.id,
                            "memory": row.content,
                            "score": row.score,
                            "created_at": row.created_at or None,
                            **(
                                {
                                    "explanation": {
                                        "semantic": {
                                            "rank": row.explanation.semantic.rank,
                                            "score": row.explanation.semantic.score,
                                        }
                                        if row.explanation.semantic is not None
                                        else None,
                                        "fulltext": {
                                            "rank": row.explanation.fulltext.rank,
                                            "score": row.explanation.fulltext.score,
                                        }
                                        if row.explanation.fulltext is not None
                                        else None,
                                        "pre_rerank_score": row.explanation.pre_rerank_score,
                                        "pre_rerank_score_kind": row.explanation.pre_rerank_score_kind.value,
                                        "final_score_kind": row.explanation.final_score_kind.value,
                                    }
                                }
                                if explain and row.explanation is not None
                                else {}
                            ),
                        }
                        for row in result.records
                    ],
                    "score_kind": result.score_kind.value,
                    "limit": page_size,
                }
                if explain and result.stage_outcomes is not None:
                    payload["stage_outcomes"] = {
                        "semantic": result.stage_outcomes.semantic.value,
                        "fulltext": result.stage_outcomes.fulltext.value,
                        "reranker": result.stage_outcomes.reranker.value,
                    }
                return payload
            except Exception as e:
                self._raise_tool_error(
                    self.get_name(),
                    e,
                    error_cls=SearchError,
                    message="Failed to search memory records",
                    query_length=len(query),
                )

        return search_memories
