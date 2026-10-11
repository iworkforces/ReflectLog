import inspect
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from reflectlog.application.memory.add_phases import AddResult
from reflectlog.application.memory.manager import (
    MemoryManager,
    RecordSearchResult,
    ScoredMemoryRecord,
)
from reflectlog.core.enums import SearchScoreKind
from tests.test_get_all_memories import AddResponse, Page, SearchResponse

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.integration
@pytest.mark.asyncio
async def test_demo_workflow(
    add_memories_tool: Callable[..., object],
    get_all_memories_tool: Callable[..., object],
    search_memories_tool: Callable[..., object],
) -> None:
    memories = [
        "Python tutorial for beginners",
        "Advanced Python programming",
        "JavaScript fundamentals",
        "Python data science guide",
        "Java enterprise patterns",
        "Python machine learning",
        "React frontend development",
        "Python web scraping",
        "Docker containerization",
        "Python automation scripts",
    ]
    workspace = "test_project"
    rows = [
        ScoredMemoryRecord(id=i, content=memory, score=1.0, created_at=None)
        for i, memory in enumerate(memories, 1)
    ]

    async def search(query: str, limit: int | None = None) -> RecordSearchResult:
        matching = tuple(row for row in rows if query.lower() in row.content.lower())
        return RecordSearchResult(matching[:limit], SearchScoreKind.NONE)

    with (
        patch.object(
            MemoryManager, "add_memories_async", return_value=AddResult(stored_count=10)
        ),
        patch.object(
            MemoryManager, "get_page_records_with_total", return_value=(rows, 10)
        ),
        patch.object(MemoryManager, "search_records", side_effect=search),
    ):
        added = add_memories_tool(memories, workspace_id=workspace)
        assert inspect.isawaitable(added)
        assert AddResponse.model_validate(await added).stored_count == 10
        listed = get_all_memories_tool(workspace_id=workspace)
        assert inspect.isawaitable(listed)
        page = Page.model_validate(await listed)
        assert [record.memory for record in page.records] == memories
        for query, expected in (("Python", 6), ("JavaScript", 1)):
            searched = search_memories_tool(query, workspace_id=workspace, limit=10)
            assert inspect.isawaitable(searched)
            response = SearchResponse.model_validate(await searched)
            assert len(response.results) == expected
            assert {row.memory for row in response.results} == {
                memory for memory in memories if query in memory
            }
            print(query, response.model_dump())
