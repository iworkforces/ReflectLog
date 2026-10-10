from unittest.mock import AsyncMock, MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import RecordSearchResult, ScoredMemoryRecord
from reflectlog.application.tools.search_records import SearchRecordsTool
from reflectlog.core.enums import SearchScoreKind
from reflectlog.core.exceptions import InconsistentStateError, SearchError


async def test_search_records_shape(
    mock_config: Config, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
) -> None:
    mock_memory_manager.search_records = AsyncMock(
        return_value=RecordSearchResult(
            (ScoredMemoryRecord(7, "text", 0.03, None),), SearchScoreKind.RRF
        )
    )
    tool = SearchRecordsTool(mock_config, mock_memory_manager, mock_tool_logger)
    result = await tool.get_handler()("query", 2)
    assert result == {
        "results": [{"id": 7, "memory": "text", "score": 0.03, "created_at": None}],
        "score_kind": "rrf",
        "limit": 2,
    }
    mock_memory_manager.search_records.assert_awaited_once_with("query", limit=2)


@pytest.mark.parametrize(
    ("query", "limit"),
    [("", None), ("  ", None), ("x" * 1001, None), ("x", 0), ("x", 51)],
)
async def test_search_records_validation(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    query: str,
    limit: int | None,
) -> None:
    tool = SearchRecordsTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(ValueError):
        await tool.get_handler()(query, limit)
    mock_memory_manager.search_records.assert_not_called()


@pytest.mark.parametrize("inconsistent", [False, True])
async def test_search_records_errors(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    inconsistent: bool,
) -> None:
    error = InconsistentStateError("split") if inconsistent else RuntimeError("failed")
    mock_memory_manager.search_records = AsyncMock(side_effect=error)
    tool = SearchRecordsTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(
        InconsistentStateError if inconsistent else SearchError
    ) as raised:
        await tool.get_handler()("query")
    assert (
        (raised.value is error) if inconsistent else (raised.value.__cause__ is error)
    )


async def test_search_records_default_limit(
    mock_config: Config, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
) -> None:
    mock_memory_manager.search_records = AsyncMock(
        return_value=RecordSearchResult((), SearchScoreKind.NONE)
    )
    tool = SearchRecordsTool(mock_config, mock_memory_manager, mock_tool_logger)
    result = await tool.get_handler()("query")
    assert result == {
        "results": [],
        "score_kind": "none",
        "limit": mock_config.search_limit,
    }
    assert tool.get_name() == "search_records"
    assert "search_records" in tool.get_instruction_snippet()
