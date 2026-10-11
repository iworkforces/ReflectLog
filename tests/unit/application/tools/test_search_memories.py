from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import RecordSearchResult, ScoredMemoryRecord
from reflectlog.application.tools.search_memories import SearchMemoriesTool
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
    tool = SearchMemoriesTool(mock_config, mock_memory_manager, mock_tool_logger)
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
    tool = SearchMemoriesTool(mock_config, mock_memory_manager, mock_tool_logger)
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
    tool = SearchMemoriesTool(mock_config, mock_memory_manager, mock_tool_logger)
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
    tool = SearchMemoriesTool(mock_config, mock_memory_manager, mock_tool_logger)
    result = await tool.get_handler()("query")
    assert result == {
        "results": [],
        "score_kind": "none",
        "limit": mock_config.search_limit,
    }
    assert tool.get_name() == "search_memories"
    assert "search_memories" in tool.get_instruction_snippet()


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("has_results", [False, True])
async def test_search_logs_only_metadata(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    verbose: bool,
    has_results: bool,
) -> None:
    query = "private query text"
    records = (
        (ScoredMemoryRecord(9, "private memory text", 0.7, None),)
        if has_results
        else ()
    )
    mock_memory_manager.search_records.return_value = RecordSearchResult(
        records, SearchScoreKind.RRF
    )
    tool = SearchMemoriesTool(
        replace(mock_config, log_search_results_verbose=verbose),
        mock_memory_manager,
        mock_tool_logger,
    )

    result = await tool.get_handler()(query)

    assert result["results"] == [
        {"id": row.id, "memory": row.content, "score": row.score, "created_at": None}
        for row in records
    ]
    calls = mock_tool_logger.info.call_args_list
    assert any("invoked" in str(call) for call in calls)
    assert any("completed" in str(call) for call in calls)
    assert "private query text" not in str(mock_tool_logger.mock_calls)
    assert "private memory text" not in str(mock_tool_logger.mock_calls)
    assert any("FINAL RESULTS" in str(call) for call in calls) is verbose
    if verbose:
        assert any("result_count" in call.kwargs.get("extra", {}) for call in calls)


async def test_search_error_logs_query_length_not_text(
    search_tool_instance: SearchMemoriesTool,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
) -> None:
    mock_memory_manager.search_records.side_effect = RuntimeError("backend failed")
    with pytest.raises(SearchError, match="Failed to search memory records"):
        await search_tool_instance.get_handler()("sensitive query")
    assert mock_tool_logger.error.called
    assert "sensitive query" not in str(mock_tool_logger.mock_calls)
    assert any(
        "query_length" in call.kwargs.get("extra", {})
        for call in mock_tool_logger.error.call_args_list
    )
