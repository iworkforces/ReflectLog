from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.tools.get_all_memories import GetAllMemoriesTool
from reflectlog.core.exceptions import InconsistentStateError, StorageError
from reflectlog.core.types import CursorPage
from reflectlog.infrastructure.memory_store import MemoryRecord


@pytest.mark.parametrize(
    ("limit", "offset", "expected_limit", "expected_offset"),
    [(None, 0, 2, 0), (99, -1, 2, 0), (-1, 3, 0, 3), (0, 0, 0, 0), (0, 5, 0, 5)],
)
async def test_list_records_paging(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    limit: int | None,
    offset: int,
    expected_limit: int,
    expected_offset: int,
) -> None:
    mock_memory_manager.get_page_records_with_total.return_value = (
        [MemoryRecord(1, "workspace", "text")] if expected_limit else [],
        5,
    )
    tool = GetAllMemoriesTool(
        replace(mock_config, get_all_limit=2), mock_memory_manager, mock_tool_logger
    )
    result = await tool.get_handler()(limit, offset)
    assert result == {
        "records": [{"id": 1, "memory": "text", "created_at": None}]
        if expected_limit
        else [],
        "total": 5,
        "offset": expected_offset,
        "limit": expected_limit,
        "truncated": expected_offset < 5,
    }
    mock_memory_manager.get_page_records_with_total.assert_called_once_with(
        limit=expected_limit, offset=expected_offset
    )
    assert tool.get_name() == "get_all_memories"
    assert "get_all_memories" in tool.get_instruction_snippet()
    if result["truncated"] and expected_limit:
        mock_tool_logger.warning.assert_called_once_with(
            "get_all_memories truncated",
            extra={
                "tool": "get_all_memories",
                "returned": 1,
                "total": 5,
                "offset": expected_offset,
                "limit": expected_limit,
            },
        )
    else:
        mock_tool_logger.warning.assert_not_called()


@pytest.mark.parametrize("inconsistent", [False, True])
async def test_list_records_errors(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    inconsistent: bool,
) -> None:
    error = InconsistentStateError("split") if inconsistent else RuntimeError("failed")
    mock_memory_manager.get_page_records_with_total.side_effect = error
    tool = GetAllMemoriesTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(
        InconsistentStateError if inconsistent else StorageError
    ) as raised:
        await tool.get_handler()()
    assert (
        (raised.value is error) if inconsistent else (raised.value.__cause__ is error)
    )


@pytest.mark.parametrize("after_id", [-1, True, 1.5, "1"])
async def test_cursor_rejects_invalid_ids(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    after_id: object,
) -> None:
    tool = GetAllMemoriesTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(ValueError):
        await tool.get_handler()(after_id=after_id)


async def test_cursor_rejects_positive_offset(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
) -> None:
    tool = GetAllMemoriesTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(ValueError):
        await tool.get_handler()(after_id=0, offset=1)


@pytest.mark.parametrize("limit", [0, 1, 99, None])
async def test_cursor_response_uses_coherent_manager_page(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    limit: int | None,
) -> None:
    records = (MemoryRecord(9, "workspace", "text"),) if limit != 0 else ()
    mock_memory_manager.get_cursor_page.return_value = CursorPage(
        records, 5, True, 9 if records else 7
    )
    tool = GetAllMemoriesTool(
        replace(mock_config, get_all_limit=2), mock_memory_manager, mock_tool_logger
    )
    result = await tool.get_handler()(limit=limit, after_id=7)
    assert result == {
        "records": [{"id": 9, "memory": "text", "created_at": None}] if records else [],
        "total": 5,
        "offset": 0,
        "limit": min(2, limit) if limit is not None else 2,
        "truncated": True,
        "next_after_id": 9 if records else 7,
    }
    mock_memory_manager.get_cursor_page.assert_called_once_with(
        limit=min(2, limit) if limit is not None else 2, after_id=7
    )
    mock_memory_manager.get_page_records_with_total.assert_not_called()
