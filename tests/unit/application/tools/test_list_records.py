from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.tools.list_records import ListRecordsTool
from reflectlog.core.exceptions import InconsistentStateError, StorageError
from reflectlog.infrastructure.memory_store import MemoryRecord


@pytest.mark.parametrize(
    ("limit", "offset", "expected_limit", "expected_offset"),
    [(None, 0, 2, 0), (99, -1, 2, 0), (-1, 3, 0, 3)],
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
        [MemoryRecord(1, "workspace", "text")],
        5,
    )
    tool = ListRecordsTool(
        replace(mock_config, get_all_limit=2), mock_memory_manager, mock_tool_logger
    )
    result = await tool.get_handler()(limit, offset)
    assert result == {
        "records": [{"id": 1, "memory": "text", "created_at": None}],
        "total": 5,
        "offset": expected_offset,
        "limit": expected_limit,
        "truncated": True,
    }
    mock_memory_manager.get_page_records_with_total.assert_called_once_with(
        limit=expected_limit, offset=expected_offset
    )
    assert tool.get_name() == "list_records"
    assert "list_records" in tool.get_instruction_snippet()


@pytest.mark.parametrize("inconsistent", [False, True])
async def test_list_records_errors(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    inconsistent: bool,
) -> None:
    error = InconsistentStateError("split") if inconsistent else RuntimeError("failed")
    mock_memory_manager.get_page_records_with_total.side_effect = error
    tool = ListRecordsTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(
        InconsistentStateError if inconsistent else StorageError
    ) as raised:
        await tool.get_handler()()
    assert (
        (raised.value is error) if inconsistent else (raised.value.__cause__ is error)
    )
