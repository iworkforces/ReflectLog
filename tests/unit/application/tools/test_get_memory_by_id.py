from unittest.mock import MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.tools.get_memory_by_id import GetMemoryByIdTool
from reflectlog.core.exceptions import InconsistentStateError, StorageError
from reflectlog.infrastructure.memory_store import MemoryRecord


async def test_get_records_shape(
    mock_config: Config, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
) -> None:
    mock_memory_manager.get_records.return_value = (
        [MemoryRecord(2, "workspace", "text", "time")],
        [8],
    )
    tool = GetMemoryByIdTool(mock_config, mock_memory_manager, mock_tool_logger)
    assert await tool.get_handler()(2) == {
        "record": {"id": 2, "memory": "text", "created_at": "time"},
    }
    mock_memory_manager.get_records.assert_called_once_with([2])
    assert tool.get_name() == "get_memory_by_id"
    assert "get_memory_by_id" in tool.get_instruction_snippet()


@pytest.mark.parametrize("ids", [0, -1])
async def test_get_records_validation(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    ids: int,
) -> None:
    tool = GetMemoryByIdTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(ValueError):
        await tool.get_handler()(ids)
    mock_memory_manager.get_records.assert_not_called()


@pytest.mark.parametrize("inconsistent", [False, True])
async def test_get_records_errors(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    inconsistent: bool,
) -> None:
    error = InconsistentStateError("split") if inconsistent else RuntimeError("failed")
    mock_memory_manager.get_records.side_effect = error
    tool = GetMemoryByIdTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(
        InconsistentStateError if inconsistent else StorageError
    ) as raised:
        await tool.get_handler()(1)
    assert (
        (raised.value is error) if inconsistent else (raised.value.__cause__ is error)
    )


@pytest.mark.parametrize("record_id", [8, 999])
async def test_missing_or_foreign_record(
    get_memory_tool_instance: GetMemoryByIdTool,
    mock_memory_manager: MagicMock,
    record_id: int,
) -> None:
    mock_memory_manager.get_records.return_value = ([], [record_id])
    assert await get_memory_tool_instance.get_handler()(record_id) == {"record": None}
    mock_memory_manager.get_records.assert_called_once_with([record_id])
