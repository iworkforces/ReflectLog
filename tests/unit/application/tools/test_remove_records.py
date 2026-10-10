from unittest.mock import MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.tools.remove_records import RemoveRecordsTool
from reflectlog.core.exceptions import InconsistentStateError, StorageError


async def test_remove_records_shape(
    mock_config: Config, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
) -> None:
    mock_memory_manager.delete_records.return_value = ([2], [8])
    tool = RemoveRecordsTool(mock_config, mock_memory_manager, mock_tool_logger)
    assert await tool.get_handler()([2, 8]) == {"removed": [2], "missing": [8]}
    mock_memory_manager.delete_records.assert_called_once_with([2, 8])
    assert tool.get_name() == "remove_records"
    assert "remove_records" in tool.get_instruction_snippet()


@pytest.mark.parametrize("ids", [[0], [-1], [1] * 101])
async def test_remove_records_validation(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    ids: list[int],
) -> None:
    tool = RemoveRecordsTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(ValueError):
        await tool.get_handler()(ids)
    mock_memory_manager.delete_records.assert_not_called()


@pytest.mark.parametrize("inconsistent", [False, True])
async def test_remove_records_errors(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    inconsistent: bool,
) -> None:
    error = InconsistentStateError("split") if inconsistent else RuntimeError("failed")
    mock_memory_manager.delete_records.side_effect = error
    tool = RemoveRecordsTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(
        InconsistentStateError if inconsistent else StorageError
    ) as raised:
        await tool.get_handler()([1])
    assert (
        (raised.value is error) if inconsistent else (raised.value.__cause__ is error)
    )
