from unittest.mock import MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.tools.delete_memories import DeleteMemoriesTool
from reflectlog.core.exceptions import InconsistentStateError, StorageError


async def test_delete_memories_shape(
    mock_config: Config, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
) -> None:
    mock_memory_manager.delete_records.return_value = ([2], [8])
    tool = DeleteMemoriesTool(mock_config, mock_memory_manager, mock_tool_logger)
    assert await tool.get_handler()([2, 8]) == {"removed": [2], "missing": [8]}
    mock_memory_manager.delete_records.assert_called_once_with([2, 8])
    assert tool.get_name() == "delete_memories"
    assert "delete_memories" in tool.get_instruction_snippet()


@pytest.mark.parametrize("ids", [[0], [-1], [1] * 101])
async def test_delete_memories_validation(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    ids: list[int],
) -> None:
    tool = DeleteMemoriesTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(ValueError):
        await tool.get_handler()(ids)
    mock_memory_manager.delete_records.assert_not_called()


@pytest.mark.parametrize("inconsistent", [False, True])
async def test_delete_memories_errors(
    mock_config: Config,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    inconsistent: bool,
) -> None:
    error = InconsistentStateError("split") if inconsistent else RuntimeError("failed")
    mock_memory_manager.delete_records.side_effect = error
    tool = DeleteMemoriesTool(mock_config, mock_memory_manager, mock_tool_logger)
    with pytest.raises(
        InconsistentStateError if inconsistent else StorageError
    ) as raised:
        await tool.get_handler()([1])
    assert (
        (raised.value is error) if inconsistent else (raised.value.__cause__ is error)
    )
    assert mock_tool_logger.error.called
    assert not any(
        "completed" in str(call) for call in mock_tool_logger.info.call_args_list
    )


@pytest.mark.parametrize(
    ("ids", "removed", "missing"),
    [([], [], []), ([1], [1], []), ([3, 1, 3, 2, 1], [3, 1], [2]), ([8], [], [8])],
)
async def test_delete_reports_manager_order_and_logs_counts(
    delete_tool_instance: DeleteMemoriesTool,
    mock_memory_manager: MagicMock,
    mock_tool_logger: MagicMock,
    ids: list[int],
    removed: list[int],
    missing: list[int],
) -> None:
    mock_memory_manager.delete_records.return_value = (removed, missing)
    result = await delete_tool_instance.get_handler()(ids)
    assert result == {"removed": removed, "missing": missing}
    mock_memory_manager.delete_records.assert_called_once_with(ids)
    assert any("invoked" in str(call) for call in mock_tool_logger.info.call_args_list)
    assert any(
        call.kwargs.get("extra", {}).get("removed_count") == len(removed)
        and call.kwargs.get("extra", {}).get("missing_count") == len(missing)
        for call in mock_tool_logger.info.call_args_list
    )
