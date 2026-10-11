"""Shared fixtures for MCP tool unit tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.add_phases import AddResult
from reflectlog.application.memory.manager import MemoryManager, RecordSearchResult
from reflectlog.application.tools.add_memories import AddMemoriesTool
from reflectlog.application.tools.delete_memories import DeleteMemoriesTool
from reflectlog.application.tools.get_all_memories import GetAllMemoriesTool
from reflectlog.application.tools.get_memory_by_id import GetMemoryByIdTool
from reflectlog.application.tools.search_memories import SearchMemoriesTool
from reflectlog.core.enums import SearchScoreKind


@pytest.fixture
def mock_config(set_env_vars: dict[str, str]) -> Config:
    """Create a Config from test environment variables."""
    return Config.from_environment()


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Create a mock MemoryManager with common methods stubbed."""
    mm = MagicMock(spec=MemoryManager)
    mm.add_memories_async = AsyncMock(
        return_value=AddResult(
            stored_count=1, skipped_count=0, replaced_count=0, replacements=[]
        )
    )
    mm.search_records = AsyncMock(
        return_value=RecordSearchResult((), SearchScoreKind.NONE)
    )
    mm.get_records.return_value = ([], [])
    mm.get_page_records_with_total.return_value = ([], 0)
    mm.delete_records.return_value = ([], [])
    return mm


@pytest.fixture
def mock_tool_logger() -> MagicMock:
    """Create a mock structured logger for tool tests."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def add_tool_instance(
    mock_config: Config, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
) -> AddMemoriesTool:
    """Create an AddMemoriesTool instance with mocked dependencies."""
    return AddMemoriesTool(
        config=mock_config, memory_manager=mock_memory_manager, logger=mock_tool_logger
    )


@pytest.fixture
def search_tool_instance(
    mock_config: Config, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
) -> SearchMemoriesTool:
    """Create a SearchMemoriesTool instance with mocked dependencies."""
    return SearchMemoriesTool(
        config=mock_config, memory_manager=mock_memory_manager, logger=mock_tool_logger
    )


@pytest.fixture
def delete_tool_instance(
    mock_config: Config, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
) -> DeleteMemoriesTool:
    """Create a DeleteMemoriesTool instance with mocked dependencies."""
    return DeleteMemoriesTool(
        config=mock_config, memory_manager=mock_memory_manager, logger=mock_tool_logger
    )


@pytest.fixture
def get_all_tool_instance(
    mock_config: Config, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
) -> GetAllMemoriesTool:
    return GetAllMemoriesTool(mock_config, mock_memory_manager, mock_tool_logger)


@pytest.fixture
def get_memory_tool_instance(
    mock_config: Config, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
) -> GetMemoryByIdTool:
    return GetMemoryByIdTool(mock_config, mock_memory_manager, mock_tool_logger)
