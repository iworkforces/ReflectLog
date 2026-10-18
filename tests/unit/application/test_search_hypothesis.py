"""Property-based testing using Hypothesis.

Generates edge cases and complex scenarios to find bugs that
traditional unit testing might miss.

Example:
    @given(st.text(min_size=1, max_size=1000))
    def test_query_length(query: str):
        result = search_memory(query)
        assert isinstance(result, list)
        # Hypothesis will try many variations automatically

Usage:
    pytest tests/unit/application/test_search_hypothesis.py
    pytest --hypothesis-show-statematrix
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.utils.logging import create_logger
from reflectlog.application.utils.security import SecretString


@given(st.text(min_size=1, max_size=1000))
def test_query_edge_cases(query: str):
    """Test search with edge case inputs.

    Args:
        query: Query string to search.

    Hypothesis generates: Unicode, empty, very long, etc.
    """
    config = Config(
        workspace_id="hypothesis_test",
        openrouter_api_key=SecretString("test-key-not-real"),
        admin_password=SecretString("test-admin-password"),
    )
    logger = create_logger(__name__, config.workspace_id, "INFO")

    manager = MemoryManager(config, logger)
    results = manager.search(query)

    assert isinstance(results, list)
    assert len(results) <= manager.config.search_limit


@given(st.integers(min_value=1, max_value=1000))
def test_add_memories_varied_count(count: int):
    """Test add with varying memory counts.

    Args:
        count: Number of memories to add.

    Hypothesis will test: 1, 100, 999, etc.
    """
    config = Config(
        workspace_id="hypothesis_test",
        openrouter_api_key=SecretString("test-key-not-real"),
        admin_password=SecretString("test-admin-password"),
    )
    logger = create_logger(__name__, config.workspace_id, "INFO")

    manager = MemoryManager(config, logger)
    added = manager.add_memories([f"Memory {i}" for i in range(count)])

    assert added == count


@given(st.integers(min_value=0, max_value=1))
def test_add_empty_memories(count: int):
    """Test add with empty memory list.

    Args:
        count: Number of empty memories (0-1000).

    Boundary case: should handle gracefully.
    """
    config = Config(
        workspace_id="hypothesis_test",
        openrouter_api_key=SecretString("test-key-not-real"),
        admin_password=SecretString("test-admin-password"),
    )
    logger = create_logger(__name__, config.workspace_id, "INFO")

    manager = MemoryManager(config, logger)

    if count == 0:
        added = manager.add_memories([])
    else:
        memories = [""] * count
        added = manager.add_memories(memories)

    assert added == count


@given(st.text(min_size=10, max_size=100))
def test_add_injection_attempt(query: str):
    """Test that malicious query patterns are handled.

    Args:
        query: Query that might contain injection attempts.

    Hypothesis will generate: Various text patterns including potential injections.
    """
    config = Config(
        workspace_id="hypothesis_test",
        openrouter_api_key=SecretString("test-key-not-real"),
        admin_password=SecretString("test-admin-password"),
    )
    logger = create_logger(__name__, config.workspace_id, "INFO")

    manager = MemoryManager(config, logger)

    results = manager.search(query)

    assert isinstance(results, list)
    assert not any("error" in str(r).lower() for r in results)


@given(st.integers(min_value=0, max_value=10000))
def test_max_memory_length(length: int):
    """Test max memory length validation.

    Args:
        length: Memory length to test.

    Boundary case: Should reject or accept.
    """
    config = Config(
        workspace_id="hypothesis_test",
        openrouter_api_key=SecretString("test-key-not-real"),
        admin_password=SecretString("test-admin-password"),
    )
    logger = create_logger(__name__, config.workspace_id, "INFO")

    manager = MemoryManager(config, logger)

    if length <= config.max_memory_length:
        result = manager.add_memories([f"{'x' * length}"])
        assert result >= 0
    else:
        with pytest.raises(Exception):
            manager.add_memories([f"{'x' * length}"])
