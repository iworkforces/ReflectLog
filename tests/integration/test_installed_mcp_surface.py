"""Installed FastMCP tool surface without paid credentials."""

from __future__ import annotations

from reflectlog.application.mcp_server import AVAILABLE_TOOL_CLASSES
from reflectlog.application.tools.add import AddTool
from reflectlog.application.utils.validation import validate_memories


def test_nine_registered_tools() -> None:
    assert set(AVAILABLE_TOOL_CLASSES) == {
        "add",
        "get_all",
        "search",
        "remove",
        "health_check",
        "search_records",
        "get_records",
        "list_records",
        "remove_records",
    }
    assert AVAILABLE_TOOL_CLASSES["add"] is AddTool


def test_malformed_memories_rejected_by_validator() -> None:
    ok, error = validate_memories(["ok"], 1, 100)
    assert ok is True
    assert error is None
    ok_bad, error_bad = validate_memories([None], 1, 100)
    assert ok_bad is False
    assert error_bad is not None
