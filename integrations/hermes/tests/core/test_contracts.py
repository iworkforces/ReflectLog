# pyright: basic
"""The shared contracts work without Hermes on Python 3.14.8 and later."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

RECORD = {"id": 7, "memory": "A stored fact", "created_at": "2026-10-12T00:00:00Z"}
HIT = {**RECORD, "score": 0.8}
REPLACEMENT = {
    "old_memory": "Previous fact",
    "new_memory": "Updated fact",
    "confidence": 0.9,
    "reason": "Newer information",
}
SHAPES: dict[str, dict[str, object]] = {
    "parse_memory_record": RECORD,
    "parse_search_hit": HIT,
    "parse_search_result": {"results": [HIT], "score_kind": "rrf", "limit": 6},
    "parse_memory_page": {
        "records": [RECORD],
        "total": 3,
        "offset": 1,
        "limit": 1,
        "truncated": True,
    },
    "parse_get_result": {"record": RECORD},
    "parse_replacement": REPLACEMENT,
    "parse_add_result": {
        "stored_count": 1,
        "skipped_count": 2,
        "replaced_count": 1,
        "replacements": [REPLACEMENT],
        "dry_run": False,
    },
    "parse_delete_result": {"removed": [7], "missing": [9]},
}


@pytest.mark.parametrize("parser_name", SHAPES)
def test_parser_accepts_documented_shape(models: ModuleType, parser_name: str) -> None:
    # Given a server response and an independently constructed domain value.
    record = models.MemoryRecord(7, "A stored fact", "2026-10-12T00:00:00Z")
    hit = models.SearchHit(7, "A stored fact", 0.8, "2026-10-12T00:00:00Z")
    replacement = models.Replacement(
        "Previous fact", "Updated fact", 0.9, "Newer information"
    )
    expected = {
        "parse_memory_record": record,
        "parse_search_hit": hit,
        "parse_search_result": models.SearchResult((hit,), "rrf", 6),
        "parse_memory_page": models.MemoryPage((record,), 3, 1, 1, True),
        "parse_get_result": record,
        "parse_replacement": replacement,
        "parse_add_result": models.AddResult(1, 2, 1, (replacement,), False),
        "parse_delete_result": models.DeleteResult((7,), (9,)),
    }
    # When parsing the documented wire shape.
    result = vars(models)[parser_name](deepcopy(SHAPES[parser_name]))
    # Then every field is represented by the typed immutable contract.
    assert result == expected[parser_name]


@pytest.mark.parametrize("parser_name", SHAPES)
@pytest.mark.parametrize("payload", [None, [], "not an object", 1, True, {}])
def test_parser_rejects_wrong_top_level_shape(
    models: ModuleType, parser_name: str, payload: object
) -> None:
    # Given a payload that cannot be a structured tool result.
    parser = vars(models)[parser_name]
    # When parsing it, then the failure is classified as malformed.
    with pytest.raises(models.ReflectLogError) as caught:
        parser(payload)
    assert caught.value.code == "malformed"


@pytest.mark.parametrize("parser_name", SHAPES)
def test_parser_rejects_missing_fields(models: ModuleType, parser_name: str) -> None:
    # Given an otherwise valid response missing a documented field.
    payload = deepcopy(SHAPES[parser_name])
    _ = payload.pop(next(iter(payload)))
    # When parsing it, then the failure is classified as malformed.
    with pytest.raises(models.ReflectLogError) as caught:
        vars(models)[parser_name](payload)
    assert caught.value.code == "malformed"


@pytest.mark.parametrize("parser_name", SHAPES)
def test_parser_ignores_fields_added_by_newer_servers(
    models: ModuleType, parser_name: str
) -> None:
    # Given a valid response carrying an extra field a newer server might add.
    payload = deepcopy(SHAPES[parser_name])
    payload["future_field"] = "ignored"
    # When parsing it, then the result equals parsing the documented shape alone.
    parser = vars(models)[parser_name]
    assert parser(payload) == parser(deepcopy(SHAPES[parser_name]))


@pytest.mark.parametrize(
    ("parser_name", "field", "bad_values"),
    [
        ("parse_memory_record", "id", [True, False, 0, -1, 1.0, "7", None]),
        ("parse_memory_record", "memory", [None, 7, False, []]),
        ("parse_memory_record", "created_at", [7, True, [], {}]),
        ("parse_search_hit", "id", [True, 0, -1, "7"]),
        ("parse_search_hit", "memory", [None, 7]),
        ("parse_search_hit", "created_at", [7, False]),
        (
            "parse_search_hit",
            "score",
            [True, False, "0.8", None, float("inf"), float("nan"), 10**400],
        ),
        ("parse_search_result", "results", [None, {}, (), [RECORD], [None]]),
        ("parse_search_result", "score_kind", [None, 1, True]),
        ("parse_search_result", "limit", [True, 0, -1, 1.5, "6"]),
        ("parse_memory_page", "records", [None, {}, (), [{"id": 7}], [None]]),
        ("parse_memory_page", "total", [True, -1, 1.5, "3"]),
        ("parse_memory_page", "offset", [True, -1, 1.5, "0"]),
        ("parse_memory_page", "limit", [True, -1, 1.5, "1"]),
        ("parse_memory_page", "truncated", [0, 1, "true", None]),
        ("parse_get_result", "record", [True, [], {}, {"id": 7, "memory": "x"}]),
        ("parse_replacement", "old_memory", [None, True, 1]),
        ("parse_replacement", "new_memory", [None, True, 1]),
        ("parse_replacement", "confidence", [True, "0.9", None, float("inf")]),
        ("parse_replacement", "reason", [None, True, 1]),
        ("parse_add_result", "stored_count", [True, -1, "1", 1.5]),
        ("parse_add_result", "skipped_count", [True, -1, "1", 1.5]),
        ("parse_add_result", "replaced_count", [True, -1, "1", 1.5]),
        ("parse_add_result", "replacements", [None, {}, (), [RECORD], [None]]),
        ("parse_add_result", "dry_run", [0, 1, "false", None]),
        (
            "parse_delete_result",
            "removed",
            [None, {}, (), [True], [0], [-1], [1.0], ["7"]],
        ),
        (
            "parse_delete_result",
            "missing",
            [None, {}, (), [True], [0], [-1], [1.0], ["7"]],
        ),
    ],
)
def test_parser_rejects_wrong_field_types(
    models: ModuleType, parser_name: str, field: str, bad_values: list[object]
) -> None:
    # Given invalid values at a documented boundary field.
    for bad_value in bad_values:
        payload = deepcopy(SHAPES[parser_name])
        payload[field] = bad_value
        # When parsing it, then no coercion can hide a malformed response.
        with pytest.raises(models.ReflectLogError) as caught:
            vars(models)[parser_name](payload)
        assert caught.value.code == "malformed"


@pytest.mark.parametrize("parser_name", ["parse_memory_record", "parse_search_hit"])
def test_record_accepts_absent_timestamp(models: ModuleType, parser_name: str) -> None:
    # Given a record whose server timestamp is explicitly null.
    payload = {**SHAPES[parser_name], "created_at": None}
    # When parsing it.
    result = vars(models)[parser_name](payload)
    # Then the missing timestamp is preserved.
    assert result.created_at is None


@pytest.mark.parametrize(
    ("parser_name", "field"),
    [("parse_search_hit", "score"), ("parse_replacement", "confidence")],
)
def test_parser_normalizes_integer_scores(
    models: ModuleType, parser_name: str, field: str
) -> None:
    # Given a JSON integer in a numeric score field.
    payload = {**SHAPES[parser_name], field: 1}
    # When parsing it.
    result = vars(models)[parser_name](payload)
    # Then consumers receive a float, not a boolean or unnormalized integer.
    score = result.score if field == "score" else result.confidence
    assert score == 1.0
    assert type(score) is float


def test_get_result_accepts_missing_record(models: ModuleType) -> None:
    # Given a successful lookup with no matching record.
    payload = {"record": None}
    # When parsing it.
    result = models.parse_get_result(payload)
    # Then absence is distinct from an invalid response.
    assert result is None


def test_memory_page_accepts_zero_limit_probe(models: ModuleType) -> None:
    # Given a cheap server count probe.
    payload = {"records": [], "total": 4, "offset": 0, "limit": 0, "truncated": True}
    # When parsing it.
    result = models.parse_memory_page(payload)
    # Then zero is a valid page limit and the count survives.
    assert result == models.MemoryPage((), 4, 0, 0, True)


@pytest.mark.parametrize("code", ["disabled", "", "unknown", None, 1])
def test_error_rejects_unknown_codes(models: ModuleType, code: object) -> None:
    # Given an error code outside the closed client vocabulary.
    # When constructing it, then it cannot masquerade as a supported failure.
    with pytest.raises(ValueError):
        models.ReflectLogError(code, "Safe message")


@pytest.mark.parametrize(
    "code",
    [
        "unavailable",
        "timeout",
        "outcome_unknown",
        "tool_error",
        "unauthorized",
        "incompatible",
        "malformed",
    ],
)
def test_error_keeps_server_detail_out_of_message(
    models: ModuleType, code: str
) -> None:
    # Given untrusted server detail that a caller must opt in to expose.
    detail = "secret token and memory text"
    # When constructing a client error.
    error = models.ReflectLogError(code, "Safe message", server_detail=detail)
    # Then the normal exception surface contains only the client message.
    assert error.code == code
    assert str(error) == "Safe message"
    assert error.server_detail == detail


@pytest.mark.parametrize("field", ["workspace_id", "memories", "source"])
def test_write_job_is_immutable(models: ModuleType, field: str) -> None:
    # Given a queued job permanently bound to its original workspace.
    job = models.WriteJob("workspace", ("memory",), "mirror")
    # When assigning any field, then dataclass immutability rejects it.
    with pytest.raises(FrozenInstanceError):
        setattr(job, field, "changed")


@pytest.mark.parametrize("has_initializer", [False, True])
def test_package_loader_supports_both_waves(
    tmp_path: Path, has_initializer: bool
) -> None:
    # Given a package with or without the later lazy root initializer.
    conftest = Path(__file__).resolve().parents[1] / "conftest.py"
    spec = importlib.util.spec_from_file_location("hermes_loader_test", conftest)
    assert spec is not None and spec.loader is not None
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    if has_initializer:
        _ = (tmp_path / "__init__.py").write_text(
            '"""Lazy plugin."""\n', encoding="utf-8"
        )
    _ = (tmp_path / "child.py").write_text("VALUE = 7\n", encoding="utf-8")
    package_name = f"reflectlog_hermes_loader_{has_initializer}"
    original_path = list(sys.path)
    # When loading the package and one submodule.
    try:
        loader.load_plugin_package(tmp_path, package_name)
        child = __import__(f"{package_name}.child", fromlist=["VALUE"])
        # Then relative package imports work without adding to sys.path.
        assert child.VALUE == 7
        assert sys.path == original_path
    finally:
        _ = sys.modules.pop(f"{package_name}.child", None)
        _ = sys.modules.pop(package_name, None)


def test_core_import_does_not_load_hermes() -> None:
    # Given a fresh interpreter, isolated from this process's imported modules.
    plugin_dir = Path(__file__).resolve().parents[2]
    script = """
import importlib
import importlib.util
from pathlib import Path
import sys

plugin_dir = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("hermes_test_loader", plugin_dir / "tests/conftest.py")
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
before = set(sys.modules)
original_path = list(sys.path)
loader.load_plugin_package(plugin_dir, "reflectlog_hermes_isolated")
importlib.import_module("reflectlog_hermes_isolated.core.models")
added = set(sys.modules) - before
assert not any(name.split(".")[0] in {"agent", "hermes_cli"} for name in added), added
assert sys.path == original_path
assert str(plugin_dir) not in sys.path
print("core isolated")
"""
    # When importing only the shared core package.
    result = subprocess.run(
        [sys.executable, "-c", script, str(plugin_dir)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    # Then no Hermes module was imported or shadowed by the plugin root.
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "core isolated"
