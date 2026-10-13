# pyright: basic
from __future__ import annotations

import json

import pytest
import toolsets


def test_tool_happy_paths(hermes_provider):
    # Given a real provider and Hermes tool namespace.
    provider, fake = hermes_provider()
    assert not {s["name"] for s in provider.get_tool_schemas()} & set(
        toolsets._HERMES_CORE_TOOLS
    )
    # When the five tools operate on one record.
    stored = json.loads(
        provider.handle_tool_call("reflectlog_store", {"memories": ["cobalt widgets"]})
    )
    searched = json.loads(
        provider.handle_tool_call("reflectlog_search", {"query": "cobalt"})
    )
    listed = json.loads(provider.handle_tool_call("reflectlog_list", {}))
    record = json.loads(provider.handle_tool_call("reflectlog_get", {"id": 1}))
    deleted = json.loads(provider.handle_tool_call("reflectlog_forget", {"ids": [1]}))
    # Then their structured results preserve the server contract.
    assert stored["stored_count"] == 1
    assert searched["count"] == 1
    assert searched["attempt"] == 1
    assert searched["max_attempts"] == 5
    assert "attempt=2" in searched["next_step"]
    assert listed["total"] == 1
    assert record["record"]["memory"] == "cobalt widgets"
    assert deleted["removed"] == [1]


@pytest.mark.parametrize(
    "name,args",
    [
        ("reflectlog_search", {"query": "ok", "limit": True}),
        ("reflectlog_search", {"query": 1}),
        ("reflectlog_search", {"query": "x", "workspace_id": "other"}),
        *[
            ("reflectlog_search", {"query": "ok", "attempt": attempt})
            for attempt in [0, 6, -1, 1.5, "2", True, False, None]
        ],
        ("reflectlog_store", {"memories": "bad"}),
        ("reflectlog_store", {"memories": ["ok"], "dry_run": 1}),
        ("reflectlog_list", {"offset": -1}),
        ("reflectlog_get", {"id": True}),
        ("reflectlog_forget", {"ids": [False]}),
    ],
)
def test_argument_validation(hermes_provider, name, args):
    # Given invalid boundary input.
    provider, fake = hermes_provider()
    # When dispatched through Hermes.
    result = json.loads(provider.handle_tool_call(name, args))
    # Then it is rejected locally.
    assert result["code"] == "invalid_input"
    assert not fake.requests


@pytest.mark.parametrize("attempt", [1, 4, 5])
@pytest.mark.parametrize("has_hits", [False, True])
def test_search_guides_rewritten_retry(hermes_provider, attempt, has_hits):
    # Given a lookup with either matching or empty server results.
    provider, fake = hermes_provider()
    if has_hits:
        provider.handle_tool_call("reflectlog_store", {"memories": ["cobalt widgets"]})
    # When the caller supplies its search attempt.
    result = json.loads(
        provider.handle_tool_call(
            "reflectlog_search", {"query": "cobalt", "attempt": attempt}
        )
    )
    # Then relevance checking and the remaining search budget guide the caller.
    assert result["count"] == int(has_hits)
    assert result["attempt"] == attempt and result["max_attempts"] == 5
    guidance = result["next_step"]
    assert ("Verify each memory is relevant" in guidance) == has_hits
    if attempt < 5:
        assert "rewrite the query" in guidance.lower()
        assert f"attempt={attempt + 1}" in guidance
    else:
        assert "Search limit reached" in guidance
        assert "attempt=6" not in guidance
        assert "rewrite the query" not in guidance.lower()
    calls = [json.loads(r.content) for r in fake.requests if r.method == "POST"]
    searches = [
        r["params"]["arguments"]
        for r in calls
        if r.get("params", {}).get("name") == "search_memories"
    ]
    assert len(searches) == 1 and "attempt" not in searches[0]


def test_search_attempt_schema(hermes_provider):
    # Given the provider's public tool schema.
    provider, fake = hermes_provider()
    # When inspecting the search boundary.
    parameters = next(
        s["parameters"]
        for s in provider.get_tool_schemas()
        if s["name"] == "reflectlog_search"
    )
    # Then the bounded optional integer is advertised without extra arguments.
    assert parameters["properties"]["attempt"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 5,
        "default": 1,
    }
    assert "attempt" not in parameters["required"]
    assert parameters["additionalProperties"] is False
    assert not fake.requests


def test_five_rewritten_searches_reject_sixth(hermes_provider):
    # Given a caller following one bounded lookup without automatic recall.
    provider, fake = hermes_provider()
    # When five rewritten searches are followed by an out-of-budget attempt.
    results = [
        json.loads(
            provider.handle_tool_call(
                "reflectlog_search", {"query": query, "attempt": attempt}
            )
        )
        for attempt, query in enumerate(
            ["cobalt", "widgets", "project", "settings", "entities"], start=1
        )
    ]
    rejected = json.loads(
        provider.handle_tool_call(
            "reflectlog_search", {"query": "synonyms", "attempt": 6}
        )
    )
    # Then only the five valid searches reach the backend.
    assert [r["attempt"] for r in results] == [1, 2, 3, 4, 5]
    assert rejected["code"] == "invalid_input"
    assert fake.dispatches["search_memories"] == 5


@pytest.mark.parametrize(
    "fault,code",
    [
        ("connect", "unavailable"),
        ("read_timeout", "outcome_unknown"),
        ("validation", "tool_error"),
    ],
)
def test_error_mapping(hermes_provider, fault, code, caplog):
    # Given a server failure on a mutation.
    provider, fake = hermes_provider()
    fake.faults["add_memories"].append(fault)
    # When storing a sensitive entry.
    result = json.loads(
        provider.handle_tool_call(
            "reflectlog_store", {"memories": ["private cobalt fact"]}
        )
    )
    # Then the typed code is exposed, without logs or mutation retries.
    assert result["code"] == code
    if code == "outcome_unknown":
        assert (
            "Check with reflectlog_search or reflectlog_list before retrying."
            in result["error"]
        )
        assert fake.dispatches["add_memories"] == 1
    assert "private cobalt fact" not in caplog.text
    assert fake.token not in caplog.text


def test_unauthorized(hermes_provider):
    # Given a token rejected by the server.
    provider, fake = hermes_provider()
    fake.auth_status = 401
    fake.token = "a different server token"
    # When a read is attempted.
    result = json.loads(provider.handle_tool_call("reflectlog_list", {}))
    # Then authentication failure is classified.
    assert result["code"] == "unauthorized"


@pytest.mark.parametrize("context", ["subagent", "cron"])
def test_dry_run_allowed_on_nonprimary(hermes_provider, context):
    # Given a non-primary provider on a profile workspace.
    provider, fake = hermes_provider(
        {"workspace_scope": "profile"}, platform=context, agent_context=context
    )
    # When a non-mutating preview is requested.
    result = json.loads(
        provider.handle_tool_call(
            "reflectlog_store", {"memories": ["cobalt fact"], "dry_run": True}
        )
    )
    # Then it succeeds without storing content.
    assert result["dry_run"] is True
    assert not any(fake.store.values())


@pytest.mark.parametrize("enabled,expected", [(False, "disabled"), (True, None)])
def test_smart_replace_opt_in(hermes_provider, enabled, expected):
    # Given an explicit replacement setting.
    provider, fake = hermes_provider({"smart_replace": enabled})
    # When replacement is requested through the explicit tool.
    result = json.loads(
        provider.handle_tool_call(
            "reflectlog_store", {"memories": ["cobalt fact"], "replace": True}
        )
    )
    # Then only opt-in permits dispatch.
    assert result.get("code") == expected
    assert fake.dispatches["add_memories"] == int(enabled)


def test_tool_errors_never_return_server_detail(hermes_provider, plugin_import, caplog):
    # Given a server error whose detail echoes the configured token.
    provider, fake = hermes_provider()
    models = plugin_import("core.models")

    class FailingClient:
        def get_all_memories(self, *args, **kwargs):
            raise models.ReflectLogError(
                "tool_error",
                "ReflectLog tool failed",
                server_detail="Rejected Bearer qa-token in upstream validation",
            )

    tools = plugin_import("tool_handlers")
    context = tools.ToolContext(FailingClient(), provider.settings, "test", True)
    # When it crosses the real Hermes JSON error helper.
    result = json.loads(tools.handle_tool_call(context, "reflectlog_list", {}))
    # Then only the client's own message is returned, and nothing is logged.
    assert result == {"error": "ReflectLog tool failed", "code": "tool_error"}
    assert "qa-token" not in caplog.text
