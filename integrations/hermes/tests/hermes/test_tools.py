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
    # When the six tools operate on one record.
    stored = json.loads(
        provider.handle_tool_call("memory_store", {"memories": ["cobalt widgets"]})
    )
    searched = json.loads(
        provider.handle_tool_call("memory_search", {"query": "cobalt"})
    )
    preview = json.loads(
        provider.handle_tool_call("memory_preview", {"memories": ["teal dashboards"]})
    )
    listed = json.loads(provider.handle_tool_call("memory_list", {}))
    record = json.loads(provider.handle_tool_call("memory_get", {"id": 1}))
    deleted = json.loads(provider.handle_tool_call("memory_forget", {"ids": [1]}))
    # Then their structured results preserve the server contract.
    assert stored["stored_count"] == 1
    assert preview["dry_run"] is True
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
        ("memory_search", {"query": "ok", "limit": True}),
        ("memory_search", {"query": 1}),
        ("memory_search", {"query": "x", "workspace_id": "other"}),
        *[
            ("memory_search", {"query": "ok", "attempt": attempt})
            for attempt in [0, 6, -1, 1.5, "2", True, False, None]
        ],
        ("memory_store", {"memories": "bad"}),
        ("memory_store", {"memories": ["ok"], "dry_run": True}),
        ("memory_store", {"memories": ["ok"], "dry_run": False}),
        ("memory_preview", {"memories": "bad"}),
        ("memory_preview", {"memories": ["ok"], "replace": 1}),
        ("memory_list", {"offset": -1}),
        ("memory_get", {"id": True}),
        ("memory_forget", {"ids": [False]}),
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
        provider.handle_tool_call("memory_store", {"memories": ["cobalt widgets"]})
    # When the caller supplies its search attempt.
    result = json.loads(
        provider.handle_tool_call(
            "memory_search", {"query": "cobalt", "attempt": attempt}
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
        if s["name"] == "memory_search"
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
                "memory_search", {"query": query, "attempt": attempt}
            )
        )
        for attempt, query in enumerate(
            ["cobalt", "widgets", "project", "settings", "entities"], start=1
        )
    ]
    rejected = json.loads(
        provider.handle_tool_call("memory_search", {"query": "synonyms", "attempt": 6})
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
        provider.handle_tool_call("memory_store", {"memories": ["private cobalt fact"]})
    )
    # Then the typed code is exposed, without logs or mutation retries.
    assert result["code"] == code
    if code == "outcome_unknown":
        assert (
            "Check with memory_search or memory_list before retrying."
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
    result = json.loads(provider.handle_tool_call("memory_list", {}))
    # Then authentication failure is classified.
    assert result["code"] == "unauthorized"


@pytest.mark.parametrize("context", ["subagent", "cron"])
def test_preview_allowed_on_nonprimary(hermes_provider, context):
    # Given a non-primary provider on a profile workspace.
    provider, fake = hermes_provider(
        {"workspace_scope": "profile"}, platform=context, agent_context=context
    )
    # When a non-mutating preview is requested.
    result = json.loads(
        provider.handle_tool_call("memory_preview", {"memories": ["cobalt fact"]})
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
            "memory_store", {"memories": ["cobalt fact"], "replace": True}
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
    result = json.loads(tools.handle_tool_call(context, "memory_list", {}))
    # Then only the client's own message is returned, and nothing is logged.
    assert result == {"error": "ReflectLog tool failed", "code": "tool_error"}
    assert "qa-token" not in caplog.text


def test_six_shared_tool_names(hermes_provider):
    # Given an initialized provider in the real Hermes namespace.
    provider, fake = hermes_provider()
    expected = (
        "memory_search",
        "memory_get",
        "memory_store",
        "memory_preview",
        "memory_forget",
        "memory_list",
    )
    # When Hermes discovers the available tools.
    names = [schema["name"] for schema in provider.get_tool_schemas()]
    # Then exactly the six shared names are unique and do not shadow core tools.
    assert len(names) == len(set(names)) == 6
    assert set(names) == set(expected)
    assert set(names).isdisjoint(toolsets._HERMES_CORE_TOOLS)
    assert not fake.requests


@pytest.mark.parametrize(
    "name,args",
    [
        ("reflectlog_search", {"query": "cobalt"}),
        ("reflectlog_store", {"memories": ["cobalt fact"]}),
        ("reflectlog_list", {}),
        ("reflectlog_get", {"id": 1}),
        ("reflectlog_forget", {"ids": [1]}),
        ("reflectlog_preview", {"memories": ["cobalt fact"]}),
    ],
)
def test_legacy_names_are_unknown(hermes_provider, name, args):
    # Given an active provider and an obsolete tool name.
    provider, fake = hermes_provider()
    # When the obsolete tool is invoked.
    result = json.loads(provider.handle_tool_call(name, args))
    # Then the unknown name fails locally without a server request.
    assert result["code"] == "invalid_input"
    assert not fake.requests


@pytest.mark.parametrize("context", ["primary", "subagent", "cron"])
@pytest.mark.parametrize("approval", [False, True])
def test_preview_skips_only_explicit_write_gate(
    hermes_provider, plugin_import, monkeypatch, context, approval
):
    # Given an active workspace with explicit mutation approval or context gates.
    provider, fake = hermes_provider(
        {"workspace_scope": "profile"}, platform="cli", agent_context=context
    )
    monkeypatch.setattr(
        plugin_import("tool_handlers"),
        "write_approval_enabled",
        lambda subsystem: approval and subsystem == "memory",
    )
    # When the same fact is previewed and then offered for storage.
    preview = json.loads(
        provider.handle_tool_call("memory_preview", {"memories": ["cobalt fact"]})
    )
    stored = json.loads(
        provider.handle_tool_call("memory_store", {"memories": ["cobalt fact"]})
    )
    # Then preview always dispatches, while storage obeys the mutation gate.
    blocked = approval or context != "primary"
    assert preview["dry_run"] is True
    assert stored.get("code") == ("disabled" if blocked else None)
    assert fake.dispatches["add_memories"] == (1 if blocked else 2)
    assert bool(any(fake.store.values())) == (not blocked)


@pytest.mark.parametrize("enabled", [False, True])
def test_preview_replacement_opt_in(hermes_provider, enabled):
    # Given explicit smart replacement configuration.
    provider, fake = hermes_provider({"smart_replace": enabled})
    # When replacement is requested through preview.
    result = json.loads(
        provider.handle_tool_call(
            "memory_preview", {"memories": ["cobalt fact"], "replace": True}
        )
    )
    # Then opt-in is required even though preview cannot mutate storage.
    assert result.get("code") == (None if enabled else "disabled")
    assert fake.dispatches["add_memories"] == int(enabled)
    assert not any(fake.store.values())
    if enabled:
        calls = [json.loads(r.content) for r in fake.requests if r.method == "POST"]
        adds = [
            r["params"]["arguments"]
            for r in calls
            if r.get("params", {}).get("name") == "add_memories"
        ]
        assert len(adds) == 1
        assert adds[0]["dry_run"] is True
        assert adds[0]["allow_replace"] is True
    else:
        assert not fake.requests


def test_store_forwards_mutating_add(hermes_provider):
    # Given an active primary provider.
    provider, fake = hermes_provider()
    # When storing a fact without preview arguments.
    result = json.loads(
        provider.handle_tool_call("memory_store", {"memories": ["cobalt fact"]})
    )
    # Then the MCP add explicitly selects mutation.
    calls = [json.loads(r.content) for r in fake.requests if r.method == "POST"]
    adds = [
        r["params"]["arguments"]
        for r in calls
        if r.get("params", {}).get("name") == "add_memories"
    ]
    assert len(adds) == 1
    assert adds[0]["dry_run"] is False
    assert result["dry_run"] is False
    assert result["stored_count"] == 1
