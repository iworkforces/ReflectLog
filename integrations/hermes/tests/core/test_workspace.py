# pyright: basic
from __future__ import annotations

import hashlib
import json
import re

import pytest


def binding(module, **changes):
    values = dict(
        namespace="hermes",
        profile="Work Profile",
        hermes_home="/home/.hermes",
        scope="user",
        override=None,
        platform="cli",
        agent_context="primary",
        user_id=None,
        user_id_alt=None,
        chat_type=None,
        shared_chats=False,
    )
    return module.resolve_binding(**(values | changes))


def test_derivation_matches_versioned_hash(plugin_import):
    # Given exact derivation inputs, when resolving, then the stable digest matches.
    workspace = plugin_import("core.workspace")
    digest = hashlib.sha256(
        json.dumps(
            ["reflectlog-hermes/v1", "hermes", "Work Profile", None],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()[:24]
    assert binding(workspace).workspace_id == f"h-work-profile-{digest}"
    assert binding(workspace) == binding(workspace)


def test_cli_owner_workspace(plugin_import):
    # Given a CLI owner, when resolving, then reads and writes are active.
    result = binding(plugin_import("core.workspace"))
    assert (
        result.active
        and result.writes_allowed
        and result.participant_ids == frozenset()
    )


def test_gateway_users_and_platforms_are_isolated(plugin_import):
    # Given differing gateway participants/platforms, when resolving, then ids differ.
    workspace = plugin_import("core.workspace")
    ids = {
        binding(workspace, platform=p, user_id=u).workspace_id
        for p, u in [("telegram", "1"), ("telegram", "2"), ("discord", "1")]
    }
    assert len(ids) == 3


def test_alt_participant_preferred_and_both_author_ids_kept(plugin_import):
    # Given aliased user identity, when resolving, then the alternate owns routing.
    workspace = plugin_import("core.workspace")
    result = binding(workspace, platform="whatsapp", user_id="jid", user_id_alt="lid")
    assert (
        result.workspace_id
        == binding(workspace, platform="whatsapp", user_id="lid").workspace_id
    )
    assert result.participant_ids == frozenset({"jid", "lid"})


@pytest.mark.parametrize(
    "changes",
    [
        {"platform": "telegram"},
        {"platform": "subagent", "agent_context": "subagent", "user_id": "1"},
        {"platform": "cron", "agent_context": "cron", "user_id_alt": "1"},
    ],
)
def test_unresolved_identity_fails_closed(plugin_import, changes):
    # Given ambiguous identity, when resolving, then no workspace is exposed.
    result = binding(plugin_import("core.workspace"), **changes)
    assert not result.active and result.workspace_id is None and result.reason


@pytest.mark.parametrize("context", ["cron", "subagent"])
def test_nonprimary_without_participant_is_owner_readonly(plugin_import, context):
    # Given an owner background task, when resolving, then owner reads are allowed only.
    workspace = plugin_import("core.workspace")
    result = binding(workspace, platform=context, agent_context=context)
    assert result.active and not result.writes_allowed
    assert result.workspace_id == binding(workspace).workspace_id


@pytest.mark.parametrize("allowed", [False, True])
def test_shared_chat_gate(plugin_import, allowed):
    # Given a group chat, when resolving, then explicit opt-in controls activity.
    result = binding(
        plugin_import("core.workspace"),
        platform="telegram",
        user_id="1",
        chat_type="group",
        shared_chats=allowed,
    )
    assert result.active is allowed


@pytest.mark.parametrize("changes", [{"scope": "profile"}, {"override": "CUSTOM"}])
def test_explicit_scope_ignores_routing_participants(plugin_import, changes):
    # Given explicit profile/override routing, when resolving, then participants do not retarget.
    workspace = plugin_import("core.workspace")
    assert (
        binding(workspace, **changes).workspace_id
        == binding(
            workspace,
            platform="subagent",
            agent_context="subagent",
            user_id="1",
            **changes,
        ).workspace_id
    )


@pytest.mark.parametrize(
    "value",
    [".", "a..b", "a/b", "x" * 65, "h-profile-" + "a" * 24, "H-PROFILE-" + "A" * 24],
)
def test_invalid_override(plugin_import, value):
    # Given invalid/reserved routing, when validating, then it is rejected.
    workspace = plugin_import("core.workspace")
    with pytest.raises(ValueError):
        workspace.validate_workspace_override(value)


def test_override_normalizes(plugin_import):
    # Given mixed case and outer whitespace, when validating, then server normalization applies.
    assert (
        plugin_import("core.workspace").validate_workspace_override("  A_B.c-1  ")
        == "a_b.c-1"
    )


@pytest.mark.parametrize(
    "profile, home, expected",
    [
        (None, "/home/.hermes", "default"),
        ("", "/home/profiles/work", "work"),
        (None, "/home/profiles/other/", "other"),
    ],
)
def test_profile_home_fallback(plugin_import, profile, home, expected):
    # Given missing identity, when resolving, then the home basename defines the profile.
    workspace = plugin_import("core.workspace")
    assert (
        binding(workspace, profile=profile, hermes_home=home).workspace_id
        == binding(workspace, profile=expected).workspace_id
    )


@pytest.mark.parametrize("profile", ["?!", "A" * 200, "A__B  C", "世界"])
def test_slug_produces_valid_bounded_id(plugin_import, profile):
    # Given arbitrary profiles, when deriving, then every id meets server limits.
    workspace = plugin_import("core.workspace")
    result = binding(workspace, profile=profile)
    assert len(result.workspace_id) <= 64
    assert re.fullmatch(r"[a-z0-9_.-]+", result.workspace_id)
    assert ".." not in result.workspace_id
    assert len(workspace.profile_slug(profile)) <= 24
