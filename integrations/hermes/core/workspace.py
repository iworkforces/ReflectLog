from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Literal


def _validate_id(value: str) -> str:
    normalized = value.strip().lower()
    if (
        not re.fullmatch(r"[a-z0-9_.-]{1,64}", normalized)
        or normalized == "."
        or ".." in normalized
    ):
        raise ValueError("workspace: invalid workspace identifier")
    return normalized


def validate_workspace_override(value: str) -> str:
    normalized = _validate_id(value)
    if re.fullmatch(r"h-[a-z0-9-]+-[0-9a-f]{24}", normalized):
        raise ValueError("workspace: reserved derived identifier")
    return normalized


def profile_slug(profile: str) -> str:
    slug = re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", profile.lower())).strip("-")
    return slug[:24] or "profile"


def derive_workspace_id(namespace: str, profile: str, subject: str | None) -> str:
    payload = json.dumps(
        ["reflectlog-hermes/v1", namespace, profile, subject],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return _validate_id(f"h-{profile_slug(profile)}-{digest}")


@dataclass(frozen=True, slots=True)
class Binding:
    active: bool
    reason: str
    workspace_id: str | None
    participant_ids: frozenset[str]
    writes_allowed: bool
    scope_label: str


def resolve_binding(
    *,
    namespace: str,
    profile: str | None,
    hermes_home: str | Path,
    scope: Literal["user", "profile"],
    override: str | None,
    platform: str,
    agent_context: str,
    user_id: str | None = None,
    user_id_alt: str | None = None,
    chat_type: str | None = None,
    shared_chats: bool = False,
) -> Binding:
    home_profile = Path(hermes_home).name
    effective_profile = profile or (
        "default" if home_profile == ".hermes" else home_profile
    )
    participants = frozenset(value for value in (user_id, user_id_alt) if value)
    label = "override" if override else scope
    if chat_type and chat_type != "dm" and not shared_chats:
        return Binding(False, "shared_chat_disabled", None, participants, False, label)
    if override:
        workspace_id = validate_workspace_override(override)
    elif scope == "profile":
        workspace_id = derive_workspace_id(namespace, effective_profile, None)
    else:
        participant = user_id_alt or user_id
        if participant and agent_context != "primary":
            return Binding(
                False, "unresolved_subject", None, participants, False, label
            )
        if participant:
            subject = f"{platform}:{participant}"
        elif platform == "cli" or agent_context in {"cron", "subagent"}:
            subject = None
        else:
            return Binding(
                False, "unresolved_subject", None, participants, False, label
            )
        workspace_id = derive_workspace_id(namespace, effective_profile, subject)
    return Binding(
        True, "active", workspace_id, participants, agent_context == "primary", label
    )
