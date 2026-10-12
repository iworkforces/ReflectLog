from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from .workspace import validate_workspace_override

if TYPE_CHECKING:
    from collections.abc import Mapping


class ConfigError(ValueError):
    pass


class _ConfigObject(dict[str, object]):
    pass


@dataclass(frozen=True, slots=True)
class Settings:
    url: str = "http://127.0.0.1:9103/mcp"
    namespace: str = "hermes"
    workspace_scope: Literal["user", "profile"] = "user"
    workspace: str | None = None
    recall: bool = True
    recall_limit: int = 6
    recall_timeout: float = 3.0
    recall_max_chars: int = 8000
    mirror_builtin: bool = True
    capture_turns: bool = False
    smart_replace: bool = False
    shared_chats: bool = False
    connect_timeout: float = 10.0
    request_timeout: float = 30.0


def _boolean(raw: Mapping[str, object], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
    raise ConfigError(f"{key}: expected boolean")


def _number(
    raw: Mapping[str, object],
    key: str,
    default: float,
    low: float,
    high: float,
    *,
    integer: bool = False,
) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ConfigError(f"{key}: expected number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ConfigError(f"{key}: expected finite number") from error
    if (
        not math.isfinite(number)
        or not low <= number <= high
        or (integer and not number.is_integer())
    ):
        raise ConfigError(f"{key}: out of range")
    return number


def parse_settings(
    raw: Mapping[str, object], *, url_override: str | None = None
) -> Settings:
    allowed = {field.name for field in fields(Settings)}
    for key in raw:
        if key not in allowed:
            raise ConfigError(f"{key}: unknown setting")
    url = (
        url_override
        if url_override is not None
        else raw.get("url", "http://127.0.0.1:9103/mcp")
    )
    if not isinstance(url, str):
        raise ConfigError("url: expected HTTP endpoint")
    try:
        parsed = urlsplit(url)
        port = parsed.port
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "?" in url
            or "#" in url
            or any(char.isspace() for char in url)
            or (port is not None and port == 0)
        ):
            raise ValueError("invalid endpoint")
    except ValueError as error:
        raise ConfigError("url: invalid HTTP endpoint") from error
    namespace = raw.get("namespace", "hermes")
    if not isinstance(namespace, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_.-]{0,31}", namespace
    ):
        raise ConfigError("namespace: invalid namespace")
    scope = raw.get("workspace_scope", "user")
    if not isinstance(scope, str) or scope not in {"user", "profile"}:
        raise ConfigError("workspace_scope: expected user or profile")
    workspace = raw.get("workspace")
    if workspace is not None:
        if not isinstance(workspace, str):
            raise ConfigError("workspace: expected identifier")
        try:
            workspace = (
                validate_workspace_override(workspace) if workspace.strip() else None
            )
        except ValueError as error:
            raise ConfigError("workspace: invalid or reserved identifier") from error
    return Settings(
        url=url,
        namespace=namespace,
        workspace_scope="profile" if scope == "profile" else "user",
        workspace=workspace,
        recall=_boolean(raw, "recall", True),
        recall_limit=int(_number(raw, "recall_limit", 6, 1, 50, integer=True)),
        recall_timeout=_number(raw, "recall_timeout", 3, 0.5, 6),
        recall_max_chars=int(
            _number(raw, "recall_max_chars", 8000, 1000, 20000, integer=True)
        ),
        mirror_builtin=_boolean(raw, "mirror_builtin", True),
        capture_turns=_boolean(raw, "capture_turns", False),
        smart_replace=_boolean(raw, "smart_replace", False),
        shared_chats=_boolean(raw, "shared_chats", False),
        connect_timeout=_number(raw, "connect_timeout", 10, 1, 60),
        request_timeout=_number(raw, "request_timeout", 30, 1, 120),
    )


def _read_values(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as stream:
            data: object = json.load(stream, object_pairs_hook=_ConfigObject)
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, ValueError) as error:
        raise ConfigError("reflectlog.json: unable to read configuration") from error
    if not isinstance(data, _ConfigObject):
        raise ConfigError("reflectlog.json: expected object")
    result: dict[str, object] = {}
    for key, value in data.items():
        result[key] = value
    return result


def load_settings(
    hermes_home: str | Path, *, url_override: str | None = None
) -> Settings:
    return parse_settings(
        _read_values(Path(hermes_home) / "reflectlog.json"), url_override=url_override
    )


def save_settings(values: Mapping[str, object], hermes_home: str | Path) -> None:
    home = Path(hermes_home)
    path = home / "reflectlog.json"
    merged = _read_values(path) | dict(values)
    clean = {
        key: value
        for key, value in merged.items()
        if not any(
            part in key.lower() for part in ("token", "secret", "password", "key")
        )
    }
    settings = parse_settings(clean, url_override=None)
    temporary = home / f".reflectlog-{uuid4().hex}.tmp"
    try:
        home.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(asdict(settings), stream, indent=2, ensure_ascii=False)
            _ = stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise ConfigError("reflectlog.json: unable to save configuration") from error
    finally:
        temporary.unlink(missing_ok=True)
