# pyright: basic
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "value, expected",
    [
        ("true", True),
        ("YES", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("0", False),
    ],
)
def test_wizard_boolean_coercion(plugin_import, value, expected):
    # Given wizard strings, when parsing, then booleans are typed.
    config = plugin_import("core.config")
    assert (
        config.parse_settings({"recall": value}, url_override=None).recall is expected
    )


def test_wizard_numbers_and_defaults_are_frozen(plugin_import):
    # Given wizard numeric strings, when parsing, then defaults remain immutable.
    config = plugin_import("core.config")
    result = config.parse_settings(
        {"recall_limit": "8", "recall_timeout": "2.5"}, url_override=None
    )
    assert result.recall_limit == 8 and result.recall_timeout == 2.5
    assert result == config.Settings(recall_limit=8, recall_timeout=2.5)
    with pytest.raises(FrozenInstanceError):
        result.recall = False


@pytest.mark.parametrize(
    "key, value",
    [
        ("unknown", "secret-value"),
        ("token", "secret-value"),
        ("namespace", "UPPER"),
        ("workspace_scope", "chat"),
        ("recall", 2),
        ("recall_limit", True),
        ("recall_limit", "1.5"),
        ("workspace", "."),
        ("workspace", "a..b"),
    ],
)
def test_invalid_configuration_is_redacted(plugin_import, key, value):
    # Given invalid settings, when parsing, then errors identify keys only.
    config = plugin_import("core.config")
    with pytest.raises(config.ConfigError) as caught:
        config.parse_settings({key: value}, url_override=None)
    assert key in str(caught.value) and "secret-value" not in str(caught.value)


@pytest.mark.parametrize(
    "key, low, high",
    [
        ("recall_limit", 1, 50),
        ("recall_timeout", 0.5, 6.0),
        ("recall_max_chars", 1000, 20000),
        ("connect_timeout", 1, 60),
        ("request_timeout", 1, 120),
    ],
)
def test_numeric_ranges(plugin_import, key, low, high):
    # Given boundary values, when parsing, then endpoints pass and outside fails.
    config = plugin_import("core.config")
    for value in (low, high):
        assert config.parse_settings({key: value}, url_override=None)
    for value in (low - 1, high + 1, "nan", "inf", 10**400):
        with pytest.raises(config.ConfigError, match=key):
            config.parse_settings({key: value}, url_override=None)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://host/path",
        "http:///path",
        "https://user:secret@host",
        "https://host?q=secret",
        "https://host/#secret",
        "http://host:bad",
        "http://host?",
        "http://host#",
    ],
)
def test_url_restrictions(plugin_import, url):
    # Given an unsafe endpoint, when parsing, then no URL or credential leaks.
    config = plugin_import("core.config")
    with pytest.raises(config.ConfigError, match="url") as caught:
        config.parse_settings({"url": url}, url_override=None)
    assert "secret" not in str(caught.value)


def test_url_override_wins(plugin_import):
    # Given a configured endpoint, when overriding, then the override wins.
    config = plugin_import("core.config")
    assert (
        config.parse_settings(
            {"url": "https://file.example/mcp"}, url_override="https://env.example/mcp"
        ).url
        == "https://env.example/mcp"
    )


def test_missing_file_defaults(plugin_import, tmp_path):
    # Given no file, when loading, then defaults apply.
    config = plugin_import("core.config")
    assert config.load_settings(tmp_path, url_override=None) == config.Settings()


@pytest.mark.parametrize("content", ["{", "[]", "null", '{"recall":2}'])
def test_invalid_file(plugin_import, tmp_path, content):
    # Given invalid JSON/config, when loading, then the safe error is surfaced.
    config = plugin_import("core.config")
    (tmp_path / "reflectlog.json").write_text(content)
    with pytest.raises(config.ConfigError):
        config.load_settings(tmp_path, url_override=None)


def test_unreadable_file(plugin_import, tmp_path):
    # Given a directory in place of the file, when loading, then it fails closed.
    config = plugin_import("core.config")
    (tmp_path / "reflectlog.json").mkdir()
    with pytest.raises(config.ConfigError):
        config.load_settings(tmp_path, url_override=None)


def test_save_merges_drops_secrets_and_creates_private_temp(
    plugin_import, tmp_path, monkeypatch
):
    # Given prior settings and wizard secrets, observe the atomic publication seam.
    config = plugin_import("core.config")
    path = tmp_path / "reflectlog.json"
    path.write_text(json.dumps({"namespace": "existing", "old_token": "old"}))
    replace = os.replace
    observed = []

    def inspect_replace(source, destination):
        observed.append(Path(source).stat().st_mode & 0o777)
        replace(source, destination)

    monkeypatch.setattr(config.os, "replace", inspect_replace)
    # When saving a partial update.
    config.save_settings(
        {
            "recall_limit": "9",
            "token": "secret",
            "api_key": "secret",
            "password": "secret",
            "private_secret": "secret",
            "other_key": "secret",
        },
        tmp_path,
    )
    # Then merge and permission semantics hold without persisting secrets.
    data = json.loads(path.read_text())
    assert data["namespace"] == "existing" and data["recall_limit"] == 9
    assert not any("secret" in str(value) for value in data.values())
    assert observed == [0o600] and path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [path]


def test_failed_save_preserves_existing_file(plugin_import, tmp_path):
    # Given valid existing settings, when an invalid update is saved, then no write occurs.
    config = plugin_import("core.config")
    path = tmp_path / "reflectlog.json"
    path.write_text('{"namespace":"existing"}')
    with pytest.raises(config.ConfigError):
        config.save_settings({"recall_limit": 0}, tmp_path)
    assert path.read_text() == '{"namespace":"existing"}'
