from dataclasses import replace

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.config.validation import ValidationError, validate_config
from reflectlog.application.utils.security import SecretString
from reflectlog.core.exceptions import ConfigurationError


@pytest.mark.parametrize("password", [None, "", " \t\n", "\u2003"])
def test_environment_requires_nonblank_admin_password(
    monkeypatch: pytest.MonkeyPatch, password: str | None
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ALLOWED_TOOLS", "search_memories")
    if password is None:
        monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("ADMIN_PASSWORD", password)

    with pytest.raises(ConfigurationError, match="ADMIN_PASSWORD"):
        Config.from_environment()


@pytest.mark.parametrize("password", ["", " \t\n", "\u2003"])
def test_direct_configuration_rejects_blank_admin_password(password: str) -> None:
    with pytest.raises(ConfigurationError, match="ADMIN_PASSWORD"):
        Config(
            workspace_id="",
            openrouter_api_key=SecretString("test-key"),
            admin_password=SecretString(password),
        )


def test_admin_password_keeps_exact_value_and_masks_representations(
    monkeypatch: pytest.MonkeyPatch, set_env_vars: dict[str, str]
) -> None:
    password = "  private-\u5bc6\u78bc  "
    monkeypatch.setenv("ADMIN_PASSWORD", password)

    config = Config.from_environment()

    assert config.admin_password.get_secret_value() == password
    assert password not in repr(config)
    assert password not in repr(config.admin_password)
    assert password not in str(config.admin_password)


def test_admin_password_validation_error_masks_secret() -> None:
    error = ValidationError("ADMIN_PASSWORD", "private-secret", "Invalid password")

    rendered = str(error)

    assert "private-secret" not in rendered
    assert "***REDACTED***" in rendered


def test_validator_reports_blank_admin_password(set_env_vars: dict[str, str]) -> None:
    config = replace(Config.from_environment(), workspace_id="test")
    object.__setattr__(config, "admin_password", SecretString(" "))

    errors = validate_config(config)

    assert [error.field for error in errors] == ["ADMIN_PASSWORD"]
    assert errors[0].value == "***REDACTED***"
