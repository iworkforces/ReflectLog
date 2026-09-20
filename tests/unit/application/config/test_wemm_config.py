"""Configuration contract tests for the local WeMM provider."""

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.utils.security import SecretString
from reflectlog.core.config_adapters import ConfigAdapter
from reflectlog.core.enums import EmbedderProvider, WeMMDevice, WeMMModel
from reflectlog.core.exceptions import ConfigurationError
from reflectlog.infrastructure.usearch_engine import USearchConfig


@pytest.mark.parametrize(
    ("model", "native_dimensions"),
    [
        (WeMMModel.EMBEDDING_2B, 2048),
        (WeMMModel.EMBEDDING_4B, 2560),
        (WeMMModel.EMBEDDING_9B, 4096),
    ],
)
def test_wemm_native_dimension_is_derived_when_override_absent(
    monkeypatch: pytest.MonkeyPatch,
    model: WeMMModel,
    native_dimensions: int,
) -> None:
    monkeypatch.setenv("WORKSPACE_ID", "wemm-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDER_PROVIDER", EmbedderProvider.WEMM)
    monkeypatch.setenv("EMBEDDING_MODEL", model)
    monkeypatch.delenv("WEMM_EMBEDDING_DIMS", raising=False)

    config = Config.from_environment()

    assert config.embedder_provider is EmbedderProvider.WEMM
    assert config.wemm_embedding_dims == native_dimensions
    assert (
        USearchConfig.from_config(ConfigAdapter(config)).embedding_dims
        == native_dimensions
    )


@pytest.mark.parametrize(
    ("model", "dimension"),
    [
        (WeMMModel.EMBEDDING_2B, 64),
        (WeMMModel.EMBEDDING_2B, 2048),
        (WeMMModel.EMBEDDING_4B, 2560),
        (WeMMModel.EMBEDDING_9B, 2048),
        (WeMMModel.EMBEDDING_9B, 4096),
    ],
)
def test_wemm_supported_dimension_override_is_accepted(
    monkeypatch: pytest.MonkeyPatch, model: WeMMModel, dimension: int
) -> None:
    monkeypatch.setenv("WORKSPACE_ID", "wemm-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDER_PROVIDER", EmbedderProvider.WEMM)
    monkeypatch.setenv("EMBEDDING_MODEL", model)
    monkeypatch.setenv("WEMM_EMBEDDING_DIMS", str(dimension))

    config = Config.from_environment()

    assert config.wemm_embedding_dims == dimension


@pytest.mark.parametrize(
    ("model", "dimension"),
    [
        (WeMMModel.EMBEDDING_2B, 2560),
        (WeMMModel.EMBEDDING_4B, 2048),
        (WeMMModel.EMBEDDING_9B, 2560),
    ],
)
def test_wemm_unsupported_model_dimension_is_rejected(
    monkeypatch: pytest.MonkeyPatch, model: WeMMModel, dimension: int
) -> None:
    monkeypatch.setenv("WORKSPACE_ID", "wemm-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDER_PROVIDER", EmbedderProvider.WEMM)
    monkeypatch.setenv("EMBEDDING_MODEL", model)
    monkeypatch.setenv("WEMM_EMBEDDING_DIMS", str(dimension))

    with pytest.raises(ConfigurationError, match="WEMM_EMBEDDING_DIMS"):
        Config.from_environment()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("EMBEDDER_PROVIDER", "unknown"),
        ("EMBEDDING_MODEL", "tencent/WeMM-Embedding-3B"),
        ("WEMM_DEVICE", "metal"),
    ],
)
def test_wemm_closed_configuration_rejects_unknown_values(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    monkeypatch.setenv("WORKSPACE_ID", "wemm-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDER_PROVIDER", EmbedderProvider.WEMM)
    monkeypatch.setenv("EMBEDDING_MODEL", WeMMModel.EMBEDDING_2B)
    monkeypatch.setenv(field, value)

    with pytest.raises(ConfigurationError):
        Config.from_environment()


def test_existing_embedding_defaults_remain_unchanged() -> None:
    direct = Config(
        workspace_id="defaults",
        openrouter_api_key=SecretString("key"),
    )

    assert direct.embedder_provider is EmbedderProvider.OPENAI
    assert direct.embedding_model == "openai/text-embedding-3-large"
    assert direct.embedding_dims == 3072
    assert direct.qwen_embedding_dims == 4096
    assert direct.wemm_device is WeMMDevice.AUTO
