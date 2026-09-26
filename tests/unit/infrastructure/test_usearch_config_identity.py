from dataclasses import replace
from inspect import Parameter, signature

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.utils.security import SecretString
from reflectlog.core.config_adapters import ConfigAdapter
from reflectlog.core.enums import EmbedderProvider, WeMMModel
from reflectlog.infrastructure.usearch_engine import USearchConfig


@pytest.mark.parametrize(
    ("provider", "model", "dimensions"),
    [
        (EmbedderProvider.OPENAI, "openai/text-embedding-3-small", 1536),
        (EmbedderProvider.LANGCHAIN, "Qwen/Qwen3-Embedding-4B", 2560),
        (EmbedderProvider.WEMM, WeMMModel.EMBEDDING_4B, 1024),
    ],
)
def test_from_config_keeps_provider_model_and_effective_dimensions(
    provider: EmbedderProvider, model: str, dimensions: int
) -> None:
    given = Config(
        workspace_id="identity-test",
        openrouter_api_key=SecretString("test-key"),
        embedder_provider=provider,
        embedding_model=model,
        embedding_dims=1536,
        qwen_embedding_dims=2560,
        wemm_embedding_dims=1024,
    )

    result = USearchConfig.from_config(ConfigAdapter(given))

    assert (
        result.embedder_provider,
        result.embedding_model,
        result.embedding_dims,
    ) == (
        provider,
        model,
        dimensions,
    )


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        (EmbedderProvider.OPENAI, "openai/text-embedding-3-small"),
        (EmbedderProvider.LANGCHAIN, "Qwen/Qwen3-Embedding-0.6B"),
        (EmbedderProvider.WEMM, WeMMModel.EMBEDDING_2B),
    ],
)
def test_equal_width_does_not_erase_provider_or_model(
    provider: EmbedderProvider, model: str
) -> None:
    given = Config(
        workspace_id="identity-test",
        openrouter_api_key=SecretString("test-key"),
        embedder_provider=provider,
        embedding_model=model,
        embedding_dims=1024,
        qwen_embedding_dims=1024,
        wemm_embedding_dims=1024,
    )

    result = USearchConfig.from_config(ConfigAdapter(given))

    assert result.embedding_dims == 1024
    assert (result.embedder_provider, result.embedding_model) == (provider, model)
    other = replace(given, embedding_model="different/model")
    assert USearchConfig.from_config(ConfigAdapter(other)) != result


def test_from_dict_requires_provider_and_model() -> None:
    given = {"workspace_id": "identity-test", "embedding_dims": 1024}

    with pytest.raises(KeyError, match="embedder_provider"):
        USearchConfig.from_dict(given)
    with pytest.raises(KeyError, match="embedding_model"):
        USearchConfig.from_dict({**given, "embedder_provider": "openai"})


def test_direct_config_requires_provider_and_model() -> None:
    parameters = signature(USearchConfig).parameters
    assert parameters["embedder_provider"].default is Parameter.empty
    assert parameters["embedding_model"].default is Parameter.empty


def test_from_dict_parses_provider_and_keeps_model() -> None:
    given = {
        "embedder_provider": "wemm",
        "embedding_model": WeMMModel.EMBEDDING_4B.value,
        "embedding_dims": 1024,
    }

    result = USearchConfig.from_dict(given)

    assert result.embedder_provider is EmbedderProvider.WEMM
    assert result.embedding_model == WeMMModel.EMBEDDING_4B.value
    assert result.embedding_dims == 1024
