"""Composition-root tests for the WeMM embedding provider."""

from typing import cast
from unittest.mock import MagicMock, patch

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.engine_factory import EngineFactory
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.application.utils.security import SecretString
from reflectlog.core.enums import EmbedderProvider, WeMMDevice, WeMMModel
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.embeddings.cached_embeddings import CachedEmbeddings
from reflectlog.infrastructure.embeddings.wemm_embedding import WeMMEmbeddings


def _config(*, cache_enabled: bool) -> Config:
    return Config(
        workspace_id="wemm-test",
        openrouter_api_key=SecretString("test-key"),
        embedder_provider=EmbedderProvider.WEMM,
        embedding_model=WeMMModel.EMBEDDING_4B,
        wemm_embedding_dims=1024,
        wemm_device=WeMMDevice.MPS,
        embedding_batch_size=3,
        embedding_cache_enabled=cache_enabled,
        eager_initialization=False,
    )


def _logger() -> IStructuredLogger:
    return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))


def test_engine_factory_builds_wemm_and_applies_existing_cache_wrapper() -> None:
    factory = EngineFactory()
    result = factory._create_embedder(_config(cache_enabled=True), _logger())

    assert isinstance(result, CachedEmbeddings)
    assert isinstance(result.embedder, WeMMEmbeddings)
    assert result.role_separated is True
    wemm_config = result.embedder.config
    assert wemm_config.model is WeMMModel.EMBEDDING_4B
    assert wemm_config.dimensions == 1024
    assert wemm_config.device is WeMMDevice.MPS
    assert wemm_config.batch_size == 3


def test_engine_factory_keeps_remote_provider_on_existing_adapter() -> None:
    config = _config(cache_enabled=False)
    remote_config = Config(
        workspace_id=config.workspace_id,
        openrouter_api_key=config.openrouter_api_key,
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_cache_enabled=False,
        eager_initialization=False,
    )
    with (
        patch(
            "reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings"
        ) as remote_class,
        patch(
            "reflectlog.application.memory.engine_factory.WeMMEmbeddings"
        ) as wemm_class,
    ):
        result = EngineFactory()._create_embedder(remote_config, _logger())

    assert result is remote_class.return_value
    wemm_class.assert_not_called()


def test_memory_manager_production_path_builds_wemm_directly() -> None:
    config = _config(cache_enabled=True)
    manager = object.__new__(MemoryManager)
    manager.config = config
    manager.logger = _logger()
    manager._coordinator = MagicMock()
    with (
        patch("reflectlog.application.memory.manager.WeMMEmbeddings") as embedder_class,
        patch("reflectlog.application.memory.manager.USearchEngine") as engine_class,
    ):
        embedder_class.return_value = MagicMock(spec=WeMMEmbeddings)
        manager._init_semantic_engine()

    assert manager._semantic_engine is engine_class.return_value
    cached = engine_class.call_args.kwargs["embedder"]
    assert isinstance(cached, CachedEmbeddings)
    assert cached.embedder is embedder_class.return_value
    assert cached.role_separated is True


def test_cached_and_usearch_close_reach_wemm_model_idempotently(tmp_path) -> None:
    from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine

    embedder = MagicMock(spec=WeMMEmbeddings)
    cached = CachedEmbeddings(embedder=embedder)
    engine = USearchEngine(
        USearchConfig(
            workspace_id="wemm-test",
            index_path=str(tmp_path / "vectors.usearch"),
            db_path=str(tmp_path / "memories.db"),
            embedding_dims=64,
        ),
        embedder=cached,
    )

    engine.close()
    engine.close()

    assert embedder.close.call_count == 1
