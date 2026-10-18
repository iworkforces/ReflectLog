"""Composition-root tests for the WeMM embedding provider."""

from dataclasses import replace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.engine_factory import EngineFactory
from reflectlog.application.memory.manager import MemoryManager, wemm_config
from reflectlog.application.memory.workspace_registry import WorkspaceRegistry
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.application.utils.security import SecretString
from reflectlog.core.enums import EmbedderProvider, WeMMDevice, WeMMModel
from reflectlog.core.exceptions import StorageError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.embeddings.cached_embeddings import CachedEmbeddings
from reflectlog.infrastructure.embeddings.wemm_embedding import WeMMEmbeddings


def _config(*, cache_enabled: bool) -> Config:
    return Config(
        workspace_id="wemm-test",
        openrouter_api_key=SecretString("test-key"),
        admin_password=SecretString("test-admin-password"),
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
        admin_password=config.admin_password,
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


def test_memory_manager_production_path_borrows_wemm() -> None:
    config = _config(cache_enabled=True)
    manager = object.__new__(MemoryManager)
    manager.config = config
    manager.logger = _logger()
    manager._coordinator = MagicMock()
    with (
        patch("reflectlog.application.memory.manager.acquire_wemm") as acquire,
        patch("reflectlog.application.memory.manager.USearchEngine") as engine_class,
    ):
        acquire.return_value = WeMMEmbeddings(wemm_config(config))
        manager._init_semantic_engine()

    assert manager._semantic_engine is engine_class.return_value
    cached = engine_class.call_args.kwargs["embedder"]
    assert isinstance(cached, CachedEmbeddings)
    assert cached.embedder is acquire.return_value
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
            embedder_provider=EmbedderProvider.WEMM,
            embedding_model=WeMMModel.EMBEDDING_4B,
        ),
        embedder=cached,
    )

    engine.close()
    engine.close()

    assert embedder.close.call_count == 1


def test_independent_managers_share_provider_without_sharing_cache(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = replace(_config(cache_enabled=True), wemm_device=WeMMDevice.CPU)
    with (
        patch("reflectlog.application.memory.manager.USearchEngine") as engines,
        patch("reflectlog.application.memory.manager.TantivyEngine"),
        patch(
            "reflectlog.infrastructure.embeddings.wemm_service.WeMMEmbeddings",
            wraps=WeMMEmbeddings,
        ) as providers,
        patch.object(WeMMEmbeddings, "close", autospec=True) as provider_close,
    ):
        first = MemoryManager(replace(config, workspace_id="alpha"), _logger())
        second = MemoryManager(replace(config, workspace_id="beta"), _logger())
        try:
            assert providers.call_count == 1
            first_cache = engines.call_args_list[0].kwargs["embedder"]
            second_cache = engines.call_args_list[1].kwargs["embedder"]
            assert first_cache is not second_cache
            assert first_cache.embedder is not second_cache.embedder
            first.close()
            provider_close.assert_not_called()
        finally:
            first.close()
            second.close()
        provider_close.assert_called_once()


def test_failed_manager_init_and_failed_persist_keep_correct_ownership(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = replace(_config(cache_enabled=False), wemm_device=WeMMDevice.CPU)
    with (
        patch("reflectlog.application.memory.manager.USearchEngine") as engines,
        patch("reflectlog.application.memory.manager.TantivyEngine"),
        patch(
            "reflectlog.infrastructure.embeddings.wemm_service.WeMMEmbeddings",
            wraps=WeMMEmbeddings,
        ) as providers,
        patch.object(WeMMEmbeddings, "close", autospec=True) as provider_close,
    ):
        semantic = MagicMock()
        engines.side_effect = [RuntimeError("construction"), semantic]
        with pytest.raises(RuntimeError, match="construction"):
            MemoryManager(replace(config, workspace_id="failed"), _logger())
        provider_close.assert_called_once()
        manager = MemoryManager(replace(config, workspace_id="retry"), _logger())
        semantic.commit.side_effect = RuntimeError("persist")
        try:
            with pytest.raises(StorageError):
                manager.close()
            assert provider_close.call_count == 1
            semantic.commit.side_effect = None
        finally:
            manager.close()
        assert providers.call_count == 2
        assert provider_close.call_count == 2


@pytest.mark.asyncio
async def test_registry_eviction_keeps_model_until_successful_shutdown(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = replace(
        _config(cache_enabled=False), workspace_id="", wemm_device=WeMMDevice.CPU
    )
    with (
        patch("reflectlog.application.memory.manager.USearchEngine"),
        patch("reflectlog.application.memory.manager.TantivyEngine"),
        patch(
            "reflectlog.infrastructure.embeddings.wemm_service.WeMMEmbeddings",
            wraps=WeMMEmbeddings,
        ) as providers,
        patch.object(WeMMEmbeddings, "close", autospec=True) as provider_close,
    ):
        first = WorkspaceRegistry(config, idle_ttl=0)
        second = WorkspaceRegistry(
            config, lambda concrete: MemoryManager(concrete, _logger()), idle_ttl=0
        )
        async with first.acquire("alpha"):
            pass
        await first.prune()
        provider_close.assert_not_called()
        async with second.acquire("beta"):
            assert providers.call_count == 1
        await first.close()
        provider_close.assert_not_called()
        await second.close()
        provider_close.assert_called_once()


def test_engine_teardown_failure_retains_manager_borrow_until_retry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = replace(_config(cache_enabled=False), wemm_device=WeMMDevice.CPU)
    with (
        patch("reflectlog.application.memory.manager.USearchEngine") as engine_class,
        patch("reflectlog.application.memory.manager.TantivyEngine"),
        patch(
            "reflectlog.infrastructure.embeddings.wemm_service.WeMMEmbeddings",
            wraps=WeMMEmbeddings,
        ) as providers,
        patch.object(WeMMEmbeddings, "close", autospec=True) as provider_close,
    ):
        manager = MemoryManager(config, _logger())
        engine_class.return_value.close.side_effect = RuntimeError("teardown")
        with pytest.raises(StorageError):
            manager.close()
        provider_close.assert_not_called()
        other = MemoryManager(replace(config, workspace_id="other"), _logger())
        providers.assert_called_once()
        engine_class.return_value.close.side_effect = None
        manager.close()
        provider_close.assert_not_called()
        other.close()
        provider_close.assert_called_once()
        manager.close()
        other.close()
        provider_close.assert_called_once()


@pytest.mark.asyncio
async def test_failed_registry_shutdown_retains_pin_until_retry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = replace(
        _config(cache_enabled=False), workspace_id="", wemm_device=WeMMDevice.CPU
    )
    with (
        patch("reflectlog.application.memory.manager.USearchEngine") as engine_class,
        patch("reflectlog.application.memory.manager.TantivyEngine"),
        patch.object(WeMMEmbeddings, "close", autospec=True) as provider_close,
    ):
        registry = WorkspaceRegistry(config)
        async with registry.acquire("alpha"):
            pass
        engine_class.return_value.commit.side_effect = RuntimeError("persist")
        with pytest.raises(ExceptionGroup, match="could not be closed"):
            await registry.close()
        provider_close.assert_not_called()
        engine_class.return_value.commit.side_effect = None
        await registry.close()
        provider_close.assert_called_once()
