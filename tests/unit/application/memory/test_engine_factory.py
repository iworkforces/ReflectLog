"""Unit tests for EngineFactory and standalone factory functions.

Tests cover:
- EngineFactory.create_engines: Full engine creation pipeline
- EngineFactory._create_semantic_engine: USearch engine setup
- EngineFactory._create_embedder: Embedder with/without caching
- EngineFactory._create_fusion_engine: Fusion engine creation
- EngineFactoryResult: Dataclass structure
- create_cross_encoder_reranker: CrossEncoder reranker factory
- create_smart_replacer: SmartReplacer factory
"""

from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.engine_factory import (
    EngineFactory,
    EngineFactoryResult,
    create_cross_encoder_reranker,
    create_smart_replacer,
)
from reflectlog.application.memory.fusion.base import FusionEngine
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.application.utils.security import SecretString
from reflectlog.core.enums import RerankerEngine
from reflectlog.core.exceptions import InitializationError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.embedding_identity import ensure_embedding_identity
from reflectlog.infrastructure.storage_coordinator import PortalockerStorageCoordinator
from reflectlog.infrastructure.tantivy_engine import TantivyEngine
from reflectlog.infrastructure.usearch_engine import USearchEngine


@pytest.fixture
def mock_config() -> Mock:
    """Mock configuration with typical settings."""
    config = Mock(spec=Config)
    config.workspace_id = "test_project"
    config.tantivy_index_path_template = "indexes/{workspace_id}/tantivy"
    config.tantivy_normalize_scores = True
    config.tantivy_soft_delete_enabled = True
    config.tantivy_compaction_threshold_ratio = 0.2
    config.tantivy_compaction_max_tombstones = 10000
    config.tantivy_tombstone_ttl_days = 7
    config.reranker_engine = "cross_encoder"

    # Embedding settings
    config.embedder_provider = "openai"
    config.embedding_model = "openai/text-embedding-3-large"
    config.embedding_dims = 3072
    config.qwen_embedding_dims = 4096
    config.openrouter_api_key = Mock()
    config.openrouter_api_key.get_secret_value.return_value = "test-api-key"
    config.openrouter_base_url = "https://openrouter.ai/api/v1"
    config.embedding_batch_size = 512
    config.embedding_max_concurrent_batches = 4
    config.embedding_cache_enabled = True
    config.embedding_cache_size = 100

    # Fusion settings
    config.fusion_method = "rrf"
    config.fusion_normalization = None
    config.fusion_rrf_k = 60
    config.fusion_weights = None

    # Smart replacement
    config.enable_smart_replace = True

    return config


@pytest.fixture
def mock_logger() -> IStructuredLogger:
    """Mock structured logger."""
    return cast(IStructuredLogger, Mock(spec=StructuredLogger))


@pytest.fixture
def factory() -> EngineFactory:
    """Create an EngineFactory instance."""
    return EngineFactory()


@pytest.fixture(autouse=True)
def _stub_identity_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "reflectlog.application.memory.engine_factory.ensure_embedding_identity",
        lambda _config, coordinator, *, tantivy_index_path: coordinator,
    )


@pytest.mark.unit
class TestEngineFactoryResult:
    """Tests for EngineFactoryResult dataclass."""

    def test_result_stores_all_fields(self) -> None:
        """Result dataclass should store all engine references."""
        semantic = cast(USearchEngine, Mock())
        tantivy = cast(TantivyEngine, Mock())
        fusion = cast(FusionEngine, Mock())

        result = EngineFactoryResult(
            semantic_engine=semantic,
            tantivy_engine=tantivy,
            fusion_engine=fusion,
            reranker_engine=RerankerEngine.CROSS_ENCODER,
        )

        assert result.semantic_engine is semantic
        assert result.tantivy_engine is tantivy
        assert result.fusion_engine is fusion
        assert result.reranker_engine == "cross_encoder"


@pytest.mark.unit
class TestEngineFactoryInit:
    """Tests for EngineFactory initialization."""

    def test_factory_initialization(self) -> None:
        """Factory should initialize without errors."""
        factory = EngineFactory()
        assert factory is not None


@pytest.mark.unit
class TestCreateEngines:
    """Tests for EngineFactory.create_engines orchestration."""

    def test_external_tantivy_rejected_before_engines(
        self,
        factory: EngineFactory,
        mock_logger: IStructuredLogger,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "reflectlog.application.memory.engine_factory.ensure_embedding_identity",
            ensure_embedding_identity,
        )
        external = tmp_path / "external" / "tantivy"
        external.mkdir(parents=True)
        (external / "metadata.json").write_text("legacy")
        config = Config(
            workspace_id="factory_project",
            openrouter_api_key=SecretString("test-api-key"),
            tantivy_index_path_template=str(external),
        )
        coordinator = PortalockerStorageCoordinator(str(tmp_path / "indexes"))

        with (
            patch(
                "reflectlog.application.memory.engine_factory.USearchEngine"
            ) as semantic,
            patch(
                "reflectlog.application.memory.engine_factory.TantivyEngine"
            ) as tantivy,
            patch(
                "reflectlog.application.memory.engine_factory.WeMMEmbeddings"
            ) as embedder,
            pytest.raises(InitializationError, match="Legacy workspace"),
        ):
            factory.create_engines(config, mock_logger, coordinator)

        semantic.assert_not_called()
        tantivy.assert_not_called()
        embedder.assert_not_called()
        assert not Path(coordinator.paths_for(config.workspace_id).root).exists()
        assert (external / "metadata.json").read_text() == "legacy"

    def test_preflight_rejects_mismatch_before_embedder_and_shares_coordinator(
        self,
        factory: EngineFactory,
        mock_logger: IStructuredLogger,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "reflectlog.application.memory.engine_factory.ensure_embedding_identity",
            ensure_embedding_identity,
        )
        config = Config(
            workspace_id="factory_project",
            openrouter_api_key=SecretString("test-api-key"),
            embedding_cache_enabled=False,
        )
        coordinator = PortalockerStorageCoordinator(str(tmp_path / "indexes"))
        with (
            patch(
                "reflectlog.application.memory.engine_factory.USearchEngine"
            ) as semantic,
            patch(
                "reflectlog.application.memory.engine_factory.TantivyEngine"
            ) as tantivy,
            patch(
                "reflectlog.application.memory.engine_factory.WeMMEmbeddings"
            ) as embedder,
            patch("reflectlog.application.memory.engine_factory.create_fusion_engine"),
        ):
            factory.create_engines(config, mock_logger, coordinator)
            assert semantic.call_args.kwargs["coordinator"] is coordinator
            assert tantivy.call_args.kwargs["coordinator"] is coordinator
            embedder.reset_mock()

            with pytest.raises(InitializationError, match="identity"):
                factory.create_engines(
                    replace(config, embedding_model="different/model"),
                    mock_logger,
                    coordinator,
                )

            embedder.assert_not_called()

    @patch("reflectlog.application.memory.engine_factory.create_fusion_engine")
    @patch("reflectlog.application.memory.engine_factory.TantivyEngine")
    @patch("reflectlog.application.memory.engine_factory.TantivyConfig")
    @patch("reflectlog.application.memory.engine_factory.USearchEngine")
    @patch("reflectlog.application.memory.engine_factory.USearchConfig")
    @patch("reflectlog.application.memory.engine_factory.CachedEmbeddings")
    @patch("reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings")
    def test_create_engines_creates_all_engines(
        self,
        mock_embedder_cls: Mock,
        mock_cached_cls: Mock,
        mock_usearch_config_cls: Mock,
        mock_usearch_cls: Mock,
        mock_tantivy_config_cls: Mock,
        mock_tantivy_cls: Mock,
        mock_create_fusion: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        mock_config.reranker_engine = "cross_encoder"

        result = factory.create_engines(mock_config, mock_logger)

        assert isinstance(result, EngineFactoryResult)
        assert result.semantic_engine is mock_usearch_cls.return_value
        assert result.tantivy_engine is mock_tantivy_cls.return_value
        assert result.fusion_engine is mock_create_fusion.return_value
        assert result.reranker_engine == "cross_encoder"

    @patch("reflectlog.application.memory.engine_factory.create_fusion_engine")
    @patch("reflectlog.application.memory.engine_factory.TantivyEngine")
    @patch("reflectlog.application.memory.engine_factory.TantivyConfig")
    @patch("reflectlog.application.memory.engine_factory.USearchEngine")
    @patch("reflectlog.application.memory.engine_factory.USearchConfig")
    @patch("reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings")
    def test_create_engines_no_cache(
        self,
        mock_embedder_cls: Mock,
        mock_usearch_config_cls: Mock,
        mock_usearch_cls: Mock,
        mock_tantivy_config_cls: Mock,
        mock_tantivy_cls: Mock,
        mock_create_fusion: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        """create_engines should use base embedder when cache disabled."""
        mock_config.embedding_cache_enabled = False

        result = factory.create_engines(mock_config, mock_logger)

        assert result.semantic_engine is mock_usearch_cls.return_value
        # USearchEngine should receive the base embedder directly
        call_kwargs = mock_usearch_cls.call_args
        assert call_kwargs is not None


@pytest.mark.unit
class TestCreateSemanticEngine:
    """Tests for EngineFactory._create_semantic_engine."""

    @patch("reflectlog.application.memory.engine_factory.USearchEngine")
    @patch("reflectlog.application.memory.engine_factory.USearchConfig")
    @patch("reflectlog.application.memory.engine_factory.CachedEmbeddings")
    @patch("reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings")
    def test_creates_usearch_with_config_and_embedder(
        self,
        mock_embedder_cls: Mock,
        mock_cached_cls: Mock,
        mock_usearch_config_cls: Mock,
        mock_usearch_cls: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        """Semantic engine should be created with USearchConfig and embedder."""
        result = factory._create_semantic_engine(mock_config, mock_logger)

        mock_usearch_config_cls.from_config.assert_called_once()
        mock_usearch_cls.assert_called_once_with(
            mock_usearch_config_cls.from_config.return_value,
            embedder=mock_cached_cls.return_value,
            logger=mock_logger,
            coordinator=None,
        )
        assert result is mock_usearch_cls.return_value

    @patch("reflectlog.application.memory.engine_factory.USearchEngine")
    @patch("reflectlog.application.memory.engine_factory.USearchConfig")
    @patch("reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings")
    def test_creates_usearch_without_cache(
        self,
        mock_embedder_cls: Mock,
        mock_usearch_config_cls: Mock,
        mock_usearch_cls: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        """Semantic engine should use base embedder when cache disabled."""
        mock_config.embedding_cache_enabled = False

        factory._create_semantic_engine(mock_config, mock_logger)

        mock_usearch_cls.assert_called_once_with(
            mock_usearch_config_cls.from_config.return_value,
            embedder=mock_embedder_cls.return_value,
            logger=mock_logger,
            coordinator=None,
        )


@pytest.mark.unit
class TestCreateEmbedder:
    """Tests for EngineFactory._create_embedder."""

    @patch("reflectlog.application.memory.engine_factory.CachedEmbeddings")
    @patch("reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings")
    def test_cache_enabled_wraps_embedder(
        self,
        mock_embedder_cls: Mock,
        mock_cached_cls: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        """When cache enabled, embedder should be wrapped with CachedEmbeddings."""
        mock_config.embedding_cache_enabled = True
        mock_config.embedding_cache_size = 200

        result = factory._create_embedder(mock_config, mock_logger)

        mock_cached_cls.assert_called_once_with(
            embedder=mock_embedder_cls.return_value,
            cache_size=200,
            enabled=True,
            logger=mock_logger,
        )
        assert result is mock_cached_cls.return_value

    @patch("reflectlog.application.memory.engine_factory.CachedEmbeddings")
    @patch("reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings")
    def test_cache_disabled_returns_base_embedder(
        self,
        mock_embedder_cls: Mock,
        mock_cached_cls: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        """When cache disabled, base embedder should be returned directly."""
        mock_config.embedding_cache_enabled = False

        result = factory._create_embedder(mock_config, mock_logger)

        mock_cached_cls.assert_not_called()
        assert result is mock_embedder_cls.return_value

    @patch("reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings")
    def test_embedder_config_openai_provider(
        self,
        mock_embedder_cls: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        """OpenAI provider should use embedding_dims, not qwen_embedding_dims."""
        mock_config.embedding_cache_enabled = False
        mock_config.embedder_provider = "openai"
        mock_config.embedding_dims = 3072
        mock_config.qwen_embedding_dims = 4096

        factory._create_embedder(mock_config, mock_logger)

        call_kwargs = mock_embedder_cls.call_args
        config_dict = (
            call_kwargs[1]["config"]
            if "config" in call_kwargs[1]
            else call_kwargs[0][0]
            if call_kwargs[0]
            else call_kwargs[1].get("config")
        )
        assert config_dict["embedding_dims"] == 3072

    @patch("reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings")
    def test_embedder_config_langchain_provider(
        self,
        mock_embedder_cls: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        """Langchain provider should use qwen_embedding_dims."""
        mock_config.embedding_cache_enabled = False
        mock_config.embedder_provider = "langchain"
        mock_config.embedding_dims = 3072
        mock_config.qwen_embedding_dims = 4096

        factory._create_embedder(mock_config, mock_logger)

        call_kwargs = mock_embedder_cls.call_args
        config_dict = (
            call_kwargs[1]["config"]
            if "config" in call_kwargs[1]
            else call_kwargs[0][0]
            if call_kwargs[0]
            else call_kwargs[1].get("config")
        )
        assert config_dict["embedding_dims"] == 4096

    @patch("reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings")
    def test_embedder_passes_all_config_fields(
        self,
        mock_embedder_cls: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        """Embedder should receive model, api_key, base_url, and batch settings."""
        mock_config.embedding_cache_enabled = False
        mock_config.embedder_provider = "openai"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_dims = 1024
        mock_config.openrouter_api_key.get_secret_value.return_value = "sk-test"
        mock_config.openrouter_base_url = "https://test.api/v1"
        mock_config.embedding_batch_size = 256
        mock_config.embedding_max_concurrent_batches = 8

        factory._create_embedder(mock_config, mock_logger)

        call_kwargs = mock_embedder_cls.call_args
        config_dict = call_kwargs[1].get("config") or call_kwargs[0][0]

        assert config_dict["model"] == "test-model"
        assert config_dict["embedding_dims"] == 1024
        assert config_dict["api_key"] == "sk-test"
        assert config_dict["openai_base_url"] == "https://test.api/v1"
        assert config_dict["batch_size"] == 256
        assert config_dict["max_concurrent_batches"] == 8


@pytest.mark.unit
class TestCreateTantivyEngine:
    """Tests for EngineFactory._create_tantivy_engine."""

    @patch("reflectlog.application.memory.engine_factory.TantivyEngine")
    @patch("reflectlog.application.memory.engine_factory.TantivyConfig")
    def test_creates_tantivy(
        self,
        mock_tantivy_config_cls: Mock,
        mock_tantivy_cls: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        mock_config.workspace_id = "my_project"
        mock_config.tantivy_index_path_template = "indexes/{workspace_id}/tantivy"
        mock_config.tantivy_normalize_scores = True

        result = factory._create_tantivy_engine(mock_config, mock_logger)

        mock_tantivy_config_cls.assert_called_once_with(
            workspace_id="my_project",
            index_path="indexes/my_project/tantivy",
            normalize_scores=True,
            soft_delete_enabled=True,
            compaction_threshold_ratio=0.2,
            compaction_max_tombstones=10000,
            tombstone_ttl_days=7,
        )
        mock_tantivy_cls.assert_called_once_with(
            mock_tantivy_config_cls.return_value,
            logger=mock_logger,
            coordinator=None,
        )
        assert result is mock_tantivy_cls.return_value

    @patch("reflectlog.application.memory.engine_factory.TantivyEngine")
    @patch("reflectlog.application.memory.engine_factory.TantivyConfig")
    def test_index_path_lowercased(
        self,
        mock_tantivy_config_cls: Mock,
        mock_tantivy_cls: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        """Tantivy index path should be lowercased."""
        mock_config.workspace_id = "MyProject"
        mock_config.tantivy_index_path_template = "indexes/{workspace_id}/tantivy"
        mock_config.tantivy_normalize_scores = False

        factory._create_tantivy_engine(mock_config, mock_logger)

        call_kwargs = mock_tantivy_config_cls.call_args
        assert call_kwargs[1]["index_path"] == "indexes/myproject/tantivy"


@pytest.mark.unit
class TestCreateFusionEngine:
    """Tests for EngineFactory._create_fusion_engine."""

    @patch("reflectlog.application.memory.engine_factory.create_fusion_engine")
    def test_creates_fusion_with_config(
        self,
        mock_create_fusion: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        """Fusion engine should be created with config parameters."""
        mock_config.fusion_method = "rrf"
        mock_config.fusion_normalization = "min-max"
        mock_config.fusion_rrf_k = 30

        result = factory._create_fusion_engine(mock_config, mock_logger)

        mock_create_fusion.assert_called_once_with(
            method="rrf",
            normalization="min-max",
            rrf_k=30,
            weights=None,
            logger=mock_logger,
        )
        assert result is mock_create_fusion.return_value

    @patch("reflectlog.application.memory.engine_factory.create_fusion_engine")
    def test_creates_fusion_with_none_normalization(
        self,
        mock_create_fusion: Mock,
        mock_config: Mock,
        mock_logger: Mock,
        factory: EngineFactory,
    ) -> None:
        """Fusion engine should handle None normalization."""
        mock_config.fusion_method = "sum"
        mock_config.fusion_normalization = None
        mock_config.fusion_rrf_k = 60

        factory._create_fusion_engine(mock_config, mock_logger)

        mock_create_fusion.assert_called_once_with(
            method="sum",
            normalization=None,
            rrf_k=60,
            weights=None,
            logger=mock_logger,
        )


@pytest.mark.unit
class TestCreateCrossEncoderReranker:
    """Tests for create_cross_encoder_reranker standalone factory function."""

    @patch("reflectlog.application.memory.engine_factory.CrossEncoderReranker")
    @patch("reflectlog.application.memory.engine_factory.CrossEncoderConfig")
    def test_creates_reranker_when_engine_is_cross_encoder(
        self,
        mock_config_cls: Mock,
        mock_reranker_cls: Mock,
        mock_config: Mock,
        mock_logger: Mock,
    ) -> None:
        """Should create CrossEncoderReranker when reranker_engine is "cross_encoder"."""
        mock_config.reranker_engine = "cross_encoder"

        result = create_cross_encoder_reranker(mock_config, mock_logger)

        mock_config_cls.from_config.assert_called_once()
        mock_reranker_cls.assert_called_once_with(
            config=mock_config_cls.from_config.return_value,
            logger=mock_logger,
        )
        assert result is mock_reranker_cls.return_value

    def test_returns_none_when_engine_is_not_cross_encoder(
        self,
        mock_config: Mock,
        mock_logger: Mock,
    ) -> None:
        """Should return None when reranker_engine is not "cross_encoder"."""
        mock_config.reranker_engine = "none"

        result = create_cross_encoder_reranker(mock_config, mock_logger)

        assert result is None

    def test_returns_none_when_engine_is_none(
        self,
        mock_config: Mock,
        mock_logger: Mock,
    ) -> None:
        """Should return None when reranker_engine is "none"."""
        mock_config.reranker_engine = "none"

        result = create_cross_encoder_reranker(mock_config, mock_logger)

        assert result is None


@pytest.mark.unit
class TestCreateSmartReplacer:
    """Tests for create_smart_replacer standalone factory function."""

    @patch("reflectlog.application.memory.engine_factory.SmartReplacer")
    @patch("reflectlog.application.memory.engine_factory.SmartReplacerConfig")
    def test_creates_replacer_when_enabled(
        self,
        mock_config_cls: Mock,
        mock_replacer_cls: Mock,
        mock_config: Mock,
        mock_logger: Mock,
    ) -> None:
        """Should create SmartReplacer when enable_smart_replace is True."""
        mock_config.enable_smart_replace = True

        result = create_smart_replacer(mock_config, mock_logger)

        mock_config_cls.from_config.assert_called_once()
        mock_replacer_cls.assert_called_once_with(
            config=mock_config_cls.from_config.return_value,
            logger=mock_logger,
        )
        assert result is mock_replacer_cls.return_value

    def test_returns_none_when_disabled(
        self,
        mock_config: Mock,
        mock_logger: Mock,
    ) -> None:
        """Should return None when enable_smart_replace is False."""
        mock_config.enable_smart_replace = False

        result = create_smart_replacer(mock_config, mock_logger)

        assert result is None


@pytest.mark.unit
class TestEndToEnd:
    """Integration-style tests for EngineFactory orchestration."""

    @patch("reflectlog.application.memory.engine_factory.create_fusion_engine")
    @patch("reflectlog.application.memory.engine_factory.TantivyEngine")
    @patch("reflectlog.application.memory.engine_factory.TantivyConfig")
    @patch("reflectlog.application.memory.engine_factory.USearchEngine")
    @patch("reflectlog.application.memory.engine_factory.USearchConfig")
    @patch("reflectlog.application.memory.engine_factory.CachedEmbeddings")
    @patch("reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings")
    def test_full_pipeline_with_all_features(
        self,
        mock_embedder_cls: Mock,
        mock_cached_cls: Mock,
        mock_usearch_config_cls: Mock,
        mock_usearch_cls: Mock,
        mock_tantivy_config_cls: Mock,
        mock_tantivy_cls: Mock,
        mock_create_fusion: Mock,
        mock_config: Mock,
        mock_logger: Mock,
    ) -> None:
        """Full pipeline should wire embedder -> USearch, Tantivy, Fusion correctly."""
        mock_config.embedding_cache_enabled = True
        mock_config.reranker_engine = "cross_encoder"

        factory = EngineFactory()
        result = factory.create_engines(mock_config, mock_logger)

        # Verify embedder is wrapped with cache
        mock_cached_cls.assert_called_once()

        # Verify USearch receives cached embedder
        usearch_call = mock_usearch_cls.call_args
        assert usearch_call[1]["embedder"] is mock_cached_cls.return_value

        # Verify Tantivy is created
        mock_tantivy_cls.assert_called_once()

        # Verify fusion engine is created
        mock_create_fusion.assert_called_once()

        # Verify result fields
        assert result.reranker_engine == "cross_encoder"

    @patch("reflectlog.application.memory.engine_factory.create_fusion_engine")
    @patch("reflectlog.application.memory.engine_factory.TantivyEngine")
    @patch("reflectlog.application.memory.engine_factory.TantivyConfig")
    @patch("reflectlog.application.memory.engine_factory.USearchEngine")
    @patch("reflectlog.application.memory.engine_factory.USearchConfig")
    @patch("reflectlog.application.memory.engine_factory.LangchainQwenEmbeddings")
    def test_pipeline_without_cache_or_reranker(
        self,
        mock_embedder_cls: Mock,
        mock_usearch_config_cls: Mock,
        mock_usearch_cls: Mock,
        mock_tantivy_config_cls: Mock,
        mock_tantivy_cls: Mock,
        mock_create_fusion: Mock,
        mock_config: Mock,
        mock_logger: Mock,
    ) -> None:
        mock_config.embedding_cache_enabled = False
        mock_config.reranker_engine = "none"

        factory = EngineFactory()
        result = factory.create_engines(mock_config, mock_logger)

        # Verify base embedder used directly
        usearch_call = mock_usearch_cls.call_args
        assert usearch_call[1]["embedder"] is mock_embedder_cls.return_value

        assert result.tantivy_engine is mock_tantivy_cls.return_value
        assert result.reranker_engine == "none"
