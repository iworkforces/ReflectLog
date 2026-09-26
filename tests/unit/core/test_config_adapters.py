"""Tests for reflectlog.core.config_adapters module.

Verifies protocol conformance for all adapter classes that bridge
the monolithic Config dataclass to fine-grained protocol interfaces.
"""

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.utils.security import SecretString
from reflectlog.core.config import (
    IAppConfig,
    IEmbedderConfig,
    IReplacementConfig,
    IRerankerConfig,
    ISearchConfig,
    IServerConfig,
    IStorageConfig,
)
from reflectlog.core.config_adapters import (
    ConfigAdapter,
    EmbedderConfigAdapter,
    ReplacementConfigAdapter,
    RerankerConfigAdapter,
    SearchConfigAdapter,
    ServerConfigAdapter,
    StorageConfigAdapter,
    _coerce_reranker_engine,
    create_config_adapter,
    create_embedder_config_adapter,
    create_replacement_config_adapter,
    create_reranker_config_adapter,
    create_search_config_adapter,
    create_server_config_adapter,
    create_storage_config_adapter,
)
from reflectlog.core.enums import (
    CrossEncoderDevice,
    EmbedderProvider,
    RerankerEngine,
    TransportMode,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config() -> Config:
    """Config with only required fields, all others at defaults."""
    return Config(
        workspace_id="test-project",
        openrouter_api_key=SecretString("sk-test-key-12345"),
    )


@pytest.fixture
def custom_config() -> Config:
    """Config with all adapter-relevant fields set to non-default values."""
    return Config(
        workspace_id="custom-proj",
        openrouter_api_key=SecretString("sk-custom-key-99999"),
        transport=TransportMode.HTTP,
        host="0.0.0.0",
        port=8080,
        path="/custom-mcp",
        log_level="DEBUG",
        search_limit=20,
        enable_rrf_fusion=False,
        fusion_rrf_k=30,
        fusion_ranking_threshold=0.5,
        reranker_engine=RerankerEngine.CROSS_ENCODER,
        search_score_threshold=0.8,
        enable_recency_boost=False,
        recency_decay_rate=0.05,
        embedding_dims=1536,
        llm_model="openai/gpt-4o",
        openrouter_base_url="https://api.example.com/v1",
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_device=CrossEncoderDevice.CUDA,
        reranker_batch_normalize=False,
        embedding_model="openai/text-embedding-3-small",
        embedder_provider=EmbedderProvider.LANGCHAIN,
        qwen_embedding_dims=2048,
        embedding_batch_size=256,
        embedding_max_concurrent_batches=8,
        embedding_cache_enabled=False,
        embedding_cache_size=50,
        enable_smart_replace=False,
        smart_replace_threshold=0.5,
        smart_replace_min_similarity=0.8,
        smart_replace_candidate_limit=5,
    )


# ---------------------------------------------------------------------------
# Helper: _coerce_reranker_engine (includes exhaustiveness validation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRerankerEngineHelpers:
    """Tests for _coerce_reranker_engine."""

    @pytest.mark.parametrize(
        "engine",
        [RerankerEngine.CROSS_ENCODER, RerankerEngine.NONE, "cross_encoder", "none"],
    )
    def test_coerce_reranker_engine_valid(self, engine: str) -> None:
        """Valid string and enum values coerce to RerankerEngine."""
        result = _coerce_reranker_engine(engine)
        assert result == engine

    def test_coerce_reranker_engine_unknown_fails_closed(self) -> None:
        """Unknown reranker engine strings raise ConfigurationError."""
        from reflectlog.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Invalid RERANKER_ENGINE"):
            _ = _coerce_reranker_engine("llm")
        with pytest.raises(ConfigurationError, match="Invalid RERANKER_ENGINE"):
            _ = _coerce_reranker_engine("unknown")


# ---------------------------------------------------------------------------
# ConfigAdapter (IAppConfig — all protocols combined)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigAdapter:
    """Tests for ConfigAdapter satisfying IAppConfig (all sub-protocols)."""

    def test_isinstance_iappconfig(self, minimal_config: Config) -> None:
        """ConfigAdapter satisfies the IAppConfig protocol."""
        adapter = ConfigAdapter(minimal_config)
        assert isinstance(adapter, IAppConfig)

    def test_isinstance_all_sub_protocols(self, minimal_config: Config) -> None:
        """ConfigAdapter satisfies every individual sub-protocol."""
        adapter = ConfigAdapter(minimal_config)
        assert isinstance(adapter, IServerConfig)
        assert isinstance(adapter, ISearchConfig)
        assert isinstance(adapter, IStorageConfig)
        assert isinstance(adapter, IRerankerConfig)
        assert isinstance(adapter, IEmbedderConfig)
        assert isinstance(adapter, IReplacementConfig)

    # -- IServerConfig properties --

    def test_transport_default(self, minimal_config: Config) -> None:
        """Default transport is 'stdio'."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.transport == "stdio"

    def test_host_default(self, minimal_config: Config) -> None:
        """Default host is 127.0.0.1."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.host == "127.0.0.1"

    def test_port_default(self, minimal_config: Config) -> None:
        """Default port is 9103."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.port == 9103

    def test_path_default(self, minimal_config: Config) -> None:
        """Default path is /mcp."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.path == "/mcp"

    def test_log_level_default(self, minimal_config: Config) -> None:
        """Default log level is INFO."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.log_level == "INFO"

    def test_workspace_id(self, minimal_config: Config) -> None:
        """workspace_id delegates to Config."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.workspace_id == "test-project"

    # -- ISearchConfig properties --

    def test_search_limit_default(self, minimal_config: Config) -> None:
        """Default search_limit is 5."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.search_limit == 5

    def test_enable_rrf_fusion_default(self, minimal_config: Config) -> None:
        """Default enable_rrf_fusion is True."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.enable_rrf_fusion is True

    def test_fusion_rrf_k_default(self, minimal_config: Config) -> None:
        """Default fusion_rrf_k is 60."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.fusion_rrf_k == 60

    def test_fusion_threshold_delegates_to_fusion_ranking_threshold(
        self, minimal_config: Config
    ) -> None:
        """fusion_threshold maps to Config.fusion_ranking_threshold."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.fusion_threshold == 0.0

    def test_reranker_engine_default(self, minimal_config: Config) -> None:
        """Default reranker_engine is 'cross_encoder'."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.reranker_engine == "cross_encoder"

    def test_reranker_engine_coercion(self, custom_config: Config) -> None:
        """reranker_engine coerces through _coerce_reranker_engine."""
        adapter = ConfigAdapter(custom_config)
        assert adapter.reranker_engine == "cross_encoder"

    def test_search_score_threshold_default(self, minimal_config: Config) -> None:
        """Default search_score_threshold is 0.5."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.search_score_threshold == 0.5

    def test_enable_recency_boost_default(self, minimal_config: Config) -> None:
        """Default enable_recency_boost is True."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.enable_recency_boost is True

    def test_recency_decay_rate_default(self, minimal_config: Config) -> None:
        """Default recency_decay_rate is 0.001."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.recency_decay_rate == 0.001

    # -- IStorageConfig properties --

    def test_storage_path_hardcoded(self, minimal_config: Config) -> None:
        """storage_path is always 'indexes'."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.storage_path == "indexes"

    def test_usearch_index_path(self, minimal_config: Config) -> None:
        """usearch_index_path includes workspace_id."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.usearch_index_path == "indexes/test-project/usearch"

    def test_tantivy_index_path(self, minimal_config: Config) -> None:
        """tantivy_index_path includes workspace_id."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.tantivy_index_path == "indexes/test-project/tantivy"

    def test_embedding_dims_default(self, minimal_config: Config) -> None:
        """Default embedding_dims is 2048."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.embedding_dims == 2048

    def test_metric_hardcoded(self, minimal_config: Config) -> None:
        """metric is always 'cosine'."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.metric == "cosine"

    # -- IRerankerConfig properties --

    def test_llm_model_default(self, minimal_config: Config) -> None:
        """Default llm_model is 'x-ai/grok-4.1-fast'."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.llm_model == "x-ai/grok-4.1-fast"

    def test_llm_api_base_url_delegates_to_openrouter_base_url(
        self, minimal_config: Config
    ) -> None:
        """llm_api_base_url maps to Config.openrouter_base_url."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.llm_api_base_url == "https://openrouter.ai/api/v1"

    def test_cross_encoder_model_default(self, minimal_config: Config) -> None:
        """Default cross_encoder_model."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.cross_encoder_model == "BAAI/bge-reranker-v2-m3"

    def test_cross_encoder_device_default(self, minimal_config: Config) -> None:
        """Default cross_encoder_device is 'cpu'."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.cross_encoder_device == "cpu"

    def test_reranker_batch_normalize_default(self, minimal_config: Config) -> None:
        """Default reranker_batch_normalize is True."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.reranker_batch_normalize is True

    # -- IEmbedderConfig properties --

    def test_embedding_model_default(self, minimal_config: Config) -> None:
        """Default embedding_model."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.embedding_model == "tencent/WeMM-Embedding-2B"

    def test_embedder_provider_default(self, minimal_config: Config) -> None:
        """Default embedder_provider is 'wemm'."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.embedder_provider == "wemm"

    def test_qwen_embedding_dims_default(self, minimal_config: Config) -> None:
        """Default qwen_embedding_dims is 2048."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.qwen_embedding_dims == 2048

    def test_embedding_batch_size_default(self, minimal_config: Config) -> None:
        """Default embedding_batch_size is 512."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.embedding_batch_size == 512

    def test_embedding_max_concurrent_batches_default(
        self, minimal_config: Config
    ) -> None:
        """Default embedding_max_concurrent_batches is 4."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.embedding_max_concurrent_batches == 4

    def test_embedding_cache_enabled_default(self, minimal_config: Config) -> None:
        """Default embedding_cache_enabled is True."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.embedding_cache_enabled is True

    def test_embedding_cache_size_default(self, minimal_config: Config) -> None:
        """Default embedding_cache_size is 100."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.embedding_cache_size == 100

    # -- IReplacementConfig properties --

    def test_enable_smart_replace_default(self, minimal_config: Config) -> None:
        """Default enable_smart_replace is True."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.enable_smart_replace is True

    def test_smart_replace_threshold_default(self, minimal_config: Config) -> None:
        """Default smart_replace_threshold is 0.7."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.smart_replace_threshold == 0.7

    def test_smart_replace_min_similarity_default(self, minimal_config: Config) -> None:
        """Default smart_replace_min_similarity is 0.9."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.smart_replace_min_similarity == 0.9

    def test_smart_replace_candidate_limit_default(
        self, minimal_config: Config
    ) -> None:
        """Default smart_replace_candidate_limit is 3."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter.smart_replace_candidate_limit == 3

    # -- Custom config delegation --

    def test_all_properties_with_custom_config(self, custom_config: Config) -> None:
        """All properties delegate correctly with non-default Config values."""
        adapter = ConfigAdapter(custom_config)

        # IServerConfig
        assert adapter.transport == "http"
        assert adapter.host == "0.0.0.0"
        assert adapter.port == 8080
        assert adapter.path == "/custom-mcp"
        assert adapter.log_level == "DEBUG"
        assert adapter.workspace_id == "custom-proj"

        # ISearchConfig
        assert adapter.search_limit == 20
        assert adapter.enable_rrf_fusion is False
        assert adapter.fusion_rrf_k == 30
        assert adapter.fusion_threshold == 0.5
        assert adapter.reranker_engine == "cross_encoder"
        assert adapter.search_score_threshold == 0.8
        assert adapter.enable_recency_boost is False
        assert adapter.recency_decay_rate == 0.05

        # IStorageConfig
        assert adapter.storage_path == "indexes"
        assert adapter.usearch_index_path == "indexes/custom-proj/usearch"
        assert adapter.tantivy_index_path == "indexes/custom-proj/tantivy"
        assert adapter.embedding_dims == 1536
        assert adapter.metric == "cosine"

        # IRerankerConfig
        assert adapter.llm_model == "openai/gpt-4o"
        assert adapter.llm_api_base_url == "https://api.example.com/v1"
        assert adapter.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert adapter.cross_encoder_device == "cuda"
        assert adapter.reranker_batch_normalize is False

        # IEmbedderConfig
        assert adapter.embedding_model == "openai/text-embedding-3-small"
        assert adapter.embedder_provider == "langchain"
        assert adapter.qwen_embedding_dims == 2048
        assert adapter.embedding_batch_size == 256
        assert adapter.embedding_max_concurrent_batches == 8
        assert adapter.embedding_cache_enabled is False
        assert adapter.embedding_cache_size == 50

        # IReplacementConfig
        assert adapter.enable_smart_replace is False
        assert adapter.smart_replace_threshold == 0.5
        assert adapter.smart_replace_min_similarity == 0.8
        assert adapter.smart_replace_candidate_limit == 5


# ---------------------------------------------------------------------------
# ServerConfigAdapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServerConfigAdapter:
    """Tests for ServerConfigAdapter satisfying IServerConfig."""

    def test_isinstance_iserver_config(self, minimal_config: Config) -> None:
        """ServerConfigAdapter satisfies IServerConfig protocol."""
        adapter = ServerConfigAdapter(minimal_config)
        assert isinstance(adapter, IServerConfig)

    def test_does_not_satisfy_other_protocols(self, minimal_config: Config) -> None:
        """ServerConfigAdapter does NOT satisfy unrelated protocols."""
        adapter = ServerConfigAdapter(minimal_config)
        assert not isinstance(adapter, ISearchConfig)
        assert not isinstance(adapter, IStorageConfig)
        assert not isinstance(adapter, IRerankerConfig)
        assert not isinstance(adapter, IEmbedderConfig)
        assert not isinstance(adapter, IReplacementConfig)

    def test_transport(self, custom_config: Config) -> None:
        """transport delegates to Config.transport."""
        adapter = ServerConfigAdapter(custom_config)
        assert adapter.transport == "http"

    def test_host(self, custom_config: Config) -> None:
        """host delegates to Config.host."""
        adapter = ServerConfigAdapter(custom_config)
        assert adapter.host == "0.0.0.0"

    def test_port(self, custom_config: Config) -> None:
        """port delegates to Config.port."""
        adapter = ServerConfigAdapter(custom_config)
        assert adapter.port == 8080

    def test_path(self, custom_config: Config) -> None:
        """path delegates to Config.path."""
        adapter = ServerConfigAdapter(custom_config)
        assert adapter.path == "/custom-mcp"

    def test_log_level(self, custom_config: Config) -> None:
        """log_level delegates to Config.log_level."""
        adapter = ServerConfigAdapter(custom_config)
        assert adapter.log_level == "DEBUG"

    def test_workspace_id(self, custom_config: Config) -> None:
        """workspace_id delegates to Config.workspace_id."""
        adapter = ServerConfigAdapter(custom_config)
        assert adapter.workspace_id == "custom-proj"

    def test_defaults(self, minimal_config: Config) -> None:
        """All defaults match Config defaults."""
        adapter = ServerConfigAdapter(minimal_config)
        assert adapter.transport == "stdio"
        assert adapter.host == "127.0.0.1"
        assert adapter.port == 9103
        assert adapter.path == "/mcp"
        assert adapter.log_level == "INFO"
        assert adapter.workspace_id == "test-project"


# ---------------------------------------------------------------------------
# SearchConfigAdapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchConfigAdapter:
    """Tests for SearchConfigAdapter satisfying ISearchConfig."""

    def test_isinstance_isearch_config(self, minimal_config: Config) -> None:
        """SearchConfigAdapter satisfies ISearchConfig protocol."""
        adapter = SearchConfigAdapter(minimal_config)
        assert isinstance(adapter, ISearchConfig)

    def test_does_not_satisfy_other_protocols(self, minimal_config: Config) -> None:
        """SearchConfigAdapter does NOT satisfy unrelated protocols."""
        adapter = SearchConfigAdapter(minimal_config)
        assert not isinstance(adapter, IServerConfig)
        assert not isinstance(adapter, IStorageConfig)
        assert not isinstance(adapter, IRerankerConfig)
        assert not isinstance(adapter, IEmbedderConfig)
        assert not isinstance(adapter, IReplacementConfig)

    def test_search_limit(self, custom_config: Config) -> None:
        """search_limit delegates to Config."""
        adapter = SearchConfigAdapter(custom_config)
        assert adapter.search_limit == 20

    def test_enable_rrf_fusion(self, custom_config: Config) -> None:
        """enable_rrf_fusion delegates to Config."""
        adapter = SearchConfigAdapter(custom_config)
        assert adapter.enable_rrf_fusion is False

    def test_fusion_rrf_k(self, custom_config: Config) -> None:
        """fusion_rrf_k delegates to Config."""
        adapter = SearchConfigAdapter(custom_config)
        assert adapter.fusion_rrf_k == 30

    def test_fusion_threshold(self, custom_config: Config) -> None:
        """fusion_threshold maps to Config.fusion_ranking_threshold."""
        adapter = SearchConfigAdapter(custom_config)
        assert adapter.fusion_threshold == 0.5

    def test_reranker_engine(self, custom_config: Config) -> None:
        """reranker_engine coerces through _coerce_reranker_engine."""
        adapter = SearchConfigAdapter(custom_config)
        assert adapter.reranker_engine == "cross_encoder"

    def test_reranker_engine_coercion_invalid(self) -> None:
        """Invalid reranker_engine in Config fails closed."""
        from reflectlog.core.exceptions import ConfigurationError

        config = Config(
            workspace_id="test",
            openrouter_api_key=SecretString("sk-key"),
        )
        object.__setattr__(config, "reranker_engine", "invalid_engine")
        adapter = SearchConfigAdapter(config)
        with pytest.raises(ConfigurationError, match="Invalid RERANKER_ENGINE"):
            _ = adapter.reranker_engine

    def test_search_score_threshold(self, custom_config: Config) -> None:
        """search_score_threshold delegates to Config."""
        adapter = SearchConfigAdapter(custom_config)
        assert adapter.search_score_threshold == 0.8

    def test_enable_recency_boost(self, custom_config: Config) -> None:
        """enable_recency_boost delegates to Config."""
        adapter = SearchConfigAdapter(custom_config)
        assert adapter.enable_recency_boost is False

    def test_recency_decay_rate(self, custom_config: Config) -> None:
        """recency_decay_rate delegates to Config."""
        adapter = SearchConfigAdapter(custom_config)
        assert adapter.recency_decay_rate == 0.05

    def test_defaults(self, minimal_config: Config) -> None:
        """All defaults match Config defaults."""
        adapter = SearchConfigAdapter(minimal_config)
        assert adapter.search_limit == 5
        assert adapter.enable_rrf_fusion is True
        assert adapter.fusion_rrf_k == 60
        assert adapter.fusion_threshold == 0.0
        assert adapter.reranker_engine == "cross_encoder"
        assert adapter.search_score_threshold == 0.5
        assert adapter.enable_recency_boost is True
        assert adapter.recency_decay_rate == 0.001


# ---------------------------------------------------------------------------
# StorageConfigAdapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStorageConfigAdapter:
    """Tests for StorageConfigAdapter satisfying IStorageConfig."""

    def test_isinstance_istorage_config(self, minimal_config: Config) -> None:
        """StorageConfigAdapter satisfies IStorageConfig protocol."""
        adapter = StorageConfigAdapter(minimal_config)
        assert isinstance(adapter, IStorageConfig)

    def test_does_not_satisfy_other_protocols(self, minimal_config: Config) -> None:
        """StorageConfigAdapter does NOT satisfy unrelated protocols."""
        adapter = StorageConfigAdapter(minimal_config)
        assert not isinstance(adapter, IServerConfig)
        assert not isinstance(adapter, ISearchConfig)
        assert not isinstance(adapter, IRerankerConfig)
        assert not isinstance(adapter, IEmbedderConfig)
        assert not isinstance(adapter, IReplacementConfig)

    def test_storage_path_hardcoded(self, minimal_config: Config) -> None:
        """storage_path is always 'indexes'."""
        adapter = StorageConfigAdapter(minimal_config)
        assert adapter.storage_path == "indexes"

    def test_usearch_index_path(self, custom_config: Config) -> None:
        """usearch_index_path includes workspace_id from Config."""
        adapter = StorageConfigAdapter(custom_config)
        assert adapter.usearch_index_path == "indexes/custom-proj/usearch"

    def test_tantivy_index_path(self, custom_config: Config) -> None:
        """tantivy_index_path includes workspace_id from Config."""
        adapter = StorageConfigAdapter(custom_config)
        assert adapter.tantivy_index_path == "indexes/custom-proj/tantivy"

    def test_embedding_dims(self, custom_config: Config) -> None:
        """embedding_dims delegates to Config."""
        adapter = StorageConfigAdapter(custom_config)
        assert adapter.embedding_dims == 1536

    def test_metric_hardcoded(self, minimal_config: Config) -> None:
        """metric is always 'cosine'."""
        adapter = StorageConfigAdapter(minimal_config)
        assert adapter.metric == "cosine"

    def test_defaults(self, minimal_config: Config) -> None:
        """All defaults match expected values."""
        adapter = StorageConfigAdapter(minimal_config)
        assert adapter.storage_path == "indexes"
        assert adapter.usearch_index_path == "indexes/test-project/usearch"
        assert adapter.tantivy_index_path == "indexes/test-project/tantivy"
        assert adapter.embedding_dims == 2048
        assert adapter.metric == "cosine"
        assert adapter.usearch_exact_search is False
        assert adapter.usearch_exact_search_threshold == 256

    def test_usearch_exact_search_forwards_non_defaults(
        self, custom_config: Config
    ) -> None:
        """Exact-search knobs must not be dropped by the adapter."""
        from dataclasses import replace

        from reflectlog.infrastructure.usearch_engine import USearchConfig

        tuned = replace(
            custom_config,
            usearch_exact_search=True,
            usearch_exact_search_threshold=256,
        )
        adapter = ConfigAdapter(tuned)
        assert adapter.usearch_exact_search is True
        built = USearchConfig.from_config(adapter)
        assert built.exact_search is True
        assert built.exact_search_threshold == 256


# ---------------------------------------------------------------------------
# RerankerConfigAdapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRerankerConfigAdapter:
    """Tests for RerankerConfigAdapter satisfying IRerankerConfig."""

    def test_isinstance_ireranker_config(self, minimal_config: Config) -> None:
        """RerankerConfigAdapter satisfies IRerankerConfig protocol."""
        adapter = RerankerConfigAdapter(minimal_config)
        assert isinstance(adapter, IRerankerConfig)

    def test_does_not_satisfy_other_protocols(self, minimal_config: Config) -> None:
        """RerankerConfigAdapter does NOT satisfy unrelated protocols."""
        adapter = RerankerConfigAdapter(minimal_config)
        assert not isinstance(adapter, IServerConfig)
        assert not isinstance(adapter, ISearchConfig)
        assert not isinstance(adapter, IStorageConfig)
        assert not isinstance(adapter, IEmbedderConfig)
        assert not isinstance(adapter, IReplacementConfig)

    def test_llm_model(self, custom_config: Config) -> None:
        """llm_model delegates to Config."""
        adapter = RerankerConfigAdapter(custom_config)
        assert adapter.llm_model == "openai/gpt-4o"

    def test_llm_api_base_url(self, custom_config: Config) -> None:
        """llm_api_base_url maps to Config.openrouter_base_url."""
        adapter = RerankerConfigAdapter(custom_config)
        assert adapter.llm_api_base_url == "https://api.example.com/v1"

    def test_cross_encoder_model(self, custom_config: Config) -> None:
        """cross_encoder_model delegates to Config."""
        adapter = RerankerConfigAdapter(custom_config)
        assert adapter.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_cross_encoder_device(self, custom_config: Config) -> None:
        """cross_encoder_device delegates to Config."""
        adapter = RerankerConfigAdapter(custom_config)
        assert adapter.cross_encoder_device == "cuda"

    def test_reranker_batch_normalize(self, custom_config: Config) -> None:
        """reranker_batch_normalize delegates to Config."""
        adapter = RerankerConfigAdapter(custom_config)
        assert adapter.reranker_batch_normalize is False

    def test_defaults(self, minimal_config: Config) -> None:
        """All defaults match Config defaults."""
        adapter = RerankerConfigAdapter(minimal_config)
        assert adapter.llm_model == "x-ai/grok-4.1-fast"
        assert adapter.llm_api_base_url == "https://openrouter.ai/api/v1"
        assert adapter.cross_encoder_model == "BAAI/bge-reranker-v2-m3"
        assert adapter.cross_encoder_device == "cpu"
        assert adapter.reranker_batch_normalize is True


# ---------------------------------------------------------------------------
# EmbedderConfigAdapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedderConfigAdapter:
    """Tests for EmbedderConfigAdapter satisfying IEmbedderConfig."""

    def test_isinstance_iembedder_config(self, minimal_config: Config) -> None:
        """EmbedderConfigAdapter satisfies IEmbedderConfig protocol."""
        adapter = EmbedderConfigAdapter(minimal_config)
        assert isinstance(adapter, IEmbedderConfig)

    def test_does_not_satisfy_other_protocols(self, minimal_config: Config) -> None:
        """EmbedderConfigAdapter does NOT satisfy unrelated protocols."""
        adapter = EmbedderConfigAdapter(minimal_config)
        assert not isinstance(adapter, IServerConfig)
        assert not isinstance(adapter, ISearchConfig)
        assert not isinstance(adapter, IStorageConfig)
        assert not isinstance(adapter, IRerankerConfig)
        assert not isinstance(adapter, IReplacementConfig)

    def test_embedding_model(self, custom_config: Config) -> None:
        """embedding_model delegates to Config."""
        adapter = EmbedderConfigAdapter(custom_config)
        assert adapter.embedding_model == "openai/text-embedding-3-small"

    def test_embedder_provider(self, custom_config: Config) -> None:
        """embedder_provider delegates to Config."""
        adapter = EmbedderConfigAdapter(custom_config)
        assert adapter.embedder_provider == "langchain"

    def test_qwen_embedding_dims(self, custom_config: Config) -> None:
        """qwen_embedding_dims delegates to Config."""
        adapter = EmbedderConfigAdapter(custom_config)
        assert adapter.qwen_embedding_dims == 2048

    def test_embedding_batch_size(self, custom_config: Config) -> None:
        """embedding_batch_size delegates to Config."""
        adapter = EmbedderConfigAdapter(custom_config)
        assert adapter.embedding_batch_size == 256

    def test_embedding_max_concurrent_batches(self, custom_config: Config) -> None:
        """embedding_max_concurrent_batches delegates to Config."""
        adapter = EmbedderConfigAdapter(custom_config)
        assert adapter.embedding_max_concurrent_batches == 8

    def test_embedding_cache_enabled(self, custom_config: Config) -> None:
        """embedding_cache_enabled delegates to Config."""
        adapter = EmbedderConfigAdapter(custom_config)
        assert adapter.embedding_cache_enabled is False

    def test_embedding_cache_size(self, custom_config: Config) -> None:
        """embedding_cache_size delegates to Config."""
        adapter = EmbedderConfigAdapter(custom_config)
        assert adapter.embedding_cache_size == 50

    def test_defaults(self, minimal_config: Config) -> None:
        """All defaults match Config defaults."""
        adapter = EmbedderConfigAdapter(minimal_config)
        assert adapter.embedding_model == "tencent/WeMM-Embedding-2B"
        assert adapter.embedder_provider == "wemm"
        assert adapter.qwen_embedding_dims == 2048
        assert adapter.embedding_batch_size == 512
        assert adapter.embedding_max_concurrent_batches == 4
        assert adapter.embedding_cache_enabled is True
        assert adapter.embedding_cache_size == 100


# ---------------------------------------------------------------------------
# ReplacementConfigAdapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReplacementConfigAdapter:
    """Tests for ReplacementConfigAdapter satisfying IReplacementConfig."""

    def test_isinstance_ireplacement_config(self, minimal_config: Config) -> None:
        """ReplacementConfigAdapter satisfies IReplacementConfig protocol."""
        adapter = ReplacementConfigAdapter(minimal_config)
        assert isinstance(adapter, IReplacementConfig)

    def test_does_not_satisfy_other_protocols(self, minimal_config: Config) -> None:
        """ReplacementConfigAdapter does NOT satisfy unrelated protocols."""
        adapter = ReplacementConfigAdapter(minimal_config)
        assert not isinstance(adapter, IServerConfig)
        assert not isinstance(adapter, ISearchConfig)
        assert not isinstance(adapter, IStorageConfig)
        assert not isinstance(adapter, IRerankerConfig)
        assert not isinstance(adapter, IEmbedderConfig)

    def test_enable_smart_replace(self, custom_config: Config) -> None:
        """enable_smart_replace delegates to Config."""
        adapter = ReplacementConfigAdapter(custom_config)
        assert adapter.enable_smart_replace is False

    def test_smart_replace_threshold(self, custom_config: Config) -> None:
        """smart_replace_threshold delegates to Config."""
        adapter = ReplacementConfigAdapter(custom_config)
        assert adapter.smart_replace_threshold == 0.5

    def test_smart_replace_min_similarity(self, custom_config: Config) -> None:
        """smart_replace_min_similarity delegates to Config."""
        adapter = ReplacementConfigAdapter(custom_config)
        assert adapter.smart_replace_min_similarity == 0.8

    def test_smart_replace_candidate_limit(self, custom_config: Config) -> None:
        """smart_replace_candidate_limit delegates to Config."""
        adapter = ReplacementConfigAdapter(custom_config)
        assert adapter.smart_replace_candidate_limit == 5

    def test_defaults(self, minimal_config: Config) -> None:
        """All defaults match Config defaults."""
        adapter = ReplacementConfigAdapter(minimal_config)
        assert adapter.enable_smart_replace is True
        assert adapter.smart_replace_threshold == 0.7
        assert adapter.smart_replace_min_similarity == 0.9
        assert adapter.smart_replace_candidate_limit == 3


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFactoryFunctions:
    """Tests for create_*_adapter factory functions."""

    def test_create_config_adapter(self, minimal_config: Config) -> None:
        """create_config_adapter returns a ConfigAdapter."""
        adapter = create_config_adapter(minimal_config)
        assert isinstance(adapter, ConfigAdapter)
        assert isinstance(adapter, IAppConfig)

    def test_create_server_config_adapter(self, minimal_config: Config) -> None:
        """create_server_config_adapter returns a ServerConfigAdapter."""
        adapter = create_server_config_adapter(minimal_config)
        assert isinstance(adapter, ServerConfigAdapter)
        assert isinstance(adapter, IServerConfig)

    def test_create_search_config_adapter(self, minimal_config: Config) -> None:
        """create_search_config_adapter returns a SearchConfigAdapter."""
        adapter = create_search_config_adapter(minimal_config)
        assert isinstance(adapter, SearchConfigAdapter)
        assert isinstance(adapter, ISearchConfig)

    def test_create_storage_config_adapter(self, minimal_config: Config) -> None:
        """create_storage_config_adapter returns a StorageConfigAdapter."""
        adapter = create_storage_config_adapter(minimal_config)
        assert isinstance(adapter, StorageConfigAdapter)
        assert isinstance(adapter, IStorageConfig)

    def test_create_reranker_config_adapter(self, minimal_config: Config) -> None:
        """create_reranker_config_adapter returns a RerankerConfigAdapter."""
        adapter = create_reranker_config_adapter(minimal_config)
        assert isinstance(adapter, RerankerConfigAdapter)
        assert isinstance(adapter, IRerankerConfig)

    def test_create_embedder_config_adapter(self, minimal_config: Config) -> None:
        """create_embedder_config_adapter returns an EmbedderConfigAdapter."""
        adapter = create_embedder_config_adapter(minimal_config)
        assert isinstance(adapter, EmbedderConfigAdapter)
        assert isinstance(adapter, IEmbedderConfig)

    def test_create_replacement_config_adapter(self, minimal_config: Config) -> None:
        """create_replacement_config_adapter returns a ReplacementConfigAdapter."""
        adapter = create_replacement_config_adapter(minimal_config)
        assert isinstance(adapter, ReplacementConfigAdapter)
        assert isinstance(adapter, IReplacementConfig)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    """Edge case tests for config adapters."""

    def test_storage_paths_with_special_workspace_id(self) -> None:
        """Storage paths handle workspace_id with dots and hyphens."""
        config = Config(
            workspace_id="my-project.v2",
            openrouter_api_key=SecretString("sk-key"),
        )
        adapter = ConfigAdapter(config)
        assert adapter.usearch_index_path == "indexes/my-project.v2/usearch"
        assert adapter.tantivy_index_path == "indexes/my-project.v2/tantivy"

    def test_storage_paths_lowercase_workspace_component(self) -> None:
        """Index paths lowercase the workspace component; identity stays mixed-case."""
        config = Config(
            workspace_id="MyProject",
            openrouter_api_key=SecretString("sk-key"),
        )
        adapter = ConfigAdapter(config)
        storage = StorageConfigAdapter(config)
        assert config.workspace_id == "MyProject"
        assert adapter.usearch_index_path == "indexes/myproject/usearch"
        assert adapter.tantivy_index_path == "indexes/myproject/tantivy"
        assert storage.usearch_index_path == "indexes/myproject/usearch"
        assert storage.tantivy_index_path == "indexes/myproject/tantivy"

    def test_reranker_engine_none_string(self) -> None:
        """Config with reranker_engine='none' coerces correctly."""
        config = Config(
            workspace_id="test",
            openrouter_api_key=SecretString("sk-key"),
            reranker_engine=RerankerEngine.NONE,
        )
        adapter = ConfigAdapter(config)
        assert adapter.reranker_engine == "none"

    def test_all_transport_modes(self) -> None:
        """All transport literal values are supported."""
        for transport in TransportMode:
            config = Config(
                workspace_id="test",
                openrouter_api_key=SecretString("sk-key"),
                transport=transport,
            )
            adapter = ConfigAdapter(config)
            assert adapter.transport == transport

    def test_same_config_shared_by_multiple_adapters(
        self, minimal_config: Config
    ) -> None:
        """Multiple adapter types can wrap the same Config instance."""
        server = ServerConfigAdapter(minimal_config)
        search = SearchConfigAdapter(minimal_config)
        storage = StorageConfigAdapter(minimal_config)
        reranker = RerankerConfigAdapter(minimal_config)
        embedder = EmbedderConfigAdapter(minimal_config)
        replacement = ReplacementConfigAdapter(minimal_config)
        full = ConfigAdapter(minimal_config)

        # All share the same workspace_id source
        assert server.workspace_id == "test-project"
        assert full.workspace_id == "test-project"

        # Verify they each satisfy only their own protocol
        assert isinstance(server, IServerConfig)
        assert isinstance(search, ISearchConfig)
        assert isinstance(storage, IStorageConfig)
        assert isinstance(reranker, IRerankerConfig)
        assert isinstance(embedder, IEmbedderConfig)
        assert isinstance(replacement, IReplacementConfig)
        assert isinstance(full, IAppConfig)

    def test_adapter_wraps_config_not_copies(self, minimal_config: Config) -> None:
        """Adapter references the Config instance, not a copy."""
        adapter = ConfigAdapter(minimal_config)
        assert adapter._config is minimal_config

    def test_zero_fusion_threshold(self) -> None:
        """Config with zero fusion_ranking_threshold works."""
        config = Config(
            workspace_id="test",
            openrouter_api_key=SecretString("sk-key"),
            fusion_ranking_threshold=0.0,
        )
        adapter = ConfigAdapter(config)
        assert adapter.fusion_threshold == 0.0

    def test_extreme_search_limit(self) -> None:
        """Config with high search_limit delegates correctly."""
        config = Config(
            workspace_id="test",
            openrouter_api_key=SecretString("sk-key"),
            search_limit=1000,
        )
        adapter = SearchConfigAdapter(config)
        assert adapter.search_limit == 1000

    def test_metric_always_cosine_regardless_of_config(self) -> None:
        """metric is hardcoded to 'cosine', not from Config."""
        config = Config(
            workspace_id="test",
            openrouter_api_key=SecretString("sk-key"),
        )
        # StorageConfigAdapter hardcodes "cosine"
        adapter = StorageConfigAdapter(config)
        assert adapter.metric == "cosine"
        # ConfigAdapter also hardcodes "cosine"
        full = ConfigAdapter(config)
        assert full.metric == "cosine"

    def test_storage_path_always_indexes(self) -> None:
        """storage_path is hardcoded to 'indexes', not from Config."""
        config = Config(
            workspace_id="any-id",
            openrouter_api_key=SecretString("sk-key"),
        )
        adapter = StorageConfigAdapter(config)
        assert adapter.storage_path == "indexes"
        full = ConfigAdapter(config)
        assert full.storage_path == "indexes"
