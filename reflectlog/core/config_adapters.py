"""Configuration adapters for protocol-based dependency injection.

This module provides adapter classes that wrap application configuration
and expose specific protocol interfaces. This enables components to depend
on protocol abstractions (IServerConfig, ISearchConfig, etc.) rather than
concrete configuration classes, following the Dependency Inversion Principle.

Example:
    Instead of depending on Config directly:

        class MemoryManager:
            def __init__(self, config: Config):  # Tight coupling

    Components can depend on protocol interfaces:

        class MemoryManager:
            def __init__(self, config: IAppConfig):  # Protocol abstraction

    The adapter is created at the composition root:

        config = Config.from_environment()
        manager = MemoryManager(config=ConfigAdapter(config))

    Or use specific protocol adapters:

        class SearchService:
            def __init__(self, search_config: ISearchConfig):
                ...

        service = SearchService(search_config=SearchConfigAdapter(config))
"""

from typing import TYPE_CHECKING, assert_never, final

from reflectlog.core.config import (
    IAppConfig,
    IEmbedderConfig,
    IReplacementConfig,
    IRerankerConfig,
    ISearchConfig,
    IServerConfig,
    IStorageConfig,
)
from reflectlog.core.enums import (
    CrossEncoderDevice,
    DistanceMetric,
    EmbedderProvider,
    LlmProvider,
    RerankerEngine,
    TransportMode,
    WeMMDevice,
    parse_str_enum,
)

if TYPE_CHECKING:
    from reflectlog.application.config.settings import Config


def _validated_reranker_engine(reranker_engine: RerankerEngine) -> RerankerEngine:
    if reranker_engine == RerankerEngine.CROSS_ENCODER:
        return RerankerEngine.CROSS_ENCODER
    if reranker_engine == RerankerEngine.NONE:
        return RerankerEngine.NONE
    assert_never(reranker_engine)


def _coerce_reranker_engine(value: str) -> RerankerEngine:
    return _validated_reranker_engine(
        parse_str_enum(RerankerEngine, value, field="RERANKER_ENGINE")
    )


@final
class ConfigAdapter(IAppConfig):
    """Adapter that makes Config satisfy IAppConfig protocol.

    This adapter wraps the application Config dataclass and provides
    all properties required by the IAppConfig protocol. Since Python
    uses structural typing for runtime-checkable protocols, this class
    simply forwards property access to the wrapped Config instance.

    Args:
        config: The application Config instance to wrap.

    Example:
        config = Config.from_environment()
        adapter = ConfigAdapter(config)

        # Now adapter satisfies IAppConfig protocol
        assert isinstance(adapter, IAppConfig)  # True
    """

    def __init__(self, config: Config) -> None:
        """Initialize adapter with Config instance.

        Args:
            config: Application configuration instance.
        """
        self._config = config

    # IServerConfig properties
    @property
    def transport(self) -> TransportMode:
        """Transport mode for MCP server."""
        return self._config.transport

    @property
    def host(self) -> str:
        """Server host for network transports."""
        return self._config.host

    @property
    def port(self) -> int:
        """Server port for network transports."""
        return self._config.port

    @property
    def path(self) -> str:
        """Server path for network transports."""
        return self._config.path

    @property
    def log_level(self) -> str:
        """Logging level: DEBUG, INFO, WARNING, ERROR."""
        return self._config.log_level

    @property
    def workspace_id(self) -> str:
        """Unique workspace identifier."""
        return self._config.workspace_id

    # ISearchConfig properties
    @property
    def search_limit(self) -> int:
        """Maximum number of results to return."""
        return self._config.search_limit

    @property
    def enable_rrf_fusion(self) -> bool:
        """Enable Reciprocal Rank Fusion for result ranking."""
        return self._config.enable_rrf_fusion

    @property
    def fusion_rrf_k(self) -> int:
        """RRF constant k for fusion ranking."""
        return self._config.fusion_rrf_k

    @property
    def fusion_threshold(self) -> float:
        """Minimum fusion score to include result."""
        return self._config.fusion_ranking_threshold

    @property
    def reranker_engine(self) -> RerankerEngine:
        """Reranking engine type."""
        return _coerce_reranker_engine(self._config.reranker_engine)

    @property
    def search_score_threshold(self) -> float:
        """Minimum relevance score used by search filtering."""
        return self._config.search_score_threshold

    @property
    def enable_recency_boost(self) -> bool:
        """Include memory age in reranking."""
        return self._config.enable_recency_boost

    @property
    def recency_decay_rate(self) -> float:
        """Exponential decay rate per hour."""
        return self._config.recency_decay_rate

    # IStorageConfig properties
    @property
    def storage_path(self) -> str:
        """Base path for index storage."""
        return "indexes"

    @property
    def usearch_index_path(self) -> str:
        """Path to USearch index files."""
        return f"indexes/{self._config.workspace_id.lower()}/usearch"

    @property
    def tantivy_index_path(self) -> str:
        """Path to Tantivy index files."""
        return self._config.tantivy_index_path_template.format(
            workspace_id=self._config.workspace_id.lower()
        )

    @property
    def usearch_exact_search(self) -> bool:
        """Whether USearch should use exact search."""
        return self._config.usearch_exact_search

    @property
    def usearch_exact_search_threshold(self) -> int:
        """Index size below which USearch switches to exact search."""
        return self._config.usearch_exact_search_threshold

    @property
    def tantivy_normalize_scores(self) -> bool:
        """Whether Tantivy scores are normalized."""
        return self._config.tantivy_normalize_scores

    @property
    def tantivy_soft_delete_enabled(self) -> bool:
        """Whether Tantivy uses tombstones instead of rebuilds."""
        return self._config.tantivy_soft_delete_enabled

    @property
    def tantivy_compaction_threshold_ratio(self) -> float:
        """Tombstone ratio that triggers compaction."""
        return self._config.tantivy_compaction_threshold_ratio

    @property
    def tantivy_compaction_max_tombstones(self) -> int:
        """Tombstone count that forces compaction."""
        return self._config.tantivy_compaction_max_tombstones

    @property
    def tantivy_tombstone_ttl_days(self) -> int:
        """Days before tombstones are eligible for removal."""
        return self._config.tantivy_tombstone_ttl_days

    @property
    def embedding_dims(self) -> int:
        """Embedding vector dimensions."""
        return self._config.embedding_dims

    @property
    def metric(self) -> DistanceMetric:
        """Similarity metric for vector search."""
        return DistanceMetric.COSINE

    # IRerankerConfig properties
    @property
    def llm_model(self) -> str:
        """LLM model for smart replacement."""
        return self._config.llm_model

    @property
    def llm_api_base_url(self) -> str:
        """API base URL for LLM provider."""
        return self._config.openrouter_base_url

    @property
    def cross_encoder_model(self) -> str:
        """Cross-encoder model name."""
        return self._config.cross_encoder_model

    @property
    def cross_encoder_device(self) -> CrossEncoderDevice:
        """Device for cross-encoder: cpu, cuda, mps."""
        return self._config.cross_encoder_device

    @property
    def reranker_batch_normalize(self) -> bool:
        """Enable batch normalization for reranker scores."""
        return self._config.reranker_batch_normalize

    @property
    def llm_api_key(self) -> str:
        """API key for LLM provider (plain text)."""
        return self._config.openrouter_api_key.get_secret_value()

    @property
    def llm_provider(self) -> LlmProvider:
        """LLM provider: openai or anthropic."""
        return self._config.llm_provider

    @property
    def rerank_max_concurrency(self) -> int:
        """Maximum parallel LLM calls for smart replacement."""
        return self._config.rerank_max_concurrency

    @property
    def cross_encoder_top_k(self) -> int:
        """Number of top results to return after cross-encoder reranking."""
        return self._config.cross_encoder_top_k

    @property
    def cross_encoder_batch_size(self) -> int:
        """Batch size for cross-encoder inference."""
        return self._config.cross_encoder_batch_size

    @property
    def cross_encoder_score_threshold(self) -> float:
        """Minimum cross-encoder score to keep results."""
        return self._config.cross_encoder_score_threshold

    @property
    def cross_encoder_use_fp16(self) -> bool:
        """Enable FP16 for faster cross-encoder computation."""
        return self._config.cross_encoder_use_fp16

    @property
    def cross_encoder_normalize(self) -> bool:
        """Apply sigmoid to normalize cross-encoder scores."""
        return self._config.cross_encoder_normalize

    @property
    def cross_encoder_max_length(self) -> int:
        """Maximum token length for query-document pairs."""
        return self._config.cross_encoder_max_length

    @property
    def reranker_min_results(self) -> int:
        """Safety net: minimum results to return (0 = disabled)."""
        return self._config.reranker_min_results

    # IEmbedderConfig properties
    @property
    def embedding_model(self) -> str:
        """Embedding model name."""
        return self._config.embedding_model

    @property
    def embedder_provider(self) -> EmbedderProvider:
        """Embedding backend."""
        return self._config.embedder_provider

    @property
    def qwen_embedding_dims(self) -> int:
        """Qwen embedding dimensions."""
        return self._config.qwen_embedding_dims

    @property
    def wemm_embedding_dims(self) -> int:
        """Validated WeMM embedding dimensions."""
        return self._config.wemm_embedding_dims

    @property
    def wemm_device(self) -> WeMMDevice:
        """Device selection for local WeMM inference."""
        return self._config.wemm_device

    @property
    def embedding_batch_size(self) -> int:
        """Batch size for embedding operations."""
        return self._config.embedding_batch_size

    @property
    def embedding_max_concurrent_batches(self) -> int:
        """Maximum concurrent embedding batches."""
        return self._config.embedding_max_concurrent_batches

    @property
    def embedding_cache_enabled(self) -> bool:
        """Enable embedding cache."""
        return self._config.embedding_cache_enabled

    @property
    def embedding_cache_size(self) -> int:
        """Maximum cache entries."""
        return self._config.embedding_cache_size

    # IReplacementConfig properties
    @property
    def enable_smart_replace(self) -> bool:
        """Enable smart memory replacement detection."""
        return self._config.enable_smart_replace

    @property
    def smart_replace_threshold(self) -> float:
        """Minimum confidence for replacement detection."""
        return self._config.smart_replace_threshold

    @property
    def smart_replace_min_similarity(self) -> float:
        """Minimum embedding similarity for candidates."""
        return self._config.smart_replace_min_similarity

    @property
    def smart_replace_candidate_limit(self) -> int:
        """Maximum candidates to check."""
        return self._config.smart_replace_candidate_limit

    @property
    def smart_replace_max_retries(self) -> int:
        """Maximum retry attempts for smart replacement LLM calls."""
        return self._config.smart_replace_max_retries

    @property
    def smart_replace_retry_delay(self) -> float:
        """Base delay in seconds for exponential backoff."""
        return self._config.smart_replace_retry_delay


@final
class ServerConfigAdapter(IServerConfig):
    """Adapter exposing only server configuration protocol.

    Use this when a component only needs server-level settings.

    Args:
        config: The application Config instance to adapt.

    Example:
        config = Config.from_environment()
        server_adapter = ServerConfigAdapter(config)

        # Use where IServerConfig is expected
        start_server(transport=server_adapter.transport, port=server_adapter.port)
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def transport(self) -> TransportMode:
        return self._config.transport

    @property
    def host(self) -> str:
        return self._config.host

    @property
    def port(self) -> int:
        return self._config.port

    @property
    def path(self) -> str:
        return self._config.path

    @property
    def log_level(self) -> str:
        return self._config.log_level

    @property
    def workspace_id(self) -> str:
        return self._config.workspace_id


@final
class SearchConfigAdapter(ISearchConfig):
    """Adapter exposing only search configuration protocol.

    Use this when a component only needs search-related settings.

    Args:
        config: The application Config instance to adapt.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def search_limit(self) -> int:
        return self._config.search_limit

    @property
    def enable_rrf_fusion(self) -> bool:
        return self._config.enable_rrf_fusion

    @property
    def fusion_rrf_k(self) -> int:
        return self._config.fusion_rrf_k

    @property
    def fusion_threshold(self) -> float:
        return self._config.fusion_ranking_threshold

    @property
    def reranker_engine(self) -> RerankerEngine:
        return _coerce_reranker_engine(self._config.reranker_engine)

    @property
    def search_score_threshold(self) -> float:
        return self._config.search_score_threshold

    @property
    def enable_recency_boost(self) -> bool:
        return self._config.enable_recency_boost

    @property
    def recency_decay_rate(self) -> float:
        return self._config.recency_decay_rate


@final
class StorageConfigAdapter(IStorageConfig):
    """Adapter exposing only storage configuration protocol.

    Use this when a component only needs storage-related settings.

    Args:
        config: The application Config instance to adapt.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def storage_path(self) -> str:
        return "indexes"

    @property
    def usearch_index_path(self) -> str:
        return f"indexes/{self._config.workspace_id.lower()}/usearch"

    @property
    def tantivy_index_path(self) -> str:
        return self._config.tantivy_index_path_template.format(
            workspace_id=self._config.workspace_id.lower()
        )

    @property
    def embedding_dims(self) -> int:
        return self._config.embedding_dims

    @property
    def metric(self) -> DistanceMetric:
        return DistanceMetric.COSINE

    @property
    def usearch_exact_search(self) -> bool:
        return self._config.usearch_exact_search

    @property
    def usearch_exact_search_threshold(self) -> int:
        return self._config.usearch_exact_search_threshold

    @property
    def tantivy_normalize_scores(self) -> bool:
        return self._config.tantivy_normalize_scores

    @property
    def tantivy_soft_delete_enabled(self) -> bool:
        return self._config.tantivy_soft_delete_enabled

    @property
    def tantivy_compaction_threshold_ratio(self) -> float:
        return self._config.tantivy_compaction_threshold_ratio

    @property
    def tantivy_compaction_max_tombstones(self) -> int:
        return self._config.tantivy_compaction_max_tombstones

    @property
    def tantivy_tombstone_ttl_days(self) -> int:
        return self._config.tantivy_tombstone_ttl_days


@final
class RerankerConfigAdapter(IRerankerConfig):
    """Adapter exposing only reranker configuration protocol.

    Use this when a component only needs reranker settings.

    Args:
        config: The application Config instance to adapt.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def llm_model(self) -> str:
        return self._config.llm_model

    @property
    def llm_api_base_url(self) -> str:
        return self._config.openrouter_base_url

    @property
    def cross_encoder_model(self) -> str:
        return self._config.cross_encoder_model

    @property
    def cross_encoder_device(self) -> CrossEncoderDevice:
        return self._config.cross_encoder_device

    @property
    def reranker_batch_normalize(self) -> bool:
        return self._config.reranker_batch_normalize

    @property
    def llm_api_key(self) -> str:
        return self._config.openrouter_api_key.get_secret_value()

    @property
    def llm_provider(self) -> LlmProvider:
        return self._config.llm_provider

    @property
    def rerank_max_concurrency(self) -> int:
        return self._config.rerank_max_concurrency

    @property
    def cross_encoder_top_k(self) -> int:
        return self._config.cross_encoder_top_k

    @property
    def cross_encoder_batch_size(self) -> int:
        return self._config.cross_encoder_batch_size

    @property
    def cross_encoder_score_threshold(self) -> float:
        return self._config.cross_encoder_score_threshold

    @property
    def cross_encoder_use_fp16(self) -> bool:
        return self._config.cross_encoder_use_fp16

    @property
    def cross_encoder_normalize(self) -> bool:
        return self._config.cross_encoder_normalize

    @property
    def cross_encoder_max_length(self) -> int:
        return self._config.cross_encoder_max_length

    @property
    def reranker_min_results(self) -> int:
        return self._config.reranker_min_results


@final
class EmbedderConfigAdapter(IEmbedderConfig):
    """Adapter exposing only embedder configuration protocol.

    Use this when a component only needs embedding settings.

    Args:
        config: The application Config instance to adapt.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def embedding_model(self) -> str:
        return self._config.embedding_model

    @property
    def embedder_provider(self) -> EmbedderProvider:
        return self._config.embedder_provider

    @property
    def qwen_embedding_dims(self) -> int:
        return self._config.qwen_embedding_dims

    @property
    def wemm_embedding_dims(self) -> int:
        return self._config.wemm_embedding_dims

    @property
    def wemm_device(self) -> WeMMDevice:
        return self._config.wemm_device

    @property
    def embedding_batch_size(self) -> int:
        return self._config.embedding_batch_size

    @property
    def embedding_max_concurrent_batches(self) -> int:
        return self._config.embedding_max_concurrent_batches

    @property
    def embedding_cache_enabled(self) -> bool:
        return self._config.embedding_cache_enabled

    @property
    def embedding_cache_size(self) -> int:
        return self._config.embedding_cache_size


@final
class ReplacementConfigAdapter(IReplacementConfig):
    """Adapter exposing only replacement configuration protocol.

    Use this when a component only needs smart replacement settings.

    Args:
        config: The application Config instance to adapt.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def enable_smart_replace(self) -> bool:
        return self._config.enable_smart_replace

    @property
    def smart_replace_threshold(self) -> float:
        return self._config.smart_replace_threshold

    @property
    def smart_replace_min_similarity(self) -> float:
        return self._config.smart_replace_min_similarity

    @property
    def smart_replace_candidate_limit(self) -> int:
        return self._config.smart_replace_candidate_limit

    @property
    def smart_replace_max_retries(self) -> int:
        return self._config.smart_replace_max_retries

    @property
    def smart_replace_retry_delay(self) -> float:
        return self._config.smart_replace_retry_delay


def create_config_adapter(config: Config) -> ConfigAdapter:
    """Factory function to create a full ConfigAdapter.

    Args:
        config: Application configuration instance.

    Returns:
        ConfigAdapter wrapping the provided configuration.
    """
    return ConfigAdapter(config)


def create_server_config_adapter(
    config: Config,
) -> ServerConfigAdapter:
    """Factory function to create a ServerConfigAdapter.

    Args:
        config: Application configuration instance.

    Returns:
        ServerConfigAdapter wrapping the provided configuration.
    """
    return ServerConfigAdapter(config)


def create_search_config_adapter(
    config: Config,
) -> SearchConfigAdapter:
    """Factory function to create a SearchConfigAdapter.

    Args:
        config: Application configuration instance.

    Returns:
        SearchConfigAdapter wrapping the provided configuration.
    """
    return SearchConfigAdapter(config)


def create_storage_config_adapter(
    config: Config,
) -> StorageConfigAdapter:
    """Factory function to create a StorageConfigAdapter.

    Args:
        config: Application configuration instance.

    Returns:
        StorageConfigAdapter wrapping the provided configuration.
    """
    return StorageConfigAdapter(config)


def create_reranker_config_adapter(
    config: Config,
) -> RerankerConfigAdapter:
    """Factory function to create a RerankerConfigAdapter.

    Args:
        config: Application configuration instance.

    Returns:
        RerankerConfigAdapter wrapping the provided configuration.
    """
    return RerankerConfigAdapter(config)


def create_embedder_config_adapter(
    config: Config,
) -> EmbedderConfigAdapter:
    """Factory function to create an EmbedderConfigAdapter.

    Args:
        config: Application configuration instance.

    Returns:
        EmbedderConfigAdapter wrapping the provided configuration.
    """
    return EmbedderConfigAdapter(config)


def create_replacement_config_adapter(
    config: Config,
) -> ReplacementConfigAdapter:
    """Factory function to create a ReplacementConfigAdapter.

    Args:
        config: Application configuration instance.

    Returns:
        ReplacementConfigAdapter wrapping the provided configuration.
    """
    return ReplacementConfigAdapter(config)
