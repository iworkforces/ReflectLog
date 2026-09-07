"""Configuration protocols for ReflectLog.

This module defines protocols that abstract configuration sources from
configuration consumers. Components depend on these protocols rather than
concrete configuration classes, enabling:

- Configuration from environment variables, files, or remote services
- Testing with fixture configurations
- Runtime configuration updates without component modification

Example:
    def search_memories(
        query: str,
        config: ISearchConfig,
        store: IMemoryStore,
    ) -> list[str]:
        limit = config.search_limit
        ...
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from reflectlog.core.enums import (
        CrossEncoderDevice,
        DistanceMetric,
        LlmProvider,
        RerankerEngine,
        TransportMode,
    )


@runtime_checkable
class IServerConfig(Protocol):
    """Protocol for server-level configuration."""

    @property
    def transport(self) -> TransportMode:
        """Transport mode for MCP server."""
        ...

    @property
    def host(self) -> str:
        """Server host for network transports."""
        ...

    @property
    def port(self) -> int:
        """Server port for network transports."""
        ...

    @property
    def path(self) -> str:
        """Server path for network transports."""
        ...

    @property
    def log_level(self) -> str:
        """Logging level: DEBUG, INFO, WARNING, ERROR."""
        ...

    @property
    def workspace_id(self) -> str:
        """Unique workspace identifier."""
        ...


@runtime_checkable
class ISearchConfig(Protocol):
    """Protocol for search-related configuration."""

    @property
    def search_limit(self) -> int:
        """Maximum number of results to return."""
        ...

    @property
    def enable_rrf_fusion(self) -> bool:
        """Enable Reciprocal Rank Fusion for result ranking."""
        ...

    @property
    def fusion_rrf_k(self) -> int:
        """RRF constant k for fusion ranking."""
        ...

    @property
    def fusion_threshold(self) -> float:
        """Minimum fusion score to include result."""
        ...

    @property
    def reranker_engine(self) -> RerankerEngine:
        """Reranking engine type."""
        ...

    @property
    def search_score_threshold(self) -> float:
        """Minimum relevance score used by search filtering."""
        ...

    @property
    def enable_recency_boost(self) -> bool:
        """Include memory age in reranking."""
        ...

    @property
    def recency_decay_rate(self) -> float:
        """Exponential decay rate per hour."""
        ...


@runtime_checkable
class IStorageConfig(Protocol):
    """Protocol for storage-related configuration."""

    @property
    def storage_path(self) -> str:
        """Base path for index storage."""
        ...

    @property
    def usearch_index_path(self) -> str:
        """Path to USearch index files."""
        ...

    @property
    def tantivy_index_path(self) -> str:
        """Path to Tantivy index files."""
        ...

    @property
    def embedding_dims(self) -> int:
        """Embedding vector dimensions."""
        ...

    @property
    def metric(self) -> DistanceMetric:
        """Similarity metric for vector search."""
        ...

    @property
    def usearch_exact_search(self) -> bool:
        """Whether USearch should use exact search."""
        ...

    @property
    def usearch_exact_search_threshold(self) -> int:
        """Index size below which USearch switches to exact search."""
        ...

    @property
    def tantivy_normalize_scores(self) -> bool:
        """Whether Tantivy scores are normalized."""
        ...

    @property
    def tantivy_soft_delete_enabled(self) -> bool:
        """Whether Tantivy uses tombstones instead of rebuilds."""
        ...

    @property
    def tantivy_compaction_threshold_ratio(self) -> float:
        """Tombstone ratio that triggers compaction."""
        ...

    @property
    def tantivy_compaction_max_tombstones(self) -> int:
        """Tombstone count that forces compaction."""
        ...

    @property
    def tantivy_tombstone_ttl_days(self) -> int:
        """Days before tombstones are eligible for removal."""
        ...


@runtime_checkable
class IRerankerConfig(Protocol):
    """Protocol for reranker configuration."""

    @property
    def llm_model(self) -> str:
        """LLM model for smart replacement."""
        ...

    @property
    def llm_api_key(self) -> str:
        """API key for LLM provider (plain text, not SecretString)."""
        ...

    @property
    def llm_provider(self) -> LlmProvider:
        """LLM provider: openai or anthropic."""
        ...

    @property
    def rerank_max_concurrency(self) -> int:
        """Maximum parallel LLM calls for smart replacement."""
        ...

    @property
    def cross_encoder_top_k(self) -> int:
        """Number of top results to return after cross-encoder reranking."""
        ...

    @property
    def cross_encoder_batch_size(self) -> int:
        """Batch size for cross-encoder inference."""
        ...

    @property
    def cross_encoder_score_threshold(self) -> float:
        """Minimum cross-encoder score to keep results."""
        ...

    @property
    def cross_encoder_use_fp16(self) -> bool:
        """Enable FP16 for faster cross-encoder computation."""
        ...

    @property
    def cross_encoder_normalize(self) -> bool:
        """Apply sigmoid to normalize cross-encoder scores."""
        ...

    @property
    def cross_encoder_max_length(self) -> int:
        """Maximum token length for query-document pairs."""
        ...

    @property
    def reranker_min_results(self) -> int:
        """Safety net: minimum results to return (0 = disabled)."""
        ...

    @property
    def llm_api_base_url(self) -> str:
        """API base URL for LLM provider."""
        ...

    @property
    def cross_encoder_model(self) -> str:
        """Cross-encoder model name."""
        ...

    @property
    def cross_encoder_device(self) -> CrossEncoderDevice:
        """Device for cross-encoder: cpu, cuda, mps."""
        ...

    @property
    def reranker_batch_normalize(self) -> bool:
        """Enable batch normalization for reranker scores."""
        ...


@runtime_checkable
class IEmbedderConfig(Protocol):
    """Protocol for embedding provider configuration."""

    @property
    def embedding_model(self) -> str:
        """Embedding model name."""
        ...

    @property
    def embedder_provider(self) -> str:
        """Embedder provider: langchain or openai."""
        ...

    @property
    def qwen_embedding_dims(self) -> int:
        """Qwen embedding dimensions."""
        ...

    @property
    def embedding_batch_size(self) -> int:
        """Batch size for embedding operations."""
        ...

    @property
    def embedding_max_concurrent_batches(self) -> int:
        """Maximum concurrent embedding batches."""
        ...

    @property
    def embedding_cache_enabled(self) -> bool:
        """Enable embedding cache."""
        ...

    @property
    def embedding_cache_size(self) -> int:
        """Maximum cache entries."""
        ...


@runtime_checkable
class IReplacementConfig(Protocol):
    """Protocol for smart replacement configuration."""

    @property
    def enable_smart_replace(self) -> bool:
        """Enable smart memory replacement detection."""
        ...

    @property
    def smart_replace_threshold(self) -> float:
        """Minimum confidence for replacement detection."""
        ...

    @property
    def smart_replace_min_similarity(self) -> float:
        """Minimum embedding similarity for candidates."""
        ...

    @property
    def smart_replace_candidate_limit(self) -> int:
        """Maximum candidates to check."""
        ...

    @property
    def smart_replace_max_retries(self) -> int:
        """Maximum retry attempts for smart replacement LLM calls."""
        ...

    @property
    def smart_replace_retry_delay(self) -> float:
        """Base delay in seconds for exponential backoff."""
        ...


@runtime_checkable
class IAppConfig(
    IServerConfig,
    ISearchConfig,
    IStorageConfig,
    IRerankerConfig,
    IEmbedderConfig,
    IReplacementConfig,
    Protocol,
):
    """Combined protocol for all application configuration.

    Components can depend on this protocol to access all configuration
    without needing to inject multiple configuration objects.
    """
