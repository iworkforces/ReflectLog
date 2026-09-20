"""Configuration management for ReflectLog Server."""

from dataclasses import dataclass
import math
import os
import re
import threading
from typing import TypedDict

from reflectlog.core.enums import (
    CrossEncoderDevice,
    EmbedderProvider,
    FusionMethod,
    FusionNormalization,
    LlmProvider,
    RerankerEngine,
    TransportMode,
    WeMMDevice,
    WeMMModel,
    parse_str_enum,
)
from reflectlog.core.exceptions import ConfigurationError

from ..utils.security import SecretString
from .presets import apply_preset_to_env, get_active_preset

# Note: LangchainQwenEmbeddings is imported lazily in MemoryManager
# to avoid unnecessary initialization when not using langchain provider


def _env_field_value(name: str, default: str) -> str:
    raw = os.environ.get(name, default)
    if raw.strip() == "":
        raise ConfigurationError(f"Invalid {name}: value is empty")
    return raw


def _parse_env_int(
    name: str,
    default: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = _env_field_value(name, default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid {name}: expected an integer") from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"Invalid {name}: below the allowed range")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"Invalid {name}: above the allowed range")
    return value


def _parse_env_float(
    name: str,
    default: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = _env_field_value(name, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid {name}: expected a number") from exc
    if not math.isfinite(value):
        raise ConfigurationError(f"Invalid {name}: expected a finite number")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"Invalid {name}: below the allowed range")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"Invalid {name}: above the allowed range")
    return value


def _parse_port() -> int:
    if "PORT" in os.environ:
        return _parse_env_int("PORT", "9103", minimum=1, maximum=65535)
    return _parse_env_int("MCP_PORT", "9103", minimum=1, maximum=65535)


def _parse_fusion_weights() -> list[float] | None:
    raw = os.environ.get("FUSION_WEIGHTS", "")
    if raw.strip() == "":
        return None
    weights: list[float] = []
    for token in raw.split(","):
        piece = token.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError as exc:
            raise ConfigurationError(
                "Invalid FUSION_WEIGHTS: expected a number"
            ) from exc
        if not math.isfinite(value):
            raise ConfigurationError("Invalid FUSION_WEIGHTS: expected a finite number")
        weights.append(value)
    return weights or None


class TransportConfigDict(TypedDict):
    transport: TransportMode
    port: int
    host: str
    path: str
    openrouter_base_url: str


class EmbeddingConfigDict(TypedDict):
    embedder_provider: EmbedderProvider
    embedding_model: str
    embedding_dims: int
    qwen_embedding_dims: int
    wemm_embedding_dims: int
    wemm_device: WeMMDevice
    embedding_batch_size: int
    embedding_max_concurrent_batches: int
    embedding_cache_enabled: bool
    embedding_cache_size: int


class SearchConfigDict(TypedDict):
    search_limit: int
    remove_search_limit: int
    tantivy_index_path_template: str
    overfetch_multiplier: int
    overfetch_adaptive: bool
    overfetch_min_multiplier: float
    overfetch_max_multiplier: float
    usearch_exact_search: bool
    usearch_exact_search_threshold: int
    fusion_method: FusionMethod
    fusion_normalization: FusionNormalization | None
    fusion_rrf_k: int
    fusion_ranking_threshold: float
    enable_rrf_fusion: bool
    fusion_weights: list[float] | None


class RerankerConfigDict(TypedDict):
    reranker_engine: RerankerEngine
    llm_provider: LlmProvider
    llm_model: str
    search_score_threshold: float
    rerank_max_concurrency: int
    cross_encoder_model: str
    cross_encoder_top_k: int
    cross_encoder_device: CrossEncoderDevice
    cross_encoder_batch_size: int
    cross_encoder_score_threshold: float
    cross_encoder_use_fp16: bool
    cross_encoder_normalize: bool
    cross_encoder_max_length: int
    reranker_min_results: int
    reranker_batch_normalize: bool
    enable_recency_boost: bool
    recency_decay_rate: float


class StorageConfigDict(TypedDict):
    tantivy_soft_delete_enabled: bool
    tantivy_compaction_threshold_ratio: float
    tantivy_compaction_max_tombstones: int
    tantivy_tombstone_ttl_days: int
    tantivy_normalize_scores: bool


class SecurityConfigDict(TypedDict):
    max_memory_length: int
    min_memory_length: int
    deduplicate_memories: bool
    max_add_batch: int
    max_add_chars: int
    get_all_limit: int


class LoggingConfigDict(TypedDict):
    log_level: str
    log_search_results_verbose: bool
    log_search_result_limit: int


class MetricsConfigDict(TypedDict):
    enable_smart_replace: bool
    smart_replace_threshold: float
    smart_replace_min_similarity: float
    smart_replace_candidate_limit: int
    smart_replace_archive_ttl_days: int
    smart_replace_max_retries: int
    smart_replace_retry_delay: float


class ServerConfigDict(TypedDict):
    add_max_concurrency: int
    eager_initialization: bool
    eager_initialize_search_engines: bool | None
    eager_initialize_reranker: bool | None
    eager_initialize_smart_replacer: bool | None


def _parse_optional_bool(value: str | None) -> bool | None:
    """Parse an optional boolean environment variable.

    Returns:
        True if value is "true" or "1", False if value is "false" or "0",
        None if value is None or empty string.

    Examples:
        _parse_optional_bool("true") -> True
        _parse_optional_bool("false") -> False
        _parse_optional_bool(None) -> None
        _parse_optional_bool("") -> None
    """
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    return None


@dataclass(frozen=True)
class Config:
    """Configuration settings for ReflectLog Server.

    All settings are read from environment variables with sensible defaults.
    This class is immutable (frozen) to prevent accidental modifications at runtime.
    """

    # Core settings (required)
    workspace_id: str
    openrouter_api_key: SecretString  # Wrapped for security - use .get_secret_value()

    # Transport settings
    transport: TransportMode = TransportMode.STDIO
    port: int = 9103
    host: str = "127.0.0.1"
    path: str = "/mcp"

    # API settings
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Embedding settings
    embedder_provider: EmbedderProvider = EmbedderProvider.OPENAI
    embedding_model: str = "openai/text-embedding-3-large"
    embedding_dims: int = 3072
    qwen_embedding_dims: int = 4096
    wemm_embedding_dims: int = 2048
    wemm_device: WeMMDevice = WeMMDevice.AUTO

    # Embedding performance settings
    embedding_batch_size: int = 512  # Texts per API request for async batching
    embedding_max_concurrent_batches: int = 4  # Max parallel batch requests
    embedding_cache_enabled: bool = True  # Cache query embeddings (LRU)
    embedding_cache_size: int = 100  # Max cached embeddings

    # Search settings
    search_limit: int = 5
    remove_search_limit: int = 5

    # Hybrid search settings
    tantivy_index_path_template: str = "indexes/{workspace_id}/tantivy"
    overfetch_multiplier: int = 3  # Base multiplier (Fetch N * search_limit)
    overfetch_adaptive: bool = True  # Enable adaptive overfetch based on index size
    overfetch_min_multiplier: float = 1.5  # Min multiplier (for large indexes)
    overfetch_max_multiplier: float = 3.0  # Max multiplier (for small indexes)

    # Tantivy soft-delete settings (O(1) delete vs O(n) rebuild)
    tantivy_soft_delete_enabled: bool = True  # Use tombstone marking instead of rebuild
    tantivy_compaction_threshold_ratio: float = (
        0.2  # Compact when tombstones > 20% of docs
    )
    tantivy_compaction_max_tombstones: int = 10000  # Force compaction above this count
    tantivy_tombstone_ttl_days: int = 7  # Days before tombstones eligible for removal

    # Tantivy BM25 score normalization (batch min-max to 0-1 range)
    tantivy_normalize_scores: bool = (
        True  # Normalize BM25 scores for threshold filtering
    )

    # USearch exact search settings
    usearch_exact_search: bool = False  # HNSW approximate search (exact is opt-in)
    usearch_exact_search_threshold: int = (
        256  # Auto-switch to exact when index is smaller than this
    )

    # Fusion settings (ranx-based)
    fusion_method: FusionMethod = FusionMethod.RRF
    fusion_normalization: FusionNormalization | None = None
    fusion_rrf_k: int = 60  # RRF k parameter (lower = more weight to top ranks)
    fusion_ranking_threshold: float = 0.0  # Raw RRF cutoff; 0.0 keeps all fused hits
    enable_rrf_fusion: bool = True  # Enable RRF fusion (false = concatenate results)
    fusion_weights: list[float] | None = None  # Per-run weights for weighted RRF

    # Memory validation settings
    max_memory_length: int = 30720
    min_memory_length: int = 1
    max_add_batch: int = 100
    max_add_chars: int = 500_000
    get_all_limit: int = 1000

    # Reranker engine selection
    reranker_engine: RerankerEngine = RerankerEngine.CROSS_ENCODER

    # LLM settings used by smart replacement (not search reranking)
    llm_model: str = "x-ai/grok-4.1-fast"
    search_score_threshold: float = 0.5
    rerank_max_concurrency: int = 10

    # Cross-encoder reranking settings (used when reranker_engine="cross_encoder")
    # Uses FlagEmbedding's FlagReranker for BGE reranker models
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"  # HuggingFace model
    cross_encoder_top_k: int = 20  # Max results to return after reranking
    cross_encoder_device: CrossEncoderDevice = CrossEncoderDevice.CPU
    cross_encoder_batch_size: int = 32  # Batch size for inference
    cross_encoder_score_threshold: float = 0.5
    cross_encoder_use_fp16: bool = True  # Enable FP16 for faster inference
    cross_encoder_normalize: bool = True  # Normalize scores to 0-1 with sigmoid
    cross_encoder_max_length: int = 512  # Max token length for query-doc pairs

    # Unified reranker settings
    reranker_min_results: int = 1  # Safety net: keep at least the best CE hit
    reranker_batch_normalize: bool = True  # Enable batch min-max normalization

    # Temporal-aware reranking settings (recency boost for memories)
    enable_recency_boost: bool = True  # Include memory age in reranking context
    recency_decay_rate: float = 0.001  # ~29-day half-life; 0.01 is opt-in

    # Memory behavior
    deduplicate_memories: bool = True

    # Smart memory replacement settings
    enable_smart_replace: bool = True  # Enable smart memory replacement detection
    smart_replace_threshold: float = 0.7  # Min LLM confidence to trigger replacement
    smart_replace_min_similarity: float = (
        0.9  # Min embedding similarity to trigger LLM check
    )
    smart_replace_candidate_limit: int = 3  # Max candidates to check for replacement
    smart_replace_archive_ttl_days: int = (
        30  # Days to keep archived memories (0 = permanent)
    )
    smart_replace_max_retries: int = 3  # Max LLM call retries
    smart_replace_retry_delay: float = (
        1.0  # Base delay (seconds) for exponential backoff
    )
    llm_provider: LlmProvider = LlmProvider.ANTHROPIC

    # Concurrency settings
    add_max_concurrency: int = 4  # Max concurrent memory additions

    # Initialization settings
    eager_initialization: bool = True  # Pre-warm engines during MemoryManager init
    enable_llm_infer: bool = False  # Enable LLM memory inference

    # Granular eager initialization settings (for fine-grained control)
    # These override eager_initialization when set
    eager_initialize_search_engines: bool | None = (
        None  # Pre-warm USearch/Tantivy (default: true if eager_initialization)
    )
    eager_initialize_reranker: bool | None = (
        None  # Pre-load reranker (default: false - lazy load on first search)
    )
    eager_initialize_smart_replacer: bool | None = (
        None  # Pre-load SmartReplacer (default: false - lazy load on first add)
    )

    # Logging settings
    log_level: str = "INFO"
    log_search_results_verbose: bool = False  # Log individual search results
    log_search_result_limit: int = 3  # Max results to log when verbose enabled

    # Tool registration settings
    allowed_tools: tuple[str, ...] | None = None

    # Static helper methods for parsing configuration sections
    # These methods improve testability and maintainability by extracting
    # the large from_environment method into focused, single-responsibility functions.

    @staticmethod
    def _parse_transport_config() -> TransportConfigDict:
        """Parse transport-related configuration from environment variables."""
        transport = parse_str_enum(
            TransportMode,
            os.environ.get("MCP_TRANSPORT", TransportMode.STDIO),
            field="MCP_TRANSPORT",
        )

        return {
            "transport": transport,
            "port": _parse_port(),
            "host": os.environ.get("MCP_HOST", "127.0.0.1"),
            "path": os.environ.get("MCP_PATH", "/mcp"),
            "openrouter_base_url": os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
        }

    @staticmethod
    def _parse_embedding_config() -> EmbeddingConfigDict:
        """Parse embedding-related configuration from environment variables."""
        provider = parse_str_enum(
            EmbedderProvider,
            os.environ.get("EMBEDDER_PROVIDER", EmbedderProvider.OPENAI),
            field="EMBEDDER_PROVIDER",
        )
        embedding_model = os.environ.get(
            "EMBEDDING_MODEL", "openai/text-embedding-3-large"
        )
        wemm_dimensions = 2048
        if provider is EmbedderProvider.WEMM:
            model = WeMMModel.from_config(embedding_model)
            raw_dimensions = os.environ.get("WEMM_EMBEDDING_DIMS")
            override = (
                _parse_env_int("WEMM_EMBEDDING_DIMS", raw_dimensions, minimum=1)
                if raw_dimensions is not None
                else None
            )
            wemm_dimensions = model.resolve_dimensions(override)
            embedding_model = model.value
        return {
            "embedder_provider": provider,
            "embedding_model": embedding_model,
            "embedding_dims": _parse_env_int("EMBEDDING_DIMS", "3072", minimum=1),
            "qwen_embedding_dims": _parse_env_int(
                "QWEN_EMBEDDING_DIMS", "4096", minimum=1
            ),
            "wemm_embedding_dims": wemm_dimensions,
            "wemm_device": parse_str_enum(
                WeMMDevice,
                os.environ.get("WEMM_DEVICE", WeMMDevice.AUTO),
                field="WEMM_DEVICE",
            ),
            "embedding_batch_size": _parse_env_int(
                "EMBEDDING_BATCH_SIZE", "512", minimum=1
            ),
            "embedding_max_concurrent_batches": _parse_env_int(
                "EMBEDDING_MAX_CONCURRENT_BATCHES", "4", minimum=1
            ),
            "embedding_cache_enabled": os.environ.get(
                "EMBEDDING_CACHE_ENABLED", "true"
            ).lower()
            == "true",
            "embedding_cache_size": _parse_env_int(
                "EMBEDDING_CACHE_SIZE", "100", minimum=1
            ),
        }

    @staticmethod
    def _parse_search_config() -> SearchConfigDict:
        """Parse search-related configuration from environment variables."""
        return {
            "search_limit": _parse_env_int("SEARCH_LIMIT", "5", minimum=1),
            "remove_search_limit": _parse_env_int(
                "REMOVE_SEARCH_LIMIT", "5", minimum=1
            ),
            "tantivy_index_path_template": "indexes/{workspace_id}/tantivy",
            "overfetch_multiplier": _parse_env_int(
                "OVERFETCH_MULTIPLIER", "3", minimum=1
            ),
            "overfetch_adaptive": os.environ.get("OVERFETCH_ADAPTIVE", "true").lower()
            == "true",
            "overfetch_min_multiplier": _parse_env_float(
                "OVERFETCH_MIN_MULTIPLIER", "1.5", minimum=1.0
            ),
            "overfetch_max_multiplier": _parse_env_float(
                "OVERFETCH_MAX_MULTIPLIER", "3.0", minimum=1.0
            ),
            "usearch_exact_search": os.environ.get(
                "USEARCH_EXACT_SEARCH", "false"
            ).lower()
            == "true",
            "usearch_exact_search_threshold": _parse_env_int(
                "USEARCH_EXACT_SEARCH_THRESHOLD", "256", minimum=0
            ),
            "fusion_method": parse_str_enum(
                FusionMethod,
                os.environ.get("FUSION_METHOD", FusionMethod.RRF),
                field="FUSION_METHOD",
            ),
            "fusion_normalization": (
                parse_str_enum(
                    FusionNormalization,
                    fusion_normalization_raw,
                    field="FUSION_NORMALIZATION",
                )
                if (fusion_normalization_raw := os.environ.get("FUSION_NORMALIZATION"))
                else None
            ),
            "fusion_rrf_k": _parse_env_int("FUSION_RRF_K", "60", minimum=1),
            "fusion_ranking_threshold": _parse_env_float(
                "FUSION_RANKING_THRESHOLD", "0.0"
            ),
            "enable_rrf_fusion": os.environ.get("ENABLE_RRF_FUSION", "true").lower()
            == "true",
            "fusion_weights": _parse_fusion_weights(),
        }

    @staticmethod
    def _parse_tantivy_config() -> StorageConfigDict:
        """Parse Tantivy-specific configuration from environment variables."""
        return {
            "tantivy_soft_delete_enabled": os.environ.get(
                "TANTIVY_SOFT_DELETE_ENABLED", "true"
            ).lower()
            == "true",
            "tantivy_compaction_threshold_ratio": _parse_env_float(
                "TANTIVY_COMPACTION_THRESHOLD_RATIO",
                "0.2",
                minimum=0.01,
                maximum=1.0,
            ),
            "tantivy_compaction_max_tombstones": _parse_env_int(
                "TANTIVY_COMPACTION_MAX_TOMBSTONES", "10000", minimum=100
            ),
            "tantivy_tombstone_ttl_days": _parse_env_int(
                "TANTIVY_TOMBSTONE_TTL_DAYS", "7", minimum=0
            ),
            "tantivy_normalize_scores": os.environ.get(
                "TANTIVY_NORMALIZE_SCORES", "true"
            ).lower()
            == "true",
        }

    @staticmethod
    def _parse_reranker_config() -> RerankerConfigDict:
        """Parse reranker-related configuration from environment variables."""
        reranker_engine = parse_str_enum(
            RerankerEngine,
            os.environ.get("RERANKER_ENGINE", RerankerEngine.CROSS_ENCODER),
            field="RERANKER_ENGINE",
        )
        llm_provider = parse_str_enum(
            LlmProvider,
            os.environ.get("LLM_PROVIDER", LlmProvider.ANTHROPIC),
            field="LLM_PROVIDER",
        )

        return {
            "reranker_engine": reranker_engine,
            "llm_provider": llm_provider,
            "llm_model": os.environ.get("LLM_MODEL", "x-ai/grok-4.1-fast"),
            "search_score_threshold": _parse_env_float("SEARCH_SCORE_THRESHOLD", "0.5"),
            "rerank_max_concurrency": _parse_env_int(
                "RERANK_MAX_CONCURRENCY", "10", minimum=1
            ),
            "cross_encoder_model": os.environ.get(
                "CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"
            ),
            "cross_encoder_top_k": _parse_env_int(
                "CROSS_ENCODER_TOP_K", "20", minimum=1
            ),
            "cross_encoder_device": parse_str_enum(
                CrossEncoderDevice,
                os.environ.get("CROSS_ENCODER_DEVICE", CrossEncoderDevice.CPU),
                field="CROSS_ENCODER_DEVICE",
            ),
            "cross_encoder_batch_size": _parse_env_int(
                "CROSS_ENCODER_BATCH_SIZE", "32", minimum=1
            ),
            "cross_encoder_score_threshold": _parse_env_float(
                "CROSS_ENCODER_SCORE_THRESHOLD", "0.5"
            ),
            "cross_encoder_use_fp16": os.environ.get(
                "CROSS_ENCODER_USE_FP16", "true"
            ).lower()
            == "true",
            "cross_encoder_normalize": os.environ.get(
                "CROSS_ENCODER_NORMALIZE", "true"
            ).lower()
            == "true",
            "cross_encoder_max_length": _parse_env_int(
                "CROSS_ENCODER_MAX_LENGTH", "512", minimum=1
            ),
            "reranker_min_results": _parse_env_int(
                "RERANKER_MIN_RESULTS", "1", minimum=0
            ),
            "reranker_batch_normalize": os.environ.get(
                "RERANKER_BATCH_NORMALIZE", "true"
            ).lower()
            == "true",
            "enable_recency_boost": os.environ.get(
                "ENABLE_RECENCY_BOOST", "true"
            ).lower()
            == "true",
            "recency_decay_rate": _parse_env_float(
                "RECENCY_DECAY_RATE", "0.001", minimum=0.0
            ),
        }

    @staticmethod
    def _parse_memory_config() -> SecurityConfigDict:
        """Parse memory-related configuration from environment variables."""
        return {
            "max_memory_length": _parse_env_int(
                "MAX_MEMORY_LENGTH", "30720", minimum=1
            ),
            "min_memory_length": _parse_env_int("MIN_MEMORY_LENGTH", "1", minimum=1),
            "deduplicate_memories": os.environ.get(
                "DEDUPLICATE_MEMORIES", "true"
            ).lower()
            == "true",
            "max_add_batch": _parse_env_int("MAX_ADD_BATCH", "100", minimum=1),
            "max_add_chars": _parse_env_int("MAX_ADD_CHARS", "500000", minimum=1),
            "get_all_limit": _parse_env_int("GET_ALL_LIMIT", "1000", minimum=1),
        }

    @staticmethod
    def _parse_smart_replace_config() -> MetricsConfigDict:
        """Parse smart replacement configuration from environment variables."""
        return {
            "enable_smart_replace": os.environ.get(
                "ENABLE_SMART_REPLACE", "true"
            ).lower()
            == "true",
            "smart_replace_threshold": _parse_env_float(
                "SMART_REPLACE_THRESHOLD", "0.7"
            ),
            "smart_replace_min_similarity": _parse_env_float(
                "SMART_REPLACE_MIN_SIMILARITY", "0.9"
            ),
            "smart_replace_candidate_limit": _parse_env_int(
                "SMART_REPLACE_CANDIDATE_LIMIT", "3", minimum=1
            ),
            "smart_replace_archive_ttl_days": _parse_env_int(
                "SMART_REPLACE_ARCHIVE_TTL_DAYS", "30", minimum=0
            ),
            "smart_replace_max_retries": _parse_env_int(
                "SMART_REPLACE_MAX_RETRIES", "3", minimum=1
            ),
            "smart_replace_retry_delay": _parse_env_float(
                "SMART_REPLACE_RETRY_DELAY", "1.0", minimum=0.1
            ),
        }

    @staticmethod
    def _parse_init_config() -> ServerConfigDict:
        """Parse initialization-related configuration from environment variables."""
        return {
            "add_max_concurrency": _parse_env_int(
                "ADD_MAX_CONCURRENCY", "4", minimum=1
            ),
            "eager_initialization": os.environ.get(
                "EAGER_INITIALIZATION", "true"
            ).lower()
            == "true",
            "eager_initialize_search_engines": _parse_optional_bool(
                os.environ.get("EAGER_INITIALIZE_SEARCH_ENGINES")
            ),
            "eager_initialize_reranker": _parse_optional_bool(
                os.environ.get("EAGER_INITIALIZE_RERANKER")
            ),
            "eager_initialize_smart_replacer": _parse_optional_bool(
                os.environ.get("EAGER_INITIALIZE_SMART_REPLACER")
            ),
        }

    @staticmethod
    def _parse_logging_config() -> LoggingConfigDict:
        """Parse logging-related configuration from environment variables."""
        return {
            "log_level": os.environ.get("LOG_LEVEL", "INFO"),
            "log_search_results_verbose": os.environ.get(
                "LOG_SEARCH_RESULTS_VERBOSE", "false"
            ).lower()
            == "true",
            "log_search_result_limit": _parse_env_int(
                "LOG_SEARCH_RESULT_LIMIT", "3", minimum=0
            ),
        }

    @staticmethod
    def _parse_allowed_tools() -> tuple[str, ...] | None:
        """Parse allowed tools configuration from environment variables."""

        def _normalize_tool_token(raw_token: str) -> str:
            """Normalize tool identifiers to snake_case."""
            token = raw_token.strip()
            if not token:
                return ""

            token = token.replace("-", "_")
            token = re.sub(r"\s+", "_", token)
            if token.isupper():
                token = token.lower()
            else:
                token = re.sub(r"(?<!^)(?=[A-Z])", "_", token).lower()
            token = re.sub(r"_+", "_", token)
            token = token.strip("_")
            return token

        allowed_tools_env = os.environ.get("ALLOWED_TOOLS")
        allowed_tools: tuple[str, ...] | None = None
        if allowed_tools_env is not None:
            entries = [
                entry.strip() for entry in allowed_tools_env.split(",") if entry.strip()
            ]

            normalized: list[str] = []
            allow_all = False

            for entry in entries:
                token = _normalize_tool_token(entry)
                if token in {"all", "*"}:
                    allow_all = True
                    break
                normalized.append(token)

            if allow_all:
                allowed_tools = None
            else:
                if normalized and all(
                    token in {"none", "disabled"} for token in normalized
                ):
                    allowed_tools = ()
                else:
                    # Preserve order while removing duplicates
                    seen: dict[str, None] = {}
                    for token in normalized:
                        if token not in seen:
                            seen[token] = None
                    allowed_tools = tuple(seen.keys())

        # If ALLOWED_TOOLS was set to an empty/blank value, respect it
        if allowed_tools_env is not None and allowed_tools_env.strip() == "":
            allowed_tools = ()

        return allowed_tools

    @classmethod
    def from_environment(cls) -> Config:
        """Create configuration from environment variables.

        This method centralizes environment parsing and performs strict validation
        for all critical settings. It fails fast with clear, non-secret-leaking
        errors when configuration is invalid.

        Raises:
            ConfigurationError: If required environment variables are missing or invalid.
        """
        # Validate required environment variables
        workspace_id = os.environ.get("WORKSPACE_ID")
        if not workspace_id:
            raise ConfigurationError(
                "The WORKSPACE_ID environment variable has not been configured."
            )

        # Enforce a safe WORKSPACE_ID format to avoid path traversal and
        # filesystem issues. Keep the rule intentionally strict.
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", workspace_id):
            raise ConfigurationError(
                "Invalid WORKSPACE_ID: only A-Za-z0-9_.- allowed, max length 64."
            )

        # Check for path traversal patterns (not caught by regex)
        if ".." in workspace_id or workspace_id.startswith("/"):
            raise ConfigurationError(
                f"Invalid WORKSPACE_ID: path traversal patterns not allowed: {workspace_id}"
            )

        preset = get_active_preset()
        if preset:
            apply_preset_to_env(preset)

        openrouter_api_key_raw = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key_raw:
            raise ConfigurationError(
                "The OPENROUTER_API_KEY environment variable has not been configured."
            )
        # Wrap in SecretString to prevent accidental logging
        openrouter_api_key = SecretString(openrouter_api_key_raw)

        # Parse configuration sections using helper methods
        transport_config = cls._parse_transport_config()
        embedding_config = cls._parse_embedding_config()
        search_config = cls._parse_search_config()
        tantivy_config = cls._parse_tantivy_config()
        reranker_config = cls._parse_reranker_config()
        memory_config = cls._parse_memory_config()
        smart_replace_config = cls._parse_smart_replace_config()
        init_config = cls._parse_init_config()
        logging_config = cls._parse_logging_config()
        allowed_tools = cls._parse_allowed_tools()

        # Create Config instance with all parsed settings
        config = cls(
            # Required settings
            workspace_id=workspace_id,
            openrouter_api_key=openrouter_api_key,
            # Parsed configuration sections
            **transport_config,
            **embedding_config,
            **search_config,
            **tantivy_config,
            **reranker_config,
            **memory_config,
            **smart_replace_config,
            **init_config,
            **logging_config,
            allowed_tools=allowed_tools,
        )

        from reflectlog.application.config.validation import validate_config

        errors = validate_config(config)
        if errors:
            raise ConfigurationError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {error.field}: {error.message}" for error in errors)
            )

        return config


# Thread-safe lazy singleton pattern - config is only created when first accessed
_config: Config | None = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """Get the configuration singleton (lazy initialization).

    This delays environment variable parsing until first access,
    avoiding startup overhead for imports that don't need config.

    Thread-safe: Uses double-checked locking pattern to ensure only
    one Config instance is created even under concurrent access.

    Returns:
        The Config singleton instance.
    """
    global _config

    # Fast path: already initialized (use local var for type narrowing)
    local_config = _config
    if local_config is not None:
        return local_config

    # Slow path: need to initialize with lock
    with _config_lock:
        # Double-check inside lock to prevent race condition
        local_config = _config
        if local_config is not None:
            return local_config

        # Create and cache the singleton
        new_config = Config.from_environment()
        _config = new_config
        return new_config


# Lazy config proxy for deferred initialization
# Note: This will trigger lazy init on attribute access, not import
class _LazyConfig:
    """Lazy proxy for deferred Config initialization.

    This proxy class forwards all attribute access to the lazily-initialized
    Config singleton, enabling imports like `from config import config` without
    triggering immediate environment variable parsing.
    """

    def __getattr__(self, name: str) -> object:
        return object.__getattribute__(get_config(), name)

    def __repr__(self) -> str:
        return f"_LazyConfig(initialized={_config is not None})"


def setup_config_reload() -> Config:
    """Setup runtime configuration reload via SIGHUP.

    Must be called after Config singleton is initialized.
    """
    from ..utils.config_reload import setup_signal_handler

    config = Config.from_environment()
    _ = setup_signal_handler(lambda: Config.from_environment())

    return config


# Note: This is typed as Any to allow type checkers to accept it
# where Config is expected
# The actual Config object is accessed lazily through __getattr__
config: Config | _LazyConfig = _LazyConfig()
