"""Closed-set string identifiers used across ReflectLog.

These ``StrEnum`` values serialize as their existing TEXT/JSON strings, so
SQLite rows and MCP payloads stay unchanged. Compare and assign the members
instead of repeating the literals.
"""

from enum import StrEnum

from reflectlog.core.exceptions import ConfigurationError


class TransitionKind(StrEnum):
    """Journal intent recorded in ``replacement_transitions.kind``."""

    ADD = "add"
    DELETE = "delete"
    REPLACE = "replace"

    @classmethod
    def from_stored(cls, value: str) -> TransitionKind:
        """Parse a stored kind. Unknown values raise ``ValueError``."""
        return cls(value)


class TransitionStatus(StrEnum):
    """Journal lifecycle recorded in ``replacement_transitions.status``."""

    PENDING = "pending"
    COMPLETED = "completed"

    @classmethod
    def from_stored(cls, value: str) -> TransitionStatus:
        """Parse a stored status. Unknown values raise ``ValueError``."""
        return cls(value)


class TransportMode(StrEnum):
    """MCP transport selected by ``MCP_TRANSPORT``."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"


class RerankerEngine(StrEnum):
    """Search reranker selected by ``RERANKER_ENGINE``."""

    CROSS_ENCODER = "cross_encoder"
    NONE = "none"


class FusionMethod(StrEnum):
    """Hybrid-search fusion algorithm selected by ``FUSION_METHOD``."""

    RRF = "rrf"
    SUM = "sum"
    MNZ = "mnz"
    MAX = "max"
    BORDAFUSE = "bordafuse"


class FusionNormalization(StrEnum):
    """Score normalization applied before fusion."""

    MIN_MAX = "min-max"
    MAX = "max"
    SUM = "sum"
    ZMUV = "zmuv"
    RANK = "rank"
    BORDA = "borda"


class DistanceMetric(StrEnum):
    """Vector similarity metric used by USearch."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    INNER_PRODUCT = "inner_product"


class LlmProvider(StrEnum):
    """LLM vendor used by smart replacement."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class CrossEncoderDevice(StrEnum):
    """Device for FlagReranker inference."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class EmbedderProvider(StrEnum):
    """Embedding backend selected by ``EMBEDDER_PROVIDER``."""

    OPENAI = "openai"
    LANGCHAIN = "langchain"
    WEMM = "wemm"


class WeMMModel(StrEnum):
    """Supported Tencent WeMM embedding checkpoints."""

    EMBEDDING_2B = "tencent/WeMM-Embedding-2B"
    EMBEDDING_4B = "tencent/WeMM-Embedding-4B"
    EMBEDDING_9B = "tencent/WeMM-Embedding-9B"

    @classmethod
    def from_config(cls, value: str) -> WeMMModel:
        try:
            return cls(value.strip())
        except ValueError:
            valid = ", ".join(model.value for model in cls)
            raise ConfigurationError(
                f"Invalid EMBEDDING_MODEL: '{value}'. Valid options: {valid}"
            ) from None

    @property
    def native_dimensions(self) -> int:
        match self:
            case WeMMModel.EMBEDDING_2B:
                return 2048
            case WeMMModel.EMBEDDING_4B:
                return 2560
            case WeMMModel.EMBEDDING_9B:
                return 4096

    @property
    def allowed_dimensions(self) -> frozenset[int]:
        shared = {64, 128, 256, 512, 1024}
        match self:
            case WeMMModel.EMBEDDING_2B:
                return frozenset({*shared, 2048})
            case WeMMModel.EMBEDDING_4B:
                return frozenset({*shared, 2560})
            case WeMMModel.EMBEDDING_9B:
                return frozenset({*shared, 2048, 4096})

    def resolve_dimensions(self, override: int | None) -> int:
        dimensions = self.native_dimensions if override is None else override
        if dimensions not in self.allowed_dimensions:
            allowed = ", ".join(str(value) for value in sorted(self.allowed_dimensions))
            raise ConfigurationError(
                f"Invalid WEMM_EMBEDDING_DIMS for {self.value}: {dimensions}. "
                f"Valid options: {allowed}"
            )
        return dimensions


class WeMMDevice(StrEnum):
    """Device selection for local WeMM inference."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class EngineReadiness(StrEnum):
    """Public search-engine readiness reported by health checks."""

    INITIALIZED = "initialized"
    PENDING = "pending"
    NOT_INITIALIZED = "not_initialized"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class HealthStatus(StrEnum):
    """Overall server health reported by ``health_check``."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ToolName(StrEnum):
    """Registered MCP tool identifiers."""

    ADD = "add"
    GET_ALL = "get_all"
    SEARCH = "search"
    REMOVE = "remove"
    HEALTH_CHECK = "health_check"


class ConfigProfile(StrEnum):
    """Named configuration preset selected by ``REFLECTLOG_PROFILE``."""

    SIMPLE = "simple"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    CUSTOM = "custom"


def parse_str_enum[E: StrEnum](
    enum_cls: type[E],
    value: str,
    *,
    field: str,
    default: E | None = None,
) -> E:
    """Parse a case-insensitive identifier into ``enum_cls``.

    Args:
        enum_cls: Target string enumeration.
        value: Raw configuration string.
        field: Environment or config field name used in error text.
        default: Value returned when ``value`` is not a member. When omitted,
            unknown values raise ``ConfigurationError``.
    """
    try:
        return enum_cls(value.strip().lower())
    except ValueError:
        if default is not None:
            return default
        valid = ", ".join(member.value for member in enum_cls)
        raise ConfigurationError(
            f"Invalid {field}: '{value}'. Valid options: {valid}"
        ) from None
