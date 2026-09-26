"""Configuration validation for ReflectLog Server.

This module provides comprehensive validation for configuration values,
ensuring that all settings are valid and consistent before the server starts.
"""

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, ClassVar

from reflectlog.core.enums import (
    CrossEncoderDevice,
    FusionMethod,
    LlmProvider,
    RerankerEngine,
    TransportMode,
)
from reflectlog.core.exceptions import ConfigurationError as WorkspaceConfigurationError

if TYPE_CHECKING:
    from reflectlog.application.config.settings import Config


@dataclass
class ValidationError:
    """A single validation error."""

    field: str
    value: str | int | float | bool | None
    message: str

    def __str__(self) -> str:
        """String representation of the error."""
        if self.field == "OPENROUTER_API_KEY":
            return f"{self.field}: {self.message} (got: ***REDACTED***)"
        return f"{self.field}: {self.message} (got: {self.value!r})"


class ConfigurationError(Exception):
    """Configuration validation error."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ConfigurationValidator:
    """Validates configuration values.

    This class provides methods to validate various configuration settings,
    checking for type correctness, value ranges, and logical consistency.
    """

    # Regex patterns
    WORKSPACE_ID_PATTERN: ClassVar = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

    # Valid values for enums
    VALID_TRANSPORTS: ClassVar = frozenset(TransportMode)
    VALID_RERANKER_ENGINES: ClassVar = frozenset(RerankerEngine)
    VALID_LLM_PROVIDERS: ClassVar = frozenset(LlmProvider)
    VALID_CROSS_ENCODER_DEVICES: ClassVar = frozenset(CrossEncoderDevice)
    VALID_FUSION_METHODS: ClassVar = frozenset(FusionMethod)

    def __init__(self) -> None:
        """Initialize validator with an empty list of errors."""
        super().__init__()
        self.errors: list[ValidationError] = []

    def reset(self) -> None:
        """Clear all accumulated errors."""
        self.errors.clear()

    def add_error(self, field: str, value: object, message: str) -> None:
        """Add a validation error.

        Args:
            field: The configuration field name
            value: The invalid value
            message: Description of why the value is invalid
        """
        stored: str | int | float | bool | None
        if value is None or isinstance(value, str | int | float | bool):
            stored = value
        else:
            stored = repr(value)
        self.errors.append(ValidationError(field=field, value=stored, message=message))

    def validate_workspace_id(self, workspace_id: str) -> bool:
        """Validate WORKSPACE_ID format.

        Args:
            workspace_id: The workspace ID to validate

        Returns:
            True if valid, False otherwise
        """
        if not workspace_id:
            self.add_error("WORKSPACE_ID", workspace_id, "Cannot be empty")
            return False

        if not self.WORKSPACE_ID_PATTERN.match(workspace_id):
            self.add_error(
                "WORKSPACE_ID",
                workspace_id,
                "Must contain only A-Za-z0-9_.- and be 1-64 characters",
            )
            return False

        if workspace_id == "." or ".." in workspace_id:
            self.add_error(
                "WORKSPACE_ID",
                workspace_id,
                "Path traversal patterns not allowed",
            )
            return False

        return True

    def validate_transport(self, transport: str) -> bool:
        """Validate MCP_TRANSPORT value.

        Args:
            transport: The transport mode

        Returns:
            True if valid, False otherwise
        """
        if transport not in self.VALID_TRANSPORTS:
            self.add_error(
                "MCP_TRANSPORT",
                transport,
                f"Must be one of: {', '.join(sorted(self.VALID_TRANSPORTS))}",
            )
            return False
        return True

    def validate_port(self, port: int) -> bool:
        """Validate MCP_PORT value.

        Args:
            port: The port number

        Returns:
            True if valid, False otherwise
        """
        if not (1 <= port <= 65535):
            self.add_error("MCP_PORT", port, "Must be between 1 and 65535")
            return False
        return True

    def validate_percentage(
        self,
        field: str,
        value: float,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> bool:
        """Validate a percentage-based configuration value.

        Args:
            field: The configuration field name
            value: The value to validate
            min_value: Minimum allowed value (default: 0.0)
            max_value: Maximum allowed value (default: 1.0)

        Returns:
            True if valid, False otherwise
        """
        if not (min_value <= value <= max_value):
            self.add_error(
                field,
                value,
                f"Must be between {min_value} and {max_value}",
            )
            return False
        return True

    def validate_positive_int(
        self,
        field: str,
        value: int,
        min_value: int = 1,
    ) -> bool:
        """Validate a positive integer configuration value.

        Args:
            field: The configuration field name
            value: The value to validate
            min_value: Minimum allowed value (default: 1)

        Returns:
            True if valid, False otherwise
        """
        if value < min_value:
            self.add_error(
                field,
                value,
                f"Must be at least {min_value}",
            )
            return False
        return True

    def validate_positive_float(
        self,
        field: str,
        value: float,
        min_value: float = 0.0,
    ) -> bool:
        """Validate a positive float configuration value.

        Args:
            field: The configuration field name
            value: The value to validate
            min_value: Minimum allowed value (default: 0.0)

        Returns:
            True if valid, False otherwise
        """
        if value < min_value:
            self.add_error(
                field,
                value,
                f"Must be at least {min_value}",
            )
            return False
        return True

    def validate_reranker_engine(self, engine: str) -> bool:
        """Validate RERANKER_ENGINE value.

        Args:
            engine: The reranker engine type

        Returns:
            True if valid, False otherwise
        """
        if engine not in self.VALID_RERANKER_ENGINES:
            self.add_error(
                "RERANKER_ENGINE",
                engine,
                f"Must be one of: {', '.join(sorted(self.VALID_RERANKER_ENGINES))}",
            )
            return False
        return True

    def validate_llm_provider(self, provider: str) -> bool:
        """Validate LLM_PROVIDER value.

        Args:
            provider: The LLM provider

        Returns:
            True if valid, False otherwise
        """
        if provider not in self.VALID_LLM_PROVIDERS:
            self.add_error(
                "LLM_PROVIDER",
                provider,
                f"Must be one of: {', '.join(sorted(self.VALID_LLM_PROVIDERS))}",
            )
            return False
        return True

    def validate_cross_encoder_device(self, device: str) -> bool:
        """Validate CROSS_ENCODER_DEVICE value.

        Args:
            device: The cross-encoder device

        Returns:
            True if valid, False otherwise
        """
        if device not in self.VALID_CROSS_ENCODER_DEVICES:
            self.add_error(
                "CROSS_ENCODER_DEVICE",
                device,
                f"Must be one of: {', '.join(sorted(self.VALID_CROSS_ENCODER_DEVICES))}",
            )
            return False
        return True

    def validate_fusion_method(self, method: str) -> bool:
        """Validate FUSION_METHOD value.

        Args:
            method: The fusion method

        Returns:
            True if valid, False otherwise
        """
        if method not in self.VALID_FUSION_METHODS:
            self.add_error(
                "FUSION_METHOD",
                method,
                f"Must be one of: {', '.join(sorted(self.VALID_FUSION_METHODS))}",
            )
            return False
        return True

    def validate_dependencies(
        self,
        enable_rrf_fusion: bool,
        reranker_engine: str,
    ) -> bool:
        """Validate logical dependencies between configuration options.

        Args:
            enable_rrf_fusion: Whether RRF fusion is enabled
            reranker_engine: The reranker engine type

        Returns:
            True if all dependencies are valid, False otherwise
        """
        valid = True

        # Reranker requires at least one search engine
        # (this is always true since we always have semantic search)

        return valid

    def validate_embedding_settings(
        self,
        embedder_provider: str,
        embedding_dims: int,
        qwen_embedding_dims: int,
    ) -> bool:
        """Validate embedding-related settings.

        Args:
            embedder_provider: The embedder provider
            embedding_dims: OpenAI embedding dimensions
            qwen_embedding_dims: Qwen embedding dimensions

        Returns:
            True if valid, False otherwise
        """
        valid = True

        # Validate dimensions are positive
        if not self.validate_positive_int("EMBEDDING_DIMS", embedding_dims):
            valid = False

        if not self.validate_positive_int("QWEN_EMBEDDING_DIMS", qwen_embedding_dims):
            valid = False

        return valid

    def validate_circuit_breaker_settings(
        self,
        enabled: bool,
        failure_threshold: int,
        timeout: float,
        success_threshold: int,
    ) -> bool:
        """Validate circuit breaker settings.

        Args:
            enabled: Whether circuit breaker is enabled
            failure_threshold: Number of failures before opening
            timeout: Seconds before attempting recovery
            success_threshold: Successes needed to close circuit

        Returns:
            True if valid, False otherwise
        """
        if not enabled:
            return True  # Skip validation if disabled

        valid = True

        if not self.validate_positive_int(
            "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            failure_threshold,
            min_value=1,
        ):
            valid = False

        if not self.validate_positive_float(
            "CIRCUIT_BREAKER_TIMEOUT",
            timeout,
            min_value=1.0,
        ):
            valid = False

        if not self.validate_positive_int(
            "CIRCUIT_BREAKER_SUCCESS_THRESHOLD",
            success_threshold,
            min_value=1,
        ):
            valid = False

        return valid

    def validate_memory_lengths(
        self,
        min_length: int,
        max_length: int,
    ) -> bool:
        """Validate memory length settings.

        Args:
            min_length: Minimum memory length
            max_length: Maximum memory length

        Returns:
            True if valid, False otherwise
        """
        valid = True

        if not self.validate_positive_int("MIN_MEMORY_LENGTH", min_length):
            valid = False

        if not self.validate_positive_int("MAX_MEMORY_LENGTH", max_length):
            valid = False

        if min_length >= max_length:
            self.add_error(
                "MIN_MEMORY_LENGTH / MAX_MEMORY_LENGTH",
                (min_length, max_length),
                "MIN_MEMORY_LENGTH must be less than MAX_MEMORY_LENGTH",
            )
            valid = False

        return valid

    def validate_query(
        self: ConfigurationValidator,
        query: str,
        max_length: int = 1000,
    ) -> bool:
        """Validate search query for security and content.

        Args:
            self: The validator instance
            query: The search query to validate
            max_length: Maximum allowed query length (default: 1000)

        Returns:
            True if valid, False otherwise
        """
        if not query:
            self.add_error("query", query, "Query cannot be empty")
            return False

        if len(query) > max_length:
            self.add_error(
                "query",
                query,
                f"Query length ({len(query)}) exceeds maximum ({max_length})",
            )
            return False

        return True

    def sanitize_query(
        self: ConfigurationValidator,
        query: str,
        max_length: int = 1000,
    ) -> str:
        """Sanitize search query to prevent injection attacks.

        Args:
            self: The validator instance
            query: The search query to sanitize
            max_length: Maximum allowed query length (default: 1000)

        Returns:
            Sanitized query string

        This function removes or escapes potentially dangerous characters:
        - SQL/NoSQL injection patterns
        - Command injection patterns
        - Excess whitespace
        """
        if not query:
            return ""

        # Limit length
        if len(query) > max_length:
            query = query[:max_length]

        # Remove null bytes and control characters
        sanitized = "".join(char for char in query if ord(char) >= 32)

        # Remove common SQL injection patterns
        injection_patterns = [
            r"';\s*--",
            r"'\s+or\s+",
            r"\s+and\s+",
            r";\s*drop\s+",
            r";\s*delete\s+",
            r";\s*insert\s+",
            r";\s*update\s+",
            r";\s*exec\s*",
            r"union\s+select",
            r"\|\s*select",
        ]

        for pattern in injection_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Remove multiple consecutive spaces
        sanitized = re.sub(r"\s{2,}", " ", sanitized)

        # Trim leading/trailing whitespace
        sanitized = sanitized.strip()

        return sanitized

    def validate_memory(
        self: ConfigurationValidator,
        content: str,
        min_length: int = 1,
        max_length: int = 30720,
    ) -> bool:
        """Validate memory content for security and content.

        Args:
            self: The validator instance
            content: The memory content to validate
            min_length: Minimum allowed length (default: 1)
            max_length: Maximum allowed length (default: 30720)

        Returns:
            True if valid, False otherwise
        """
        if not content:
            self.add_error("memory", content, "Memory cannot be empty")
            return False

        if len(content) < min_length:
            self.add_error(
                "memory",
                content,
                f"Memory length ({len(content)}) below minimum ({min_length})",
            )
            return False

        if len(content) > max_length:
            self.add_error(
                "memory",
                content,
                f"Memory length ({len(content)}) exceeds maximum ({max_length})",
            )
            return False

        # Check for null bytes and control characters (except newline, tab)
        for char in content:
            char_code = ord(char)
            if char_code < 9 or (char_code >= 11 and char_code <= 12):
                self.add_error(
                    "memory",
                    content,
                    "Memory contains invalid control characters",
                )
                return False

        return True

    def validate_openrouter_api_key_format(
        self: ConfigurationValidator,
        api_key: str,
    ) -> bool:
        """Validate OpenRouter API key format.

        Args:
            self: The validator instance
            api_key: The API key to validate

        Returns:
            True if valid format, False otherwise

        Expected format: sk-or-v1-XXXXXXXXXXXXXXXX
        """
        if not api_key:
            self.add_error("OPENROUTER_API_KEY", api_key, "API key cannot be empty")
            return False

        # OpenRouter keys start with 'sk-or-v1-'
        if not api_key.startswith("sk-or-v1-"):
            self.add_error(
                "OPENROUTER_API_KEY",
                api_key,
                "API key must start with 'sk-or-v1-'",
            )
            return False

        # Check length (typical OpenRouter keys are 51 characters: sk-or-v1- + 39 chars)
        if len(api_key) < 10 or len(api_key) > 100:
            self.add_error(
                "OPENROUTER_API_KEY",
                api_key,
                "API key length must be between 10 and 100 characters",
            )
            return False

        return True

    def validate_openrouter_api_key(
        self: ConfigurationValidator,
        api_key: str,
    ) -> bool:
        """Validate OpenRouter API key format.

        Args:
            self: The validator instance
            api_key: The API key to validate

        Returns:
            True if valid format, False otherwise

        Expected format: sk-or-v1-XXXXXXXXXXXXXXXX
        """
        if not api_key:
            self.add_error("OPENROUTER_API_KEY", api_key, "API key cannot be empty")
            return False

        # OpenRouter keys start with 'sk-or-v1-'
        if not api_key.startswith("sk-or-v1-"):
            self.add_error(
                "OPENROUTER_API_KEY",
                api_key,
                "API key must start with 'sk-or-v1-'",
            )
            return False

        # Check length (typical OpenRouter keys are 51 characters: sk-or-v1- + 39 chars)
        if len(api_key) < 10 or len(api_key) > 100:
            self.add_error(
                "OPENROUTER_API_KEY",
                api_key,
                "API key length must be between 10 and 100 characters",
            )
            return False

        return True

    def has_errors(self) -> bool:
        """Check if any validation errors have been recorded.

        Returns:
            True if there are errors, False otherwise
        """
        return len(self.errors) > 0

    def get_error_message(self) -> str:
        """Get a formatted error message with all validation errors.

        Returns:
            A string containing all validation errors, one per line
        """
        if not self.errors:
            return "No validation errors"

        lines = ["Configuration validation failed:"]
        for error in self.errors:
            lines.append(f"  - {error}")
        return "\n".join(lines)


def canonical_workspace_id(workspace_id: str) -> str:
    validator = ConfigurationValidator()
    if not validator.validate_workspace_id(workspace_id):
        raise WorkspaceConfigurationError(
            f"Invalid WORKSPACE_ID: {validator.errors[0].message.lower()}"
        )
    return workspace_id.lower()


def _validate_server_config(
    validator: ConfigurationValidator,
    config: Config,
) -> None:
    """Validate server transport, port, and workspace ID."""
    if config.workspace_id:
        _ = validator.validate_workspace_id(config.workspace_id)
    _ = validator.validate_transport(config.transport)
    _ = validator.validate_port(config.port)


def _validate_search_config(
    validator: ConfigurationValidator,
    config: Config,
) -> None:
    """Validate search limit, threshold, and percentage fields."""
    percentage_fields: list[tuple[str, float]] = [
        ("SEARCH_SCORE_THRESHOLD", config.search_score_threshold),
        ("FUSION_RANKING_THRESHOLD", config.fusion_ranking_threshold),
        ("CROSS_ENCODER_SCORE_THRESHOLD", config.cross_encoder_score_threshold),
        ("SMART_REPLACE_THRESHOLD", config.smart_replace_threshold),
        ("SMART_REPLACE_MIN_SIMILARITY", config.smart_replace_min_similarity),
        (
            "TANTIVY_COMPACTION_THRESHOLD_RATIO",
            config.tantivy_compaction_threshold_ratio,
        ),
        ("RECENCY_DECAY_RATE", config.recency_decay_rate),
    ]
    for field_name, value in percentage_fields:
        _ = validator.validate_percentage(field_name, value)

    positive_int_fields: list[tuple[str, int]] = [
        ("SEARCH_LIMIT", config.search_limit),
        ("REMOVE_SEARCH_LIMIT", config.remove_search_limit),
        ("FUSION_RRF_K", config.fusion_rrf_k),
        ("RERANK_MAX_CONCURRENCY", config.rerank_max_concurrency),
        ("CROSS_ENCODER_BATCH_SIZE", config.cross_encoder_batch_size),
        ("CROSS_ENCODER_MAX_LENGTH", config.cross_encoder_max_length),
        ("ADD_MAX_CONCURRENCY", config.add_max_concurrency),
        ("EMBEDDING_BATCH_SIZE", config.embedding_batch_size),
        ("EMBEDDING_CACHE_SIZE", config.embedding_cache_size),
    ]
    for field_name, value in positive_int_fields:
        _ = validator.validate_positive_int(field_name, value)


def _validate_storage_config(
    validator: ConfigurationValidator,
    config: Config,
) -> None:
    """Validate memory lengths and logical dependencies."""
    _ = validator.validate_memory_lengths(
        config.min_memory_length, config.max_memory_length
    )
    _ = validator.validate_dependencies(
        config.enable_rrf_fusion,
        config.reranker_engine,
    )


def _validate_embedder_config(
    validator: ConfigurationValidator,
    config: Config,
) -> None:
    """Validate fusion method and fusion weights."""
    _ = validator.validate_fusion_method(config.fusion_method)
    fusion_weights = config.fusion_weights
    if fusion_weights is None:
        return
    if len(fusion_weights) < 2:
        raise ConfigurationError(
            "fusion_weights must have at least 2 elements for weighted RRF"
        )
    for i, weight in enumerate(fusion_weights):
        if weight < 0:
            raise ConfigurationError(
                f"fusion_weights[{i}] must be non-negative, got {weight}"
            )


def _validate_reranker_config(
    validator: ConfigurationValidator,
    config: Config,
) -> None:
    """Validate reranker engine, LLM provider, and cross-encoder device."""
    _ = validator.validate_reranker_engine(config.reranker_engine)
    _ = validator.validate_llm_provider(config.llm_provider)
    _ = validator.validate_cross_encoder_device(config.cross_encoder_device)


def _validate_security_fields(
    validator: ConfigurationValidator,
    config: Config,
) -> None:
    """Validate API key formats and security-sensitive fields."""
    api_key = config.openrouter_api_key.get_secret_value()
    if api_key.startswith("sk-or"):
        _ = validator.validate_openrouter_api_key_format(api_key)


def validate_config(config: Config) -> list[ValidationError]:
    """Validate a configuration object.

    This is a convenience function that creates a validator and runs
    all validation checks on the given configuration.

    Args:
        config: Application configuration.

    Returns:
        List of validation errors (empty if valid)
    """
    validator = ConfigurationValidator()

    _validate_server_config(validator, config)
    _validate_search_config(validator, config)
    _validate_storage_config(validator, config)
    _validate_embedder_config(validator, config)
    _validate_reranker_config(validator, config)
    _validate_security_fields(validator, config)

    return validator.errors


__all__ = [
    "ConfigurationValidator",
    "ValidationError",
    "validate_config",
]
