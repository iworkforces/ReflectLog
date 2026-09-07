"""Tests for reflectlog.application.config.validation module."""

from dataclasses import replace

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.config.validation import (
    ConfigurationValidator,
    ValidationError,
    validate_config,
)
from reflectlog.application.utils.security import SecretString
from reflectlog.core.enums import (
    CrossEncoderDevice,
    FusionMethod,
    LlmProvider,
    RerankerEngine,
    TransportMode,
)

# ---------------------------------------------------------------------------
# ValidationError dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidationError:
    """Tests for the ValidationError dataclass."""

    def test_str_representation(self):
        """Test __str__ includes field, message, and repr of value."""
        error = ValidationError(field="PORT", value=99999, message="out of range")
        result = str(error)
        assert "PORT" in result
        assert "out of range" in result
        assert "99999" in result

    def test_str_with_string_value(self):
        """Test __str__ with a string value shows repr quotes."""
        error = ValidationError(field="NAME", value="bad", message="invalid")
        result = str(error)
        assert "'bad'" in result

    def test_fields_accessible(self):
        """Test dataclass fields are directly accessible."""
        error = ValidationError(field="f", value="v", message="m")
        assert error.field == "f"
        assert error.value == "v"
        assert error.message == "m"


# ---------------------------------------------------------------------------
# ConfigurationValidator — lifecycle helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidatorLifecycle:
    """Tests for ConfigurationValidator init, reset, add_error, has_errors, get_error_message."""

    def test_init_has_no_errors(self):
        """Freshly created validator has no errors."""
        v = ConfigurationValidator()
        assert v.errors == []
        assert v.has_errors() is False

    def test_add_error_records_error(self):
        """add_error appends a ValidationError to the list."""
        v = ConfigurationValidator()
        v.add_error("FIELD", "val", "msg")
        assert len(v.errors) == 1
        assert v.errors[0].field == "FIELD"
        assert v.errors[0].value == "val"
        assert v.errors[0].message == "msg"

    def test_has_errors_true_after_add(self):
        """has_errors returns True after adding an error."""
        v = ConfigurationValidator()
        v.add_error("X", 1, "bad")
        assert v.has_errors() is True

    def test_reset_clears_errors(self):
        """reset() clears accumulated errors."""
        v = ConfigurationValidator()
        v.add_error("A", 1, "err1")
        v.add_error("B", 2, "err2")
        assert len(v.errors) == 2
        v.reset()
        assert v.errors == []
        assert v.has_errors() is False

    def test_get_error_message_no_errors(self):
        """get_error_message with no errors returns helpful text."""
        v = ConfigurationValidator()
        assert v.get_error_message() == "No validation errors"

    def test_get_error_message_with_errors(self):
        """get_error_message formats multiple errors line-by-line."""
        v = ConfigurationValidator()
        v.add_error("A", 1, "first issue")
        v.add_error("B", 2, "second issue")
        msg = v.get_error_message()
        assert "Configuration validation failed:" in msg
        assert "first issue" in msg
        assert "second issue" in msg
        # Each error on its own line with indent
        lines = msg.split("\n")
        assert len(lines) == 3  # header + 2 error lines


# ---------------------------------------------------------------------------
# validate_workspace_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateWorkspaceId:
    """Tests for ConfigurationValidator.validate_workspace_id."""

    def test_valid_alphanumeric(self):
        """Simple alphanumeric ID is valid."""
        v = ConfigurationValidator()
        assert v.validate_workspace_id("my_project123") is True
        assert not v.has_errors()

    def test_valid_with_dots_and_dashes(self):
        """Dots and dashes are allowed characters."""
        v = ConfigurationValidator()
        assert v.validate_workspace_id("my-project.v1") is True

    def test_empty_string_invalid(self):
        """Empty workspace ID is rejected."""
        v = ConfigurationValidator()
        assert v.validate_workspace_id("") is False
        assert v.has_errors()
        assert "Cannot be empty" in v.errors[0].message

    def test_invalid_characters(self):
        """Characters outside [A-Za-z0-9_.-] are rejected."""
        v = ConfigurationValidator()
        assert v.validate_workspace_id("bad@chars!") is False
        assert "Must contain only" in v.errors[0].message

    def test_too_long(self):
        """ID longer than 64 characters is rejected."""
        v = ConfigurationValidator()
        assert v.validate_workspace_id("a" * 65) is False

    def test_exactly_64_chars_valid(self):
        """ID of exactly 64 characters is valid."""
        v = ConfigurationValidator()
        assert v.validate_workspace_id("a" * 64) is True

    def test_path_traversal_double_dot(self):
        """Double-dot path traversal is rejected."""
        v = ConfigurationValidator()
        assert v.validate_workspace_id("a..b") is False
        assert "Path traversal" in v.errors[0].message

    def test_path_traversal_leading_slash(self):
        """Leading slash path traversal is rejected."""
        v = ConfigurationValidator()
        # Leading slash also fails the regex first, so just test it returns False
        assert v.validate_workspace_id("/etc") is False
        assert v.has_errors()

    def test_single_char_valid(self):
        """Single character ID is valid."""
        v = ConfigurationValidator()
        assert v.validate_workspace_id("x") is True


# ---------------------------------------------------------------------------
# validate_transport
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateTransport:
    """Tests for ConfigurationValidator.validate_transport."""

    @pytest.mark.parametrize("transport", ["stdio", "http", "sse", "streamable-http"])
    def test_valid_transports(self, transport: str) -> None:
        """All valid transport modes are accepted."""
        v = ConfigurationValidator()
        assert v.validate_transport(transport) is True
        assert not v.has_errors()

    def test_invalid_transport(self):
        """Unknown transport mode is rejected."""
        v = ConfigurationValidator()
        assert v.validate_transport("websocket") is False
        assert "Must be one of" in v.errors[0].message


# ---------------------------------------------------------------------------
# validate_port
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidatePort:
    """Tests for ConfigurationValidator.validate_port."""

    def test_valid_port(self):
        """Normal port number is accepted."""
        v = ConfigurationValidator()
        assert v.validate_port(8080) is True

    def test_port_min_boundary(self):
        """Port 1 is valid (lower boundary)."""
        v = ConfigurationValidator()
        assert v.validate_port(1) is True

    def test_port_max_boundary(self):
        """Port 65535 is valid (upper boundary)."""
        v = ConfigurationValidator()
        assert v.validate_port(65535) is True

    def test_port_zero_invalid(self):
        """Port 0 is invalid."""
        v = ConfigurationValidator()
        assert v.validate_port(0) is False
        assert "Must be between 1 and 65535" in v.errors[0].message

    def test_port_too_high_invalid(self):
        """Port above 65535 is invalid."""
        v = ConfigurationValidator()
        assert v.validate_port(70000) is False

    def test_port_negative_invalid(self):
        """Negative port is invalid."""
        v = ConfigurationValidator()
        assert v.validate_port(-1) is False


# ---------------------------------------------------------------------------
# validate_percentage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidatePercentage:
    """Tests for ConfigurationValidator.validate_percentage."""

    def test_valid_percentage(self):
        """Value within default 0.0-1.0 range is accepted."""
        v = ConfigurationValidator()
        assert v.validate_percentage("THRESHOLD", 0.5) is True

    def test_min_boundary(self):
        """Value at 0.0 is accepted."""
        v = ConfigurationValidator()
        assert v.validate_percentage("THRESHOLD", 0.0) is True

    def test_max_boundary(self):
        """Value at 1.0 is accepted."""
        v = ConfigurationValidator()
        assert v.validate_percentage("THRESHOLD", 1.0) is True

    def test_below_min_invalid(self):
        """Value below minimum is rejected."""
        v = ConfigurationValidator()
        assert v.validate_percentage("THRESHOLD", -0.1) is False
        assert "Must be between" in v.errors[0].message

    def test_above_max_invalid(self):
        """Value above maximum is rejected."""
        v = ConfigurationValidator()
        assert v.validate_percentage("THRESHOLD", 1.1) is False

    def test_custom_range(self):
        """Custom min/max range works."""
        v = ConfigurationValidator()
        assert (
            v.validate_percentage("FIELD", 5.0, min_value=2.0, max_value=10.0) is True
        )
        assert (
            v.validate_percentage("FIELD", 1.0, min_value=2.0, max_value=10.0) is False
        )


# ---------------------------------------------------------------------------
# validate_positive_int
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidatePositiveInt:
    """Tests for ConfigurationValidator.validate_positive_int."""

    def test_valid_positive_int(self):
        """Value above min_value is accepted."""
        v = ConfigurationValidator()
        assert v.validate_positive_int("LIMIT", 10) is True

    def test_at_min_value(self):
        """Value at default min_value (1) is accepted."""
        v = ConfigurationValidator()
        assert v.validate_positive_int("LIMIT", 1) is True

    def test_below_min_invalid(self):
        """Value below min_value is rejected."""
        v = ConfigurationValidator()
        assert v.validate_positive_int("LIMIT", 0) is False
        assert "Must be at least" in v.errors[0].message

    def test_custom_min_value(self):
        """Custom min_value works."""
        v = ConfigurationValidator()
        assert v.validate_positive_int("FIELD", 5, min_value=5) is True
        assert v.validate_positive_int("FIELD", 4, min_value=5) is False

    def test_negative_value_invalid(self):
        """Negative value is rejected."""
        v = ConfigurationValidator()
        assert v.validate_positive_int("FIELD", -5) is False


# ---------------------------------------------------------------------------
# validate_positive_float
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidatePositiveFloat:
    """Tests for ConfigurationValidator.validate_positive_float."""

    def test_valid_positive_float(self):
        """Value above min_value is accepted."""
        v = ConfigurationValidator()
        assert v.validate_positive_float("RATE", 0.5) is True

    def test_at_min_value(self):
        """Value at default min_value (0.0) is accepted."""
        v = ConfigurationValidator()
        assert v.validate_positive_float("RATE", 0.0) is True

    def test_below_min_invalid(self):
        """Value below min_value is rejected."""
        v = ConfigurationValidator()
        assert v.validate_positive_float("RATE", -0.1) is False
        assert "Must be at least" in v.errors[0].message

    def test_custom_min_value(self):
        """Custom min_value works."""
        v = ConfigurationValidator()
        assert v.validate_positive_float("TIMEOUT", 5.0, min_value=1.0) is True
        assert v.validate_positive_float("TIMEOUT", 0.5, min_value=1.0) is False


# ---------------------------------------------------------------------------
# validate_reranker_engine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateRerankerEngine:
    """Tests for ConfigurationValidator.validate_reranker_engine."""

    @pytest.mark.parametrize("engine", ["cross_encoder", "none"])
    def test_valid_engines(self, engine: str) -> None:
        """All valid reranker engines are accepted."""
        v = ConfigurationValidator()
        assert v.validate_reranker_engine(engine) is True

    def test_invalid_engine(self):
        """Unknown engine is rejected."""
        v = ConfigurationValidator()
        assert v.validate_reranker_engine("transformers") is False
        assert "Must be one of" in v.errors[0].message


# ---------------------------------------------------------------------------
# validate_llm_provider
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateLlmProvider:
    """Tests for ConfigurationValidator.validate_llm_provider."""

    @pytest.mark.parametrize("provider", ["openai", "anthropic"])
    def test_valid_providers(self, provider: str) -> None:
        """All valid LLM providers are accepted."""
        v = ConfigurationValidator()
        assert v.validate_llm_provider(provider) is True

    def test_invalid_provider(self):
        """Unknown provider is rejected."""
        v = ConfigurationValidator()
        assert v.validate_llm_provider("google") is False
        assert "Must be one of" in v.errors[0].message


# ---------------------------------------------------------------------------
# validate_cross_encoder_device
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateCrossEncoderDevice:
    """Tests for ConfigurationValidator.validate_cross_encoder_device."""

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    def test_valid_devices(self, device: str) -> None:
        """All valid devices are accepted."""
        v = ConfigurationValidator()
        assert v.validate_cross_encoder_device(device) is True

    def test_invalid_device(self):
        """Unknown device is rejected."""
        v = ConfigurationValidator()
        assert v.validate_cross_encoder_device("tpu") is False
        assert "Must be one of" in v.errors[0].message


# ---------------------------------------------------------------------------
# validate_fusion_method
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateFusionMethod:
    """Tests for ConfigurationValidator.validate_fusion_method."""

    @pytest.mark.parametrize("method", ["rrf", "sum", "mnz", "max", "bordafuse"])
    def test_valid_methods(self, method: str) -> None:
        """All valid fusion methods are accepted."""
        v = ConfigurationValidator()
        assert v.validate_fusion_method(method) is True

    def test_invalid_method(self):
        """Unknown method is rejected."""
        v = ConfigurationValidator()
        assert v.validate_fusion_method("average") is False
        assert "Must be one of" in v.errors[0].message


# ---------------------------------------------------------------------------
# validate_dependencies
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateDependencies:
    """Tests for ConfigurationValidator.validate_dependencies."""

    def test_all_valid_returns_true(self):
        """Valid dependency combination returns True."""
        v = ConfigurationValidator()
        result = v.validate_dependencies(
            enable_rrf_fusion=True,
            reranker_engine="cross_encoder",
        )
        assert result is True
        assert not v.has_errors()

    def test_no_reranker_still_valid(self):
        """No reranker engine is allowed."""
        v = ConfigurationValidator()
        result = v.validate_dependencies(
            enable_rrf_fusion=False,
            reranker_engine="none",
        )
        assert result is True


# ---------------------------------------------------------------------------
# validate_embedding_settings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateEmbeddingSettings:
    """Tests for ConfigurationValidator.validate_embedding_settings."""

    def test_valid_settings(self):
        """Valid embedding dimensions are accepted."""
        v = ConfigurationValidator()
        result = v.validate_embedding_settings("openai", 4096, 1024)
        assert result is True
        assert not v.has_errors()

    def test_zero_embedding_dims_invalid(self):
        """Zero embedding dimensions are rejected."""
        v = ConfigurationValidator()
        result = v.validate_embedding_settings("openai", 0, 1024)
        assert result is False

    def test_zero_qwen_dims_invalid(self):
        """Zero Qwen embedding dimensions are rejected."""
        v = ConfigurationValidator()
        result = v.validate_embedding_settings("qwen", 4096, 0)
        assert result is False

    def test_both_invalid(self):
        """Both invalid dimensions are caught."""
        v = ConfigurationValidator()
        result = v.validate_embedding_settings("openai", 0, 0)
        assert result is False
        assert len(v.errors) == 2


# ---------------------------------------------------------------------------
# validate_circuit_breaker_settings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateCircuitBreakerSettings:
    """Tests for ConfigurationValidator.validate_circuit_breaker_settings."""

    def test_disabled_skips_validation(self):
        """When disabled, validation is skipped entirely."""
        v = ConfigurationValidator()
        result = v.validate_circuit_breaker_settings(
            enabled=False,
            failure_threshold=0,
            timeout=0.0,
            success_threshold=0,
        )
        assert result is True
        assert not v.has_errors()

    def test_valid_enabled_settings(self):
        """Valid enabled settings are accepted."""
        v = ConfigurationValidator()
        result = v.validate_circuit_breaker_settings(
            enabled=True,
            failure_threshold=5,
            timeout=60.0,
            success_threshold=2,
        )
        assert result is True
        assert not v.has_errors()

    def test_invalid_failure_threshold(self):
        """Zero failure threshold is rejected."""
        v = ConfigurationValidator()
        result = v.validate_circuit_breaker_settings(
            enabled=True,
            failure_threshold=0,
            timeout=60.0,
            success_threshold=2,
        )
        assert result is False

    def test_invalid_timeout(self):
        """Timeout below 1.0 is rejected."""
        v = ConfigurationValidator()
        result = v.validate_circuit_breaker_settings(
            enabled=True,
            failure_threshold=5,
            timeout=0.5,
            success_threshold=2,
        )
        assert result is False

    def test_invalid_success_threshold(self):
        """Zero success threshold is rejected."""
        v = ConfigurationValidator()
        result = v.validate_circuit_breaker_settings(
            enabled=True,
            failure_threshold=5,
            timeout=60.0,
            success_threshold=0,
        )
        assert result is False

    def test_all_invalid_when_enabled(self):
        """All invalid settings produce multiple errors."""
        v = ConfigurationValidator()
        result = v.validate_circuit_breaker_settings(
            enabled=True,
            failure_threshold=0,
            timeout=0.0,
            success_threshold=0,
        )
        assert result is False
        assert len(v.errors) == 3


# ---------------------------------------------------------------------------
# validate_message_lengths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateMemoryLengths:
    """Tests for ConfigurationValidator.validate_memory_lengths."""

    def test_valid_lengths(self):
        """Valid min < max is accepted."""
        v = ConfigurationValidator()
        result = v.validate_memory_lengths(1, 30720)
        assert result is True
        assert not v.has_errors()

    def test_min_equals_max_invalid(self):
        """min == max is rejected."""
        v = ConfigurationValidator()
        result = v.validate_memory_lengths(100, 100)
        assert result is False
        assert any("must be less than" in e.message.lower() for e in v.errors)

    def test_min_greater_than_max_invalid(self):
        """min > max is rejected."""
        v = ConfigurationValidator()
        result = v.validate_memory_lengths(200, 100)
        assert result is False

    def test_zero_min_invalid(self):
        """Zero min length is rejected."""
        v = ConfigurationValidator()
        result = v.validate_memory_lengths(0, 100)
        assert result is False

    def test_zero_max_invalid(self):
        """Zero max length is rejected."""
        v = ConfigurationValidator()
        result = v.validate_memory_lengths(0, 0)
        assert result is False


# ---------------------------------------------------------------------------
# validate_query
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateQuery:
    """Tests for ConfigurationValidator.validate_query."""

    def test_valid_query(self):
        """Normal query is accepted."""
        v = ConfigurationValidator()
        assert v.validate_query("search term") is True
        assert not v.has_errors()

    def test_empty_query_invalid(self):
        """Empty query is rejected."""
        v = ConfigurationValidator()
        assert v.validate_query("") is False
        assert "cannot be empty" in v.errors[0].message.lower()

    def test_query_exceeds_max_length(self):
        """Query exceeding max_length is rejected."""
        v = ConfigurationValidator()
        assert v.validate_query("x" * 1001) is False
        assert "exceeds maximum" in v.errors[0].message.lower()

    def test_query_at_max_length_valid(self):
        """Query at exactly max_length is accepted."""
        v = ConfigurationValidator()
        assert v.validate_query("x" * 1000) is True

    def test_custom_max_length(self):
        """Custom max_length is respected."""
        v = ConfigurationValidator()
        assert v.validate_query("x" * 50, max_length=50) is True
        assert v.validate_query("x" * 51, max_length=50) is False


# ---------------------------------------------------------------------------
# sanitize_query
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSanitizeQuery:
    """Tests for ConfigurationValidator.sanitize_query."""

    def test_empty_query_returns_empty(self):
        """Empty input returns empty string."""
        v = ConfigurationValidator()
        assert v.sanitize_query("") == ""

    def test_normal_query_unchanged(self):
        """Normal text is returned as-is."""
        v = ConfigurationValidator()
        assert v.sanitize_query("hello world") == "hello world"

    def test_truncation_to_max_length(self):
        """Long query is truncated to max_length."""
        v = ConfigurationValidator()
        result = v.sanitize_query("a" * 2000, max_length=100)
        assert len(result) <= 100

    def test_removes_null_bytes(self):
        """Null bytes and control characters are removed."""
        v = ConfigurationValidator()
        result = v.sanitize_query("hello\x00world\x01test")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_removes_sql_injection_semicolon_drop(self):
        """SQL injection pattern "; DROP" is removed."""
        v = ConfigurationValidator()
        result = v.sanitize_query("test; drop table users")
        assert "drop" not in result.lower()

    def test_removes_sql_injection_union_select(self):
        """SQL injection pattern "UNION SELECT" is removed."""
        v = ConfigurationValidator()
        result = v.sanitize_query("test union select * from users")
        assert "union select" not in result.lower()

    def test_removes_sql_injection_semicolon_delete(self):
        """SQL injection pattern "; DELETE" is removed."""
        v = ConfigurationValidator()
        result = v.sanitize_query("test; delete from users")
        assert "delete" not in result.lower()

    def test_removes_sql_injection_semicolon_insert(self):
        """SQL injection pattern "; INSERT" is removed."""
        v = ConfigurationValidator()
        result = v.sanitize_query("test; insert into users")
        assert "insert" not in result.lower()

    def test_removes_sql_injection_semicolon_update(self):
        """SQL injection pattern "; UPDATE" is removed."""
        v = ConfigurationValidator()
        result = v.sanitize_query("test; update users set")
        assert "update" not in result.lower()

    def test_removes_sql_injection_semicolon_exec(self):
        """SQL injection pattern "; EXEC" is removed."""
        v = ConfigurationValidator()
        result = v.sanitize_query("test; exec xp_cmdshell")
        assert "exec" not in result.lower()

    def test_removes_sql_injection_quote_or(self):
        """SQL injection pattern "' OR" is removed."""
        v = ConfigurationValidator()
        result = v.sanitize_query("admin' or 1=1")
        assert "' or" not in result.lower()

    def test_removes_sql_injection_quote_comment(self):
        """SQL injection pattern "'; --" is removed."""
        v = ConfigurationValidator()
        result = v.sanitize_query("admin'; -- comment")
        assert "'; --" not in result

    def test_removes_pipe_select(self):
        """SQL injection pattern "| SELECT" is removed."""
        v = ConfigurationValidator()
        result = v.sanitize_query("test| select password")
        assert "| select" not in result.lower()

    def test_removes_and_injection(self):
        """SQL injection pattern " AND " is removed."""
        v = ConfigurationValidator()
        result = v.sanitize_query("1=1 and 1=1")
        assert " and " not in result.lower()

    def test_collapses_multiple_spaces(self):
        """Multiple consecutive spaces are collapsed to one."""
        v = ConfigurationValidator()
        result = v.sanitize_query("hello     world")
        assert result == "hello world"

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        v = ConfigurationValidator()
        result = v.sanitize_query("  hello world  ")
        assert result == "hello world"

    def test_case_insensitive_injection_removal(self):
        """SQL injection detection is case-insensitive."""
        v = ConfigurationValidator()
        result = v.sanitize_query("test; DROP TABLE users")
        assert "DROP" not in result

    def test_custom_max_length(self):
        """Custom max_length truncates before sanitization."""
        v = ConfigurationValidator()
        result = v.sanitize_query("x" * 200, max_length=50)
        assert len(result) <= 50


# ---------------------------------------------------------------------------
# validate_message
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateMemory:
    """Tests for ConfigurationValidator.validate_memory."""

    def test_valid_memory(self):
        """Normal memory is accepted."""
        v = ConfigurationValidator()
        assert v.validate_memory("Hello, world!") is True
        assert not v.has_errors()

    def test_empty_memory_invalid(self):
        """Empty memory is rejected."""
        v = ConfigurationValidator()
        assert v.validate_memory("") is False
        assert "cannot be empty" in v.errors[0].message.lower()

    def test_memory_below_min_length(self):
        """Memory shorter than min_length is rejected."""
        v = ConfigurationValidator()
        assert v.validate_memory("a", min_length=5) is False
        assert "below minimum" in v.errors[0].message.lower()

    def test_memory_above_max_length(self):
        """Memory exceeding max_length is rejected."""
        v = ConfigurationValidator()
        assert v.validate_memory("x" * 31000, max_length=30720) is False
        assert "exceeds maximum" in v.errors[0].message.lower()

    def test_memory_at_min_length_valid(self):
        """Memory at exactly min_length is accepted."""
        v = ConfigurationValidator()
        assert v.validate_memory("ab", min_length=2) is True

    def test_memory_at_max_length_valid(self):
        """Memory at exactly max_length is accepted."""
        v = ConfigurationValidator()
        assert v.validate_memory("x" * 100, max_length=100) is True

    def test_memory_with_newlines_valid(self):
        """Memory with newlines (char code 10) is valid."""
        v = ConfigurationValidator()
        assert v.validate_memory("line1\nline2") is True

    def test_memory_with_tabs_valid(self):
        """Memory with tabs (char code 9) is valid."""
        v = ConfigurationValidator()
        assert v.validate_memory("col1\tcol2") is True

    def test_memory_with_null_byte_invalid(self):
        """Memory with null byte (char code 0) is rejected."""
        v = ConfigurationValidator()
        assert v.validate_memory("hello\x00world") is False
        assert "control characters" in v.errors[0].message.lower()

    def test_memory_with_low_control_char_invalid(self):
        """Memory with control char < 9 (e.g. 0x01) is rejected."""
        v = ConfigurationValidator()
        assert v.validate_memory("hello\x01world") is False
        assert "control characters" in v.errors[0].message.lower()

    def test_memory_with_char_11_invalid(self):
        """Memory with vertical tab (char code 11) is rejected."""
        v = ConfigurationValidator()
        assert v.validate_memory("hello\x0bworld") is False

    def test_memory_with_char_12_invalid(self):
        """Memory with form feed (char code 12) is rejected."""
        v = ConfigurationValidator()
        assert v.validate_memory("hello\x0cworld") is False

    def test_memory_with_carriage_return_valid(self):
        """Memory with carriage return (char code 13) is valid."""
        v = ConfigurationValidator()
        assert v.validate_memory("line1\rline2") is True

    def test_memory_with_unicode_valid(self):
        """Memory with unicode characters is valid."""
        v = ConfigurationValidator()
        assert v.validate_memory("Hello 世界 🌍") is True


# ---------------------------------------------------------------------------
# validate_openrouter_api_key_format
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateOpenrouterApiKeyFormat:
    """Tests for ConfigurationValidator.validate_openrouter_api_key_format."""

    def test_valid_key(self):
        """Valid OpenRouter key format is accepted."""
        v = ConfigurationValidator()
        key = "sk-or-v1-" + "a" * 42
        assert v.validate_openrouter_api_key_format(key) is True

    def test_empty_key_invalid(self):
        """Empty key is rejected."""
        v = ConfigurationValidator()
        assert v.validate_openrouter_api_key_format("") is False
        assert "cannot be empty" in v.errors[0].message.lower()

    def test_wrong_prefix_invalid(self):
        """Key without sk-or-v1- prefix is rejected."""
        v = ConfigurationValidator()
        assert v.validate_openrouter_api_key_format("sk-abc123def456") is False
        assert "must start with" in v.errors[0].message.lower()

    def test_too_short_invalid(self):
        """Key shorter than 10 characters is rejected."""
        v = ConfigurationValidator()
        assert v.validate_openrouter_api_key_format("sk-or-v1-") is False
        assert "length" in v.errors[0].message.lower()

    def test_too_long_invalid(self):
        """Key longer than 100 characters is rejected."""
        v = ConfigurationValidator()
        key = "sk-or-v1-" + "a" * 100  # 109 total
        assert v.validate_openrouter_api_key_format(key) is False

    def test_at_min_length_valid(self):
        """Key at exactly 10 characters is accepted."""
        v = ConfigurationValidator()
        key = "sk-or-v1-x"  # 10 chars
        assert v.validate_openrouter_api_key_format(key) is True

    def test_at_max_length_valid(self):
        """Key at exactly 100 characters is accepted."""
        v = ConfigurationValidator()
        key = "sk-or-v1-" + "a" * 91  # 100 total
        assert v.validate_openrouter_api_key_format(key) is True


# ---------------------------------------------------------------------------
# validate_openrouter_api_key (duplicate method)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateOpenrouterApiKey:
    """Tests for ConfigurationValidator.validate_openrouter_api_key."""

    def test_valid_key(self):
        """Valid OpenRouter key format is accepted."""
        v = ConfigurationValidator()
        key = "sk-or-v1-" + "b" * 42
        assert v.validate_openrouter_api_key(key) is True

    def test_empty_key_invalid(self):
        """Empty key is rejected."""
        v = ConfigurationValidator()
        assert v.validate_openrouter_api_key("") is False

    def test_wrong_prefix_invalid(self):
        """Key without sk-or-v1- prefix is rejected."""
        v = ConfigurationValidator()
        assert v.validate_openrouter_api_key("wrong-prefix-key") is False

    def test_too_short_invalid(self):
        """Key shorter than 10 characters is rejected."""
        v = ConfigurationValidator()
        assert v.validate_openrouter_api_key("sk-or-v1-") is False

    def test_too_long_invalid(self):
        """Key longer than 100 characters is rejected."""
        v = ConfigurationValidator()
        key = "sk-or-v1-" + "b" * 100
        assert v.validate_openrouter_api_key(key) is False


# ---------------------------------------------------------------------------
# validate_config convenience function
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateConfig:
    """Tests for the validate_config() convenience function."""

    def _make_config(self, **overrides: object) -> Config:
        """Create a Config with valid defaults and optional field overrides."""
        config = Config(
            workspace_id="my-project",
            openrouter_api_key=SecretString("sk-or-v1-" + "a" * 42),
            transport=TransportMode.STDIO,
            port=9103,
            search_score_threshold=0.5,
            fusion_ranking_threshold=0.3,
            cross_encoder_score_threshold=0.7,
            smart_replace_threshold=0.7,
            smart_replace_min_similarity=0.5,
            tantivy_compaction_threshold_ratio=0.2,
            recency_decay_rate=0.01,
            search_limit=10,
            remove_search_limit=5,
            fusion_rrf_k=60,
            rerank_max_concurrency=4,
            cross_encoder_batch_size=32,
            cross_encoder_max_length=512,
            add_max_concurrency=8,
            embedding_batch_size=32,
            embedding_cache_size=100,
            reranker_engine=RerankerEngine.CROSS_ENCODER,
            llm_provider=LlmProvider.OPENAI,
            cross_encoder_device=CrossEncoderDevice.CPU,
            fusion_method=FusionMethod.RRF,
            min_memory_length=1,
            max_memory_length=30720,
            enable_rrf_fusion=True,
        )
        if not overrides:
            return config
        return replace(config, **overrides)

    def test_valid_config_no_errors(self):
        """Valid configuration produces no errors."""
        cfg = self._make_config()
        errors = validate_config(cfg)
        assert errors == []

    def test_invalid_workspace_id(self):
        """Invalid workspace_id produces error."""
        cfg = self._make_config(workspace_id="bad@id!")
        errors = validate_config(cfg)
        assert any(e.field == "WORKSPACE_ID" for e in errors)

    def test_invalid_port(self):
        """Invalid port produces error."""
        cfg = self._make_config(port=0)
        errors = validate_config(cfg)
        assert any(e.field == "MCP_PORT" for e in errors)

    def test_invalid_percentage_field(self):
        """Out-of-range percentage produces error."""
        cfg = self._make_config(search_score_threshold=2.0)
        errors = validate_config(cfg)
        assert any(e.field == "SEARCH_SCORE_THRESHOLD" for e in errors)

    def test_invalid_positive_int_field(self):
        """Zero positive int field produces error."""
        cfg = self._make_config(search_limit=0)
        errors = validate_config(cfg)
        assert any(e.field == "SEARCH_LIMIT" for e in errors)

    def test_invalid_memory_lengths(self):
        """min >= max memory lengths produces error."""
        cfg = self._make_config(min_memory_length=500, max_memory_length=100)
        errors = validate_config(cfg)
        assert len(errors) > 0

    def test_openrouter_api_key_validated(self):
        """OpenRouter API key format is validated when present."""
        cfg = self._make_config(openrouter_api_key=SecretString("sk-or-bad"))
        errors = validate_config(cfg)
        assert any("OPENROUTER_API_KEY" in e.field for e in errors)

    def test_valid_openrouter_api_key(self):
        """Valid OpenRouter API key produces no error."""
        cfg = self._make_config(openrouter_api_key=SecretString("sk-or-v1-" + "a" * 42))
        errors = validate_config(cfg)
        assert not any("OPENROUTER_API_KEY" in e.field for e in errors)

    def test_dependencies_validated(self):
        cfg = self._make_config(
            enable_rrf_fusion=True,
            reranker_engine=RerankerEngine.CROSS_ENCODER,
        )
        errors = validate_config(cfg)
        assert errors == []

    def test_all_percentage_fields_validated(self):
        """All percentage fields are validated."""
        cfg = self._make_config(
            search_score_threshold=2.0,
            fusion_ranking_threshold=2.0,
            cross_encoder_score_threshold=-1.0,
            smart_replace_threshold=-1.0,
            smart_replace_min_similarity=5.0,
            tantivy_compaction_threshold_ratio=3.0,
            recency_decay_rate=10.0,
        )
        errors = validate_config(cfg)
        percentage_fields = {
            "SEARCH_SCORE_THRESHOLD",
            "FUSION_RANKING_THRESHOLD",
            "CROSS_ENCODER_SCORE_THRESHOLD",
            "SMART_REPLACE_THRESHOLD",
            "SMART_REPLACE_MIN_SIMILARITY",
            "TANTIVY_COMPACTION_THRESHOLD_RATIO",
            "RECENCY_DECAY_RATE",
        }
        error_fields = {e.field for e in errors}
        assert percentage_fields.issubset(error_fields)

    def test_all_positive_int_fields_validated(self):
        """All positive int fields are validated."""
        cfg = self._make_config(
            search_limit=0,
            remove_search_limit=0,
            fusion_rrf_k=0,
            rerank_max_concurrency=0,
            cross_encoder_batch_size=0,
            cross_encoder_max_length=0,
            add_max_concurrency=0,
            embedding_batch_size=0,
            embedding_cache_size=0,
        )
        errors = validate_config(cfg)
        int_fields = {
            "SEARCH_LIMIT",
            "REMOVE_SEARCH_LIMIT",
            "FUSION_RRF_K",
            "RERANK_MAX_CONCURRENCY",
            "CROSS_ENCODER_BATCH_SIZE",
            "CROSS_ENCODER_MAX_LENGTH",
            "ADD_MAX_CONCURRENCY",
            "EMBEDDING_BATCH_SIZE",
            "EMBEDDING_CACHE_SIZE",
        }
        error_fields = {e.field for e in errors}
        assert int_fields.issubset(error_fields)
