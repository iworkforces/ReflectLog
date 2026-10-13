"""Unit tests for SmartReplacer and replacement providers."""

import asyncio
from collections.abc import AsyncGenerator
import json
import traceback
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import claude_agent_sdk as claude
from openai import AsyncOpenAI
import pytest

from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.memory_store import MemoryStore
from reflectlog.infrastructure.smart_replacer import (
    AnthropicReplacementProvider,
    OpenAIReplacementProvider,
    ReplacementDecision,
    SmartReplacer,
    SmartReplacerConfig,
    create_replacement_provider,
)

if TYPE_CHECKING:
    from pathlib import Path


def create_mock_logger() -> IStructuredLogger:
    """Create a properly typed mock logger for testing."""
    return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))


class RecordingLogger:
    """Capture all structured logging without hiding exception rendering."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any] | None, bool]] = []
        self.exception_text: list[str] = []

    def debug(
        self, message: str, extra: dict[str, Any] | None = None, exc_info: bool = False
    ) -> None:
        self.records.append(("debug", message, extra, exc_info))
        if exc_info:
            self.exception_text.append(traceback.format_exc())

    def info(
        self, message: str, extra: dict[str, Any] | None = None, exc_info: bool = False
    ) -> None:
        self.records.append(("info", message, extra, exc_info))
        if exc_info:
            self.exception_text.append(traceback.format_exc())

    def warning(
        self, message: str, extra: dict[str, Any] | None = None, exc_info: bool = False
    ) -> None:
        self.records.append(("warning", message, extra, exc_info))
        if exc_info:
            self.exception_text.append(traceback.format_exc())

    def error(
        self, message: str, extra: dict[str, Any] | None = None, exc_info: bool = False
    ) -> None:
        self.records.append(("error", message, extra, exc_info))
        if exc_info:
            self.exception_text.append(traceback.format_exc())

    def is_enabled_for(self, level: int) -> bool:
        return True


@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        *(("should_replace", value) for value in ["false", "true", 0, 1, None]),
        *(
            ("confidence", value)
            for value in [
                True,
                False,
                "0.9",
                None,
                float("nan"),
                float("inf"),
                -float("inf"),
            ]
        ),
        *(("reason", value) for value in [123, None, ["x"]]),
    ],
)
async def test_providers_reject_malformed_decisions(
    provider_name: str, field: str, value: object
) -> None:
    decision = {"should_replace": True, "confidence": 0.9, "reason": "private reply"}
    decision[field] = value
    logger = RecordingLogger()
    with patch("reflectlog.utility.utility.init_credentials", autospec=True):
        provider = (
            OpenAIReplacementProvider(
                "key", "https://example.com", "model", logger=logger
            )
            if provider_name == "openai"
            else AnthropicReplacementProvider(logger=logger)
        )
    with (
        patch.object(
            OpenAIReplacementProvider,
            "_call_llm_with_structured_output",
            autospec=True,
            return_value=decision,
        ),
        patch(
            "reflectlog.utility.utility.generate_content",
            autospec=True,
            return_value=json.dumps(decision),
        ),
    ):
        result = await provider.detect_replacement("private memory", 2, 0)
    assert result[:2] == (False, 0.0)
    assert result[2].startswith("Error")
    assert "private" not in repr(logger.records)
    assert len(logger.records) >= 2


@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ({}, (False, 0.0, "No reason provided")),
        (
            {
                "should_replace": True,
                "confidence": 0.9,
                "reason": "unchanged",
                "extra": 1,
            },
            (True, 0.9, "unchanged"),
        ),
        *(
            (
                {"should_replace": True, "confidence": value},
                (True, expected, "No reason provided"),
            )
            for value, expected in [(1.7, 1.0), (-0.2, 0.0), (5, 1.0), (10**400, 1.0)]
        ),
    ],
)
async def test_providers_preserve_valid_decisions(
    provider_name: str, decision: dict[str, object], expected: tuple[bool, float, str]
) -> None:
    with patch("reflectlog.utility.utility.init_credentials", autospec=True):
        provider = (
            OpenAIReplacementProvider("key", "https://example.com", "model")
            if provider_name == "openai"
            else AnthropicReplacementProvider()
        )
    with (
        patch.object(
            OpenAIReplacementProvider,
            "_call_llm_with_structured_output",
            autospec=True,
            return_value=decision,
        ),
        patch(
            "reflectlog.utility.utility.generate_content",
            autospec=True,
            return_value=json.dumps(decision),
        ),
    ):
        result = await provider.detect_replacement("prompt", 1, 0)
    assert result == expected


@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
@pytest.mark.parametrize("raw", [[], None, "model reply"])
async def test_providers_reject_non_object_root(
    provider_name: str, raw: object
) -> None:
    with patch("reflectlog.utility.utility.init_credentials", autospec=True):
        provider = (
            OpenAIReplacementProvider("key", "https://example.com", "model")
            if provider_name == "openai"
            else AnthropicReplacementProvider()
        )
    with (
        patch.object(
            OpenAIReplacementProvider,
            "_call_llm_with_structured_output",
            autospec=True,
            return_value=raw,
        ),
        patch(
            "reflectlog.utility.utility.generate_content",
            autospec=True,
            return_value=json.dumps(raw),
        ),
    ):
        result = await provider.detect_replacement("prompt", 1, 0)
    assert result[:2] == (False, 0.0)
    assert result[2].startswith("Error")
    assert "model reply" not in result[2]


@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
@pytest.mark.parametrize("scenario", ["accepted", "rejected", "invalid", "json", "sdk"])
async def test_replacement_flow_keeps_text_private(
    provider_name: str, scenario: str, tmp_path: Path
) -> None:
    sentinel = "PRIVATE_SENTINEL_MODEL_MEMORY"
    logger = RecordingLogger()
    decision = {
        "should_replace": scenario != "rejected",
        "confidence": 0.9,
        "reason": sentinel,
    }
    if scenario == "invalid":
        decision["confidence"] = sentinel
    error = (
        json.JSONDecodeError(sentinel, sentinel, 0)
        if scenario == "json"
        else RuntimeError(sentinel)
    )
    with patch("reflectlog.utility.utility.init_credentials", autospec=True):
        provider = (
            OpenAIReplacementProvider(
                "key", "https://example.com", "model", logger=logger
            )
            if provider_name == "openai"
            else AnthropicReplacementProvider(logger=logger)
        )
    with patch(
        "reflectlog.infrastructure.smart_replacer.create_replacement_provider",
        autospec=True,
        return_value=provider,
    ):
        replacer = SmartReplacer(
            config=SmartReplacerConfig(
                "key",
                "https://example.com",
                "model",
                threshold=0.0,
                max_retries=2,
                retry_delay=0,
            ),
            logger=logger,
        )
    with (
        patch.object(
            OpenAIReplacementProvider,
            "_call_llm_with_structured_output",
            autospec=True,
            return_value=decision,
            side_effect=error if scenario in {"json", "sdk"} else None,
        ),
        patch(
            "reflectlog.utility.utility.generate_content",
            autospec=True,
            return_value=sentinel if scenario == "json" else json.dumps(decision),
            side_effect=error if scenario == "sdk" else None,
        ),
    ):
        result = await replacer.check_replacement(sentinel, sentinel)
    assert logger.records
    assert sentinel not in repr(logger.records)
    assert sentinel not in repr(logger.exception_text)
    assert all(not record[3] for record in logger.records)
    if scenario in {"accepted", "rejected"}:
        assert result == (scenario == "accepted", 0.9, sentinel)
        assert any(
            record[2] and record[2].get("reason_length") == len(sentinel)
            for record in logger.records
        )
    else:
        assert result[:2] == (False, 0.0)
        assert sentinel not in result[2]
    if scenario == "accepted":
        store = MemoryStore(str(tmp_path / "archive.db"))
        try:
            store.begin_replacement_transition(
                old_memory_id=1,
                workspace_id="test",
                old_content="old memory",
                new_content="new memory",
                reason=result[2],
                confidence=result[1],
            )
            assert store.get_archived("test")[0].reason == sentinel
        finally:
            store.close()


@pytest.mark.parametrize("scenario", ["transport", "fallback_transport"])
async def test_openai_transport_failures_keep_text_private(scenario: str) -> None:
    """SDK errors can echo the prompt; the real client boundary must not leak it."""
    sentinel = "PRIVATE_SENTINEL_TRANSPORT"
    fallback_message = "Model doesn't support structured outputs, falling back"
    logger = RecordingLogger()
    provider = OpenAIReplacementProvider(
        "key", "https://example.com", "model", logger=logger
    )
    attempts = 2
    errors = (
        [RuntimeError(f"connection reset {sentinel}") for _ in range(attempts)]
        if scenario == "transport"
        else [
            RuntimeError(f"{message} {sentinel}")
            for _ in range(attempts)
            for message in ("json_schema unsupported", "upstream failed")
        ]
    )
    async with AsyncOpenAI(api_key="key") as client:
        with (
            patch.object(provider, "_get_client", autospec=True, return_value=client),
            patch.object(
                client.chat.completions,
                "create",
                new=AsyncMock(spec=client.chat.completions.create, side_effect=errors),
            ) as create,
        ):
            result = await provider.detect_replacement(sentinel, attempts, 0)
    assert result == (False, 0.0, "Error: replacement detection failed (RuntimeError)")
    assert create.await_count == len(errors)
    assert sentinel not in repr(logger.records)
    assert sentinel not in repr(logger.exception_text)
    assert all(not record[3] for record in logger.records)
    failures = [
        record[2]
        for record in logger.records
        if record[1] == "OpenAI replacement detection failed"
    ]
    assert [failure and failure["exception_type"] for failure in failures] == [
        "RuntimeError"
    ] * attempts
    fallbacks = [record for record in logger.records if fallback_message in record[1]]
    assert len(fallbacks) == (attempts if scenario == "fallback_transport" else 0)


class TestReplacementDecision:
    """Test ReplacementDecision Pydantic schema."""

    def test_valid_decision_should_replace(self) -> None:
        """Test valid replacement decision."""
        decision = ReplacementDecision(
            should_replace=True,
            confidence=0.85,
            reason="Same topic with updated preference",
        )
        assert decision.should_replace is True
        assert decision.confidence == 0.85
        assert decision.reason == "Same topic with updated preference"

    def test_valid_decision_no_replace(self) -> None:
        """Test valid no-replacement decision."""
        decision = ReplacementDecision(
            should_replace=False,
            confidence=0.3,
            reason="Different topics",
        )
        assert decision.should_replace is False
        assert decision.confidence == 0.3

    def test_valid_confidence_zero(self) -> None:
        """Test minimum valid confidence."""
        decision = ReplacementDecision(
            should_replace=False,
            confidence=0.0,
            reason="No match",
        )
        assert decision.confidence == 0.0

    def test_valid_confidence_one(self) -> None:
        """Test maximum valid confidence."""
        decision = ReplacementDecision(
            should_replace=True,
            confidence=1.0,
            reason="Exact replacement",
        )
        assert decision.confidence == 1.0

    def test_invalid_confidence_below_zero(self) -> None:
        """Test that confidence below 0.0 is rejected."""
        with pytest.raises(ValueError):
            ReplacementDecision(
                should_replace=False,
                confidence=-0.1,
                reason="Invalid",
            )

    def test_invalid_confidence_above_one(self) -> None:
        """Test that confidence above 1.0 is rejected."""
        with pytest.raises(ValueError):
            ReplacementDecision(
                should_replace=True,
                confidence=1.1,
                reason="Invalid",
            )

    def test_json_schema_generation(self) -> None:
        """Test that JSON schema is generated correctly."""
        schema = ReplacementDecision.model_json_schema()
        assert "should_replace" in schema.get("properties", {})
        assert "confidence" in schema.get("properties", {})
        assert "reason" in schema.get("properties", {})

    def test_json_serialization(self) -> None:
        """Test JSON serialization round-trip."""
        decision = ReplacementDecision(
            should_replace=True,
            confidence=0.75,
            reason="Updated preference",
        )
        json_str = decision.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["should_replace"] is True
        assert parsed["confidence"] == 0.75
        assert parsed["reason"] == "Updated preference"


class TestSmartReplacerConfig:
    """Test SmartReplacerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
        )

        assert config.threshold == 0.7
        assert config.enabled is True
        assert config.timeout == 30.0
        assert config.provider == "openai"

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://custom.url/api/v1",
            model="custom/model",
            threshold=0.8,
            enabled=False,
            timeout=60.0,
            provider="anthropic",
        )

        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.url/api/v1"
        assert config.model == "custom/model"
        assert config.threshold == 0.8
        assert config.enabled is False
        assert config.timeout == 60.0
        assert config.provider == "anthropic"

    def test_from_app_config(self) -> None:
        """Test factory method from application config."""
        mock_app_config = MagicMock()
        mock_app_config.llm_api_key = "api-key"
        mock_app_config.llm_api_base_url = "https://openrouter.ai/api/v1"
        mock_app_config.llm_model = "x-ai/grok-4.1-fast"
        mock_app_config.smart_replace_threshold = 0.8
        mock_app_config.enable_smart_replace = True
        mock_app_config.smart_replace_max_retries = 3
        mock_app_config.smart_replace_retry_delay = 1.0
        mock_app_config.llm_provider = "openai"

        config = SmartReplacerConfig.from_config(mock_app_config)

        assert config.api_key == "api-key"
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.model == "x-ai/grok-4.1-fast"
        assert config.threshold == 0.8
        assert config.enabled is True
        assert config.provider == "openai"

    def test_from_app_config_anthropic_provider(self) -> None:
        """Test factory method with anthropic provider."""
        mock_app_config = MagicMock()
        mock_app_config.llm_api_key = "api-key"
        mock_app_config.llm_api_base_url = "https://openrouter.ai/api/v1"
        mock_app_config.llm_model = "claude-sonnet-4-20250514"
        mock_app_config.smart_replace_threshold = 0.7
        mock_app_config.enable_smart_replace = True
        mock_app_config.smart_replace_max_retries = 3
        mock_app_config.smart_replace_retry_delay = 1.0
        mock_app_config.llm_provider = "anthropic"

        config = SmartReplacerConfig.from_config(mock_app_config)

        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"


class TestCreateReplacementProvider:
    """Test create_replacement_provider factory function."""

    def test_forwards_anthropic_timeout(self) -> None:
        config = SmartReplacerConfig(
            "key", "url", "model", timeout=0.125, provider="anthropic"
        )
        with (
            patch("reflectlog.utility.utility.init_credentials"),
            patch(
                "reflectlog.infrastructure.smart_replacer.AnthropicReplacementProvider",
                autospec=True,
            ) as constructor,
        ):
            create_replacement_provider(config)
        constructor.assert_called_once_with(model="model", timeout=0.125, logger=None)

    def test_create_openai_provider(self) -> None:
        """Test creating OpenAI provider."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            provider="openai",
        )

        with patch(
            "reflectlog.infrastructure.llm_provider_base.AsyncOpenAI"
        ) as mock_async_client:
            provider = create_replacement_provider(config)

            assert isinstance(provider, OpenAIReplacementProvider)
            mock_async_client.assert_not_called()

    def test_create_anthropic_provider(self) -> None:
        """Test creating Anthropic provider."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="claude-sonnet-4-20250514",
            provider="anthropic",
        )

        with patch("reflectlog.utility.utility.init_credentials") as mock_init:
            provider = create_replacement_provider(config)

            assert isinstance(provider, AnthropicReplacementProvider)
            mock_init.assert_called_once_with(verbose=False)

    def test_create_invalid_provider_raises_error(self) -> None:
        """Test that invalid provider raises ValueError."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test-model",
            provider="invalid",
        )

        with pytest.raises(ValueError, match="Unsupported provider"):
            create_replacement_provider(config)


class TestOpenAIReplacementProvider:
    """Test OpenAIReplacementProvider."""

    @pytest.fixture
    def mock_provider(self) -> OpenAIReplacementProvider:
        """Create a mocked OpenAI provider instance."""
        with patch("reflectlog.infrastructure.llm_provider_base.AsyncOpenAI"):
            provider = OpenAIReplacementProvider(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="x-ai/grok-4.1-fast",
            )
            return provider

    @pytest.mark.asyncio
    async def test_detect_replacement_success(
        self, mock_provider: OpenAIReplacementProvider
    ) -> None:
        """Test successful replacement detection."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": True,
                "confidence": 0.9,
                "reason": "Same topic with updated preference",
            }
        )

        mock_provider._client = AsyncMock()
        mock_provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        should_replace, confidence, reason = await mock_provider.detect_replacement(
            prompt="test prompt",
            max_retries=3,
            retry_delay=1.0,
        )

        assert should_replace is True
        assert confidence == 0.9
        assert reason == "Same topic with updated preference"

    @pytest.mark.asyncio
    async def test_detect_replacement_no_replace(
        self, mock_provider: OpenAIReplacementProvider
    ) -> None:
        """Test no replacement needed."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": False,
                "confidence": 0.2,
                "reason": "Different topics entirely",
            }
        )

        mock_provider._client = AsyncMock()
        mock_provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        should_replace, confidence, reason = await mock_provider.detect_replacement(
            prompt="test prompt",
            max_retries=3,
            retry_delay=1.0,
        )

        assert should_replace is False
        assert confidence == 0.2
        assert reason == "Different topics entirely"

    @pytest.mark.asyncio
    async def test_detect_replacement_clamps_confidence(
        self, mock_provider: OpenAIReplacementProvider
    ) -> None:
        """Test that out-of-range confidence is clamped to 0-1."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": True,
                "confidence": 1.5,
                "reason": "Clamped",
            }
        )

        mock_provider._client = AsyncMock()
        mock_provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        _, confidence, _ = await mock_provider.detect_replacement(
            prompt="test prompt",
            max_retries=3,
            retry_delay=1.0,
        )
        assert confidence == 1.0

    @pytest.mark.asyncio
    async def test_fallback_on_json_schema_not_supported(
        self, mock_provider: OpenAIReplacementProvider
    ) -> None:
        """Test fallback to json_object mode when structured outputs not supported."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": False,
                "confidence": 0.3,
                "reason": "Different topics",
            }
        )

        mock_provider._client = AsyncMock()
        mock_provider._client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("json_schema is not supported for this model"),
                mock_response,
            ]
        )

        mock_logger = create_mock_logger()
        mock_provider._logger = mock_logger

        should_replace, confidence, _reason = await mock_provider.detect_replacement(
            prompt="test prompt",
            max_retries=3,
            retry_delay=1.0,
        )

        # Verify result after fallback
        assert should_replace is False
        assert confidence == 0.3

        # Verify second call used json_object mode
        assert mock_provider._client.chat.completions.create.call_count == 2
        second_call_kwargs = (
            mock_provider._client.chat.completions.create.call_args_list[1][1]
        )
        assert second_call_kwargs["response_format"]["type"] == "json_object"

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_returns_safe_defaults(
        self, mock_provider: OpenAIReplacementProvider
    ) -> None:
        """Test that exhausted retries return safe defaults."""
        mock_provider._client = AsyncMock()
        mock_provider._client.chat.completions.create = AsyncMock(
            side_effect=Exception("Network timeout error")
        )

        mock_logger = create_mock_logger()
        mock_provider._logger = mock_logger

        should_replace, confidence, reason = await mock_provider.detect_replacement(
            prompt="test prompt",
            max_retries=2,
            retry_delay=0.01,  # Fast retries for testing
        )

        assert should_replace is False
        assert confidence == 0.0
        assert "Error" in reason


class TestAnthropicReplacementProvider:
    """Test AnthropicReplacementProvider."""

    @pytest.mark.parametrize("max_retries", [2, 3])
    async def test_stalled_stream_times_out_every_attempt(
        self, monkeypatch: pytest.MonkeyPatch, max_retries: int
    ) -> None:
        attempts = 0
        closed = 0
        delays: list[float] = []

        async def query(
            *, prompt: str, options: claude.ClaudeAgentOptions
        ) -> AsyncGenerator[claude.Message]:
            nonlocal attempts, closed
            attempts += 1
            try:
                yield claude.SystemMessage("init", {})
                await asyncio.Event().wait()
            finally:
                closed += 1

        async def sleep(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr("reflectlog.utility.utility._claude.query", query)
        monkeypatch.setattr(
            "reflectlog.infrastructure.smart_replacer.asyncio.sleep", sleep
        )
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(timeout=0.05)
        result = await asyncio.wait_for(
            provider.detect_replacement("Test", max_retries, 1.0), timeout=2.0
        )
        assert result == (
            False,
            0.0,
            "Error: replacement detection failed (TimeoutError)",
        )
        assert attempts == closed == max_retries
        assert delays == [float(2**i) for i in range(max_retries - 1)]

    async def test_timeout_then_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = 0
        closed = 0

        async def query(
            *, prompt: str, options: claude.ClaudeAgentOptions
        ) -> AsyncGenerator[claude.Message]:
            nonlocal attempts, closed
            attempts += 1
            try:
                yield claude.SystemMessage("init", {})
                if attempts == 1:
                    await asyncio.Event().wait()
                yield claude.AssistantMessage(
                    [
                        claude.TextBlock(
                            '{"should_replace": true, "confidence": 0.9, "reason": "Updated"}'
                        )
                    ],
                    "test-model",
                )
            finally:
                closed += 1

        monkeypatch.setattr("reflectlog.utility.utility._claude.query", query)
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(timeout=0.05)
        result = await asyncio.wait_for(
            provider.detect_replacement("Test", 2, 0.0), timeout=2.0
        )
        assert result == (True, 0.9, "Updated")
        assert attempts == closed == 2

    async def test_external_cancellation_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entered = asyncio.Event()
        attempts = 0
        closed = 0
        delays: list[float] = []

        async def query(
            *, prompt: str, options: claude.ClaudeAgentOptions
        ) -> AsyncGenerator[claude.Message]:
            nonlocal attempts, closed
            attempts += 1
            try:
                yield claude.SystemMessage("init", {})
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed += 1

        async def sleep(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr("reflectlog.utility.utility._claude.query", query)
        monkeypatch.setattr(
            "reflectlog.infrastructure.smart_replacer.asyncio.sleep", sleep
        )
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(timeout=30.0)
        task = asyncio.create_task(provider.detect_replacement("Test", 3, 1.0))
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert attempts == closed == 1
        assert delays == []

    def test_extract_json_pure_json(self) -> None:
        """Test extracting pure JSON from response."""
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(model="claude-sonnet-4-20250514")

        result = provider._extract_json_from_response(
            '{"should_replace": true, "confidence": 0.9, "reason": "Test"}'
        )

        assert result["should_replace"] is True
        assert result["confidence"] == 0.9
        assert result["reason"] == "Test"

    def test_extract_json_markdown_code_block(self) -> None:
        """Test extracting JSON from markdown code block."""
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(model="claude-sonnet-4-20250514")

        response = """Here's the analysis:

```json
{"should_replace": true, "confidence": 0.85, "reason": "Updated preference"}
```

That's my assessment."""

        result = provider._extract_json_from_response(response)

        assert result["should_replace"] is True
        assert result["confidence"] == 0.85

    def test_extract_json_embedded_json(self) -> None:
        """Test extracting embedded JSON from text."""
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(model="claude-sonnet-4-20250514")

        response = 'Result: {"should_replace": false, "confidence": 0.3} done.'

        result = provider._extract_json_from_response(response)

        assert result["should_replace"] is False
        assert result["confidence"] == 0.3

    def test_extract_json_markdown_block_invalid_json_falls_through(self) -> None:
        """Test markdown code block with invalid JSON falls through to embedded strategy."""
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(model="claude-sonnet-4-20250514")

        response = """Here's my analysis:

```json
{not valid json at all!!!}
```

But the result is {"should_replace": true, "confidence": 0.8, "reason": "Updated"}"""

        result = provider._extract_json_from_response(response)

        assert result["should_replace"] is True
        assert result["confidence"] == 0.8

    def test_extract_json_markdown_block_invalid_json_no_fallback(self) -> None:
        """Test markdown code block with invalid JSON and no valid embedded JSON raises."""
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(model="claude-sonnet-4-20250514")

        response = """Here's my analysis:

```json
{this is not valid json}
```

No valid JSON anywhere else either."""

        with pytest.raises(ValueError, match="Could not extract JSON"):
            provider._extract_json_from_response(response)

    def test_extract_json_embedded_invalid_json_continues(self) -> None:
        """Test embedded JSON patterns with invalid JSON continue to next match."""
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(model="claude-sonnet-4-20250514")

        response = 'Result: {invalid json} but also {"should_replace": false, "confidence": 0.4, "reason": "No match"}'

        result = provider._extract_json_from_response(response)

        assert result["should_replace"] is False
        assert result["confidence"] == 0.4

    def test_extract_json_all_embedded_invalid_raises(self) -> None:
        """Test all embedded JSON patterns invalid raises ValueError."""
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(model="claude-sonnet-4-20250514")

        response = "Result: {bad json} and also {more bad} end"

        with pytest.raises(ValueError, match="Could not extract JSON"):
            provider._extract_json_from_response(response)

    def test_extract_json_fails_raises_error(self) -> None:
        """Test that extraction failure raises ValueError."""
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(model="claude-sonnet-4-20250514")

        with pytest.raises(ValueError, match="Could not extract JSON"):
            provider._extract_json_from_response("No JSON here at all!")

    @pytest.mark.asyncio
    async def test_detect_replacement_success(self) -> None:
        """Test successful replacement detection with Anthropic."""
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(model="claude-sonnet-4-20250514")

        with patch("reflectlog.utility.utility.generate_content") as mock_generate:
            mock_generate.return_value = json.dumps(
                {
                    "should_replace": True,
                    "confidence": 0.88,
                    "reason": "Same topic updated",
                }
            )

            should_replace, confidence, reason = await provider.detect_replacement(
                prompt="test prompt",
                max_retries=3,
                retry_delay=1.0,
            )

            assert should_replace is True
            assert confidence == 0.88
            assert reason == "Same topic updated"
            mock_generate.assert_called_once_with(
                prompt="test prompt",
                model="claude-sonnet-4-20250514",
                allowed_tools=[],
            )

    @pytest.mark.asyncio
    async def test_detect_replacement_with_markdown_response(self) -> None:
        """Test replacement detection with markdown-wrapped response."""
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(model="claude-sonnet-4-20250514")

        with patch("reflectlog.utility.utility.generate_content") as mock_generate:
            mock_generate.return_value = """Analysis complete:

```json
{"should_replace": true, "confidence": 0.92, "reason": "Preference changed"}
```"""

            should_replace, confidence, _reason = await provider.detect_replacement(
                prompt="test prompt",
                max_retries=3,
                retry_delay=1.0,
            )

            assert should_replace is True
            assert confidence == 0.92

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_returns_safe_defaults(self) -> None:
        """Test that exhausted retries return safe defaults."""
        with patch("reflectlog.utility.utility.init_credentials"):
            provider = AnthropicReplacementProvider(model="claude-sonnet-4-20250514")

        with patch("reflectlog.utility.utility.generate_content") as mock_generate:
            mock_generate.side_effect = Exception("API Error")

            mock_logger = create_mock_logger()
            provider._logger = mock_logger

            should_replace, confidence, reason = await provider.detect_replacement(
                prompt="test prompt",
                max_retries=2,
                retry_delay=0.01,
            )

            assert should_replace is False
            assert confidence == 0.0
            assert "Error" in reason


class TestSmartReplacerInitialization:
    """Test SmartReplacer initialization."""

    def test_initialization_creates_openai_provider_when_enabled(self) -> None:
        """Test that initialization creates OpenAI provider when enabled."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            enabled=True,
            provider="openai",
        )

        with patch(
            "reflectlog.infrastructure.llm_provider_base.AsyncOpenAI"
        ) as mock_async_client:
            replacer = SmartReplacer(config=config)

            mock_async_client.assert_not_called()
            assert replacer._provider is not None
            assert isinstance(replacer._provider, OpenAIReplacementProvider)

    def test_initialization_creates_anthropic_provider_when_enabled(self) -> None:
        """Test that initialization creates Anthropic provider when enabled."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="claude-sonnet-4-20250514",
            enabled=True,
            provider="anthropic",
        )

        with patch("reflectlog.utility.utility.init_credentials") as mock_init:
            replacer = SmartReplacer(config=config)

            mock_init.assert_called_once_with(verbose=False)
            assert replacer._provider is not None
            assert isinstance(replacer._provider, AnthropicReplacementProvider)

    def test_initialization_no_provider_when_disabled(self) -> None:
        """Test that no provider is created when disabled."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            enabled=False,
        )

        with patch(
            "reflectlog.infrastructure.llm_provider_base.AsyncOpenAI"
        ) as mock_async_client:
            replacer = SmartReplacer(config=config)

            mock_async_client.assert_not_called()
            assert replacer._provider is None


class TestCheckReplacement:
    """Test check_replacement method."""

    @pytest.fixture
    def mock_replacer(self) -> SmartReplacer:
        """Create a mocked SmartReplacer instance."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            threshold=0.7,
            provider="openai",
        )

        with patch("reflectlog.infrastructure.llm_provider_base.AsyncOpenAI"):
            replacer = SmartReplacer(config=config)
            return replacer

    @pytest.mark.asyncio
    async def test_check_replacement_should_replace(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test successful replacement detection."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": True,
                "confidence": 0.9,
                "reason": "Same topic with updated preference",
            }
        )

        provider = cast(OpenAIReplacementProvider, mock_replacer._provider)
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="I don't like cats anymore",
            existing_memory="I like cats",
        )

        assert should_replace is True
        assert confidence == 0.9
        assert reason == "Same topic with updated preference"

    @pytest.mark.asyncio
    async def test_check_replacement_no_replace(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test no replacement needed."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": False,
                "confidence": 0.2,
                "reason": "Different topics entirely",
            }
        )

        provider = cast(OpenAIReplacementProvider, mock_replacer._provider)
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="I like dogs",
            existing_memory="The weather is nice today",
        )

        assert should_replace is False
        assert confidence == 0.2
        assert reason == "Different topics entirely"

    @pytest.mark.asyncio
    async def test_check_replacement_below_threshold(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test that replacement is not triggered when confidence is below threshold."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": True,
                "confidence": 0.5,  # Below threshold of 0.7
                "reason": "Possibly related",
            }
        )

        provider = cast(OpenAIReplacementProvider, mock_replacer._provider)
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        should_replace, confidence, _reason = await mock_replacer.check_replacement(
            new_memory="I like dogs",
            existing_memory="I like cats",
        )

        # should_replace should be False because confidence is below threshold
        assert should_replace is False
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_check_replacement_disabled(self) -> None:
        """Test that disabled replacer returns no replacement."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            enabled=False,
        )

        with patch("reflectlog.infrastructure.llm_provider_base.AsyncOpenAI"):
            replacer = SmartReplacer(config=config)

        should_replace, confidence, reason = await replacer.check_replacement(
            new_memory="test",
            existing_memory="test2",
        )

        assert should_replace is False
        assert confidence == 0.0
        assert reason == "Smart replacement disabled"

    @pytest.mark.asyncio
    async def test_check_replacement_provider_not_initialized(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test handling when provider is not initialized."""
        mock_replacer._provider = None

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="test",
            existing_memory="test2",
        )

        assert should_replace is False
        assert confidence == 0.0
        assert reason == "Provider not initialized"

    @pytest.mark.asyncio
    async def test_check_replacement_api_error(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test handling of API error with retry behavior."""
        provider = cast(OpenAIReplacementProvider, mock_replacer._provider)
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        mock_logger = create_mock_logger()
        mock_replacer.logger = mock_logger

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="test", existing_memory="test2"
        )

        assert should_replace is False
        assert confidence == 0.0
        assert "Error" in reason

    @pytest.mark.asyncio
    async def test_check_replacement_json_decode_error(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test handling of JSONDecodeError from provider."""
        mock_provider = AsyncMock()
        mock_provider.detect_replacement = AsyncMock(
            side_effect=json.JSONDecodeError("Expecting value", "doc", 0)
        )
        mock_replacer._provider = mock_provider

        mock_logger = create_mock_logger()
        mock_replacer.logger = mock_logger

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="test", existing_memory="test2"
        )

        assert should_replace is False
        assert confidence == 0.0
        assert "JSON parse error" in reason
        mock_logger.warning.assert_called_once()  # type: ignore

    @pytest.mark.asyncio
    async def test_check_replacement_json_decode_error_no_logger(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test JSONDecodeError handling without logger."""
        mock_provider = AsyncMock()
        mock_provider.detect_replacement = AsyncMock(
            side_effect=json.JSONDecodeError("Expecting value", "doc", 0)
        )
        mock_replacer._provider = mock_provider
        mock_replacer.logger = None

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="test", existing_memory="test2"
        )

        assert should_replace is False
        assert confidence == 0.0
        assert "JSON parse error" in reason

    @pytest.mark.asyncio
    async def test_check_replacement_generic_exception(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test handling of generic Exception from provider."""
        mock_provider = AsyncMock()
        mock_provider.detect_replacement = AsyncMock(
            side_effect=RuntimeError("Unexpected failure")
        )
        mock_replacer._provider = mock_provider

        mock_logger = create_mock_logger()
        mock_replacer.logger = mock_logger

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="test", existing_memory="test2"
        )

        assert should_replace is False
        assert confidence == 0.0
        assert reason == "Error: replacement check failed (RuntimeError)"
        assert "Unexpected failure" not in reason
        mock_logger.warning.assert_called_once()  # type: ignore

    @pytest.mark.asyncio
    async def test_check_replacement_generic_exception_no_logger(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test generic Exception handling without logger."""
        mock_provider = AsyncMock()
        mock_provider.detect_replacement = AsyncMock(
            side_effect=RuntimeError("Unexpected failure")
        )
        mock_replacer._provider = mock_provider
        mock_replacer.logger = None

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="test", existing_memory="test2"
        )

        assert should_replace is False
        assert confidence == 0.0
        assert reason == "Error: replacement check failed (RuntimeError)"
        assert "Unexpected failure" not in reason


class TestSmartReplacerIntegration:
    """Integration-style tests for full replacement flow."""

    @pytest.mark.asyncio
    async def test_full_replacement_flow_openai_replace(self) -> None:
        """Test complete replacement flow with OpenAI provider."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            threshold=0.7,
            provider="openai",
        )

        with patch("reflectlog.infrastructure.llm_provider_base.AsyncOpenAI"):
            replacer = SmartReplacer(config=config, logger=create_mock_logger())

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(
                {
                    "should_replace": True,
                    "confidence": 0.9,
                    "reason": "Updated preference about cats",
                }
            )

            provider = cast(OpenAIReplacementProvider, replacer._provider)
            provider._client = AsyncMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            should_replace, confidence, reason = await replacer.check_replacement(
                new_memory="I don't like cats anymore, I prefer dogs",
                existing_memory="I like cats",
            )

            assert should_replace is True
            assert confidence == 0.9
            assert "Updated preference" in reason

    @pytest.mark.asyncio
    async def test_full_replacement_flow_anthropic_replace(self) -> None:
        """Test complete replacement flow with Anthropic provider."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="claude-sonnet-4-20250514",
            threshold=0.7,
            provider="anthropic",
        )

        with patch("reflectlog.utility.utility.init_credentials"):
            replacer = SmartReplacer(config=config, logger=create_mock_logger())

            with patch("reflectlog.utility.utility.generate_content") as mock_generate:
                mock_generate.return_value = json.dumps(
                    {
                        "should_replace": True,
                        "confidence": 0.85,
                        "reason": "Updated location",
                    }
                )

                should_replace, confidence, reason = await replacer.check_replacement(
                    new_memory="I moved to San Francisco",
                    existing_memory="I live in New York",
                )

                assert should_replace is True
                assert confidence == 0.85
                assert "Updated location" in reason

    @pytest.mark.asyncio
    async def test_full_replacement_flow_no_replace(self) -> None:
        """Test complete replacement flow with no replacement."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            threshold=0.7,
            provider="openai",
        )

        with patch("reflectlog.infrastructure.llm_provider_base.AsyncOpenAI"):
            replacer = SmartReplacer(config=config, logger=create_mock_logger())

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(
                {
                    "should_replace": False,
                    "confidence": 0.2,
                    "reason": "Different topics - cats vs weather",
                }
            )

            provider = cast(OpenAIReplacementProvider, replacer._provider)
            provider._client = AsyncMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            should_replace, confidence, _reason = await replacer.check_replacement(
                new_memory="The weather is nice today",
                existing_memory="I like cats",
            )

            assert should_replace is False
            assert confidence == 0.2
