"""Graceful degradation path tests.

These tests verify that the system degrades gracefully when external
dependencies (LLM providers, APIs) fail or are unavailable.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.add_phases import SmartReplacementPhase
from reflectlog.application.utils.security import SecretString
from reflectlog.core.types import ISemanticSearchEngine
from reflectlog.infrastructure.smart_replacer import (
    OpenAIReplacementProvider,
    SmartReplacer,
    SmartReplacerConfig,
)
from tests.unit.infrastructure.test_smart_replacer import RecordingLogger


@pytest.mark.parametrize(
    "scenario", ["accepted", "rejected", "candidate_error", "search_error"]
)
async def test_candidate_checks_keep_memory_and_reply_private(scenario: str) -> None:
    sentinel = "PRIVATE_SENTINEL_CANDIDATE"
    logger = RecordingLogger()
    phase = SmartReplacementPhase(
        Mock(spec=ISemanticSearchEngine),
        Config(
            workspace_id="test",
            openrouter_api_key=SecretString("key"),
            admin_password=SecretString("test-admin-password"),
        ),
        logger,
        None,
    )
    replacer = SmartReplacer(
        config=SmartReplacerConfig("key", "https://example.com", "model", enabled=False)
    )
    with (
        patch.object(
            SmartReplacer,
            "check_replacement",
            autospec=True,
            return_value=(scenario == "accepted", 0.9, sentinel),
            side_effect=RuntimeError(sentinel)
            if scenario == "candidate_error"
            else None,
        ),
        patch.object(
            SmartReplacementPhase,
            "_search_candidates",
            autospec=True,
            side_effect=RuntimeError(sentinel),
        ),
    ):
        result = (
            await phase._check_for_replacement(sentinel, replacer)
            if scenario == "search_error"
            else await phase._check_candidates_with_llm(
                [(sentinel, 0.9)], sentinel, replacer
            )
        )
    assert logger.records
    assert sentinel not in repr(logger.records)
    assert sentinel not in repr(logger.exception_text)
    assert all(not record[3] for record in logger.records)
    if scenario == "accepted":
        assert len(result) == 1
        assert result[0].reason == sentinel
        assert result[0].old_memory == sentinel
        assert result[0].new_memory == sentinel
    else:
        assert result == []
    if scenario in {"accepted", "rejected"}:
        assert logger.records[0][2]
        assert logger.records[0][2]["reason_length"] == len(sentinel)
        assert logger.records[0][2]["confidence"] == 0.9
    else:
        assert logger.records[0][2]
        assert logger.records[0][2]["exception_type"] == "RuntimeError"


@pytest.mark.unit
class TestSmartReplacerDegradation:
    """Tests for SmartReplacer graceful degradation."""

    @pytest.mark.asyncio
    async def test_replacer_adds_normally_on_llm_failure(self):
        """Test that memory is added normally when LLM replacement check fails."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Mock the LLM client to raise an exception
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API unavailable")
        )

        # Mock logger
        provider._logger = Mock()

        # Call replacement detection - should gracefully fallback to "no replacement"
        prompt = (
            "Should 'I don't like cats anymore, I like dogs' replace 'I like cats'?"
        )
        result = await provider.detect_replacement(
            prompt=prompt,
            max_retries=3,
            retry_delay=1.0,
        )

        # Should return (should_replace=False, confidence=0.0, reason)
        should_replace, confidence, reason = result
        assert should_replace is False
        assert confidence == 0.0
        assert "failed" in reason.lower() or "error" in reason.lower()

        # Should have logged a warning
        provider._logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_replacer_with_retry_success_after_transient_failure(self):
        """Test that transient failures are retried successfully."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Mock the LLM client to fail twice, then succeed
        # Use ConnectionError which is a retryable exception type
        call_count = 0

        async def mock_create(*args: str, **kwargs: str) -> Mock:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Temporary API error")
            # Return valid response on third attempt
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[
                0
            ].message.content = '{"should_replace": true, "confidence": 0.9, "reason": "User changed preference"}'
            return mock_response

        # Create mock client with proper structure
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
        provider._client = mock_client

        # Mock logger
        provider._logger = Mock()

        # Call replacement detection - should retry and succeed
        prompt = (
            "Should 'I don't like cats anymore, I like dogs' replace 'I like cats'?"
        )
        result = await provider.detect_replacement(
            prompt=prompt,
            max_retries=3,
            retry_delay=0.01,  # Use short delay for faster test
        )

        # Should succeed after retries
        should_replace, confidence, reason = result
        assert should_replace is True
        assert confidence == 0.9
        assert "changed preference" in reason.lower()

        # Should have made 3 attempts (2 failures + 1 success)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_replacer_invalid_json_fallback(self):
        """Test that invalid JSON responses are handled gracefully."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Mock the LLM client to return invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is not valid JSON"

        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock logger
        provider._logger = Mock()

        # Call replacement detection - should fallback to no replacement
        prompt = "Should 'new' replace 'old'?"
        result = await provider.detect_replacement(
            prompt=prompt,
            max_retries=3,
            retry_delay=1.0,
        )

        # Should return safe defaults
        should_replace, confidence, _reason = result
        assert should_replace is False
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_replacer_missing_fields_fallback(self):
        """Test that missing fields in JSON are handled gracefully."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Test cases for missing fields with expected results
        test_cases = [
            (
                '{"should_replace": true}',
                True,
                0.0,
            ),  # Missing confidence and reason - should_replace is True
            (
                '{"confidence": 0.9}',
                False,
                0.9,
            ),  # Missing should_replace - defaults to False
            (
                '{"reason": "User changed mind"}',
                False,
                0.0,
            ),  # Missing should_replace and confidence
            ("{}", False, 0.0),  # Empty JSON - all defaults
        ]

        for json_response, expected_should_replace, expected_confidence in test_cases:
            # Mock the LLM client
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json_response

            provider._client = AsyncMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            # Mock logger
            provider._logger = Mock()

            # Call replacement detection
            prompt = "Should 'new' replace 'old'?"
            result = await provider.detect_replacement(
                prompt=prompt,
                max_retries=3,
                retry_delay=1.0,
            )

            # Should return expected values
            should_replace, confidence, _reason = result
            assert should_replace == expected_should_replace, (
                f"For JSON '{json_response}': expected {expected_should_replace}, got {should_replace}"
            )
            assert confidence == expected_confidence, (
                f"For JSON '{json_response}': expected confidence {expected_confidence}, got {confidence}"
            )

    @pytest.mark.asyncio
    async def test_replacer_confidence_clamping(self):
        """Test that confidence values are clamped to [0, 1]."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Test cases for out-of-range confidence values
        test_cases = [
            ('{"should_replace": true, "confidence": 1.5, "reason": "test"}', 1.0),
            ('{"should_replace": true, "confidence": -0.5, "reason": "test"}', 0.0),
            ('{"should_replace": true, "confidence": 999.0, "reason": "test"}', 1.0),
        ]

        for json_response, expected_clamped in test_cases:
            # Mock the LLM client
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json_response

            provider._client = AsyncMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            # Mock logger
            provider._logger = Mock()

            # Call replacement detection
            prompt = "Should 'new' replace 'old'?"
            result = await provider.detect_replacement(
                prompt=prompt,
                max_retries=3,
                retry_delay=1.0,
            )

            # Confidence should be clamped
            assert result[1] == expected_clamped

    @pytest.mark.asyncio
    async def test_replacer_timeout_handling(self):
        """Test that timeouts are handled gracefully."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
            timeout=0.001,  # Very short timeout
        )

        # Mock the LLM client to take too long
        async def mock_create(*args: str, **kwargs: str) -> Mock:
            await asyncio.sleep(1)  # Sleep longer than timeout
            mock_response = Mock()
            mock_response.choices = [Mock()]
            return mock_response

        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(side_effect=mock_create)

        # Mock logger
        provider._logger = Mock()

        # Call replacement detection - should timeout and fallback
        prompt = "Should 'new' replace 'old'?"
        result = await provider.detect_replacement(
            prompt=prompt,
            max_retries=1,  # Only retry once to speed up test
            retry_delay=0.1,
        )

        # Should return safe defaults (likely due to timeout)
        should_replace, _confidence, _reason = result
        # Should default to False on error
        assert should_replace is False
