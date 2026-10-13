"""Unit tests for reflectlog/utility/utility.py module."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
import os
from typing import Any
from unittest.mock import MagicMock, patch

import claude_agent_sdk as claude
import pytest

from reflectlog.utility.types import OAUTH_TOKEN_PREFIX


async def _empty_query(**kwargs: Any) -> AsyncIterator[object]:
    _ = kwargs
    if False:
        yield None


class TestGetClaudeCodeApiKey:
    """Tests for get_claude_code_api_key() function."""

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_returns_credential_from_retriever(
        self, mock_get_retriever: MagicMock
    ) -> None:
        """Should return credential when retriever finds one."""
        mock_retriever = MagicMock()
        mock_retriever.get_credential.return_value = "sk-ant-test-key-123"
        mock_get_retriever.return_value = mock_retriever

        from reflectlog.utility.utility import get_claude_code_api_key

        result = get_claude_code_api_key()

        assert result == "sk-ant-test-key-123"
        mock_retriever.get_credential.assert_called_once()

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_returns_none_when_no_retriever(
        self, mock_get_retriever: MagicMock
    ) -> None:
        """Should return None when get_platform_retriever returns None."""
        mock_get_retriever.return_value = None

        from reflectlog.utility.utility import get_claude_code_api_key

        result = get_claude_code_api_key()

        assert result is None

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_returns_none_on_exception(self, mock_get_retriever: MagicMock) -> None:
        """Should return None silently on any exception."""
        mock_get_retriever.side_effect = RuntimeError("Platform error")

        from reflectlog.utility.utility import get_claude_code_api_key

        result = get_claude_code_api_key()

        assert result is None

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_returns_none_when_retriever_raises(
        self, mock_get_retriever: MagicMock
    ) -> None:
        """Should return None when retriever.get_credential() raises."""
        mock_retriever = MagicMock()
        mock_retriever.get_credential.side_effect = OSError("Keychain error")
        mock_get_retriever.return_value = mock_retriever

        from reflectlog.utility.utility import get_claude_code_api_key

        result = get_claude_code_api_key()

        assert result is None

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_verbose_mode_prints_to_stderr(
        self, mock_get_retriever: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should print debug message to stderr when verbose=True."""
        mock_retriever = MagicMock()
        mock_retriever.get_credential.return_value = "sk-ant-test"
        mock_get_retriever.return_value = mock_retriever

        from reflectlog.utility.utility import get_claude_code_api_key

        _ = get_claude_code_api_key(verbose=True)

        captured = capsys.readouterr()
        assert "credential store" in captured.err

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_verbose_false_no_stderr_output(
        self, mock_get_retriever: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should not print to stderr when verbose=False."""
        mock_retriever = MagicMock()
        mock_retriever.get_credential.return_value = "sk-ant-test"
        mock_get_retriever.return_value = mock_retriever

        from reflectlog.utility.utility import get_claude_code_api_key

        _ = get_claude_code_api_key(verbose=False)

        captured = capsys.readouterr()
        assert captured.err == ""

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_returns_none_when_credential_is_none(
        self, mock_get_retriever: MagicMock
    ) -> None:
        """Should return None when retriever returns None credential."""
        mock_retriever = MagicMock()
        mock_retriever.get_credential.return_value = None
        mock_get_retriever.return_value = mock_retriever

        from reflectlog.utility.utility import get_claude_code_api_key

        result = get_claude_code_api_key()

        assert result is None


class TestGetAnthropicApiKey:
    """Tests for get_anthropic_api_key() function."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env-key"}, clear=False)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_returns_from_env_var(self, mock_get_key: MagicMock) -> None:
        """Should return API key from ANTHROPIC_API_KEY env var with source=env."""
        from reflectlog.utility.utility import get_anthropic_api_key

        result = get_anthropic_api_key()

        assert result is not None
        assert result["api_key"] == "sk-ant-env-key"
        assert result["source"] == "env"
        # Should not attempt keychain retrieval
        mock_get_key.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_falls_back_to_keychain(self, mock_get_key: MagicMock) -> None:
        """Should fall back to keychain when env var not set."""
        mock_get_key.return_value = "sk-ant-keychain-key"

        from reflectlog.utility.utility import get_anthropic_api_key

        result = get_anthropic_api_key()

        assert result is not None
        assert result["api_key"] == "sk-ant-keychain-key"
        assert result["source"] == "claude-code"
        mock_get_key.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_returns_none_when_neither_available(self, mock_get_key: MagicMock) -> None:
        """Should return None when no env var and no keychain credential."""
        mock_get_key.return_value = None

        from reflectlog.utility.utility import get_anthropic_api_key

        result = get_anthropic_api_key()

        assert result is None

    @patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "sk-ant-env-key"},
        clear=False,
    )
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_env_var_takes_priority_over_keychain(
        self, mock_get_key: MagicMock
    ) -> None:
        """Env var should take priority even if keychain has a key."""
        mock_get_key.return_value = "sk-ant-keychain-key"

        from reflectlog.utility.utility import get_anthropic_api_key

        result = get_anthropic_api_key()

        assert result is not None
        assert result["api_key"] == "sk-ant-env-key"
        assert result["source"] == "env"
        mock_get_key.assert_not_called()


class TestInitCredentials:
    """Tests for init_credentials() function."""

    @patch.dict(
        os.environ,
        {"ANTHROPIC_AUTH_TOKEN": "sk-ant-oat01-existing-token-abc"},
        clear=False,
    )
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_returns_existing_env_token(self, mock_get_key: MagicMock) -> None:
        """Should return existing ANTHROPIC_AUTH_TOKEN from env."""
        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=False)

        assert result == "sk-ant-oat01-existing-token-abc"
        mock_get_key.assert_not_called()

    @patch.dict(
        os.environ,
        {"ANTHROPIC_AUTH_TOKEN": "sk-ant-oat01-existing-token-abc"},
        clear=False,
    )
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_verbose_prints_existing_token_prefix(
        self,
        mock_get_key: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should print truncated token when verbose=True and token exists."""
        from reflectlog.utility.utility import init_credentials

        _ = init_credentials(verbose=True)

        captured = capsys.readouterr()
        assert "ANTHROPIC_AUTH_TOKEN:" in captured.out
        assert "<set>" in captured.out
        assert "sk-ant-oat01-existin" not in captured.out

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_retrieves_oauth_from_keychain(self, mock_get_key: MagicMock) -> None:
        """Should retrieve OAuth token from keychain and set env var."""
        oauth_token = f"{OAUTH_TOKEN_PREFIX}keychain-oauth-token-xyz"
        mock_get_key.return_value = oauth_token

        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=False)

        assert result == oauth_token
        assert os.environ.get("ANTHROPIC_AUTH_TOKEN") == oauth_token

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_sets_env_var_when_oauth_found_in_keychain(
        self, mock_get_key: MagicMock
    ) -> None:
        """Should set ANTHROPIC_AUTH_TOKEN env var when OAuth token found."""
        oauth_token = f"{OAUTH_TOKEN_PREFIX}keychain-token-abc123"
        mock_get_key.return_value = oauth_token

        from reflectlog.utility.utility import init_credentials

        _ = init_credentials(verbose=False)

        assert os.environ["ANTHROPIC_AUTH_TOKEN"] == oauth_token

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_verbose_prints_keychain_oauth_token(
        self,
        mock_get_key: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should print OAuth token info when verbose and found in keychain."""
        oauth_token = f"{OAUTH_TOKEN_PREFIX}keychain-oauth-token-xyz"
        mock_get_key.return_value = oauth_token

        from reflectlog.utility.utility import init_credentials

        _ = init_credentials(verbose=True)

        captured = capsys.readouterr()
        assert "ANTHROPIC_AUTH_TOKEN:" in captured.out
        assert "retrieved from keychain" in captured.out

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_handles_non_oauth_token(
        self,
        mock_get_key: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should handle regular API key (not OAuth) from keychain."""
        mock_get_key.return_value = "sk-ant-api-regular-key-123"

        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=True)

        assert result is None
        captured = capsys.readouterr()
        assert "not an OAuth token" in captured.out
        assert "OAuth tokens are only available" in captured.out
        assert "SDK will use stored credentials" in captured.out

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_handles_non_oauth_token_verbose_false(
        self, mock_get_key: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should not print when verbose=False and non-OAuth token found."""
        mock_get_key.return_value = "sk-ant-api-regular-key-123"

        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=False)

        assert result is None
        captured = capsys.readouterr()
        assert captured.out == ""

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_no_token_anywhere(
        self,
        mock_get_key: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should return None and print status when no token found."""
        mock_get_key.return_value = None

        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=True)

        assert result is None
        captured = capsys.readouterr()
        assert "ANTHROPIC_AUTH_TOKEN: <not set>" in captured.out
        assert "SDK will use stored credentials" in captured.out

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_no_token_verbose_false(
        self, mock_get_key: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should not print when verbose=False and no token found."""
        mock_get_key.return_value = None

        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=False)

        assert result is None
        captured = capsys.readouterr()
        assert captured.out == ""


class TestGenerateContent:
    """Tests for generate_content() async function."""

    async def test_tolerates_all_frame_variants(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reflectlog.utility.utility import generate_content

        frames: list[claude.Message] = [
            claude.SystemMessage("init", {}),
            claude.UserMessage("User input"),
            claude.StreamEvent("event", "session", {}),
            claude.RateLimitEvent(claude.RateLimitInfo("allowed"), "rate", "session"),
            claude.ConversationResetMessage("conversation", "reset", "session"),
            claude.AssistantMessage(
                [
                    claude.ThinkingBlock("Thinking", "signature"),
                    claude.TextBlock(" Hello"),
                    claude.ToolUseBlock("tool", "Read", {}),
                    claude.TextBlock(" world! "),
                ],
                "test-model",
            ),
            claude.ResultMessage("success", 0, 0, False, 1, "session"),
        ]

        async def query(
            *, prompt: str, options: claude.ClaudeAgentOptions
        ) -> AsyncGenerator[claude.Message]:
            for frame in frames:
                yield frame

        monkeypatch.setattr("reflectlog.utility.utility._claude.query", query)
        assert await generate_content("Test") == "Hello world!"

    @pytest.mark.parametrize("error", [False, True])
    async def test_closes_owned_stream(
        self, monkeypatch: pytest.MonkeyPatch, error: bool
    ) -> None:
        from reflectlog.utility.utility import generate_content

        closed = False

        async def query(
            *, prompt: str, options: claude.ClaudeAgentOptions
        ) -> AsyncGenerator[claude.Message]:
            nonlocal closed
            try:
                if error:
                    yield claude.AssistantMessage([], "test-model", error="rate_limit")
                else:
                    yield claude.AssistantMessage(
                        [claude.TextBlock("Text")], "test-model"
                    )
                # An error must close the generator before this suspension.
                if error:
                    await asyncio.Event().wait()
            finally:
                closed = True

        monkeypatch.setattr("reflectlog.utility.utility._claude.query", query)
        if error:
            with pytest.raises(RuntimeError, match="^Query failed: rate_limit$"):
                await asyncio.wait_for(generate_content("Test"), timeout=1.0)
        else:
            # No ResultMessage is required for success.
            assert await generate_content("Test") == "Text"
        assert closed

    async def test_closes_stream_on_external_cancellation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reflectlog.utility.utility import generate_content

        entered = asyncio.Event()
        closed = False

        async def query(
            *, prompt: str, options: claude.ClaudeAgentOptions
        ) -> AsyncGenerator[claude.Message]:
            nonlocal closed
            try:
                yield claude.AssistantMessage([], "test-model")
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed = True

        monkeypatch.setattr("reflectlog.utility.utility._claude.query", query)
        task = asyncio.create_task(generate_content("Test"))
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed

    @patch("reflectlog.utility.utility._claude.query")
    @patch("reflectlog.utility.utility._claude.ClaudeAgentOptions")
    async def test_returns_text_from_assistant_message(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should return concatenated text from TextBlock content."""
        from claude_agent_sdk import AssistantMessage, TextBlock

        from reflectlog.utility.utility import generate_content

        mock_block = TextBlock(text="Hello, world!")
        mock_message = AssistantMessage(content=[mock_block], model="test-model")

        # Make query return an async iterator
        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_message

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Say hello")

        assert result == "Hello, world!"

    @patch("reflectlog.utility.utility._claude.query")
    @patch("reflectlog.utility.utility._claude.ClaudeAgentOptions")
    async def test_concatenates_multiple_text_blocks(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should concatenate text from multiple TextBlocks."""
        from claude_agent_sdk import AssistantMessage, TextBlock

        from reflectlog.utility.utility import generate_content

        mock_block1 = TextBlock(text="Hello, ")
        mock_block2 = TextBlock(text="world!")
        mock_message = AssistantMessage(
            content=[mock_block1, mock_block2], model="test-model"
        )

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_message

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Say hello")

        assert result == "Hello, world!"

    @patch("reflectlog.utility.utility._claude.query")
    @patch("reflectlog.utility.utility._claude.ClaudeAgentOptions")
    async def test_raises_runtime_error_on_error_message(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should raise RuntimeError when ResultMessage is an error."""
        from claude_agent_sdk import ResultMessage

        from reflectlog.utility.utility import generate_content

        mock_error = ResultMessage("api_error", 0, 0, True, 1, "session")

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_error

        mock_query.side_effect = mock_query_iter

        with pytest.raises(RuntimeError, match="Query failed: api_error"):
            _ = await generate_content("Will fail")

    @patch("reflectlog.utility.utility._claude.query")
    @patch("reflectlog.utility.utility._claude.ClaudeAgentOptions")
    async def test_ignores_non_error_result_message(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should ignore ResultMessage that is not an error."""
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        from reflectlog.utility.utility import generate_content

        mock_block = TextBlock(text="Content")
        mock_assistant = AssistantMessage(content=[mock_block], model="test-model")
        mock_result = ResultMessage("success", 0, 0, False, 1, "session")

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_assistant
            yield mock_result

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Test prompt")

        assert result == "Content"

    @patch("reflectlog.utility.utility._claude.query")
    @patch("reflectlog.utility.utility._claude.ClaudeAgentOptions")
    async def test_ignores_non_text_blocks(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should skip content blocks that are not TextBlock instances."""
        from claude_agent_sdk import AssistantMessage, TextBlock, ThinkingBlock

        from reflectlog.utility.utility import generate_content

        mock_text_block = TextBlock(text="Valid text")
        mock_other_block = ThinkingBlock(thinking="Thinking", signature="signature")
        mock_message = AssistantMessage(
            content=[mock_other_block, mock_text_block], model="test-model"
        )

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_message

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Test")

        assert result == "Valid text"

    @patch("reflectlog.utility.utility._claude.query")
    @patch("reflectlog.utility.utility._claude.ClaudeAgentOptions")
    async def test_strips_result_text(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should strip leading/trailing whitespace from result."""
        from claude_agent_sdk import AssistantMessage, TextBlock

        from reflectlog.utility.utility import generate_content

        mock_block = TextBlock(text="  Padded content  ")
        mock_message = AssistantMessage(content=[mock_block], model="test-model")

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_message

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Test")

        assert result == "Padded content"

    @patch("reflectlog.utility.utility._claude.query")
    @patch("reflectlog.utility.utility._claude.ClaudeAgentOptions")
    async def test_empty_response_returns_empty_string(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should return empty string when no messages received."""
        from reflectlog.utility.utility import generate_content

        mock_query.side_effect = _empty_query

        result = await generate_content("Test")

        assert result == ""

    @patch("reflectlog.utility.utility._claude.query")
    @patch("reflectlog.utility.utility._claude.ClaudeAgentOptions")
    async def test_passes_options_correctly(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should pass correct options to ClaudeAgentOptions."""
        from reflectlog.utility.utility import generate_content

        mock_query.side_effect = _empty_query

        _ = await generate_content(
            "Test prompt",
            model="claude-sonnet-4-5-20250929",
            system_prompt="Be helpful",
            allowed_tools=["read", "write"],
        )

        mock_options_cls.assert_called_once_with(
            model="claude-sonnet-4-5-20250929",
            system_prompt="Be helpful",
            tools=[],
            allowed_tools=["read", "write"],
            permission_mode="default",
        )

    @patch("reflectlog.utility.utility._claude.query")
    @patch("reflectlog.utility.utility._claude.ClaudeAgentOptions")
    async def test_default_options_empty_tools(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should default to empty allowed_tools list."""
        from reflectlog.utility.utility import generate_content

        mock_query.side_effect = _empty_query

        _ = await generate_content("Test")

        mock_options_cls.assert_called_once_with(
            model=None,
            system_prompt=None,
            tools=[],
            allowed_tools=[],
            permission_mode="default",
        )

    @patch("reflectlog.utility.utility._claude.query")
    async def test_default_query_options_disable_builtin_tools(
        self, mock_query: MagicMock
    ) -> None:
        """Should hand query options with all built-in tools disabled."""
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions

        from reflectlog.utility.utility import generate_content

        captured_options: list[ClaudeAgentOptions] = []
        mock_message = AssistantMessage(content=[], model="test-model")

        async def mock_query_iter(*, prompt: str, options: ClaudeAgentOptions) -> Any:
            captured_options.append(options)
            yield mock_message

        mock_query.side_effect = mock_query_iter

        _ = await generate_content("Test")

        assert len(captured_options) == 1
        assert captured_options[0].tools == []
        assert captured_options[0].allowed_tools == []

    @patch("reflectlog.utility.utility._claude.query")
    async def test_allowed_tools_do_not_enable_builtin_tools(
        self, mock_query: MagicMock
    ) -> None:
        """Should preserve auto-approval settings without enabling built-in tools."""
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions

        from reflectlog.utility.utility import generate_content

        captured_options: list[ClaudeAgentOptions] = []
        mock_message = AssistantMessage(content=[], model="test-model")

        async def mock_query_iter(*, prompt: str, options: ClaudeAgentOptions) -> Any:
            captured_options.append(options)
            yield mock_message

        mock_query.side_effect = mock_query_iter

        _ = await generate_content("Test", allowed_tools=["Read"])

        assert len(captured_options) == 1
        assert captured_options[0].tools == []
        assert captured_options[0].allowed_tools == ["Read"]

    @patch("reflectlog.utility.utility._claude.query")
    @patch("reflectlog.utility.utility._claude.ClaudeAgentOptions")
    async def test_multiple_assistant_messages(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should concatenate text across multiple AssistantMessages."""
        from claude_agent_sdk import AssistantMessage, TextBlock

        from reflectlog.utility.utility import generate_content

        mock_block1 = TextBlock(text="Part 1. ")
        mock_msg1 = AssistantMessage(content=[mock_block1], model="test-model")
        mock_block2 = TextBlock(text="Part 2.")
        mock_msg2 = AssistantMessage(content=[mock_block2], model="test-model")

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_msg1
            yield mock_msg2

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Multi-part")

        assert result == "Part 1. Part 2."

    @patch("reflectlog.utility.utility._claude.query")
    @patch("reflectlog.utility.utility._claude.ClaudeAgentOptions")
    async def test_passes_prompt_to_query(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should pass prompt to query function."""
        from reflectlog.utility.utility import generate_content

        mock_options_instance = MagicMock()
        mock_options_cls.return_value = mock_options_instance

        mock_query.side_effect = _empty_query

        _ = await generate_content("My test prompt")

        mock_query.assert_called_once_with(
            prompt="My test prompt",
            options=mock_options_instance,
        )
