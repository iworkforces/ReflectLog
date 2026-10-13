"""Anthropic replacement deadlines through MCP tools and real storage engines."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import claude_agent_sdk as claude
import pytest

from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.core.config_adapters import ConfigAdapter
from reflectlog.infrastructure.smart_replacer import SmartReplacerConfig
from tests.integration.test_memory_manager_usearch import (
    MockEmbedder,
    create_usearch_config,
)


@pytest.mark.integration
@pytest.mark.parametrize("stalled", [False, True], ids=["replace", "deadline-fallback"])
async def test_anthropic_replacement_real_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stalled: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    config = replace(
        create_usearch_config(str(tmp_path)),
        enable_smart_replace=True,
        llm_provider="anthropic",
        smart_replace_max_retries=2,
        smart_replace_retry_delay=0.0,
        # MockEmbedder uses hash-seeded random vectors, not semantic embeddings.
        search_score_threshold=-1.0,
        smart_replace_min_similarity=-1.0,
    )
    original_from_config = SmartReplacerConfig.from_config
    replacement_config = replace(
        original_from_config(ConfigAdapter(config)), timeout=0.05
    )
    attempts = 0
    closed = 0

    async def query(
        *, prompt: str, options: claude.ClaudeAgentOptions
    ) -> AsyncGenerator[claude.Message]:
        nonlocal attempts, closed
        attempts += 1
        try:
            yield claude.SystemMessage("init", {})
            if stalled:
                await asyncio.Event().wait()
            yield claude.AssistantMessage(
                [
                    claude.TextBlock(
                        '{"should_replace": true, "confidence": 0.9, "reason": "Changed preference"}'
                    )
                ],
                "test-model",
            )
            yield claude.ResultMessage("success", 0, 0, False, 1, "session")
        finally:
            closed += 1

    monkeypatch.setattr("reflectlog.utility.utility._claude.query", query)
    old = "My preferred editor is Vim."
    new = "My preferred editor is Emacs, not Vim."
    with (
        patch("reflectlog.utility.utility.init_credentials"),
        patch(
            "reflectlog.application.memory.manager.LangchainQwenEmbeddings",
            return_value=MockEmbedder(),
        ),
        patch(
            "reflectlog.application.memory.manager.SmartReplacerConfig.from_config",
            return_value=replacement_config,
        ),
    ):
        server = FastMCPServer(config)
        try:
            initial = await server._add_memories([old], config.workspace_id)
            assert initial["stored_count"] == 1
            added = await server._add_memories([new], config.workspace_id)
            assert added["stored_count"] == 1
            assert added["replaced_count"] == (0 if stalled else 1)
            assert attempts == (config.smart_replace_max_retries if stalled else 1)
            assert closed == attempts
            page = await server._get_all_memories(config.workspace_id)
            records = page["records"]
            assert isinstance(records, list)
            memories: set[str] = set()
            for record in records:
                assert isinstance(record, dict)
                memory = record["memory"]
                assert isinstance(memory, str)
                memories.add(memory)
            assert memories == ({old, new} if stalled else {new})
        finally:
            await server._registry.close()
