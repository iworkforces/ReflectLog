"""Coordinated shutdown handoff and native signal registration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import os
from pathlib import Path
import signal
import socket
import sys
import threading
from typing import cast
from unittest.mock import MagicMock, PropertyMock, patch

from fastmcp import Client
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.mcp_server import FastMCPServer
from reflectlog.application.utils.security import SecretString
from reflectlog.core.config_adapters import ConfigAdapter
from reflectlog.core.enums import EmbedderProvider, RerankerEngine
from reflectlog.core.exceptions import StorageError
from reflectlog.infrastructure.embeddings.qwen3_embedding import LangchainQwenEmbeddings
from reflectlog.infrastructure.storage_coordinator import PortalockerStorageCoordinator
from reflectlog.server import _start_server


def test_lease_handoff_after_close(tmp_path: Path) -> None:
    import logging

    from reflectlog.application.config.settings import Config
    from reflectlog.application.memory.manager import MemoryManager
    from reflectlog.application.utils.logging import StructuredLogger
    from reflectlog.application.utils.security import SecretString
    from reflectlog.core.enums import LlmProvider, RerankerEngine

    logger = StructuredLogger(logging.getLogger("shutdown-handoff"))
    coordinator = PortalockerStorageCoordinator(str(tmp_path), timeout=1.0)
    config = Config(
        workspace_id="ws",
        openrouter_api_key=SecretString("test"),
        admin_password=SecretString("test-admin-password"),
        tantivy_index_path_template=str(tmp_path / "{workspace_id}" / "tantivy"),
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="openai/text-embedding-3-large",
        embedding_dims=3072,
        enable_smart_replace=False,
        llm_provider=LlmProvider.OPENAI,
        reranker_engine=RerankerEngine.NONE,
        embedding_cache_enabled=False,
        eager_initialization=False,
    )
    with (
        patch.object(
            ConfigAdapter, "usearch_index_path", new_callable=PropertyMock
        ) as index_path,
        patch("reflectlog.application.memory.manager.USearchEngine") as usearch_cls,
        patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"),
        patch("reflectlog.application.memory.manager.TantivyEngine"),
    ):
        index_path.return_value = str(tmp_path / "ws" / "usearch")
        usearch_cls.return_value = MagicMock()
        first = MemoryManager(config, logger, coordinator=coordinator)
        first.close()
        with pytest.raises(StorageError, match="closed"):
            first.get_all()
        second = MemoryManager(config, logger, coordinator=coordinator)
        assert second._coordinator is coordinator
        second.close()


def test_graceful_signal_registers_posix_handlers() -> None:
    registered: dict[int, object] = {}

    def _capture(signum: int, handler: object) -> None:
        registered[signum] = handler

    with (
        patch("reflectlog.server.signal.signal", side_effect=_capture),
        patch("reflectlog.server._server_cls") as server_cls,
    ):
        server_cls.return_value = lambda: MagicMock()
        _ = _start_server(sys.stderr, 0.0, {})
    assert signal.SIGINT in registered
    assert signal.SIGTERM in registered
    if sys.platform == "win32":
        assert signal.SIGBREAK in registered


def test_second_signal_restores_default() -> None:
    registered: dict[int, Callable[[int, object | None], None]] = {}

    def _capture(signum: int, handler: object) -> object:
        registered[signum] = cast(Callable[[int, object | None], None], handler)
        return None

    with (
        patch("reflectlog.server.signal.signal", side_effect=_capture),
        patch("reflectlog.server.signal.raise_signal") as raise_signal,
        patch("reflectlog.server._server_cls") as server_cls,
        patch("reflectlog.server.sys.exit"),
    ):
        server_cls.return_value = lambda: MagicMock()
        _ = _start_server(sys.stderr, 0.0, {})
        handler = registered[signal.SIGINT]
        handler(signal.SIGINT, None)
        handler(signal.SIGINT, None)
        raise_signal.assert_called_once_with(signal.SIGINT)
        assert registered[signal.SIGINT] is signal.SIG_DFL
        assert registered[signal.SIGTERM] is signal.SIG_DFL
        if sys.platform == "win32":
            assert registered[signal.SIGBREAK] is signal.SIG_DFL


@pytest.mark.integration
@pytest.mark.asyncio
async def test_signal_shutdown_drains_real_storage_and_reopens(
    tmp_path: Path, set_env_vars: None
) -> None:
    config = Config(
        workspace_id="",
        openrouter_api_key=SecretString("test-key"),
        admin_password=SecretString("test-admin-password"),
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="openai/text-embedding-3-small",
        embedding_dims=128,
        reranker_engine=RerankerEngine.NONE,
        enable_smart_replace=False,
        embedding_cache_enabled=False,
        eager_initialization=False,
        tantivy_index_path_template=str(tmp_path / "{workspace_id}" / "tantivy"),
    )
    handlers: dict[int, Callable[[int, object], None]] = {}
    completed = asyncio.Event()

    def capture_signal(number: int, handler: Callable[[int, object], None]) -> None:
        handlers[number] = handler

    def vector(_self: LangchainQwenEmbeddings, _text: str) -> list[float]:
        return [1.0, *([0.0] * 127)]

    def vectors(_self: LangchainQwenEmbeddings, texts: list[str]) -> list[list[float]]:
        return [vector(_self, text) for text in texts]

    async def async_vectors(
        embedder: LangchainQwenEmbeddings, texts: list[str]
    ) -> list[list[float]]:
        return vectors(embedder, texts)

    entered = threading.Event()
    release = threading.Event()
    close_calls = 0
    with (
        patch.object(LangchainQwenEmbeddings, "embed_documents", vectors),
        patch.object(LangchainQwenEmbeddings, "aembed_documents", async_vectors),
        patch.object(LangchainQwenEmbeddings, "embed_query", vector),
        patch("reflectlog.server.signal.signal", side_effect=capture_signal),
        patch(
            "reflectlog.server.signal.raise_signal",
            side_effect=lambda _number: completed.set(),
        ),
        patch(
            "reflectlog.server._server_cls", return_value=lambda: FastMCPServer(config)
        ),
    ):
        server = _start_server(sys.stderr, 0.0, {})
        client_factory = cast("Callable[[object], Client]", Client)
        async with client_factory(server.mcp) as client:
            _ = await client.call_tool(
                "add_memories",
                {"workspace_id": "durable", "memories": ["shutdown survives restart"]},
            )

            async with server._registry.acquire("durable") as manager:
                original_close = manager.close

            def close_with_barrier() -> None:
                nonlocal close_calls
                close_calls += 1
                entered.set()
                assert release.wait(timeout=10)
                original_close()

            with patch.object(manager, "close", side_effect=close_with_barrier):
                async with server._registry.acquire("durable"):
                    asyncio.get_running_loop().call_soon(
                        handlers[signal.SIGTERM], signal.SIGTERM, None
                    )
                    await asyncio.sleep(0)
                    assert close_calls == 0
                try:
                    assert await asyncio.wait_for(asyncio.to_thread(entered.wait), 10)
                    waiter = asyncio.create_task(server.aclose())
                    await asyncio.sleep(0)
                    waiter.cancel()
                finally:
                    release.set()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
            with pytest.raises(RuntimeError, match="closed"):
                async with server._registry.acquire("other"):
                    pass
        assert close_calls == 1
        await asyncio.wait_for(completed.wait(), 10)

        reopened = FastMCPServer(config)
        async with client_factory(reopened.mcp) as client:
            page = await client.call_tool(
                "get_all_memories", {"workspace_id": "durable"}
            )
            assert "shutdown survives restart" in page.content[0].text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_installed_http_signal_persists(
    set_env_vars: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX SIGTERM path")
    monkeypatch.setenv("NUMBA_WARMUP", "false")
    monkeypatch.setenv("REFLECTLOG_MCP_AUTH_TOKEN", "shutdown-test-token")
    monkeypatch.setenv("MCP_HOST", "127.0.0.1")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "reflectlog.server",
        "--transport",
        "http",
        "--port",
        str(port),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    try:
        startup = await asyncio.wait_for(
            process.stdout.readuntil(b"Application startup complete"), 30
        )
        process.send_signal(signal.SIGTERM)
        output = startup + await asyncio.wait_for(process.stdout.read(), 15)
        assert b"Received SIGTERM" in output
        assert b"Server shutdown complete" in output
        assert await asyncio.wait_for(process.wait(), 2) == -signal.SIGTERM
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()


def _child_wait_for_term(path: str) -> None:
    import signal as sig
    import time

    def _handler(signum: int, frame: object) -> None:
        Path(path).write_text(f"got-{signum}", encoding="utf-8")
        raise SystemExit(0)

    _ = sig.signal(sig.SIGTERM, _handler)
    _ = sig.signal(sig.SIGINT, _handler)
    Path(path).write_text("ready", encoding="utf-8")
    for _ in range(200):
        time.sleep(0.05)


def test_graceful_signal_sigterm_child(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX SIGTERM path")
    marker = tmp_path / "marker.txt"
    ctx = __import__("multiprocessing").get_context("spawn")
    child = ctx.Process(target=_child_wait_for_term, args=(str(marker),))
    child.start()
    try:
        for _ in range(300):
            if marker.exists() and marker.read_text(encoding="utf-8") == "ready":
                break
            __import__("time").sleep(0.05)
        assert marker.exists()
        child_pid = child.pid
        assert child_pid is not None
        os.kill(child_pid, signal.SIGTERM)
        child.join(timeout=10.0)
        assert child.exitcode == 0
        assert marker.read_text(encoding="utf-8") == f"got-{signal.SIGTERM}"
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5.0)
