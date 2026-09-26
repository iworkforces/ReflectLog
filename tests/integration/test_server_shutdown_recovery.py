"""Coordinated shutdown handoff and native signal registration."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import signal
import sys
from typing import cast
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from reflectlog.core.config_adapters import ConfigAdapter
from reflectlog.core.enums import EmbedderProvider
from reflectlog.core.exceptions import StorageError
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
        for _ in range(100):
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
