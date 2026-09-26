"""Real-process writer coordination for disjoint and duplicate adds."""

from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import os
from pathlib import Path

import numpy as np
import pytest

from reflectlog.core.enums import EmbedderProvider
from reflectlog.core.types import Embeddings
from reflectlog.infrastructure.storage_coordinator import PortalockerStorageCoordinator
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_USEARCH_CONCURRENCY_TESTS") != "1",
    reason="Set RUN_USEARCH_CONCURRENCY_TESTS=1 to run USearch concurrency tests",
)


class _HashEmbedder(Embeddings):
    def __init__(self, dims: int = 32) -> None:
        super().__init__()
        self.dims = dims

    def embed_query(self, text: str) -> list[float]:
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dims).astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def _writer(
    root: str,
    workspace_id: str,
    contents: list[str],
    ready: multiprocessing.synchronize.Event,
    go: multiprocessing.synchronize.Event,
    result: multiprocessing.Queue[int],
) -> None:
    coordinator = PortalockerStorageCoordinator(root, timeout=30.0)
    config = USearchConfig(
        workspace_id=workspace_id,
        index_path=os.path.join(root, workspace_id, "usearch", "vectors.usearch"),
        db_path=os.path.join(root, workspace_id, "usearch", "memories.db"),
        embedding_dims=32,
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="test/hash-32",
    )
    engine = USearchEngine(
        config=config, embedder=_HashEmbedder(), coordinator=coordinator
    )
    ready.set()
    if not go.wait(timeout=30.0):
        result.put(-1)
        engine.close()
        return
    with coordinator.acquire(workspace_id):
        added = engine.add_batch(workspace_id, contents, infer=False)
        engine.commit()
    result.put(len(added))
    engine.close()


@pytest.mark.integration
def test_disjoint_and_duplicate_multiprocess_writes(tmp_path: Path) -> None:
    root = str(tmp_path / "indexes")
    workspace_id = "ws"
    ctx = multiprocessing.get_context("spawn")
    ready_events = [ctx.Event() for _ in range(4)]
    go = ctx.Event()
    queue: multiprocessing.Queue[int] = ctx.Queue()
    payloads = [
        [f"w0-{i}" for i in range(200)] + ["shared-dup"],
        [f"w1-{i}" for i in range(200)] + ["shared-dup"],
        [f"w2-{i}" for i in range(200)] + ["shared-dup"],
        [f"w3-{i}" for i in range(200)] + ["shared-dup"],
    ]
    processes = [
        ctx.Process(
            target=_writer,
            args=(root, workspace_id, payload, ready, go, queue),
        )
        for payload, ready in zip(payloads, ready_events, strict=True)
    ]
    for process in processes:
        process.start()
    try:
        assert all(ready.wait(timeout=20.0) for ready in ready_events)
        go.set()
        for process in processes:
            process.join(timeout=60.0)
            assert process.exitcode == 0
        counts = [queue.get(timeout=1.0) for _ in processes]
        assert all(count > 0 for count in counts)
        coordinator = PortalockerStorageCoordinator(root, timeout=5.0)
        config = USearchConfig(
            workspace_id=workspace_id,
            index_path=os.path.join(root, workspace_id, "usearch", "vectors.usearch"),
            db_path=os.path.join(root, workspace_id, "usearch", "memories.db"),
            embedding_dims=32,
            embedder_provider=EmbedderProvider.OPENAI,
            embedding_model="test/hash-32",
        )
        inspector = USearchEngine(
            config=config, embedder=_HashEmbedder(), coordinator=coordinator
        )
        try:
            rows = inspector.get_all(workspace_id)
            assert len(rows) == 801
            assert rows.count("shared-dup") == 1
            assert inspector.get_id_by_content(workspace_id, "shared-dup") is not None
            assert len(inspector.index) == 801
        finally:
            inspector.close()
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)


def _mutator(
    root: str,
    workspace_id: str,
    action: str,
    content: str,
    ready: multiprocessing.synchronize.Event,
    go: multiprocessing.synchronize.Event,
    result: multiprocessing.Queue[str],
) -> None:
    coordinator = PortalockerStorageCoordinator(root, timeout=30.0)
    config = USearchConfig(
        workspace_id=workspace_id,
        index_path=os.path.join(root, workspace_id, "usearch", "vectors.usearch"),
        db_path=os.path.join(root, workspace_id, "usearch", "memories.db"),
        embedding_dims=32,
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="test/hash-32",
    )
    engine = USearchEngine(
        config=config, embedder=_HashEmbedder(), coordinator=coordinator
    )
    ready.set()
    if not go.wait(timeout=30.0):
        result.put("timeout")
        engine.close()
        return
    if action == "add":
        engine.add(workspace_id, content, infer=False)
        engine.commit()
        result.put("added")
    elif action == "delete":
        mem_id = engine.get_id_by_content(workspace_id, content)
        if mem_id is not None:
            engine.delete(str(mem_id))
            engine.commit()
            result.put("deleted")
        else:
            result.put("missing")
    engine.close()


@pytest.mark.integration
def test_delete_and_readd_later_write_wins(tmp_path: Path) -> None:
    root = str(tmp_path / "indexes")
    workspace_id = "ws"
    ctx = multiprocessing.get_context("spawn")
    seed_ready = ctx.Event()
    seed_go = ctx.Event()
    queue: multiprocessing.Queue[str] = ctx.Queue()
    seeder = ctx.Process(
        target=_mutator,
        args=(root, workspace_id, "add", "race-row", seed_ready, seed_go, queue),
    )
    seeder.start()
    assert seed_ready.wait(timeout=20.0)
    seed_go.set()
    seeder.join(timeout=30.0)
    assert seeder.exitcode == 0
    assert queue.get(timeout=1.0) == "added"

    delete_ready = ctx.Event()
    add_ready = ctx.Event()
    go = ctx.Event()
    deleter = ctx.Process(
        target=_mutator,
        args=(root, workspace_id, "delete", "race-row", delete_ready, go, queue),
    )
    adder = ctx.Process(
        target=_mutator,
        args=(root, workspace_id, "add", "race-row", add_ready, go, queue),
    )
    deleter.start()
    adder.start()
    try:
        assert delete_ready.wait(timeout=20.0)
        assert add_ready.wait(timeout=20.0)
        go.set()
        deleter.join(timeout=30.0)
        adder.join(timeout=30.0)
        assert deleter.exitcode == 0
        assert adder.exitcode == 0
        coordinator = PortalockerStorageCoordinator(root, timeout=5.0)
        config = USearchConfig(
            workspace_id=workspace_id,
            index_path=os.path.join(root, workspace_id, "usearch", "vectors.usearch"),
            db_path=os.path.join(root, workspace_id, "usearch", "memories.db"),
            embedding_dims=32,
            embedder_provider=EmbedderProvider.OPENAI,
            embedding_model="test/hash-32",
        )
        inspector = USearchEngine(
            config=config, embedder=_HashEmbedder(), coordinator=coordinator
        )
        try:
            rows = inspector.get_all(workspace_id)
            assert rows.count("race-row") <= 1
        finally:
            inspector.close()
    finally:
        for process in (deleter, adder):
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
