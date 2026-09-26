"""Process-death coverage for atomic USearch publication."""

from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import os
from pathlib import Path

import numpy as np
import pytest

from reflectlog.core.enums import EmbedderProvider
from reflectlog.core.types import Embeddings
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


def _writer_die_at(
    index_path: str,
    db_path: str,
    workspace_id: str,
    step: str,
    ready: multiprocessing.synchronize.Event,
) -> None:
    def boom(name: str) -> None:
        if name == step:
            os._exit(20)

    config = USearchConfig(
        workspace_id=workspace_id,
        index_path=index_path,
        db_path=db_path,
        embedding_dims=32,
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="test/hash-32",
    )
    first = USearchEngine(config=config, embedder=_HashEmbedder())
    first.add(workspace_id, "kept", infer=False)
    first.commit()
    first.close()
    ready.set()
    second = USearchEngine(config=config, embedder=_HashEmbedder(), publish_hook=boom)
    second.add(workspace_id, "new-row", infer=False)
    second.commit()


@pytest.mark.integration
@pytest.mark.parametrize(
    "step",
    [
        "before_save",
        "after_temp_save",
        "after_temp_validate",
        "after_fsync",
        "before_replace",
    ],
)
def test_kill_at_publish_failpoint_keeps_valid_index(tmp_path: Path, step: str) -> None:
    index_path = str(tmp_path / "vectors.usearch")
    db_path = str(tmp_path / "memories.db")
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    child = ctx.Process(
        target=_writer_die_at,
        args=(index_path, db_path, "ws", step, ready),
    )
    child.start()
    try:
        assert ready.wait(timeout=20.0)
        child.join(timeout=30.0)
        assert child.exitcode == 20
        inspector = USearchEngine(
            config=USearchConfig(
                workspace_id="ws",
                index_path=index_path,
                db_path=db_path,
                embedding_dims=32,
                embedder_provider=EmbedderProvider.OPENAI,
                embedding_model="test/hash-32",
            ),
            embedder=_HashEmbedder(),
        )
        try:
            _ = inspector.index
            temps = [name for name in os.listdir(tmp_path) if name.endswith(".tmp")]
            assert temps == []
            rows = inspector.get_all("ws")
            assert "kept" in rows
            assert len(inspector.index) == 1
        finally:
            inspector.close()
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5.0)
