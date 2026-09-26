"""Characterization of legacy persisted data and public MCP contracts.

Fixtures are generated at test time. No schema migration and no real user data.
"""

from __future__ import annotations

from inspect import signature
import os
import sqlite3
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.tools.add import AddTool
from reflectlog.application.tools.get_all import GetAllTool
from reflectlog.application.tools.health_check import HealthCheckTool
from reflectlog.application.tools.remove import RemoveTool
from reflectlog.application.tools.search import SearchTool
from reflectlog.application.utils.security import SecretString
from reflectlog.core.enums import (
    EmbedderProvider,
    EngineReadiness,
    HealthStatus,
    ToolName,
    TransitionKind,
    TransitionStatus,
)
from reflectlog.core.exceptions import InitializationError, StorageError
from reflectlog.core.types import ReplacementTransitionRequest
from reflectlog.infrastructure.embedding_identity import ensure_embedding_identity
from reflectlog.infrastructure.memory_store import MemoryStore
from reflectlog.infrastructure.tantivy_engine import TantivyEngine
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine
from tests.integration.test_memory_manager_usearch import (
    MockEmbedder,
    cleanup_manager,
    create_memory_manager,
    create_usearch_config,
)

WORKSPACE = "legacy-compat"
LIVE_A = "legacy live memory alpha"
LIVE_B = "legacy live memory beta"


def _direct_config(tmp_path: str) -> tuple[USearchConfig, str]:
    usearch_dir = os.path.join(tmp_path, "indexes", WORKSPACE, "usearch")
    os.makedirs(usearch_dir, exist_ok=True)
    config = USearchConfig(
        workspace_id=WORKSPACE,
        index_path=os.path.join(usearch_dir, "vectors.usearch"),
        db_path=os.path.join(usearch_dir, "memories.db"),
        embedding_dims=128,
        embedder_provider=EmbedderProvider.OPENAI,
        embedding_model="test/mock-128",
    )
    return config, usearch_dir


def _sqlite_memory_rows(db_path: str) -> list[tuple[int, str, str, object]]:
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT id, workspace_id, content, created_at FROM memories ORDER BY id"
        ).fetchall()
        return [(int(row[0]), str(row[1]), str(row[2]), row[3]) for row in rows]
    finally:
        connection.close()


@pytest.mark.integration
def test_legacy_sqlite_rows_unique_identity_and_empty_timestamps() -> None:
    """SQLite identity is unique (workspace, content); missing timestamps stay empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "memories.db")
        store = MemoryStore(db_path=db_path)
        try:
            first_id = store.insert(WORKSPACE, LIVE_A)
            other_id = store.insert("other-workspace", LIVE_A)
            assert first_id != other_id
            with pytest.raises(StorageError, match="Duplicate memory"):
                _ = store.insert(WORKSPACE, LIVE_A)
        finally:
            store.close()

        connection = sqlite3.connect(db_path)
        try:
            _ = connection.execute(
                "UPDATE memories SET created_at = NULL WHERE id = ?",
                (first_id,),
            )
            connection.commit()
        finally:
            connection.close()

        reopened = MemoryStore(db_path=db_path)
        try:
            record = reopened.get(first_id)
            assert record is not None
            assert record.workspace_id == WORKSPACE
            assert record.content == LIVE_A
            assert record.created_at == ""
            other = reopened.get(other_id)
            assert other is not None
            assert other.workspace_id == "other-workspace"
        finally:
            reopened.close()


@pytest.mark.integration
def test_journal_kinds_and_pending_transition_rows() -> None:
    """Journal kinds are add|delete|replace with pending later-write-wins order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(db_path=os.path.join(tmpdir, "memories.db"))
        try:
            old_id = store.insert(WORKSPACE, "old-text")
            adds = store.begin_add_intents(WORKSPACE, ["new-text"])
            deletes = store.begin_delete_intents(WORKSPACE, [(old_id, "old-text")])
            replaces = store.begin_replacement_transitions(
                [
                    ReplacementTransitionRequest(
                        old_memory_id=old_id,
                        workspace_id=WORKSPACE,
                        old_content="old-text",
                        new_content="replacement-text",
                        reason="updated convention",
                        confidence=0.91,
                    )
                ]
            )
            pending = store.list_pending_transitions()
            kinds = [row.kind for row in pending]
            assert kinds == [
                TransitionKind.ADD,
                TransitionKind.DELETE,
                TransitionKind.REPLACE,
            ]
            assert {row.status for row in pending} == {TransitionStatus.PENDING}
            assert adds[0].new_content == "new-text"
            assert deletes[0].old_content == "old-text"
            assert replaces[0].new_content == "replacement-text"
            assert pending[0].id < pending[1].id < pending[2].id
        finally:
            store.close()


@pytest.mark.integration
def test_generated_usearch_and_tantivy_reopen_without_sidecars() -> None:
    """Current HNSW + Tantivy reopen with the same identity and no lock sidecars."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = create_usearch_config(tmpdir, project_suffix="legacyreopen")
        manager, _ = create_memory_manager(config)
        workspace = manager.workspace_id
        try:
            stored = manager.add_memories([LIVE_A, LIVE_B])
            assert stored == 2
            first = manager.get_all()
            assert set(first) == {LIVE_A, LIVE_B}
            engine = manager._semantic_engine
            assert isinstance(engine, USearchEngine)
            assert len(engine.index) == 2
            tantivy = manager._tantivy_engine
            assert isinstance(tantivy, TantivyEngine)
            assert LIVE_A in tantivy.find_by_exact_match(workspace, LIVE_A)
            assert LIVE_B in tantivy.find_by_exact_match(workspace, LIVE_B)
            usearch_dir = os.path.dirname(engine.config.index_path)
            assert engine.config.index_path.endswith(
                os.path.join(workspace.lower(), "usearch", "vectors.usearch")
            )
            assert engine.config.db_path.endswith(
                os.path.join(workspace.lower(), "usearch", "memories.db")
            )
            assert tantivy.config.index_path.endswith(
                os.path.join(workspace.lower(), "tantivy")
            )
            assert not os.path.exists(
                os.path.join(usearch_dir, ".reflectlog.writer.lock")
            )
            assert not os.path.exists(
                os.path.join(usearch_dir, ".reflectlog.storage-generation")
            )
            sqlite_rows = _sqlite_memory_rows(engine.config.db_path)
            assert {row[2] for row in sqlite_rows} == {LIVE_A, LIVE_B}
            manager.close()
            reopened, _ = create_memory_manager(config)
            try:
                assert set(reopened.get_all()) == {LIVE_A, LIVE_B}
                engine = reopened._semantic_engine
                assert isinstance(engine, USearchEngine)
                assert len(engine.index) == 2
                tantivy = reopened._tantivy_engine
                assert isinstance(tantivy, TantivyEngine)
                assert LIVE_A in tantivy.find_by_exact_match(workspace, LIVE_A)
                assert LIVE_B in tantivy.find_by_exact_match(workspace, LIVE_B)
            finally:
                cleanup_manager(reopened)
        except Exception:
            cleanup_manager(manager)
            raise


@pytest.mark.integration
def test_later_write_wins_does_not_resurrect_superseded_add() -> None:
    """A later DELETE of the same text completes a pending ADD without inserting it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = create_usearch_config(tmpdir, project_suffix="laterwin")
        manager, _ = create_memory_manager(config)
        try:
            assert manager.add_memories(["keep-me"]) == 1
            store = manager._semantic_engine.memory_store
            assert isinstance(store, MemoryStore)
            _ = store.begin_add_intents(manager.workspace_id, ["ghost-content"])
            _ = store.begin_delete_intents(manager.workspace_id, [(0, "ghost-content")])
            completed = manager.reconcile_pending_replacements()
            assert completed >= 1
            assert manager.get_all() == ["keep-me"]
            assert store.list_pending_transitions() == []
        finally:
            cleanup_manager(manager)


@pytest.mark.integration
def test_corrupt_or_missing_hnsw_with_sqlite_rows_fails_closed() -> None:
    """Populated SQLite plus corrupt/missing HNSW must not create an empty index."""
    embedder = MockEmbedder(dims=128)
    with tempfile.TemporaryDirectory() as tmpdir:
        config, _usearch_dir = _direct_config(tmpdir)
        ensure_embedding_identity(config)
        store = MemoryStore(db_path=config.db_path)
        kept_id = store.insert(WORKSPACE, LIVE_A)
        store.close()
        before = _sqlite_memory_rows(config.db_path)
        assert before == [(kept_id, WORKSPACE, LIVE_A, before[0][3])]

        with open(config.index_path, "wb") as handle:
            handle.write(b"not-a-valid-usearch-index")
        corrupt = USearchEngine(config=config, embedder=embedder)
        # Real Index.restore() currently raises ValueError; the engine wraps
        # that as RuntimeError and must not replace the file with an empty HNSW.
        with pytest.raises(RuntimeError, match="Failed to initialize USearch index"):
            _ = corrupt.index
        assert _sqlite_memory_rows(config.db_path) == before
        with open(config.index_path, "rb") as handle:
            assert handle.read() == b"not-a-valid-usearch-index"

        os.remove(config.index_path)
        missing = USearchEngine(config=config, embedder=embedder)
        with pytest.raises(InitializationError, match="missing but SQLite"):
            _ = missing.index
        assert not os.path.exists(config.index_path)
        assert _sqlite_memory_rows(config.db_path) == before


@pytest.mark.integration
async def test_mcp_tool_signatures_and_public_result_shapes() -> None:
    """The five registered tools keep their public names, params, and result keys."""
    config = Config(
        workspace_id="legacy-tools",
        openrouter_api_key=SecretString("test-key"),
    )
    logger = MagicMock()
    memory = MagicMock(spec=MemoryManager)
    memory.add_memories_async = AsyncMock()
    memory.search = AsyncMock(return_value=[LIVE_A])
    memory.get_all = MagicMock(return_value=[LIVE_A])
    memory.count = MagicMock(return_value=1)
    memory.search_engine_status = MagicMock(
        return_value={
            "semantic_engine": EngineReadiness.INITIALIZED,
            "tantivy_engine": EngineReadiness.INITIALIZED,
        }
    )
    memory.pending_intent_count = MagicMock(return_value=0)
    memory.startup_metrics = None

    add = AddTool(config, memory, logger)
    search = SearchTool(config, memory, logger)
    get_all = GetAllTool(config, memory, logger)
    remove = RemoveTool(config, memory, logger)
    health = HealthCheckTool(config, memory, logger)

    assert [tool.get_name() for tool in (add, search, get_all, remove, health)] == [
        ToolName.ADD,
        ToolName.SEARCH,
        ToolName.GET_ALL,
        ToolName.REMOVE,
        ToolName.HEALTH_CHECK,
    ]
    assert [str(name) for name in ToolName] == [
        "add",
        "get_all",
        "search",
        "remove",
        "health_check",
    ]

    add_params = list(signature(add.get_handler()).parameters)
    search_params = list(signature(search.get_handler()).parameters)
    get_all_params = list(signature(get_all.get_handler()).parameters)
    remove_params = list(signature(remove.get_handler()).parameters)
    health_params = list(signature(health.get_handler()).parameters)
    assert add_params == ["memories", "dry_run"]
    assert search_params == ["query"]
    assert get_all_params == ["limit", "offset"]
    assert remove_params == ["memories"]
    assert health_params == []

    empty_add = await add.get_handler()([])
    assert empty_add == {
        "stored_count": 0,
        "skipped_count": 0,
        "replaced_count": 0,
        "replacements": [],
        "dry_run": False,
    }
    search_hits = await search.get_handler()("legacy")
    assert search_hits == [LIVE_A]
    page = await get_all.get_handler()()
    assert page == {
        "memories": [LIVE_A],
        "total": 1,
        "offset": 0,
        "limit": config.get_all_limit,
        "truncated": False,
    }
    assert await remove.get_handler()([]) is None
    health_payload = await health.get_handler()()
    assert health_payload["status"] == HealthStatus.HEALTHY
    assert health_payload["workspace_id"] == "legacy-tools"
    assert set(health_payload) >= {
        "status",
        "workspace_id",
        "semantic_engine",
        "tantivy_engine",
        "reranker_engine",
        "hybrid_search_enabled",
        "rrf_fusion_enabled",
        "recency_boost_enabled",
        "pending_intent_count",
    }
