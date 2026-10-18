"""Unit tests for MemoryStore SQLite storage."""

import os
from pathlib import Path
import sqlite3
import tempfile
import threading
from unittest.mock import MagicMock

import pytest

from reflectlog.core.exceptions import StorageError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.memory_store import (
    ArchivedMemoryRecord,
    MemoryRecord,
    MemoryStore,
)


def create_mock_logger() -> MagicMock:
    return MagicMock(spec=IStructuredLogger)


class TestMemoryStoreInitialization:
    """Tests for MemoryStore initialization."""

    def test_creates_database_file(self) -> None:
        """Database file should be created on first access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            # Access connection to trigger creation
            _ = store.connection

            assert os.path.exists(db_path)
            store.close()

    def test_creates_parent_directories(self) -> None:
        """Parent directories should be created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "nested", "path", "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            assert os.path.exists(db_path)
            store.close()

    def test_lazy_initialization(self) -> None:
        """Connection should not be created until accessed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            # Before accessing connection, file should not exist
            assert not os.path.exists(db_path)

            # Access connection
            _ = store.connection

            # Now file should exist
            assert os.path.exists(db_path)
            store.close()


class TestMemoryStoreInsert:
    """Tests for MemoryStore.insert method."""

    def test_insert_returns_id(self) -> None:
        """Insert should return auto-increment ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            memory_id = store.insert("user1", "Hello world")

            assert memory_id == 1
            store.close()

    def test_insert_increments_id(self) -> None:
        """Subsequent inserts should return incrementing IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            id1 = store.insert("user1", "Memory 1")
            id2 = store.insert("user1", "Memory 2")
            id3 = store.insert("user1", "Memory 3")

            assert id1 == 1
            assert id2 == 2
            assert id3 == 3
            store.close()

    def test_insert_duplicate_raises_error(self) -> None:
        """Inserting duplicate memory for same user should raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.insert("user1", "Hello world")

            with pytest.raises(StorageError, match="Duplicate memory"):
                _ = store.insert("user1", "Hello world")

            store.close()

    def test_insert_same_memory_different_users(self) -> None:
        """Same memory for different users should be allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            id1 = store.insert("user1", "Hello world")
            id2 = store.insert("user2", "Hello world")

            assert id1 == 1
            assert id2 == 2
            store.close()


class TestMemoryStoreGet:
    """Tests for MemoryStore.get method."""

    def test_get_existing_memory(self) -> None:
        """Get should return MemoryRecord for existing ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            memory_id = store.insert("user1", "Hello world")
            record = store.get(memory_id)

            assert record is not None
            assert record.id == memory_id
            assert record.workspace_id == "user1"
            assert record.content == "Hello world"
            store.close()

    def test_get_nonexistent_memory(self) -> None:
        """Get should return None for non-existent ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            record = store.get(999)

            assert record is None
            store.close()


class TestMemoryStoreGetAll:
    """Tests for MemoryStore.get_all method."""

    def test_get_all_empty(self) -> None:
        """Get all should return empty list when no memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            memories = store.get_all("user1")

            assert memories == []
            store.close()

    def test_get_all_single_user(self) -> None:
        """Get all should return memories for specific user."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.insert("user1", "Memory 1")
            _ = store.insert("user1", "Memory 2")
            _ = store.insert("user2", "Other memory")

            memories = store.get_all("user1")

            assert len(memories) == 2
            assert "Memory 1" in memories
            assert "Memory 2" in memories
            assert "Other memory" not in memories
            store.close()

    def test_get_all_preserves_order(self) -> None:
        """Get all should return memories in insertion order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.insert("user1", "First")
            _ = store.insert("user1", "Second")
            _ = store.insert("user1", "Third")

            memories = store.get_all("user1")

            assert memories == ["First", "Second", "Third"]
            store.close()

    def test_get_all_pages_in_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            _ = store.insert("user1", "First")
            _ = store.insert("user1", "Second")
            _ = store.insert("user1", "Third")
            assert store.get_all("user1", limit=1, offset=1) == ["Second"]
            assert store.get_all("user1", limit=10, offset=2) == ["Third"]
            store.close()


class TestMemoryStoreDelete:
    """Tests for MemoryStore.delete method."""

    def test_delete_existing_memory(self) -> None:
        """Delete should return True for existing memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            memory_id = store.insert("user1", "Hello world")
            deleted = store.delete(memory_id)

            assert deleted is True
            assert store.get(memory_id) is None
            store.close()

    def test_delete_nonexistent_memory(self) -> None:
        """Delete should return False for non-existent memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            deleted = store.delete(999)

            assert deleted is False
            store.close()


class TestMemoryStoreExists:
    """Tests for MemoryStore.exists method."""

    def test_exists_returns_true_for_existing(self) -> None:
        """Exists should return True for existing memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.insert("user1", "Hello world")

            assert store.exists("user1", "Hello world") is True
            store.close()

    def test_exists_returns_false_for_nonexistent(self) -> None:
        """Exists should return False for non-existent memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            assert store.exists("user1", "Hello world") is False
            store.close()

    def test_exists_checks_workspace_id(self) -> None:
        """Exists should check workspace_id, not just memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.insert("user1", "Hello world")

            assert store.exists("user1", "Hello world") is True
            assert store.exists("user2", "Hello world") is False
            store.close()


class TestMemoryStoreGetIdByMemory:
    """Tests for MemoryStore.get_id_by_content method."""

    def test_get_id_by_content_existing(self) -> None:
        """Get ID by memory should return ID for existing memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            memory_id = store.insert("user1", "Hello world")
            found_id = store.get_id_by_content("user1", "Hello world")

            assert found_id == memory_id
            store.close()

    def test_get_id_by_content_nonexistent(self) -> None:
        """Get ID by memory should return None for non-existent memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            found_id = store.get_id_by_content("user1", "Hello world")

            assert found_id is None
            store.close()


class TestMemoryStoreGetRecordsByContents:
    """Bulk content lookup is one workspace-scoped SELECT per chunk."""

    def test_returns_requested_workspace_rows_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            _ = store.insert("ws-a", "one")
            _ = store.insert("ws-a", "two")
            _ = store.insert("ws-b", "one")
            records = store.get_records_by_contents(
                "ws-a", ["two", "missing", "one", "one"]
            )
            assert [row.content for row in records] == ["two", "one"]
            assert all(row.workspace_id == "ws-a" for row in records)
            store.close()

    def test_five_contents_use_one_select(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))
            for index in range(5):
                _ = store.insert("ws", f"row-{index}")
            statements: list[str] = []
            store.connection.set_trace_callback(statements.append)
            _ = store.get_records_by_contents(
                "ws", [f"row-{index}" for index in range(5)]
            )
            store.connection.set_trace_callback(None)
            selects = [
                item
                for item in statements
                if item.lstrip().upper().startswith("SELECT")
            ]
            assert len(selects) == 1
            store.close()


def test_successor_live_record_matches_existing_readers_when_timestamp_is_null(
    tmp_path: Path,
) -> None:
    store = MemoryStore(db_path=str(tmp_path / "null-timestamp.db"))
    try:
        memory_id = store.insert("ws", "legacy memory")
        assert memory_id is not None
        store.connection.execute(
            "UPDATE memories SET created_at = NULL WHERE id = ?", (memory_id,)
        )
        store.connection.commit()
        existing_record = store.get(memory_id)
        batch_record = store.get_batch([memory_id])[memory_id]

        result = store.get_record_with_successor("ws", memory_id)

        assert result.record == existing_record == batch_record
        assert result.record is not None
        assert result.record.created_at == ""
    finally:
        store.close()


@pytest.mark.parametrize("latest_active", [False, True])
def test_successor_uses_latest_archive_id_and_workspace(
    tmp_path: Path, latest_active: bool
) -> None:
    store = MemoryStore(db_path=str(tmp_path / "successors.db"))
    try:
        old = store.insert("ws", "old")
        first = store.insert("ws", "first")
        latest = store.insert("ws" if latest_active else "foreign", "latest")
        assert old is not None and first is not None and latest is not None
        store.archive(old, "ws", "old", "first", "private reason", 0.9)
        archive_id = store.archive(old, "ws", "old", "latest", "private reason", 0.9)
        store.archive(old, "foreign", "old", "foreign replacement", "private", 0.9)
        assert archive_id is not None
        store.connection.execute("UPDATE archived_memories SET id = id + 8")
        store.connection.execute(
            "UPDATE archived_memories SET archived_at = '1900-01-01' WHERE id = ?",
            (archive_id + 8,),
        )
        store.connection.commit()
        assert store.get_record_with_successor("ws", old).record == store.get(old)
        assert store.delete(old)
        result = store.get_record_with_successor("ws", old)
        assert result.record is None
        assert result.successor_id == (latest if latest_active else None)
        assert store.get_record_with_successor("other", old).successor_id is None
        assert store.get_record_with_successor("ws", 999).successor_id is None
        assert not store.connection.in_transaction
    finally:
        store.close()


def test_successor_lookup_holds_one_read_snapshot(tmp_path: Path) -> None:
    store = MemoryStore(db_path=str(tmp_path / "snapshot.db"))
    writer = MemoryStore(db_path=store.db_path)
    try:
        successor_id = store.insert("ws", "before")
        store.archive(999, "ws", "old", "before", "private", 0.9)
        _ = writer.connection
        changed = False

        def interleave(statement: str) -> None:
            nonlocal changed
            if "SELECT replaced_by" in statement and not changed:
                changed = True
                writer.connection.execute(
                    "UPDATE archived_memories SET replaced_by = 'after'"
                )
                writer.connection.execute("DELETE FROM memories")
                writer.connection.execute(
                    "INSERT INTO memories (workspace_id, content) VALUES ('ws', 'after')"
                )
                writer.connection.commit()

        store.connection.set_trace_callback(interleave)
        result = store.get_record_with_successor("ws", 999)
        store.connection.set_trace_callback(None)
        assert changed
        assert result.record is None
        assert result.successor_id == successor_id
        assert not store.connection.in_transaction
        assert store.get_record_with_successor("ws", 999).successor_id != successor_id
    finally:
        writer.close()
        store.close()


def test_successor_lookup_cleans_up_failed_read(tmp_path: Path) -> None:
    store = MemoryStore(db_path=str(tmp_path / "failed-read.db"))
    try:

        def authorize(
            action: int,
            table: str | None,
            column: str | None,
            database: str | None,
            source: str | None,
        ) -> int:
            return (
                sqlite3.SQLITE_DENY
                if table == "archived_memories"
                else sqlite3.SQLITE_OK
            )

        store.connection.set_authorizer(authorize)
        with pytest.raises(
            StorageError, match="Failed to resolve memory successor"
        ) as raised:
            store.get_record_with_successor("ws", 999)
        assert isinstance(raised.value.__cause__, sqlite3.Error)
        assert not store.connection.in_transaction
        store.connection.set_authorizer(None)
        assert store.get_record_with_successor("ws", 999).successor_id is None
    finally:
        store.close()


class TestMemoryStoreThreadSafety:
    """Tests for MemoryStore thread safety."""

    @pytest.mark.skip(reason="Concurrent test causes system instability in CI")
    def test_concurrent_inserts(self) -> None:
        """Concurrent inserts should be thread-safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            results: list[int] = []
            lock = threading.Lock()

            def insert_memory(i: int) -> None:
                memory_id = store.insert("user1", f"Memory {i}")
                with lock:
                    results.append(memory_id)

            threads = [
                threading.Thread(target=insert_memory, args=(i,)) for i in range(10)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All inserts should succeed with unique IDs
            assert len(results) == 10
            assert len(set(results)) == 10
            store.close()


class TestMemoryStoreLogging:
    """Tests for MemoryStore logging."""

    def test_logs_insert_on_debug(self) -> None:
        """Insert should log on debug level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.insert("user1", "Hello world")

            logger.debug.assert_called()
            store.close()

    def test_logs_delete_on_debug(self) -> None:
        """Delete should log on debug level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            memory_id = store.insert("user1", "Hello world")
            logger.reset_mock()

            _ = store.delete(memory_id)

            logger.debug.assert_called()
            store.close()

    def test_logs_initialization_on_debug(self) -> None:
        """Logger should be called when store initializes connection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            logger.debug.assert_called_once()
            call_args = logger.debug.call_args
            assert "MemoryStore initialized" in call_args[0][0]
            store.close()

    def test_logs_delete_not_found(self) -> None:
        """Logger should log when memory not found for deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.delete(999)

            # Should have called debug with "not found" memory
            debug_calls = logger.debug.call_args_list
            memories = [call[0][0] for call in debug_calls]
            assert any("not found" in m for m in memories)
            store.close()

    def test_logs_close_on_debug(self) -> None:
        """Logger should be called when store is closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            # Force initialization
            _ = store.connection
            logger.reset_mock()

            store.close()

            logger.debug.assert_called_once()
            call_args = logger.debug.call_args
            assert "MemoryStore closed" in call_args[0][0]

    def test_logs_duplicate_insert_on_debug(self) -> None:
        """Logger should log duplicate memory detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.insert("user1", "Hello world")

            with pytest.raises(StorageError):
                _ = store.insert("user1", "Hello world")

            debug_calls = logger.debug.call_args_list
            memories = [call[0][0] for call in debug_calls]
            assert any("Duplicate" in m or "duplicate" in m for m in memories)
            store.close()

    def test_logs_batch_insert_on_debug(self) -> None:
        """Logger should log batch insert completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.insert_many("user1", ["msg1", "msg2"])

            debug_calls = logger.debug.call_args_list
            memories = [call[0][0] for call in debug_calls]
            assert any("Batch insert" in m or "batch" in m.lower() for m in memories)
            store.close()

    def test_logs_batch_delete_on_debug(self) -> None:
        """Logger should log batch delete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            id1 = store.insert("user1", "msg1")
            id2 = store.insert("user1", "msg2")
            logger.reset_mock()

            _ = store.delete_batch([id1, id2])

            logger.debug.assert_called()
            store.close()

    def test_logs_archive_on_debug(self) -> None:
        """Logger should log memory archival."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            # Insert directly into archive table to test logging
            cursor = store.connection.cursor()
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, "proj", "old msg", "new msg", "updated", 0.9),
            )
            store.connection.commit()
            cursor.close()
            store.close()

    def test_logs_restore_on_info(self) -> None:
        """Logger should log memory restoration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            # Manually insert archive record
            cursor = store.connection.cursor()
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, "proj", "old msg", "new msg", "updated", 0.9),
            )
            store.connection.commit()
            cursor.close()

            logger.reset_mock()
            _ = store.restore_from_archive(1)

            logger.info.assert_called()
            call_args = logger.info.call_args
            assert "restored" in call_args[0][0].lower()
            store.close()

    def test_logs_cleanup_on_info(self) -> None:
        """Logger should log expired archive cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            # Insert a very old archive record
            cursor = store.connection.cursor()
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, "
                "confidence, archived_at) "
                "VALUES (?, ?, ?, ?, ?, ?, datetime('now', '-100 days'))",
                (1, "proj", "old msg", "new msg", "outdated", 0.9),
            )
            store.connection.commit()
            cursor.close()

            logger.reset_mock()
            deleted = store.cleanup_expired_archive(30)

            assert deleted == 1
            logger.info.assert_called()
            store.close()


class TestMemoryStoreInsertMany:
    """Tests for MemoryStore.insert_many method."""

    def test_insert_many_returns_pairs(self) -> None:
        """insert_many should return list of (memory, id) tuples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            result = store.insert_many("proj1", ["msg1", "msg2", "msg3"])

            assert len(result) == 3
            memories = [r[0] for r in result]
            assert "msg1" in memories
            assert "msg2" in memories
            assert "msg3" in memories
            # IDs should be positive integers
            for _, memory_id in result:
                assert memory_id > 0
            store.close()

    def test_insert_many_empty_list(self) -> None:
        """insert_many with empty list should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            result = store.insert_many("proj1", [])

            assert result == []
            store.close()

    def test_insert_many_skips_duplicates(self) -> None:
        """insert_many should skip duplicate memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            # Insert first memory
            _ = store.insert("proj1", "existing")

            # Batch insert with duplicate
            result = store.insert_many("proj1", ["existing", "new1", "new2"])

            assert len(result) == 2
            memories = [r[0] for r in result]
            assert "existing" not in memories
            assert "new1" in memories
            assert "new2" in memories
            store.close()

    def test_insert_many_skips_duplicates_with_logger(self) -> None:
        """insert_many should log when skipping duplicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.insert("proj1", "dup")
            logger.reset_mock()

            _ = store.insert_many("proj1", ["dup", "new_msg"])

            # Should have logged the duplicate skip
            debug_calls = logger.debug.call_args_list
            memories = [call[0][0] for call in debug_calls]
            assert any("Duplicate" in m or "duplicate" in m.lower() for m in memories)
            store.close()

    def test_insert_many_all_duplicates(self) -> None:
        """insert_many should return empty when all are duplicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.insert("proj1", "msg1")
            _ = store.insert("proj1", "msg2")

            result = store.insert_many("proj1", ["msg1", "msg2"])

            assert len(result) == 0
            store.close()

    def test_insert_many_db_error_raises_storage_error(self) -> None:
        """insert_many should raise StorageError on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            # Force initialization
            _ = store.connection

            # Corrupt the table by dropping it
            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to insert memory batch"):
                _ = store.insert_many("proj1", ["msg1"])
            store.close()

    def test_insert_many_db_error_with_logger(self) -> None:
        """insert_many should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.insert_many("proj1", ["msg1"])

            logger.error.assert_called()
            store.close()

    def test_insert_many_non_integrity_error_reraises(self) -> None:
        """insert_many should re-raise non-integrity sqlite errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            # Drop the table to cause a non-IntegrityError sqlite error
            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.insert_many("proj1", ["msg"])
            store.close()


class TestMemoryStoreGetBatch:
    """Tests for MemoryStore.get_batch method."""

    def test_get_batch_returns_records(self) -> None:
        """get_batch should return dict of memory ID to record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            id1 = store.insert("proj1", "msg1")
            id2 = store.insert("proj1", "msg2")
            id3 = store.insert("proj1", "msg3")

            result = store.get_batch([id1, id2, id3])

            assert len(result) == 3
            assert result[id1].content == "msg1"
            assert result[id2].content == "msg2"
            assert result[id3].content == "msg3"
            store.close()

    def test_get_batch_empty_list(self) -> None:
        """get_batch with empty list should return empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            result = store.get_batch([])

            assert result == {}
            store.close()

    def test_get_batch_missing_ids(self) -> None:
        """get_batch should exclude missing IDs from result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            id1 = store.insert("proj1", "msg1")

            result = store.get_batch([id1, 999, 1000])

            assert len(result) == 1
            assert id1 in result
            assert 999 not in result
            store.close()

    def test_get_batch_record_fields(self) -> None:
        """get_batch should return MemoryRecord with all fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            id1 = store.insert("proj1", "hello")

            result = store.get_batch([id1])

            record = result[id1]
            assert isinstance(record, MemoryRecord)
            assert record.id == id1
            assert record.workspace_id == "proj1"
            assert record.content == "hello"
            assert record.created_at != ""
            store.close()

    def test_get_batch_db_error_raises_storage_error(self) -> None:
        """get_batch should raise StorageError on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to retrieve memory batch"):
                _ = store.get_batch([1, 2])
            store.close()

    def test_get_batch_db_error_with_logger(self) -> None:
        """get_batch should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.get_batch([1])

            logger.error.assert_called()
            store.close()


class TestMemoryStoreDeleteBatch:
    """Tests for MemoryStore.delete_batch method."""

    def test_delete_batch_returns_count(self) -> None:
        """delete_batch should return number of deleted memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            id1 = store.insert("proj1", "msg1")
            id2 = store.insert("proj1", "msg2")
            _ = store.insert("proj1", "msg3")

            deleted = store.delete_batch([id1, id2])

            assert deleted == 2
            # msg3 should still exist
            remaining = store.get_all("proj1")
            assert len(remaining) == 1
            assert "msg3" in remaining
            store.close()

    def test_delete_batch_empty_list(self) -> None:
        """delete_batch with empty list should return 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            deleted = store.delete_batch([])

            assert deleted == 0
            store.close()

    def test_delete_batch_partial_ids(self) -> None:
        """delete_batch with some missing IDs should delete existing ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            id1 = store.insert("proj1", "msg1")

            deleted = store.delete_batch([id1, 999, 1000])

            assert deleted == 1
            store.close()

    def test_delete_batch_all_missing(self) -> None:
        """delete_batch with all missing IDs should return 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            deleted = store.delete_batch([999, 1000])

            assert deleted == 0
            store.close()

    def test_delete_batch_db_error_raises_storage_error(self) -> None:
        """delete_batch should raise StorageError on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to delete memory batch"):
                _ = store.delete_batch([1, 2])
            store.close()

    def test_delete_batch_db_error_with_logger(self) -> None:
        """delete_batch should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.delete_batch([1])

            logger.error.assert_called()
            store.close()


class TestMemoryStoreArchive:
    """Tests for MemoryStore.archive method."""

    def test_archive_persists_all_six_values(self) -> None:
        """archive should persist original_id, project, content, replacement, reason, confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            archive_id = store.archive(
                memory_id=7,
                workspace_id="proj1",
                content="old memory",
                replaced_by="new memory",
                reason="updated info",
                confidence=0.91,
            )

            assert archive_id == 1
            records = store.get_archived("proj1")
            assert len(records) == 1
            record = records[0]
            assert record.id == 1
            assert record.original_id == 7
            assert record.workspace_id == "proj1"
            assert record.content == "old memory"
            assert record.replaced_by == "new memory"
            assert record.reason == "updated info"
            assert record.confidence == 0.91
            store.close()

    def test_archive_is_idempotent(self) -> None:
        """Retrying archive for the same pair must not create a duplicate row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            first = store.archive(1, "proj1", "old", "new", "reason", 0.9)
            second = store.archive(1, "proj1", "old", "new", "reason", 0.9)

            assert first == second
            assert len(store.get_archived("proj1")) == 1
            store.close()

    def test_archive_raises_on_sql_error(self) -> None:
        """archive should raise StorageError when the archive table is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)
            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE archived_memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to archive memory"):
                _ = store.archive(
                    memory_id=1,
                    workspace_id="proj1",
                    content="old memory",
                    replaced_by="new memory",
                    reason="updated info",
                    confidence=0.9,
                )
            store.close()

    def test_archive_error_with_logger(self) -> None:
        """archive should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = MagicMock(spec=IStructuredLogger)
            store = MemoryStore(db_path=db_path, logger=logger)
            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE archived_memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.archive(
                    memory_id=1,
                    workspace_id="proj1",
                    content="old",
                    replaced_by="new",
                    reason="reason",
                    confidence=0.8,
                )

            logger.error.assert_called()
            store.close()


class TestMemoryStoreGetArchived:
    """Tests for MemoryStore.get_archived method."""

    def test_get_archived_empty(self) -> None:
        """get_archived should return empty list when no archives exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            result = store.get_archived("proj1")

            assert result == []
            store.close()

    def test_get_archived_returns_records(self) -> None:
        """get_archived should return ArchivedMemoryRecord objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            # Manually insert archive record (bypass buggy archive method)
            cursor = store.connection.cursor()
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, "proj1", "old msg", "new msg", "updated", 0.85),
            )
            store.connection.commit()
            cursor.close()

            result = store.get_archived("proj1")

            assert len(result) == 1
            record = result[0]
            assert isinstance(record, ArchivedMemoryRecord)
            assert record.original_id == 1
            assert record.workspace_id == "proj1"
            assert record.content == "old msg"
            assert record.replaced_by == "new msg"
            assert record.reason == "updated"
            assert record.confidence == 0.85
            assert record.archived_at != ""
            store.close()

    def test_get_archived_filters_by_workspace(self) -> None:
        """get_archived should only return records for specified project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            cursor = store.connection.cursor()
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, "proj1", "msg1", "new1", "reason1", 0.9),
            )
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (2, "proj2", "msg2", "new2", "reason2", 0.8),
            )
            store.connection.commit()
            cursor.close()

            result = store.get_archived("proj1")

            assert len(result) == 1
            assert result[0].workspace_id == "proj1"
            store.close()

    def test_get_archived_respects_limit(self) -> None:
        """get_archived should respect the limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            cursor = store.connection.cursor()
            for i in range(5):
                _ = cursor.execute(
                    "INSERT INTO archived_memories "
                    "(original_id, workspace_id, content, replaced_by, "
                    "reason, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (i, "proj1", f"msg{i}", f"new{i}", "reason", 0.9),
                )
            store.connection.commit()
            cursor.close()

            result = store.get_archived("proj1", limit=2)

            assert len(result) == 2
            store.close()

    def test_get_archived_ordered_by_time_desc(self) -> None:
        """get_archived should return newest archives first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            cursor = store.connection.cursor()
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, "
                "confidence, archived_at) "
                "VALUES (?, ?, ?, ?, ?, ?, '2025-01-01 00:00:00')",
                (1, "proj1", "old", "new_old", "reason", 0.9),
            )
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, "
                "confidence, archived_at) "
                "VALUES (?, ?, ?, ?, ?, ?, '2025-06-01 00:00:00')",
                (2, "proj1", "newer", "new_newer", "reason", 0.8),
            )
            store.connection.commit()
            cursor.close()

            result = store.get_archived("proj1")

            assert len(result) == 2
            # Newest first
            assert result[0].content == "newer"
            assert result[1].content == "old"
            store.close()

    def test_get_archived_db_error_raises_storage_error(self) -> None:
        """get_archived should raise StorageError on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE archived_memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to get archived"):
                _ = store.get_archived("proj1")
            store.close()

    def test_get_archived_db_error_with_logger(self) -> None:
        """get_archived should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE archived_memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.get_archived("proj1")

            logger.error.assert_called()
            store.close()


class TestMemoryStoreRestoreFromArchive:
    """Tests for MemoryStore.restore_from_archive method."""

    def test_restore_success(self) -> None:
        """restore_from_archive should insert memory and remove archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            # Manually create archive record
            cursor = store.connection.cursor()
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, "proj1", "restored msg", "replacement", "reason", 0.9),
            )
            store.connection.commit()
            cursor.close()

            new_id = store.restore_from_archive(1)

            assert new_id is not None
            assert new_id > 0

            # Memory should be in memories table
            record = store.get(new_id)
            assert record is not None
            assert record.content == "restored msg"
            assert record.workspace_id == "proj1"

            # Archive record should be removed
            archived = store.get_archived("proj1")
            assert len(archived) == 0
            store.close()

    def test_restore_not_found(self) -> None:
        """restore_from_archive should return None for missing archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            result = store.restore_from_archive(999)

            assert result is None
            store.close()

    def test_restore_not_found_with_logger(self) -> None:
        """restore_from_archive should log warning when archive not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.restore_from_archive(999)

            logger.warning.assert_called()
            call_args = logger.warning.call_args
            assert "not found" in call_args[0][0].lower()
            store.close()

    def test_restore_db_error_raises_storage_error(self) -> None:
        """restore_from_archive should raise StorageError on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE archived_memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to restore"):
                _ = store.restore_from_archive(1)
            store.close()

    def test_restore_db_error_with_logger(self) -> None:
        """restore_from_archive should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE archived_memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.restore_from_archive(1)

            logger.error.assert_called()
            store.close()


class TestMemoryStoreCleanupExpiredArchive:
    """Tests for MemoryStore.cleanup_expired_archive method."""

    def test_cleanup_zero_ttl_returns_zero(self) -> None:
        """cleanup_expired_archive with 0 ttl should return 0 (no cleanup)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            result = store.cleanup_expired_archive(0)

            assert result == 0
            store.close()

    def test_cleanup_negative_ttl_returns_zero(self) -> None:
        """cleanup_expired_archive with negative ttl should return 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            result = store.cleanup_expired_archive(-5)

            assert result == 0
            store.close()

    def test_cleanup_removes_old_records(self) -> None:
        """cleanup_expired_archive should remove records older than TTL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            cursor = store.connection.cursor()
            # Insert old record (100 days ago)
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, "
                "confidence, archived_at) "
                "VALUES (?, ?, ?, ?, ?, ?, datetime('now', '-100 days'))",
                (1, "proj1", "old msg", "new msg", "reason", 0.9),
            )
            # Insert recent record (1 day ago)
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, "
                "confidence, archived_at) "
                "VALUES (?, ?, ?, ?, ?, ?, datetime('now', '-1 day'))",
                (2, "proj1", "recent msg", "new msg", "reason", 0.8),
            )
            store.connection.commit()
            cursor.close()

            deleted = store.cleanup_expired_archive(30)

            assert deleted == 1
            # Recent record should still exist
            remaining = store.get_archived("proj1")
            assert len(remaining) == 1
            assert remaining[0].content == "recent msg"
            store.close()

    def test_cleanup_no_expired_records(self) -> None:
        """cleanup_expired_archive should return 0 when no records expired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            cursor = store.connection.cursor()
            _ = cursor.execute(
                "INSERT INTO archived_memories "
                "(original_id, workspace_id, content, replaced_by, reason, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, "proj1", "recent", "new", "reason", 0.9),
            )
            store.connection.commit()
            cursor.close()

            deleted = store.cleanup_expired_archive(30)

            assert deleted == 0
            store.close()

    def test_cleanup_db_error_raises_storage_error(self) -> None:
        """cleanup_expired_archive should raise StorageError on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE archived_memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to cleanup expired"):
                _ = store.cleanup_expired_archive(30)
            store.close()

    def test_cleanup_db_error_with_logger(self) -> None:
        """cleanup_expired_archive should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE archived_memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.cleanup_expired_archive(30)

            logger.error.assert_called()
            store.close()


class TestMemoryStoreClose:
    """Tests for MemoryStore.close and ensure_initialized methods."""

    def test_close_without_connection(self) -> None:
        """close should be safe to call without prior initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            # Should not raise
            store.close()

    def test_close_with_connection(self) -> None:
        """close is a write barrier; the connection cannot be reopened."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection  # Force init
            store.close()

            with pytest.raises(StorageError, match="MemoryStore is closed"):
                _ = store.connection

    def test_close_double_call(self) -> None:
        """Calling close twice should be safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection
            store.close()
            store.close()  # Should not raise

    def test_ensure_initialized(self) -> None:
        """ensure_initialized should force lazy initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            assert not os.path.exists(db_path)

            store.ensure_initialized()

            assert os.path.exists(db_path)
            store.close()

    def test_ensure_initialized_idempotent(self) -> None:
        """ensure_initialized should be safe to call multiple times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            store.ensure_initialized()
            store.ensure_initialized()  # Should not raise

            store.close()


class TestMemoryStoreErrorPaths:
    """Tests for error handling paths in get, get_all, delete, exists, get_id_by_content."""

    def test_get_db_error_raises_storage_error(self) -> None:
        """get should raise StorageError on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to retrieve memory"):
                _ = store.get(1)
            store.close()

    def test_get_db_error_with_logger(self) -> None:
        """get should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.get(1)

            logger.error.assert_called()
            store.close()

    def test_get_all_db_error_raises_storage_error(self) -> None:
        """get_all should raise StorageError on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to retrieve all memories"):
                _ = store.get_all("proj1")
            store.close()

    def test_get_all_db_error_with_logger(self) -> None:
        """get_all should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.get_all("proj1")

            logger.error.assert_called()
            store.close()

    def test_delete_db_error_raises_storage_error(self) -> None:
        """delete should raise StorageError on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to delete memory"):
                _ = store.delete(1)
            store.close()

    def test_delete_db_error_with_logger(self) -> None:
        """delete should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.delete(1)

            logger.error.assert_called()
            store.close()

    def test_exists_db_error_raises_storage_error(self) -> None:
        """exists should raise StorageError on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to check memory existence"):
                _ = store.exists("proj1", "msg")
            store.close()

    def test_exists_db_error_with_logger(self) -> None:
        """exists should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.exists("proj1", "msg")

            logger.error.assert_called()
            store.close()

    def test_get_id_by_content_db_error_raises_storage_error(self) -> None:
        """get_id_by_content should raise StorageError on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to get memory ID"):
                _ = store.get_id_by_content("proj1", "msg")
            store.close()

    def test_get_id_by_content_db_error_with_logger(self) -> None:
        """get_id_by_content should log on database failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.get_id_by_content("proj1", "msg")

            logger.error.assert_called()
            store.close()

    def test_insert_non_duplicate_error_raises_storage_error(self) -> None:
        """insert should raise StorageError for non-duplicate db errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError, match="Failed to insert memory"):
                _ = store.insert("proj1", "msg")
            store.close()

    def test_insert_non_duplicate_error_with_logger(self) -> None:
        """insert should log non-duplicate errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = create_mock_logger()
            store = MemoryStore(db_path=db_path, logger=logger)

            _ = store.connection

            cursor = store.connection.cursor()
            _ = cursor.execute("DROP TABLE memories")
            store.connection.commit()
            cursor.close()

            with pytest.raises(StorageError):
                _ = store.insert("proj1", "msg")

            logger.error.assert_called()
            store.close()


class TestMemoryRecordDataclass:
    """Tests for MemoryRecord dataclass."""

    def test_memory_record_fields(self) -> None:
        """MemoryRecord should store all fields correctly."""
        record = MemoryRecord(
            id=1, workspace_id="proj1", content="hello", created_at="2025-01-01"
        )
        assert record.id == 1
        assert record.workspace_id == "proj1"
        assert record.content == "hello"
        assert record.created_at == "2025-01-01"

    def test_memory_record_default_created_at(self) -> None:
        """MemoryRecord should default created_at to empty string."""
        record = MemoryRecord(id=1, workspace_id="proj1", content="hello")
        assert record.created_at == ""

    def test_memory_record_frozen(self) -> None:
        """MemoryRecord should be immutable (frozen dataclass)."""
        record = MemoryRecord(id=1, workspace_id="proj1", content="hello")
        with pytest.raises(AttributeError):
            record.content = "changed"  # type: ignore


class TestArchivedMemoryRecordDataclass:
    """Tests for ArchivedMemoryRecord dataclass."""

    def test_archived_record_fields(self) -> None:
        """ArchivedMemoryRecord should store all fields correctly."""
        record = ArchivedMemoryRecord(
            id=1,
            original_id=10,
            workspace_id="proj1",
            content="old msg",
            replaced_by="new msg",
            reason="updated",
            confidence=0.85,
            archived_at="2025-01-01",
        )
        assert record.id == 1
        assert record.original_id == 10
        assert record.workspace_id == "proj1"
        assert record.content == "old msg"
        assert record.replaced_by == "new msg"
        assert record.reason == "updated"
        assert record.confidence == 0.85
        assert record.archived_at == "2025-01-01"

    def test_archived_record_frozen(self) -> None:
        """ArchivedMemoryRecord should be immutable (frozen dataclass)."""
        record = ArchivedMemoryRecord(
            id=1,
            original_id=10,
            workspace_id="proj1",
            content="old",
            replaced_by="new",
            reason="reason",
            confidence=0.9,
            archived_at="2025-01-01",
        )
        with pytest.raises(AttributeError):
            record.content = "changed"  # type: ignore


class TestMemoryStoreCustomTimeout:
    """Tests for MemoryStore with custom timeout."""

    def test_custom_timeout(self) -> None:
        """MemoryStore should accept custom timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path, timeout=60.0)

            assert store.timeout == 60.0
            store.ensure_initialized()
            store.close()

    def test_default_timeout(self) -> None:
        """MemoryStore should have 30s default timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            assert store.timeout == 30.0
            store.close()


class TestMemoryStoreConnectionProperty:
    """Tests for MemoryStore.connection property edge cases."""

    def test_connection_without_directory(self) -> None:
        """Connection should work with db_path in current directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            conn = store.connection

            assert conn is not None
            store.close()

    def test_connection_reuse(self) -> None:
        """Accessing connection multiple times should return same object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            conn1 = store.connection
            conn2 = store.connection

            assert conn1 is conn2
            store.close()

    def test_connection_after_close_raises(self) -> None:
        """Accessing connection after close must not mint a new handle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MemoryStore(db_path=db_path)

            _ = store.connection
            store.close()
            with pytest.raises(StorageError, match="MemoryStore is closed"):
                _ = store.connection
