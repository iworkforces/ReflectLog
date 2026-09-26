"""Unit tests for USearchEngine."""

from collections.abc import Sequence
import gc
import multiprocessing
import os
import tempfile
from typing import Generator, cast
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.enums import EmbedderProvider
from reflectlog.core.exceptions import StorageError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.types import Embeddings
from reflectlog.infrastructure.embedding_identity import ensure_embedding_identity
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine


def create_mock_logger() -> IStructuredLogger:
    """Create a properly typed mock logger for testing."""
    return cast(IStructuredLogger, MagicMock(spec=StructuredLogger))


class MockEmbedder(Embeddings):
    """Mock embedder for testing."""

    def __init__(self, dims: int = 128) -> None:
        super().__init__()
        self.dims = dims
        self.call_count = 0

    def embed_query(self, text: str) -> list[float]:
        """Return deterministic embeddings based on text hash."""
        self.call_count += 1
        np.random.seed(hash(text) % (2**32))
        embedding: list[float] = np.random.randn(self.dims).astype(np.float32).tolist()
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class FailingQueryEmbedder(MockEmbedder):
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("Embedding API down")


class FailingBatchEmbedder(MockEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Embedding batch failed")


class MismatchedSizeEmbedder(MockEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dims]


class ToggleableFailingQueryEmbedder(MockEmbedder):
    def __init__(self, dims: int = 128) -> None:
        super().__init__(dims)
        self.should_fail = False

    def embed_query(self, text: str) -> list[float]:
        if self.should_fail:
            raise RuntimeError("API failure")
        return super().embed_query(text)


def _spawn_exit_immediately() -> None:
    return


@pytest.fixture
def temp_engine() -> Generator[tuple[USearchConfig, MockEmbedder, str], None, None]:
    """Create a temporary engine configuration and embedder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = USearchConfig(
            workspace_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            embedder_provider=EmbedderProvider.OPENAI,
            embedding_model="test/mock-128",
        )
        embedder = MockEmbedder(dims=128)
        yield config, embedder, tmpdir


class TestUSearchConfigFromAppConfig:
    """Tests for USearchConfig.from_app_config factory method."""

    def test_creates_config_from_app_config(self) -> None:
        """Factory should create config from application Config."""
        mock_config = MagicMock()
        mock_config.workspace_id = "test-project"
        mock_config.embedder_provider = EmbedderProvider.OPENAI
        mock_config.embedding_model = "openai/text-embedding-3-large"
        mock_config.embedding_dims = 3072
        mock_config.qwen_embedding_dims = 4096
        mock_config.usearch_index_path = "indexes/test-project/usearch"
        mock_config.usearch_exact_search = False
        mock_config.usearch_exact_search_threshold = 256

        with patch("os.getcwd", return_value="/tmp"):
            config = USearchConfig.from_config(mock_config)

        assert config.workspace_id == "test-project"
        assert config.embedding_dims == 3072
        assert "test-project" in config.index_path
        assert "test-project" in config.db_path

    def test_uses_qwen_dims_for_langchain_provider(self) -> None:
        """Factory should use qwen_embedding_dims for langchain provider."""
        mock_config = MagicMock()
        mock_config.workspace_id = "test-project"
        mock_config.embedder_provider = EmbedderProvider.LANGCHAIN
        mock_config.embedding_model = "Qwen/Qwen3-Embedding-4B"
        mock_config.embedding_dims = 3072
        mock_config.qwen_embedding_dims = 4096
        mock_config.usearch_index_path = "indexes/test-project/usearch"
        mock_config.usearch_exact_search = False
        mock_config.usearch_exact_search_threshold = 256

        with patch("os.getcwd", return_value="/tmp"):
            config = USearchConfig.from_config(mock_config)

        assert config.embedding_dims == 4096


class TestUSearchEngineInitialization:
    """Tests for USearchEngine initialization."""

    def test_engine_name_is_usearch(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Engine name should be 'usearch'."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        try:
            assert engine.name == "usearch"
        finally:
            engine.close()

    def test_lazy_initialization(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Index and memory store should be lazily initialized."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            # Before accessing, files should not exist
            assert not os.path.exists(config.db_path)

            # After ensure_initialized, files should exist
            engine.ensure_initialized()
            assert os.path.exists(config.db_path)
        finally:
            engine.close()

    def test_is_ready_false_until_initialized(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """is_ready must not restore the index."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        try:
            assert engine.is_ready() is False
            assert engine._index is None
            engine.ensure_initialized()
            assert engine.is_ready() is True
        finally:
            engine.close()


class TestUSearchEngineAdd:
    """Tests for USearchEngine.add method."""

    def test_add_content(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Add should store content in both index and SQLite."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)

            # Content should be in SQLite
            contents = engine.get_all("user1")
            assert "Hello world" in contents

            # Embedder should have been called
            assert embedder.call_count == 1
        finally:
            engine.close()

    def test_add_indexes_document_role_vector(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, _, _ = temp_engine
        embedder = MagicMock(spec=Embeddings)
        query_vector = [1.0, *([0.0] * 127)]
        document_vector = [0.0, 1.0, *([0.0] * 126)]
        embedder.embed_query.return_value = query_vector
        embedder.embed_documents.return_value = [document_vector]
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "persisted document", infer=False)
            memory_id = engine.get_id_by_content("user1", "persisted document")

            assert memory_id is not None
            embedder.embed_documents.assert_called_once_with(["persisted document"])
            embedder.embed_query.assert_not_called()
            stored_vector = engine.index.get(memory_id)
            assert stored_vector is not None
            np.testing.assert_allclose(stored_vector, document_vector)
        finally:
            engine.close()

    @pytest.mark.parametrize(
        "document_vectors",
        [[], [[]], [[0.0] * 128, [1.0] * 128]],
    )
    def test_add_rejects_invalid_document_embedding_batch(
        self,
        temp_engine: tuple[USearchConfig, MockEmbedder, str],
        document_vectors: list[list[float]],
    ) -> None:
        config, _, _ = temp_engine
        embedder = MagicMock(spec=Embeddings)
        embedder.embed_query.return_value = [1.0, *([0.0] * 127)]
        embedder.embed_documents.return_value = document_vectors
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            with pytest.raises(RuntimeError):
                engine.add("user1", "persisted document", infer=False)

            assert engine.count("user1") == 0
            embedder.embed_query.assert_not_called()
        finally:
            engine.close()

    def test_add_skips_duplicates(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Add should skip duplicate contents."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.add("user1", "Hello world", infer=False)  # Duplicate

            # Only one content should be stored
            contents = engine.get_all("user1")
            assert len(contents) == 1

            # Embedder should only be called once
            assert embedder.call_count == 1
        finally:
            engine.close()

    def test_add_with_infer_logs_warning(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Add with infer=True should log a warning."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "Hello world", infer=True)

            # Should log warning about infer not being supported
            logger.warning.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineSearch:
    """Tests for USearchEngine.search method."""

    def test_search_empty_index(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search on empty index should return empty list."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.ensure_initialized()

            results = engine.search("hello", "user1", limit=5)

            assert results == []
        finally:
            engine.close()

    def test_search_returns_matches(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search should return matching contents with scores."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.add("user1", "Goodbye world", infer=False)

            results = engine.search("Hello", "user1", limit=5)

            assert len(results) > 0
            # Results should be (content, score, created_at) tuples
            assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
        finally:
            engine.close()

    def test_search_filters_by_workspace_id(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search should only return contents for the specified project."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "User1 content", infer=False)
            engine.add("user2", "User2 content", infer=False)

            results = engine.search("content", "user1", limit=5)

            # Should only return user1's content
            contents = [r[0] for r in results]
            assert "User1 content" in contents
            assert "User2 content" not in contents
        finally:
            engine.close()

    def test_search_respects_limit(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search should respect the limit parameter."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            # Add many contents
            for i in range(10):
                engine.add("user1", f"Content {i}", infer=False)

            results = engine.search("content", "user1", limit=3)

            assert len(results) <= 3
        finally:
            engine.close()


class TestUSearchEngineGetAll:
    """Tests for USearchEngine.get_all method."""

    def test_get_all_empty(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Get all on empty store should return empty list."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            contents = engine.get_all("user1")
            assert contents == []
        finally:
            engine.close()

    def test_get_all_returns_contents(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Get all should return all contents for user."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Content 1", infer=False)
            engine.add("user1", "Content 2", infer=False)

            contents = engine.get_all("user1")

            assert len(contents) == 2
            assert "Content 1" in contents
            assert "Content 2" in contents
        finally:
            engine.close()


class TestUSearchEngineDelete:
    """Tests for USearchEngine.delete method."""

    def test_delete_existing_content(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Delete should remove content from both index and SQLite."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)

            # Get the content ID from SQLite
            content_id = engine.memory_store.get_id_by_content("user1", "Hello world")
            assert content_id is not None

            engine.delete(str(content_id))

            # Content should be removed
            contents = engine.get_all("user1")
            assert "Hello world" not in contents
        finally:
            engine.close()

    def test_delete_nonexistent_logs_warning(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Delete of non-existent content should log warning."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            engine.delete("999")

            # Should log warning about not found
            logger.warning.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_delete_invalid_id_raises_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Delete with invalid ID format should raise error."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.ensure_initialized()

            with pytest.raises(RuntimeError, match="Invalid memory_id format"):
                engine.delete("not-an-integer")
        finally:
            engine.close()


class TestUSearchEngineCommit:
    """Tests for USearchEngine.commit method."""

    def test_commit_saves_index(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Commit should save the USearch index to disk."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.commit()

            # Index file should exist
            assert os.path.exists(config.index_path)
        finally:
            engine.close()


class TestUSearchEnginePersistence:
    """Tests for USearchEngine persistence across restarts."""

    def test_persistence_across_restart(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Contents should persist across engine restart."""
        config, embedder, _ = temp_engine

        # First engine instance
        engine1 = USearchEngine(config=config, embedder=embedder)
        try:
            engine1.add("user1", "Persistent content", infer=False)
            engine1.commit()
        finally:
            engine1.close()

        # Second engine instance (simulating restart)
        engine2 = USearchEngine(config=config, embedder=embedder)
        try:
            # Content should still be accessible
            contents = engine2.get_all("user1")
            assert "Persistent content" in contents
        finally:
            engine2.close()


class TestUSearchEngineExactSearch:
    """Tests for USearchEngine exact vs approximate search modes."""

    def test_default_uses_exact_search(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Default config should use HNSW unless exact search is forced."""
        config, embedder, _ = temp_engine
        default_config = USearchConfig(
            workspace_id=config.workspace_id,
            index_path=config.index_path,
            db_path=config.db_path,
            embedding_dims=config.embedding_dims,
            embedder_provider=config.embedder_provider,
            embedding_model=config.embedding_model,
        )
        engine = USearchEngine(config=default_config, embedder=embedder)

        try:
            assert default_config.exact_search is False
            assert default_config.exact_search_threshold == 0
            assert engine._should_use_exact_search() is False
        finally:
            engine.close()

    def test_exact_search_when_forced(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Should use exact search when exact_search=True."""
        _, embedder, tmpdir = temp_engine
        config = USearchConfig(
            workspace_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            embedder_provider=EmbedderProvider.OPENAI,
            embedding_model="test/mock-128",
            exact_search=True,  # Force exact search
        )
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            assert engine._should_use_exact_search() is True
        finally:
            engine.close()

    def test_exact_search_auto_switch_below_threshold(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Should auto-switch to exact search when index size < threshold."""
        _, embedder, tmpdir = temp_engine
        config = USearchConfig(
            workspace_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            embedder_provider=EmbedderProvider.OPENAI,
            embedding_model="test/mock-128",
            exact_search=False,
            exact_search_threshold=1000,  # Auto-switch when < 1000 vectors
        )
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            # Add a few contents (below threshold)
            for i in range(5):
                engine.add("user1", f"Content {i}", infer=False)

            # Index size is 5, below threshold of 1000
            assert len(engine.index) == 5
            assert engine._should_use_exact_search() is True
        finally:
            engine.close()

    def test_approximate_search_above_threshold(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Should use approximate search when index size >= threshold."""
        _, embedder, tmpdir = temp_engine
        config = USearchConfig(
            workspace_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            embedder_provider=EmbedderProvider.OPENAI,
            embedding_model="test/mock-128",
            exact_search=False,
            exact_search_threshold=3,  # Auto-switch when < 3 vectors
        )
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            # Add contents to exceed threshold
            for i in range(5):
                engine.add("user1", f"Content {i}", infer=False)

            # Index size is 5, above threshold of 3
            assert len(engine.index) == 5
            assert engine._should_use_exact_search() is False
        finally:
            engine.close()

    def test_exact_search_returns_valid_results(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Exact search should return valid search results."""
        _, embedder, tmpdir = temp_engine
        config = USearchConfig(
            workspace_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            embedder_provider=EmbedderProvider.OPENAI,
            embedding_model="test/mock-128",
            exact_search=True,  # Force exact search
        )
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.add("user1", "Goodbye world", infer=False)

            results = engine.search("Hello", "user1", limit=5)

            # Should return valid results (3-tuples: content, score, created_at)
            assert len(results) > 0
            assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
        finally:
            engine.close()

    def test_search_logs_search_mode(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search should log the search mode being used."""
        _, embedder, tmpdir = temp_engine
        config = USearchConfig(
            workspace_id="test",
            index_path=os.path.join(tmpdir, "index.usearch"),
            db_path=os.path.join(tmpdir, "messages.db"),
            embedding_dims=128,
            embedder_provider=EmbedderProvider.OPENAI,
            embedding_model="test/mock-128",
            exact_search=True,
        )
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "Hello world", infer=False)
            engine.search("Hello", "user1", limit=5)

            assert engine.config.exact_search is True
        finally:
            engine.close()

    def test_config_from_app_config_includes_exact_search(self) -> None:
        """USearchConfig.from_app_config should include exact search settings."""
        mock_config = MagicMock()
        mock_config.workspace_id = "test-project"
        mock_config.embedder_provider = EmbedderProvider.OPENAI
        mock_config.embedding_model = "openai/text-embedding-3-large"
        mock_config.embedding_dims = 3072
        mock_config.qwen_embedding_dims = 4096
        mock_config.usearch_exact_search = True
        mock_config.usearch_exact_search_threshold = 5000
        mock_config.usearch_index_path = "indexes/test-project/usearch"

        with patch("os.getcwd", return_value="/tmp"):
            config = USearchConfig.from_config(mock_config)

        assert config.exact_search is True
        assert config.exact_search_threshold == 5000
        assert config.index_path.endswith("vectors.usearch")


class TestUSearchConfigFromDict:
    """Tests for USearchConfig.from_dict factory method."""

    def test_from_dict_with_full_data(self) -> None:
        """from_dict should create config from a complete dictionary."""
        data = {
            "workspace_id": "my-proj",
            "index_path": "/tmp/index.usearch",
            "db_path": "/tmp/messages.db",
            "embedding_dims": 256,
            "embedder_provider": "openai",
            "embedding_model": "test/mock-256",
            "metric": "l2",
            "connectivity": 32,
            "expansion_add": 256,
            "expansion_search": 128,
            "exact_search": False,
            "exact_search_threshold": 500,
        }
        config = USearchConfig.from_dict(data)

        assert config.workspace_id == "my-proj"
        assert config.index_path == "/tmp/index.usearch"
        assert config.db_path == "/tmp/messages.db"
        assert config.embedding_dims == 256
        assert config.metric == "l2"
        assert config.connectivity == 32
        assert config.expansion_add == 256
        assert config.expansion_search == 128
        assert config.exact_search is False
        assert config.exact_search_threshold == 500

    def test_from_dict_with_defaults(self) -> None:
        """from_dict should use defaults for missing keys."""
        config = USearchConfig.from_dict(
            {"embedder_provider": "openai", "embedding_model": "test/mock-3072"}
        )

        assert config.workspace_id == ""
        assert config.index_path == ""
        assert config.db_path == ""
        assert config.embedding_dims == 3072
        assert config.metric == "cos"
        assert config.connectivity == 16
        assert config.expansion_add == 128
        assert config.expansion_search == 64
        assert config.exact_search is False
        assert config.exact_search_threshold == 0


class TestUSearchEngineInitWithDict:
    """Tests for USearchEngine initialization with dict config."""

    def test_init_with_dict_config(self) -> None:
        """Engine should accept a dict and convert to USearchConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dict = {
                "workspace_id": "test",
                "index_path": os.path.join(tmpdir, "index.usearch"),
                "db_path": os.path.join(tmpdir, "messages.db"),
                "embedding_dims": 128,
                "embedder_provider": "openai",
                "embedding_model": "test/mock-128",
            }
            embedder = MockEmbedder(dims=128)
            engine = USearchEngine(config=config_dict, embedder=embedder)

            try:
                assert isinstance(engine.config, USearchConfig)
                assert engine.config.workspace_id == "test"
            finally:
                engine.close()


class TestUSearchEngineIndexInit:
    """Tests for USearch index lazy initialization error paths."""

    def test_index_restore_returns_none_creates_new(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """When Index.restore returns None, should create a new index."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            idx = engine.index
            assert idx is not None
            logger.info.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_corrupt_index_file_with_sqlite_rows_fails_closed(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """An existing corrupt index file must not be replaced when SQLite has rows."""
        import sqlite3

        from reflectlog.core.exceptions import InitializationError

        config, embedder, _ = temp_engine
        ensure_embedding_identity(config)
        os.makedirs(os.path.dirname(config.db_path), exist_ok=True)
        os.makedirs(os.path.dirname(config.index_path), exist_ok=True)
        with open(config.index_path, "wb") as handle:
            handle.write(b"not-an-index")
        connection = sqlite3.connect(config.db_path)
        try:
            _ = connection.execute(
                "CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT)"
            )
            _ = connection.execute("INSERT INTO memories(content) VALUES ('kept')")
            connection.commit()
        finally:
            connection.close()

        engine = USearchEngine(config=config, embedder=embedder)
        with patch("reflectlog.infrastructure.usearch_engine.Index") as mock_index_cls:
            mock_index_cls.restore.side_effect = RuntimeError("corrupt")
            with pytest.raises(InitializationError, match="Refusing to create"):
                _ = engine.index

    def test_corrupt_index_with_unreadable_sqlite_fails_closed(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """A present HNSW file plus unreadable SQLite must not be overwritten."""
        from reflectlog.core.exceptions import InitializationError

        config, embedder, _ = temp_engine
        ensure_embedding_identity(config)
        os.makedirs(os.path.dirname(config.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(config.db_path), exist_ok=True)
        with open(config.index_path, "wb") as handle:
            handle.write(b"not-an-index")
        with open(config.db_path, "wb") as handle:
            handle.write(b"")

        engine = USearchEngine(config=config, embedder=embedder)
        with (
            patch("reflectlog.infrastructure.usearch_engine.Index") as mock_index_cls,
            patch(
                "reflectlog.infrastructure.usearch_engine._sqlite_memory_count",
                return_value=None,
            ),
        ):
            mock_index_cls.restore.side_effect = RuntimeError("corrupt")
            with pytest.raises(InitializationError, match="unreadable"):
                _ = engine.index

    def test_corrupt_index_with_missing_sqlite_fails_closed(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from reflectlog.core.exceptions import InitializationError

        config, embedder, _ = temp_engine
        ensure_embedding_identity(config)
        os.makedirs(os.path.dirname(config.index_path), exist_ok=True)
        with open(config.index_path, "wb") as handle:
            handle.write(b"not-an-index")
        if os.path.exists(config.db_path):
            os.remove(config.db_path)

        engine = USearchEngine(config=config, embedder=embedder)
        with patch("reflectlog.infrastructure.usearch_engine.Index") as mock_index_cls:
            mock_index_cls.restore.side_effect = RuntimeError("corrupt")
            with pytest.raises(InitializationError, match="missing"):
                _ = engine.index

    def test_missing_index_with_populated_sqlite_fails_closed(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from reflectlog.core.exceptions import InitializationError
        from reflectlog.infrastructure.memory_store import MemoryStore

        config, embedder, _ = temp_engine
        ensure_embedding_identity(config)
        store = MemoryStore(db_path=config.db_path)
        _ = store.insert(config.workspace_id, "already stored")
        store.close()
        if os.path.exists(config.index_path):
            os.remove(config.index_path)

        engine = USearchEngine(config=config, embedder=embedder)
        with patch("reflectlog.infrastructure.usearch_engine.Index") as mock_index_cls:
            mock_index_cls.restore.side_effect = FileNotFoundError("no file")
            with pytest.raises(InitializationError, match="missing but SQLite"):
                _ = engine.index
            mock_index_cls.assert_not_called()

    def test_new_index_is_not_saved_until_commit(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from reflectlog.infrastructure.memory_store import MemoryStore

        config, embedder, _ = temp_engine
        if os.path.exists(config.db_path):
            os.remove(config.db_path)
        if os.path.exists(config.index_path):
            os.remove(config.index_path)

        created = MagicMock()
        created.__len__.return_value = 0
        engine = USearchEngine(config=config, embedder=embedder)
        with patch("reflectlog.infrastructure.usearch_engine.Index") as mock_index_cls:
            mock_index_cls.restore.side_effect = FileNotFoundError("no file")
            mock_index_cls.return_value = created
            _ = engine.index
            created.save.assert_not_called()
        store = MemoryStore(db_path=config.db_path)
        assert store.count(config.workspace_id) == 0
        store.close()

    def test_empty_restored_index_with_populated_sqlite_fails_closed(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from reflectlog.core.exceptions import InitializationError
        from reflectlog.infrastructure.memory_store import MemoryStore

        config, embedder, _ = temp_engine
        ensure_embedding_identity(config)
        store = MemoryStore(db_path=config.db_path)
        _ = store.insert(config.workspace_id, "already stored")
        store.close()

        restored = MagicMock()
        restored.__len__.return_value = 0
        engine = USearchEngine(config=config, embedder=embedder)
        with patch("reflectlog.infrastructure.usearch_engine.Index") as mock_index_cls:
            mock_index_cls.restore.return_value = restored
            with pytest.raises(InitializationError, match="empty"):
                _ = engine.index

    def test_restore_success_with_missing_sqlite_fails_closed(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from reflectlog.core.exceptions import InitializationError

        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        restored = MagicMock()
        restored.__len__.return_value = 3
        with (
            patch("reflectlog.infrastructure.usearch_engine.Index") as mock_index_cls,
            patch(
                "reflectlog.infrastructure.usearch_engine.os.path.exists",
                return_value=False,
            ),
        ):
            mock_index_cls.restore.return_value = restored
            with pytest.raises(InitializationError, match="missing"):
                _ = engine.index
            mock_index_cls.assert_not_called()

    def test_index_init_failure_raises_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """If both restore and create fail, should raise RuntimeError."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        with patch("reflectlog.infrastructure.usearch_engine.Index") as mock_index_cls:
            mock_index_cls.restore.side_effect = FileNotFoundError("no file")
            mock_index_cls.side_effect = MemoryError("out of memory")

            with pytest.raises(
                RuntimeError, match="Failed to initialize USearch index"
            ):
                _ = engine.index

            logger.error.assert_called()  # type: ignore

    def test_index_init_logs_debug_on_create(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Creating new index should log debug and info messages."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            _ = engine.index
            debug_calls = [str(c) for c in logger.debug.call_args_list]  # type: ignore
            info_calls = [str(c) for c in logger.info.call_args_list]  # type: ignore
            assert any("not found" in c.lower() for c in debug_calls)
            assert any("Created new" in c for c in info_calls)
        finally:
            engine.close()

    def test_index_loaded_from_existing_file(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Restoring an existing index should log loading info."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()

        engine1 = USearchEngine(config=config, embedder=embedder)
        try:
            engine1.add("test", "Hello", infer=False)
            engine1.commit()
        finally:
            engine1.close()

        engine2 = USearchEngine(config=config, embedder=embedder, logger=logger)
        try:
            _ = engine2.index
            info_calls = [str(c) for c in logger.info.call_args_list]  # type: ignore
            assert any("Loaded existing" in c for c in info_calls)
        finally:
            engine2.close()


class TestUSearchEngineAddErrorPaths:
    """Tests for add() error handling paths."""

    def test_add_embedding_failure_rolls_back(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, _, _ = temp_engine
        failing_embedder = FailingQueryEmbedder(dims=128)
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=failing_embedder, logger=logger)

        try:
            with pytest.raises(RuntimeError, match="Failed to generate embedding"):
                engine.add("user1", "rollback test", infer=False)

            contents = engine.get_all("user1")
            assert "rollback test" not in contents
            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_add_duplicate_via_storage_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """StorageError with "Duplicate message" should be silently skipped."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            original_store = engine.memory_store
            original_store.close()
            mock_store = MagicMock()
            mock_store.insert.side_effect = StorageError("Duplicate memory detected")
            object.__setattr__(engine, "_memory_store", mock_store)

            engine.add("user1", "dup msg", infer=False)

            logger.debug.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_add_unexpected_exception_wraps_in_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Unexpected exceptions in add should be wrapped in RuntimeError."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            original_store = engine.memory_store
            original_store.close()
            mock_store = MagicMock()
            mock_store.insert.side_effect = TypeError("unexpected type error")
            object.__setattr__(engine, "_memory_store", mock_store)

            with pytest.raises(RuntimeError, match="Failed to add memory"):
                engine.add("user1", "error msg", infer=False)

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_add_non_duplicate_runtime_error_reraises(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """RuntimeError without "Duplicate message" should be re-raised."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            original_store = engine.memory_store
            original_store.close()
            mock_store = MagicMock()
            mock_store.insert.side_effect = RuntimeError("connection lost")
            object.__setattr__(engine, "_memory_store", mock_store)

            with pytest.raises(RuntimeError, match="connection lost"):
                engine.add("user1", "error msg", infer=False)

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineAddBatch:
    """Tests for add_batch() method."""

    def test_add_batch_empty_list(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """add_batch with empty list should return empty list."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            result = engine.add_batch("user1", [], infer=False)
            assert result == []
        finally:
            engine.close()

    def test_add_batch_success(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """add_batch should add multiple contents and return them."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            contents = ["Batch msg 1", "Batch msg 2", "Batch msg 3"]
            result = engine.add_batch("user1", contents, infer=False)

            assert len(result) == 3
            assert set(result) == set(contents)

            all_msgs = engine.get_all("user1")
            assert len(all_msgs) == 3

            logger.debug.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_add_batch_with_infer_logs_warning(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """add_batch with infer=True should log a warning."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add_batch("user1", ["msg"], infer=True)
            warning_calls = [str(c) for c in logger.warning.call_args_list]  # type: ignore
            assert any("infer=True" in c for c in warning_calls)
        finally:
            engine.close()

    def test_add_batch_all_duplicates_returns_empty(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """add_batch where insert_many returns empty should return empty list."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        real_store = None

        try:
            engine.ensure_initialized()
            real_store = engine._memory_store
            mock_store = MagicMock()
            mock_store.insert_many.return_value = []
            object.__setattr__(engine, "_memory_store", mock_store)

            result = engine.add_batch("user1", ["dup1", "dup2"], infer=False)
            assert result == []
        finally:
            if real_store is not None:
                real_store.close()
            engine.close()

    def test_add_batch_embedding_failure_rolls_back(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, _, _ = temp_engine
        failing_embedder = FailingBatchEmbedder(dims=128)
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=failing_embedder, logger=logger)

        try:
            engine.ensure_initialized()

            with pytest.raises(RuntimeError, match="Failed to add memory batch"):
                engine.add_batch("user1", ["rb1", "rb2"], infer=False)

            contents = engine.get_all("user1")
            assert "rb1" not in contents
            assert "rb2" not in contents
        finally:
            engine.close()

    def test_add_batch_index_failure_rolls_back_sqlite(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """A mid-batch index.add failure must not leave SQLite rows behind."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        try:
            engine.ensure_initialized()
            original_add = engine.index.add
            calls = {"n": 0}

            def boom(
                key: int, vector: npt.NDArray[np.float32] | Sequence[float]
            ) -> None:
                calls["n"] += 1
                if calls["n"] > 1:
                    raise RuntimeError("index full")
                original_add(key, vector)

            engine.index.add = boom  # ty: ignore[invalid-assignment]
            with pytest.raises(RuntimeError, match="Failed to add memory batch"):
                _ = engine.add_batch("user1", ["idx1", "idx2"], infer=False)
            assert engine.get_all("user1") == []
            assert len(engine.index) == 0
        finally:
            engine.close()

    def test_add_batch_vectors_remap_after_unique_skip(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Precomputed vectors must follow remaining rows after UNIQUE skip."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        try:
            engine.ensure_initialized()
            first = [0.11] * 128
            second = [0.22] * 128
            third = [0.33] * 128
            _ = engine.add_batch("user1", ["keep-a"], infer=False, vectors=[first])

            captured: dict[str, object] = {}
            original = engine._index_vectors

            def wrap(
                inserted: list[tuple[str, int]],
                vectors: list[list[float]],
                indexed_ids: list[int],
            ) -> None:
                captured["contents"] = [content for content, _mem_id in inserted]
                captured["vectors"] = [list(vector) for vector in vectors]
                original(inserted, vectors, indexed_ids)

            with patch.object(engine, "_index_vectors", wrap):
                stored = engine.add_batch(
                    "user1",
                    ["keep-a", "keep-b", "keep-c"],
                    infer=False,
                    vectors=[first, second, third],
                )
            assert stored == ["keep-b", "keep-c"]
            assert captured["contents"] == ["keep-b", "keep-c"]
            assert captured["vectors"] == [second, third]
        finally:
            engine.close()
            gc.collect()

    def test_add_batch_embedding_size_mismatch(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, _, _ = temp_engine
        bad_embedder = MismatchedSizeEmbedder(dims=128)
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=bad_embedder, logger=logger)

        try:
            with pytest.raises(RuntimeError, match="Failed to add memory batch"):
                engine.add_batch("user1", ["m1", "m2"], infer=False)
        finally:
            engine.close()


class TestUSearchEngineDistanceScoring:
    """Tests for _rank_scores and _distances_to_scores methods."""

    def test_rank_scores_empty(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """_rank_scores with count=0 should return empty array."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            scores = engine._rank_scores(0)
            assert len(scores) == 0
        finally:
            engine.close()

    def test_rank_scores_single(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """_rank_scores with count=1 should return [1.0]."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            scores = engine._rank_scores(1)
            assert len(scores) == 1
            assert float(scores[0]) == pytest.approx(1.0)
        finally:
            engine.close()

    def test_rank_scores_multiple(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """_rank_scores with count>1 should return descending scores."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            scores = engine._rank_scores(5)
            assert len(scores) == 5
            assert float(scores[0]) == pytest.approx(1.0)
            assert float(scores[-1]) == pytest.approx(0.0)
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1]
        finally:
            engine.close()

    def test_distances_to_scores_l2_metric(self) -> None:
        """L2 metric should use 1/(1+d) conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = USearchConfig(
                workspace_id="test",
                index_path=os.path.join(tmpdir, "index.usearch"),
                db_path=os.path.join(tmpdir, "messages.db"),
                embedding_dims=128,
                embedder_provider=EmbedderProvider.OPENAI,
                embedding_model="test/mock-128",
                metric="l2",
            )
            embedder = MockEmbedder(dims=128)
            engine = USearchEngine(config=config, embedder=embedder)

            try:
                distances = np.array([0.0, 1.0, 4.0], dtype=np.float32)
                scores = engine._distances_to_scores(distances)

                assert float(scores[0]) == pytest.approx(1.0)
                assert float(scores[1]) == pytest.approx(0.5)
                assert float(scores[2]) == pytest.approx(0.2)
            finally:
                engine.close()

    def test_distances_to_scores_unknown_metric_uses_rank(self) -> None:
        """Unknown metric should fall back to rank-based scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = USearchConfig(
                workspace_id="test",
                index_path=os.path.join(tmpdir, "index.usearch"),
                db_path=os.path.join(tmpdir, "messages.db"),
                embedding_dims=128,
                embedder_provider=EmbedderProvider.OPENAI,
                embedding_model="test/mock-128",
                metric="ip",
            )
            embedder = MockEmbedder(dims=128)
            logger = create_mock_logger()
            engine = USearchEngine(config=config, embedder=embedder, logger=logger)

            try:
                distances = np.array([0.1, 0.5, 0.9], dtype=np.float32)
                scores = engine._distances_to_scores(distances)

                assert len(scores) == 3
                assert float(scores[0]) == pytest.approx(1.0)
                assert float(scores[-1]) == pytest.approx(0.0)
                logger.warning.assert_called()  # type: ignore
            finally:
                engine.close()


class TestUSearchEngineSearchErrorPaths:
    """Tests for search() error handling and edge cases."""

    def test_search_empty_index_with_logger(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search on empty index with logger should log debug message."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            results = engine.search("hello", "user1", limit=5)

            assert results == []
            debug_calls = [str(c) for c in logger.debug.call_args_list]  # type: ignore
            assert any("empty" in c.lower() for c in debug_calls)
        finally:
            engine.close()

    def test_search_no_matching_workspace_returns_empty(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Search where no results match workspace_id should return empty."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "Hello world", infer=False)

            results = engine.search("Hello", "nonexistent-project", limit=5)
            assert results == []
        finally:
            engine.close()

    def test_search_exception_wraps_in_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, _, _ = temp_engine
        failing_embedder = ToggleableFailingQueryEmbedder(dims=128)
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=failing_embedder, logger=logger)

        try:
            engine.add("user1", "Test msg", infer=False)

            failing_embedder.should_fail = True

            with pytest.raises(RuntimeError, match="USearch search failed"):
                engine.search("broken query", "user1", limit=5)

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineGetAllErrorPaths:
    """Tests for get_all() error handling paths."""

    def test_get_all_with_logger(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """get_all with logger should log content count."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "msg1", infer=False)
            engine.get_all("user1")
            debug_calls = [str(c) for c in logger.debug.call_args_list]  # type: ignore
            assert any("Retrieved all" in c for c in debug_calls)
        finally:
            engine.close()

    def test_get_all_error_wraps_in_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Exceptions in get_all should be wrapped in RuntimeError."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            original_store = engine.memory_store
            original_store.close()
            mock_store = MagicMock()
            mock_store.get_all.side_effect = OSError("disk read error")
            object.__setattr__(engine, "_memory_store", mock_store)

            with pytest.raises(RuntimeError, match="Failed to retrieve memories"):
                engine.get_all("user1")

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineDeleteErrorPaths:
    """Tests for delete() additional error handling."""

    def test_delete_existing_with_logger_logs_debug(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Deleting an existing content with logger should log debug."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "to delete", infer=False)
            content_id = engine.memory_store.get_id_by_content("user1", "to delete")
            assert content_id is not None

            engine.delete(str(content_id))

            debug_calls = [str(c) for c in logger.debug.call_args_list]  # type: ignore
            assert any("deleted" in c.lower() for c in debug_calls)
        finally:
            engine.close()

    def test_delete_invalid_id_with_logger(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Invalid memory_id with logger should log error."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            with pytest.raises(RuntimeError, match="Invalid memory_id format"):
                engine.delete("not-a-number")
            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()

    def test_delete_general_exception_wraps_in_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Unexpected exceptions in delete should be wrapped in RuntimeError."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            engine._index = MagicMock()
            engine._index.__contains__ = MagicMock(
                side_effect=OSError("corrupted index")
            )

            with pytest.raises(RuntimeError, match="Failed to delete memory"):
                engine.delete("42")

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineCommitErrorPaths:
    """Tests for commit() error handling and edge cases."""

    def test_commit_no_index_is_noop(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Commit when _index is None should be a no-op."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.commit()
        finally:
            engine.close()

    def test_commit_with_logger(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Commit with logger should log save info."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.add("user1", "commit test", infer=False)
            engine.commit()

            debug_calls = [str(c) for c in logger.debug.call_args_list]  # type: ignore
            assert any("saved" in c.lower() for c in debug_calls)
        finally:
            engine.close()

    def test_commit_error_wraps_in_runtime_error(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Exceptions in commit should be wrapped in RuntimeError."""
        config, embedder, _ = temp_engine
        logger = create_mock_logger()
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        try:
            engine.ensure_initialized()
            engine._index = MagicMock()
            engine._dirty = True
            engine._index.save = MagicMock(side_effect=OSError("disk full"))

            with pytest.raises(RuntimeError, match="Failed to save USearch index"):
                engine.commit()

            logger.error.assert_called()  # type: ignore
        finally:
            engine.close()


class TestUSearchEngineGetIdByContent:
    """Tests for get_id_by_content() method."""

    def test_get_id_by_content_found(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """get_id_by_content should return ID for existing content."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.add("user1", "findable msg", infer=False)
            content_id = engine.get_id_by_content("user1", "findable msg")
            assert content_id is not None
            assert isinstance(content_id, int)
        finally:
            engine.close()

    def test_get_id_by_content_not_found(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """get_id_by_content should return None for non-existent content."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)

        try:
            engine.ensure_initialized()
            content_id = engine.get_id_by_content("user1", "nonexistent")
            assert content_id is None
        finally:
            engine.close()

    def test_get_records_by_contents_delegates(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        try:
            engine.add("user1", "bulk-one", infer=False)
            engine.add("user1", "bulk-two", infer=False)
            records = engine.get_records_by_contents(
                "user1", ["bulk-two", "missing", "bulk-one"]
            )
            assert [row.content for row in records] == ["bulk-two", "bulk-one"]
        finally:
            engine.close()


class TestUSearchEngineContextManager:
    """Tests for context manager protocol (__enter__/__exit__)."""

    def test_context_manager_initializes_and_closes(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """Context manager should initialize on enter and close on exit."""
        config, embedder, _ = temp_engine

        with USearchEngine(config=config, embedder=embedder) as engine:
            assert engine is not None
            assert engine.name == "usearch"
            engine.add("user1", "ctx msg", infer=False)

    def test_context_manager_does_not_suppress_exceptions(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """__exit__ should return False to not suppress exceptions."""
        config, embedder, _ = temp_engine

        with pytest.raises(ValueError, match="test error"):
            with USearchEngine(config=config, embedder=embedder) as _engine:
                raise ValueError("test error")

    def test_close_with_no_memory_store(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        """close() when _memory_store is None should be a no-op."""
        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        engine.close()

    def test_index_and_store_raise_after_close(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from reflectlog.core.exceptions import StorageError

        config, embedder, _ = temp_engine
        engine = USearchEngine(config=config, embedder=embedder)
        _ = engine.index
        engine.close()
        with pytest.raises(StorageError, match="USearchEngine is closed"):
            _ = engine.index
        with pytest.raises(StorageError, match="USearchEngine is closed"):
            _ = engine.memory_store


class TestAtomicUSearchPublication:
    """External refresh and atomic HNSW publication."""

    def test_stale_second_engine_refreshes_before_mutate(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, embedder, _ = temp_engine
        stale = USearchEngine(config=config, embedder=embedder)
        writer = USearchEngine(config=config, embedder=embedder)
        try:
            _ = stale.index
            writer.add("test", "first-row", infer=False)
            writer.commit()
            stale.add("test", "second-row", infer=False)
            stale.commit()
        finally:
            stale.close()
            writer.close()

        reopened = USearchEngine(config=config, embedder=embedder)
        try:
            assert set(reopened.get_all("test")) == {"first-row", "second-row"}
            assert len(reopened.index) == 2
        finally:
            reopened.close()

    def test_failpoint_before_replace_keeps_previous_index(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, embedder, tmpdir = temp_engine
        first = USearchEngine(config=config, embedder=embedder)
        first.add("test", "kept", infer=False)
        first.commit()
        first.close()

        def boom(step: str) -> None:
            if step == "before_replace":
                raise RuntimeError("injected publish fail")

        second = USearchEngine(config=config, embedder=embedder, publish_hook=boom)
        try:
            second.add("test", "new-row", infer=False)
            with pytest.raises(RuntimeError, match="Failed to save USearch index"):
                second.commit()
        finally:
            second.close()

        temps = [name for name in os.listdir(tmpdir) if name.endswith(".tmp")]
        assert temps == []
        reopened = USearchEngine(config=config, embedder=embedder)
        try:
            assert "kept" in reopened.get_all("test")
            assert len(reopened.index) >= 1
        finally:
            reopened.close()

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
    def test_failpoints_keep_previous_or_complete_index(
        self,
        temp_engine: tuple[USearchConfig, MockEmbedder, str],
        step: str,
    ) -> None:
        config, embedder, tmpdir = temp_engine
        first = USearchEngine(config=config, embedder=embedder)
        first.add("test", "kept", infer=False)
        first.commit()
        first.close()

        def boom(name: str) -> None:
            if name == step:
                raise RuntimeError(f"injected {step}")

        second = USearchEngine(config=config, embedder=embedder, publish_hook=boom)
        try:
            second.add("test", "new-row", infer=False)
            with pytest.raises(RuntimeError, match="Failed to save USearch index"):
                second.commit()
        finally:
            second.close()

        temps = [name for name in os.listdir(tmpdir) if name.endswith(".tmp")]
        assert temps == []
        reopened = USearchEngine(config=config, embedder=embedder)
        try:
            assert "kept" in reopened.get_all("test")
            assert len(reopened.index) >= 1
        finally:
            reopened.close()

    def test_publish_does_not_advance_generation(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from reflectlog.infrastructure.storage_coordinator import (
            PortalockerStorageCoordinator,
        )

        config, embedder, tmpdir = temp_engine
        config = USearchConfig(
            workspace_id="test",
            index_path=os.path.join(tmpdir, "test", "usearch", "vectors.usearch"),
            db_path=os.path.join(tmpdir, "test", "usearch", "memories.db"),
            embedding_dims=config.embedding_dims,
            embedder_provider=config.embedder_provider,
            embedding_model=config.embedding_model,
        )
        coordinator = PortalockerStorageCoordinator(tmpdir, timeout=1.0)
        engine = USearchEngine(
            config=config, embedder=embedder, coordinator=coordinator
        )
        try:
            assert coordinator.read_generation(config.workspace_id) == 0
            engine.add("test", "gen-row", infer=False)
            engine.commit()
            assert coordinator.read_generation(config.workspace_id) == 0
        finally:
            engine.close()

    def test_startup_removes_orphan_temp_files(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, embedder, _tmpdir = temp_engine
        ensure_embedding_identity(config)
        orphan = f"{config.index_path}.{os.getpid()}.9999.tmp"
        foreign = f"{config.index_path}.{os.getppid()}.9999.tmp"
        with open(orphan, "wb") as handle:
            handle.write(b"orphan")
        with open(foreign, "wb") as handle:
            handle.write(b"foreign")
        engine = USearchEngine(config=config, embedder=embedder)
        try:
            _ = engine.index
            assert not os.path.exists(orphan)
            assert os.path.exists(foreign)
        finally:
            engine.close()

    def test_startup_removes_dead_pid_temp_files(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from reflectlog.infrastructure.usearch_engine import _pid_is_alive

        config, embedder, _tmpdir = temp_engine
        ensure_embedding_identity(config)
        ctx = multiprocessing.get_context("spawn")
        child = ctx.Process(target=_spawn_exit_immediately)
        child.start()
        child.join(timeout=10.0)
        assert child.exitcode == 0
        dead_pid = child.pid
        assert dead_pid is not None
        assert _pid_is_alive(dead_pid) is False
        leftover = f"{config.index_path}.{dead_pid}.9999.tmp"
        with open(leftover, "wb") as handle:
            handle.write(b"dead")
        engine = USearchEngine(config=config, embedder=embedder)
        try:
            _ = engine.index
            assert not os.path.exists(leftover)
        finally:
            engine.close()

    def test_fsync_path_accepts_regular_file(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from reflectlog.infrastructure.usearch_engine import _fsync_path

        _config, _embedder, tmpdir = temp_engine
        path = os.path.join(tmpdir, "fsync.bin")
        with open(path, "wb") as handle:
            handle.write(b"durable")
        _fsync_path(path)

    def test_pid_is_alive_rejects_invalid_and_accepts_self(self) -> None:
        from reflectlog.infrastructure.usearch_engine import _pid_is_alive

        assert _pid_is_alive(os.getpid()) is True
        assert _pid_is_alive(0) is False
        assert _pid_is_alive(-1) is False

    def test_empty_hnsw_over_populated_sqlite_fails_closed(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from usearch.index import Index

        from reflectlog.core.exceptions import InitializationError

        config, embedder, _tmpdir = temp_engine
        writer = USearchEngine(config=config, embedder=embedder)
        writer.add("test", "kept", infer=False)
        writer.commit()
        writer.close()
        empty = Index(ndim=config.embedding_dims, metric=config.metric, dtype="f32")
        empty.save(config.index_path)
        with pytest.raises(InitializationError, match="empty"):
            doomed = USearchEngine(config=config, embedder=embedder)
            try:
                _ = doomed.index
            finally:
                doomed.close()

    def test_empty_hnsw_with_unreadable_sqlite_fails_closed(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from usearch.index import Index

        from reflectlog.core.exceptions import InitializationError

        config, embedder, _tmpdir = temp_engine
        ensure_embedding_identity(config)
        empty = Index(ndim=config.embedding_dims, metric=config.metric, dtype="f32")
        empty.save(config.index_path)
        with patch(
            "reflectlog.infrastructure.usearch_engine._sqlite_memory_count",
            return_value=None,
        ):
            with pytest.raises(InitializationError, match="unreadable"):
                doomed = USearchEngine(config=config, embedder=embedder)
                try:
                    _ = doomed.index
                finally:
                    doomed.close()

    def test_empty_search_reloads_newer_published_file(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        config, embedder, _tmpdir = temp_engine
        stale = USearchEngine(config=config, embedder=embedder)
        writer = USearchEngine(config=config, embedder=embedder)
        try:
            _ = stale.index
            assert stale.search("kept", "test", 5) == []
            writer.add("test", "kept", infer=False)
            writer.commit()
            hits = stale.search("kept", "test", 5)
            assert any(content == "kept" for content, _score, _ts in hits)
        finally:
            stale.close()
            writer.close()

    def test_dirty_commit_refuses_newer_live_file(
        self, temp_engine: tuple[USearchConfig, MockEmbedder, str]
    ) -> None:
        from reflectlog.core.exceptions import StorageError

        config, embedder, _tmpdir = temp_engine
        seed = USearchEngine(config=config, embedder=embedder)
        seed.add("test", "seed-row", infer=False)
        seed.commit()
        seed.close()
        stale = USearchEngine(config=config, embedder=embedder)
        writer = USearchEngine(config=config, embedder=embedder)
        try:
            _ = stale.index
            stale.add("test", "stale-row", infer=False)
            writer.add("test", "newer-row", infer=False)
            writer.commit()
            with pytest.raises(StorageError, match="stale"):
                stale.commit()
        finally:
            stale.close()
            writer.close()
