"""USearch-based semantic search engine.

This module provides a semantic search engine using USearch for vector
similarity search, with SQLite for memory content storage. It implements
the ISemanticSearchEngine protocol for compatibility with MemoryManager.

Clean Architecture Compliance:
    This module implements the ISemanticSearchEngine protocol defined in
    ``reflectlog.core.types.ISemanticSearchEngine``, following the
    Dependency Inversion Principle from SOLID.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Self, final

if TYPE_CHECKING:
    from typing import TypeGuard

    from reflectlog.core.config import IAppConfig
    from reflectlog.infrastructure.memory_store import MemoryRecord

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr
from usearch.index import BatchMatches, Index, Match

from reflectlog.core.enums import EmbedderProvider, parse_str_enum
from reflectlog.core.exceptions import InitializationError, StorageError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.storage_coordination import IStorageCoordinator, LeaseMode
from reflectlog.core.types import Closable, Embeddings, IStoredMemory
from reflectlog.infrastructure.embedding_identity import ensure_embedding_identity
from reflectlog.infrastructure.memory_store import MemoryStore
from reflectlog.utility.scoring import distance_to_similarity_cosine
from reflectlog.utility.security import validate_workspace_id


def _is_dict_config(config: object) -> TypeGuard[dict[str, Any]]:
    """Type guard to check if config is a dict."""
    return isinstance(config, dict)


def _index_file_identity(path: str) -> tuple[int, int] | None:
    """Return (mtime_ns, size) for a live HNSW file, or None if missing."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _fsync_path(path: str) -> None:
    # Windows FlushFileBuffers requires GENERIC_WRITE. O_RDONLY yields EBADF.
    flags = os.O_RDWR if os.name == "nt" else os.O_RDONLY
    handle = os.open(path, flags)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _fsync_directory(path: str) -> None:
    directory = os.path.dirname(path) or "."
    try:
        _fsync_path(directory)
    except OSError:
        return


def _windows_pid_is_alive(pid: int) -> bool:
    """Return True only while the Windows process is still running.

    ``os.kill(pid, 0)`` uses OpenProcess and succeeds for an exited child
    while the parent still holds a handle. GetExitCodeProcess distinguishes
    STILL_ACTIVE from that leftover object.
    """
    import ctypes
    from ctypes import CDLL, c_ulong

    # 64-bit Windows uses one calling convention; CDLL is typed on all platforms.
    kernel32 = CDLL("kernel32", use_last_error=True)
    process_query_limited_information = 0x1000
    still_active = 259
    handle = int(kernel32["OpenProcess"](process_query_limited_information, 0, pid))
    if handle == 0:
        return False
    try:
        exit_code = c_ulong()
        queried = kernel32["GetExitCodeProcess"](handle, ctypes.byref(exit_code))
        if int(queried) == 0:
            return False
        return int(exit_code.value) == still_active
    finally:
        _ = kernel32["CloseHandle"](handle)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _hnsw_temp_owner_pid(name: str, base: str) -> int | None:
    prefix = f"{base}."
    if not name.startswith(prefix) or not name.endswith(".tmp"):
        return None
    pid_text = name[len(prefix) : -4].split(".", 1)[0]
    try:
        return int(pid_text)
    except ValueError:
        return None


def _cleanup_orphan_hnsw_temps(index_path: str, *, only_pid: int | None = None) -> None:
    directory = os.path.dirname(index_path) or "."
    base = os.path.basename(index_path)
    own_prefix = f"{base}.{os.getpid()}."
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if not name.startswith(f"{base}.") or not name.endswith(".tmp"):
            continue
        if only_pid is not None:
            if not name.startswith(own_prefix):
                continue
        else:
            owner = _hnsw_temp_owner_pid(name, base)
            if owner is not None and owner != os.getpid() and _pid_is_alive(owner):
                continue
        try:
            os.remove(os.path.join(directory, name))
        except OSError:
            continue


def _sqlite_memory_count(db_path: str) -> int | None:
    """Return memory row count, 0 if the DB is absent, or None if unreadable."""
    if not os.path.exists(db_path):
        return 0
    import sqlite3

    try:
        connection = sqlite3.connect(db_path, timeout=5.0)
        try:
            _ = connection.execute("PRAGMA busy_timeout = 5000")
            row = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return int(row[0])


@dataclass(frozen=True)
class USearchConfig:
    """Configuration for USearchEngine.

    This dataclass defines the settings needed for USearch vector search
    with SQLite memory storage.

    Attributes:
        workspace_id: Workspace identifier for filtering.
        index_path: Path to the USearch index file.
        db_path: Path to the SQLite memory database.
        embedder_provider: Provider used to produce the embeddings.
        embedding_model: Model used to produce the embeddings.
        embedding_dims: Vector embedding dimensions.
        metric: Distance metric (cos, l2, ip).
        connectivity: HNSW M parameter.
        expansion_add: HNSW efConstruction parameter.
        expansion_search: HNSW ef search parameter.
        exact_search: Force exact brute-force search instead of HNSW approximation.
            When True, bypasses HNSW indexing and uses SIMD-optimized similarity
            metrics from SimSIMD for guaranteed exact results. Best for small
            collections (< 10k vectors). Default: False.
        exact_search_threshold: Auto-switch to exact search when index size is
            below this threshold. Set to 0 to disable auto-switching.
            When index size < threshold, exact search is used regardless of
            exact_search setting. Default: 0 (disabled).
    """

    workspace_id: str
    index_path: str
    db_path: str
    embedding_dims: int
    embedder_provider: EmbedderProvider
    embedding_model: str
    metric: str = "cos"
    connectivity: int = 16
    expansion_add: int = 128
    expansion_search: int = 64
    exact_search: bool = False
    exact_search_threshold: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> USearchConfig:
        """Create USearchConfig from a dictionary with validation.

        This method properly validates dict contents and returns a typed instance,
        avoiding the type: ignore needed when using **dict unpacking.

        Args:
            data: Dictionary with configuration values.

        Returns:
            Validated USearchConfig instance.
        """
        return cls(
            workspace_id=data.get("workspace_id", "") or "",
            index_path=data.get("index_path", "") or "",
            db_path=data.get("db_path", "") or "",
            embedding_dims=int(data.get("embedding_dims", 3072)),
            embedder_provider=parse_str_enum(
                EmbedderProvider, data["embedder_provider"], field="embedder_provider"
            ),
            embedding_model=data["embedding_model"],
            metric=data.get("metric", "cos") or "cos",
            connectivity=int(data.get("connectivity", 16)),
            expansion_add=int(data.get("expansion_add", 128)),
            expansion_search=int(data.get("expansion_search", 64)),
            exact_search=bool(data.get("exact_search", False)),
            exact_search_threshold=int(data.get("exact_search_threshold", 0)),
        )

    @classmethod
    def from_config(cls, config: IAppConfig) -> USearchConfig:
        """Create USearchConfig from IAppConfig protocol.

        Args:
            config: Configuration satisfying IAppConfig protocol.

        Returns:
            USearchConfig with extracted settings.
        """
        # Validate workspace_id to prevent path traversal attacks
        workspace_id = validate_workspace_id(config.workspace_id)
        base_path = config.usearch_index_path
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.getcwd(), base_path)

        match config.embedder_provider:
            case EmbedderProvider.OPENAI:
                embedding_dims = config.embedding_dims
            case EmbedderProvider.LANGCHAIN:
                embedding_dims = config.qwen_embedding_dims
            case EmbedderProvider.WEMM:
                embedding_dims = config.wemm_embedding_dims

        return cls(
            workspace_id=workspace_id,
            index_path=os.path.join(base_path, "vectors.usearch"),
            db_path=os.path.join(base_path, "memories.db"),
            embedding_dims=embedding_dims,
            embedder_provider=config.embedder_provider,
            embedding_model=config.embedding_model,
            exact_search=config.usearch_exact_search,
            exact_search_threshold=config.usearch_exact_search_threshold,
        )


@final
class USearchEngine(BaseModel):
    """USearch-based semantic search engine.

    Implements ISemanticSearchEngine protocol via structural subtyping.
    Uses USearch for vector similarity search and SQLite for memory content storage.

    Uses lazy initialization for both USearch index and SQLite connection
    with thread-safe double-checked locking.

    Example:
        ```python
        from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine
        from reflectlog.core.enums import EmbedderProvider
        from langchain_openai import OpenAIEmbeddings

        config = USearchConfig(
            workspace_id="my-project",
            index_path="indexes/my-project/usearch/vectors.usearch",
            db_path="indexes/my-project/usearch/memories.db",
            embedding_dims=3072,
            embedder_provider=EmbedderProvider.OPENAI,
            embedding_model="text-embedding-3-large",
        )
        embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        engine.add("my-project", "Hello world", infer=False)
        results = engine.search("hello", "my-project", limit=5)
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: USearchConfig
    embedder: Embeddings
    logger: IStructuredLogger | None = None
    coordinator: IStorageCoordinator | None = None
    publish_hook: Callable[[str], None] | None = None

    _index: Index | None = PrivateAttr(default=None)
    _memory_store: MemoryStore | None = PrivateAttr(default=None)
    # Instance-level lock for thread-safe lazy initialization (prevents cross-instance contention)
    _init_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _index_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    _dirty: bool = PrivateAttr(default=False)
    _closed: bool = PrivateAttr(default=False)
    _seen_identity: tuple[int, int] | None = PrivateAttr(default=None)

    def __init__(
        self,
        config: USearchConfig | dict[str, Any],
        embedder: Embeddings,
        logger: IStructuredLogger | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize USearchEngine.

        Args:
            config: USearchConfig or dict with engine settings.
            embedder: Embedding provider implementing Embeddings interface.
            logger: Optional StructuredLogger instance.
            **kwargs: Additional arguments passed to BaseModel.
        """
        if _is_dict_config(config):
            config = USearchConfig.from_dict(config)

        super().__init__(config=config, embedder=embedder, logger=logger, **kwargs)
        self.coordinator = ensure_embedding_identity(self.config, self.coordinator)

    def _reject_populated_hnsw_without_db(self, hnsw_size: int) -> None:
        """Refuse a loaded HNSW that cannot be paired with SQLite SoT."""
        sqlite_rows = _sqlite_memory_count(self.config.db_path)
        if hnsw_size <= 0:
            if sqlite_rows is None:
                raise InitializationError(
                    "USearch index is empty and SQLite is unreadable "
                    f"at {self.config.db_path}. Refusing to load an empty HNSW."
                )
            if sqlite_rows > 0:
                raise InitializationError(
                    "USearch index is empty but SQLite has "
                    f"{sqlite_rows} memories at {self.config.db_path}. "
                    "Refusing to load an empty HNSW over populated rows."
                )
            return
        db_missing = not os.path.exists(self.config.db_path)
        if sqlite_rows is not None and sqlite_rows > 0:
            return
        detail = (
            "missing"
            if db_missing
            else "unreadable"
            if sqlite_rows is None
            else "empty"
        )
        raise InitializationError(
            f"USearch index has {hnsw_size} vectors but SQLite is {detail} "
            f"at {self.config.db_path}. Refusing to load HNSW without a "
            "readable memory store."
        )

    @property
    def name(self) -> str:
        """Engine name for identification."""
        return "usearch"

    @property
    def index(self) -> Index:
        """Get USearch index (thread-safe lazy initialization).

        Returns:
            USearch Index instance.

        Raises:
            RuntimeError: If index initialization fails.
        """
        if self._closed:
            raise StorageError("USearchEngine is closed")
        if self._index is not None:
            return self._index

        with self._init_lock:
            if self._closed:
                raise StorageError("USearchEngine is closed")
            if self._index is None:
                try:
                    # Ensure directory exists
                    index_dir = os.path.dirname(self.config.index_path)
                    if index_dir:
                        os.makedirs(index_dir, exist_ok=True)
                    _cleanup_orphan_hnsw_temps(self.config.index_path)

                    # Optimization: Try restore first, avoid extra os.path.exists() call
                    # This is faster for existing indices (one less syscall)
                    try:
                        loaded_index = Index.restore(self.config.index_path)
                        if loaded_index is None:
                            raise RuntimeError(
                                f"Index.restore() returned None for {self.config.index_path}"
                            )
                        self._reject_populated_hnsw_without_db(len(loaded_index))
                        self._index = loaded_index
                        self._seen_identity = _index_file_identity(
                            self.config.index_path
                        )
                        if self.logger:
                            self.logger.info(
                                "Loaded existing USearch index",
                                extra={
                                    "workspace_id": self.config.workspace_id,
                                    "index_path": self.config.index_path,
                                    "size": len(loaded_index),
                                },
                            )
                    except (RuntimeError, FileNotFoundError, OSError) as restore_error:
                        index_exists = os.path.exists(self.config.index_path)
                        db_missing = not os.path.exists(self.config.db_path)
                        sqlite_rows = _sqlite_memory_count(self.config.db_path)
                        sqlite_unknown = sqlite_rows is None
                        sqlite_populated = sqlite_rows is not None and sqlite_rows > 0
                        if index_exists and (
                            sqlite_populated or sqlite_unknown or db_missing
                        ):
                            detail = (
                                "missing"
                                if db_missing
                                else "unreadable"
                                if sqlite_unknown
                                else f"{sqlite_rows} memories"
                            )
                            raise InitializationError(
                                "USearch index is corrupt but SQLite is "
                                f"{detail} at {self.config.db_path}. "
                                "Refusing to create an empty HNSW that would "
                                "overwrite the file."
                            ) from restore_error
                        if sqlite_populated:
                            raise InitializationError(
                                "USearch index is missing but SQLite has "
                                f"{sqlite_rows} memories at {self.config.db_path}. "
                                "Refusing to create an empty HNSW that would "
                                "hide existing rows."
                            ) from restore_error
                        if self.logger:
                            self.logger.debug(
                                "USearch index not found, creating new index",
                                extra={
                                    "workspace_id": self.config.workspace_id,
                                    "index_path": self.config.index_path,
                                },
                            )
                        new_index = Index(
                            ndim=self.config.embedding_dims,
                            metric=self.config.metric,
                            dtype="f32",
                            connectivity=self.config.connectivity,
                            expansion_add=self.config.expansion_add,
                            expansion_search=self.config.expansion_search,
                        )
                        self._index = new_index
                        self._dirty = False
                        self._seen_identity = _index_file_identity(
                            self.config.index_path
                        )
                        if self.logger:
                            self.logger.info(
                                "Created new USearch index",
                                extra={
                                    "workspace_id": self.config.workspace_id,
                                    "index_path": self.config.index_path,
                                    "dims": self.config.embedding_dims,
                                },
                            )

                except InitializationError:
                    raise
                except Exception as exc:
                    if self.logger:
                        self.logger.error(
                            "Failed to initialize USearch index",
                            extra={
                                "workspace_id": self.config.workspace_id,
                                "error": str(exc),
                            },
                            exc_info=True,
                        )
                    raise RuntimeError(
                        f"Failed to initialize USearch index: {exc}"
                    ) from exc

        # _index is guaranteed to be non-None after successful initialization
        # An exception would have been raised above if initialization failed
        return self._index

    @property
    def memory_store(self) -> MemoryStore:
        """Get MemoryStore (thread-safe lazy initialization).

        Returns:
            MemoryStore instance.
        """
        if self._closed:
            raise StorageError("USearchEngine is closed")
        if self._memory_store is not None:
            return self._memory_store

        with self._init_lock:
            if self._closed:
                raise StorageError("USearchEngine is closed")
            if self._memory_store is None:
                from reflectlog.infrastructure.memory_store import MemoryStore

                self._memory_store = MemoryStore(
                    db_path=self.config.db_path,
                    logger=self.logger,
                )

        return self._memory_store

    def _emit_publish_hook(self, step: str) -> None:
        hook = self.publish_hook
        if hook is not None:
            hook(step)

    @contextmanager
    def _write_lease(self) -> Generator[None]:
        coordinator = self.coordinator
        if coordinator is None or coordinator.is_held(
            self.config.workspace_id, LeaseMode.EXCLUSIVE
        ):
            yield
            return
        with coordinator.acquire(self.config.workspace_id, LeaseMode.EXCLUSIVE):
            yield

    @contextmanager
    def _read_lease(self) -> Generator[None]:
        coordinator = self.coordinator
        if coordinator is None or coordinator.is_held(self.config.workspace_id):
            yield
            return
        with coordinator.acquire(self.config.workspace_id, LeaseMode.SHARED):
            yield

    def refresh(self) -> None:
        """Reload a newer HNSW published by another writer."""
        _ = self.index
        self._maybe_reload_external()

    def _maybe_reload_external(self, *, refuse_stale: bool = False) -> None:
        """Reload the HNSW when another writer published a newer file."""
        current = _index_file_identity(self.config.index_path)
        if self._dirty:
            if refuse_stale and current is not None and current != self._seen_identity:
                raise StorageError(
                    "Refusing to publish a stale in-memory HNSW over a newer file"
                )
            return
        if current == self._seen_identity:
            return
        if current is None:
            return
        loaded = Index.restore(self.config.index_path)
        if loaded is None:
            raise RuntimeError(
                f"Index.restore() returned None for {self.config.index_path}"
            )
        self._reject_populated_hnsw_without_db(len(loaded))
        self._index = loaded
        self._seen_identity = current

    def _publish_index(self) -> None:
        """Atomically replace the live HNSW with a validated temp snapshot."""
        live_path = self.config.index_path
        directory = os.path.dirname(live_path) or "."
        os.makedirs(directory, exist_ok=True)
        temp_path = f"{live_path}.{os.getpid()}.{time.time_ns()}.tmp"
        coordinator = self.coordinator
        if coordinator is None or coordinator.is_held(
            self.config.workspace_id, LeaseMode.EXCLUSIVE
        ):
            _cleanup_orphan_hnsw_temps(live_path)
        else:
            _cleanup_orphan_hnsw_temps(live_path, only_pid=os.getpid())
        self._emit_publish_hook("before_save")
        try:
            self.index.save(temp_path)
            self._emit_publish_hook("after_temp_save")
            validated = Index.restore(temp_path)
            if validated is None:
                raise RuntimeError("Temp HNSW restore returned None")
            if len(validated) != len(self.index):
                raise RuntimeError("Temp HNSW size does not match in-memory index")
            self._emit_publish_hook("after_temp_validate")
            _fsync_path(temp_path)
            self._emit_publish_hook("after_fsync")
            self._emit_publish_hook("before_replace")
            os.replace(temp_path, live_path)
            self._emit_publish_hook("after_replace")
            _fsync_directory(live_path)
            self._dirty = False
            self._seen_identity = _index_file_identity(live_path)
        except Exception:
            if os.path.exists(temp_path):
                with suppress(OSError):
                    os.remove(temp_path)
            raise

    def add(
        self,
        workspace_id: str,
        content: str,
        infer: bool,
    ) -> None:
        """Add a memory to the USearch index.

        Args:
            workspace_id: Workspace identifier for filtering.
            content: Memory content to index.
            infer: Whether to enable LLM-based memory inference (not supported).

        Raises:
            RuntimeError: If add operation fails.
        """
        try:
            if isinstance(self.get_id_by_content(workspace_id, content), int):
                if self.logger:
                    self.logger.debug(
                        "Skipping duplicate memory (detected by DB constraint)",
                        extra={"workspace_id": self.config.workspace_id},
                    )
                return
            try:
                vectors = self.embedder.embed_documents([content])
                if len(vectors) != 1:
                    raise RuntimeError("Embedding batch size mismatch for USearch add")
                vector = vectors[0]
                if not vector:
                    raise RuntimeError("Embedding produced an empty vector")
            except Exception as embed_error:
                raise RuntimeError(
                    f"Failed to generate embedding: {embed_error}"
                ) from embed_error
            with self._write_lease():
                _ = self.index
                self._maybe_reload_external()
                self._add_unlocked(workspace_id, content, infer, vector)
            return
        except (RuntimeError, StorageError) as e:
            if "Duplicate memory" in str(e):
                if self.logger:
                    self.logger.debug(
                        "Skipping duplicate memory (detected by DB constraint)",
                        extra={"workspace_id": self.config.workspace_id},
                    )
                return
            if self.logger:
                self.logger.error(
                    "Failed to add memory to USearch index",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "error": str(e),
                    },
                )
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to add memory to USearch index",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to add memory: {e}") from e

    def _add_unlocked(
        self,
        workspace_id: str,
        content: str,
        infer: bool,
        vector: list[float],
    ) -> None:
        try:
            _ = self.index
            if infer and self.logger:
                self.logger.warning(
                    "infer=True not supported by USearchEngine, proceeding as infer=False",
                    extra={"workspace_id": self.config.workspace_id},
                )

            # Insert into SQLite (relies on UNIQUE INDEX for dedup - no pre-check needed)
            mem_id = self.memory_store.insert(workspace_id, content)

            vector_np = np.array(vector, dtype=np.float32)
            if len(vector_np) == 0:
                _ = self.memory_store.delete(mem_id)
                raise RuntimeError("Embedding produced an empty vector")

            try:
                with self._index_lock:
                    self.index.add(mem_id, vector_np)
                self._dirty = True
            except Exception as index_error:
                _ = self.memory_store.delete(mem_id)
                raise RuntimeError(
                    f"Failed to add vector to USearch index: {index_error}"
                ) from index_error

            if self.logger:
                self.logger.debug(
                    "Memory added to USearch index",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "memory_id": mem_id,
                        "memory_length": len(content),
                    },
                )

        except (RuntimeError, StorageError) as e:
            # Handle duplicate memory (detected by database UNIQUE constraint)
            if "Duplicate memory" in str(e):
                if self.logger:
                    self.logger.debug(
                        "Skipping duplicate memory (detected by DB constraint)",
                        extra={"workspace_id": self.config.workspace_id},
                    )
                return
            # Re-raise other errors
            if self.logger:
                self.logger.error(
                    "Failed to add memory to USearch index",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "error": str(e),
                    },
                )
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to add memory to USearch index",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to add memory: {e}") from e

    def add_batch(
        self,
        workspace_id: str,
        contents: list[str],
        infer: bool,
        vectors: list[list[float]] | None = None,
    ) -> list[str]:
        """Add multiple memories to the USearch index in a single batch.

        Args:
            workspace_id: Workspace identifier for filtering.
            contents: List of memory texts to index.
            infer: Whether to enable LLM-based memory inference (not supported).
            vectors: Optional precomputed embeddings aligned with ``contents``.
                When provided, this method does not call the embedder.

        Returns:
            List of memory contents successfully added (duplicates skipped).
        """
        if not contents:
            return []

        prepared = vectors
        if prepared is None:
            try:
                prepared = self.embedder.embed_documents(contents)
            except Exception as embed_error:
                raise RuntimeError(
                    f"Failed to add memory batch: {embed_error}"
                ) from embed_error
            if len(prepared) != len(contents) or any(not item for item in prepared):
                raise RuntimeError(
                    "Failed to add memory batch: Embedding batch size mismatch "
                    "for USearch add_batch"
                )
        with self._write_lease():
            _ = self.index
            self._maybe_reload_external()
            return self._add_batch_unlocked(workspace_id, contents, infer, prepared)

    def _add_batch_unlocked(
        self,
        workspace_id: str,
        contents: list[str],
        infer: bool,
        vectors: list[list[float]] | None,
    ) -> list[str]:
        _ = self.index
        if infer and self.logger:
            self.logger.warning(
                "infer=True not supported by USearchEngine, proceeding as infer=False",
                extra={"workspace_id": self.config.workspace_id},
            )

        inserted = []  # Track successfully inserted (content, mem_id) pairs
        inserted_ids = []  # Track IDs for rollback on embedding failure

        try:
            # First, insert all memories into SQLite
            inserted = self.memory_store.insert_many(workspace_id, contents)
            if not inserted:
                return []

            inserted_contents = [content for content, _ in inserted]
            inserted_ids = [mem_id for _, mem_id in inserted]

            # Generate embeddings with rollback on failure
            try:
                if vectors is None:
                    computed = self.embedder.embed_documents(inserted_contents)
                else:
                    content_to_vector = dict(zip(contents, vectors, strict=True))
                    computed = [
                        content_to_vector[content] for content in inserted_contents
                    ]
                if len(computed) != len(inserted_contents):
                    raise RuntimeError(
                        "Embedding batch size mismatch for USearch add_batch"
                    )
                if any(not vector for vector in computed):
                    raise RuntimeError("Embedding batch contained an empty vector")
            except Exception as embed_error:
                # Rollback all SQLite inserts if embedding fails
                if self.logger:
                    self.logger.error(
                        "Embedding generation failed, rolling back batch",
                        extra={
                            "workspace_id": self.config.workspace_id,
                            "count": len(inserted_ids),
                            "error": str(embed_error),
                        },
                    )
                for mem_id in inserted_ids:
                    _ = self.memory_store.delete(mem_id)
                raise RuntimeError(
                    f"Failed to generate embeddings for batch: {embed_error}"
                ) from embed_error

            # Add vectors to USearch index
            indexed_ids: list[int] = []
            try:
                self._index_vectors(inserted, computed, indexed_ids)
                self._dirty = True
            except Exception:
                self._rollback_batch_inserts(inserted_ids, indexed_ids)
                raise

            if self.logger:
                self.logger.debug(
                    "Batch added memories to USearch index",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "memory_count": len(inserted_contents),
                    },
                )

            return inserted_contents

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to add memory batch to USearch index",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to add memory batch: {e}") from e

    def _index_vectors(
        self,
        inserted: list[tuple[str, int]],
        vectors: list[list[float]],
        indexed_ids: list[int],
    ) -> None:
        """Add vectors one by one, appending each key before the next add."""
        keys = [mem_id for _, mem_id in inserted]
        matrix = np.asarray(vectors, dtype=np.float32)
        with self._index_lock:
            for mem_id, vector in zip(keys, matrix, strict=True):
                key = int(mem_id)
                self.index.add(key, vector)
                indexed_ids.append(key)

    def _rollback_batch_inserts(
        self, inserted_ids: list[int], indexed_ids: list[int]
    ) -> None:
        """Undo SQLite rows and any vectors added before a mid-batch failure."""
        with self._index_lock:
            for mem_id in indexed_ids:
                self.index.remove(int(mem_id))
        for mem_id in inserted_ids:
            _ = self.memory_store.delete(mem_id)

    def _should_use_exact_search(self) -> bool:
        """Determine if exact search should be used based on config and index size.

        Returns:
            True if exact (brute-force) search should be used, False for HNSW approximate.
        """
        # If exact search is explicitly forced, use it
        if self.config.exact_search:
            return True

        # If threshold is set and index size is below it, auto-switch to exact
        if self.config.exact_search_threshold > 0:
            index_size = len(self.index)
            if index_size < self.config.exact_search_threshold:
                return True

        # Default to approximate search (HNSW)
        return False

    def _rank_scores(self, count: int) -> np.ndarray:
        """Create rank-based scores from 1.0 to 0.0.

        Used when distance-to-similarity conversion is ambiguous.
        """
        if count <= 0:
            return np.array([], dtype=np.float32)
        if count == 1:
            return np.array([1.0], dtype=np.float32)
        return np.linspace(1.0, 0.0, num=count, dtype=np.float32)

    def _distances_to_scores(self, distances: np.ndarray) -> np.ndarray:
        """Convert index distances to similarity scores.

        Cosine distances are mapped to [0, 1] similarity.
        L2 distances are converted with a monotonic 1 / (1 + d) transform.
        Other metrics fall back to rank-based scoring to preserve ordering.
        """
        metric = self.config.metric.lower()
        if metric in ("cos", "cosine"):
            return distance_to_similarity_cosine(distances)
        if metric in ("l2", "euclidean"):
            return np.float32(1.0) / (np.float32(1.0) + distances)

        if self.logger:
            self.logger.warning(
                "USearch metric has ambiguous distance scale; using rank-based scores",
                extra={"metric": self.config.metric},
            )
        return self._rank_scores(len(distances))

    def search(
        self,
        query: str,
        workspace_id: str,
        limit: int,
    ) -> list[tuple[str, float, str]]:
        """Execute semantic search.

        Supports both exact (brute-force) and approximate (HNSW) search modes:
        - **Exact search**: Bypasses HNSW indexing and performs brute-force search
          using SIMD-optimized similarity metrics from SimSIMD. Best for small
          collections (< 10k vectors) where guaranteed exact results are needed.
        - **Approximate search**: Uses HNSW algorithm for fast nearest neighbor
          search. Best for larger collections where slight accuracy tradeoff
          for speed is acceptable.

        Search mode is determined by:
        1. `exact_search=True` in config: Always use exact search
        2. `exact_search_threshold > 0` and index size < threshold: Auto-switch to exact
        3. Otherwise: Use approximate (HNSW) search

        Args:
            query: Search query string.
            workspace_id: Filter results by workspace_id.
            limit: Maximum number of results.

        Returns:
            List of (content, score, created_at) tuples sorted by relevance.
            created_at is an ISO format timestamp string (may be empty for
            backward compatibility with older data).
        """
        try:
            with self._read_lease():
                self._maybe_reload_external()
                if len(self.index) == 0:
                    if self.logger:
                        self.logger.debug(
                            "USearch index is empty",
                            extra={"workspace_id": self.config.workspace_id},
                        )
                    return []

            use_exact = self._should_use_exact_search()
            self._log_search_mode(use_exact)

            matches = self._execute_index_search(query, limit, use_exact)
            filtered_matches = self._filter_matches_by_workspace(
                matches, workspace_id, limit
            )
            results = self._build_search_results(filtered_matches)

            if self.logger:
                self.logger.debug(
                    "USearch search completed",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "query_length": len(query),
                        "search_mode": "exact" if use_exact else "approximate",
                        "matches_found": len(matches),
                        "results_after_filter": len(results),
                        "numba_batch_size": len(filtered_matches),
                    },
                )

            return results

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "USearch search failed",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "query_length": len(query),
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"USearch search failed: {e}") from e

    def _log_search_mode(self, use_exact: bool) -> None:
        """Log the selected search mode with configuration details."""
        if not self.logger:
            return
        search_mode = "exact (brute-force)" if use_exact else "approximate (HNSW)"
        self.logger.debug(
            f"USearch search mode: {search_mode}",
            extra={
                "workspace_id": self.config.workspace_id,
                "search_mode": "exact" if use_exact else "approximate",
                "index_size": len(self.index),
                "exact_search_config": self.config.exact_search,
                "exact_search_threshold": self.config.exact_search_threshold,
            },
        )

    def _execute_index_search(
        self, query: str, limit: int, use_exact: bool
    ) -> BatchMatches:
        """Generate query embedding and execute index search with overfetch."""
        query_vector = self.embedder.embed_query(query)
        query_np = np.array(query_vector, dtype=np.float32)
        overfetch_limit = min(max(limit * 3, 1), len(self.index))
        with self._read_lease():
            self._maybe_reload_external()
            with self._index_lock:
                return self.index.search(query_np, overfetch_limit, exact=use_exact)

    def _filter_matches_by_workspace(
        self,
        matches: BatchMatches,
        workspace_id: str,
        limit: int,
    ) -> list[tuple[MemoryRecord, Match]]:
        """Filter matches by workspace_id using batch record fetch."""
        keys = [int(match.key) for match in matches]
        records = self.memory_store.get_batch(keys)

        filtered: list[tuple[MemoryRecord, Match]] = []
        for match in matches:
            key = int(match.key)
            record = records.get(key)
            if record is not None and record.workspace_id == workspace_id:
                filtered.append((record, match))
                if len(filtered) >= limit:
                    break
        return filtered

    def _build_search_results(
        self, filtered_matches: list[tuple[MemoryRecord, Match]]
    ) -> list[tuple[str, float, str]]:
        """Convert filtered matches to scored results using numba distance conversion."""
        if not filtered_matches:
            return []

        distances = np.array(
            [match.distance for _, match in filtered_matches],
            dtype=np.float32,
        )
        similarities = self._distances_to_scores(distances)

        return [
            (record.content, float(similarities[i]), record.created_at)
            for i, (record, _) in enumerate(filtered_matches)
        ]

    def count(self, workspace_id: str) -> int:
        """Return how many memories exist for a workspace."""
        return self.memory_store.count(workspace_id)

    def get_all(
        self,
        workspace_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[str]:
        """Retrieve stored memories for a workspace, optionally paged.

        Args:
            workspace_id: Workspace identifier for filtering.
            limit: Maximum rows to return.
            offset: Rows to skip.

        Returns:
            List of memories stored for the project.
        """
        try:
            memories = self.memory_store.get_all(
                workspace_id, limit=limit, offset=offset
            )

            if self.logger:
                self.logger.debug(
                    "Retrieved all memories",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "count": len(memories),
                    },
                )

            return memories

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to retrieve memories",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to retrieve memories: {e}") from e

    def delete(self, memory_id: str) -> None:
        """Delete a memory entry by its ID.

        Args:
            memory_id: The ID of the memory to delete (SQLite row ID as string).

        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            mem_id = int(memory_id)

            with self._write_lease():
                _ = self.index
                self._maybe_reload_external()
                deleted = self.memory_store.delete(mem_id)
                with self._index_lock:
                    if mem_id in self.index:
                        self.index.remove(mem_id)
                        self._dirty = True

            if self.logger:
                if deleted:
                    self.logger.debug(
                        "Memory deleted from USearch index",
                        extra={
                            "workspace_id": self.config.workspace_id,
                            "memory_id": memory_id,
                        },
                    )
                else:
                    self.logger.warning(
                        "Memory not found for deletion",
                        extra={
                            "workspace_id": self.config.workspace_id,
                            "memory_id": memory_id,
                        },
                    )

        except ValueError as e:
            if self.logger:
                self.logger.error(
                    "Invalid memory_id format",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "memory_id": memory_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Invalid memory_id format: {e}") from e
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to delete memory",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "memory_id": memory_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to delete memory: {e}") from e

    def contains_id(self, memory_id: int) -> bool | None:
        """Return whether the HNSW index contains ``memory_id`` under the index lock."""
        if self._index is None:
            return None
        with self._index_lock:
            try:
                return memory_id in self.index
            except TypeError:
                return None

    def commit(self) -> None:
        """Commit pending changes to the index.

        Saves the USearch index to disk. SQLite auto-commits.
        """
        try:
            if self._index is not None and self._dirty:
                with self._write_lease():
                    self._maybe_reload_external(refuse_stale=True)
                    with self._index_lock:
                        self._publish_index()
                if self.logger:
                    self.logger.debug(
                        "USearch index saved",
                        extra={
                            "workspace_id": self.config.workspace_id,
                            "index_path": self.config.index_path,
                            "size": len(self.index),
                        },
                    )
        except StorageError:
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to save USearch index",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to save USearch index: {e}") from e

    def ensure_initialized(self) -> None:
        """Ensure the engine is fully initialized (thread-safe).

        Forces the lazy initialization of USearch index and MemoryStore
        to complete. Call this before parallel operations.
        """
        _ = self.index
        self.memory_store.ensure_initialized()

    def is_ready(self) -> bool:
        """Return True if the USearch index and SQLite store are loaded."""
        return (
            self._index is not None
            and self._memory_store is not None
            and self._memory_store.is_ready()
        )

    def get_id_by_content(self, workspace_id: str, content: str) -> int | None:
        """Get the ID of a memory by its content.

        Args:
            workspace_id: Workspace identifier.
            content: Memory text to look up.

        Returns:
            The memory ID if found, None otherwise.
        """
        return self.memory_store.get_id_by_content(workspace_id, content)

    def get_records_by_contents(
        self, workspace_id: str, contents: list[str]
    ) -> list[IStoredMemory]:
        """Return stored rows for the requested contents in one workspace."""
        return list(self.memory_store.get_records_by_contents(workspace_id, contents))

    def close(self) -> None:
        """Close resources and cleanup.

        Closes the MemoryStore SQLite connection.
        """
        with self._init_lock:
            if self._closed:
                return
            self._closed = True
            if self._memory_store is not None:
                self._memory_store.close()
            self._index = None
            if isinstance(self.embedder, Closable):
                self.embedder.close()

    def __enter__(self) -> Self:
        """Enter context manager.

        Ensures the engine is initialized before use.

        Returns:
            self
        """
        self.ensure_initialized()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        """Exit context manager.

        Ensures resources are cleaned up.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised

        Returns:
            False to not suppress exceptions.
        """
        self.close()
        return False  # Don't suppress exceptions
