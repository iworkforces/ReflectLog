"""Memory management wrapper for hybrid search integration.

This module provides the MemoryManager class that orchestrates all memory
operations. It combines semantic vector search (USearch) with full-text
search (Tantivy) using Reciprocal Rank Fusion (RRF) for intelligent
result ranking.

Protocol Compliance:
    MemoryManager implements the IMemoryManager protocol from core/memory.py.
    This enables:
    - Dependency injection with mock implementations for testing
    - Runtime component substitution (different backends)
    - Type-safe interaction with the memory system

Example:
    from reflectlog.core.memory import IMemoryManager
    from reflectlog.core.config_adapters import ConfigAdapter

    # Protocol-based usage
    manager: IMemoryManager = MemoryManager(config, logger)

    # With dependency injection
    mock_backend = create_mock_backend()
    manager = MemoryManager(
        config=config,
        logger=logger,
        semantic_backend=mock_backend,
    )
"""

from contextlib import contextmanager, suppress
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, final, runtime_checkable

from asyncer import asyncify

from reflectlog.application.constants import LOG_ADD_MEMORY_PREVIEW_LIMIT
from reflectlog.core.exceptions import (
    ConfigurationError,
    InconsistentStateError,
    InitializationError,
    SearchError,
    StorageError,
)
from reflectlog.infrastructure.cross_encoder_reranker import (
    CrossEncoderConfig,
    CrossEncoderReranker,
)
from reflectlog.infrastructure.embeddings.cached_embeddings import CachedEmbeddings
from reflectlog.infrastructure.embeddings.qwen3_embedding import LangchainQwenEmbeddings
from reflectlog.infrastructure.smart_replacer import SmartReplacer, SmartReplacerConfig
from reflectlog.infrastructure.tantivy_engine import TantivyConfig, TantivyEngine
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine

from ...core.config_adapters import ConfigAdapter
from ...core.enums import (
    EmbedderProvider,
    EngineReadiness,
    RerankerEngine,
    TransitionKind,
)
from ...core.storage_coordination import IStorageCoordinator, LeaseMode
from ...infrastructure.storage_coordinator import PortalockerStorageCoordinator
from .add_phases import (
    AddPipeline,
    AddResult,
    DuplicateDetectionPhase,
    SmartReplacementPhase,
    StoragePhase,
)
from .fusion import create_fusion_engine
from .match_utils import has_exact_match
from .replacement_recovery import (
    reconcile_pending_replacements as apply_pending_replacements,
)
from .search_strategies import (
    SearchContext,
    SearchPipeline,
    calculate_adaptive_overfetch,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from ...core.logging import IStructuredLogger
    from ...core.types import ISemanticSearchEngine, ReplacementTransition
    from ..config.settings import Config
    from .fusion.base import FusionEngine


class _ReadyEngine(Protocol):
    """Engine that can report whether lazy initialization finished."""

    def is_ready(self) -> bool: ...


@runtime_checkable
class _RefreshableEngine(Protocol):
    def refresh(self) -> None: ...


@final
class MemoryManager:
    """Manages memory storage and retrieval using USearch with SQLite backend."""

    def __init__(
        self,
        config: Config,
        logger: IStructuredLogger | None,
        *,
        coordinator: IStorageCoordinator | None = None,
        orchestration_hook: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize Hybrid MemoryManager with USearch semantic
        & Tantivy full-text engines.

        Args:
            config: Application configuration.
            logger: Structured logger instance.
            coordinator: Optional workspace storage coordinator.
            orchestration_hook: Test-only failpoint invoked at generation/intent seams.
        """
        super().__init__()
        if logger is None:
            raise ValueError("logger is required")

        self.config = config
        self.logger: IStructuredLogger = logger
        self.workspace_id = config.workspace_id
        self.orchestration_hook = orchestration_hook

        self._init_locks()
        self._coordinator = coordinator or self._create_coordinator()
        try:
            self._init_semantic_engine()
            self._init_search_engine()
            self._init_fusion_engine()
            self._init_rerankers()
            self._init_smart_replacer()
            self._init_pipelines()
            self._log_configuration()
            try:
                _ = self.reconcile_pending_replacements()
            except InitializationError:
                raise
            except Exception as exc:
                self.logger.error(
                    "Startup replacement reconcile failed; continuing with pending rows",
                    extra={"error": str(exc)},
                )
        except BaseException:
            self._dispose_partial_init()
            raise

        if config.eager_initialization:
            self._eager_initialize_engines()

    def _init_locks(self) -> None:
        """Create all threading locks and startup metrics placeholder.

        Lock hierarchy (outer to inner) — ALWAYS follow this order
        to prevent deadlocks:
          1. _write_lock — for all write operations across async/sync boundary
          2. _lock — for read operations and inner critical sections

        Usage pattern:
          - Read operations: use _lock only
          - Write operations: acquire _write_lock first, then _lock if needed
          - Never acquire _lock before _write_lock (risk of deadlock)
        """
        self._lock = threading.RLock()
        self._write_lock = threading.RLock()
        self._closed = False
        self._closing = False
        self.startup_metrics: dict[str, float] | None = None
        self._reranker_lock = threading.RLock()
        self._smart_replacer_lock = threading.RLock()

    def _create_coordinator(self) -> IStorageCoordinator:
        root = os.path.abspath(ConfigAdapter(self.config).storage_path)
        return PortalockerStorageCoordinator(root)

    def _emit_orchestration_hook(self, step: str) -> None:
        hook = self.orchestration_hook
        if hook is not None:
            hook(step)

    @contextmanager
    def _exclusive_workspace(self) -> Generator[None]:
        with self._coordinator.acquire(self.workspace_id, LeaseMode.EXCLUSIVE):
            yield

    @contextmanager
    def _shared_workspace(self) -> Generator[None]:
        with self._coordinator.acquire(self.workspace_id, LeaseMode.SHARED):
            yield

    def _publish_converged_generation(self) -> None:
        current = self._coordinator.read_generation(self.workspace_id)
        self._emit_orchestration_hook("before_generation")
        self._coordinator.publish_generation(self.workspace_id, current + 1)
        self._emit_orchestration_hook("after_generation")

    def _complete_intents_after_generation(self, completer: Callable[[], None]) -> None:
        self._publish_converged_generation()
        self._emit_orchestration_hook("before_intent")
        completer()
        self._emit_orchestration_hook("after_intent")

    def _refresh_engines(self) -> None:
        engine = self._semantic_engine
        if isinstance(engine, _RefreshableEngine):
            engine.refresh()

    def _tantivy_has(self, content: str) -> bool | None:
        engine = self._tantivy_engine
        if engine is None:
            return None
        return content in engine.find_by_exact_match(self.workspace_id, content)

    def _validate_contents_present(self, contents: list[str]) -> None:
        for content in contents:
            sqlite_id = self.get_id_by_content(content)
            tantivy_has = self._tantivy_has(content)
            if sqlite_id is None and tantivy_has is True:
                raise InconsistentStateError(
                    "SQLite is missing a memory after a coordinated write"
                )
            if sqlite_id is not None:
                has_vector = self._semantic_engine.contains_id(sqlite_id)
                if has_vector is False:
                    raise InconsistentStateError(
                        "USearch is missing a vector after a coordinated write"
                    )
            if sqlite_id is not None and tantivy_has is False:
                raise InconsistentStateError(
                    "Tantivy is missing a memory after a coordinated write"
                )

    def _validate_contents_absent(self, contents: list[str]) -> None:
        for content in contents:
            sqlite_id = self.get_id_by_content(content)
            tantivy_has = self._tantivy_has(content)
            if sqlite_id is not None and tantivy_has is True:
                raise InconsistentStateError(
                    "SQLite and Tantivy still have a memory after a coordinated delete"
                )
            if sqlite_id is None and tantivy_has is True:
                raise InconsistentStateError(
                    "Tantivy still has a memory after a coordinated delete"
                )

    def _init_semantic_engine(self) -> None:
        """Create USearch semantic engine with optional embedding cache."""
        config = self.config
        usearch_config = USearchConfig.from_config(ConfigAdapter(config))
        base_embedder = LangchainQwenEmbeddings(
            config={
                "model": config.embedding_model,
                "embedding_dims": config.qwen_embedding_dims
                if config.embedder_provider == EmbedderProvider.LANGCHAIN
                else config.embedding_dims,
                "api_key": config.openrouter_api_key.get_secret_value(),
                "openai_base_url": config.openrouter_base_url,
                "batch_size": config.embedding_batch_size,
                "max_concurrent_batches": config.embedding_max_concurrent_batches,
            }
        )
        if config.embedding_cache_enabled:
            embedder = CachedEmbeddings(
                embedder=base_embedder,
                cache_size=config.embedding_cache_size,
                enabled=True,
                logger=self.logger,
            )
        else:
            embedder = base_embedder
        self._semantic_engine: ISemanticSearchEngine = USearchEngine(
            usearch_config,
            embedder=embedder,
            logger=self.logger,
            coordinator=self._coordinator,
        )

    def _init_search_engine(self) -> None:
        """Create the Tantivy full-text engine."""
        tantivy_config = TantivyConfig(
            workspace_id=self.workspace_id,
            index_path=self.config.tantivy_index_path_template.format(
                workspace_id=self.workspace_id
            ).lower(),
            normalize_scores=self.config.tantivy_normalize_scores,
            soft_delete_enabled=self.config.tantivy_soft_delete_enabled,
            compaction_threshold_ratio=self.config.tantivy_compaction_threshold_ratio,
            compaction_max_tombstones=self.config.tantivy_compaction_max_tombstones,
            tombstone_ttl_days=self.config.tantivy_tombstone_ttl_days,
        )
        self._tantivy_engine: TantivyEngine | None = TantivyEngine(
            tantivy_config,
            logger=self.logger,
            coordinator=self._coordinator,
        )

    def _init_fusion_engine(self) -> None:
        """Create fusion engine for hybrid ranking."""
        fusion_weights = self.config.fusion_weights
        self._fusion_engine: FusionEngine = create_fusion_engine(
            method=self.config.fusion_method,
            normalization=self.config.fusion_normalization,
            rrf_k=self.config.fusion_rrf_k,
            weights=fusion_weights if isinstance(fusion_weights, list) else None,
            logger=self.logger,
        )

    def _init_rerankers(self) -> None:
        """Set up reranker references for lazy initialization via properties.

        The cross-encoder is created on first search to avoid startup overhead.
        """
        self._cross_encoder_reranker: CrossEncoderReranker | None = None

        config = self.config
        if config.reranker_engine == RerankerEngine.CROSS_ENCODER:
            self.logger.info(
                f"CrossEncoder reranker configured (lazy init) "
                f"[model={config.cross_encoder_model}]",
                extra={
                    "reranker_engine": RerankerEngine.CROSS_ENCODER,
                    "model": config.cross_encoder_model,
                    "device": config.cross_encoder_device,
                },
            )
        elif config.reranker_engine == RerankerEngine.NONE:
            self.logger.info(
                "Reranking disabled (RERANKER_ENGINE=none)",
                extra={"reranker_engine": RerankerEngine.NONE},
            )
        else:
            raise ConfigurationError(
                f"Invalid RERANKER_ENGINE: '{config.reranker_engine}'. "
                "Valid options: cross_encoder, none"
            )

    def _init_smart_replacer(self) -> None:
        """Set up smart replacer reference for lazy initialization via property.

        SmartReplacer is created on first add operation
        to avoid startup overhead.
        """
        self._smart_replacer: SmartReplacer | None = None

        config = self.config
        if config.enable_smart_replace:
            self.logger.info(
                f"SmartReplacer configured (lazy init) [model={config.llm_model}, "
                f"threshold={config.smart_replace_threshold}]",
                extra={
                    "smart_replacer": "enabled",
                    "model": config.llm_model,
                    "threshold": config.smart_replace_threshold,
                },
            )
        else:
            self.logger.info(
                "Smart memory replacement disabled (ENABLE_SMART_REPLACE=false)",
                extra={"smart_replacer": "disabled"},
            )

    def _init_pipelines(self) -> None:
        """Create SearchPipeline and AddPipeline with their phase components."""
        self._search_pipeline = SearchPipeline(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            fusion_engine=self._fusion_engine,
            config=self.config,
            logger=self.logger,
            memory_manager=self,  # Pass self for lazy reranker fetching
        )

        self._duplicate_detection_phase = DuplicateDetectionPhase(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            config=self.config,
            logger=self.logger,
        )
        self._smart_replacement_phase = SmartReplacementPhase(
            semantic_engine=self._semantic_engine,
            config=self.config,
            logger=self.logger,
            memory_manager=self,  # Pass self for lazy SmartReplacer fetching
        )
        self._storage_phase = StoragePhase(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            config=self.config,
            logger=self.logger,
            write_lock=self._write_lock,
            lock=self._lock,
            ensure_open=self._ensure_open,
            coordinator=self._coordinator,
            orchestration_hook=self._emit_orchestration_hook,
        )
        self._add_pipeline = AddPipeline(
            duplicate_detection_phase=self._duplicate_detection_phase,
            smart_replacement_phase=self._smart_replacement_phase,
            storage_phase=self._storage_phase,
            config=self.config,
            logger=self.logger,
        )

    def reconcile_pending_replacements(self) -> int:
        """Finish replacements interrupted by a previous process stop.

        Called at startup and at the start of the next add persist.
        Not invoked by health_check. Acquires ``_write_lock`` then ``_lock``.
        """
        return apply_pending_replacements(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            write_lock=self._write_lock,
            lock=self._lock,
            logger=self.logger,
            coordinator=self._coordinator,
            workspace_id=self.workspace_id,
            orchestration_hook=self._emit_orchestration_hook,
        )

    def pending_replacement_count(self) -> int:
        """Return how many journal intents are still pending (all kinds)."""
        return self.pending_intent_count()

    def pending_intent_count(self) -> int:
        """Return how many add/delete/replace intents are still pending.

        Listing failures propagate so health cannot report a healthy zero
        while the journal is unreadable.
        """
        return len(self._semantic_engine.memory_store.list_pending_transitions())

    def _log_configuration(self) -> None:
        """Log the final configuration state after initialization."""
        self.logger.info(
            f"Initialized Hybrid MemoryManager [workspace_id={self.workspace_id}, "
            f"semantic_backend=usearch, "
            f"embedding_model={self.config.embedding_model}, "
        )
        self.logger.info(
            f"tantivy_index={self.config.tantivy_index_path_template.format(workspace_id=self.workspace_id)}, "
            f"embedder={self.config.embedder_provider}]",
        )

    def _eager_initialize_engines(self) -> None:
        """Pre-warm storage engines for faster first operation.

        This method forces initialization of lazy-loaded resources based on
        granular configuration settings:
        - USearch index and libSQL connection (when eager_initialize_search_engines)
        - Tantivy index, writer, and searcher (when eager_initialize_search_engines)
        - Reranker (when eager_initialize_reranker)
        - SmartReplacer (when eager_initialize_smart_replacer)

        Granular settings take precedence over the general EAGER_INITIALIZATION flag.
        If granular settings are None (not explicitly configured), falls back to
        EAGER_INITIALIZATION for search engines only (rerankers are lazy by default).

        Useful for reducing first-request latency in production deployments.
        Called during __init__ when enabled.
        """
        start_time = time.time()

        # Determine which components to initialize
        # Priority: granular setting > general eager initialization
        # > default (search only)
        should_init_search = (
            self.config.eager_initialize_search_engines
            if self.config.eager_initialize_search_engines is not None
            else self.config.eager_initialization
        )
        should_init_reranker = (
            self.config.eager_initialize_reranker
            if self.config.eager_initialize_reranker is not None
            else False  # Rerankers are lazy by default
        )
        should_init_smart_replacer = (
            self.config.eager_initialize_smart_replacer
            if self.config.eager_initialize_smart_replacer is not None
            else False  # SmartReplacer is lazy by default
        )

        engines_initialized: list[str] = []

        # Pre-warm search engines if configured
        if should_init_search:
            self.logger.info(
                "Starting eager search engine initialization...",
                extra={"workspace_id": self.workspace_id},
            )

            # Pre-warm USearch semantic engine
            self._semantic_engine.ensure_initialized()
            engines_initialized.append("usearch")

            # Pre-warm Tantivy full-text engine
            if self._tantivy_engine is not None:
                self._tantivy_engine.ensure_initialized()
                engines_initialized.append("tantivy")

        # Pre-warm reranker if explicitly configured (lazy by default)
        if should_init_reranker:
            # Validate that reranker engine type is supported
            if self.config.reranker_engine != RerankerEngine.CROSS_ENCODER:
                raise ValueError(
                    f"Invalid reranker_engine for eager initialization: "
                    f"{self.config.reranker_engine!r}. "
                    f"Must be 'cross_encoder', or set "
                    f"eager_initialize_reranker=false for lazy loading."
                )

            self.logger.info(
                "Starting eager reranker initialization...",
                extra={
                    "workspace_id": self.workspace_id,
                    "reranker_engine": self.config.reranker_engine,
                },
            )
            reranker = self.get_reranker()
            if reranker is not None:
                _ = reranker.model
                engines_initialized.append(f"reranker_{self.config.reranker_engine}")
            else:
                self.logger.warning(
                    "Eager reranker initialization requested but no reranker configured",
                    extra={
                        "workspace_id": self.workspace_id,
                        "reranker_engine": self.config.reranker_engine,
                    },
                )

        # Pre-warm SmartReplacer if explicitly configured (lazy by default)
        if should_init_smart_replacer:
            # Validate that smart replacement is enabled
            if not self.config.enable_smart_replace:
                raise ValueError(
                    "Eager SmartReplacer initialization requested but "
                    "enable_smart_replace is disabled. Set enable_smart_replace=true "
                    "or set eager_initialize_smart_replacer=false for lazy loading."
                )

            self.logger.info(
                "Starting eager SmartReplacer initialization...",
                extra={"workspace_id": self.workspace_id},
            )
            _ = self.smart_replacer
            engines_initialized.append("smart_replacer")

        elapsed_ms = (time.time() - start_time) * 1000

        if engines_initialized:
            self.logger.info(
                f"Eager initialization complete [{elapsed_ms:.1f}ms]",
                extra={
                    "workspace_id": self.workspace_id,
                    "elapsed_ms": elapsed_ms,
                    "engines_initialized": engines_initialized,
                },
            )
        else:
            self.logger.info(
                "Eager initialization skipped (all components set to lazy loading)",
                extra={"workspace_id": self.workspace_id},
            )

    @property
    def cross_encoder_reranker(self) -> CrossEncoderReranker | None:
        """Get CrossEncoder reranker (lazy initialization with thread-safety).

        Returns:
            CrossEncoderReranker instance if configured, None otherwise.

        Raises:
            RuntimeError: If reranker_engine is 'cross_encoder' but initialization fails.
        """
        # Fast path: already initialized or not configured
        if (
            self._cross_encoder_reranker is not None
            or self.config.reranker_engine != RerankerEngine.CROSS_ENCODER
        ):
            return self._cross_encoder_reranker

        # Slow path: need to initialize with lock
        with self._reranker_lock:
            # Double-check after acquiring lock
            if (
                self._cross_encoder_reranker is not None
                or self.config.reranker_engine != RerankerEngine.CROSS_ENCODER
            ):
                return self._cross_encoder_reranker

            # Initialize CrossEncoder reranker
            ce_config = CrossEncoderConfig.from_config(ConfigAdapter(self.config))
            self._cross_encoder_reranker = CrossEncoderReranker(
                config=ce_config, logger=self.logger
            )
            self.logger.info(
                f"Lazy initialized CrossEncoder reranker [model={self.config.cross_encoder_model}]",
                extra={
                    "reranker_engine": RerankerEngine.CROSS_ENCODER,
                    "model": self.config.cross_encoder_model,
                    "device": self.config.cross_encoder_device,
                },
            )
            return self._cross_encoder_reranker

    @property
    def smart_replacer(self) -> SmartReplacer | None:
        """Get SmartReplacer (lazy initialization with thread-safety).

        Returns:
            SmartReplacer instance if smart replacement is enabled, None otherwise.

        Raises:
            RuntimeError: If ENABLE_SMART_REPLACE=true but initialization fails.
        """
        # Fast path: already initialized or not configured
        if self._smart_replacer is not None or not self.config.enable_smart_replace:
            return self._smart_replacer

        # Slow path: need to initialize with lock
        with self._smart_replacer_lock:
            # Double-check after acquiring lock
            if self._smart_replacer is not None or not self.config.enable_smart_replace:
                return self._smart_replacer

            # Initialize SmartReplacer
            smart_replacer_config = SmartReplacerConfig.from_config(
                ConfigAdapter(self.config)
            )
            self._smart_replacer = SmartReplacer(
                config=smart_replacer_config, logger=self.logger
            )
            self.logger.info(
                f"Lazy initialized SmartReplacer [model={self.config.llm_model}]",
                extra={
                    "smart_replacer": "enabled",
                    "model": self.config.llm_model,
                },
            )
            return self._smart_replacer

    def get_reranker(self) -> CrossEncoderReranker | None:
        """Get the configured cross-encoder reranker, or None if disabled."""
        if self.config.reranker_engine == RerankerEngine.CROSS_ENCODER:
            return self.cross_encoder_reranker
        return None

    def add_memories(self, memories: list[str]) -> int:
        """Add multiple memories to memory store (thread-safe).

        Thread-safe: Uses RLock to ensure consistent state during batch addition.

        Args:
            memories: List of memories to store.

        Returns:
            Number of memories actually stored (duplicates skipped).

        Raises:
            RuntimeError: If storage operation fails.
        """
        self._ensure_open()
        try:
            _ = self.reconcile_pending_replacements()
        except InitializationError:
            raise
        except Exception as exc:
            self.logger.error(
                "Pre-add reconcile failed; continuing with pending rows",
                extra={"error": str(exc)},
            )

        self._ensure_open()
        memories_to_add: list[str] = []
        seen_memories: set[str] = set()
        log_limit = min(len(memories), LOG_ADD_MEMORY_PREVIEW_LIMIT)
        for idx, memory in enumerate(memories, 1):
            if idx <= log_limit:
                self.logger.info(
                    f"  ⏳ [{idx}/{len(memories)}] Processing memory",
                    extra={
                        "memory_index": idx,
                        "total_memories": len(memories),
                        "memory_length": len(memory),
                    },
                )
            if memory in seen_memories:
                if idx <= log_limit:
                    self.logger.info(
                        "    Skipped (duplicate in batch)",
                        extra={
                            "memory_index": idx,
                            "reason": "batch_duplicate",
                        },
                    )
                continue
            seen_memories.add(memory)
            memories_to_add.append(memory)
        if len(memories) > log_limit:
            self.logger.info(
                f"  ... {len(memories) - log_limit} more memory(s) omitted from logs",
                extra={
                    "omitted_count": len(memories) - log_limit,
                    "total_memories": len(memories),
                },
            )

        if not memories_to_add:
            return 0

        vectors = self._semantic_engine.embedder.embed_documents(memories_to_add)
        if len(vectors) != len(memories_to_add) or any(not item for item in vectors):
            raise StorageError("Embedding batch size mismatch or empty vector")

        with self._exclusive_workspace(), self._write_lock, self._lock:
            self._ensure_open()
            self._refresh_engines()
            persist_memories: list[str] = []
            persist_vectors: list[list[float]] = []
            for memory, vector in zip(memories_to_add, vectors, strict=True):
                if self.config.deduplicate_memories and self._has_exact_match(memory):
                    continue
                persist_memories.append(memory)
                persist_vectors.append(vector)
            if not persist_memories:
                return 0
            add_intents = self._record_add_intents(persist_memories)
            inserted_memories = self._semantic_engine.add_batch(
                workspace_id=self.workspace_id,
                contents=persist_memories,
                infer=self.config.enable_llm_infer,
                vectors=persist_vectors,
            )

            if self._tantivy_engine is not None:
                self._tantivy_engine.add_batch(self.workspace_id, inserted_memories)

            stored_count = len(inserted_memories)
            if stored_count != len(persist_memories):
                self.logger.warning(
                    "    Skipped during batch insert",
                    extra={
                        "reason": "batch_insert_skipped",
                        "expected_count": len(persist_memories),
                        "stored_count": stored_count,
                    },
                )

            if self._tantivy_engine is not None:
                self._tantivy_engine.commit()
                self.logger.info(
                    "  Tantivy index committed",
                    extra={"engine": "tantivy"},
                )

            self._semantic_engine.commit()
            self.logger.info(
                "  USearch index committed",
                extra={"engine": "usearch"},
            )
            self._validate_contents_present(inserted_memories)
            self._complete_intents_after_generation(
                lambda: self._complete_add_intents(add_intents)
            )

            return stored_count

    def add_messages(self, messages: list[str]) -> int:
        return self.add_memories(messages)

    async def add_memories_async(
        self, memories: list[str], dry_run: bool = False
    ) -> AddResult:
        """Add multiple memories with phased parallel processing (Sprint 2.2).

        Uses a 3-phase approach for optimal performance via AddPipeline:
        - Phase 1 (Parallel): Duplicate detection for all memories
        - Phase 2 (Parallel): Smart replacement detection for non-duplicates
        - Phase 3 (Sequential): Database writes to avoid SQLite corruption

        This approach provides 5-8x speedup over sequential processing for
        multiple memories by maximizing I/O parallelism in phases 1-2 while
        maintaining data consistency in phase 3.

        Args:
            memories: List of memories to store.
            dry_run: If True, only check for replacements without making changes.
                Returns what WOULD happen without actually storing or deleting.

        Returns:
            AddResult with detailed information about stored, skipped, and replaced
            memories, including full replacement details.

        Raises:
            RuntimeError: If storage operation fails (not raised in dry_run mode).
        """
        self._ensure_open()
        return await self._add_pipeline.execute(memories, dry_run)

    async def add_messages_async(
        self, messages: list[str], dry_run: bool = False
    ) -> AddResult:
        return await self.add_memories_async(messages, dry_run)

    def count(self) -> int:
        """Return how many memories exist in this workspace."""
        self._ensure_open()
        with self._shared_workspace(), self._lock:
            self._ensure_open()
            self._refresh_engines()
            return self._semantic_engine.count(self.workspace_id)

    def get_all(self, limit: int | None = None, offset: int = 0) -> list[str]:
        """Retrieve stored memories, paged at the semantic store.

        Thread-safe: Uses RLock to ensure consistent state during retrieval.

        Returns:
            Page of memories from USearchEngine (source of truth).

        Raises:
            RuntimeError: If retrieval operation fails.
        """
        self._ensure_open()
        try:
            with self._shared_workspace(), self._lock:
                self._ensure_open()
                self._refresh_engines()
                memories = self._semantic_engine.get_all(
                    workspace_id=self.workspace_id,
                    limit=limit,
                    offset=offset,
                )
            self.logger.info(
                f"Retrieved {len(memories)} memories (USearchEngine={len(memories)})",
                extra={
                    "workspace_id": self.workspace_id,
                    "count": len(memories),
                    "limit": limit,
                    "offset": offset,
                },
            )
            return memories
        except Exception as e:
            raise StorageError(f"Failed to retrieve memories: {e}") from e

    async def search(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[str]:
        """Hybrid semantic + full-text search with RRF ranking (async).

        This async method uses the SearchPipeline to execute a 4-step search:
        1. Parallel semantic + full-text search
        2. RRF fusion or concatenation
        3. Fusion threshold filtering (when RRF enabled)
        4. Reranking (CrossEncoder)

        Args:
            query: Search query string.
            limit: Maximum number of results (uses config default if None).

        Returns:
            List of hybrid-ranked memories from both engines.

        Raises:
            SearchError: If search operation fails.

        Note:
            Cancelling this await does not abort native USearch or Tantivy
            work already running in a worker thread.
        """
        self._ensure_open()
        if not query.strip():
            return []
        try:
            _ = await asyncify(self.reconcile_pending_replacements)()
        except InitializationError:
            raise
        except Exception as exc:
            self.logger.error(
                "Pre-search reconcile failed; continuing with pending rows",
                extra={"error": str(exc)},
            )
        self._ensure_open()

        # Use defaults from config if not provided
        if limit is None:
            limit = self.config.search_limit

        # Index restore / len() can block; keep it off the event loop.
        index_size: int = await asyncify(self._semantic_index_size)()
        overfetch_limit = calculate_adaptive_overfetch(limit, index_size, self.config)

        # Create search context
        context = SearchContext(
            query=query,
            limit=limit,
            overfetch_limit=overfetch_limit,
            enable_rrf_fusion=self.config.enable_rrf_fusion,
            reranker_engine=self.config.reranker_engine,
            workspace_id=self.workspace_id,
        )

        # Backend reads take their own short shared leases. Do not hold
        # SHARED across embed, fusion, or cross-encoder rerank.
        self._refresh_engines()
        result = await self._search_pipeline.execute(context)

        return result.memories

    def _semantic_index_size(self) -> int:
        """Return the workspace memory count used to size overfetch."""
        try:
            return self._semantic_engine.count(self.workspace_id)
        except Exception:
            return 0

    def search_engine_status(self) -> dict[str, EngineReadiness]:
        """Return public readiness of search engines for health checks.

        Values are ``initialized`` (warmed), ``pending`` (constructed but
        not yet ``ensure_initialized``), ``not_initialized``, or ``disabled``.
        """
        return {
            "semantic_engine": self._engine_readiness(
                self._semantic_engine, absent=EngineReadiness.NOT_INITIALIZED
            ),
            "tantivy_engine": self._engine_readiness(
                self._tantivy_engine, absent=EngineReadiness.DISABLED
            ),
        }

    def _engine_readiness(
        self, engine: _ReadyEngine | None, *, absent: EngineReadiness
    ) -> EngineReadiness:
        if engine is None:
            return absent
        try:
            if engine.is_ready():
                return EngineReadiness.INITIALIZED
        except Exception:
            return EngineReadiness.PENDING
        return EngineReadiness.PENDING

    def search_for_removal(
        self, query: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Search for memories to potentially remove using direct database lookup.

        Uses O(log n) indexed database lookup instead of O(n) iteration through
        all memories. This is a significant performance improvement for large
        collections.

        Args:
            query: The exact memory content to find for removal.
            limit: Maximum number of candidates (uses config default if None).
                   Note: Since we're looking for exact match, we'll return at most 1.

        Returns:
            List of candidate records with 'id', 'memory', and 'score' fields.
            Returns at most 1 candidate since exact match is unique.

        Raises:
            SearchError: If the lookup fails.
        """
        self._ensure_open()
        # Note: limit is kept for API compatibility but exact match returns at most 1
        _ = limit  # Unused, kept for API compatibility

        try:
            candidates: list[dict[str, Any]] = []

            # Direct database lookup - O(log n) instead of O(n) get_all() + iteration
            mem_id = self.get_id_by_content(query)

            if mem_id is not None:
                # Found exact match - return as candidate
                candidates.append(
                    {
                        "id": str(mem_id),  # Use database ID directly
                        "memory": query,
                        "score": 1.0,  # Exact match = perfect score
                    }
                )

            self.logger.debug(
                f"Direct lookup removal candidates: {len(candidates)}",
                extra={"workspace_id": self.workspace_id, "query_length": len(query)},
            )
            return candidates

        except Exception as e:
            raise SearchError(f"Failed to search for removal: {e}") from e

    def delete_by_id(self, memory_id: str) -> None:
        """Delete a memory entry by its ID.

        Args:
            memory_id: The ID of the memory to delete.

        Raises:
            StorageError: If deletion fails.
        """
        self._ensure_open()
        with self._exclusive_workspace(), self._write_lock, self._lock:
            self._ensure_open()
            self._refresh_engines()
            try:
                numeric_id = int(memory_id)
                record = self._semantic_engine.memory_store.get(numeric_id)
                content = record.content if record is not None else None
                if content is None:
                    self._finish_orphan_delete_by_id(numeric_id)
                    return
                delete_intents = self._record_delete_intents([(numeric_id, content)])
                self._semantic_engine.delete(memory_id=memory_id)
                self._semantic_engine.commit()
                if self._tantivy_engine is not None:
                    deleted = self._tantivy_engine.delete(
                        self.workspace_id, content, verify_exists=True
                    )
                    if deleted is not True:
                        raise InconsistentStateError(
                            "USearch deletion succeeded but Tantivy "
                            f"did not delete memory_id={memory_id}"
                        )
                    self._tantivy_engine.commit()
                self._validate_contents_absent([content])
                if delete_intents:
                    self._complete_intents_after_generation(
                        lambda: self._complete_delete_intents(delete_intents)
                    )
            except InconsistentStateError:
                raise
            except Exception as e:
                raise StorageError(f"Failed to delete memory: {e}") from e

    def delete_by_memory(self, memory: str) -> bool:
        """Delete a memory entry by its exact memory content (thread-safe).

        Thread-safe: Uses RLock to ensure consistent state during deletion.

        This method looks up the memory in the database and deletes it from
        both USearch (semantic) and Tantivy (full-text) engines atomically.
        If Tantivy deletion fails after USearch deletion succeeds, the operation
        raises an error to alert about the inconsistent state.

        Args:
            memory: The exact memory content to delete.

        Returns:
            True if the memory was found and deleted, False if not found.

        Raises:
            RuntimeError: If deletion fails or results in inconsistent state.
        """
        self._ensure_open()
        with self._exclusive_workspace(), self._write_lock, self._lock:
            self._ensure_open()
            self._refresh_engines()
            try:
                # 1. Look up the SQLite ID from the memory content
                mem_id = self.get_id_by_content(memory)

                if mem_id is None:
                    if self._finish_orphan_tantivy_delete(memory):
                        return True
                    self.logger.debug(
                        "Memory not found for deletion",
                        extra={
                            "workspace_id": self.workspace_id,
                            "memory_length": len(memory),
                        },
                    )
                    return False

                delete_intents = self._record_delete_intents([(mem_id, memory)])
                # 2. Delete from USearch semantic engine using the numeric ID
                self._semantic_engine.delete(memory_id=str(mem_id))
                self._semantic_engine.commit()

                # 3. Delete from Tantivy full-text engine
                # If this fails after USearch deletion, we have inconsistent state
                if self._tantivy_engine is not None:
                    try:
                        deleted = self._tantivy_engine.delete(
                            self.workspace_id, memory, verify_exists=True
                        )
                        if not deleted:
                            raise InconsistentStateError(
                                "USearch deletion succeeded but Tantivy "
                                f"did not delete memory_id={mem_id}"
                            )
                    except InconsistentStateError:
                        raise
                    except Exception as tantivy_error:
                        # Log critical error - state is now inconsistent
                        self.logger.error(
                            "Tantivy deletion failed after USearch deletion - INCONSISTENT STATE",
                            extra={
                                "workspace_id": self.workspace_id,
                                "memory_id": mem_id,
                                "error": str(tantivy_error),
                            },
                        )
                        raise InconsistentStateError(
                            "USearch deletion succeeded but Tantivy deletion failed: "
                            f"{tantivy_error}"
                        ) from tantivy_error

                self._validate_contents_absent([memory])
                self._complete_intents_after_generation(
                    lambda: self._complete_delete_intents(delete_intents)
                )
                self.logger.debug(
                    "Memory deleted from hybrid storage",
                    extra={
                        "workspace_id": self.workspace_id,
                        "memory_id": mem_id,
                        "engines": ["usearch", "tantivy"]
                        if self._tantivy_engine
                        else ["usearch"],
                    },
                )

                return True

            except InconsistentStateError:
                raise  # Re-raise inconsistent state errors with full context
            except Exception as e:
                raise StorageError(f"Failed to delete memory: {e}") from e

    def delete_by_message(self, message: str) -> bool:
        return self.delete_by_memory(message)

    def delete_memories(self, memories: list[str]) -> list[str]:
        """Delete many memories under one write lock and one Tantivy commit.

        Returns:
            Contents that were found and deleted.
        """
        self._ensure_open()
        unique = list(dict.fromkeys(memories))
        if not unique:
            return []
        with self._exclusive_workspace(), self._write_lock, self._lock:
            self._ensure_open()
            self._refresh_engines()
            found: list[tuple[int, str]] = []
            for memory in unique:
                mem_id = self.get_id_by_content(memory)
                if mem_id is not None:
                    found.append((mem_id, memory))
            contents = [memory for _mem_id, memory in found]
            delete_intents = self._record_delete_intents(found)
            for mem_id, _memory in found:
                self._semantic_engine.delete(memory_id=str(mem_id))
            if found and self._tantivy_engine is not None:
                try:
                    deleted_count = self._tantivy_engine.delete_batch(
                        self.workspace_id,
                        contents,
                        verify_exists=True,
                    )
                    if deleted_count < len(contents):
                        raise InconsistentStateError(
                            "USearch deletion succeeded but Tantivy "
                            f"deleted {deleted_count}/{len(contents)}"
                        )
                except InconsistentStateError:
                    raise
                except Exception as tantivy_error:
                    raise InconsistentStateError(
                        "USearch deletion succeeded but Tantivy deletion failed: "
                        f"{tantivy_error}"
                    ) from tantivy_error
            if found:
                self._semantic_engine.commit()
                self._validate_contents_absent(contents)
                self._complete_intents_after_generation(
                    lambda: self._complete_delete_intents(delete_intents)
                )
            orphaned = [memory for memory in unique if memory not in contents]
            for memory in orphaned:
                if self._finish_orphan_tantivy_delete(memory):
                    contents.append(memory)
            return contents

    def _record_add_intents(self, memories: list[str]) -> list[ReplacementTransition]:
        """Persist add intents before either backend mutates."""
        if not memories:
            return []
        return self._semantic_engine.memory_store.begin_add_intents(
            self.workspace_id, memories
        )

    def _complete_add_intents(self, intents: list[ReplacementTransition]) -> None:
        """Mark add intents complete when SQLite and Tantivy both have the text."""
        store = self._semantic_engine.memory_store
        for intent in intents:
            if intent.kind != TransitionKind.ADD:
                continue
            if self.get_id_by_content(intent.new_content) is None:
                continue
            if self._tantivy_engine is not None:
                matches = self._tantivy_engine.find_by_exact_match(
                    self.workspace_id, intent.new_content
                )
                if intent.new_content not in matches:
                    continue
            store.complete_replacement_transition(intent.id)

    def _record_delete_intents(
        self, items: list[tuple[int, str]]
    ) -> list[ReplacementTransition]:
        """Persist delete intents before either backend mutates."""
        if not items:
            return []
        return self._semantic_engine.memory_store.begin_delete_intents(
            self.workspace_id, items
        )

    def _complete_delete_intents(self, intents: list[ReplacementTransition]) -> None:
        """Mark delete intents complete when the recorded id and FTS copy are gone."""
        store = self._semantic_engine.memory_store
        for intent in intents:
            if intent.kind != TransitionKind.DELETE:
                continue
            if self.get_id_by_content(intent.old_content) == intent.old_memory_id:
                continue
            if self._semantic_engine.contains_id(intent.old_memory_id) is not False:
                continue
            if self._tantivy_engine is not None:
                matches = self._tantivy_engine.find_by_exact_match(
                    self.workspace_id, intent.old_content
                )
                if intent.old_content in matches:
                    continue
            store.complete_replacement_transition(intent.id)

    def _finish_orphan_tantivy_delete(self, content: str) -> bool:
        """Tombstone leftover FTS after USearch already dropped the row."""
        if self._tantivy_engine is None or not content:
            return False
        deleted = self._tantivy_engine.delete(
            self.workspace_id, content, verify_exists=True
        )
        if deleted is not True:
            return False
        self._tantivy_engine.commit()
        self._complete_intents_after_generation(
            lambda: self._complete_matching_delete_intents(content)
        )
        return True

    def _finish_orphan_delete_by_id(self, memory_id: int) -> None:
        """Finish a delete whose SQLite row is already gone."""
        content: str | None = None
        for row in self._semantic_engine.memory_store.list_pending_transitions():
            if (
                row.old_memory_id == memory_id
                and row.kind in {TransitionKind.DELETE, TransitionKind.REPLACE}
                and row.old_content
            ):
                content = row.old_content
                break
        self._semantic_engine.delete(memory_id=str(memory_id))
        self._semantic_engine.commit()
        if content is None:
            return
        if self._tantivy_engine is not None:
            _ = self._tantivy_engine.delete(
                self.workspace_id, content, verify_exists=False
            )
            self._tantivy_engine.commit()
        self._complete_intents_after_generation(
            lambda: self._complete_matching_delete_intents(content)
        )

    def _complete_matching_delete_intents(self, content: str) -> None:
        """Mark pending delete rows complete when FTS no longer has the text."""
        store = self._semantic_engine.memory_store
        for row in store.list_pending_transitions():
            if row.workspace_id != self.workspace_id:
                continue
            if row.kind != TransitionKind.DELETE:
                continue
            if row.old_content != content:
                continue
            if (
                self._tantivy_engine is not None
                and content
                in self._tantivy_engine.find_by_exact_match(self.workspace_id, content)
            ):
                continue
            store.complete_replacement_transition(row.id)

    def get_id_by_content(self, content: str) -> int | None:
        """Get SQLite ID for exact memory content match.

        Uses get_id_by_content() when available, with fallback to a
        legacy ID lookup API for compatibility.
        """
        return self._semantic_engine.get_id_by_content(self.workspace_id, content)

    def _has_exact_match(self, content: str) -> bool:
        """Check whether the exact memory already exists in storage.

        Uses the unique SQLite (workspace_id, content) index. Tantivy is not
        consulted for identity because stemming cannot do exact match.
        """
        return has_exact_match(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            workspace_id=self.workspace_id,
            content=content,
            logger=self.logger,
        )

    def _ensure_open(self) -> None:
        """Reject writes and searches after close() has started."""
        if self._closed or self._closing:
            raise StorageError("MemoryManager is closed")

    def _dispose_partial_init(self) -> None:
        """Close engines created before a constructor failure."""
        if "_tantivy_engine" in vars(self) and self._tantivy_engine is not None:
            with suppress(Exception):
                self._tantivy_engine.close()
        if "_semantic_engine" in vars(self):
            with suppress(Exception):
                self._semantic_engine.close()

    def close(self) -> None:
        """Close all resources and persist data to disk (thread-safe).

        This method ensures all data is properly saved before shutdown:
        1. Commits and closes Tantivy full-text engine (if enabled)
        2. Commits and closes USearch semantic engine

        Thread-safe: Uses RLock to ensure consistent state during shutdown.
        Re-entrant and concurrent close() calls are no-ops after the first.

        Should be called during graceful shutdown (e.g., on SIGINT/SIGTERM)
        to prevent data loss.
        """
        persist_ok = True
        with self._exclusive_workspace(), self._write_lock, self._lock:
            if self._closed or self._closing:
                return
            self._closing = True
            self.logger.info(
                "Closing MemoryManager - persisting data to disk...",
                extra={"workspace_id": self.workspace_id},
            )

            # Persist first. Close engines only after both commits succeed so a
            # failed close can retry instead of leaving a half-closed pair.
            if self._tantivy_engine is not None:
                try:
                    self._tantivy_engine.flush()
                    self.logger.info(
                        "Tantivy engine persisted",
                        extra={"workspace_id": self.workspace_id, "engine": "tantivy"},
                    )
                except Exception as e:
                    persist_ok = False
                    self.logger.error(
                        f"Error persisting Tantivy engine: {e}",
                        extra={
                            "workspace_id": self.workspace_id,
                            "engine": "tantivy",
                            "error": str(e),
                        },
                    )

            try:
                self._semantic_engine.commit()
                self.logger.info(
                    "USearch engine persisted",
                    extra={"workspace_id": self.workspace_id, "engine": "usearch"},
                )
            except Exception as e:
                persist_ok = False
                self.logger.error(
                    f"Error persisting USearch engine: {e}",
                    extra={
                        "workspace_id": self.workspace_id,
                        "engine": "usearch",
                        "error": str(e),
                    },
                )

            if persist_ok:
                if self._tantivy_engine is not None:
                    try:
                        self._tantivy_engine.close()
                    except Exception as e:
                        persist_ok = False
                        self.logger.error(
                            f"Error closing Tantivy engine: {e}",
                            extra={
                                "workspace_id": self.workspace_id,
                                "engine": "tantivy",
                                "error": str(e),
                            },
                        )
                try:
                    self._semantic_engine.close()
                except Exception as e:
                    persist_ok = False
                    self.logger.error(
                        f"Error closing USearch engine: {e}",
                        extra={
                            "workspace_id": self.workspace_id,
                            "engine": "usearch",
                            "error": str(e),
                        },
                    )

            self._closing = False
            if persist_ok:
                self._closed = True
                self.logger.info(
                    "MemoryManager closed - all data persisted",
                    extra={"workspace_id": self.workspace_id},
                )
                return
            self.logger.error(
                "MemoryManager persist incomplete; close not finalized",
                extra={"workspace_id": self.workspace_id},
            )
            raise StorageError("MemoryManager persist incomplete during close")
