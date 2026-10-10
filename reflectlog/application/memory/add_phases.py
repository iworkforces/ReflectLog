"""Add operation phases for the 3-phase parallel add pipeline.

This module extracts the add operation logic from MemoryManager into
separate phase classes. The 3 phases are:
1. Duplicate Detection (parallel)
2. Smart Replacement Detection (parallel)
3. Sequential Storage

Each phase is implemented as a separate class that takes inputs and
produces outputs for the next phase.
"""

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from reflectlog.infrastructure.smart_replacer import SmartReplacer
    from reflectlog.infrastructure.tantivy_engine import TantivyEngine

    from ...core.logging import IStructuredLogger
    from ..config.settings import Config

import anyio
from asyncer import asyncify, create_task_group

from reflectlog.core.enums import TransitionKind
from reflectlog.core.exceptions import (
    InconsistentStateError,
    InitializationError,
    StorageError,
)
from reflectlog.core.storage_coordination import IStorageCoordinator, LeaseMode

from ...core.types import (
    ISemanticSearchEngine,
    ReplacementTransition,
    ReplacementTransitionRequest,
)
from .match_utils import has_exact_match
from .replacement_recovery import (
    reconcile_pending_replacements,
    replacement_converged,
)


@dataclass
class ReplacementInfo:
    """Information about a memory replacement that occurred.

    Attributes:
        old_memory: The memory that was replaced (deleted).
        new_memory: The new memory that replaced it.
        confidence: LLM confidence score for the replacement decision (0.0-1.0).
        reason: LLM's explanation for why replacement was appropriate.
        similarity_score: Embedding similarity between old and new memories.
    """

    old_memory: str
    new_memory: str
    confidence: float
    reason: str
    similarity_score: float = 0.0


@dataclass
class AddResult:
    """Result of adding memories to memory storage.

    Provides detailed information about what happened during the add operation,
    including which memories were stored, skipped (duplicates), and replaced.

    Attributes:
        stored_count: Number of memories successfully stored.
        skipped_count: Number of memories skipped (duplicates).
        replaced_count: Number of existing memories that were replaced.
        replacements: Detailed info about each replacement that occurred.
    """

    stored_count: int = 0
    skipped_count: int = 0
    replaced_count: int = 0
    replacements: list[ReplacementInfo] = field(default_factory=lambda: [])


class SmartReplacerProvider(Protocol):
    @property
    def smart_replacer(self) -> SmartReplacer | None: ...


@dataclass
class Phase1Result:
    """Result of Phase 1: Duplicate Detection.

    Attributes:
        unique_memories: Memories that are unique within the batch.
        storage_duplicates: Memories that already exist in storage.
        batch_duplicates_count: Count of duplicates within the batch.
        duration: Time taken for phase 1 execution.
    """

    unique_memories: list[str]
    storage_duplicates: list[str]
    batch_duplicates_count: int
    duration: float


@dataclass
class Phase2Result:
    """Result of Phase 2: Smart Replacement Detection.

    Attributes:
        replacement_map: Mapping from memory to list of ReplacementInfo objects.
        total_replacements: Total number of replacements detected.
        duration: Time taken for phase 2 execution.
    """

    replacement_map: dict[str, list[ReplacementInfo]]
    total_replacements: int
    duration: float


@dataclass
class Phase3Result:
    """Result of Phase 3: Sequential Storage.

    Attributes:
        stored_count: Number of memories stored.
        replaced_count: Number of memories replaced.
        replacements: List of ReplacementInfo objects.
        duration: Time taken for phase 3 execution.
    """

    stored_count: int
    replaced_count: int
    replacements: list[ReplacementInfo]
    duration: float


class DuplicateDetectionPhase:
    """Phase 1: Parallel duplicate detection.

    This phase:
    1. Deduplicates within the batch itself
    2. Checks against existing storage in parallel
    3. Returns unique memories and storage duplicates
    """

    def __init__(
        self,
        semantic_engine: ISemanticSearchEngine,
        tantivy_engine: TantivyEngine | None,
        config: Config,
        logger: IStructuredLogger | None,
    ) -> None:
        """Initialize duplicate detection phase.

        Args:
            semantic_engine: USearchEngine for semantic search and direct database lookup.
            tantivy_engine: TantivyEngine for fast exact phrase matching.
            config: Application configuration.
            logger: Structured logger instance.
        """
        if logger is None:
            raise ValueError("logger is required")

        self._semantic_engine = semantic_engine
        self._tantivy_engine = tantivy_engine
        self.config = config
        self.logger: IStructuredLogger = logger
        self._workspace_id = config.workspace_id

    async def execute(self, memories: list[str]) -> Phase1Result:
        """Execute Phase 1: Parallel duplicate detection.

        Args:
            memories: List of memories to check for duplicates.

        Returns:
            Phase1Result with unique memories and duplicate information.
        """
        phase_start = time.perf_counter()

        # Step 1: Deduplicate within the batch itself (preserve order, keep first)
        seen: dict[str, int] = {}
        unique_memories: list[str] = []
        batch_duplicate_indices: list[int] = []

        for idx, memory in enumerate(memories):
            if memory in seen:
                batch_duplicate_indices.append(idx)
            else:
                seen[memory] = idx
                unique_memories.append(memory)

        batch_duplicates_count = len(batch_duplicate_indices)
        if batch_duplicates_count > 0:
            self.logger.info(
                f"Phase 1: Found {batch_duplicates_count} duplicates within batch",
                extra={
                    "batch_duplicates": batch_duplicates_count,
                    "unique_count": len(unique_memories),
                },
            )

        # Step 2: Batch duplicate detection against existing storage
        duplicate_flags: dict[str, bool] = {}
        store = self._semantic_engine.memory_store
        present = await asyncify(store.exists_many)(self._workspace_id, unique_memories)
        for memory in unique_memories:
            duplicate_flags[memory] = memory in present

        # Separate duplicates from non-duplicates
        storage_duplicates: list[str] = []
        non_duplicate_memories: list[str] = []

        for memory in unique_memories:
            if duplicate_flags.get(memory, False):
                storage_duplicates.append(memory)
            else:
                non_duplicate_memories.append(memory)

        duration = time.perf_counter() - phase_start
        self.logger.info(
            f"Phase 1 complete: {len(non_duplicate_memories)} unique, "
            f"{len(storage_duplicates)} storage duplicates ({duration:.3f}s)",
            extra={
                "phase": 1,
                "duration_ms": int(duration * 1000),
                "unique_count": len(non_duplicate_memories),
                "storage_duplicates": len(storage_duplicates),
                "batch_duplicates": batch_duplicates_count,
            },
        )

        return Phase1Result(
            unique_memories=non_duplicate_memories,
            storage_duplicates=storage_duplicates,
            batch_duplicates_count=batch_duplicates_count,
            duration=duration,
        )

    def _has_exact_match(self, content: str) -> bool:
        """Check whether the exact memory already exists in storage.

        Uses the unique SQLite (workspace_id, content) index. Tantivy is not
        consulted for identity because stemming cannot do exact match.
        """
        return has_exact_match(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            workspace_id=self._workspace_id,
            content=content,
            logger=self.logger,
        )


class SmartReplacementPhase:
    """Phase 2: Parallel smart replacement detection.

    This phase:
    1. Finds similar existing memories via semantic search
    2. Filters by similarity threshold
    3. Checks candidates in parallel with LLM
    4. Returns replacement map
    """

    def __init__(
        self,
        semantic_engine: ISemanticSearchEngine,
        config: Config,
        logger: IStructuredLogger | None,
        memory_manager: SmartReplacerProvider | None,
    ) -> None:
        """Initialize smart replacement phase.

        Args:
            semantic_engine: USearchEngine for semantic search.
            config: Application configuration.
            logger: Structured logger instance.
            memory_manager: MemoryManager instance for lazy SmartReplacer fetching (optional).
        """
        if logger is None:
            raise ValueError("logger is required")

        self._semantic_engine = semantic_engine
        self.config = config
        self.logger: IStructuredLogger = logger
        self._workspace_id = config.workspace_id
        self._memory_manager = memory_manager

    def _get_smart_replacer(self) -> SmartReplacer | None:
        """Get SmartReplacer (lazy loading via memory_manager).

        Returns:
            SmartReplacer instance if configured, None otherwise.
        """
        if self._memory_manager is None:
            return None
        return self._memory_manager.smart_replacer

    @property
    def smart_replace_enabled(self) -> bool:
        """Check if smart replacement is configured and available.

        Returns:
            True if SmartReplacer is configured, False otherwise.
        """
        return self._get_smart_replacer() is not None

    async def execute(self, memories: list[str]) -> Phase2Result:
        """Execute Phase 2: Parallel smart replacement detection.

        Args:
            memories: List of non-duplicate memories to check for replacements.

        Returns:
            Phase2Result with replacement information.
        """
        phase_start = time.perf_counter()

        # Map: memory -> list of replacement infos
        replacement_map: dict[str, list[ReplacementInfo]] = {}

        # Get SmartReplacer (lazy loaded via memory_manager if available)
        smart_replacer = self._get_smart_replacer()

        if smart_replacer is None or not memories:
            return Phase2Result(
                replacement_map=replacement_map,
                total_replacements=0,
                duration=time.perf_counter() - phase_start,
            )

        # Capture in local variable for type narrowing in nested function
        smart_replacer = smart_replacer

        # Limit concurrent replacement checks to avoid overload
        semaphore = anyio.Semaphore(self.config.add_max_concurrency)

        # asyncer task group required: soonify captures SoonValue results from parallel replacement checks
        replacement_results: list[tuple[str, list[ReplacementInfo]]] = []
        async with create_task_group() as tg:

            async def check_replacement_for_memory(memory: str) -> None:
                """Check replacement for a single memory."""
                async with semaphore:
                    infos = await self._check_for_replacement(memory, smart_replacer)
                replacement_results.append((memory, infos))

            for memory in memories:
                _ = tg.soonify(check_replacement_for_memory)(memory)

        # Collect results
        for memory, infos in replacement_results:
            replacement_map[memory] = infos

        duration = time.perf_counter() - phase_start
        total_replacements = sum(len(infos) for infos in replacement_map.values())

        self.logger.info(
            f"Phase 2 complete: {total_replacements} replacements detected ({duration:.3f}s)",
            extra={
                "phase": 2,
                "duration_ms": int(duration * 1000),
                "memories_checked": len(memories),
                "total_replacements": total_replacements,
            },
        )

        return Phase2Result(
            replacement_map=replacement_map,
            total_replacements=total_replacements,
            duration=duration,
        )

    async def _search_candidates(self, new_memory: str) -> list[tuple[str, float, str]]:
        """Step 1: Find top N most similar memories via semantic search.

        Args:
            new_memory: The new memory being added.

        Returns:
            List of (content, score, metadata) tuples from semantic search.
            Empty list if no similar memories found.
        """
        candidate_limit = self.config.smart_replace_candidate_limit
        similar_results = await asyncify(self._semantic_engine.search)(
            query=new_memory,
            workspace_id=self._workspace_id,
            limit=candidate_limit,
        )

        if not similar_results:
            self.logger.debug(
                "No similar memories found for replacement check",
                extra={
                    "workspace_id": self._workspace_id,
                    "new_memory_length": len(new_memory),
                },
            )
        return similar_results

    def _filter_by_similarity(
        self,
        similar_results: list[tuple[str, float, str]],
        new_memory: str,
    ) -> list[tuple[str, float]]:
        """Step 2: Filter candidates by similarity threshold (pre-filter).

        Args:
            similar_results: Raw search results from semantic engine.
            new_memory: The new memory (excluded from candidates).

        Returns:
            Filtered list of (content, score) tuples above the threshold.
        """
        min_similarity = self.config.smart_replace_min_similarity
        filtered_candidates = [
            (mem, score)
            for mem, score, _ in similar_results
            if score >= min_similarity and mem != new_memory
        ]

        if not filtered_candidates:
            self.logger.debug(
                f"All candidates below similarity threshold ({min_similarity})",
                extra={
                    "workspace_id": self._workspace_id,
                    "candidate_count": len(similar_results),
                    "min_similarity": min_similarity,
                },
            )
        else:
            self.logger.debug(
                f"Found {len(filtered_candidates)} candidates above similarity threshold",
                extra={
                    "workspace_id": self._workspace_id,
                    "candidate_count": len(filtered_candidates),
                    "min_similarity": min_similarity,
                    "top_similarity": filtered_candidates[0][1]
                    if filtered_candidates
                    else 0,
                },
            )

        return filtered_candidates

    async def _check_candidates_with_llm(
        self,
        filtered_candidates: list[tuple[str, float]],
        new_memory: str,
        smart_replacer: SmartReplacer,
    ) -> list[ReplacementInfo]:
        """Step 3: Check candidates in parallel with LLM.

        Args:
            filtered_candidates: Pre-filtered (content, score) tuples.
            new_memory: The new memory being added.
            smart_replacer: SmartReplacer instance for LLM checks.

        Returns:
            List of ReplacementInfo for memories that should be replaced.
        """
        semaphore = anyio.Semaphore(self.config.rerank_max_concurrency)

        async def check_single_candidate(
            existing_memory: str, similarity_score: float
        ) -> ReplacementInfo | None:
            """Check a single candidate for replacement (with semaphore)."""
            async with semaphore:
                try:
                    (
                        should_replace,
                        confidence,
                        reason,
                    ) = await smart_replacer.check_replacement(
                        new_memory=new_memory,
                        existing_memory=existing_memory,
                    )

                    if should_replace:
                        self.logger.info(
                            f"Smart replacement triggered (confidence={confidence:.2f})",
                            extra={
                                "workspace_id": self._workspace_id,
                                "should_replace": True,
                                "confidence": confidence,
                                "similarity_score": similarity_score,
                                "reason_length": len(reason),
                                "old_memory_length": len(existing_memory),
                                "new_memory_length": len(new_memory),
                            },
                        )
                        return ReplacementInfo(
                            old_memory=existing_memory,
                            new_memory=new_memory,
                            confidence=confidence,
                            reason=reason,
                            similarity_score=similarity_score,
                        )
                    else:
                        self.logger.debug(
                            f"No replacement needed (confidence={confidence:.2f})",
                            extra={
                                "workspace_id": self._workspace_id,
                                "should_replace": False,
                                "confidence": confidence,
                                "reason_length": len(reason),
                            },
                        )
                        return None
                except Exception as candidate_error:
                    # Graceful degradation: log warning and skip this candidate
                    self.logger.warning(
                        "LLM check failed for candidate",
                        extra={
                            "workspace_id": self._workspace_id,
                            "exception_type": type(candidate_error).__name__,
                            "existing_memory_length": len(existing_memory),
                        },
                    )
                    return None

        # Run all candidate checks in parallel
        results: list[ReplacementInfo | None] = []

        async def collect_result(existing_memory: str, similarity_score: float) -> None:
            """Run check and collect result."""
            result = await check_single_candidate(existing_memory, similarity_score)
            results.append(result)

        # anyio task group sufficient: fire-and-forget pattern for candidate checks
        async with anyio.create_task_group() as tg:
            for existing_memory, similarity_score in filtered_candidates:
                _ = tg.start_soon(collect_result, existing_memory, similarity_score)

        # Filter out None results (failed checks or no replacement needed)
        return [r for r in results if r is not None]

    async def _check_for_replacement(
        self, new_memory: str, smart_replacer: SmartReplacer
    ) -> list[ReplacementInfo]:
        """Check if new memory should replace existing memories.

        Orchestrates a 3-step pipeline: search candidates, filter by
        similarity threshold, then check with LLM.

        Args:
            new_memory: The new memory being added.
            smart_replacer: SmartReplacer instance.

        Returns:
            List of ReplacementInfo objects for memories that should be replaced.
            Empty list if no replacements needed.
        """
        try:
            similar_results = await self._search_candidates(new_memory)
            if not similar_results:
                return []

            filtered_candidates = self._filter_by_similarity(
                similar_results, new_memory
            )
            if not filtered_candidates:
                return []

            return await self._check_candidates_with_llm(
                filtered_candidates, new_memory, smart_replacer
            )

        except Exception as e:
            # Graceful degradation: log warning and proceed without replacement
            self.logger.warning(
                "Smart replacement check failed",
                extra={
                    "workspace_id": self._workspace_id,
                    "exception_type": type(e).__name__,
                    "new_memory_length": len(new_memory),
                },
            )
            return []


class StoragePhase:
    """Phase 3: Sequential storage with replacement handling.

    This phase:
    1. Records a durable SQLite replacement transition (archive + intent)
    2. Deletes old memories from the active indexes
    3. Adds new memories to storage
    4. Commits each engine independently, then marks transitions complete

    SQLite archive + transition is one local transaction. USearch and
    Tantivy commits are not claimed as one atomic unit; unfinished
    transitions are reconciled on startup and at the start of the next persist.
    """

    def __init__(
        self,
        semantic_engine: ISemanticSearchEngine,
        tantivy_engine: TantivyEngine | None,
        config: Config,
        logger: IStructuredLogger | None,
        write_lock: AbstractContextManager[object] | None = None,
        lock: AbstractContextManager[object] | None = None,
        ensure_open: Callable[[], None] | None = None,
        coordinator: IStorageCoordinator | None = None,
        orchestration_hook: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize storage phase.

        Args:
            semantic_engine: USearchEngine for semantic storage.
            tantivy_engine: TantivyEngine for full-text storage.
            config: Application configuration.
            logger: Structured logger instance.
            write_lock: Exclusive write lock shared with MemoryManager.
            lock: Inner read lock; acquired after ``write_lock`` during reconcile.
            ensure_open: Optional closed-manager check run under the write lock.
        """
        if logger is None:
            raise ValueError("logger is required")

        self._semantic_engine = semantic_engine
        self._tantivy_engine = tantivy_engine
        self.config = config
        self.logger: IStructuredLogger = logger
        self._workspace_id = config.workspace_id
        self._write_lock = write_lock
        self._lock = lock
        self._ensure_open = ensure_open
        self._coordinator = coordinator
        self._orchestration_hook = orchestration_hook

    async def execute(
        self,
        memories: list[str],
        replacement_map: dict[str, list[ReplacementInfo]],
        dry_run: bool = False,
    ) -> Phase3Result:
        """Execute Phase 3: Sequential storage.

        Args:
            memories: List of non-duplicate memories to store.
            replacement_map: Mapping from memory to replacement info.
            dry_run: If True, only check without making changes.

        Returns:
            Phase3Result with storage information.
        """
        phase_start = time.perf_counter()
        if dry_run:
            return self._dry_run_result(memories, replacement_map, phase_start)

        await asyncify(self._reconcile_pending)()
        vectors = (
            self._validate_persist_vectors(
                memories,
                await self._semantic_engine.embedder.aembed_documents(memories),
            )
            if memories
            else None
        )
        stored_count, replacements = await asyncify(self._persist_with_vectors)(
            memories, replacement_map, vectors
        )

        duration = time.perf_counter() - phase_start
        self.logger.info(
            f"Phase 3 complete: {stored_count} stored ({duration:.3f}s)",
            extra={
                "phase": 3,
                "duration_ms": int(duration * 1000),
                "stored_count": stored_count,
                "replaced_count": len(replacements),
            },
        )
        return Phase3Result(
            stored_count=stored_count,
            replaced_count=len(replacements),
            replacements=replacements,
            duration=duration,
        )

    def _dry_run_result(
        self,
        memories: list[str],
        replacement_map: dict[str, list[ReplacementInfo]],
        phase_start: float,
    ) -> Phase3Result:
        """Count planned replacements without touching storage."""
        _planned, replacements = self._plan_replacements(
            memories, replacement_map, dry_run=True
        )
        return Phase3Result(
            stored_count=len(memories),
            replaced_count=len(replacements),
            replacements=replacements,
            duration=time.perf_counter() - phase_start,
        )

    def _persist_replacements(
        self,
        memories: list[str],
        replacement_map: dict[str, list[ReplacementInfo]],
    ) -> tuple[int, list[ReplacementInfo]]:
        """Embed outside the write lock, then record, delete, add, and commit."""
        vectors = self._embed_for_persist(memories)
        return self._persist_with_vectors(memories, replacement_map, vectors)

    def _persist_with_vectors(
        self,
        memories: list[str],
        replacement_map: dict[str, list[ReplacementInfo]],
        vectors: list[list[float]] | None,
    ) -> tuple[int, list[ReplacementInfo]]:
        if self._coordinator is not None:
            with self._coordinator.acquire(self._workspace_id, LeaseMode.EXCLUSIVE):
                return self._persist_under_locks(memories, replacement_map, vectors)
        return self._persist_under_locks(memories, replacement_map, vectors)

    def _persist_under_locks(
        self,
        memories: list[str],
        replacement_map: dict[str, list[ReplacementInfo]],
        vectors: list[list[float]] | None,
    ) -> tuple[int, list[ReplacementInfo]]:
        if self._write_lock is None:
            self._require_open()
            return self._persist_replacements_unlocked(
                memories, replacement_map, vectors
            )
        if self._lock is not None:
            with self._write_lock, self._lock:
                self._require_open()
                return self._persist_replacements_unlocked(
                    memories, replacement_map, vectors
                )
        with self._write_lock:
            self._require_open()
            return self._persist_replacements_unlocked(
                memories, replacement_map, vectors
            )

    def _require_open(self) -> None:
        """Reject persist after MemoryManager.close() has started."""
        if self._ensure_open is not None:
            self._ensure_open()

    def _revalidate_persist_inputs(
        self,
        memories: list[str],
        vectors: list[list[float]] | None,
        replacement_map: dict[str, list[ReplacementInfo]],
    ) -> tuple[list[str], list[list[float]] | None]:
        """Drop identities that appeared after phase-2 decisions."""
        kept: list[str] = []
        kept_vectors: list[list[float]] = []
        for index, memory in enumerate(memories):
            if (
                memory not in replacement_map
                and self.config.deduplicate_memories
                and self._has_exact_match(memory)
            ):
                continue
            kept.append(memory)
            if vectors is not None:
                kept_vectors.append(vectors[index])
        if vectors is None:
            return kept, None
        return kept, kept_vectors

    def _emit_orchestration_hook(self, step: str) -> None:
        hook = self._orchestration_hook
        if hook is not None:
            hook(step)

    def _publish_then_complete(
        self,
        transitions: list[ReplacementTransition],
        add_intents: list[ReplacementTransition],
        *,
        replacement_olds: list[str],
        should_publish: bool,
    ) -> None:
        coordinator = self._coordinator
        if coordinator is not None and should_publish:
            current = coordinator.read_generation(self._workspace_id)
            self._emit_orchestration_hook("before_generation")
            coordinator.publish_generation(self._workspace_id, current + 1)
            self._emit_orchestration_hook("after_generation")
        self._emit_orchestration_hook("before_intent")
        self._complete_if_converged(transitions)
        self._complete_add_intents_for_contents(replacement_olds)
        self._complete_add_intents(add_intents)
        self._emit_orchestration_hook("after_intent")

    def _embed_for_persist(self, memories: list[str]) -> list[list[float]] | None:
        """Compute document embeddings before taking the write lock."""
        if not memories:
            return None
        computed = self._semantic_engine.embedder.embed_documents(memories)
        return self._validate_persist_vectors(memories, computed)

    @staticmethod
    def _validate_persist_vectors(
        memories: list[str], computed: list[list[float]]
    ) -> list[list[float]]:
        if len(computed) != len(memories):
            raise StorageError("Embedding batch size mismatch for persist")
        typed: list[list[float]] = []
        for item in computed:
            if not item:
                raise StorageError("Embedding produced an empty or invalid vector")
            typed.append(item)
        return typed

    def _persist_replacements_unlocked(
        self,
        memories: list[str],
        replacement_map: dict[str, list[ReplacementInfo]],
        vectors: list[list[float]] | None = None,
    ) -> tuple[int, list[ReplacementInfo]]:
        """Mutate indexes while the caller already holds ``_write_lock``."""
        if self._coordinator is not None:
            memories, vectors = self._revalidate_persist_inputs(
                memories, vectors, replacement_map
            )
        planned, replacements = self._plan_replacements(memories, replacement_map)
        replacement_olds = {info.old_memory for info, _memory, _old_id in planned}
        persist_memories, persist_vectors = self._exclude_replacement_olds(
            memories, vectors, replacement_olds
        )
        transitions = self._record_planned_transitions(planned)
        add_intents = self._record_add_intents(persist_memories)
        stored_count = self._store_and_commit(
            persist_memories, dry_run=False, locked=True, vectors=persist_vectors
        )
        self._delete_recorded_olds(transitions)
        if transitions:
            if self._tantivy_engine is not None:
                self._tantivy_engine.commit()
            self._semantic_engine.commit()
        should_publish = bool(transitions) or stored_count > 0
        self._publish_then_complete(
            transitions,
            add_intents,
            replacement_olds=list(replacement_olds),
            should_publish=should_publish,
        )
        return stored_count, replacements

    def _plan_replacements(
        self,
        memories: list[str],
        replacement_map: dict[str, list[ReplacementInfo]],
        *,
        dry_run: bool = False,
    ) -> tuple[
        list[tuple[ReplacementInfo, str, int]],
        list[ReplacementInfo],
    ]:
        """Keep the highest-confidence successor for each old id."""
        candidates = self._replacement_candidates(memories, replacement_map)
        winners = _highest_confidence_by_old_id(candidates)
        planned: list[tuple[ReplacementInfo, str, int]] = []
        for info, memory, old_id in candidates:
            winner = winners[old_id]
            if winner is not info:
                self.logger.info(
                    "Keeping highest-confidence successor for old memory",
                    extra={
                        "old_memory_id": old_id,
                        "kept_confidence": winner.confidence,
                        "skipped_confidence": info.confidence,
                        "dry_run": dry_run,
                    },
                )
                continue
            planned.append((info, memory, old_id))
        planned = self._drop_conflicting_successors(planned, dry_run=dry_run)
        replacements: list[ReplacementInfo] = []
        for info, memory, _old_id in planned:
            self.logger.info(
                "Replacing old memory with new one",
                extra={
                    "action": "replace",
                    "old_length": len(info.old_memory),
                    "new_length": len(memory),
                    "confidence": info.confidence,
                    "similarity": info.similarity_score,
                    "dry_run": dry_run,
                },
            )
            replacements.append(info)
        return planned, replacements

    def _replacement_candidates(
        self,
        memories: list[str],
        replacement_map: dict[str, list[ReplacementInfo]],
    ) -> list[tuple[ReplacementInfo, str, int]]:
        """Resolve live old ids for every intended replacement."""
        candidates: list[tuple[ReplacementInfo, str, int]] = []
        for idx, memory in enumerate(memories):
            for info in replacement_map.get(memory, []):
                old_id = self._semantic_engine.get_id_by_content(
                    self._workspace_id, info.old_memory
                )
                if old_id is None:
                    self.logger.warning(
                        "Old memory not found for replacement delete",
                        extra={"memory_index": idx + 1, "action": "delete_missing"},
                    )
                    continue
                candidates.append((info, memory, old_id))
        return candidates

    def _drop_conflicting_successors(
        self,
        planned: list[tuple[ReplacementInfo, str, int]],
        *,
        dry_run: bool,
    ) -> list[tuple[ReplacementInfo, str, int]]:
        """Omit successors that would collide with an exclusive leftover row."""
        store = self._semantic_engine.memory_store
        kept: list[tuple[ReplacementInfo, str, int]] = []
        for info, memory, old_id in planned:
            existing = store.get_transition_for_old_memory(self._workspace_id, old_id)
            if (
                isinstance(existing, ReplacementTransition)
                and existing.new_content != memory
            ):
                self.logger.info(
                    "Skipping successor that conflicts with an existing transition",
                    extra={
                        "old_memory_id": old_id,
                        "existing_new_length": len(existing.new_content),
                        "skipped_new_length": len(memory),
                        "dry_run": dry_run,
                    },
                )
                continue
            kept.append((info, memory, old_id))
        return kept

    def _record_planned_transitions(
        self, planned: list[tuple[ReplacementInfo, str, int]]
    ) -> list[ReplacementTransition]:
        """Persist every archive + pending row before the first delete."""
        if not planned:
            return []
        requests = [
            ReplacementTransitionRequest(
                old_memory_id=old_id,
                workspace_id=self._workspace_id,
                old_content=info.old_memory,
                new_content=new_memory,
                reason=info.reason,
                confidence=info.confidence,
            )
            for info, new_memory, old_id in planned
        ]
        store = self._semantic_engine.memory_store
        try:
            return store.begin_replacement_transitions(requests)
        except Exception as e:
            raise StorageError(f"Failed to record replacement transition: {e}") from e

    def _delete_recorded_olds(self, transitions: list[ReplacementTransition]) -> None:
        """Delete each recorded old memory after transitions are durable."""
        contents: list[str] = []
        for transition in transitions:
            try:
                self._semantic_engine.delete(memory_id=str(transition.old_memory_id))
                contents.append(transition.old_content)
            except InitializationError:
                raise
            except Exception as delete_error:
                self.logger.warning(
                    f"Failed to delete old memory: {delete_error}",
                    extra={
                        "action": "delete_failed",
                        "transition_id": transition.id,
                        "error": str(delete_error),
                    },
                )
                raise StorageError(
                    f"Failed to delete old memory after recording transition "
                    f"{transition.id}: {delete_error}"
                ) from delete_error
        if contents and self._tantivy_engine is not None:
            try:
                deleted_count = self._tantivy_engine.delete_batch(
                    self._workspace_id,
                    contents,
                    verify_exists=True,
                )
                if deleted_count < len(contents):
                    raise InconsistentStateError(
                        "USearch replacement delete succeeded but Tantivy "
                        f"deleted {deleted_count}/{len(contents)}"
                    )
            except InconsistentStateError:
                raise
            except Exception as tantivy_error:
                raise InconsistentStateError(
                    "USearch replacement delete succeeded but Tantivy deletion "
                    f"failed: {tantivy_error}"
                ) from tantivy_error

    def _store_and_commit(
        self,
        memories_to_add: list[str],
        dry_run: bool,
        *,
        locked: bool = False,
        vectors: list[list[float]] | None = None,
    ) -> int:
        """Batch-add new memories and commit changes to engines."""
        if dry_run or not memories_to_add:
            return 0

        stored_memories = self._add_memories_batch(
            memories_to_add, locked=locked, vectors=vectors
        )
        stored_count = len(stored_memories)
        if stored_count != len(memories_to_add):
            self.logger.warning(
                "Batch add stored fewer memories than expected",
                extra={
                    "expected_count": len(memories_to_add),
                    "stored_count": stored_count,
                },
            )

        if self._tantivy_engine is not None:
            self._tantivy_engine.commit()
        self._semantic_engine.commit()
        return stored_count

    def _add_memories_batch(
        self,
        memories: list[str],
        *,
        locked: bool = False,
        vectors: list[list[float]] | None = None,
    ) -> list[str]:
        """Add multiple memories to both semantic and full-text engines."""
        if not memories:
            return []

        try:
            if self._write_lock is not None and not locked:
                with self._write_lock:
                    return self._add_memories_unlocked(memories, vectors)
            return self._add_memories_unlocked(memories, vectors)
        except InitializationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to add memory batch: {e}") from e

    def _add_memories_unlocked(
        self,
        memories: list[str],
        vectors: list[list[float]] | None = None,
    ) -> list[str]:
        """Add memories assuming the caller already serialized writes."""
        if vectors is None:
            inserted_memories = self._semantic_engine.add_batch(
                workspace_id=self._workspace_id,
                contents=memories,
                infer=self.config.enable_llm_infer,
            )
        else:
            inserted_memories = self._semantic_engine.add_batch(
                workspace_id=self._workspace_id,
                contents=memories,
                infer=self.config.enable_llm_infer,
                vectors=vectors,
            )
        if self._tantivy_engine is not None:
            self._tantivy_engine.add_batch(self._workspace_id, inserted_memories)
        self.logger.debug(
            "Batch added memories to hybrid storage",
            extra={
                "workspace_id": self._workspace_id,
                "memory_count": len(inserted_memories),
                "engines": ["semantic", "tantivy"],
            },
        )
        return inserted_memories

    def _has_exact_match(self, content: str) -> bool:
        """Check whether the exact memory already exists in storage.

        Uses Tantivy for fast exact phrase matching when hybrid search is enabled,
        falling back to direct database lookup otherwise. Both paths are O(log n)
        avoiding the ~100-500ms embedding API call overhead.
        """
        return has_exact_match(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            workspace_id=self._workspace_id,
            content=content,
            logger=self.logger,
        )

    def _record_add_intents(self, memories: list[str]) -> list[ReplacementTransition]:
        """Persist add intents before mutating either index."""
        if not memories:
            return []
        return self._semantic_engine.memory_store.begin_add_intents(
            self._workspace_id, memories
        )

    def _complete_add_intents(self, intents: list[ReplacementTransition]) -> None:
        """Mark add intents complete when both backends have the content."""
        if not intents:
            return
        live_contents = (
            set(self._tantivy_engine.get_all(self._workspace_id))
            if self._tantivy_engine is not None
            else None
        )
        store = self._semantic_engine.memory_store
        for intent in intents:
            new_id = self._semantic_engine.get_id_by_content(
                self._workspace_id, intent.new_content
            )
            if new_id is None:
                continue
            if live_contents is not None and intent.new_content not in live_contents:
                continue
            store.complete_replacement_transition(intent.id)

    def _complete_add_intents_for_contents(self, contents: list[str]) -> None:
        """Complete leftover ADD rows for texts this persist just replaced."""
        wanted = {content for content in contents if content}
        if not wanted:
            return
        store = self._semantic_engine.memory_store
        for row in store.list_pending_transitions():
            if row.workspace_id != self._workspace_id:
                continue
            if row.kind != TransitionKind.ADD:
                continue
            if row.new_content not in wanted:
                continue
            store.complete_replacement_transition(row.id)

    def _exclude_replacement_olds(
        self,
        memories: list[str],
        vectors: list[list[float]] | None,
        replacement_olds: set[str],
    ) -> tuple[list[str], list[list[float]] | None]:
        """Drop replacement olds so they are not re-journaled as ADD intents."""
        if not replacement_olds:
            return memories, vectors
        persist_memories: list[str] = []
        persist_vectors: list[list[float]] | None = [] if vectors is not None else None
        for idx, memory in enumerate(memories):
            if memory in replacement_olds:
                continue
            persist_memories.append(memory)
            if persist_vectors is not None and vectors is not None:
                persist_vectors.append(vectors[idx])
        return persist_memories, persist_vectors

    def _complete_if_converged(self, transitions: list[ReplacementTransition]) -> None:
        """Mark a transition complete only when both indexes agree."""
        store = self._semantic_engine.memory_store
        for transition in transitions:
            if not replacement_converged(
                transition,
                semantic_engine=self._semantic_engine,
                tantivy_engine=self._tantivy_engine,
            ):
                self.logger.warning(
                    "Leaving replacement transition pending; indexes disagree",
                    extra={"transition_id": transition.id},
                )
                continue
            store.complete_replacement_transition(transition.id)

    def _reconcile_pending(self) -> None:
        """Apply leftover transitions from an earlier failed persist."""
        if self._write_lock is None:
            return
        _ = reconcile_pending_replacements(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            write_lock=self._write_lock,
            lock=self._lock,
            logger=self.logger,
            coordinator=self._coordinator,
            workspace_id=self._workspace_id,
            orchestration_hook=self._orchestration_hook,
        )


def _persist_list_for_add(
    *,
    unique_memories: list[str],
    storage_duplicates: list[str],
    replacement_map: dict[str, list[ReplacementInfo]],
    dry_run: bool,
) -> tuple[list[str], int]:
    """Build the persist list and how many storage dups are already skipped.

    Replacement olds are never persisted or journaled as ADD intents. Dry-run
    also omits every storage duplicate so the preview matches live unique-skip.
    """
    replacement_olds = {
        info.old_memory for infos in replacement_map.values() for info in infos
    }
    if dry_run:
        return list(unique_memories), len(storage_duplicates)
    persist_memories = list(unique_memories)
    skipped = 0
    for memory in storage_duplicates:
        if memory in replacement_olds:
            skipped += 1
            continue
        persist_memories.append(memory)
    return persist_memories, skipped


def _highest_confidence_by_old_id(
    candidates: list[tuple[ReplacementInfo, str, int]],
) -> dict[int, ReplacementInfo]:
    """Pick one successor per old id; ties keep the first candidate."""
    winners: dict[int, ReplacementInfo] = {}
    for info, _memory, old_id in candidates:
        current = winners.get(old_id)
        if current is None or info.confidence > current.confidence:
            winners[old_id] = info
    return winners


class AddPipeline:
    """Orchestrates the 3-phase add pipeline.

    This class coordinates the execution of the three phases:
    1. DuplicateDetectionPhase
    2. SmartReplacementPhase
    3. StoragePhase

    It combines the results from each phase into the final AddResult.
    """

    def __init__(
        self,
        duplicate_detection_phase: DuplicateDetectionPhase,
        smart_replacement_phase: SmartReplacementPhase,
        storage_phase: StoragePhase,
        config: Config,
        logger: IStructuredLogger | None,
    ) -> None:
        """Initialize add pipeline.

        Args:
            duplicate_detection_phase: Phase 1 instance.
            smart_replacement_phase: Phase 2 instance.
            storage_phase: Phase 3 instance.
            config: Application configuration.
            logger: Structured logger instance.
        """
        if logger is None:
            raise ValueError("logger is required")

        self._phase1 = duplicate_detection_phase
        self._phase2 = smart_replacement_phase
        self._phase3 = storage_phase
        self.config = config
        self.logger: IStructuredLogger = logger

    async def execute(
        self, memories: list[str], dry_run: bool = False, allow_replace: bool = True
    ) -> AddResult:
        """Execute the 3-phase add pipeline.

        Args:
            memories: List of memories to add.
            dry_run: If True, only check for replacements without making changes.

        Returns:
            AddResult with detailed information about the operation.

        Raises:
            StorageError: If storage operation fails (not raised in dry_run mode).
        """
        result = AddResult()

        if not memories:
            return result

        mode_str = "DRY RUN" if dry_run else "LIVE"
        self.logger.info(
            f"Starting phased parallel memory addition [{mode_str}]: {len(memories)} memories",
            extra={
                "total_memories": len(memories),
                "smart_replace_enabled": self._phase2.smart_replace_enabled,
                "dry_run": dry_run,
                "optimization": "phased_parallel",
            },
        )

        try:
            # Phase 1: Parallel duplicate detection
            phase1_result = await self._phase1.execute(memories)

            # Phase 2: Parallel smart replacement detection
            phase2_result = (
                await self._phase2.execute(phase1_result.unique_memories)
                if allow_replace
                else Phase2Result(
                    replacement_map={}, total_replacements=0, duration=0.0
                )
            )

            persist_memories, skipped_storage = _persist_list_for_add(
                unique_memories=phase1_result.unique_memories,
                storage_duplicates=phase1_result.storage_duplicates,
                replacement_map=phase2_result.replacement_map,
                dry_run=dry_run,
            )
            result.skipped_count = (
                phase1_result.batch_duplicates_count + skipped_storage
            )

            # Phase 3: Sequential database writes
            phase3_result = await self._phase3.execute(
                persist_memories,
                phase2_result.replacement_map,
                dry_run,
            )

            result.stored_count = phase3_result.stored_count
            result.replaced_count = phase3_result.replaced_count
            result.replacements = phase3_result.replacements
            persist_skipped = len(persist_memories) - phase3_result.stored_count
            if persist_skipped > 0:
                result.skipped_count += persist_skipped

        except InconsistentStateError, InitializationError:
            raise
        except Exception as e:
            mode_str = "DRY_RUN" if dry_run else "LIVE"
            self.logger.error(
                f"Phased parallel memory addition failed [{mode_str}]: {e}",
                extra={
                    "mode": mode_str,
                    "dry_run": dry_run,
                    "total_memories": len(memories),
                    "stored_count": result.stored_count,
                    "replaced_count": result.replaced_count,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            if not dry_run:
                raise StorageError(f"Failed to add memories: {e}") from e

        self.logger.info(
            f"Phased parallel addition completed [{mode_str}]: "
            f"{result.stored_count}/{len(memories)} stored, "
            f"{result.replaced_count} replaced, {result.skipped_count} skipped",
            extra={
                "total_memories": len(memories),
                "stored_count": result.stored_count,
                "replaced_count": result.replaced_count,
                "skipped_count": result.skipped_count,
                "replacement_count": result.replaced_count,
                "dry_run": dry_run,
                "optimization": "phased_parallel",
            },
        )

        return result
