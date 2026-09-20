"""Restart-safe reconciliation of unfinished smart replacements.

SQLite archive + transition rows are one local transaction. USearch and
Tantivy commits are independent. Recovery *attempts* to converge leftover
intent; it can be skipped, and it will not mark a row complete while
SQLite or hybrid Tantivy still disagree.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import os
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from reflectlog.core.enums import TransitionKind
from reflectlog.core.storage_coordination import IStorageCoordinator, LeaseMode
from reflectlog.core.types import (
    IArchiveMemoryStore,
    ISemanticSearchEngine,
    ReplacementTransition,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from reflectlog.core.logging import IStructuredLogger
    from reflectlog.infrastructure.tantivy_engine import TantivyEngine


@runtime_checkable
class _RefreshableEngine(Protocol):
    def refresh(self) -> None: ...


def _refresh_engine(engine: object) -> None:
    if isinstance(engine, _RefreshableEngine):
        engine.refresh()


def _complete_converged_intent(
    store: IArchiveMemoryStore,
    transition_id: int,
    *,
    coordinator: IStorageCoordinator | None,
    workspace_id: str,
    hook: Callable[[str], None] | None,
    leftover_add_content: str = "",
) -> None:
    """Publish generation, then complete leftover ADDs and the durable intent."""
    if coordinator is not None and workspace_id:
        if hook is not None:
            hook("before_generation")
        generation = coordinator.read_generation(workspace_id)
        coordinator.publish_generation(workspace_id, generation + 1)
        if hook is not None:
            hook("after_generation")
    if leftover_add_content:
        _complete_pending_adds_of(
            store,
            workspace_id=workspace_id,
            content=leftover_add_content,
        )
    if hook is not None:
        hook("before_intent")
    store.complete_replacement_transition(transition_id)
    if hook is not None:
        hook("after_intent")


def reconcile_pending_replacements(
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    write_lock: AbstractContextManager[object],
    lock: AbstractContextManager[object] | None = None,
    logger: IStructuredLogger,
    coordinator: IStorageCoordinator | None = None,
    workspace_id: str = "",
    orchestration_hook: Callable[[str], None] | None = None,
) -> int:
    """Finish pending replacements using the semantic store as source of truth.

    Acquires ``write_lock`` before ``lock`` when both are provided.

    Returns:
        Number of pending transitions that were marked complete.
    """
    store = _recovery_store(semantic_engine, logger)
    if store is None:
        return 0

    pending = _pending_rows(store.list_pending_transitions())
    if not pending:
        return 0

    semantic_engine.ensure_initialized()
    if tantivy_engine is not None:
        tantivy_engine.ensure_initialized()

    precomputed = _precompute_add_vectors(pending, semantic_engine, logger)

    completed = 0
    inner_lock = lock if lock is not None else nullcontext()
    lease = (
        coordinator.acquire(workspace_id, LeaseMode.EXCLUSIVE)
        if coordinator is not None and workspace_id
        else nullcontext()
    )
    with lease, write_lock, inner_lock:
        semantic_engine.ensure_initialized()
        if tantivy_engine is not None:
            tantivy_engine.ensure_initialized()
        _refresh_engine(semantic_engine)
        if tantivy_engine is not None:
            _refresh_engine(tantivy_engine)
        snapshot = _pending_rows(store.list_pending_transitions())
        pending_ids = {row.id for row in snapshot}
        for transition in snapshot:
            if transition.id not in pending_ids:
                continue
            try:
                if apply_pending_transition(
                    transition,
                    semantic_engine=semantic_engine,
                    tantivy_engine=tantivy_engine,
                    logger=logger,
                    precomputed_vectors=precomputed,
                    coordinator=coordinator,
                    orchestration_hook=orchestration_hook,
                ):
                    completed += 1
            except Exception as exc:
                logger.error(
                    "Skipping pending replacement after recovery error",
                    extra={
                        "transition_id": transition.id,
                        "error": str(exc),
                    },
                )
            pending_ids = {
                row.id for row in _pending_rows(store.list_pending_transitions())
            }

    if completed:
        logger.info(
            "Reconciled unfinished replacement transitions",
            extra={"reconciled_count": completed},
        )
    return completed


def apply_pending_transition(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    logger: IStructuredLogger,
    precomputed_vectors: dict[str, list[float]] | None = None,
    coordinator: IStorageCoordinator | None = None,
    orchestration_hook: Callable[[str], None] | None = None,
) -> bool:
    """Converge both indexes to the replacement recorded in ``transition``.

    Returns:
        True when the transition was marked complete.
    """
    store = semantic_engine.memory_store
    pending_ids = {row.id for row in _pending_rows(store.list_pending_transitions())}
    if transition.id not in pending_ids:
        return False

    if transition.kind == TransitionKind.ADD:
        return _apply_pending_add(
            transition,
            semantic_engine=semantic_engine,
            tantivy_engine=tantivy_engine,
            logger=logger,
            precomputed_vectors=precomputed_vectors,
            coordinator=coordinator,
            orchestration_hook=orchestration_hook,
        )
    if transition.kind == TransitionKind.DELETE:
        return _apply_pending_delete(
            transition,
            semantic_engine=semantic_engine,
            tantivy_engine=tantivy_engine,
            logger=logger,
            coordinator=coordinator,
            orchestration_hook=orchestration_hook,
        )

    if _later_intent_exists(
        store, transition, kind=TransitionKind.DELETE, content=transition.new_content
    ):
        _remove_recorded_old(transition, semantic_engine, tantivy_engine)
        if tantivy_engine is not None:
            tantivy_engine.commit()
        semantic_engine.commit()
        if not _delete_converged(
            transition,
            semantic_engine,
            tantivy_engine,
            later_add=False,
        ):
            logger.warning(
                "Superseded replace not complete; old text still live",
                extra={"transition_id": transition.id},
            )
            return False
        _complete_converged_intent(
            store,
            transition.id,
            coordinator=coordinator,
            workspace_id=transition.workspace_id,
            hook=orchestration_hook,
        )
        logger.info(
            "Completed replace intent superseded by a later delete or replace",
            extra={"transition_id": transition.id},
        )
        return True

    replacement_live = _ensure_replacement_present(
        transition,
        semantic_engine=semantic_engine,
        tantivy_engine=tantivy_engine,
        precomputed_vectors=precomputed_vectors,
    )
    if not replacement_live:
        return False
    _remove_recorded_old(transition, semantic_engine, tantivy_engine)

    if tantivy_engine is not None:
        tantivy_engine.commit()
    semantic_engine.commit()

    if not replacement_converged(
        transition, semantic_engine=semantic_engine, tantivy_engine=tantivy_engine
    ):
        logger.warning(
            "Replacement transition not complete; indexes have not converged",
            extra={
                "transition_id": transition.id,
                "old_memory_id": transition.old_memory_id,
                "workspace_id": transition.workspace_id,
            },
        )
        return False

    _complete_converged_intent(
        store,
        transition.id,
        coordinator=coordinator,
        workspace_id=transition.workspace_id,
        hook=orchestration_hook,
        leftover_add_content=transition.old_content,
    )
    logger.info(
        "Applied pending replacement transition",
        extra={
            "transition_id": transition.id,
            "archive_id": transition.archive_id,
            "old_memory_id": transition.old_memory_id,
            "workspace_id": transition.workspace_id,
        },
    )
    return True


def replacement_converged(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
) -> bool:
    """Return True when NEW is live, the recorded OLD id is gone, and Tantivy agrees."""
    new_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    if new_id is None:
        return False
    if not _vector_present(semantic_engine, new_id):
        return False
    if not _old_id_gone(transition, semantic_engine):
        return False

    if tantivy_engine is None:
        return True
    if not _tantivy_has(
        tantivy_engine, transition.workspace_id, transition.new_content
    ):
        return False
    if not _tantivy_has(
        tantivy_engine, transition.workspace_id, transition.old_content
    ):
        return True
    return _old_text_live_under_new_id(transition, semantic_engine)


def _old_id_gone(
    transition: ReplacementTransition, semantic_engine: ISemanticSearchEngine
) -> bool:
    """Return True when the recorded old id is gone from SQLite and USearch."""
    if _sqlite_id_for(transition, semantic_engine, transition.old_content) == (
        transition.old_memory_id
    ):
        return False
    return _vector_absent(semantic_engine, transition.old_memory_id)


def _old_text_live_under_new_id(
    transition: ReplacementTransition, semantic_engine: ISemanticSearchEngine
) -> bool:
    """Return True when old text is stored under a different live id."""
    current_id = _sqlite_id_for(transition, semantic_engine, transition.old_content)
    return current_id is not None and current_id != transition.old_memory_id


def _sqlite_id_for(
    transition: ReplacementTransition,
    semantic_engine: ISemanticSearchEngine,
    content: str,
) -> int | None:
    """Look up the live SQLite id for ``content`` in this transition's workspace."""
    return semantic_engine.get_id_by_content(transition.workspace_id, content)


def _apply_pending_add(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    logger: IStructuredLogger,
    precomputed_vectors: dict[str, list[float]] | None = None,
    coordinator: IStorageCoordinator | None = None,
    orchestration_hook: Callable[[str], None] | None = None,
) -> bool:
    """Ensure NEW content exists unless a later delete/replace of that text won."""
    store = semantic_engine.memory_store
    if _later_intent_exists(
        store, transition, kind=TransitionKind.DELETE, content=transition.new_content
    ):
        _complete_converged_intent(
            store,
            transition.id,
            coordinator=coordinator,
            workspace_id=transition.workspace_id,
            hook=orchestration_hook,
        )
        logger.info(
            "Completed add intent superseded by a later delete or replace",
            extra={"transition_id": transition.id},
        )
        return True

    existing_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    vector = (
        None
        if precomputed_vectors is None
        else precomputed_vectors.get(transition.new_content)
    )
    if existing_id is None:
        if precomputed_vectors is not None and vector is None:
            logger.warning(
                "Add intent not complete; precomputed vector missing",
                extra={"transition_id": transition.id},
            )
            return False
        _insert_recovered_add(semantic_engine, transition, vector=vector)
    else:
        if (
            precomputed_vectors is not None
            and not _vector_present(semantic_engine, existing_id)
            and vector is None
        ):
            logger.warning(
                "Add intent not complete; precomputed vector missing",
                extra={"transition_id": transition.id},
            )
            return False
        _reindex_if_vector_missing(
            semantic_engine,
            existing_id,
            transition,
            vector=vector,
        )

    if tantivy_engine is not None and not _tantivy_has(
        tantivy_engine, transition.workspace_id, transition.new_content
    ):
        tantivy_engine.add(transition.workspace_id, transition.new_content)

    if tantivy_engine is not None:
        tantivy_engine.commit()
    semantic_engine.commit()

    if not _add_converged(transition, semantic_engine, tantivy_engine):
        logger.warning(
            "Add intent not complete; indexes have not converged",
            extra={"transition_id": transition.id},
        )
        return False
    _complete_converged_intent(
        store,
        transition.id,
        coordinator=coordinator,
        workspace_id=transition.workspace_id,
        hook=orchestration_hook,
    )
    return True


def _apply_pending_delete(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    logger: IStructuredLogger,
    coordinator: IStorageCoordinator | None = None,
    orchestration_hook: Callable[[str], None] | None = None,
) -> bool:
    """Remove the recorded old id; do not wipe a later re-add of the same text."""
    store = semantic_engine.memory_store
    later_add = _later_intent_exists(
        store, transition, kind=TransitionKind.ADD, content=transition.old_content
    )
    semantic_engine.delete(memory_id=str(transition.old_memory_id))
    if (
        tantivy_engine is not None
        and not later_add
        and not _old_text_live_under_new_id(transition, semantic_engine)
    ):
        _ = tantivy_engine.delete(
            transition.workspace_id,
            transition.old_content,
            verify_exists=False,
        )

    if tantivy_engine is not None:
        tantivy_engine.commit()
    semantic_engine.commit()

    if not _delete_converged(
        transition,
        semantic_engine,
        tantivy_engine,
        later_add=later_add,
    ):
        logger.warning(
            "Delete intent not complete; old id or Tantivy copy still present",
            extra={"transition_id": transition.id},
        )
        return False
    _complete_converged_intent(
        store,
        transition.id,
        coordinator=coordinator,
        workspace_id=transition.workspace_id,
        hook=orchestration_hook,
    )
    return True


def _add_converged(
    transition: ReplacementTransition,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
) -> bool:
    """Return True when the added content is live in every required backend."""
    new_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    if new_id is None or not _vector_present(semantic_engine, new_id):
        return False
    if tantivy_engine is None:
        return True
    return _tantivy_has(tantivy_engine, transition.workspace_id, transition.new_content)


def _delete_converged(
    transition: ReplacementTransition,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    *,
    later_add: bool,
) -> bool:
    """Return True when the recorded old id is gone and Tantivy agrees."""
    if not _old_id_gone(transition, semantic_engine):
        return False
    if tantivy_engine is None or later_add:
        return True
    if _old_text_live_under_new_id(transition, semantic_engine):
        return True
    return not _tantivy_has(
        tantivy_engine, transition.workspace_id, transition.old_content
    )


def _complete_pending_adds_of(
    store: IArchiveMemoryStore,
    *,
    workspace_id: str,
    content: str,
) -> None:
    """Complete leftover pending ADD rows for text that a replace just removed.

    Snapshot-visible leftover ADDs of the old text (including ``id`` greater
    than the replace) must not be applied after this replace converges.
    A genuine later ADD is journaled only after this replace is complete.
    """
    if not content:
        return
    for row in store.list_pending_transitions():
        if row.workspace_id != workspace_id:
            continue
        if row.kind != TransitionKind.ADD:
            continue
        if row.new_content != content:
            continue
        store.complete_replacement_transition(row.id)


def _later_intent_exists(
    store: IArchiveMemoryStore,
    transition: ReplacementTransition,
    *,
    kind: TransitionKind,
    content: str,
) -> bool:
    """Return True when a later add/delete/replace intent for the text exists.

    Listing failures fall through to ``has_later_intent`` rather than treating
    the later write as absent. Pending rows are workspace-scoped.
    """
    pending: list[ReplacementTransition] | None = None
    try:
        pending = store.list_pending_transitions()
    except Exception:
        pending = None
    if pending is not None:
        for other in pending:
            if other.workspace_id != transition.workspace_id:
                continue
            if other.id <= transition.id:
                continue
            if (
                kind == TransitionKind.DELETE
                and other.kind in {TransitionKind.DELETE, TransitionKind.REPLACE}
                and other.old_content == content
            ):
                return True
            if (
                kind == TransitionKind.ADD
                and other.kind == TransitionKind.ADD
                and other.new_content == content
            ):
                return True
    return store.has_later_intent(
        workspace_id=transition.workspace_id,
        kind=kind,
        content=content,
        after_id=transition.id,
    )


def _remove_recorded_old(
    transition: ReplacementTransition,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
) -> None:
    """Delete the recorded old id; tombstone Tantivy only if that text is not live."""
    semantic_engine.delete(memory_id=str(transition.old_memory_id))
    if tantivy_engine is None:
        return

    if _old_text_live_under_new_id(transition, semantic_engine):
        return
    _ = tantivy_engine.delete(
        transition.workspace_id,
        transition.old_content,
        verify_exists=False,
    )


def _recovery_store(
    semantic_engine: ISemanticSearchEngine,
    logger: IStructuredLogger,
) -> IArchiveMemoryStore | None:
    """Return a transition store when pending rows can be listed."""
    store = semantic_engine.memory_store
    if store.db_path and not os.path.exists(store.db_path):
        return None
    return store


def _ensure_replacement_present(
    transition: ReplacementTransition,
    *,
    semantic_engine: ISemanticSearchEngine,
    tantivy_engine: TantivyEngine | None,
    precomputed_vectors: dict[str, list[float]] | None = None,
) -> bool:
    """Ensure the replacement has both SQLite identity and a live vector."""
    existing_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    vector = (
        None
        if precomputed_vectors is None
        else precomputed_vectors.get(transition.new_content)
    )
    if existing_id is None:
        if precomputed_vectors is not None and vector is None:
            return False
        _insert_recovered_add(semantic_engine, transition, vector=vector)
    elif not _vector_present(semantic_engine, existing_id):
        if precomputed_vectors is not None and vector is None:
            return False
        _reindex_if_vector_missing(
            semantic_engine, existing_id, transition, vector=vector
        )

    replacement_id = semantic_engine.get_id_by_content(
        transition.workspace_id, transition.new_content
    )
    if replacement_id is None or not _vector_present(semantic_engine, replacement_id):
        return False
    if tantivy_engine is None:
        return True
    if not _tantivy_has(
        tantivy_engine, transition.workspace_id, transition.new_content
    ):
        tantivy_engine.add(transition.workspace_id, transition.new_content)
    return True


def _reindex_if_vector_missing(
    semantic_engine: ISemanticSearchEngine,
    existing_id: int,
    transition: ReplacementTransition,
    vector: list[float] | None = None,
) -> None:
    """Re-add a SQLite row whose USearch vector was not committed."""
    if _vector_present(semantic_engine, existing_id):
        return

    semantic_engine.delete(memory_id=str(existing_id))
    if vector is not None:
        _ = semantic_engine.add_batch(
            transition.workspace_id,
            [transition.new_content],
            infer=False,
            vectors=[vector],
        )
        return
    semantic_engine.add(
        workspace_id=transition.workspace_id,
        content=transition.new_content,
        infer=False,
    )


def _vector_present(semantic_engine: ISemanticSearchEngine, memory_id: int) -> bool:
    """Return True only when a real index contains ``memory_id``."""
    return semantic_engine.contains_id(memory_id) is True


def _vector_absent(semantic_engine: ISemanticSearchEngine, memory_id: int) -> bool:
    """Return True only when a real index is missing ``memory_id``."""
    return semantic_engine.contains_id(memory_id) is False


def _precompute_add_vectors(
    pending: list[ReplacementTransition],
    semantic_engine: ISemanticSearchEngine,
    logger: IStructuredLogger,
) -> dict[str, list[float]]:
    """Embed missing add/replace text outside the write lock."""
    embedder = semantic_engine.embedder
    needed: list[str] = []
    seen: set[str] = set()
    for transition in pending:
        if transition.kind not in {TransitionKind.ADD, TransitionKind.REPLACE}:
            continue
        content = transition.new_content
        if not content or content in seen:
            continue
        existing_id = semantic_engine.get_id_by_content(
            transition.workspace_id, content
        )
        if existing_id is not None and _vector_present(semantic_engine, existing_id):
            continue
        seen.add(content)
        needed.append(content)
    if not needed:
        return {}

    try:
        raw_vectors = embedder.embed_documents(needed)
    except Exception as exc:
        logger.warning(
            "Pre-embed for recovery add failed; leaving intent pending",
            extra={"error": str(exc)},
        )
        return {}
    if len(raw_vectors) != len(needed):
        logger.warning(
            "Pre-embed for recovery add failed; leaving intent pending",
            extra={"error": "Embedding batch size mismatch for recovery add"},
        )
        return {}

    vectors: dict[str, list[float]] = {}
    for content, raw in zip(needed, raw_vectors, strict=True):
        converted = _as_floats(raw)
        if converted is None:
            logger.warning(
                "Pre-embed for recovery add failed; leaving intent pending",
                extra={"error": "Embedding batch contained an empty vector"},
            )
            return {}
        vectors[content] = converted
    return vectors


def _insert_recovered_add(
    semantic_engine: ISemanticSearchEngine,
    transition: ReplacementTransition,
    *,
    vector: list[float] | None,
) -> None:
    """Insert recovered content, preferring a precomputed vector."""
    if vector is not None:
        _ = semantic_engine.add_batch(
            transition.workspace_id,
            [transition.new_content],
            infer=False,
            vectors=[vector],
        )
        return
    semantic_engine.add(
        workspace_id=transition.workspace_id,
        content=transition.new_content,
        infer=False,
    )


def _pending_rows(raw: object) -> list[ReplacementTransition]:
    """Accept only real transition rows from list_pending_transitions()."""
    rows: list[ReplacementTransition] = []
    for item in _as_objects(raw):
        if isinstance(item, ReplacementTransition):
            rows.append(item)
    return rows


def _as_objects(raw: object) -> list[object]:
    """Treat a dynamic list result as ``list[object]``."""
    if not isinstance(raw, list):
        return []
    return cast("list[object]", raw)


def _as_floats(raw: object) -> list[float] | None:
    """Return a float list when ``raw`` is a non-empty sequence of numbers."""
    items = _as_objects(raw)
    if not items:
        return None
    values: list[float] = []
    for item in items:
        if not isinstance(item, (int, float)):
            return None
        values.append(float(item))
    return values


def _tantivy_has(engine: TantivyEngine, workspace_id: str, content: str) -> bool:
    """Return True when exact-match results include ``content``."""
    return content in engine.find_by_exact_match(workspace_id, content)
