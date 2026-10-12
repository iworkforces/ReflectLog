# Memory Pipeline Unit Tests

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW

3-phase add, 4-step search, factory, journal replay. Engines mocked. The Numba JIT is disabled suite-wide by `tests/conftest.py`.

## STRUCTURE

```
tests/unit/application/memory/
├── test_manager.py              # MemoryManager internals
├── test_add_phases.py
├── test_search_strategies.py
├── test_search_pipeline.py
├── test_engine_factory.py
├── test_workspace_registry.py
├── test_wemm_wiring.py
├── test_replacement_recovery.py
└── reranking/test_normalization.py
```

Sibling (not here): `../test_memory_manager.py` — dual facade file; both collect.

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| `test_manager.py` | `_write_lock` then `_lock`; `delete_memories` → `list[str]`; Tantivy fail → `InconsistentStateError` |
| `test_add_phases.py` | Embed outside write lock; persist NEW then OLD |
| `test_search_pipeline.py` | `SearchPipeline` / `SearchContext` |
| `test_replacement_recovery.py` | Journal kinds `add\|delete\|replace` |

## CONVENTIONS

Patch on the manager import site:

```python
MODULE = "reflectlog.application.memory.manager"
patch(f"{MODULE}.USearchEngine")
patch(f"{MODULE}.TantivyEngine")
```

Stub `is_ready.return_value = False` and `add_batch.side_effect`. JIT off here; other disable is `../utils/conftest.py` only.

## ANTI-PATTERNS

- Never enable JIT in this tree (breaks coverage).
- Never wrap `InconsistentStateError` as `StorageError`.
- Never construct real USearch/Tantivy here.

## NOTES

`delete_memories` returns `list[str]`, not a count. CE skipped when ≤1 hit.
