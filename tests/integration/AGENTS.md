# Integration Tests

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW

Real USearch + Tantivy + SQLite. In default `testpaths`. NUMBA JIT enabled. Marked `@pytest.mark.integration`; some `@pytest.mark.slow`.

## STRUCTURE

```
tests/integration/
├── test_memory_manager_usearch.py
├── test_mcp_workflows.py
├── test_concurrent_operations.py      # needs RUN_USEARCH_CONCURRENCY_TESTS=1
├── test_thread_safety.py
├── test_chaos.py
├── test_replacement_recovery.py
├── test_layer_boundaries.py
├── test_qwen_embeddings_integration.py
├── test_storage_coordinator_processes.py
├── test_multiprocess_writers.py       # needs RUN_USEARCH_CONCURRENCY_TESTS=1
├── test_multiprocess_tantivy.py
├── test_usearch_atomic_recovery.py    # needs RUN_USEARCH_CONCURRENCY_TESTS=1
├── test_server_shutdown_recovery.py
├── test_legacy_storage_compatibility.py
└── test_installed_mcp_surface.py
```

## CONVENTIONS

- Real engines. One manager per test; always `cleanup_manager` / close indexes.
- `MemoryManager.delete_memories` returns `list[str]`; MCP `delete_memories` takes ids and returns `removed` / `missing`.
- Dual-engine split after USearch success is `InconsistentStateError`, not `StorageError`.
- Set `RUN_USEARCH_CONCURRENCY_TESTS=1` for multiprocess/concurrency files (focused CI sets this).

## ANTI-PATTERNS

- Never mock USearch/Tantivy here.
- Never share a `MemoryManager` across tests.
- Never compact-on-delete assertions.
- Never skip cleanup.

## NOTES

`platform-storage.yml` runs a subset, not this whole folder. `pr-quality.yml` runs the default suite with coverage on Ubuntu. WeMM files stay skipped unless `RUN_LOCAL_MODEL_TESTS=1`.
