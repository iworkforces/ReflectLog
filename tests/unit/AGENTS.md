# Unit Tests

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW

Unit tree mirrors `reflectlog/`. Application/core/plugins/utility mock engines. **Infrastructure unit uses real USearch/Tantivy/SQLite on tmp dirs.** In default `testpaths`.

## STRUCTURE

```
tests/unit/
├── test_server.py       # CLI, signals, Numba warmup, FastMCP boot
├── application/         # MCP, manager, tools, config, utils
├── core/                # Adapters, prompts
├── infrastructure/      # Real engines on TemporaryDirectory
├── plugins/             # Discovery/loading (not wired at startup)
└── utility/             # HTTP factory, retry, scoring helpers
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| `test_server.py` | `reflectlog.server:main`, SIGINT/SIGTERM persist, warmup |
| `application/` | Facade + tools + config; mock MemoryManager |
| `infrastructure/` | Real USearch/Tantivy/SQLite on tmp dirs |
| `utility/` | `HttpClientFactory`, retry |

## CONVENTIONS

- Patch constructors on the **import site**: `reflectlog.application.memory.manager.USearchEngine`.
- MagicMock `spec=`; stub `is_ready=False`, `add_batch` `side_effect`.
- `NUMBA_DISABLE_JIT=1` is set once, at the top of `tests/conftest.py`, before any reflectlog import. No subdirectory conftest touches numba.
- Dual manager files: `application/test_memory_manager.py` vs `application/memory/test_manager.py`.

## ANTI-PATTERNS

- Never construct a real `MemoryManager` in application/ unit tests.
- Never skip `reset_env_after_test`.
- Never mark these `@pytest.mark.integration`.
- Never test `_rebuild_index_with_docs` (removed).

## NOTES

Coverage fail-under 90% is suite-wide. Focused CI does not run this whole tree.
