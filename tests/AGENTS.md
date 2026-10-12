# Test Suite

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW

Pytest suite. Default `testpaths`: `tests/unit`, `tests/integration`, `tests/security`. Coverage `fail_under = 90` (`[tool.coverage.report]` + `COVERAGE_MIN` in `start-unittest.sh`). `filterwarnings = ["error", ...]`.

## STRUCTURE

```
tests/
├── conftest.py          # Shared fixtures; reset_env_after_test autouse
├── unit/                # Mirrors reflectlog/; app mocked, infra real tmp
├── integration/         # Real USearch + Tantivy
├── security/            # In default testpaths
├── load/                # Locust; not pytest
└── test_*.py            # Root demos; not in default testpaths
```

## COLLECTION

| Invoker | Collects |
|---------|----------|
| Bare `pytest` / default `./start-unittest.sh` | `testpaths` only |
| `--parallel` / `--pattern` (`pytest tests/`) | Also root `test_*.py` demos |

`tests/load/` is Locust, not pytest.

## FIXTURES

1. `tests/conftest.py` top — `NUMBA_DISABLE_JIT=1` + `numba.core.config.reload_config()` before any reflectlog import, so the whole session runs without the JIT (coverage traces scoring bodies; same as CI). Never purge or reload numba in a subdirectory conftest: dispatchers created earlier fail with `cannot augment Function(...)`.
2. `tests/conftest.py` fixtures — `mock_usearch_engine` (`is_ready=False`, `add_batch` `side_effect`), `mock_memory_class` (patches `reflectlog.application.memory.manager.USearchEngine`), `set_env_vars`, `reset_env_after_test`.

## CONVENTIONS

- Async: `asyncio_mode=auto`.
- MagicMock: `spec=` real types; stub `is_ready=False`, `add_batch` `side_effect`. No auto-attrs as APIs.
- Patch constructors on the manager import site.
- Tools mock `MemoryManager`, not engines.
- Dual manager tests: `unit/application/test_memory_manager.py` and `unit/application/memory/test_manager.py`.

## ANTI-PATTERNS

- Never treat `fail_under` 90 as a warning.
- Never invent MagicMock auto-attrs.
- Never log memory text or secrets.

## CI

`pr-quality.yml` runs `./start-type-check.sh --check`, `./start-lint.sh --check`, and `NUMBA_DISABLE_JIT=1 ./start-unittest.sh --coverage` on Ubuntu. Triggers: PRs to `develop`/`main`, push to `main`.

`platform-storage.yml` runs `scripts/run_platform_gates.py --focused` plus a **subset** of coordinator/engine tests on macOS, Ubuntu, and Windows. It does **not** run the full suite, lint, typecheck, or coverage. Push to `develop` hits this workflow only.

WeMM tests need `RUN_LOCAL_MODEL_TESTS=1` (`slow`). Qwen HTTP needs `RUN_REAL_API_TESTS=1`.

## RUNNING

```bash
./start-unittest.sh --coverage   # fail_under=90
./start-unittest.sh --parallel
pytest                           # testpaths only
```
