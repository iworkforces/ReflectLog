# Utility Unit Tests

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW
Unit tests for HTTP pool, retry, and workspace-id equivalence. Network mocked.

## STRUCTURE
```
tests/unit/utility/
├── test_http.py       # HttpClientFactory singletons
├── test_retry.py      # async_retry_with_backoff
└── test_security.py   # validate_workspace_id vs application.utils.security
```

## WHERE TO LOOK
| Test | Purpose |
|------|---------|
| `test_http.py` | Pooled httpx / aiohttp; reset singletons |
| `test_retry.py` | Transient exceptions + backoff |
| `test_security.py` | Dual `validate_workspace_id` characterization |

## CONVENTIONS

## ANTI-PATTERNS
- Never log credential values in assertions or fixtures.
- Ban `getattr`, `optional_attr()`, and `type(obj).__dict__`.
- No `type: ignore`.

## NOTES
HTTP factory is `reflectlog.utility.http` (no leftover `http_client.py`). Scoring lives in `utility/scoring.py` and is covered under application/utils numba tests.
