# ReflectLog Utility

**Generated:** 2026-10-11
**Commit:** 4159261
**Branch:** develop

## OVERVIEW
HTTP pool, Numba scoring, retry, OS credentials. **Not** `application/utils/`.

## STRUCTURE

```
utility/
├── http.py          # HttpClientFactory (httpx + aiohttp pool)
├── scoring.py       # Numba RRF, min-max, recency, warmup_numba_functions
├── retry.py         # async_retry_with_backoff — tests only
├── security.py      # validate_workspace_id
├── utility.py       # get_anthropic_api_key, init_credentials
├── types.py         # TOKEN_PREFIX, SERVICE_NAME, ApiKeyResult
└── platforms/       # OS credential retrievers (see child AGENTS)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| HTTP pool | http.py | `HttpClientFactory`; `close_all_sync` on shutdown |
| RRF / CE math | scoring.py | `compute_rrf_scores_batch`, `normalize_reranker_scores`, `apply_recency_decay` |
| Retry | retry.py | **No production caller.** Only `tests/unit/utility/test_retry.py` |
| Credentials | utility.py + platforms/ | `get_anthropic_api_key()` → `ApiKeyResult` |

## CONVENTIONS

- This layer: HTTP / scoring / retry / credentials.
- Array min-max of equal scores is `0.5`. `normalize_reranker_scores` maps single and all-equal batches to `1.0`; an empty batch returns `[]`.
- Infra imports `utility.security.validate_workspace_id`, not the copy in `application/utils/security.py`.
- App layer `application/utils/`: `StructuredLogger`, `SecretString`, `validate_memories`.
- `retry.py` is unused at runtime. `SmartReplacer` retries inline; do not wire `async_retry_with_backoff` without an explicit product change.
- No leftover `http_client.py` here or under `application/utils/`.
- `access.py` deleted. No `optional_attr` / `invoke_if_callable`.

## ANTI-PATTERNS

- Do not log tokens or memory text.
- Do not add metrics/circuit-breaker modules here (removed).
- Do not import `application/` from this package.
- Do not treat `retry.py` as a live production path.
- Credential parse/timeout rules live in `platforms/` — do not duplicate here.
