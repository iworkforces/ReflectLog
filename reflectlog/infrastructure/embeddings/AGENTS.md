# Agent Guidelines for reflectlog/infrastructure/embeddings/

**Generated:** 2026-10-11
**Commit:** 4159261
**Branch:** develop

## OVERVIEW
Only populated child of `infrastructure/`. Local WeMM, Qwen OpenRouter, and an LRU cache.

## STRUCTURE

```
embeddings/
├── cached_embeddings.py  # SHA-256 LRU; fail-closed batches
├── wemm_embedding.py     # Local SentenceTransformer; text only
├── wemm_service.py       # acquire_wemm() -> WeMMBorrow; one shared model
└── qwen3_embedding.py    # OpenAI-compat OpenRouter client
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| LRU wrap | `cached_embeddings.py` | `role_separated=True` only for WeMM |
| WeMM | `wemm_embedding.py` | `local_files_only=True`, then download on `OSError` |
| Qwen HTTP | `qwen3_embedding.py` | `LangchainQwenEmbeddings` name leftover |
| HTTP pool | `utility/http.py` | `HttpClientFactory` |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `CachedEmbeddings` | Class | `cached_embeddings.py` | LRU; fail-closed short/empty |
| `WeMMEmbeddings` | Class | `wemm_embedding.py` | Lazy, thread-safe, text-only |
| `LangchainQwenEmbeddings` | Class | `qwen3_embedding.py` | OpenRouter OpenAI-compat |

## CONVENTIONS

- Imports: `reflectlog.infrastructure.embeddings.*`. Not at package root.
- WeMM loads through `wemm_service.acquire_wemm(config)`: one shared, refcounted `WeMMEmbeddings`. A concurrent acquire with a different `WeMMEmbeddingConfig` raises `ValueError`.
- WeMM query and document encoders differ. The manager sets `CachedEmbeddings.role_separated=True` for WeMM only. Qwen shares one SHA-256 key.
- WeMM load tries `local_files_only=True`, then downloads on `OSError`. Empty, short, non-finite, or zero vectors raise.
- Qwen is OpenAI-compatible OpenRouter. Langchain name is leftover.
- Fail-closed on short/empty batches. Cache raises `RuntimeError` on size mismatch or empty vector. Qwen raises if `len(results) != len(texts)` or any empty item.
- `aembed_documents` may init slots as `[]`; leftover empty slots raise. Never treat that as a successful pad.
- No pad with `[]` to fake a complete batch.

## ANTI-PATTERNS

- Never put these modules back on `infrastructure/` root.
- Never treat empty/short embed batches as success.
- Never pad missing vectors with `[]`.
- Never log query text or API keys.

## NOTES

Parent engines stay FLAT. Empty sibling markers `llm/`, `memory/`, `reranking/` get no guides.
