# Infrastructure Layer

**Generated:** 2026-10-11
**Commit:** 4159261
**Branch:** develop

## OVERVIEW
Protocol wrappers. Live engines stay FLAT here. `embeddings/` is the only populated child.

## STRUCTURE

```
infrastructure/
├── storage_coordinator.py      # Portalocker + generation sidecar
├── embedding_identity.py       # .reflectlog.embedding-identity.json
├── usearch_engine.py           # HNSW + SQLite SoT
├── tantivy_engine.py           # FTS + tombstones
├── memory_store.py             # Identity + journal
├── index_integrity.py          # Pure SQLite-vs-HNSW id-set comparison
├── cross_encoder_reranker.py   # Local FlagReranker; sigmoid default
├── openrouter_reranker.py      # Remote POST /rerank; RERANKER_ENGINE=openrouter
├── reranker_post_processor.py  # CE + temporal compose
├── smart_replacer.py           # LLM replacement
├── llm_provider_base.py        # OpenAI provider base
├── embeddings/                 # ONLY populated child
│   ├── cached_embeddings.py
│   ├── qwen3_embedding.py
│   ├── wemm_embedding.py
│   └── wemm_service.py         # Shared refcounted WeMM instance
├── search/                     # empty marker + guide
├── llm/                        # empty marker (no guide)
├── memory/                     # empty marker (no guide)
└── reranking/                  # empty marker (no guide)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Workspace lease | `storage_coordinator.py` | Thread-owned SHARED; `is_held` |
| Semantic + SoT | `usearch_engine.py` | `get_all`/`count` via MemoryStore |
| Embed identity | `embedding_identity.py` | `ensure_embedding_identity` before HNSW open |
| FTS | `tantivy_engine.py` | Tombstone+commit; `compact()` only |
| Identity/journal | `memory_store.py` | unique `(workspace_id, content)` |
| Salvage archive | `salvage_archive.py` | `ArchiveInspector`; fingerprints SQLite files + sidecars, copies database/WAL to scratch |
| Salvage snapshot | `salvage_snapshot.py` | `SqliteSalvageSnapshotReader`; active rows and raw journal evidence in one read-only transaction |
| Salvage publication | `salvage_publication.py` | `FileSystemBundlePublisher`; exclusive mkdir/files, `INCOMPLETE` first, manifest last |
| CE Step 4 | `cross_encoder_reranker.py` | `rerank`: `[]` for empty input, pass-through when disabled, else score → normalize → CE threshold → recency |
| Remote rerank | `openrouter_reranker.py` | `RERANKER_ENGINE=openrouter`; reuses `RerankerPostProcessor` |
| Index drift | `index_integrity.py` | `evaluate_index_integrity` → `IndexIntegrityReport` |
| Embed cache | `embeddings/cached_embeddings.py` | SHA-256 LRU |

## CONVENTIONS

- `index_integrity.py` is pure id-set math: SQLite ids vs live vector keys. Only same-workspace pending journal rows excuse a difference; a non-empty journal is never a blanket pass. No I/O, and reports carry ids and counts, never memory text.
- First-create HNSW is in-memory only (`Index(...)`). No live `Index.save`. Commit is temp+validate+`os.replace`.
- Tantivy rewrite: `_rewrite_index_in_place`. `_rebuild_index_with_docs` deleted.
- MemoryStore schema: `_create_memories_schema` / `_create_archive_schema` / `_create_transition_schema` only. No `_create_schema`.
- Thread-owned SHARED. Engines skip nested lease when `coordinator.is_held`.
- HNSW fail-closed if SQLite missing/empty/unreadable.
- Embedding identity must match provider, model, and dims. Mismatch tells the caller to export and rebuild offline. Index and SQLite must share a workspace directory.
- Tantivy delete = tombstone+commit. `compact()` is maintenance only.
- Factories: `from_config()`, not `from_app_config()`.
- Search `OSError` → `SearchError`, never `[]`.
- Salvage reads only a disposable SQLite copy with `mode=ro`, never `immutable=1`; committed WAL content stays visible. Never construct `MemoryStore`, replay journal rows, or open vector/FTS engines for salvage.
- Salvage never writes to the archive. Fingerprint inspected files before copying and re-check before publication; reject symlinks at the archive root, `usearch/`, and inspected files. SHM is fingerprinted, but SQLite rebuilds it in scratch.
- Salvage output must not exist (including dangling symlinks), must have an existing parent, and must be outside archive/scratch. Mode `0700` directory, `0600` files; complete only with `manifest.json` present and `INCOMPLETE` absent. No overwrite; incomplete bundles are unusable.

## ANTI-PATTERNS

- No first-create `Index.save(live)`.
- No `_rebuild_index_with_docs`. No MemoryStore `_create_schema`.
- No nested lease when the calling thread already holds the coordinator.
- No HNSW load when SQLite is missing/empty/unreadable.
- No embedding-identity rewrite to accept a different provider, model, or dims. Same width is still incompatible.
- No compact-on-delete. No leftover rows after mid-batch `index.add` fail.
- No CE skip ≤1 inside `CrossEncoderReranker.rerank` — SearchPipeline owns that.
- No extra AGENTS.md under `llm/`, `memory/`, `reranking/`.

## NOTES

Children: only `embeddings/` (full) and `search/` (marker). Empty `llm/`, `memory/`, `reranking/` stay markers.
