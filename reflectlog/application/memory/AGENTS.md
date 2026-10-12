# Memory Package

**Generated:** 2026-10-11
**Commit:** 4159261
**Branch:** develop

## OVERVIEW
`MemoryManager` facade for add, search, delete, journal replay. Persist is batch-only.

## STRUCTURE
```
memory/
├── manager.py                 # Locks, public API, lazy CE / SmartReplacer
├── workspace_registry.py      # acquire() pins one manager per workspace
├── add_phases.py              # 3-phase add (dedup → replace → store)
├── search_strategies.py       # 4-step SearchPipeline
├── engine_factory.py          # Exists; unused at runtime
├── match_utils.py             # has_exact_match only
├── search_health.py           # SearchFailureRecorder: counts + exception type name
├── replacement_recovery.py    # Restart converge of unfinished intents
├── fusion/                    # RanxFusionEngine (see child guide)
└── reranking/                 # Pointer only (see child guide)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Public API | `manager.py` | Portalocker → `_write_lock` → `_lock` |
| Per-workspace pin | `workspace_registry.py` | Builds `MemoryManager`; idle TTL 900s, max 8 idle; optional `WORKSPACE_MAX_RESIDENT` |
| Persist / replace | `add_phases.py` | Embed outside exclusive; NEW then OLD |
| Restart converge | `replacement_recovery.py` | Generation after converge; leftover ADD then replace complete |
| Exact identity | `match_utils.has_exact_match` | SQLite `(workspace_id, content)` only |
| Hybrid search | `search_strategies.py` | Parallel backends → raw RRF → CE if >1 |

## CONVENTIONS

- Registry capacity counts canonical ids across loaded/building/closing/failed-close managers; resident reuse takes no slot. At capacity, persist/close the longest-idle manager before loading; await existing closes. Failed closes stay counted; shutdown retries without a timeout.
- 3-phase add is the async path only. Sync `add_memories` journals `ADD` and skips smart replace.
- `DuplicateDetectionPhase` → `SmartReplacementPhase` → `StoragePhase`.
- 4-step search: parallel backends → RRF/concat → fusion threshold → CE if >1 hit.
- Embed outside exclusive. Phase 3: journal → persist NEW + commit → delete OLD → extra commit → publish generation after converge → leftover ADD complete → replace complete.
- Identity is SQLite unique `(workspace_id, content)`. Tantivy `en_stem` is not exact match.
- `match_utils` exports `has_exact_match` only. Query escaping is `TantivyEngine._escape_tantivy_query`, a private staticmethod.
- `SearchFailureRecorder` counts failures per component under its own lock and keeps only the exception type name and a UTC time. Counters stay internal; there is no MCP health tool.
- Search: backends take short SHARED leases. No manager SHARED across embed/CE.
- Dual-backend search outage (or semantic error + empty FTS) raises `SearchError`. Tantivy delete shortfall after USearch success raises `InconsistentStateError`.

## ANTI-PATTERNS

- No `_add_memory` / single-item persist path; use `_add_memories_batch`.
- No `_delete_memory` / `_record_replacement_transition` wrappers (deleted).
- Do not treat Tantivy phrase/stem hits as identity.
- Do not apply recency before CE normalize + threshold.
- Do not compact Tantivy inside `delete` / `delete_batch`.
- Do not hold manager SHARED across embed, fusion, or CE.

## NOTES

`fusion/` and `reranking/` have their own guides. Scoring helpers live in `utility/scoring.py`.
