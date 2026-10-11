# ReflectLog Core Protocols

**Generated:** 2026-10-11
**Commit:** 4159261
**Branch:** develop

## OVERVIEW
Protocols, StrEnums, adapters, exceptions, replacement prompts. No engines.

## STRUCTURE

```
core/
├── types.py             # MemoryRecord, Embeddings, IStoredMemory, ISemanticSearchEngine
├── enums.py             # StrEnums: RerankerEngine, EmbedderProvider, WeMMModel, TransitionKind, SearchComponent, SearchScoreKind
├── config.py            # IServerConfig … IAppConfig (6 sub-protocols)
├── config_adapters.py   # ConfigAdapter + fine-grained + create_*_adapter()
├── storage_coordination.py  # IStorageCoordinator, LeaseMode, IStorageLease
├── memory.py            # IMemoryStore, IMemoryBackend, IMemoryManager
├── reranking.py         # IReranker, IRerankerProvider, IRankingResult
├── search_health.py     # Frozen ComponentFailureSnapshot / SearchFailureSnapshot
├── tools.py             # ITool, IToolRegistry, ToolParameter, ToolDefinition
├── logging.py           # ILoggingService, ILogSink
├── prompts.py           # replacement + MCP instructions only
└── exceptions.py        # ReflectLogError → ConfigurationError, SearchError, …
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Semantic contract | types.py | `ISemanticSearchEngine.embedder: Embeddings`; `search` → `(memory, score, created_at)` |
| Stored row | types.py | `IStoredMemory`: id, workspace_id, content, created_at |
| Workspace lease | storage_coordination.py | `IStorageCoordinator`, `LeaseMode` SHARED\|EXCLUSIVE |
| Salvage contracts | salvage.py | Frozen observed records, raw journal values, sidecar/schema reports, bundle, archive/snapshot/publication protocols |
| Salvage failures | exceptions.py | `SalvageError`: input, unsupported schema, malformed record, damaged database, archive changed, unsafe output, publication |
| Closed sets | enums.py | `RerankerEngine.CROSS_ENCODER` / `OPENROUTER` / `NONE`; `parse_str_enum` |
| Config wrap | config_adapters.py | wraps frozen `Config`; `_coerce_reranker_engine` |
| Journal kinds | enums.py | `TransitionKind` add\|delete\|replace |
| Search failures | search_health.py | Snapshots for `SearchComponent`: semantic, tantivy, cross_encoder, openrouter_reranker |

## CONVENTIONS

- `RerankerEngine` is `StrEnum`, not `Literal["cross_encoder", "openrouter", "none"]`. Unknown → `ConfigurationError` via `parse_str_enum`.
- `search_health.py` snapshots hold a count, a UTC timestamp, and an exception type name. No exception objects, no memory text.
- `IAppConfig` = `IServerConfig` + `ISearchConfig` + `IStorageConfig` + `IRerankerConfig` + `IEmbedderConfig` + `IReplacementConfig`.
- Adapters: `ConfigAdapter(config)` or `create_*_adapter()`.
- Lease protocol lives here; Portalocker implementation is `infrastructure/storage_coordinator.py`.
- `access.py` gone. No `getattr` / `optional_attr` / `invoke_if_callable` / `type(obj).__dict__`.
- `prompts.py` is replacement + MCP instructions only.
- `salvage.py` defines `ISalvageSnapshotReader`, `IArchiveInspection`, `IArchiveInspector`, and `IBundlePublisher`; no I/O or engines. Raw journal evidence is separate from active records, never resolved into content.
- Salvage constants: `RECORDS_FILENAME="memories.jsonl"`, `MANIFEST_FILENAME="manifest.json"`, `INCOMPLETE_MARKER="INCOMPLETE"`. `SalvageError` messages carry paths, row ids, column names, and reasons, never memory content.

## ANTI-PATTERNS

- Do not depend on concrete `Config` below the composition root — use `IAppConfig`.
- Do not add scoring/rerank prompts here.
- Do not treat `ISemanticSearchEngine` as embedder-less.
- Do not put Portalocker or sidecar I/O in this package.
