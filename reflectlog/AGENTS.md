# reflectlog Package

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW
Installed hatch package. CLI is `server.py:main`. Plugins present; not imported at startup.

## STRUCTURE

```
reflectlog/
├── server.py            # CLI, Numba warmup, SIGINT/SIGTERM persist
├── version.py           # importlib.metadata; fallback "0.0.0"
├── application/         # MCP, Config, WorkspaceRegistry, tools
│   └── utils/           # logging, SecretString, validation, unused SIGHUP
├── core/                # Protocols + StrEnums + adapters + leases
├── infrastructure/      # Engines FLAT; embeddings/ is the only full child
│   ├── embeddings/      # WeMM local + Qwen + LRU
│   ├── embedding_identity.py  # provider/model/dims sidecar
│   └── storage_coordinator.py  # Portalocker + generation sidecar
├── plugins/             # discovery/registry/loading; unwired
└── utility/             # HttpClientFactory, Numba scoring, retry, OS credentials
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Startup | `server.py` | Warmup; no plugin load; no `setup_config_reload` |
| Offline salvage CLI | `application/salvage/__init__.py`, `application/salvage/cli.py` | Lazy dispatch from `server.py` before server env/config loading |
| Salvage service + manifest | `application/salvage/service.py`, `application/salvage/manifest.py` | `ContentSalvageService.from_defaults()`; deterministic JSONL + manifest |
| Salvage contracts | `core/salvage.py`, `core/exceptions.py` | Snapshot/bundle protocols and `SalvageError` family |
| Salvage I/O | `infrastructure/salvage_archive.py`, `infrastructure/salvage_snapshot.py`, `infrastructure/salvage_publication.py` | Fingerprint, disposable read-only SQLite, exclusive publication |
| MCP tools | `application/mcp_server.py` | `workspace_id` on every call; registry then tool |
| Engines | `application/memory/manager.py` | Inlines `from_config()`; `EngineFactory` unused at runtime |
| Workspace pin | `application/memory/workspace_registry.py` | Built on first tool call; idle TTL 900s |
| Workspace lease | `infrastructure/storage_coordinator.py` | Lives here, not in `core/` |
| Lease protocol | `core/storage_coordination.py` | `IStorageCoordinator`, `LeaseMode` |
| Embeddings | `infrastructure/embeddings/` | `WeMMEmbeddings`, `LangchainQwenEmbeddings`, `CachedEmbeddings` |
| Embed identity | `infrastructure/embedding_identity.py` | `.reflectlog.embedding-identity.json` before HNSW |
| HTTP | `utility/http.py` | `HttpClientFactory`; no leftover `http_client.py` |

## CONVENTIONS

- Two utility layers: `utility/` = HTTP/scoring/retry/credentials; `application/utils/` = logging/`SecretString`/validation.
- Factories: `from_config()` (not `from_app_config()`).
- Offline salvage uses `ContentSalvageService.from_defaults()`, not config-driven startup. No `MemoryManager`, `MemoryStore`, engines, journal replay, or model loading.
- Lock order: Portalocker lease → `_write_lock` → `_lock`. `threading.RLock`.
- `access.py` deleted. No `getattr` / `optional_attr` / `invoke_if_callable` / `type(obj).__dict__`.

## ANTI-PATTERNS

- Do not treat `plugins/` as live at startup.
- Do not construct engines via `EngineFactory` in production paths.
- Do not put embedding modules back on `infrastructure/` root.
- Do not move `storage_coordinator.py` out of `infrastructure/`.
- Do not resurrect `http_client.py`, `metrics.py`, or `circuit_breaker.py` as APIs.
