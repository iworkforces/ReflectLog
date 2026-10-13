# ReflectLog Knowledge Base

**Generated:** 2026-10-11
**Commit:** 4159261
**Branch:** develop

## OVERVIEW

Python 3.14 MCP memory server, served over HTTP only (`http`, `streamable-http`; stdio and sse are gone). One `MemoryManager` per workspace. USearch + SQLite identity + Tantivy FTS + raw RRF, optional local FlagReranker or remote OpenRouter rerank. Local WeMM is the field default; env load still defaults to OpenAI. LLM is for smart replacement only. A separate TypeScript OpenClaw plugin lives in `integrations/openclaw/`. A separate Python Hermes memory-provider plugin lives in `integrations/hermes/`.

## STRUCTURE

```
./
├── reflectlog/          # Flat hatch package (not src/)
│   ├── server.py        # Installed CLI: reflectlog.server:main
│   ├── application/     # Config, MCP, workspace registry, add/search, tools
│   ├── core/            # Protocols, StrEnums, adapters, leases
│   ├── infrastructure/  # Engines FLAT; embeddings/ is the only full child
│   ├── plugins/         # Present; not wired at startup
│   └── utility/         # HTTP pool, Numba scoring, OS credentials
├── tests/               # unit/ mirrors package; integration uses real engines
├── integrations/openclaw/  # TypeScript plugin: pnpm + vitest + compiled smoke; scoped CI, no root script
|-- integrations/hermes/    # Python memory provider: uv + pytest; no root script or CI
├── stubs/               # ty extra-paths + pyright stubPath
├── scripts/             # Copied git hooks + platform gates + ad-hoc
├── .github/workflows/   # pr-quality, platform-storage, cd (release)
└── start-*.sh           # Local lint / typecheck / pytest wrappers
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI + signals | `reflectlog/server.py` | Warmup; persist on SIGINT/SIGTERM/SIGBREAK |
| MCP + auth | `application/mcp_server.py` | Tools via `MemoryManager` only |
| MCP surface | `application/tools/` | `add_memories`, `search_memories`, `get_all_memories`, `get_memory_by_id`, `delete_memories`; explicit `workspace_id` |
| Memory facade | `application/memory/manager.py` | Locks, journal, public API |
| Workspace pin | `application/memory/workspace_registry.py` | `acquire()` per tool call; idle TTL 900s; optional `WORKSPACE_MAX_RESIDENT` |
| Embedding identity | `infrastructure/embedding_identity.py` | Sidecar before HNSW open |
| Add / search | `application/memory/` | 3-phase add; 4-step hybrid search |
| Journal replay | `application/memory/replacement_recovery.py` | Restart converge |
| Offline content salvage | `application/salvage/` + `infrastructure/salvage_*.py` | Archive export and read-only bundle verification; contracts in `core/salvage.py` |
| Workspace lease | `infrastructure/storage_coordinator.py` | Portalocker + generation sidecar |
| Config | `application/config/settings.py` | Frozen env `Config` |
| Protocols | `core/` | `ISemanticSearchEngine`, `IStorageCoordinator` |
| Backends | `infrastructure/` | USearch, Tantivy, SQLite, CE |
| Embeddings | `infrastructure/embeddings/` | WeMM local + Qwen OpenRouter + LRU |
| HTTP | `utility/http.py` | `HttpClientFactory` |
| Score math | `utility/scoring.py` | Numba RRF / min-max / recency |
| OpenClaw plugin | `integrations/openclaw/` | MCP client for the five tools; see child guide |
| Hermes plugin | `integrations/hermes/` | Memory provider, MCP client for the five tools; see child guide |
| Release | `.github/workflows/cd.yml` | Tags `v<pyproject version>` after CI passes on `main` |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `main` | Function | `server.py` | CLI | Installed entry |
| `FastMCPServer` | Class | `application/mcp_server.py` | High | Tool registry + transports |
| `MemoryManager` | Class | `application/memory/manager.py` | High | Add/search/delete facade |
| `WorkspaceRegistry` | Class | `application/memory/workspace_registry.py` | High | One manager per workspace |
| `Config` | Dataclass | `application/config/settings.py` | High | Frozen env settings |
| `EmbeddingIdentity` | Dataclass | `infrastructure/embedding_identity.py` | Med | Provider/model/dims sidecar |
| `WeMMEmbeddings` | Class | `infrastructure/embeddings/wemm_embedding.py` | Med | Local SentenceTransformer |
| `ConfigAdapter` | Class | `core/config_adapters.py` | High | `Config` → protocols |
| `USearchEngine` | Class | `infrastructure/usearch_engine.py` | High | HNSW + SQLite SoT |
| `TantivyEngine` | Class | `infrastructure/tantivy_engine.py` | High | FTS + tombstones |
| `MemoryStore` | Class | `infrastructure/memory_store.py` | High | Identity + journal |
| `PortalockerStorageCoordinator` | Class | `storage_coordinator.py` | High | Exclusive/shared lease |
| `RanxFusionEngine` | Class | `memory/fusion/ranx_fusion.py` | Med | RRF Numba; other ranx |
| `CrossEncoderReranker` | Class | `cross_encoder_reranker.py` | High | Search Step 4 |
| `delete_memories` | Method | `manager.py` | High | Returns deleted `list[str]` |

## CONVENTIONS

- 3.14+ native unions; Ruff `UP` + `ANN`/`TC`/`PYI`; double quotes including docstrings.
- Protocols at layer boundaries; factories are `from_config()`.
- Lock order: Portalocker lease → `_write_lock` → `_lock`. `threading.RLock`, not `asyncio.Lock`.
- Embed add/batch **outside** exclusive. Search: no manager SHARED across embed/CE.
- `raise ... from e`. Never log memory text or secrets.
- Dual typecheck: `ty` + pyright. Both must pass. No `type: ignore`.

## ANTI-PATTERNS (THIS PROJECT)

- No `getattr`, `optional_attr()`, `invoke_if_callable()`, or `type(obj).__dict__`.
- No MagicMock auto-attrs as APIs; put the method on a protocol.
- Tools never touch engines; go through `MemoryManager`.
- Resident capacity admission belongs in `WorkspaceRegistry`, never in tools.
- No compact-on-delete. No embed-batch pad with `[]`. No single-list fusion min-max.
- No recency before CE normalize/threshold. Skip CE if ≤1 hit (pipeline owns skip).
- No HNSW load when SQLite is missing/empty/unreadable.
- No first-create `Index.save(live)`. First durable write is temp+replace on commit.

## UNIQUE STYLES

- Add: dedup/replace → embed outside lease → persist NEW then OLD.
- Search: parallel backends → raw RRF (threshold 0.0) → CE if >1 hit.
- Identity: unique `(workspace_id, content)` in SQLite. Tantivy is not exact match.
- `get_all()` / `count()` SoT is USearch/SQLite.
- Journal kinds `add|delete|replace`; later-write-wins.
- Publish generation after engines converge, then leftover ADD complete, then replace complete.
- Coverage fail-under 90%. Pytest warnings are errors.

## COMMANDS

```bash
uv sync
./start-type-check.sh
./start-lint.sh --check
./start-unittest.sh --coverage
uv run reflectlog --transport http --port 9103
uv run reflectlog salvage --archive ARCHIVED_WORKSPACE_DIR --workspace-id ID --output NEW_DIR
uv run reflectlog salvage-verify --bundle BUNDLE_DIR --workspace-id ID
(cd integrations/openclaw && pnpm check)   # TypeScript plugin: typecheck + vitest + build + compiled smoke
uv run --project integrations/hermes/dev --locked pytest -c integrations/hermes/pyproject.toml integrations/hermes/tests  # Python plugin; see child guide for all checks
```

## NOTES

- Installed script: `reflectlog = reflectlog.server:main`. `mcp_server.main` is a thinner second entry.
- `salvage` and `salvage-verify` dispatch lazily before server env/config loading; no manager, engines, journal replay, models, or credentials. Export uses disposable SQLite copies; verification reads bundle bytes only and creates no files.
- `ty` + pyright include `tests`; tests execution env relaxes mock noise.
- Default pytest paths (`tests/unit`, `tests/integration`, `tests/security`) omit `tests/load/` and root `tests/test_*.py`. `start-unittest.sh --parallel` and `-k` run `pytest tests` instead, which bypasses that list.
- Hooks: edit `scripts/git-hooks/`, install via `scripts/setup-git-hooks.sh`. Pre-push = typecheck + lint `--all` (writes). No pytest in hooks. `--all` never fails the push on leftover lint; CI `--check` is the gate.
- pyright reads `pyrightconfig.json` (strict, `pythonPlatform` Darwin); the `[tool.pyright]` block in `pyproject.toml` is ignored. ty reads `[tool.ty]`. Neither checks `scripts/`; ruff does.
- `pr-quality.yml`: Ubuntu typecheck + lint `--check` + coverage. PRs to `develop`/`main`, push to `main` only.
- `platform-storage.yml`: macOS/Ubuntu/Windows storage gates. Push to `develop` does not run `pr-quality`.
- `cd.yml`: after both workflows pass on a `main` push, tags and releases `v<pyproject version>`.
- CLI version: `reflectlog/version.py` reads `importlib.metadata.version("reflectlog")`, falling back to `"0.0.0"` when not installed. The number itself comes from `pyproject.toml`.
- Python: root and PEP 723 scripts require `>=3.14.8`; `.python-version`, `pr-quality.yml`, and `platform-storage.yml` use 3.14.8. Type/lint targets remain `3.14`/`py314`.
- `integrations/openclaw` is not run by root scripts or Python workflows; its scoped `openclaw-quality` workflow runs `pnpm check` on Ubuntu with Node 24/26.
- `integrations/hermes` is not run by any root script or workflow; run its `uv` checks by hand.
- Lease, generation, and embedding-identity rules: `docs/storage-coordination.md`.

## GUIDANCE HIERARCHY

Children: `reflectlog/{application,core,infrastructure,plugins,utility}`, `integrations/openclaw` (TypeScript, own toolchain), `integrations/hermes` (Python, own uv project), and `tests/` mirrors.
Deep: `memory/fusion`, `memory/reranking` (pointer), `utility/platforms`, `infrastructure/embeddings`, `infrastructure/search` (marker).
`workspace_registry.py` and `embedding_identity.py` stay in their parent guides.
No guides on empty `infrastructure/{llm,memory,reranking}`.
