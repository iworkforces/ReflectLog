# ReflectLog Application Layer

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW
Orchestration: MCP + `MemoryManager`. Tools never touch engines.

## STRUCTURE

```
application/
├── mcp_server.py        # FastMCPServer(server_config: Config | None = None)
├── constants.py         # MIN_OVERFETCH_LIMIT, log truncate
├── config/              # Frozen Config; settings.setup_config_reload unused
├── memory/
│   ├── manager.py       # Public API; inlines USearch/Tantivy/fusion
│   ├── workspace_registry.py # One manager per workspace_id
│   ├── engine_factory.py # EngineFactory exists; unused at runtime
│   ├── add_phases.py    # 3-phase add; embed outside lease
│   ├── search_strategies.py # 4-step SearchPipeline
│   └── replacement_recovery.py
├── tools/               # BaseTool subclasses
└── utils/               # logging, security.SecretString, validation
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Tool wiring | `mcp_server.py` | `WorkspaceRegistry.acquire` then a per-call tool |
| Facade | `memory/manager.py` | Portalocker → `_write_lock` → `_lock` |
| Workspace pin | `memory/workspace_registry.py` | Idle TTL 900s, max 8 idle; reaper in lifespan |
| Add | `memory/add_phases.py` | Embed outside exclusive; leftover ADD after generation |
| Search | `memory/search_strategies.py` | 4-step; skip CE if ≤1 hit |
| Config reload | `config/settings.py` `setup_config_reload` | SIGHUP helper; not called at startup |
| App utils | `utils/` | Not HTTP/scoring — those are `utility/` |

## CONVENTIONS

- `FastMCPServer(server_config: Config | None = None)`; defaults to `get_config()`.
- Engines: manager calls `USearchConfig.from_config(ConfigAdapter(config))` (CE/replacer same). Do not route startup through `EngineFactory`.
- Tools: `AddMemoriesTool` / `SearchMemoriesTool` / `GetAllMemoriesTool` / `GetMemoryByIdTool` / `DeleteMemoriesTool` only. Each MCP call passes `workspace_id`; all five handlers return `Awaitable[dict[str, object]]`. Never import engines or fusion.
- `get_all_memories` returns an id-ordered records page and total from one read. Sync `MemoryManager.add_memories` skips smart replace; MCP `add_memories` uses the async pipeline.
- Add: embed outside lease. After engines converge, publish generation, then leftover ADD complete, then replace complete.
- `application/utils/` has no `http_client.py` / `metrics.py` / `retry.py` / `circuit_breaker.py`.
- `access.py` deleted. No `getattr` / `optional_attr` / `invoke_if_callable`.

## ANTI-PATTERNS

- Tools must not import engines or fusion.
- Do not call `setup_config_reload()` from `server.py` / `FastMCPServer`.
- Do not treat leftover `utils/*.pyc` (`http_client`, `metrics`, `circuit_breaker`) as APIs.
- Do not construct engines via `EngineFactory` at runtime.
