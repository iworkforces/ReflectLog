# MCP Tools

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW

MCP tool implementations. Each `BaseTool` subclass validates input, logs redacted metadata, and delegates to `MemoryManager`. No engine imports.

## STRUCTURE

```
tools/
├── base.py           # ABC + log + InconsistentStateError re-raise
├── add_memories.py   # AddMemoriesTool; counts + replacement previews
├── search_memories.py # SearchMemoriesTool; records + final scores
├── get_all_memories.py # GetAllMemoriesTool; id-ordered page + total
├── get_memory_by_id.py # GetMemoryByIdTool; one record or null
└── delete_memories.py # DeleteMemoriesTool; removed/missing ids
```

Registry lives in `application/mcp_server.py` as `AVAILABLE_TOOL_CLASSES`.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add | `add_memories.py` | `add_memories_async`; dry-run and optional smart replacement |
| Search | `search_memories.py` | Stable ids and final pipeline scores; optional limit 1..50 |
| Get all | `get_all_memories.py` | Atomic page/total read; cap `Config.get_all_limit` / `GET_ALL_LIMIT` (default 1000); limit 0 probes connectivity |
| Get by id | `get_memory_by_id.py` | Positive id; record or null within the workspace |
| Delete | `delete_memories.py` | Up to 100 positive ids; duplicates collapse in first-seen order; actual removed/missing ids |
| Errors | `base.py` | `InconsistentStateError` is re-raised, not wrapped |

## CONVENTIONS

- Every MCP handler takes `workspace_id`. `FastMCPServer` pins a manager with `WorkspaceRegistry.acquire`, then builds the tool.
- Tools never touch USearch/Tantivy/SQLite. Go through `MemoryManager` only.
- Handlers are typed. All five `get_handler()` methods return `Callable[..., Awaitable[dict[str, object]]]`.
- `add_memories` accepts `memories: list[str]`, `dry_run: bool = False`, `allow_replace: bool = True`; exact dedup still applies when replacement is disabled.
- Tests mock `MemoryManager` (`MagicMock(spec=MemoryManager)`), not engines.
- Internal `MemoryManager.delete_memories` returns deleted contents (`list[str]`); MCP `delete_memories` accepts ids and returns a removed/missing dict.
- `BaseTool._raise_tool_error`: if `isinstance(error, InconsistentStateError): raise error` — never flatten to `StorageError` / `SearchError`.
- Names come from `core.enums.ToolName`: `ADD_MEMORIES`, `SEARCH_MEMORIES`, `GET_ALL_MEMORIES`, `GET_MEMORY_BY_ID`, `DELETE_MEMORIES`. Instruction snippets feed dynamic MCP instructions.

## ANTI-PATTERNS

- Never import infrastructure engines from this package.
- Never log memory text or secrets.
- Never wrap or flatten `InconsistentStateError`.
- Never assume both backends return the same hit count.
- Never expose manager health or search-failure counters as an MCP tool.
