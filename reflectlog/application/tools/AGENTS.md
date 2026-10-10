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
├── add.py            # add(memories, dry_run=False) → counts dict
├── get_all.py        # get_page_with_total(limit, offset)
├── search.py         # hybrid search(query) → list[str]
├── remove.py         # remove(memories); exact match
└── health_check.py   # read-only component status
```

Registry lives in `application/mcp_server.py` as `AVAILABLE_TOOL_CLASSES`.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add | `add.py` | `validate_memories` + `validate_add_batch`; `add_memories_async` |
| Search | `search.py` | Manager runs RRF + optional CE; tool returns texts |
| Remove | `remove.py` | `delete_memories` → `list[str]`; missing ids are a no-op |
| Get all | `get_all.py` | `get_page_with_total`; cap `Config.get_all_limit` (default 1000) |
| Health | `health_check.py` | Read-only; leftovers are not reconciled here |
| Errors | `base.py` | `InconsistentStateError` is re-raised, not wrapped |
| Search records | `search_records.py` | Stable ids and final pipeline scores; optional limit 1..50 |
| Get records | `get_records.py` | Up to 100 positive ids; request order, duplicates collapsed |
| List records | `list_records.py` | Atomic page/total read; same cap as get_all |
| Remove records | `remove_records.py` | Actual removed/missing ids; release shared before delete |

## CONVENTIONS

- Every MCP handler takes `workspace_id`. `FastMCPServer` pins a manager with `WorkspaceRegistry.acquire`, then builds the tool.
- Tools never touch USearch/Tantivy/SQLite. Go through `MemoryManager` only.
- Handlers are typed. `get_handler()` returns `Callable[..., Awaitable[...]]`:
  - `add` → `Awaitable[dict[str, object]]` (`memories: list[str]`, `dry_run: bool`, `allow_replace: bool = True`)
  - `search_records`, `get_records`, `list_records`, `remove_records` → `Awaitable[dict[str, object]]`
  - `search` → `Awaitable[list[str]]`
  - `remove` → `Awaitable[None]`
  - `get_all` → `Awaitable[dict[str, object]]`
  - `health_check` → `Awaitable[dict[str, Any]]`
- Tests mock `MemoryManager` (`MagicMock(spec=MemoryManager)`), not engines.
- `delete_memories` returns deleted contents (`list[str]`), not an int.
- `BaseTool._raise_tool_error`: if `isinstance(error, InconsistentStateError): raise error` — never flatten to `StorageError` / `SearchError`.
- Names come from `core.enums.ToolName`. Instruction snippets feed dynamic MCP instructions.

## ANTI-PATTERNS

- Never import infrastructure engines from this package.
- Never log memory text or secrets.
- Never wrap or flatten `InconsistentStateError`.
- Never assume both backends return the same hit count.
- Never run reconcile or compact from `health_check`.
