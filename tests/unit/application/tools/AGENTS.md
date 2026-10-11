# MCP Tool Unit Tests

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW
Unit tests for MCP tools. Tools go through `MemoryManager` only — engines stay mocked.

## STRUCTURE
```
tests/unit/application/tools/
├── conftest.py              # mock_config, mock_memory_manager, tool fixtures
├── test_add_memories.py
├── test_search_memories.py
├── test_get_all_memories.py
├── test_get_memory_by_id.py
├── test_delete_memories.py
└── test_base.py
```

## WHERE TO LOOK
| Test | Purpose |
|------|---------|
| `test_add_memories.py` | `add_memories_async` counts / dry_run / allow_replace |
| `test_search_memories.py` | Query and limit validation, records and score_kind, `SearchError` |
| `test_get_all_memories.py` | Id-ordered records, page/total, clamping/cap, limit 0 |
| `test_get_memory_by_id.py` | Positive id; workspace-local record or null |
| `test_delete_memories.py` | Positive ids, batch cap, ordered dedup, actual removed/missing ids |
| `test_base.py` | `BaseTool` name / handler / validation |

## CONVENTIONS
- `MagicMock(spec=MemoryManager)` in conftest. Put methods on the protocol.
- No real engines. No real `MemoryManager` construction.
- Tools must not import infrastructure engines.

## ANTI-PATTERNS
- Never call a real `MemoryManager` or engine.
- Never skip input validation tests.
- Ban `getattr`, `optional_attr()`, and `type(obj).__dict__`.
- No `type: ignore`.

## NOTES
Registration is `application/mcp_server.py`. All five handlers return `Awaitable[dict[str, object]]`. Local: MagicMock(spec=MemoryManager) only.
