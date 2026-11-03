# Type Stubs Directory

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW

Minimal third-party stubs. `ty.extra-paths` + `pyright.stubPath`. Dual typecheck must pass. No `type: ignore`. Only stub used APIs.

## STRUCTURE

```
stubs/
├── fastmcp/
├── tantivy/
├── usearch/
├── ranx/
├── numba/
├── sentence_transformers/
├── hypothesis/
└── flagembedding/
```

## WHERE TO LOOK

| Stub | Purpose |
|------|---------|
| `fastmcp/` | Server, client, exceptions, middleware |
| `tantivy/` | FTS engine |
| `usearch/` | HNSW index |
| `ranx/` | Fusion (RRF and others) |

## CONVENTIONS

- Only stub symbols the tree actually imports.
- Native 3.14 unions; no `Any` unless the library is untyped at that call site.
- `[tool.ty] extra-paths = ["stubs"]` in `pyproject.toml`.
- `stubPath: "stubs"` in `pyrightconfig.json`.

```python
# stubs/newlib/__init__.pyi
def important_function(arg: str) -> int: ...
```

## ANTI-PATTERNS

- Never stub already-typed libraries.
- Never leave stubs stale after a dependency bump.

## NOTES

Keep `fastmcp/` in sync with the installed fastmcp version.
