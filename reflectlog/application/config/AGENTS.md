# Application Config

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW

Frozen env-backed `Config`. Enum fields are `StrEnum` from `core.enums` (parsed via `parse_str_enum`). LLM prompt templates are in `core/prompts.py`, not here.

## STRUCTURE

```
config/
├── settings.py      # @dataclass(frozen=True) Config; get_config() singleton
├── validation.py    # ConfigurationValidator + WORKSPACE_ID_PATTERN
├── presets.py       # REFLECTLOG_PROFILE = simple|balanced|performance|quality
└── __init__.py      # Re-exports
```

Adapters (`Config` → protocol configs) live in `core/config_adapters.py` (`from_config()`).

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Env load | `settings.py` `Config.from_environment` | Fail-fast `ConfigurationError` |
| Singleton | `get_config()` / `_config_lock` | Tests reset via `reset_env_after_test` |
| Reranker | `RERANKER_ENGINE` | `cross_encoder` (default) or `none`. `llm` is invalid |
| Embedder | `EMBEDDER_PROVIDER` | Env default `openai` / `text-embedding-3-large` / dims 3072 |
| WeMM | `EMBEDDER_PROVIDER=wemm` | `WeMMModel` + `WEMM_EMBEDDING_DIMS` + `WEMM_DEVICE` |
| Presets | `presets.py` | `apply_preset_to_env` overwrites the keys it sets |
| Workspace | regex `^[A-Za-z0-9_.-]{1,64}$` | Path `..` / leading `/` rejected |
| Resident capacity | `WORKSPACE_MAX_RESIDENT` | `Config.workspace_max_resident: int \| None = None`; unset means unlimited; positive integers only |

## CONVENTIONS

- `Config` is frozen. Mutate env + clear `_config`, do not assign fields.
- Typed enum fields include `EmbedderProvider`, `WeMMModel`, `WeMMDevice`, `TransportMode`, `RerankerEngine`, `FusionMethod`.
- Dataclass field defaults are WeMM-2B / 2048. `from_environment()` does not use those when `EMBEDDER_PROVIDER` is unset.
- MCP tools take `workspace_id` per call. Do not treat a process-global `WORKSPACE_ID` as the selector.
- Capacity parses only when present; blank, zero, negative, fractional, or non-numeric values fail at startup. `_validate_workspace_capacity` also rejects boolean/non-integer config values.
- API keys are `SecretString`; use `.get_secret_value()`. Never log them.
- `setup_config_reload()` exists but is not registered at process startup.

## ANTI-PATTERNS

- Never treat `RERANKER_ENGINE=llm` as valid.
- Never treat the dataclass WeMM defaults as the env-load defaults.
- Never use `str.format` on prompts with user text (`core/prompts.py` uses `string.Template`).
- Never log raw `OPENROUTER_API_KEY`.
- Never skip `__post_init__` / `ConfigurationValidator` when adding fields.
- Never allow path-like `WORKSPACE_ID`.
