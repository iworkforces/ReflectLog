# Hermes Memory Plugin

**Generated:** 2026-10-12
**Commit:** 6b8afc9
**Branch:** develop

## OVERVIEW

Standalone Python Hermes memory-provider plugin (`reflectlog`) for Hermes Agent `v0.21.7`, pinned to `71d34bad1045610252eda02ee79549436896e784`. Python >=3.14.8; dev and QA use exactly 3.14.8. Stateful streamable HTTP MCP, automatic recall, add-only native mirroring, opt-in turn capture, six tools. One external `memory.provider`; native memory stays active. Own uv project and checks; root scripts do not cover it. The scoped `hermes-quality` workflow runs them in CI.

## STRUCTURE

```text
hermes/
|-- plugin.yaml           # Name, version, requires_env
|-- after-install.md      # Installer instructions
|-- __init__.py           # Lazy provider registration
|-- provider.py           # Hermes lifecycle, config wizard, recall, write gates
|-- tool_handlers.py      # Six tool schemas and argument validation
|-- core/                 # Python >=3.14.8; no Hermes imports
|   |-- client.py          # MCP handshake, framing, deadlines, retry policy
|   |-- config.py          # Frozen settings, validation, private atomic save
|   |-- workspace.py       # Derived ids, context and participant binding
|   |-- policy.py          # Normalization, injection checks, turn capture
|   |-- recall.py          # Bounded stored-data recall formatting
|   |-- writer.py          # Best-effort queue and automatic breaker
|   `-- models.py          # Typed results and error codes
|-- pyproject.toml         # Runtime dependencies + pytest, ruff, pyright config only
|-- dev/                   # Dev uv project: Hermes pin, tools, uv.lock
|-- tests/                 # core/, hermes/, support/fake_server.py
`-- qa/                    # Real CLI, editable Hermes, real ReflectLog server
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Entry, registration | `__init__.py`, `plugin.yaml` | `register`; no network at registration |
| Provider lifecycle | `provider.py` | `ReflectLogMemoryProvider`; initialize before workspace access |
| Config | `core/config.py`, `provider.py` | `Settings`, `get_config_schema`; `$HERMES_HOME/reflectlog.json` |
| Workspace routing | `core/workspace.py` | `resolve_binding`, `derive_workspace_id`; user/profile/override |
| Agent tools | `tool_handlers.py` | Six `memory_*` tools, same names as the OpenClaw plugin; no workspace argument |
| MCP client | `core/client.py` | Every remote call, stateful handshake, no ambiguous mutation retry |
| Write policy | `core/policy.py`, `core/writer.py` | 30720-character cap; 64 jobs; breaker after 5 failures for 120 s |
| Recall | `core/recall.py`, `provider.py` | Synchronous deadline; Hermes owns `<memory-context>` fence |
| Contracts | `core/models.py` | Typed records and closed error-code set |
| Tests, QA | `tests/`, `qa/run.py`, `qa/scenarios.py` | Real Hermes imports; fake MCP responses / real surfaces |

## CONNECTION CONTRACT

- Stateful streamable HTTP MCP via `httpx`; default `http://127.0.0.1:9103/mcp`. `REFLECTLOG_URL` overrides config. JSON and SSE response framing are supported.
- httpx timeouts are per I/O operation, so `_bounded` in `core/client.py` runs each exchange on a daemon thread and the caller stops at one absolute deadline: connect, headers, body, and the close-time session `DELETE`. On expiry the caller starts `_Exchange.cancel` on its own thread and returns; never interrupt or close on the caller. Cancel only shuts down a response that is still `attached`, since a detached connection may be back in the pool. `prefetch` also spends `recall_timeout` on provider lock waits.
- Bearer token from `REFLECTLOG_MCP_AUTH_TOKEN` in the Hermes profile environment; must equal server `REFLECTLOG_MCP_AUTH_TOKEN`. `requires_env` prompts on install; never persist the token in `reflectlog.json`.
- Handshake needs `add_memories`, `search_memories`, `get_all_memories`, `get_memory_by_id`, `delete_memories`, and `add_memories.allow_replace`, else `incompatible`.
- Every remote call passes `workspace_id`. Derived ids are `h-<profile slug>-<24 hex>` from SHA-256 of compact JSON `["reflectlog-hermes/v1", namespace, profile, subject]`. ReflectLog uses ids for routing; plugin gates enforce access.
- Default user routing fails closed for missing gateway participants and non-primary contexts with participants. Owner CLI/cron/subagent routing has subject `null`. Profile scope and explicit overrides bypass user-derived routing checks. Shared chats default inactive; mismatched per-turn authors block access until the next `on_turn_start`. `on_session_switch` (compression runs it mid-turn) must not clear the block.

## CONVENTIONS

- `core/` is Hermes-free Python >=3.14.8, standard library plus `httpx`. Hermes-bound code belongs only in `provider.py`, `tool_handlers.py`, and `__init__.py`.
- Never name a plugin module after Hermes top-level packages: `agent`, `tools`, `plugins`, `providers`, `cron`, `gateway`, `pm`, `hermes_cli`. Do not add plugin directories to `sys.path`.
- Tests use real installed Hermes and `tests/support/fake_server.py`; no fake Hermes modules. Run both integration checks and the Hermes-free core check on Python 3.14.8.
- Pyright strict paths: `core`, `provider.py`, `tool_handlers.py`, `__init__.py`, `qa`. The remaining project uses basic mode. Ruff targets `py314`, line length `88`, double quotes; Pyright uses `3.14`.
- Automatic writes require primary context and policy checks. Explicit store/forget mutations and capture require `memory.write_approval` off; approved native mirroring still runs. Preserve per-turn author and generation checks.
- Unknown config keys and tool arguments fail validation. Keep typed result parsing and error codes in `core/models.py`; absence is a nullable record.

## ANTI-PATTERNS

- Never resend a possibly dispatched mutation; report `outcome_unknown`. Reads and dry-run adds may retry once. The only proof that allows resending a mutation is the MCP SDK's session rejection: HTTP 400/404 with `jsonrpc` `"2.0"`, error `-32600`, a message containing `session` (case-insensitive), and an explicit `"id": null` (`_session_rejection`). Any other 400/404, including an unreadable, unparsable, or withheld body, still ends the session: `_rpc` invalidates on the status line, before `attached()` or the deadline can abandon the exchange. A body read that hits the deadline stays `timeout`. Every `json.loads` of server bytes also catches `RecursionError`. An unreadable add/delete result is `outcome_unknown`, not `malformed`.
- Reject text that cannot encode as UTF-8 in `core/policy.py`; it must never reach request serialization.
- No smart replacement on automatic writes: always `allow_replace=false`, even when `smart_replace` is enabled.
- No destructive mirroring. Mirror new add/replace content only; ignore removes. Remote forgetting is explicit.
- No network in `is_available`, status, `identity_signature`, or `register`. Availability is a local config/token check.
- Never log tokens, memory text, or queries. Never log or return server error text (`server_detail`); tool errors carry the plugin's own message. Do not put secrets in configuration or status output.
- No lazy workspace on an uninitialized provider. Require the initialized, active binding; do not infer authority from a workspace id.
- Do not promise queue durability or exactly-once writes. Breaker state and queued jobs are process-local; full queues drop work.
- Do not wrap recall in a second `<memory-context>` fence or treat recalled text as instructions.
- Hermes v0.21.7 defaults `memory.prefetch_spill_enabled` to `false`; it is an upstream opt-in. Recall is already bounded by `recall_max_chars` and needs no spill opt-in.

## SYNC HAZARDS

- `plugin.yaml` version == `pyproject.toml` version == `client_version` in `core/client.py`: `"0.1.1"`.
- `get_config_schema` in `provider.py` exposes wizard fields; keep their names, coercion, and defaults consistent with `core/config.py`. Advanced settings remain in the JSON file; secrets go to `.env`.
- Tool schemas and handler argument sets both live in `tool_handlers.py`. Update both together; unknown arguments are rejected.
- Automatic recall is attempt 1; keep the `attempt` schema bounds and allowlist in `tool_handlers.py` in sync with `MAX_SEARCH_ATTEMPTS` in `core/recall.py`.
- Hermes copies the plugin `pyproject.toml` into a uv workspace rooted at `hermes-agent`. Keep `[tool.uv]` and `[dependency-groups]` out of it; a `hermes-agent` source there makes `hermes plugins install` fail to activate the provider. `dev/pyproject.toml` dependencies must equal the plugin's `[project].dependencies`. `tests/hermes/test_loading.py` checks both.
- Hermes pin in `dev/pyproject.toml` and resolved commit in `dev/uv.lock` must agree. QA also pins the release commit. The dev wheel uses `HERMES_NIX_BUILD=1`; real QA installs an editable checkout.

## COMMANDS

```bash
# run from the repository root
uv sync --project integrations/hermes/dev --locked --python 3.14.8
uv run --project integrations/hermes/dev --locked pytest -c integrations/hermes/pyproject.toml integrations/hermes/tests
uv run --project integrations/hermes/dev --locked ruff check --config integrations/hermes/pyproject.toml integrations/hermes
uv run --project integrations/hermes/dev --locked ruff format --check --config integrations/hermes/pyproject.toml integrations/hermes
uv run --project integrations/hermes/dev --locked pyright --project integrations/hermes
uv run --no-project --python 3.14.8 --with pytest --with "httpx>=0.27,<1" python -m pytest -c integrations/hermes/pyproject.toml integrations/hermes/tests/core
```

```text
uv run --project integrations/hermes/dev --locked python integrations/hermes/qa/run.py [--scenario S1 ...] [--repo PATH] [--env-file PATH] [--hermes-checkout PATH] [--report PATH] [--keep]
```

Repeat `--scenario` for a subset; default is `S1` through `S6`. `S6` may skip without usable LLM credentials. Exit `0` requires no `FAIL`. The scoped `.github/workflows/hermes-quality.yml` runs exactly the six commands above as the `dev` job (first five) and a separate Hermes-free `core` job (last), on Python 3.14.8 with no secrets, server, or model downloads. No root script runs them. `qa/run.py` stays manual: it needs a real server and may `SKIP`.
