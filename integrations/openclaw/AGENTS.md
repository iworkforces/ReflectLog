# OpenClaw Memory Plugin

**Generated:** 2026-10-11
**Commit:** 4159261
**Branch:** develop

## OVERVIEW

TypeScript OpenClaw memory-slot plugin (`memory-reflectlog`, kind `memory`). It replaces OpenClaw's built-in memory with ReflectLog over streamable HTTP. pnpm + `tsc` + vitest. The Python rules in the root guide (uv, ruff, ty, pyright) do not apply here.

## STRUCTURE

```
openclaw/
├── openclaw.plugin.json   # Manifest; configSchema mirrors config.ts by hand
├── package.json           # Scripts, engines, openclaw.* host pins
├── pnpm-workspace.yaml    # Settings only (no `packages:`)
├── tsconfig{,.build}.json # Typecheck src+tests / emit src -> dist
├── src/                   # Flat modules + reflectlog/ (client, parse, types)
└── tests/                 # unit/ + support/ fakes; compiled/ native Node smoke + guarded host adapter
```

`dist/` is gitignored. Never commit it.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Entry, registration | `src/index.ts`, `src/plugin.ts` | Only default export; `registerReflectLogMemory`: tools, service, slot warnings |
| Config | `src/config.ts` | Zod `parseConfig` + hand-written `configJsonSchema` |
| Workspace ids, authz | `src/access.ts` | `deriveWorkspaceId`, `resolve*Scope`, `MemoryAccessDeniedError` |
| Every ReflectLog call | `src/service.ts` | `MemoryService`: limits, validation, injection check |
| Agent tools | `src/tools.ts` | Six `memory_*` tools; TypeBox params + Zod input |
| Provider, CLI | `src/provider.ts`, `src/cli.ts` | Native search/get/health runtime; `reflectlog` subcommands |
| Flush | `src/flush.ts`, `src/flush-ledger.ts`, `src/flush-coordinator.ts` | Plan; reserve/pin/success-TTL ledger; per-waiter authority; owned by `MemoryService` |
| Import | `src/importer.ts`, `src/import-run.ts`, `src/import-archive.ts`, `src/import-receipt.ts` | Capture/split/classify; orchestration; exclusive archive writer; receipt/journal model |
| MCP client | `src/reflectlog/mcp-client.ts` | Connect, handshake, retry policy |
| Contracts | `src/reflectlog/types.ts` | `ReflectLogClient`, `ReflectLogError` codes; refs live in `src/references.ts` |
| Compiled smoke | `tests/compiled/` | Package export to `dist/index.js`; real OpenClaw registry + Commander; no loader or mocked SDK; no network/token/gateway |

## CONNECTION CONTRACT

- Transport is `StreamableHTTPClientTransport`. The `transport` enum is only `"http"`, meaning streamable HTTP. Do not add stdio or sse.
- Default URL `http://127.0.0.1:9103/mcp`. Bearer token read at connect time from `process.env[authTokenEnv]` (default `REFLECTLOG_MCP_AUTH_TOKEN`); missing token fails `unauthorized`.
- Handshake needs all five ReflectLog tools and `add_memories.allow_replace`, else `incompatible`. A server `ALLOWED_TOOLS` filter can trigger this.
- Every call passes an explicit `workspace_id`: SHA-256 of `[namespace, agent, partition, session]`, shaped `p-…` (owner) or `c-…` (conversation, keyed on the root session key). ReflectLog does not treat `workspace_id` as an auth boundary; `access.ts` is the boundary.

## CONVENTIONS

- ESM + NodeNext: relative imports end in `.js`; `import type` for type-only (`verbatimModuleSyntax`). Class state uses `#private`.
- `exactOptionalPropertyTypes`: type optionals `?: T | undefined` or use conditional spreads. `noUncheckedIndexedAccess`: indexed reads need a fallback.
- Zod 4 validates config and ReflectLog results; TypeBox builds the host's tool schemas. Absence is a value (`{status: "not_found"}`); failure is a thrown `ReflectLogError` with a closed `code` union.
- Dependencies are injected (`createClient`, `getService`, `connect`, `env`). Tests use `FakeReflectLogClient`, not module mocks; a new client method means updating the fake.
- vitest runs `tests/unit/**/*.test.ts` with `restoreMocks: true`. No linter or formatter is configured.

## ANTI-PATTERNS

- Never retry a mutation after it may have been sent; surface `outcome_unknown`. Reads and `dry_run` adds retry once.
- Never call `ReflectLogClient` outside `MemoryService`. Tools, provider, and CLI go through `getService()`.
- Call `guard.assertCurrent()` before a backend call and after each await in service paths. After an awaited mutation, use `#assertMutationCurrent`: a revoked caller gets `outcome_unknown`, never the result or a pre-dispatch code.
- Never put the token in config, logs, or CLI output. Never log memory text. Text matching `looksLikePromptInjection` is rejected before any remote mutation; an explicitly requested `--archive` still keeps captured raw bytes (including rejected text) under `sources/`.
- Flush goes through `MemoryService.flush` (always add-only); import passes `allowReplace: false` in `import-run.ts`. Never enable replacement for either.
- Workspace overrides: not `.` or `..`, not the derived-id shape (both also enforced by the host JSON Schema), unique (runtime-only). Refs never resolve outside the scope that issued them.
- Never rename/delete live import sources or overwrite an archive.
- Never treat paginated export verification as a snapshot or infer per-entry success from aggregate counts.
- A mutation failure is retryable only with proof it failed before dispatch.

## SYNC HAZARDS

- Config schema exists three times: Zod `configSchema`, `configJsonSchema` (`config.ts`), and `openclaw.plugin.json`. Manifest == `configJsonSchema` is tested, and `tests/unit/config.test.ts` differentially tests both against the pinned host validator. The JSON Schema mirrors the http(s) scheme prefix (`HTTP_URL_PREFIX_PATTERN`) and the reserved/derived override bans (`RESERVED_WORKSPACE_IDS`, `DERIVED_WORKSPACE_ID_RE`); duplicate override values and full URL parsing stay runtime-only (Zod). The JSON Schema must never narrow Zod acceptance.
- `plugin.ts` pairs tool factories with `MEMORY_TOOL_NAMES` by index. Reorder both or neither.
- The version literal `"0.1.0"` in `mcp-client.ts` must match `package.json`. Host pins live in `devDependencies.openclaw`, `peerDependencies`, `openclaw.build`, `openclaw.compat`, and `minimumReleaseAgeExclude`.
- Host pin changes also require reviewing `tests/compiled/support/openclaw-host.mjs`: bump `SUPPORTED_HOST_VERSION` and re-check its header seam notes. Adapter locates the internal registry chunk by content; version/seam mismatch fails closed.
- Flush ledger lives per `MemoryService`: default 512 entries, 6 h success TTL from settlement; pending/unknown pinned. `plugin.ts` lifecycle retirement disposes it; process-local only.

## COMMANDS

```bash
# run inside integrations/openclaw/
pnpm typecheck   # tsc -p tsconfig.json --noEmit
pnpm test        # vitest run
pnpm build       # tsc -p tsconfig.build.json -> dist/
pnpm test:dist   # native Node compiled-entry smoke; build first
pnpm check       # typecheck && test && build && test:dist
```

Root scripts and Python workflows do not run these. Scoped `openclaw-quality` runs on Ubuntu, Node 24/26, pnpm 12.9.1: frozen install then `pnpm check`. Pushes to develop/main and PRs are filtered to `integrations/openclaw/**` or the workflow file; manual dispatch also works. Node: `>=24.16.0 <25 || >=26.1.0`. Unit tests import `src/`; compiled smoke imports the package export and rejects credentials, network attempts, warnings, unhandled rejections, and lingering handles.
