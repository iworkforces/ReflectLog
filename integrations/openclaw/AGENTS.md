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
└── tests/                 # unit/*.test.ts + support/fake-client.ts + support/tracked-fs.ts
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

## CONNECTION CONTRACT

- Transport is `StreamableHTTPClientTransport`. The `transport` enum is only `"http"`, meaning streamable HTTP. Do not add stdio or sse.
- Default URL `http://127.0.0.1:9103/mcp`. Bearer token read at connect time from `process.env[authTokenEnv]` (default `REFLECTLOG_MCP_TOKEN`); missing token fails `unauthorized`.
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
- Workspace overrides: not `.` or `..`, not the derived-id shape, unique. Refs never resolve outside the scope that issued them.
- Never rename/delete live import sources or overwrite an archive.
- Never treat paginated export verification as a snapshot or infer per-entry success from aggregate counts.
- A mutation failure is retryable only with proof it failed before dispatch.

## SYNC HAZARDS

- Config schema exists three times: Zod `configSchema`, `configJsonSchema` (`config.ts`), and `openclaw.plugin.json`. Only manifest == `configJsonSchema` is tested; Zod `superRefine` rules are absent from the JSON Schema.
- `plugin.ts` pairs tool factories with `MEMORY_TOOL_NAMES` by index. Reorder both or neither.
- The version literal `"0.1.0"` in `mcp-client.ts` must match `package.json`. Host pins live in `devDependencies.openclaw`, `peerDependencies`, `openclaw.build`, `openclaw.compat`, and `minimumReleaseAgeExclude`.
- Flush ledger lives per `MemoryService`: default 512 entries, 6 h success TTL from settlement; pending/unknown pinned. `plugin.ts` lifecycle retirement disposes it; process-local only.

## COMMANDS

```bash
# run inside integrations/openclaw/
pnpm typecheck   # tsc -p tsconfig.json --noEmit
pnpm test        # vitest run
pnpm build       # tsc -p tsconfig.build.json -> dist/
pnpm check       # typecheck && test && build
```

No CI or root script runs these. Tests import `src/`; nothing smoke-tests `dist/`.
