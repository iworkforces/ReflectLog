# ReflectLog memory for OpenClaw

## Connection and configuration

`memory-reflectlog` is an OpenClaw memory-slot plugin backed by a ReflectLog
server over streamable HTTP. It supplies agent tools, a native memory provider,
and an operator CLI. See the [ReflectLog README](../../README.md) for server
setup.

The default endpoint is `http://127.0.0.1:9103/mcp`. The plugin reads its bearer
token at connection time from the environment variable named by `authTokenEnv`,
which defaults to `REFLECTLOG_MCP_AUTH_TOKEN`. Its value must match the server's
`REFLECTLOG_MCP_AUTH_TOKEN`. Keep the token out of configuration files and logs. Set it in
the environment that starts OpenClaw, for example:

```bash
REFLECTLOG_MCP_AUTH_TOKEN=... openclaw reflectlog status
```

The handshake requires `add_memories`, `search_memories`, `get_all_memories`,
`get_memory_by_id`, and `delete_memories`, plus the `allow_replace` parameter on
`add_memories`. A missing token fails with `unauthorized`; missing tools or the
parameter fail with `incompatible`. A server tool filter can cause the latter.

Select the memory slot and disable the built-in memory guidance in the OpenClaw
configuration. The plugin configuration below shows every option and its
runtime default:

```json
{
  "plugins": {
    "slots": { "memory": "memory-reflectlog" },
    "entries": {
      "memory-core": { "enabled": false },
      "memory-reflectlog": {
        "enabled": true,
        "config": {
          "transport": "http",
          "url": "http://127.0.0.1:9103/mcp",
          "authTokenEnv": "REFLECTLOG_MCP_AUTH_TOKEN",
          "namespace": "openclaw",
          "workspaceOverrides": {},
          "defaultMaxResults": 6,
          "connectTimeoutMs": 120000,
          "requestTimeoutMs": 60000,
          "smartReplace": false
        }
      }
    }
  }
}
```

- `transport`: only `"http"`, meaning streamable HTTP. No stdio or SSE.
- `url`: an HTTP or HTTPS URL; default `"http://127.0.0.1:9103/mcp"`. The host
  config schema enforces the http/https scheme prefix; full URL parsing happens
  at runtime.
- `authTokenEnv`: an environment-variable name matching
  `^[A-Za-z_][A-Za-z0-9_]*$`; default `"REFLECTLOG_MCP_AUTH_TOKEN"`.
- `namespace`: deployment namespace matching `^[a-z0-9][a-z0-9_.-]{0,31}$`;
  default `"openclaw"`.
- `workspaceOverrides`: agent-id to owner-private workspace-id map; default
  `{}`. Agent ids match `^[a-z0-9][a-z0-9_-]{0,63}$`; workspace ids match
  `^[a-z0-9_.-]{1,64}$`. The host config schema rejects `.`, `..`, and the
  reserved derived-id shape `^[pc]-[a-z0-9_-]{1,32}-[0-9a-f]{24}$`. Value
  uniqueness is checked only at runtime: a duplicate makes the plugin log
  `ReflectLog configuration rejected` and register only a disabled memory
  capability.
- `defaultMaxResults`: integer from 1 to 50; default `6`.
- `connectTimeoutMs`: integer from 1000 to 600000 milliseconds; default `120000`.
- `requestTimeoutMs`: integer from 1000 to 600000 milliseconds; default `60000`.
- `smartReplace`: boolean; default `false`. Replacement through `memory_store`
  requires both this option and an explicit `replace: true`. Flush and import
  always add only.

Unknown configuration keys are rejected. The plugin sends an explicit
`workspace_id` on every server call. Owner-private and conversation audiences
use separate derived workspaces; overrides apply only to owner-private memory.
The plugin enforces audience access. The server's workspace id is routing, not
an authorization boundary.

## Agent tools and provider

The six tools operate in the caller's authorized audience:

- `memory_search`: search by `query`, with optional `maxResults` and `minScore`
  (also accepts `max_results` and `min_score`).
- `memory_get`: read by `ref`, optionally bounded by `from` and `lines`.
- `memory_store`: store a `memories` batch of 1 to 20 items, with optional
  `replace` opt-in.
- `memory_preview`: preview the same write input without storing anything.
- `memory_forget`: delete by `refs`, from 1 to 100 references.
- `memory_list`: browse with `limit` (default 20, range 1 to 100) and `offset`
  (default 0).

The native provider supports search, reference-based get, and health checks.
It advertises the `memory` source, without provider pagination or project
filtering. Recalled memory text is untrusted data, not instructions.

## Pre-compaction flush deduplication

Before compaction, the flush plan directs the agent to look up existing memory
and save durable items with `memory_store`. Flush writes go through
`MemoryService.flush` and are always add-only: `allowReplace: false`, never
smart replacement.

Identical retries, with the same flush id, workspace, and normalized batch,
share one write while their ledger entry is retained. Each `MemoryService`
owns its ledger for one backend/service lifecycle; this is not global state.
The entry is reserved before dispatch. Pending entries and unconfirmed
`outcome_unknown` entries are pinned, never expired or evicted. A retry joins
the pending write or reports `outcome_unknown` instead of sending it again.

Only confirmed successes expire or can be evicted under capacity pressure.
The default success TTL is 6 hours from settlement, and the default capacity
is 512 entries. If pending and unknown entries fill the ledger, a distinct
new flush fails before dispatch with `FlushCapacityError`. Only a failure
proven to precede dispatch frees that key for a later write attempt. After a
success expires or is evicted, the same flush can be submitted again.

Each waiter's authority is checked before joining, at dispatch, and before
receiving a result. Dispatch requires at least one authorized waiter. One
caller's cancellation detaches that caller; it doesn't cancel the shared
write for other authorized callers.

Stopping or retiring the backend disposes its ledger. Dispatched in-flight
waiters receive `outcome_unknown`; a pending operation that hasn't dispatched
fails with `FlushDisposedError`. The disposed service rejects new mutations,
and a later lifecycle gets a fresh service and ledger.

State is process-local and lost on restart. There is no exactly-once guarantee
across restarts and no server-side idempotency. After `outcome_unknown`, check
with `memory_search`, `openclaw reflectlog list`, or `openclaw reflectlog export`
before retrying.

## Operator CLI

Use `openclaw reflectlog <cmd>`. All subcommands accept `--agent <id>`,
`--conversation <sessionKey>`, and `--json`. The agent defaults to OpenClaw's
configured default agent. Without `--conversation`, commands use its
owner-private workspace; with it, they use the specified conversation
partition. JSON output is pretty-printed by default and compact with `--json`,
except export, which always emits JSONL.

```bash
openclaw reflectlog status --json
openclaw reflectlog search "release decisions" --limit 6
openclaw reflectlog list --limit 100 --offset 0
openclaw reflectlog export --out memories.jsonl
```

- `status`: check slot selection, disabled `memory-core`, token presence,
  workspace selection, and backend connectivity. Any failed check sets exit
  code 1.
- `search <query...>`: accepts `--limit <n>` and `--min-score <x>`. The default
  limit is `defaultMaxResults`; scores used for filtering must be comparable
  on the server's reported scale.
- `list`: accepts `--limit <n>` (default 100) and `--offset <n>` (default 0).
- `get <ref>`: accepts `--from <n>` and `--lines <n>`.
- `forget <ref...>`: requires `--yes` to delete.
- `export`: read the workspace page by page; write JSONL to stdout or to
  `--out <file>`. An output file is written with mode 0600. This observation
  isn't a global snapshot; concurrent changes can make it fail or go unseen.
- `import`: capture local Markdown notes and submit add-only batches, as
  described below.

## Importing Markdown notes

```bash
openclaw reflectlog import --workspace-dir /path/to/workspace --dry-run --json
openclaw reflectlog import --workspace-dir /path/to/workspace --archive --json
```

The syntax is `openclaw reflectlog import [--workspace-dir DIR] [--dry-run]
[--archive] [--json]`, with the shared agent and conversation options above.
Without `--workspace-dir`, it uses OpenClaw's workspace directory for the
selected agent.

### Captured sources and writes

Sources are `MEMORY.md`, `memory.md`, then sorted direct `memory/*.md` children,
not nested directories. Each regular file is read once through one handle.
Final-component symlinks and non-regular files are skipped; `memory/` must
itself be a real directory. Intermediate path components resolve normally.
Duplicate file identities are read only once. The size, SHA-256 digest, and
parsed entries all come from the captured bytes, not a later reread of the
live path.

Invalid UTF-8 is retained as raw bytes and reported as `parse: "unparsed"`
with `unparsedReason: "invalid_utf8"`; it yields no entries. Markdown is split
into entries with heading context and a source suffix. Entries longer than
30720 characters are rejected as `too_long`; entries matching the
prompt-injection check are rejected as `prompt_injection`. These rejected
entries are never sent to the server.

Accepted entries are sent sequentially in batches of at most 20 through
`MemoryService.store`, with `allowReplace: false`. `--dry-run` sends only
dry-run previews, writes no memories, skips verification, and creates no
archive. Combining `--archive` with `--dry-run` is rejected before contacting
the backend.

### Copy-based archives

`--archive` copies captured bytes. It never renames, deletes, or modifies live
Markdown files, even if another tool edits or replaces them during the import.
With captured sources, the archive is created at
`<workspace-dir>/.reflectlog-imported/<YYYYMMDDTHHMMSSZ>/`, using a UTC timestamp.
New directories use mode 0700 and files use mode 0600. Every file is created
exclusively; existing archives are never overwritten.

```text
INCOMPLETE
sources/MEMORY.md
sources/memory.md
sources/memory/<name>.md
progress.jsonl
receipt.json
```

Only captured source paths appear under `sources/`. `INCOMPLETE` is created
first. All captured sources, including rejected and unparsed content, are
archived before any batch is sent. If staging them fails, no batch is attempted.
The journal `progress.jsonl` contains fsynced `run_started`, `batch_intent`
before each dispatch, and `batch_result` afterwards. A `batch_intent` without
a matching `batch_result` means the process stopped while that batch was in
flight; treat the outcome as unknown.

An archive is complete only when `receipt.json` exists and `INCOMPLETE` is
absent. Failed or interrupted runs leave incomplete archives in place with
the marker. If completion fails after the marker was removed, restoring it is
attempted; if that also fails, the error says so. Treat any archive from a run
that reported a publication failure as unusable, even without the marker.
Incomplete archives aren't cleaned up or overwritten. A second archived import in the same
workspace and second fails with `archive_exists` before sending batches.
If no source was captured, no archive is created.

### Receipt and batch states

When it can be written, `receipt.json` describes the run using these fields.
Optional fields are omitted when unavailable:

- `format`: `"reflectlog.openclaw-import"`; `formatVersion`: `1`.
- `outcome`: `"succeeded"` or `"failed"`, the run's result when the receipt was
  written. It does not establish archive completeness. A later archive
  publication failure can leave a receipt with `"succeeded"` in an incomplete
  run.
- `workspaceId`, `startedAt`, `finishedAt`: destination and ISO timestamps.
- `sources`: objects with `relativePath`, `size`, `sha256`, `archivedPath`,
  `parse` (`"parsed"` or `"unparsed"`), optional `unparsedReason`, and `entries`
  containing `total`, `accepted`, and `rejected`. `archivedPath` is relative
  to the archive, under `sources/`.
- `entries`: `accepted` objects with `source`, `line`, and `batch`; `rejected`
  objects with `source`, `line`, and `reason`.
- `batches`: objects with `index`, `size`, `state`, and optional `storedCount`,
  `skippedCount`, and `errorCode`. Batch indices start at 1.
- `totals`: `stored`, `skipped`, and `rejected`. Server counts are summed only
  over acknowledged batches; `rejected` counts local rejections.
- `verification`: `status`, `paginated: true`, `globalSnapshot: false`, optional
  `missing` (objects with `source` and `line`), and optional `errorCode`.
- Optional `failure`: `stage`, `code`, and optional `batch`. Stages are
  `"capture"`, `"archive"`, `"batch"`, `"journal"`, and `"verification"`.
- `notes`: reminders about counts, unknown outcomes, verification, and archives.

The batch states are:

- `acknowledged`: the server confirmed the batch; counts are aggregate.
- `outcome_unknown`: the batch may have been applied; it is never retried
  automatically.
- `failed_before_dispatch`: proven not sent.
- `not_attempted`: an earlier failure stopped the run before this batch.

An accepted entry was accepted for submission, not confirmed persisted one by
one. Aggregate counts never establish which entry was stored or skipped.

After all batches are acknowledged in a non-dry run, the workspace is read
page by page and every accepted text must be observed. Verification reports
`all_observed`, `missing`, or `failed`. It isn't a global snapshot. Its status
is `not_run` when an earlier step failed, nothing was accepted, or the run was
a dry run.

### CLI summary and recovery

Each completed `runImport` call prints one JSON summary, even without `--json`.
It includes:

- `ok`, `accepted`, `stored`, `skipped`, `totals`, `rejected`, and `dryRun`.
  `accepted` is a count; `rejected` is a list of `source:line` strings.
- `files`: objects with `source`, `entries`, `sha256`, `parse`, and optional
  `unparsedReason`.
- `archivedTo`: present only when the archive is complete.
- `archive`: `path`, `status`, `receipt`, and optional `error` code. `path` and
  `receipt` can be null. Status is `complete`, `incomplete`, `not_requested`,
  or `not_created`.
- `batches`: `counts` for all four batch states and `items` containing the
  batch records described above.
- `verification`, optional `failure`, optional `error` hint, and `notes`.

The legacy top-level `skipped` combines locally rejected entries and
server-skipped entries. `totals` separates `stored`, server `skipped`, and
local `rejected`. The exit code is 1 when `ok` is false. Input validation or
setup errors before a run instead print an error and set exit code 1; for
example, `--archive --dry-run` prints
`--archive cannot be combined with --dry-run.`

Remote batches and archive files commit independently. The import has no
transaction spanning them and nothing is rolled back. After a failed import:

1. Read the JSON summary and any receipt. Check batch states and the
   `INCOMPLETE` marker; don't rely on receipt `outcome` alone.
2. Run `openclaw reflectlog list` or `openclaw reflectlog export` with the same
   agent and conversation options to see what landed. For `outcome_unknown`,
   the CLI hint is `Outcome unknown; re-run list/export to check before retrying.`
3. Keep working from the untouched live notes. Retain the incomplete archive
   for inspection.
4. Re-run only after checking the destination. Server exact deduplication
   skips texts already stored, but aggregate counts still don't identify
   individual entries.

## Development

From `integrations/openclaw`, run:

```bash
pnpm check
```

This runs typecheck, vitest, build, then `pnpm test:dist`. To run the compiled
smoke alone, run `pnpm build` first. A native Node child imports
`@iworkforces/openclaw-memory-reflectlog` through its `dist/index.js` package
export, without a TypeScript loader or mocked SDK. It registers with the real
OpenClaw `2026.10.2` registry through a version-guarded adapter that locates the
internal registry chunk by content because the host exports no registry factory.
Host version or registry-seam differences fail closed.

The smoke checks the memory capability, six `memory_*` tools, recall names,
plugin service, and `reflectlog` CLI registration through real Commander. It
validates each config case against both the entry schema and the manifest copy
that the real loader uses, and requires the two verdicts to agree. Direct
registration with any rejected config, including runtime-only rejections the
host schema accepts such as duplicate override values, must fall back to a
disabled memory capability; a host-accepted custom config must register fully.
It also checks missing-token failure before connection and fresh service reuse
after provider close or service stop retires a lazily allocated service. It
needs no token, network, or gateway; credentials in its environment, network
attempts, unhandled rejections, warnings, and lingering handles fail the run.

Root scripts and Python workflows do not run these checks. The scoped
`openclaw-quality` workflow runs on Ubuntu with Node 24 and 26 and pnpm 12.9.1,
using `pnpm install --frozen-lockfile` then `pnpm check`. Pushes to `develop` or
`main` and pull requests trigger it when they touch `integrations/openclaw/**`
or `.github/workflows/openclaw-quality.yml`; it also supports manual dispatch.
Node must satisfy `>=24.16.0 <25 || >=26.1.0`.
