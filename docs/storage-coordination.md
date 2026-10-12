# Storage coordination

ReflectLog coordinates workspace writes on the **local filesystem** with
`portalocker==4.3.0`. There is no Redis or distributed lock, and no ACID
claim across SQLite, USearch, and Tantivy.

## Sidecars

Each workspace root is `indexes/<canonical_workspace_id>/` (validated, stripped,
and lowercased) and contains:

- `.reflectlog.writer.lock` — exclusive/shared Portalocker lease
- `.reflectlog.storage-generation` — integer generation published after
  SQLite, USearch, and enabled Tantivy converge
- `.reflectlog.embedding-identity.json` — versioned provider, model selector,
  and effective vector dimensions; published atomically under the exclusive
  workspace lease

NFS client-local locks and SMB/CIFS `nobrl` are **unsupported**.

## Lease nesting

An EXCLUSIVE owner may nest EXCLUSIVE and SHARED leases; a SHARED-only owner
may nest SHARED leases. SHARED-to-EXCLUSIVE requests raise `LeaseUpgradeError`
immediately, leaving the existing lease and its ownership unchanged. Release
the SHARED lease before requesting EXCLUSIVE. Other threads and processes
continue to wait up to the configured lease timeout.

## Embedding identity and offline rebuild

A workspace reopens with the same embedding provider, model selector, and
effective dimensions. A different provider or model is incompatible even when
its vector width matches; a changed effective width is incompatible too. The
model selector does not pin the upstream checkpoint revision. Pin and manage
checkpoint revisions separately if reproducibility matters.

If the identity sidecar is absent in legacy storage with nonempty memories,
unknown index occupancy, or pending intents, do not assume the old vectors are
compatible. Do not create or edit the sidecar manually to accept them. There
is no automatic migration, deletion, or full-text-only fallback.

For an upgrade, obtain and verify a complete memory-content export for each
workspace from a compatible old environment where possible, before stopping
it. Stop all processes using the workspace, then archive its complete directory
to a safe location. Include SQLite database and WAL files, vector and full-text
indexes, pending intents, generation, lock, and identity sidecars. A content
export alone is not a backup of the journal or indexes. Keep the archive intact.
Configure a distinct, empty workspace storage location for the chosen embedder,
then re-add the exported memories there. If the old environment cannot read
the content, use offline content salvage below. Preserve the full archive
rather than discarding or relabeling the unknown vectors.

### Offline content salvage

Run the command against one archived workspace directory:

```bash
uv run reflectlog salvage --archive ARCHIVED_WORKSPACE_DIR --workspace-id ID --output NEW_DIR
```

`--archive` must be a copy of `indexes/<workspace_id>/` taken after all
ReflectLog processes using it have stopped, not the whole `indexes/` root.
Include `usearch/memories.db`, any `usearch/memories.db-wal` and
`usearch/memories.db-shm` files, and the `.reflectlog.embedding-identity.json`
and `.reflectlog.storage-generation` sidecars. Preserve the complete archive,
including its indexes and journal. `--workspace-id` is explicit and never
inferred. `--output` must not exist (dangling symlinks count as existing), its
parent must exist, and it must not be inside the archive or disposable scratch
directory. An existing export is never overwritten. Scratch is created under
the temporary directory, so `TMPDIR`, `TEMP`, `TMP`, and any `SQLITE_TMPDIR`
must point outside the archive; otherwise the run fails before creating
anything.

Salvage dispatches before server environment/config loading. It constructs no
`MemoryManager`, workspace registry, `MemoryStore`, vector or full-text engine,
or embedding model. No credentials or environment configuration are needed,
and no journal replay occurs. The archive is never modified. Its inspected
SQLite files and sidecars are fingerprinted by size and SHA-256, the database
and any WAL are copied to disposable scratch, and the copied bytes are checked
against those fingerprints. SQLite reads only that copy in one transaction
with `mode=ro`, never `immutable=1`, so committed WAL content is visible.
SQLite rebuilds scratch SHM from the WAL; archive SHM is fingerprinted but not
copied. The original inspected files are re-fingerprinted before publication;
an observed change fails the run. These checks don't detect every live writer
or changes to uninspected files. Stop all writers before archiving. Symlinks at
the archive root, `usearch/`, or any inspected file are rejected.

The output directory is created with mode `0700`; files use mode `0600`.
It contains:

- `memories.jsonl`: one UTF-8 JSON object per observed active memory of the
  requested workspace, sorted by `source_id`. Each object has `content`,
  `source_created_at` (text or null), and `source_id`.
- `manifest.json`: the export contract and raw observations listed below.

Repeated runs over the same archive with the same requested workspace and
ReflectLog version produce byte-identical payload files. No export timestamp,
absolute source path, or scratch name is included.

| Manifest field | Meaning |
|---|---|
| `format`, `format_version` | `"reflectlog.content-salvage"` and `1` |
| `reflectlog_version`, `workspace_id` | Tool version and requested canonical workspace id |
| `rebuild_ready`, `unresolved_reasons` | True when no unresolved journal evidence was observed; sorted reason codes |
| `identity` | `status` and observed scalar `fields`: `version`, `workspace_id`, `provider`, `model`, `dimensions`, when present; not used to validate the old embedder |
| `generation` | `status` and stripped text `raw`, or null; a missing sidecar is not generation zero |
| `source_fingerprints` | List of inspected files with relative `path`, `sha256`, and `size` |
| `schema` | `journal_status` and `tables`, each with `name` and `columns` |
| `records` | `file` (`"memories.jsonl"`), `count`, and payload `sha256` |
| `journal` | Raw rows in `pending`, `unknown`, and `foreign_workspace`; each row has `columns` containing `name`, `type`, and `value`, never merged into active content |
| `history` | `completed_journal_rows`, a count only; completed rows are not exported |
| `foreign_active_count` | Active rows outside the requested workspace, excluded from JSONL |
| `not_preserved` | List of what a content rebuild does not preserve |

The exact `unresolved_reasons` codes are:

| Code | Meaning |
|---|---|
| `journal_pending_intent` | Known pending intent in the requested workspace |
| `journal_unknown_evidence` | Journal rows whose status, kind, or schema can't be interpreted |
| `journal_foreign_workspace_evidence` | Journal rows outside the requested workspace |
| `journal_missing` | No `replacement_transitions` table |
| `journal_unsupported` | Journal lacks required `id`, `workspace_id`, or `status` columns |

Any reason makes `rebuild_ready=false`. JSONL still holds the observed active
rows, but replacement intent has not been resolved. Keep the original archive
for investigation and decide manually before rebuilding. Pending `new_content`
is never exported as active content. A legacy journal without `kind` can count
completed rows as history; other same-workspace rows remain unknown evidence.
Missing or malformed identity/generation sidecars are reported independently
and do not add journal reason codes.

Publication uses exclusive directory creation. An `INCOMPLETE` marker is
created first, payload files are created exclusively and fsynced, and
`manifest.json` is written last. The marker is removed after the payloads are
written and the directory is synced. A bundle is complete only when
`manifest.json` exists and `INCOMPLETE` is absent. On an ordinary publication
failure, the run first restores `INCOMPLETE` if it had already been removed,
then removes its own files, the marker last, and the output directory. The
error says `rolled back` only when that cleanup finished; otherwise the
directory is left with `INCOMPLETE`, or, if even the marker could not be
restored, the error says the partial output must be deleted before use.
Process interruption may also leave a directory with `INCOMPLETE`. Treat it as
unusable and delete it before retrying.

Success prints one JSON summary to stdout with `output`, `rebuild_ready`,
`record_count`, and `unresolved_reasons`, and exits with code 0, including when
`rebuild_ready=false`. A handled salvage failure prints
`reflectlog salvage: <message>` to stderr and exits with code 1. Failures
include a missing database, unsupported memories schema, malformed active
record, damaged or unreadable database, observed archive change, unsafe
output, and publication failure. Error messages don't include memory content.
Unexpected exceptions propagate.

### Verifying a bundle

```bash
uv run reflectlog salvage-verify --bundle BUNDLE_DIR --workspace-id ID
```

The command dispatches before environment/config loading. It reads only the
bundle, without credentials, models, SQLite, engines, scratch or output files.
It never repairs, replays a journal, or imports records. Checks include:

- No `INCOMPLETE` entry, including a dangling symlink; no symlinked bundle root
  or required files, non-regular files, or changes observed during the read.
- A version 1 `reflectlog.content-salvage` manifest with exactly the exported
  keys, the canonical expected workspace id, and records file `memories.jsonl`.
- SHA-256 and record count against the same captured bytes used for parsing.
- Strict UTF-8 JSONL with a final newline and exactly `source_id`, `content`,
  `source_created_at`: non-boolean integer ids, ascending and unique; non-empty
  content, unique by exact comparison; timestamps text or null. Empty bundles
  are valid.
- `rebuild_ready` and `unresolved_reasons` consistent with the journal evidence
  recorded in the manifest, without inspecting the archive or replaying intent.

Read-time change detection covers observed changes only. Verification does not
check authorship or apply `add_memories` limits, trimming, positive-id rules, or
ISO timestamp validation. Valid bundle text can still fail re-add validation.
A matching hash shows internal agreement; `authenticity` is always
`not_verified`.

One content-free JSON line reports `status`, canonical expected `workspace_id`,
`record_count`, `rebuild_ready`, `unresolved_reasons`, `errors` (`{code, entry,
line}` objects), and `authenticity`. Exit 0 means `ready`; 1 means `not_ready`
(internally consistent with unresolved journal evidence) or `invalid`; 2 means
a usage error, with stderr only and no stdout.

### Rebuilding a workspace

To rebuild from a complete bundle after reviewing its journal evidence:

1. Keep the original archive unchanged. Storage is `indexes/` relative to the
   server's working directory, with one directory per workspace.
2. Configure the chosen embedder and run the server from a fresh working
   directory (a new `indexes/` root), or select a workspace id whose directory
   doesn't exist yet. Rebuild into a separate, empty workspace. Never write
   into the archive or reuse its embedding identity sidecar.
3. Read `memories.jsonl` and send its `content` values through the existing
   `add_memories` tool with the target `workspace_id` and
   `allow_replace=False`. Exact deduplication still applies. Limit batches to
   `MAX_ADD_BATCH` (default 100 memories) and `MAX_ADD_CHARS` (default 500000
   total characters), within the server's memory validation limits.
4. Compare `get_all_memories`'s `total` with `records.count` in `manifest.json`.

This content rebuild does not preserve original memory ids, `created_at`
timestamps (re-added memories are stamped at re-add time), vectors or the
full-text index (rebuilt from content), replacement/archive history or
completed journal rows, or the old embedding identity (the new workspace
records its own). Source ids and timestamps in JSONL are observations, not
values the add tool restores. The five MCP tools are unchanged; there is no
import tool.

## Vector index and SQLite id check

A vector key is its SQLite memory id. When the vector index is opened (first
use, or when another writer publishes a newer file) ReflectLog compares the
ids of the workspace's SQLite rows with the live keys of `vectors.usearch`.
The comparison reads SQLite through its own read-only transaction, so it
changes neither the database nor the vector file, and it logs ids and counts
only, never memory text. A matching workspace opens normally, including one
that had memories removed and the index saved and reloaded.

Crash states that restart recovery repairs legitimately leave the two sides
out of step, so **pending** journal rows of the same workspace account for
exactly these ids and nothing else:

| Pending row | Row without a vector | Vector without a row |
|---|---|---|
| ADD | the current id whose text is the added text | none |
| DELETE | the recorded old id, while its row still has the old text | the recorded old id |
| REPLACE | the replacement's current id, and the old id as for DELETE | the recorded old id only |

Completed, unknown, or other-workspace rows account for nothing, and a
non-empty journal never excuses an unrelated difference. Recovery re-checks
the ids once it has converged, while it still holds the workspace lease.

Any other difference refuses the workspace with an `InitializationError`
that lists the counts and a bounded sample of ids and ends with: *Restore a
consistent backup or rebuild this workspace offline.* `add_memories`, `search_memories`, and
`delete_memories` refuse with that text before they journal or write anything, and
`search_memories` does not fall back to full-text results. Nothing is deleted,
re-embedded, or rewritten automatically, and a refused workspace is left as it
was found: startup reads the journal read-only first, so the database is not
switched to WAL or migrated before the check runs. Restore the whole workspace
directory from a backup taken at one moment, or rebuild into an empty
workspace as described above. MCP `get_all_memories` and `get_memory_by_id`,
and internal count and health reads, use SQLite, which stays the source of truth for memory text.

An empty SQLite store with a populated vector index stays refused, including
the case where the last memory was deleted and the process stopped before the
index was saved.

A missing `vectors.usearch` next to a populated SQLite store is refused as
well, with one exception: a workspace whose first add stopped before the first
publish. When pending ADD journal rows of the workspace account for every
SQLite row and nothing else is pending (no pending row of another workspace,
of another kind, or unrecognised), the engine opens an empty in-memory index
and writes nothing. Recovery then embeds the missing vectors, indexes every
pending add, publishes the first `vectors.usearch` through the usual temp
file and replace, and advances the generation only after SQLite, Tantivy and
the vector index agree. If embedding or indexing fails, the journal rows stay
pending and the generation stays where it was. The manager retains the
internal `pending_intent_count`; the next start tries again. No MCP health-check tool exposes it.

## Engines

- **USearch** publishes HNSW snapshots via a same-directory temp file,
  validate, fsync, and `os.replace`. It does not publish generation.
- **Tantivy** scopes readers (shared) and writers (exclusive) to coordinator
  leases. Request-path delete/compact rewrite in place. Leftover
  `.rebuild-bak` restore is startup-only.

## Shutdown

- POSIX: `SIGINT` and `SIGTERM` persist then exit.
- Windows: `SIGBREAK` via `CTRL_BREAK_EVENT` to a new process group.
- Forced termination is a separate abrupt-death case. Lock files are not
  deleted to recover a live owner.

## Capacity

Supported characterization is under 10,000 records per workspace. Do not
treat unpublished speedup ratios as SLOs.
