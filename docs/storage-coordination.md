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
the content, preserve the full archive for offline recovery rather than
discarding or relabeling the unknown vectors.

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
