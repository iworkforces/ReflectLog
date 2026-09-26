# Storage coordination

ReflectLog coordinates workspace writes on the **local filesystem** with
`portalocker==4.3.0`. There is no Redis or distributed lock, and no ACID
claim across SQLite, USearch, and Tantivy.

## Sidecars

Each workspace root is `indexes/<workspace_id.lower()>/` and contains:

- `.reflectlog.writer.lock` — exclusive/shared Portalocker lease
- `.reflectlog.storage-generation` — integer generation published after
  SQLite, USearch, and enabled Tantivy converge
- `.reflectlog.embedding-identity.json` — versioned provider, model selector,
  and effective vector dimensions; published atomically under the exclusive
  workspace lease

NFS client-local locks and SMB/CIFS `nobrl` are **unsupported**.

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
