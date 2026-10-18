# ReflectLog

> An Agentic Memory Layer For Coding Agents

[![Python](https://img.shields.io/badge/python-%E2%89%A53.14.8-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

ReflectLog is an [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that provides persistent, Workspace-based memory storage for Claude Code and other AI agents. It combines semantic vector search with full-text search for intelligent memory retrieval.

## Features

- **Hybrid Search**: Combines semantic similarity (USearch) + exact phrase matching (Tantivy)
- **RRF Fusion**: Reciprocal Rank Fusion for optimal result ranking
- **Pluggable Reranking**: Local cross-encoder relevance scoring (or none)
- **Temporal-Aware Scoring**: Recency decay for handling contradictory memories
- **Smart Memory Replacement**: LLM-based detection of memory updates
- **HTTP Transports**: `http` (default) and `streamable-http`, with bearer-token auth
- **Eager engine warmup** by default, with optional lazy reranker load

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/iworkforces/ReflectLog.git
cd ReflectLog

# Install dependencies using uv
uv sync
```

### Configuration

Create a `.env` file with your API key:

```bash
OPENROUTER_API_KEY=sk-or-your-key-here
```

### Running the Server

```bash
# Start with the default HTTP transport (requires REFLECTLOG_MCP_AUTH_TOKEN; bind-all needs ALLOW_PUBLIC_BIND=true)
REFLECTLOG_MCP_AUTH_TOKEN=change-me uv run reflectlog --port 9103

# Or use the launcher
REFLECTLOG_MCP_AUTH_TOKEN=change-me ./start-reflectlog-mcp-server.sh
```

The server accepts only the `http` (default) and `streamable-http` transports.
`stdio` and `sse` are not supported.

### Connect HTTP MCP clients

Choose a secret token and use the same value for the server's `REFLECTLOG_MCP_AUTH_TOKEN`
and each client's `REFLECTLOG_MCP_AUTH_TOKEN`. For example, replace `change-me` in
the command above and export the matching value in the environment that starts
each client:

```bash
export REFLECTLOG_MCP_AUTH_TOKEN='change-me'
```

The local endpoint is `http://127.0.0.1:9103/mcp`. Each client sends
`Authorization: Bearer <token>`. A server `.env` file does not set environment
variables in other client processes. Keep the real token out of tracked config
files and shell history. For access across hosts, use HTTPS; binding to all
interfaces requires `ALLOW_PUBLIC_BIND=true`, and plaintext HTTP must not be
exposed over a network.

Pick the configuration for your client:

- **Claude Code:** Add this to your project's `.mcp.json` (which may be shared).
  [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp).

  ```json
  {
    "mcpServers": {
      "reflectlog": {
        "type": "http",
        "url": "http://127.0.0.1:9103/mcp",
        "headers": { "Authorization": "Bearer ${REFLECTLOG_MCP_AUTH_TOKEN}" }
      }
    }
  }
  ```

  Check with `claude mcp get reflectlog` or `/mcp` inside Claude Code.

- **Codex:** Add this to `~/.codex/config.toml`.
  [Codex MCP documentation](https://developers.openai.com/codex/extend/mcp#streamable-http-servers).

  ```toml
  [mcp_servers.reflectlog]
  url = "http://127.0.0.1:9103/mcp"
  bearer_token_env_var = "REFLECTLOG_MCP_AUTH_TOKEN"
  ```

  Check with `codex mcp list` or `/mcp` inside Codex.

- **Grok Build:** Add this to `~/.grok/config.toml`. Grok expands `${VAR}` in
  header values. [Grok Build MCP documentation](https://docs.x.ai/build/features/mcp-servers).

  ```toml
  [mcp_servers.reflectlog]
  url = "http://127.0.0.1:9103/mcp"
  headers = { Authorization = "Bearer ${REFLECTLOG_MCP_AUTH_TOKEN}" }
  ```

  Check with `grok mcp doctor reflectlog` or `/mcps` inside Grok Build.

- **OpenCode:** Add this entry under `mcp` in `opencode.json`.
  [OpenCode MCP documentation](https://opencode.ai/docs/mcp-servers/).

  ```json
  {
    "mcp": {
      "reflectlog": {
        "type": "remote",
        "url": "http://127.0.0.1:9103/mcp",
        "oauth": false,
        "headers": { "Authorization": "Bearer {env:REFLECTLOG_MCP_AUTH_TOKEN}" }
      }
    }
  }
  ```

  Check with `opencode mcp list`.

- **Hermes Agent:** Install the [ReflectLog memory-provider plugin](integrations/hermes/README.md), rather than adding a plain MCP server entry.

## Usage

### MCP Tools

ReflectLog provides exactly five MCP tools:

1. `add_memories(memories: list[str], workspace_id: str, dry_run: bool = False, allow_replace: bool = True) -> dict`: Store memories; returns `stored_count`, `skipped_count`, `replaced_count`, `replacements` (`old_memory`, `new_memory`, `confidence`, `reason`), and `dry_run`. With `allow_replace=True` (default) and `ENABLE_SMART_REPLACE` enabled, LLM smart replacement may replace an existing contradicting memory. Set `allow_replace=False` to skip smart replacement; exact deduplication still applies. `dry_run=True` previews without storing.
2. `search_memories(query: str, workspace_id: str, limit: int | None = None) -> dict`: Hybrid semantic + full-text search. Returns `results` (`id`, `memory`, final `score`, nullable `created_at`), `score_kind`, and effective `limit`. Queries must contain non-whitespace content and be 1..1000 characters; explicit limits are 1..50 and omitted limits use `SEARCH_LIMIT`. Score kinds are `reranker`, `rrf`, `semantic`, `fulltext`, `concatenated`, and `none`; scores are not comparable across kinds.
3. `get_all_memories(workspace_id: str, limit: int | None = None, offset: int = 0) -> dict`: Returns `records` (`id`, `memory`, nullable `created_at`), `total`, `offset`, `limit`, and `truncated`, ordered by id. Default and maximum page size is `GET_ALL_LIMIT` (default 1000). Negative limits and offsets clamp to 0. `limit=0` returns no records but the real total, useful as a cheap connectivity probe.
4. `get_memory_by_id(id: int, workspace_id: str) -> dict`: Returns `{"record": {"id": ..., "memory": ..., "created_at": ...}}` or `{"record": null}` if the id is absent in that workspace. `created_at` is nullable; foreign-workspace ids are not found. The id must be positive.
5. `delete_memories(ids: list[int], workspace_id: str) -> dict`: Delete by stable id. At most 100 positive ids; duplicates collapse in first-seen order and an empty list is a no-op. Returns `removed` (ids actually deleted) and `missing` (ids not found, including concurrent-deletion shortfalls).

Every tool call requires an explicit `workspace_id: str`.
`WORKSPACE_ID` in the environment or `.env` is not used as a default. Active
workspace managers are cached per workspace. Idle managers expire after 15
minutes without a call; a sweep runs every 60 seconds, with at most 8 idle
managers retained. `WORKSPACE_MAX_RESIDENT` optionally caps resident workspaces;
unset means unlimited. Only positive integers are accepted; blank, zero,
negative, fractional, and non-numeric values fail at startup. Canonical workspace
ids count once across loaded, building, closing, and failed-close managers.
Reusing a resident workspace takes no extra slot. At capacity, a new workspace
first closes the longest-idle manager, even before the idle timeout, and waits
for close to persist its pending writes before loading the new manager. An
in-progress close is awaited. If none is idle or closing, the call fails before
loading with `Resident workspace limit reached; no idle workspace could be released`;
retry later. Existing sessions keep working. Failed closes remain counted until
a close succeeds; shutdown retries them without a timeout. This applies to all
five tools without changing their schemas.

There is no MCP health-check tool. Search-failure counters
remain internal to each workspace manager and are not exposed over MCP.

`ALLOWED_TOOLS` optionally filters the five names with a comma-separated list.
Kebab-case, CamelCase, and an `_tool` suffix are normalized; `all` or `*`
enables all tools. Unknown names, including old tool names, are ignored with a warning.

Workspace IDs must be 1 to 64 characters from `A-Z`, `a-z`, `0-9`, `_`, `.`,
and `-`; `.` and any occurrence of `..` are rejected. Surrounding whitespace
is ignored. IDs are stored lowercase, so capitalization does not distinguish workspaces.
The MCP bearer token authenticates access to the server, not to an individual
workspace. Only give workspace access to trusted clients; `workspace_id` is not
an authorization boundary.

### Example Usage

```python
# Add memories
await add_memories(["I prefer Python for web development", "I use FastAPI for APIs"], workspace_id="my-project")

# Search semantically
results = await search_memories("web frameworks", workspace_id="my-project", limit=5)
# results["results"] contains records with ids, memory text, scores, and created_at

# Get all memories
page = await get_all_memories(workspace_id="my-project")
# Returns {"records": [...], "total": N, "offset": 0, "limit": 1000, "truncated": bool}

# Look up and delete a returned id in the same workspace
if page["records"]:
    memory_id = page["records"][0]["id"]
    record = await get_memory_by_id(id=memory_id, workspace_id="my-project")
    deleted = await delete_memories(ids=[memory_id], workspace_id="my-project")

# Probe connectivity without fetching records
probe = await get_all_memories(workspace_id="my-project", limit=0)
```

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM/embeddings and optional remote reranking |

### Optional Configuration

```bash
# Search Settings
SEARCH_LIMIT=5                    # Max results per search
RERANKER_ENGINE=cross_encoder     # cross_encoder (default), openrouter, or none
# OPENROUTER_RERANK_MODEL=voyageai/rerank-2.5-lite

# Memory Replacement
ENABLE_SMART_REPLACE=true         # LLM-based memory replacement
SMART_REPLACE_THRESHOLD=0.7       # Confidence threshold

# Server
MCP_TRANSPORT=http                # http (default) or streamable-http
MCP_PORT=9103                     # Port for HTTP transport
LOG_LEVEL=INFO                    # Logging level
# WORKSPACE_MAX_RESIDENT=         # Unset: unlimited; set to a positive integer
```

See `.env.example` for all available options.

To opt into remote reranking, set `RERANKER_ENGINE=openrouter` and provide
`OPENROUTER_API_KEY`. Search sends the query and all fused candidate memory
texts to `POST {OPENROUTER_BASE_URL}/rerank` (default
`https://openrouter.ai/api/v1/rerank`) using `OPENROUTER_RERANK_MODEL`.
OpenRouter usage may incur charges. The model setting is separate from the
smart-replacement `LLM_MODEL`. Remote relevance scores are ranked without
cross-encoder score thresholds, then optional recency decay is applied; searches
with at most one result skip reranking. On API failure, search returns the
original fused results. Use `RERANKER_ENGINE=none` to disable reranking.

### Local Tencent WeMM Embeddings

ReflectLog can run Tencent WeMM embeddings locally through the checkpoints'
official SentenceTransformer integration. The runtime surface is text-only even
though the upstream checkpoints are multimodal.

```bash
EMBEDDER_PROVIDER=wemm
EMBEDDING_MODEL=tencent/WeMM-Embedding-2B
WEMM_DEVICE=auto
EMBEDDING_BATCH_SIZE=1
```

Exactly three model selectors are accepted:

| Model | Native dimensions | Allowed `WEMM_EMBEDDING_DIMS` |
|-------|-------------------|----------------------------------|
| `tencent/WeMM-Embedding-2B` | 2048 | 64, 128, 256, 512, 1024, 2048 |
| `tencent/WeMM-Embedding-4B` | 2560 | 64, 128, 256, 512, 1024, 2560 |
| `tencent/WeMM-Embedding-9B` | 4096 | 64, 128, 256, 512, 1024, 2048, 4096 |

Omit `WEMM_EMBEDDING_DIMS` to use the selected model's native width. Set
`WEMM_DEVICE` to `auto`, `cpu`, `cuda`, or `mps`; `auto` delegates device
selection to SentenceTransformers. The first embedding call downloads and loads
the checkpoint if it is not already cached: ReflectLog first tries a local-only
load, then allows a Hugging Face download if that load raises `OSError`.
Set `HF_HOME` to a persistent, writable directory shared across restarts to
retain the checkpoint cache (for example, `export HF_HOME=/path/to/hf-cache`).
After the complete checkpoint and its dependencies are cached, the local-only
load can work without network access. For strict offline operation, set
`HF_HUB_OFFLINE=1` and ensure the cache is populated beforehand; an empty or
incomplete cache cannot load offline, and a subsequent download attempt will
fail. `SentenceTransformer(..., trust_remote_code=True)` is required
by Tencent's official integration, so only run a checkpoint revision whose
repository code you trust.

WeMM embedding inference does not call OpenRouter. ReflectLog still requires its
existing `OPENROUTER_API_KEY` configuration for features such as smart memory
replacement; that requirement is independent of the selected embedding backend.

Tencent's model cards demonstrate CUDA execution only. `cpu` and `mps` are
SentenceTransformers device selectors provided for practical host selection,
not an upstream WeMM compatibility or performance guarantee.

Tencent recommends the `qwen-vl-utils[decord]` extra. ReflectLog installs that
exact extra on Linux x86_64. `decord 0.6.0` has no macOS wheel or source
distribution, so macOS installs the same pinned `qwen-vl-utils` package without
the video-only decoder. This does not reduce ReflectLog's text-only WeMM surface.

These are multi-billion-parameter local models. Disk, system memory, accelerator
memory, and startup time increase substantially from 2B to 4B to 9B. Start with
2B, a small batch size, and an explicit device appropriate for the host. The 9B
checkpoint generally requires server-class resources or an appropriately sized
accelerator; it is not a realistic default for memory-constrained laptops.

Real checkpoint tests are opt-in and are never part of normal CI:

```bash
RUN_LOCAL_MODEL_TESTS=1 WEMM_TEST_MODEL=tencent/WeMM-Embedding-2B \
  uv run python -m pytest tests/integration/test_wemm_embeddings_integration.py
RUN_LOCAL_MODEL_TESTS=1 WEMM_TEST_MODEL=tencent/WeMM-Embedding-4B \
  uv run python -m pytest tests/integration/test_wemm_embeddings_integration.py
RUN_LOCAL_MODEL_TESTS=1 WEMM_TEST_MODEL=tencent/WeMM-Embedding-9B \
  uv run python -m pytest tests/integration/test_wemm_embeddings_integration.py
```

Each workspace records its embedding provider, model selector, and effective
vector dimensions in `.reflectlog.embedding-identity.json`. Reopening with the
same identity is supported. Changing `EMBEDDER_PROVIDER`, `EMBEDDING_MODEL`, or
effective dimensions requires an offline rebuild, even if the new embedder has
the same vector width. A model selector does not pin an upstream checkpoint
revision; manage checkpoint revisions separately.

If the old environment remains compatible, use `get_all_memories` to obtain
and verify a complete export of memory content for each workspace before
stopping it. Stop all
ReflectLog processes, then archive the complete workspace directory, including
SQLite database and WAL files, vector and full-text indexes, pending intents,
and metadata sidecars. Keep this archive intact; a content export alone does
not preserve the journal or indexes. Rebuild into a distinct, empty workspace
storage location with the chosen embedder and re-add the exported content.
Legacy storage without identity metadata that contains
memories, has unknown index occupancy, or has pending intents also requires
this offline export and rebuild. Do not create or rewrite the identity sidecar
to make unknown vectors appear compatible. ReflectLog does not migrate or
delete existing storage automatically, or fall back to full-text-only search.
If a workspace's vector index and SQLite memory ids disagree, for example after
restoring only one of them from a backup, it is refused with an error that says
to restore a consistent backup or rebuild offline; nothing is repaired automatically.
See [storage coordination](docs/storage-coordination.md) for recovery details.

#### Salvaging content from an archived workspace

`reflectlog salvage` reads active content from an archived workspace without
starting the server or loading an embedding model. It needs no credentials or
environment configuration, and leaves the archive unchanged. Use it when the
old environment can't open the workspace, or to export from a stopped archive.

1. Stop all ReflectLog processes using the workspace. Storage is under
   `indexes/` relative to the server's working directory, with one directory
   per workspace. Copy the complete `indexes/<workspace_id>/` directory to a
   safe archive location, including `usearch/memories.db`, any
   `usearch/memories.db-wal` and `usearch/memories.db-shm` files, and the
   `.reflectlog.embedding-identity.json` and `.reflectlog.storage-generation`
   sidecars. Keep the archive intact.
2. Export one archived workspace directory, not the whole `indexes/` root:

   ```bash
   uv run reflectlog salvage --archive ARCHIVED_WORKSPACE_DIR --workspace-id ID --output NEW_DIR
   ```

   `--workspace-id` is required and is never inferred. `NEW_DIR` must not
   exist, even as a dangling symlink; its parent must exist, and it must be
   outside the archive. The command inspects a disposable SQLite copy, with
   committed WAL content visible, and does not replay the journal.
3. Check `manifest.json` and confirm that `INCOMPLETE` is absent. Only then is
   the bundle complete. `memories.jsonl` has one object per observed active
   memory, sorted by `source_id`, with `content`, `source_created_at`, and
   `source_id`. If `rebuild_ready` is false, keep the original archive for
   investigation and decide manually before rebuilding: pending, unknown, or
   foreign-workspace journal evidence, or a missing or unsupported journal,
   leaves intent unresolved. Pending `new_content` is never exported as active
   content. An interrupted run may leave `INCOMPLETE`; treat that directory as
   unusable and delete it before retrying. Existing exports aren't overwritten.
4. Start the server with the chosen embedder in a fresh working directory
   (a new `indexes/` root), or use a workspace id whose directory doesn't exist
   yet. The target must be a separate, empty workspace. Never write into the
   archived directory or reuse its identity sidecar.
5. Re-add the JSONL `content` values with the existing `add_memories` tool,
   passing the target `workspace_id` and `allow_replace=False`. Exact
   deduplication still applies. Keep each batch within `MAX_ADD_BATCH`
   (default 100 memories) and `MAX_ADD_CHARS` (default 500000 total characters).
   Verify `get_all_memories`'s `total` against the manifest's `records.count`.

Content salvage doesn't preserve original memory ids or `created_at` timestamps
in the rebuilt workspace. Re-added memories receive new ids and timestamps at
re-add time. Vectors and the full-text index are rebuilt from content;
replacement/archive history and completed journal rows aren't restored. The
new workspace records its own embedding identity. The five MCP tools are
unchanged; there is no import tool. See
[offline content salvage](docs/storage-coordination.md#offline-content-salvage)
for the manifest, publication, and failure contract.

##### Verifying a bundle

```bash
uv run reflectlog salvage-verify --bundle NEW_DIR --workspace-id ID
```

Verification is offline and read-only. It loads no server configuration,
credentials, models, SQLite database, or engines, creates no files, and never
repairs, replays a journal, or imports content. It checks publication, the
manifest and workspace id, record format, SHA-256, count, and readiness against
the manifest's journal evidence. Empty bundles are valid.

Stdout contains one content-free JSON line, for example for an empty ready
bundle with workspace id `my-project`:

```json
{"authenticity": "not_verified", "errors": [], "rebuild_ready": true, "record_count": 0, "status": "ready", "unresolved_reasons": [], "workspace_id": "my-project"}
```

Exit 0 means `ready`; exit 1 means `not_ready` (consistent but unresolved journal
evidence) or `invalid`. Exit 2 means a usage error, with stderr only and no
stdout. Missing arguments print usage; an invalid workspace id prints
`reflectlog salvage-verify: --workspace-id is not a valid workspace id`.
Errors use `{code, entry, line}` objects, never memory content.

A matching hash only shows that the files agree, not who wrote them;
`authenticity` is always `not_verified`. Verification does not apply
`add_memories` validation, so valid bundle content can still be rejected on
re-add. See the storage guide for the exact checks.

## Architecture

```
ReflectLog/
├── reflectlog/
│   ├── server.py              # CLI entry point
│   ├── application/           # Business logic
│   │   ├── mcp_server.py      # FastMCPServer orchestrator
│   │   ├── memory/            # Memory management
│   │   ├── tools/             # MCP tool implementations
│   │   └── config/            # Configuration management
│   └── infrastructure/        # External integrations
│       ├── usearch_engine.py  # Semantic vector search
│       ├── tantivy_engine.py  # Full-text search
│       └── cross_encoder_reranker.py  # Local cross-encoder reranking
```

### Data Persistence

- **USearch**: `indexes/{workspace_id}/usearch/` - Vector index + SQLite messages
- **Tantivy**: `indexes/{workspace_id}/tantivy/` - Full-text index

## Development

### Commands

```bash
# Type checking
./start-type-check.sh

# Linting
./start-lint.sh --all

# Testing
./start-unittest.sh
./start-unittest.sh --coverage

# Run server (workspace_id required on every tool call)
./start-reflectlog-mcp-server.sh

# Export content from an archived, stopped workspace
uv run reflectlog salvage --archive ARCHIVED_WORKSPACE_DIR --workspace-id ID --output NEW_DIR
uv run reflectlog salvage-verify --bundle NEW_DIR --workspace-id ID
```

### Testing

```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest tests/unit/ -v

# Run with coverage
./start-unittest.sh --coverage
```

## Performance

- **Exact Search**: SIMD-optimized brute-force (default, best for <10K vectors)
- **Approximate Search**: HNSW algorithm (for large collections)
- **Phased Parallel Add**: 3-phase pipeline; historical 5-8x claims are not
  a supported SLO
- **LRU Query Cache**: Reduces embedding API calls

## Documentation

- [docs/storage-coordination.md](docs/storage-coordination.md) - Local lock contract
- [CLAUDE.md](CLAUDE.md) - Comprehensive developer documentation
- [openspec/AGENTS.md](openspec/AGENTS.md) - OpenSpec workflow guide
- [.env.example](.env.example) - Full configuration reference

## Contributing

1. Install git hooks: `./scripts/setup-git-hooks.sh`
2. Create a branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `./start-unittest.sh`
5. Commit and push

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [USearch](https://github.com/unum-cloud/usearch) - Vector search engine
- [Tantivy](https://github.com/tantivy-search/tantivy-py) - Full-text search
- [ranx](https://github.com/AmenRa/ranx) - Ranking fusion algorithms
