# ReflectLog

> An Agentic Memory Layer For Coding Agents

[![Python](https://img.shields.io/badge/python-%E2%89%A53.14.4-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

ReflectLog is an [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that provides persistent, project-based memory storage for Claude Code and other AI agents. It combines semantic vector search with full-text search for intelligent memory retrieval.

## Features

- **Hybrid Search**: Combines semantic similarity (USearch) + exact phrase matching (Tantivy)
- **RRF Fusion**: Reciprocal Rank Fusion for optimal result ranking
- **Pluggable Reranking**: Local cross-encoder relevance scoring (or none)
- **Temporal-Aware Scoring**: Recency decay for handling contradictory memories
- **Smart Memory Replacement**: LLM-based detection of memory updates
- **Multiple Transport Modes**: stdio, HTTP, SSE, streamable-http
- **Eager engine warmup** by default, with optional lazy reranker load

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/iWorkforces/ReflectLog.git
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
# Start with stdio transport (default for MCP clients)
uv run reflectlog

# Or use the launcher
./start-reflectlog-mcp-server.sh

# Start with HTTP transport (requires MCP_AUTH_TOKEN; bind-all needs ALLOW_PUBLIC_BIND=true)
MCP_AUTH_TOKEN=change-me uv run reflectlog --transport http --port 9103
```

## Usage

### MCP Tools

ReflectLog provides five MCP tools:

1. `add(memories: list[str], workspace_id: str, dry_run: bool = False) -> dict`: Store memories; returns stored/skipped/replaced counts
2. `get_all(workspace_id: str, limit: int | None = None, offset: int = 0) -> dict`: Page stored memories (default cap 1000); includes `total` and `truncated`
3. `search(query: str, workspace_id: str) -> list[str]`: Hybrid semantic + full-text search
4. `remove(memories: list[str], workspace_id: str)`: Remove memories by exact match
5. `health_check(workspace_id: str) -> dict`: Workspace-specific health, including `pending_intent_count`

Every tool call, including `health_check`, requires an explicit `workspace_id`.
`WORKSPACE_ID` in the environment or `.env` is not used as a default. Active
workspace managers are cached per workspace. Idle managers expire after 15
minutes without a call; a sweep runs every 60 seconds, with at most 8 idle
managers retained.

Workspace IDs must be 1 to 64 characters from `A-Z`, `a-z`, `0-9`, `_`, `.`,
and `-`; `.` and traversal-like IDs are rejected. Storage treats IDs without
regard to case, so names that differ only in capitalization share a workspace.
The MCP bearer token authenticates access to the server, not to an individual
workspace. Only give workspace access to trusted clients; `workspace_id` is not
an authorization boundary.

### Example Usage

```python
# Add memories
await add(["I prefer Python for web development", "I use FastAPI for APIs"], workspace_id="my-project")

# Search semantically
results = await search("web frameworks", workspace_id="my-project")
# Returns: ["I prefer Python for web development"]

# Get all memories
page = await get_all(workspace_id="my-project")
# Returns {"memories": [...], "total": N, "offset": 0, "limit": 1000, "truncated": bool}

# Remove memories
await remove(["I use FastAPI for APIs"], workspace_id="my-project")

# Check this workspace's health
status = await health_check(workspace_id="my-project")
```

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM/embeddings |

### Optional Configuration

```bash
# Search Settings
SEARCH_LIMIT=5                    # Max results per search
RERANKER_ENGINE=cross_encoder     # cross_encoder or none
# Memory Replacement
ENABLE_SMART_REPLACE=true         # LLM-based memory replacement
SMART_REPLACE_THRESHOLD=0.7       # Confidence threshold

# Server
MCP_TRANSPORT=stdio               # stdio, http, sse, streamable-http
MCP_PORT=9103                     # Port for HTTP transport
LOG_LEVEL=INFO                    # Logging level
```

See `.env.example` for all available options.

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
the checkpoint. `SentenceTransformer(..., trust_remote_code=True)` is required
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

USearch persists a fixed vector width. Changing `EMBEDDING_MODEL` or
`WEMM_EMBEDDING_DIMS` requires rebuilding that workspace's vector index. Export
or otherwise preserve the workspace memories first, stop ReflectLog, recreate
the workspace semantic storage, and re-add the memories. Existing vectors are
not migrated or made compatible automatically.

Real checkpoint tests are opt-in and are never part of normal CI:

```bash
RUN_LOCAL_MODEL_TESTS=1 WEMM_TEST_MODEL=tencent/WeMM-Embedding-2B \
  uv run python -m pytest tests/integration/test_wemm_embeddings_integration.py
RUN_LOCAL_MODEL_TESTS=1 WEMM_TEST_MODEL=tencent/WeMM-Embedding-4B \
  uv run python -m pytest tests/integration/test_wemm_embeddings_integration.py
RUN_LOCAL_MODEL_TESTS=1 WEMM_TEST_MODEL=tencent/WeMM-Embedding-9B \
  uv run python -m pytest tests/integration/test_wemm_embeddings_integration.py
```

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

Pinned runtime: `fastmcp==4.0.2`, `pydantic==2.14.0b1`, `ranx==0.3.21`,
`portalocker==4.3.0`. Mandatory ranx is loaded lazily. See
[docs/storage-coordination.md](docs/storage-coordination.md) for lock
sidecars, crash recovery, and platform signal behavior.

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
