# MCP Load Benchmark

`locustfile.py` is a standalone FastMCP Client benchmark, not a Locust scenario or a pytest file. It invokes the five-tool MCP surface (`add_memories`, `search_memories`, `get_all_memories`, `get_memory_by_id`, `delete_memories`) with `workspace_id` on every call. Connectivity probes use `get_all_memories(limit=0)`. No extra dependency is needed.

Start a local HTTP server separately, then run:

```bash
uv run --frozen --no-sync reflectlog --transport http --port 9103
uv run --frozen --no-sync python tests/load/locustfile.py --clients 2 --iterations 10
```

The default endpoint is `http://127.0.0.1:9103/mcp`. Override it with `--url` when needed. Set `REFLECTLOG_MCP_AUTH_TOKEN` for a server requiring a bearer token; it is never printed. The default workspace is a fresh `bench-<uuid>`; use `--workspace-id` only for an isolated disposable workspace. The benchmark removes only its own per-run seed and measured memories; unrelated workspace contents are preserved. If cleanup fails, the report records `cleanup_errors` and the CLI exits nonzero. Never point this benchmark at production.

One connected client is used per worker (1 to 32). Each worker adds and searches a per-run seed before the measured phase; connection and warmup durations are reported separately. Each measured iteration adds one distinct memory, searches it, reads a bounded page of 10, and probes connectivity. `--timeout` sets the per-tool deadline in seconds (default 30), including warmup and cleanup. JSON output gives count, MCP/transport/timeout error count, mean, p50 and p95 latency per tool, and aggregate throughput across the measured phase (excluding teardown and cleanup). No token, memory content, or server error text is printed. A warmup tool error stops the run rather than producing misleading metrics.

Integration coverage lives in `tests/integration/test_mcp_load_benchmark.py`; it runs the same harness against an in-process MCP server with real storage and deterministic embeddings, without a model download or external API.
