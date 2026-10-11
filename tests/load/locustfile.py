import argparse
from collections import defaultdict
from contextlib import AsyncExitStack
import json
from math import ceil, isfinite
import os
from statistics import mean
from time import perf_counter
from typing import TypedDict
from uuid import uuid4

import anyio
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel

from tests.test_get_all_memories import (
    AddResponse,
    DeleteResponse,
    GetResponse,
    Page,
    SearchResponse,
    all_records,
    parse_response,
)

RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "add_memories": AddResponse,
    "search_memories": SearchResponse,
    "get_all_memories": Page,
    "get_memory_by_id": GetResponse,
    "delete_memories": DeleteResponse,
}
TOOLS = tuple(RESPONSE_MODELS)


class ToolMetrics(TypedDict):
    count: int
    errors: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


class BenchmarkReport(TypedDict):
    workspace_id: str
    clients: int
    iterations: int
    connection_ms: list[float]
    warmup_ms: list[float]
    measured_seconds: float
    requests_per_second: float
    cleanup_errors: int
    tools: dict[str, ToolMetrics]


async def run_benchmark(
    target: str | FastMCP,
    workspace_id: str,
    clients: int,
    iterations: int,
    token: str | None = None,
    timeout_seconds: float = 30.0,
) -> BenchmarkReport:
    if (
        not 1 <= clients <= 32
        or iterations < 1
        or not (isfinite(timeout_seconds) and timeout_seconds > 0)
    ):
        raise ValueError("clients must be 1..32, iterations and timeout positive")

    ready = [anyio.Event() for _ in range(clients)]
    finished = [anyio.Event() for _ in range(clients)]
    start = anyio.Event()
    cleanup_start = anyio.Event()
    run_id = uuid4().hex
    samples: dict[str, list[float]] = defaultdict(list)
    failures: dict[str, int] = defaultdict(int)
    connection_ms: list[float] = []
    warmup_ms: list[float] = []
    cleanup_errors = 0
    warmup_lock = anyio.Lock()

    async def worker(worker_id: int, client: Client) -> None:
        nonlocal cleanup_errors
        prefix = f"reflectlog benchmark {workspace_id} run {run_id} worker {worker_id}"
        seed = f"{prefix} seed"
        memories = [seed, *(f"{prefix} item {i}" for i in range(iterations))]

        async def measured(name: str, args: dict[str, object]) -> BaseModel | None:
            started_at = perf_counter()
            try:
                with anyio.fail_after(timeout_seconds):
                    result = await client.call_tool(name, args, raise_on_error=False)
                response = parse_response(result, RESPONSE_MODELS[name])
                if isinstance(response, GetResponse) and response.record is None:
                    raise ValueError("Benchmark record missing")
                if isinstance(response, DeleteResponse) and response.missing:
                    raise ValueError("Benchmark deletion missed records")
                return response
            except Exception:
                failures[name] += 1
                return None
            finally:
                samples[name].append((perf_counter() - started_at) * 1000)

        try:
            async with warmup_lock:
                warmed_at = perf_counter()
                warmup_calls: tuple[tuple[str, dict[str, object]], ...] = (
                    (
                        "add_memories",
                        {
                            "workspace_id": workspace_id,
                            "memories": [seed],
                            "allow_replace": False,
                        },
                    ),
                    ("search_memories", {"workspace_id": workspace_id, "query": seed}),
                    ("get_all_memories", {"workspace_id": workspace_id, "limit": 0}),
                )
                for name, args in warmup_calls:
                    with anyio.fail_after(timeout_seconds):
                        result = await client.call_tool(
                            name, args, raise_on_error=False
                        )
                    parse_response(result, RESPONSE_MODELS[name])
                with anyio.fail_after(timeout_seconds):
                    seeds = await all_records(client, workspace_id)
                seed_id = next(row.id for row in seeds if row.memory == seed)
                warmup_ms.append((perf_counter() - warmed_at) * 1000)
            ready[worker_id].set()
            await start.wait()

            for iteration in range(iterations):
                memory = memories[iteration + 1]
                await measured(
                    "add_memories",
                    {
                        "workspace_id": workspace_id,
                        "memories": [memory],
                        "allow_replace": False,
                    },
                )
                searched = await measured(
                    "search_memories",
                    {"workspace_id": workspace_id, "query": memory, "limit": 50},
                )
                page = await measured(
                    "get_all_memories", {"workspace_id": workspace_id, "limit": 10}
                )
                await measured(
                    "get_all_memories", {"workspace_id": workspace_id, "limit": 0}
                )
                candidates = (
                    searched.results if isinstance(searched, SearchResponse) else []
                )
                ids = [row.id for row in candidates if row.memory == memory]
                if not ids and isinstance(page, Page):
                    ids = [row.id for row in page.records if row.memory == memory]
                await measured(
                    "get_memory_by_id",
                    {"workspace_id": workspace_id, "id": ids[0] if ids else seed_id},
                )
                await measured(
                    "delete_memories", {"workspace_id": workspace_id, "ids": ids}
                )
                if not ids:
                    failures["get_memory_by_id"] += 1
                    failures["delete_memories"] += 1
            finished[worker_id].set()
            await cleanup_start.wait()
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    with anyio.fail_after(timeout_seconds):
                        records = await all_records(client, workspace_id)
                        ids = [row.id for row in records if row.memory in memories]
                        for offset in range(0, len(ids), 100):
                            batch = ids[offset : offset + 100]
                            deleted = parse_response(
                                await client.call_tool(
                                    "delete_memories",
                                    {"workspace_id": workspace_id, "ids": batch},
                                    raise_on_error=False,
                                ),
                                DeleteResponse,
                            )
                            if set(deleted.removed) != set(batch) or deleted.missing:
                                raise ValueError(
                                    "Benchmark cleanup did not remove its records"
                                )
                except Exception:
                    cleanup_errors += 1

    elapsed = 0.0
    client_stack = AsyncExitStack()
    try:
        connected_clients: list[Client] = []
        for _ in range(clients):
            transport = (
                StreamableHttpTransport(
                    target,
                    headers={"Authorization": f"Bearer {token}"} if token else None,
                )
                if isinstance(target, str)
                else target
            )
            connected_at = perf_counter()
            connected_clients.append(
                await client_stack.enter_async_context(Client(transport))
            )
            connection_ms.append((perf_counter() - connected_at) * 1000)
        async with anyio.create_task_group() as group:
            for worker_id, client in enumerate(connected_clients):
                group.start_soon(worker, worker_id, client)
            for event in ready:
                await event.wait()
            measured_at = perf_counter()
            start.set()
            for event in finished:
                await event.wait()
            elapsed = perf_counter() - measured_at
            cleanup_start.set()
    except BaseException as error:
        if cleanup_errors:
            error.add_note(f"benchmark cleanup_errors={cleanup_errors}")
        raise
    finally:
        with anyio.CancelScope(shield=True):
            await client_stack.aclose()

    total = sum(map(len, samples.values()))
    return {
        "workspace_id": workspace_id,
        "clients": clients,
        "iterations": iterations,
        "connection_ms": connection_ms,
        "warmup_ms": warmup_ms,
        "measured_seconds": elapsed,
        "requests_per_second": total / elapsed,
        "cleanup_errors": cleanup_errors,
        "tools": {
            name: {
                "count": len(samples[name]),
                "errors": failures[name],
                "mean_ms": mean(samples[name]),
                "p50_ms": sorted(samples[name])[ceil(0.50 * len(samples[name])) - 1],
                "p95_ms": sorted(samples[name])[ceil(0.95 * len(samples[name])) - 1],
            }
            for name in TOOLS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ReflectLog MCP tools")
    parser.add_argument("--url", default="http://127.0.0.1:9103/mcp")
    parser.add_argument(
        "--clients", type=int, default=1, help="Concurrent clients (1..32)"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Workload loops per client"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Seconds per MCP tool call"
    )
    parser.add_argument(
        "--workspace-id",
        default=None,
        help="Isolated workspace (default: unique bench ID)",
    )
    args = parser.parse_args()
    if (
        not 1 <= args.clients <= 32
        or args.iterations < 1
        or not (isfinite(args.timeout) and args.timeout > 0)
    ):
        parser.error("clients must be 1..32, iterations and timeout positive")
    workspace_id = args.workspace_id or f"bench-{uuid4().hex}"
    token = os.environ.get("REFLECTLOG_MCP_TOKEN")
    result = anyio.run(
        run_benchmark,
        args.url,
        workspace_id,
        args.clients,
        args.iterations,
        token,
        args.timeout,
    )
    print(json.dumps(result, indent=2))
    if result["cleanup_errors"]:
        parser.exit(1, "benchmark cleanup failed; see cleanup_errors in report\n")


if __name__ == "__main__":
    main()
