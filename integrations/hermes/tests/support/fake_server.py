# pyright: basic
"""Deterministic MockTransport handler.

FakeReflectLogServer(token) is callable. expire_sessions() invalidates sessions;
faults[tool] queues connect/read_timeout/read_error/heartbeat/session400/session404,
validation/execution/malformed/mismatch/missing faults, plus rpc_params (JSON-RPC
-32602 before running) and rpc_internal (-32603 after running). session400/404 use the
MCP SDK's pre-dispatch body; audit400 (a 400 after running), noid400/version400
(-32600 session errors without an explicit null id / with the wrong jsonrpc after
running), nested400/big404 (unparsable / oversized session error bodies after
running), stall404 (a 404 body that ends only when closed, after running),
nested (an unparsable 200 body), bad_result (unreadable structuredContent
after running) and slow_headers (slow_s before any response after running) model
post-dispatch failures; stall answers with a body that ends only when closed, and
close_delay/close_error make that close slow or raise; delete_delay slows session
DELETE.
framing selects json/plain/comments/multiline. drop_tool and allow_replace configure
compatibility; requests, dispatches, handshakes and deletes expose receipts.
entered/release optionally gate tools/call for event-based concurrency tests. All
state is guarded by an RLock.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
import re
import threading
import time

import httpx

TOOLS = (
    "add_memories",
    "search_memories",
    "get_all_memories",
    "get_memory_by_id",
    "delete_memories",
)


class Heartbeats(httpx.SyncByteStream):
    def __iter__(self):
        while True:
            yield b": heartbeat\n\n"


class Fragments(httpx.SyncByteStream):
    def __init__(self, data):
        self.data = data.encode()

    def __iter__(self):
        for offset in range(0, len(self.data), 7):
            yield self.data[offset : offset + 7]


class Stalled(httpx.SyncByteStream):
    def __init__(self, server):
        self.server = server
        self.closed = threading.Event()

    def __iter__(self):
        self.closed.wait(5)
        yield from ()

    def close(self):
        self.closed.set()
        time.sleep(self.server.close_delay)
        if self.server.close_error:
            raise OSError("private close failure")


class FakeReflectLogServer:
    def __init__(self, token="qa-token"):
        self.token = token
        self.framing = "comments"
        self.drop_tool = None
        self.allow_replace = True
        self.paginate = True
        self.auth_status = 200
        self.sessions = set()
        self.handshakes = 0
        self.deletes = 0
        self.requests = []
        self.dispatches = Counter()
        self.calls = Counter()
        self.faults = defaultdict(deque)
        self.store = defaultdict(dict)
        self.next_id = 1
        self.entered = None
        self.release = None
        self.slow_s = 1.0
        self.delete_delay = 0.0
        self.close_delay = 0.0
        self.close_error = False
        self.lock = threading.RLock()

    def expire_sessions(self):
        with self.lock:
            self.sessions.clear()

    @staticmethod
    def session_error(status, message):
        # mcp.server.streamable_http rejects sessions before dispatching anything.
        return httpx.Response(
            status,
            json={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": message},
            },
        )

    def response(self, request, value, headers=None):
        if self.framing == "json":
            return httpx.Response(200, json=value, headers=headers)
        data = json.dumps(value, indent=2 if self.framing == "multiline" else None)
        prefix = ": heartbeat\n\nevent: message\n" if self.framing == "comments" else ""
        body = (
            prefix + "\n".join("data: " + line for line in data.splitlines()) + "\n\n"
        )
        if self.framing == "fragmented":
            noise = 'data: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
            noise += 'data: {"jsonrpc":"2.0","id":-123,"result":{}}\n\n'
            return httpx.Response(
                200,
                stream=Fragments(noise + body),
                headers={
                    "content-type": "text/event-stream",
                    **(headers or {}),
                },
            )
        return httpx.Response(
            200,
            text=body,
            headers={
                "content-type": "text/event-stream",
                **(headers or {}),
            },
        )

    def __call__(self, request):
        if request.method == "DELETE":
            with self.lock:
                self.requests.append(request)
                self.deletes += 1
                self.sessions.discard(request.headers.get("mcp-session-id"))
            time.sleep(self.delete_delay)
            return httpx.Response(200)
        with self.lock:
            self.requests.append(request)
            body = json.loads(request.content)
            method = body["method"]
            tool = body.get("params", {}).get("name", "")
            if method == "tools/call":
                self.calls[tool] += 1
            fault = self.faults[tool].popleft() if self.faults[tool] else None
            if fault == "connect":
                raise httpx.ConnectError("private connection detail", request=request)
            if request.headers.get("authorization") != "Bearer " + self.token:
                if self.auth_status != 200:
                    return httpx.Response(self.auth_status)
                return self.response(
                    request,
                    {
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "error": {"code": 0, "message": "Unauthorized"},
                    },
                )
            if method == "initialize":
                self.handshakes += 1
                session = "session-" + str(self.handshakes)
                self.sessions.add(session)
                return self.response(
                    request,
                    {
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "serverInfo": {"name": "fake", "version": "1"},
                        },
                    },
                    {"mcp-session-id": session},
                )
            session = request.headers.get("mcp-session-id")
            if not session or fault == "session400":
                return self.session_error(400, "Bad Request: Missing session ID")
            if session not in self.sessions or fault == "session404":
                return self.session_error(404, "Session not found")
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "tools/list":
                tools = [
                    {
                        "name": name,
                        "inputSchema": {
                            "type": "object",
                            "properties": {"allow_replace": {"type": "boolean"}}
                            if self.allow_replace
                            else {},
                        },
                    }
                    for name in TOOLS
                    if name != self.drop_tool
                ]
                second = body.get("params", {}).get("cursor") == "second"
                result: dict[str, object] = {
                    "tools": tools[2:]
                    if second
                    else tools[:2]
                    if self.paginate
                    else tools
                }
                if self.paginate and not second:
                    result["nextCursor"] = "second"
                return self.response(
                    request, {"jsonrpc": "2.0", "id": body["id"], "result": result}
                )
            self.dispatches[tool] += 1
            args = body["params"]["arguments"]
            if not self.valid(tool, args):
                fault = "validation"
            if fault in {"validation", "execution"}:
                text = (
                    "1 validation error: invalid arguments"
                    if fault == "validation"
                    else f"Error calling tool '{tool}': boom"
                )
                result = {"isError": True, "content": [{"type": "text", "text": text}]}
            elif fault == "rpc_params":
                result = {}
            elif fault == "bad_result":
                self.execute(tool, args)
                result = {"structuredContent": {"dry_run": False}, "isError": False}
            else:
                result = {
                    "structuredContent": self.execute(tool, args),
                    "isError": False,
                }
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait(5)
        if fault == "slow_headers":
            time.sleep(self.slow_s)
        if fault == "audit400":
            return httpx.Response(400, text="Session audit failed after applying it")
        if fault in {"noid400", "version400"}:
            forged = {
                "jsonrpc": "1.0" if fault == "version400" else "2.0",
                "error": {
                    "code": -32600,
                    "message": "Session audit failed after applying it",
                },
            }
            if fault == "version400":
                forged["id"] = None
            return httpx.Response(400, json=forged)
        if fault == "nested400":
            return httpx.Response(400, text="[" * 100_000)
        if fault == "big404":
            return httpx.Response(404, text="x" * 8192)
        if fault == "stall404":
            return httpx.Response(404, stream=Stalled(self))
        if fault == "nested":
            return httpx.Response(
                200, text="[" * 100_000, headers={"content-type": "application/json"}
            )
        if fault == "stall":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Stalled(self),
            )
        if fault == "read_timeout":
            raise httpx.ReadTimeout("lost private response", request=request)
        if fault == "read_error":
            raise httpx.ReadError("lost private response", request=request)
        if fault == "heartbeat":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=Heartbeats()
            )
        if fault == "malformed":
            return httpx.Response(
                200, text="{broken", headers={"content-type": "application/json"}
            )
        if fault in {"rpc_params", "rpc_internal"}:
            return self.response(
                request,
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "error": {
                        "code": -32602 if fault == "rpc_params" else -32603,
                        "message": "private rpc detail",
                    },
                },
            )
        if fault == "missing":
            result = {}
        return self.response(
            request,
            {
                "jsonrpc": "2.0",
                "id": -1 if fault == "mismatch" else body["id"],
                "result": result,
            },
        )

    def valid(self, tool, args):
        workspace = args.get("workspace_id")
        if (
            not isinstance(workspace, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", workspace)
            or workspace == "."
            or ".." in workspace
        ):
            return False
        if tool == "get_memory_by_id":
            identity = args.get("id")
            return type(identity) is int and identity > 0
        if tool == "delete_memories":
            ids = args.get("ids")
            return (
                isinstance(ids, list)
                and len(ids) <= 100
                and all(type(i) is int and i > 0 for i in ids)
            )
        if tool == "search_memories":
            query, limit = args.get("query"), args.get("limit")
            return (
                isinstance(query, str)
                and bool(query.strip())
                and len(query) <= 1000
                and (limit is None or type(limit) is int and 1 <= limit <= 50)
            )
        if tool == "add_memories":
            memories = args.get("memories")
            return (
                isinstance(memories, list)
                and len(memories) <= 100
                and all(
                    isinstance(m, str) and bool(m.strip()) and len(m) <= 30720
                    for m in memories
                )
                and sum(len(m) for m in memories) <= 500000
            )
        return True

    def execute(self, tool, args):
        records = self.store[args["workspace_id"].lower()]
        if tool == "add_memories":
            stored = skipped = 0
            for memory in args["memories"]:
                if any(record["memory"] == memory for record in records.values()):
                    skipped += 1
                else:
                    stored += 1
                    if not args.get("dry_run", False):
                        records[self.next_id] = {
                            "id": self.next_id,
                            "memory": memory,
                            "created_at": "2026-01-01T00:00:00+00:00",
                        }
                        self.next_id += 1
            return {
                "stored_count": stored,
                "skipped_count": skipped,
                "replaced_count": 0,
                "replacements": [],
                "dry_run": args.get("dry_run", False),
            }
        if tool == "delete_memories":
            removed, missing = [], []
            for identity in dict.fromkeys(args["ids"]):
                (removed if identity in records else missing).append(identity)
                records.pop(identity, None)
            return {"removed": removed, "missing": missing}
        if tool == "get_memory_by_id":
            return {"record": records.get(args["id"])}
        limit = args.get("limit")
        if tool == "search_memories":
            limit = limit or 10
            terms = args["query"].lower().split()
            hits = [
                {**record, "score": 1.0}
                for record in records.values()
                if any(term in record["memory"].lower() for term in terms)
            ]
            return {"results": hits[:limit], "score_kind": "fulltext", "limit": limit}
        offset = args.get("offset", 0)
        limit = len(records) if limit is None else limit
        page = list(records.values())[offset : offset + limit]
        return {
            "records": page,
            "total": len(records),
            "offset": offset,
            "limit": limit,
            "truncated": offset + len(page) < len(records),
        }
