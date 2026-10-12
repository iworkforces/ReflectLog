"""Synchronous, deadline-bounded streamable-HTTP ReflectLog client.

SIZE_OK: the session/retry state machine stays together in the assigned module.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import itertools
import json
import socket
import threading
import time
from typing import TYPE_CHECKING, Protocol, TypeGuard, runtime_checkable

import httpx

if TYPE_CHECKING:
    from collections.abc import Sequence

from .models import (
    AddResult,
    DeleteResult,
    MemoryPage,
    MemoryRecord,
    ReflectLogError,
    SearchResult,
    parse_add_result,
    parse_delete_result,
    parse_get_result,
    parse_memory_page,
    parse_search_result,
)

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
REQUIRED_TOOLS = frozenset(
    {
        "add_memories",
        "search_memories",
        "get_all_memories",
        "get_memory_by_id",
        "delete_memories",
    }
)


def _mapping(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _object(value: object) -> dict[str, object]:
    if not _mapping(value):
        raise ReflectLogError("malformed", "Invalid MCP response object")
    return value


class _SessionInvalidError(Exception):
    pass


# JSON-RPC codes the server can only return before a tool runs.
_PRE_EXECUTION_RPC_CODES = frozenset({-32700, -32600, -32601, -32602})


class _RpcError(Exception):
    def __init__(self, definitive: bool, detail: str | None) -> None:
        super().__init__()
        self.definitive = definitive
        self.detail = detail


@runtime_checkable
class _NetworkStream(Protocol):
    def get_extra_info(self, info: str) -> object: ...


def _interrupt(response: httpx.Response) -> None:
    """Unblock a silent socket as well as a cooperatively closable test stream."""
    stream: object = response.extensions.get("network_stream")
    if isinstance(stream, _NetworkStream):
        connection = stream.get_extra_info("socket")
        if isinstance(connection, socket.socket):
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
    response.close()


@dataclass(frozen=True)
class _Session:
    generation: int
    identity: str
    protocol: str


class ReflectLogClient:
    def __init__(
        self,
        url: str,
        token: str,
        *,
        connect_timeout: float,
        request_timeout: float,
        transport: httpx.BaseTransport | None = None,
        client_name: str = "reflectlog-hermes",
        client_version: str = "0.1.0",
    ) -> None:
        self._url = url
        self._token = token
        self._connect_timeout = connect_timeout
        self._request_timeout = request_timeout
        self._name = client_name
        self._version = client_version
        self._http = httpx.Client(transport=transport, trust_env=False)
        self._lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._generation = 0
        self._session: _Session | None = None
        self._closing = False
        self._in_flight = 0

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ReflectLogError("timeout", "ReflectLog deadline exceeded")
        return remaining

    def _headers(self, session: _Session | None) -> dict[str, str]:
        headers = {
            "Authorization": "Bearer " + self._token,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if session:
            headers["mcp-session-id"] = session.identity
            headers["MCP-Protocol-Version"] = session.protocol
        return headers

    def _invalidate(self, generation: int) -> None:
        with self._lock:
            if self._generation == generation:
                self._generation += 1
                self._session = None

    def _message(self, data: bytes, identity: int) -> dict[str, object] | None:
        try:
            value: object = json.loads(data)
        except (ValueError, UnicodeError):
            raise ReflectLogError("malformed", "Invalid MCP JSON response") from None
        message = _object(value)
        if message.get("id") != identity:
            return None
        error = message.get("error")
        if error is not None:
            fields = _object(error)
            detail = fields.get("message")
            if detail == "Unauthorized":
                raise ReflectLogError(
                    "unauthorized", "ReflectLog authentication failed"
                )
            raise _RpcError(
                fields.get("code") in _PRE_EXECUTION_RPC_CODES,
                detail if isinstance(detail, str) else None,
            )
        return _object(message.get("result"))

    def _read(
        self, response: httpx.Response, identity: int, deadline: float
    ) -> dict[str, object]:
        timer = threading.Timer(self._remaining(deadline), _interrupt, (response,))
        timer.daemon = True
        timer.start()
        try:
            return self._read_stream(response, identity, deadline)
        except httpx.StreamError:
            # The watchdog can close the response before iteration starts.
            _ = self._remaining(deadline)
            raise ReflectLogError("malformed", "MCP stream closed early") from None
        finally:
            timer.cancel()

    def _read_stream(
        self, response: httpx.Response, identity: int, deadline: float
    ) -> dict[str, object]:
        sse = "text/event-stream" in response.headers.get("content-type", "")
        buffer = b""
        data: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            _ = self._remaining(deadline)
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise ReflectLogError("malformed", "MCP response exceeds size limit")
            buffer += chunk
            if not sse:
                continue
            while b"\n" in buffer:
                _ = self._remaining(deadline)
                line, buffer = buffer.split(b"\n", 1)
                line = line.rstrip(b"\r")
                if line.startswith(b"data:"):
                    value = line[5:]
                    data.append(value[1:] if value.startswith(b" ") else value)
                elif not line and data:
                    result = self._message(b"\n".join(data), identity)
                    data.clear()
                    if result is not None:
                        return result
        _ = self._remaining(deadline)
        if not sse:
            result = self._message(buffer, identity)
            if result is not None:
                return result
        raise ReflectLogError(
            "malformed", "MCP stream ended without a matching response"
        )

    def _rpc(
        self,
        method: str,
        params: dict[str, object],
        session: _Session | None,
        deadline: float,
        *,
        notification: bool = False,
    ) -> tuple[dict[str, object], str | None]:
        remaining = self._remaining(deadline)
        with self._lock:
            identity = next(self._ids)
        payload: dict[str, object] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        if not notification:
            payload["id"] = identity
        timeout = httpx.Timeout(
            min(self._request_timeout, remaining),
            connect=min(self._connect_timeout, remaining),
        )
        with self._http.stream(
            "POST",
            self._url,
            json=payload,
            headers=self._headers(session),
            timeout=timeout,
        ) as response:
            _ = self._remaining(deadline)
            if response.status_code in {401, 403}:
                raise ReflectLogError(
                    "unauthorized", "ReflectLog authentication failed"
                )
            if response.status_code == 404 and session:
                raise _SessionInvalidError
            if response.status_code == 400 and session:
                # A 400 is only safe to retry when it explicitly rejects the session.
                body = bytearray()
                for chunk in response.iter_bytes():
                    _ = self._remaining(deadline)
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ReflectLogError(
                            "malformed", "MCP response exceeds size limit"
                        )
                if b"session" in bytes(body).lower():
                    raise _SessionInvalidError
            if response.status_code >= 400:
                raise ReflectLogError("unavailable", "ReflectLog HTTP request failed")
            if notification and response.status_code in {200, 202, 204}:
                return {}, response.headers.get("mcp-session-id")
            return self._read(response, identity, deadline), response.headers.get(
                "mcp-session-id"
            )

    def _ensure_session(self, deadline: float) -> _Session:
        with self._lock:
            if self._session is not None:
                return self._session
        if not self._init_lock.acquire(timeout=self._remaining(deadline)):
            raise ReflectLogError(
                "timeout", "ReflectLog initialization deadline exceeded"
            )
        try:
            with self._lock:
                if self._session is not None:
                    return self._session
                if self._closing:
                    raise ReflectLogError("unavailable", "ReflectLog client is closed")
                generation = self._generation
            result, identity = self._rpc(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": self._name, "version": self._version},
                },
                None,
                deadline,
            )
            protocol = result.get("protocolVersion")
            if not identity or not isinstance(protocol, str):
                raise ReflectLogError("malformed", "Missing MCP session negotiation")
            session = _Session(generation, identity, protocol)
            _ = self._rpc(
                "notifications/initialized", {}, session, deadline, notification=True
            )
            names: set[str] = set()
            compatible = False
            params: dict[str, object] = {}
            cursors: set[str] = set()
            while True:
                listing, _ = self._rpc("tools/list", params, session, deadline)
                tools = listing.get("tools")
                if not _array(tools):
                    raise ReflectLogError("malformed", "Invalid MCP tool list")
                for tool in tools:
                    fields = _object(tool)
                    name = fields.get("name")
                    if isinstance(name, str):
                        names.add(name)
                    if name == "add_memories":
                        properties = _object(
                            _object(fields.get("inputSchema")).get("properties", {})
                        )
                        compatible = "allow_replace" in properties
                cursor = listing.get("nextCursor")
                if cursor is None:
                    break
                if not isinstance(cursor, str) or cursor in cursors:
                    raise ReflectLogError("malformed", "Invalid MCP pagination cursor")
                cursors.add(cursor)
                params = {"cursor": cursor}
            if not names >= REQUIRED_TOOLS or not compatible:
                raise ReflectLogError(
                    "incompatible", "ReflectLog tool contract is incompatible"
                )
            with self._lock:
                if self._closing or generation != self._generation:
                    raise ReflectLogError(
                        "unavailable", "ReflectLog initialization superseded"
                    )
                self._session = session
            return session
        finally:
            self._init_lock.release()

    def _call(
        self,
        tool: str,
        arguments: dict[str, object],
        deadline: float | None,
        *,
        mutation: bool = False,
    ) -> object:
        end = time.monotonic() + self._request_timeout if deadline is None else deadline
        with self._lock:
            if self._closing:
                raise ReflectLogError("unavailable", "ReflectLog client is closed")
            self._in_flight += 1
        try:
            for attempt in range(2):
                session: _Session | None = None
                with self._lock:
                    generation = self._generation
                dispatched = False
                try:
                    session = self._ensure_session(end)
                    _ = self._remaining(end)
                    dispatched = True
                    result, _ = self._rpc(
                        "tools/call",
                        {"name": tool, "arguments": arguments},
                        session,
                        end,
                    )
                    if result.get("isError") is True:
                        content = result.get("content")
                        text = (
                            "\n".join(
                                str(_object(item).get("text", "")) for item in content
                            )
                            if _array(content)
                            else ""
                        )
                        code = (
                            "outcome_unknown"
                            if mutation and "Error calling tool" in text
                            else "tool_error"
                        )
                        raise ReflectLogError(
                            code, "ReflectLog tool failed", server_detail=text
                        )
                    return _object(result.get("structuredContent"))
                except _SessionInvalidError:
                    self._invalidate(session.generation if session else generation)
                    if attempt:
                        raise ReflectLogError(
                            "unavailable", "ReflectLog session rejected"
                        ) from None
                except _RpcError as error:
                    raise ReflectLogError(
                        "outcome_unknown"
                        if mutation and dispatched and not error.definitive
                        else "tool_error",
                        "ReflectLog RPC failed",
                        server_detail=error.detail,
                    ) from None
                except httpx.RequestError as error:
                    presend = isinstance(
                        error,
                        (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout),
                    )
                    self._invalidate(session.generation if session else generation)
                    if mutation and dispatched and not presend:
                        raise ReflectLogError(
                            "outcome_unknown", "ReflectLog mutation response lost"
                        ) from None
                    if mutation or attempt:
                        code = (
                            "timeout"
                            if isinstance(error, httpx.TimeoutException)
                            else "unavailable"
                        )
                        raise ReflectLogError(
                            code, "ReflectLog transport failed"
                        ) from None
                except ReflectLogError as error:
                    if (
                        mutation
                        and dispatched
                        and error.code in {"timeout", "unavailable"}
                    ):
                        raise ReflectLogError(
                            "outcome_unknown", "ReflectLog mutation response lost"
                        ) from None
                    if mutation and dispatched and error.code == "malformed":
                        raise ReflectLogError(
                            "outcome_unknown", "ReflectLog mutation response invalid"
                        ) from None
                    raise
            raise ReflectLogError("unavailable", "ReflectLog request failed")
        finally:
            with self._lock:
                self._in_flight -= 1
                close = self._closing and self._in_flight == 0
            if close:
                self._http.close()

    def add_memories(
        self,
        workspace_id: str,
        memories: Sequence[str],
        *,
        dry_run: bool = False,
        allow_replace: bool = False,
        deadline: float | None = None,
    ) -> AddResult:
        return parse_add_result(
            self._call(
                "add_memories",
                {
                    "workspace_id": workspace_id,
                    "memories": list(memories),
                    "dry_run": dry_run,
                    "allow_replace": allow_replace,
                },
                deadline,
                mutation=not dry_run,
            )
        )

    def search_memories(
        self,
        workspace_id: str,
        query: str,
        *,
        limit: int | None = None,
        deadline: float | None = None,
    ) -> SearchResult:
        return parse_search_result(
            self._call(
                "search_memories",
                {"workspace_id": workspace_id, "query": query, "limit": limit},
                deadline,
            )
        )

    def get_all_memories(
        self,
        workspace_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        deadline: float | None = None,
    ) -> MemoryPage:
        return parse_memory_page(
            self._call(
                "get_all_memories",
                {"workspace_id": workspace_id, "limit": limit, "offset": offset},
                deadline,
            )
        )

    def get_memory_by_id(
        self, workspace_id: str, memory_id: int, *, deadline: float | None = None
    ) -> MemoryRecord | None:
        return parse_get_result(
            self._call(
                "get_memory_by_id",
                {"workspace_id": workspace_id, "id": memory_id},
                deadline,
            )
        )

    def delete_memories(
        self, workspace_id: str, ids: Sequence[int], *, deadline: float | None = None
    ) -> DeleteResult:
        return parse_delete_result(
            self._call(
                "delete_memories",
                {"workspace_id": workspace_id, "ids": list(ids)},
                deadline,
                mutation=True,
            )
        )

    def probe(self, workspace_id: str, *, deadline: float | None = None) -> int:
        return self.get_all_memories(workspace_id, limit=0, deadline=deadline).total

    def close(self, budget_s: float) -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
            session = self._session
            active = self._in_flight > 0
        if active:
            # Closing the pool/socket would interrupt a possibly committed mutation.
            return
        try:
            if session and budget_s > 0:
                with self._http.stream(
                    "DELETE",
                    self._url,
                    headers=self._headers(session),
                    timeout=min(budget_s, self._request_timeout),
                ):
                    pass
        except httpx.HTTPError:
            pass
        finally:
            self._http.close()
