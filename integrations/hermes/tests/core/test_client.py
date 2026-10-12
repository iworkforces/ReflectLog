# pyright: basic
"""Wire-level client regression tests (no network)."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import httpx
import pytest


@pytest.fixture
def error_type(plugin_import):
    return plugin_import("core.models").ReflectLogError


@pytest.fixture
def wire(plugin_import):
    server = plugin_import("tests.support.fake_server").FakeReflectLogServer()
    module = plugin_import("core.client")
    client = module.ReflectLogClient(
        "http://fake/mcp",
        server.token,
        connect_timeout=1,
        request_timeout=2,
        transport=httpx.MockTransport(server),
    )
    yield server, client
    client.close(0.1)


@pytest.mark.parametrize("framing", ["plain", "comments", "multiline", "json"])
def test_handshake_and_response(wire, framing):
    # given
    server, client = wire
    server.framing = framing
    # when
    total = client.probe("ws")
    # then
    assert total == 0
    assert server.handshakes == 1
    assert server.requests[1].headers["mcp-session-id"] == "session-1"
    assert server.requests[-1].headers["mcp-protocol-version"] == "2025-06-18"
    assert (
        server.requests[-1].headers["accept"] == "application/json, text/event-stream"
    )
    assert (
        json.loads(server.requests[1].content)["method"] == "notifications/initialized"
    )


@pytest.mark.parametrize("status", [200, 401, 403])
def test_auth_is_safe(wire, status, caplog, error_type):
    # given
    server, client = wire
    server.token = "different-secret"
    server.auth_status = status
    # when
    with pytest.raises(error_type) as error:
        client.add_memories("ws", ["private-memory"])
    # then
    assert error.value.code == "unauthorized"
    assert "qa-token" not in str(error.value) + caplog.text
    assert "private-memory" not in str(error.value) + caplog.text


@pytest.mark.parametrize("drop", [True, False])
def test_incompatible(wire, drop, error_type):
    # given
    server, client = wire
    server.drop_tool = "search_memories" if drop else None
    server.allow_replace = drop
    # when
    with pytest.raises(error_type) as error:
        client.probe("ws")
    # then
    assert error.value.code == "incompatible"


@pytest.mark.parametrize(
    "fault",
    [
        "session400",
        "session404",
        "audit400",
        "noid400",
        "nested400",
        "read_error",
        "read_timeout",
    ],
)
def test_reads_retry_once(wire, fault):
    # given
    server, client = wire
    server.faults["get_all_memories"].append(fault)
    # when
    result = client.probe("ws")
    # then
    assert result == 0
    assert server.handshakes == 2


def test_second_session_failure(wire, error_type):
    # given
    server, client = wire
    server.faults["get_all_memories"].extend(["session404", "session404"])
    # when
    with pytest.raises(error_type) as error:
        client.probe("ws")
    # then
    assert error.value.code == "unavailable"
    assert server.handshakes == 2


@pytest.mark.parametrize(
    "fault,code,count",
    [
        ("read_error", "outcome_unknown", 1),
        ("read_timeout", "outcome_unknown", 1),
        ("connect", "unavailable", 0),
        ("session404", None, 1),
        ("session400", None, 1),
        ("audit400", "outcome_unknown", 1),
        ("noid400", "outcome_unknown", 1),
        ("version400", "outcome_unknown", 1),
        ("nested400", "outcome_unknown", 1),
        ("nested", "outcome_unknown", 1),
        ("bad_result", "outcome_unknown", 1),
        ("validation", "tool_error", 1),
        ("execution", "outcome_unknown", 1),
        ("rpc_params", "tool_error", 1),
        ("rpc_internal", "outcome_unknown", 1),
    ],
)
def test_mutation_dispatch_policy(wire, fault, code, count, error_type):
    # given
    server, client = wire
    server.faults["add_memories"].append(fault)
    # when
    if code:
        with pytest.raises(error_type) as error:
            client.add_memories("ws", ["private-memory"])
        assert error.value.code == code
        assert "private-memory" not in str(error.value)
    else:
        assert client.add_memories("ws", ["memory"]).stored_count == 1
    # then
    assert server.dispatches["add_memories"] == count


@pytest.mark.parametrize("fault", ["execution", "rpc_internal"])
def test_read_execution_error(wire, fault, error_type):
    # given
    server, client = wire
    server.faults["search_memories"].append(fault)
    # when
    with pytest.raises(error_type) as error:
        client.search_memories("ws", "query")
    # then
    assert error.value.code == "tool_error"


@pytest.mark.parametrize("fault", ["malformed", "mismatch", "missing", "nested"])
def test_malformed(wire, fault, error_type):
    # given
    server, client = wire
    server.faults["get_all_memories"].append(fault)
    # when
    with pytest.raises(error_type) as error:
        client.probe("ws")
    # then
    assert error.value.code == "malformed"


@pytest.mark.parametrize("mutation", [True, False])
def test_heartbeat_deadline(wire, mutation, error_type):
    # given
    server, client = wire
    tool = "add_memories" if mutation else "get_all_memories"
    server.faults[tool].append("heartbeat")
    start = time.monotonic()
    # when
    with pytest.raises(error_type) as error:
        if mutation:
            client.add_memories("ws", ["text"], deadline=start + 0.1)
        else:
            client.probe("ws", deadline=start + 0.1)
    # then
    assert error.value.code == ("outcome_unknown" if mutation else "timeout")
    assert time.monotonic() - start < 0.5


@pytest.mark.parametrize("mutation", [True, False])
def test_slow_headers_deadline(wire, mutation, error_type):
    # given
    server, client = wire
    client.probe("ws")
    tool = "add_memories" if mutation else "get_all_memories"
    server.faults[tool].append("slow_headers")
    before = server.dispatches[tool]
    start = time.monotonic()
    # when
    with pytest.raises(error_type) as error:
        if mutation:
            client.add_memories("ws", ["text"], deadline=start + 0.2)
        else:
            client.probe("ws", deadline=start + 0.2)
    # then
    assert error.value.code == ("outcome_unknown" if mutation else "timeout")
    assert time.monotonic() - start < 0.5
    assert server.dispatches[tool] == before + 1


@pytest.mark.parametrize("mode", ["block", "raise"])
@pytest.mark.parametrize("mutation", [True, False])
def test_timeout_cleanup_never_delays_the_caller(wire, mode, mutation, error_type):
    # given a body that ends only when closed, and a close that blocks or raises
    server, client = wire
    client.probe("ws")
    server.close_delay = 0.4 if mode == "block" else 0.0
    server.close_error = mode == "raise"
    tool = "add_memories" if mutation else "get_all_memories"
    server.faults[tool].append("stall")
    start = time.monotonic()
    # when
    with pytest.raises(error_type) as error:
        if mutation:
            client.add_memories("ws", ["text"], deadline=start + 0.1)
        else:
            client.probe("ws", deadline=start + 0.1)
    # then
    assert error.value.code == ("outcome_unknown" if mutation else "timeout")
    assert time.monotonic() - start < 0.35


def test_unreadable_session_error_body(wire, error_type, monkeypatch, plugin_import):
    # given 404 bodies too large to read, after the tool ran
    server, client = wire
    client.probe("ws")
    monkeypatch.setattr(plugin_import("core.client"), "MAX_RESPONSE_BYTES", 4096)
    server.faults["get_all_memories"].append("big404")
    server.faults["add_memories"].append("big404")
    # when
    total = client.probe("ws")
    with pytest.raises(error_type) as error:
        client.add_memories("ws", ["memory"])
    # then the read re-handshakes and the mutation is never resent
    assert total == 0
    assert server.handshakes == 2
    assert error.value.code == "outcome_unknown"
    assert server.dispatches["add_memories"] == 1


def test_unreadable_results(wire, error_type):
    # given
    server, client = wire
    client.add_memories("ws", ["Alpha"])
    record = client.get_all_memories("ws").records[0]
    server.faults["delete_memories"].append("bad_result")
    server.faults["add_memories"].append("bad_result")
    # when
    with pytest.raises(error_type) as deleted:
        client.delete_memories("ws", [record.id])
    with pytest.raises(error_type) as preview:
        client.add_memories("ws", ["Beta"], dry_run=True)
    # then
    assert deleted.value.code == "outcome_unknown"
    assert server.dispatches["delete_memories"] == 1
    assert preview.value.code == "malformed"


def test_concurrent_initialization(wire):
    # given
    server, client = wire
    barrier = threading.Barrier(8)

    def call(_):
        barrier.wait()
        return client.probe("ws")

    # when
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(call, range(8)))
    # then
    assert results == [0] * 8
    assert server.handshakes == 1


def test_stale_invalidation(wire):
    # given
    server, client = wire
    client.probe("ws")
    generation = client._generation
    server.expire_sessions()
    client.probe("ws")
    # when
    client._invalidate(generation)
    client.probe("ws")
    # then
    assert server.handshakes == 2


def test_idle_close(wire, error_type):
    # given
    server, client = wire
    client.probe("ws")
    # when
    client.close(0.1)
    # then
    assert server.deletes == 1
    with pytest.raises(error_type) as error:
        client.probe("ws")
    assert error.value.code == "unavailable"


def test_close_is_bounded(wire):
    # given
    server, client = wire
    client.probe("ws")
    server.delete_delay = 1.0
    start = time.monotonic()
    # when
    client.close(0.1)
    # then
    assert time.monotonic() - start < 0.4
    assert server.deletes == 1


def test_active_close(wire):
    # given
    server, client = wire
    server.entered = threading.Event()
    server.release = threading.Event()
    # when
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(client.probe, "ws")
        assert server.entered.wait(2)
        client.close(0.1)
        server.release.set()
        assert future.result(2) == 0
    # then
    assert server.deletes == 0


def test_crud_and_dedup(wire):
    # given
    _, client = wire
    client.add_memories("ws", ["Alpha"])
    # when
    duplicate = client.add_memories("ws", ["Alpha"])
    # then
    assert duplicate.skipped_count == 1
    assert client.search_memories("ws", "alpha").hits[0].memory == "Alpha"
    record = client.get_all_memories("ws").records[0]
    assert client.get_memory_by_id("ws", record.id) == record
    assert client.delete_memories("ws", [record.id]).removed == (record.id,)


def test_dry_run_retries_as_read(wire):
    # given
    server, client = wire
    server.faults["add_memories"].append("read_error")
    # when
    result = client.add_memories("ws", ["preview"], dry_run=True)
    # then
    assert result.dry_run
    assert server.dispatches["add_memories"] == 2
    assert client.probe("ws") == 0


def test_invalid_arguments(wire, error_type):
    # given
    _, client = wire
    # when
    with pytest.raises(error_type) as error:
        client.get_memory_by_id("ws", -1)
    # then
    assert error.value.code == "tool_error"


def test_size_cap(wire, error_type, monkeypatch, plugin_import):
    # given
    _, client = wire
    client.probe("ws")
    monkeypatch.setattr(plugin_import("core.client"), "MAX_RESPONSE_BYTES", 32)
    # when
    with pytest.raises(error_type) as error:
        client.probe("ws")
    # then
    assert error.value.code == "malformed"


def test_oversized_mutation_response_is_unknown(
    wire, error_type, monkeypatch, plugin_import
):
    # given
    server, client = wire
    client.probe("ws")
    monkeypatch.setattr(plugin_import("core.client"), "MAX_RESPONSE_BYTES", 32)
    # when
    with pytest.raises(error_type) as error:
        client.add_memories("ws", ["memory"])
    # then
    assert error.value.code == "outcome_unknown"
    assert server.dispatches["add_memories"] == 1


def test_init_mutex_deadline(wire, error_type):
    # given
    _, client = wire
    client._init_lock.acquire()
    # when
    try:
        with pytest.raises(error_type) as error:
            client.probe("ws", deadline=time.monotonic() + 0.05)
    finally:
        client._init_lock.release()
    # then
    assert error.value.code == "timeout"


def test_sse_ignores_notifications_and_other_ids(wire):
    # given
    server, client = wire
    server.framing = "fragmented"
    # when
    result = client.probe("ws")
    # then
    assert result == 0
