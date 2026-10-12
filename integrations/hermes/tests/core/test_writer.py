# pyright: basic
from __future__ import annotations

from threading import Event, Thread
import time

import pytest


def thread_factory(target, name):
    return Thread(target=target, name=name, daemon=True)


def job(models, number=1):
    return models.WriteJob("owner", (str(number),), "mirror")


def test_fifo_and_drain(plugin_import, models):
    # Given a single writer, when multiple jobs are queued, then successful submission is FIFO.
    submitted = []
    writer = plugin_import("core.writer").BackgroundWriter(
        submitted.append, thread_factory=thread_factory
    )
    try:
        for number in range(10):
            assert writer.enqueue(job(models, number))
        assert writer.drain(2)
        assert submitted == [job(models, number) for number in range(10)]
        assert writer.stats()["completed"] == writer.stats()["enqueued"] == 10
    finally:
        writer.shutdown(1)


def test_capacity_drop_counted(plugin_import, models):
    # Given an in-flight job and one queued slot, when overflowing, then no queued job is displaced.
    entered, release = Event(), Event()

    def submit(item):
        entered.set()
        assert release.wait(2)

    writer = plugin_import("core.writer").BackgroundWriter(
        submit, thread_factory=thread_factory, capacity=1
    )
    try:
        assert writer.enqueue(job(models)) and entered.wait(2)
        assert writer.enqueue(job(models, 2))
        assert not writer.enqueue(job(models, 3))
        assert writer.stats()["dropped"] == 1
        assert not writer.drain(0)
    finally:
        release.set()
        writer.shutdown(1)


@pytest.mark.parametrize("code", ["unavailable", "timeout", "outcome_unknown"])
def test_breaker_cooldown_and_success_reset(plugin_import, models, code):
    # Given a fake clock and consecutive retry-unsafe/transient failures, open then half-open.
    now = [10.0]
    fail = [True]

    def submit(item):
        if fail[0]:
            raise models.ReflectLogError(code, "safe")

    writer = plugin_import("core.writer").BackgroundWriter(
        submit,
        thread_factory=thread_factory,
        failure_threshold=2,
        cooldown_s=5,
        clock=lambda: now[0],
    )
    try:
        for _ in range(2):
            assert writer.enqueue(job(models)) and writer.drain(2)
        # When the breaker opens, then enqueue drops and expiry admits a probe.
        assert writer.breaker_open() and not writer.enqueue(job(models))
        now[0] = 15
        assert not writer.breaker_open()
        fail[0] = False
        assert writer.enqueue(job(models)) and writer.drain(2)
        assert not writer.breaker_open() and writer.stats()["completed"] == 1
        assert writer.stats()["unknown"] == (2 if code == "outcome_unknown" else 0)
        fail[0] = True
        assert writer.enqueue(job(models)) and writer.drain(2)
        assert not writer.breaker_open()
    finally:
        writer.shutdown(1)


@pytest.mark.parametrize(
    "failure", ["tool_error", "unauthorized", "incompatible", "malformed", "unexpected"]
)
def test_other_exceptions_fail_without_tripping(plugin_import, models, failure):
    # Given nonavailability failures, when submitting, then they count without opening the breaker.
    def submit(item):
        if failure == "unexpected":
            raise RuntimeError("safe")
        raise models.ReflectLogError(failure, "safe")

    writer = plugin_import("core.writer").BackgroundWriter(
        submit, thread_factory=thread_factory, failure_threshold=1
    )
    try:
        assert writer.enqueue(job(models)) and writer.drain(2)
        assert writer.stats()["failed"] == 1 and not writer.breaker_open()
    finally:
        writer.shutdown(1)


def test_shutdown_budget_and_abandoned_queue(plugin_import, models):
    # Given blocked submission and queued work, when shutting down, then only queued work is abandoned.
    entered, release = Event(), Event()

    def submit(item):
        entered.set()
        assert release.wait(3)

    writer = plugin_import("core.writer").BackgroundWriter(
        submit, thread_factory=thread_factory
    )
    assert writer.enqueue(job(models)) and entered.wait(2)
    assert writer.enqueue(job(models, 2))
    started = time.monotonic()
    try:
        writer.shutdown(0.02)
        assert time.monotonic() - started < 0.5
        assert writer.stats()["abandoned"] == 1
        assert not writer.enqueue(job(models, 3))
        writer.shutdown(0.02)
        assert writer.stats()["abandoned"] == 1
    finally:
        release.set()
        assert writer.drain(2)


def test_rejected_counter_and_snapshot(plugin_import):
    # Given a policy rejection, when recording it, then snapshots are independent copies.
    writer = plugin_import("core.writer").BackgroundWriter(
        lambda item: None, thread_factory=thread_factory
    )
    snapshot = writer.stats()
    writer.record_rejection()
    assert snapshot["rejected"] == 0 and writer.stats()["rejected"] == 1
    writer.shutdown(0)
