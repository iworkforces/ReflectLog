from __future__ import annotations

from collections import deque
import math
from threading import Condition, Thread, current_thread
import time
from typing import TYPE_CHECKING

from .models import ReflectLogError, WriteJob

if TYPE_CHECKING:
    from collections.abc import Callable


class BackgroundWriter:
    """Best-effort FIFO, not durable; shutdown cannot cancel an in-flight mutation.

    Only availability/timeout/unknown outcomes trip the breaker. Other exceptions
    count as failed and interrupt the consecutive availability-failure streak.
    """

    def __init__(
        self,
        submit: Callable[[WriteJob], None],
        *,
        thread_factory: Callable[[Callable[[], None], str], Thread],
        capacity: int = 64,
        failure_threshold: int = 5,
        cooldown_s: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            capacity < 1
            or failure_threshold < 1
            or not math.isfinite(cooldown_s)
            or cooldown_s < 0
        ):
            raise ValueError("writer: invalid capacity or breaker settings")
        self._submit = submit
        self._thread_factory = thread_factory
        self._capacity = capacity
        self._failure_threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._condition = Condition()
        self._queue: deque[WriteJob] = deque()
        self._thread: Thread | None = None
        self._closing = False
        self._inflight = False
        self._consecutive = 0
        self._open_until: float | None = None
        self._stats: dict[str, int] = dict.fromkeys(
            (
                "enqueued",
                "completed",
                "failed",
                "unknown",
                "dropped",
                "rejected",
                "abandoned",
            ),
            0,
        )

    def _breaker_open_locked(self) -> bool:
        if self._open_until is None:
            return False
        if self._clock() >= self._open_until:
            self._open_until = None
            self._consecutive = 0
            return False
        return True

    def breaker_open(self) -> bool:
        with self._condition:
            return self._breaker_open_locked()

    def enqueue(self, job: WriteJob) -> bool:
        with self._condition:
            if (
                self._closing
                or self._breaker_open_locked()
                or len(self._queue) >= self._capacity
            ):
                self._stats["dropped"] += 1
                return False
            if self._thread is None:
                thread = self._thread_factory(self._run, "reflectlog-writer")
                self._thread = thread
                thread.start()
            self._queue.append(job)
            self._stats["enqueued"] += 1
            self._condition.notify_all()
            return True

    def _record_failure_locked(self, error: Exception) -> None:
        self._stats["failed"] += 1
        if isinstance(error, ReflectLogError) and error.code in {
            "unavailable",
            "timeout",
            "outcome_unknown",
        }:
            if error.code == "outcome_unknown":
                self._stats["unknown"] += 1
            self._consecutive += 1
            if self._consecutive >= self._failure_threshold:
                self._open_until = self._clock() + self._cooldown_s
        else:
            self._consecutive = 0

    def record_failure(self, error: Exception) -> None:
        with self._condition:
            self._record_failure_locked(error)

    def record_rejection(self) -> None:
        with self._condition:
            self._stats["rejected"] += 1

    def _run(self) -> None:
        while True:
            with self._condition:
                _ = self._condition.wait_for(lambda: bool(self._queue) or self._closing)
                if not self._queue:
                    return
                job = self._queue.popleft()
                self._inflight = True
            error: Exception | None = None
            try:
                self._submit(job)
            except Exception as caught:
                error = caught
            with self._condition:
                if error is None:
                    self._stats["completed"] += 1
                    self._consecutive = 0
                    self._open_until = None
                else:
                    self._record_failure_locked(error)
                self._inflight = False
                self._condition.notify_all()

    def stats(self) -> dict[str, int]:
        with self._condition:
            return self._stats.copy()

    def drain(self, timeout: float) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: not self._queue and not self._inflight,
                timeout=max(0.0, timeout),
            )

    def shutdown(self, budget_s: float) -> None:
        with self._condition:
            if self._closing:
                return
            self._closing = True
            thread = self._thread
            self._condition.notify_all()
        if thread is not None and thread is not current_thread():
            thread.join(timeout=max(0.0, budget_s))
        with self._condition:
            self._stats["abandoned"] += len(self._queue)
            self._queue.clear()
            self._condition.notify_all()
