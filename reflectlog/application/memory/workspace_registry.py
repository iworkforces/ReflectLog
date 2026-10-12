import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from time import monotonic
from typing import TYPE_CHECKING

import anyio

from reflectlog.application.config.validation import canonical_workspace_id
from reflectlog.application.memory.manager import MemoryManager, wemm_config
from reflectlog.application.utils.logging import create_logger
from reflectlog.core.enums import EmbedderProvider
from reflectlog.core.exceptions import ConfigurationError
from reflectlog.infrastructure.embeddings.wemm_service import (
    WeMMBorrow,
    acquire_wemm,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine

    from reflectlog.application.config.settings import Config


def _create_manager(config: Config) -> MemoryManager:
    workspace_id = config.workspace_id
    if not workspace_id:
        raise ValueError("A concrete workspace is required to create a manager")
    return MemoryManager(
        config, create_logger(__name__, workspace_id, config.log_level)
    )


class WorkspaceCapacityError(RuntimeError):
    """The resident workspace limit is reached and no idle workspace can be closed."""


@dataclass
class _Entry:
    manager: MemoryManager
    active: int
    idle_since: float


async def _finish_owned[T](
    operation: Coroutine[None, None, T],
) -> tuple[asyncio.Task[T], bool]:
    task = asyncio.create_task(operation)
    return task, await _wait_owned(task)


async def _wait_owned[T](task: asyncio.Task[T]) -> bool:
    cancelled = False
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                _ = await asyncio.wait({task})
            except asyncio.CancelledError:
                cancelled = True
    return cancelled


class WorkspaceRegistry:
    def __init__(
        self,
        config: Config,
        manager_factory: Callable[[Config], MemoryManager] = _create_manager,
        *,
        clock: Callable[[], float] = monotonic,
        idle_ttl: float = 900,
        max_idle: int = 8,
        max_resident: int | None = None,
    ) -> None:
        if max_resident is not None and (
            type(max_resident) is not int or max_resident < 1
        ):
            raise ConfigurationError("max_resident must be None or an integer >= 1")
        self._config = config
        self._factory = manager_factory
        self._clock = clock
        self._idle_ttl = idle_ttl
        self._max_idle = max_idle
        self._max_resident = max_resident
        self._entries: dict[str, _Entry] = {}
        self._building: dict[str, anyio.Event] = {}
        self._evicting: dict[str, asyncio.Task[Exception | None]] = {}
        self._quarantined: dict[str, MemoryManager] = {}
        self._lock = anyio.Lock()
        self._drained = anyio.Event()
        self._drained.set()
        self._closed = anyio.Event()
        self._closing = False
        self._close_error: ExceptionGroup | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._wemm_pin: WeMMBorrow | None = None

    @asynccontextmanager
    async def acquire(self, workspace_id: str) -> AsyncGenerator[MemoryManager]:
        """Pin a canonical workspace manager through the whole tool invocation."""
        key = canonical_workspace_id(workspace_id)
        task, cancelled = await _finish_owned(self._acquire_entry(key))
        try:
            entry = task.result()
        except Exception as error:
            if cancelled:
                raise asyncio.CancelledError from error
            raise
        try:
            if cancelled:
                raise asyncio.CancelledError
            yield entry.manager
        finally:
            release_task, release_cancelled = await _finish_owned(
                self._release_entry(entry)
            )
            try:
                release_task.result()
            finally:
                if release_cancelled:
                    raise asyncio.CancelledError

    async def _release_entry(self, entry: _Entry) -> None:
        with anyio.CancelScope(shield=True):
            async with self._lock:
                entry.active -= 1
                if entry.active == 0:
                    entry.idle_since = self._clock()
                pending = self._prune_locked() if not self._closing else []
                self._notify_drained_locked()
            for task in pending:
                _ = await asyncio.wait({task})

    def _notify_drained_locked(self) -> None:
        if (
            not self._building
            and not self._evicting
            and all(entry.active == 0 for entry in self._entries.values())
        ):
            self._drained.set()

    async def _acquire_entry(self, key: str) -> _Entry:
        with anyio.CancelScope(shield=True):
            while True:
                async with self._lock:
                    if self._closing:
                        raise RuntimeError("WorkspaceRegistry is closed")
                    _ = self._prune_locked()
                    if key in self._quarantined:
                        raise RuntimeError(
                            f"Workspace {key!r} has a manager pending close"
                        )
                    pending_close = self._evicting.get(key)
                    pending_build = self._building.get(key)
                    entry = self._entries.get(key)
                    capacity_wait: set[asyncio.Task[Exception | None]] = set()
                    if pending_close is None and pending_build is None:
                        if entry is not None:
                            if self._drained.is_set():
                                self._drained = anyio.Event()
                            entry.active += 1
                            return entry
                        capacity_wait = self._make_room_locked()
                        if not capacity_wait:
                            pending_build = anyio.Event()
                            self._building[key] = pending_build
                            if self._drained.is_set():
                                self._drained = anyio.Event()
                            break
                if capacity_wait:
                    _ = await asyncio.wait(
                        capacity_wait, return_when=asyncio.FIRST_COMPLETED
                    )
                elif pending_close is not None:
                    _ = await asyncio.wait({pending_close})
                elif pending_build is not None:
                    await pending_build.wait()
            try:
                concrete = replace(self._config, workspace_id=key)
                manager = await anyio.to_thread.run_sync(self._factory, concrete)
            except Exception:
                async with self._lock:
                    _ = self._building.pop(key)
                    pending_build.set()
                    self._notify_drained_locked()
                raise
            async with self._lock:
                if (
                    type(manager) is MemoryManager
                    and manager.config.embedder_provider is EmbedderProvider.WEMM
                    and self._wemm_pin is None
                ):
                    self._wemm_pin = acquire_wemm(wemm_config(manager.config))
                entry = _Entry(manager, 0, self._clock())
                self._entries[key] = entry
                _ = self._building.pop(key)
                pending_build.set()
                if self._closing:
                    self._notify_drained_locked()
                    raise RuntimeError("WorkspaceRegistry is closed")
                entry.active += 1
                return entry
        raise RuntimeError("Workspace acquisition was cancelled")

    def _make_room_locked(self) -> set[asyncio.Task[Exception | None]]:
        """Return evictions to await before re-checking; empty admits the new key."""
        if self._max_resident is None:
            return set()
        resident = (
            self._entries.keys()
            | self._building.keys()
            | self._evicting.keys()
            | self._quarantined.keys()
        )
        if len(resident) < self._max_resident:
            return set()
        oldest = min(
            (item for item in self._entries.items() if item[1].active == 0),
            key=lambda item: item[1].idle_since,
            default=None,
        )
        if oldest is not None:
            key, entry = oldest
            return {self._schedule_eviction_locked(key, entry.manager)}
        if self._evicting:
            return set(self._evicting.values())
        raise WorkspaceCapacityError(
            "Resident workspace limit reached; no idle workspace could be released"
        )

    def _prune_locked(self) -> list[asyncio.Task[Exception | None]]:
        now = self._clock()
        idle = sorted(
            ((key, entry) for key, entry in self._entries.items() if entry.active == 0),
            key=lambda item: item[1].idle_since,
        )
        excess = max(0, len(idle) - self._max_idle)
        scheduled: list[asyncio.Task[Exception | None]] = []
        for index, (key, entry) in enumerate(idle):
            if now - entry.idle_since >= self._idle_ttl or index < excess:
                scheduled.append(self._schedule_eviction_locked(key, entry.manager))
        return scheduled

    def _schedule_eviction_locked(
        self, key: str, manager: MemoryManager
    ) -> asyncio.Task[Exception | None]:
        _ = self._entries.pop(key, None)
        if self._drained.is_set():
            self._drained = anyio.Event()
        task = asyncio.create_task(self._evict(key, manager))
        self._evicting[key] = task
        return task

    async def _evict(self, key: str, manager: MemoryManager) -> Exception | None:
        try:
            await anyio.to_thread.run_sync(manager.close)
        except Exception as error:
            async with self._lock:
                self._quarantined[key] = manager
                _ = self._evicting.pop(key)
                self._notify_drained_locked()
            return error
        async with self._lock:
            _ = self._quarantined.pop(key, None)
            _ = self._evicting.pop(key)
            self._notify_drained_locked()
        return None

    async def prune(self) -> None:
        task, cancelled = await _finish_owned(self._prune())
        try:
            task.result()
        finally:
            if cancelled:
                raise asyncio.CancelledError

    async def _prune(self) -> None:
        with anyio.CancelScope(shield=True):
            async with self._lock:
                pending = self._prune_locked() if not self._closing else []
            for task in pending:
                _ = await asyncio.wait({task})

    async def run_reaper(self, interval: float = 60) -> None:
        """Sweep idle managers until close finishes."""
        while not self._closing:
            with anyio.move_on_after(interval):
                await self._closed.wait()
            if self._closing:
                return
            await self.prune()

    async def close(self) -> None:
        """Reject new acquisitions, drain active users, then persist managers."""
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close())
        task = self._close_task
        cancelled = await _wait_owned(task)
        try:
            task.result()
        finally:
            if cancelled:
                raise asyncio.CancelledError

    async def _close(self) -> None:
        with anyio.CancelScope(shield=True):
            async with self._lock:
                if not self._closing:
                    self._closing = True
                    other_closer = False
                elif not self._closed.is_set():
                    other_closer = True
                elif not self._quarantined:
                    return
                else:
                    self._closed = anyio.Event()
                    self._close_error = None
                    other_closer = False
            if other_closer:
                await self._closed.wait()
                if self._close_error is not None:
                    raise self._close_error
                return
            await self._drained.wait()
            errors: list[Exception] = []
            try:
                async with self._lock:
                    pending = tuple(self._quarantined.items())
                    tasks = [
                        self._schedule_eviction_locked(key, entry.manager)
                        for key, entry in tuple(self._entries.items())
                    ]
                    for key, manager in pending:
                        tasks.append(self._schedule_eviction_locked(key, manager))
                for task in tasks:
                    error = await task
                    if error is not None:
                        errors.append(error)
                if errors:
                    self._close_error = ExceptionGroup(
                        "Workspace managers could not be closed", errors
                    )
                elif self._wemm_pin is not None:
                    self._wemm_pin.close()
                    self._wemm_pin = None
            finally:
                self._closed.set()
            if self._close_error is not None:
                raise self._close_error
