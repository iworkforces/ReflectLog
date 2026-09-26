from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from time import monotonic
from typing import TYPE_CHECKING

import anyio

from reflectlog.application.config.validation import canonical_workspace_id
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.utils.logging import create_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from reflectlog.application.config.settings import Config


def _create_manager(config: Config) -> MemoryManager:
    workspace_id = config.workspace_id
    if not workspace_id:
        raise ValueError("A concrete workspace is required to create a manager")
    return MemoryManager(
        config, create_logger(__name__, workspace_id, config.log_level)
    )


@dataclass
class _Entry:
    manager: MemoryManager
    active: int
    idle_since: float


class WorkspaceRegistry:
    def __init__(
        self,
        config: Config,
        manager_factory: Callable[[Config], MemoryManager] = _create_manager,
        *,
        clock: Callable[[], float] = monotonic,
        idle_ttl: float = 900,
        max_idle: int = 8,
    ) -> None:
        self._config = config
        self._factory = manager_factory
        self._clock = clock
        self._idle_ttl = idle_ttl
        self._max_idle = max_idle
        self._entries: dict[str, _Entry] = {}
        self._quarantined: dict[str, MemoryManager] = {}
        self._lock = anyio.Lock()
        self._drained = anyio.Event()
        self._drained.set()
        self._closed = anyio.Event()
        self._closing = False
        self._close_error: ExceptionGroup | None = None

    @asynccontextmanager
    async def acquire(self, workspace_id: str) -> AsyncGenerator[MemoryManager]:
        """Pin a canonical workspace manager through the whole tool invocation."""
        key = canonical_workspace_id(workspace_id)
        entry = await self._acquire_entry(key)
        try:
            yield entry.manager
        finally:
            with anyio.CancelScope(shield=True):
                async with self._lock:
                    entry.active -= 1
                    if entry.active == 0:
                        entry.idle_since = self._clock()
                    if all(item.active == 0 for item in self._entries.values()):
                        self._drained.set()
                    if not self._closing:
                        await self._prune_locked()

    async def _acquire_entry(self, key: str) -> _Entry:
        with anyio.CancelScope(shield=True):
            async with self._lock:
                if self._closing:
                    raise RuntimeError("WorkspaceRegistry is closed")
                await self._prune_locked()
                if key in self._quarantined:
                    raise RuntimeError(f"Workspace {key!r} has a manager pending close")
                entry = self._entries.get(key)
                if entry is None:
                    concrete = replace(self._config, workspace_id=key)
                    manager = await anyio.to_thread.run_sync(self._factory, concrete)
                    entry = _Entry(manager, 0, self._clock())
                    self._entries[key] = entry
                if self._drained.is_set():
                    self._drained = anyio.Event()
                entry.active += 1
                return entry
        raise RuntimeError("Workspace acquisition was cancelled")

    async def _prune_locked(self) -> None:
        now = self._clock()
        idle = sorted(
            ((key, entry) for key, entry in self._entries.items() if entry.active == 0),
            key=lambda item: item[1].idle_since,
        )
        excess = max(0, len(idle) - self._max_idle)
        for index, (key, entry) in enumerate(idle):
            if now - entry.idle_since >= self._idle_ttl or index < excess:
                _ = await self._evict_locked(key, entry.manager)

    async def _evict_locked(self, key: str, manager: MemoryManager) -> Exception | None:
        _ = self._entries.pop(key, None)
        try:
            await anyio.to_thread.run_sync(manager.close)
        except Exception as error:
            self._quarantined[key] = manager
            return error
        _ = self._quarantined.pop(key, None)
        return None

    async def prune(self) -> None:
        with anyio.CancelScope(shield=True):
            async with self._lock:
                if not self._closing:
                    await self._prune_locked()

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
                    for key, entry in tuple(self._entries.items()):
                        error = await self._evict_locked(key, entry.manager)
                        if error is not None:
                            errors.append(error)
                    for key, manager in pending:
                        error = await self._evict_locked(key, manager)
                        if error is not None:
                            errors.append(error)
                if errors:
                    self._close_error = ExceptionGroup(
                        "Workspace managers could not be closed", errors
                    )
            finally:
                self._closed.set()
            if self._close_error is not None:
                raise self._close_error
