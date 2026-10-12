from annotationlib import Format
import asyncio
from dataclasses import replace
import inspect
import threading
from typing import Any, cast
from unittest.mock import MagicMock

import anyio
import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory import workspace_registry
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.memory.workspace_registry import (
    WorkspaceCapacityError,
    WorkspaceRegistry,
)
from reflectlog.application.utils.security import SecretString
from reflectlog.core.enums import EmbedderProvider
from reflectlog.core.exceptions import ConfigurationError
from reflectlog.core.search_health import SearchFailureSnapshot
from reflectlog.infrastructure.embeddings.wemm_service import WeMMBorrow


def fake_manager(config: Config) -> MagicMock:
    manager = MagicMock(spec=MemoryManager)
    manager.search_failure_snapshot = MagicMock(return_value=SearchFailureSnapshot())
    manager.config = config
    manager.closed = False
    manager.close.side_effect = lambda: setattr(manager, "closed", True)
    return manager


def was_closed(manager: MemoryManager) -> bool:
    return vars(manager)["closed"] is True


@pytest.fixture
def config() -> Config:
    return Config(workspace_id="", openrouter_api_key=SecretString("sk-test"))


@pytest.mark.unit
async def test_isolation_and_canonical_identity(config: Config) -> None:
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    assert created == []
    async with registry.acquire("ALPHA") as alpha:
        async with registry.acquire("alpha") as same:
            async with registry.acquire("beta") as beta:
                assert alpha is same
                assert alpha is not beta
                assert alpha.config.workspace_id == "alpha"
                assert beta.config.workspace_id == "beta"
                assert config.workspace_id == ""
    assert len(created) == 2
    await registry.close()


@pytest.mark.unit
@pytest.mark.parametrize("workspace_id", [".", "..", "abc..def", "../other", "a/b", ""])
async def test_invalid_workspace_id_is_rejected(
    config: Config, workspace_id: str
) -> None:
    registry = WorkspaceRegistry(config, fake_manager)
    with pytest.raises(ConfigurationError, match="WORKSPACE_ID"):
        async with registry.acquire(workspace_id):
            pytest.fail("Invalid workspace reached manager construction")


@pytest.mark.unit
async def test_acquire_requires_explicit_workspace(config: Config) -> None:
    registry = WorkspaceRegistry(config, fake_manager)
    parameters = inspect.signature(
        registry.acquire, annotation_format=Format.STRING
    ).parameters
    assert parameters["workspace_id"].default is inspect.Parameter.empty


@pytest.mark.unit
async def test_configured_workspace_is_not_selected(config: Config) -> None:
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(
        Config(workspace_id="Default", openrouter_api_key=config.openrouter_api_key),
        factory,
    )
    assert created == []
    async with registry.acquire("Selected") as manager:
        assert manager.config.workspace_id == "selected"
    assert len(created) == 1
    await registry.close()


@pytest.mark.unit
async def test_single_flight_construction(config: Config) -> None:
    created: list[MagicMock] = []
    release = anyio.Event()
    entered = anyio.Event()

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    seen: list[MemoryManager] = []

    async def worker() -> None:
        async with registry.acquire("Shared") as manager:
            seen.append(manager)
            entered.set()
            await release.wait()

    async with anyio.create_task_group() as group:
        for _ in range(12):
            group.start_soon(worker)
        await entered.wait()
        release.set()
    assert len(created) == 1
    assert all(manager is created[0] for manager in seen)
    assert len(seen) == 12
    await registry.close()


@pytest.mark.unit
async def test_slow_construction_does_not_block_another_workspace(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    fast_entered = anyio.Event()

    def factory(concrete: Config) -> MagicMock:
        if concrete.workspace_id == "slow":
            started.set()
            assert finish.wait(timeout=5)
        return fake_manager(concrete)

    registry = WorkspaceRegistry(config, factory)

    async def slow_user() -> None:
        async with registry.acquire("slow"):
            pass

    async def fast_user() -> None:
        async with registry.acquire("fast") as manager:
            assert manager.config.workspace_id == "fast"
            fast_entered.set()

    slow_task = asyncio.create_task(slow_user())
    fast_task: asyncio.Task[None] | None = None
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        fast_task = asyncio.create_task(fast_user())
        with anyio.fail_after(1):
            await fast_entered.wait()
    finally:
        finish.set()
        await slow_task
        if fast_task is not None:
            await fast_task
        await registry.close()


@pytest.mark.unit
async def test_slow_same_key_construction_is_single_flight(config: Config) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []
    seen: list[MemoryManager] = []
    second_entered = anyio.Event()

    def factory(concrete: Config) -> MagicMock:
        started.set()
        assert finish.wait(timeout=5)
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)

    async def user(workspace_id: str) -> None:
        async with registry.acquire(workspace_id) as manager:
            seen.append(manager)
            if workspace_id == "SAME":
                second_entered.set()

    first = asyncio.create_task(user("same"))
    second: asyncio.Task[None] | None = None
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        second = asyncio.create_task(user("SAME"))
        await anyio.lowlevel.checkpoint()
        assert not second_entered.is_set()
        assert created == []
    finally:
        finish.set()
        await first
        if second is not None:
            await second
        await registry.close()
    assert len(created) == 1
    assert seen == [created[0], created[0]]
    assert was_closed(created[0])


@pytest.mark.unit
async def test_slow_idle_eviction_does_not_block_another_workspace(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    now = [0.0]

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        if concrete.workspace_id == "slow":

            def persist() -> None:
                started.set()
                assert finish.wait(timeout=5)
                manager.closed = True

            manager.close.side_effect = persist
        return manager

    registry = WorkspaceRegistry(config, factory, clock=lambda: now[0])
    fast_entered = anyio.Event()

    async def fast_user() -> None:
        async with registry.acquire("fast") as manager:
            assert manager.config.workspace_id == "fast"
            fast_entered.set()

    async with registry.acquire("slow"):
        pass
    now[0] = 901
    prune_task = asyncio.create_task(registry.prune())
    fast_task: asyncio.Task[None] | None = None
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        fast_task = asyncio.create_task(fast_user())
        with anyio.fail_after(1):
            await fast_entered.wait()
    finally:
        finish.set()
        await prune_task
        if fast_task is not None:
            await fast_task
        await registry.close()


@pytest.mark.unit
async def test_acquire_waits_for_same_key_eviction_before_rebuilding(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    entered = anyio.Event()
    created: list[MagicMock] = []
    now = [0.0]

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        if not created:

            def persist() -> None:
                started.set()
                assert finish.wait(timeout=5)
                manager.closed = True

            manager.close.side_effect = persist
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory, clock=lambda: now[0])
    async with registry.acquire("same"):
        pass
    now[0] = 901
    prune_task = asyncio.create_task(registry.prune())

    async def user() -> None:
        async with registry.acquire("SAME") as manager:
            assert manager is not created[0]
            entered.set()

    user_task: asyncio.Task[None] | None = None
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        user_task = asyncio.create_task(user())
        await anyio.lowlevel.checkpoint()
        assert not entered.is_set()
        assert len(created) == 1
    finally:
        finish.set()
        await prune_task
        if user_task is not None:
            await user_task
        await registry.close()
    assert entered.is_set()
    assert len(created) == 2
    assert all(was_closed(manager) for manager in created)


@pytest.mark.unit
async def test_close_waits_for_inflight_construction(config: Config) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        started.set()
        assert finish.wait(timeout=5)
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)

    async def user() -> None:
        with pytest.raises(RuntimeError, match="closed"):
            async with registry.acquire("slow"):
                pytest.fail("Closing registry returned an in-flight manager")

    user_task = asyncio.create_task(user())
    close_task: asyncio.Task[None] | None = None
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        close_task = asyncio.create_task(registry.close())
        while not registry._closing:
            await anyio.lowlevel.checkpoint()
        assert not close_task.done()
        assert created == []
    finally:
        finish.set()
        await user_task
        if close_task is not None:
            await close_task
    assert len(created) == 1
    assert was_closed(created[0])
    assert created[0].close.call_count == 1


@pytest.mark.unit
async def test_sliding_ttl_and_active_protection(config: Config) -> None:
    now = [0.0]
    registry = WorkspaceRegistry(config, fake_manager, clock=lambda: now[0])
    async with registry.acquire("one") as first:
        now[0] = 899
        await registry.prune()
        assert not was_closed(first)
    now[0] = 1700
    async with registry.acquire("ONE") as reused:
        assert reused is first
    now[0] = 2600
    await registry.prune()
    assert was_closed(first)
    async with registry.acquire("one") as replacement:
        assert replacement is not first
    await registry.close()


@pytest.mark.unit
async def test_idle_cap_never_evicts_active(config: Config) -> None:
    now = [0.0]
    registry = WorkspaceRegistry(config, fake_manager, clock=lambda: now[0], max_idle=2)
    async with registry.acquire("pinned") as pinned:
        idle: list[MemoryManager] = []
        for workspace in ("one", "two", "three"):
            now[0] += 1
            async with registry.acquire(workspace) as manager:
                idle.append(manager)
        assert was_closed(idle[0])
        assert not was_closed(idle[1])
        assert not was_closed(idle[2])
        assert not was_closed(pinned)
    await registry.close()
    assert was_closed(pinned)


@pytest.mark.unit
async def test_default_idle_cap_is_eight(config: Config) -> None:
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    for index in range(9):
        async with registry.acquire(f"workspace-{index}"):
            pass
    assert was_closed(created[0])
    assert all(not was_closed(manager) for manager in created[1:])
    await registry.close()


@pytest.mark.unit
async def test_close_drains_and_rejects_new_acquisitions(config: Config) -> None:
    registry = WorkspaceRegistry(config, fake_manager)
    acquired = anyio.Event()
    release = anyio.Event()
    completed = anyio.Event()
    held: list[MemoryManager] = []

    async def user() -> None:
        async with registry.acquire("live") as manager:
            held.append(manager)
            acquired.set()
            await release.wait()

    async def closer() -> None:
        await registry.close()
        completed.set()

    async with anyio.create_task_group() as group:
        group.start_soon(user)
        await acquired.wait()
        group.start_soon(closer)
        await anyio.lowlevel.checkpoint()
        assert not completed.is_set()
        assert not was_closed(held[0])
        release.set()
    assert completed.is_set()
    assert was_closed(held[0])
    with pytest.raises(RuntimeError, match="closed"):
        async with registry.acquire("live"):
            pytest.fail("Closed registry returned a manager")
    await registry.close()


@pytest.mark.unit
async def test_cancelled_construction_waits_for_native_worker(config: Config) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        started.set()
        finish.wait(timeout=5)
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    completed = anyio.Event()

    async def user() -> None:
        async with registry.acquire("slow"):
            completed.set()

    async with anyio.create_task_group() as group:
        group.start_soon(user)
        await anyio.to_thread.run_sync(started.wait)
        group.cancel_scope.cancel()
        finish.set()
    assert len(created) == 1
    assert created[0].closed is False
    await registry.close()
    assert created[0].closed


@pytest.mark.unit
async def test_direct_cancel_during_construction_keeps_completed_manager_owned(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        started.set()
        assert finish.wait(timeout=5)
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)

    async def user() -> None:
        async with registry.acquire("slow"):
            pytest.fail("Cancelled acquisition returned a manager")

    user_task = asyncio.create_task(user())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        user_task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await user_task
        assert len(created) == 1
        assert registry._entries["slow"].manager is created[0]
        assert registry._entries["slow"].active == 0
        await registry.close()
        assert was_closed(created[0])
        assert created[0].close.call_count == 1
    finally:
        finish.set()


@pytest.mark.unit
async def test_direct_cancel_during_failed_construction_propagates_cancellation(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()

    def factory(_: Config) -> MemoryManager:
        started.set()
        assert finish.wait(timeout=5)
        raise OSError("constructor failed")

    registry = WorkspaceRegistry(config, factory)

    async def user() -> None:
        async with registry.acquire("broken"):
            pytest.fail("Failed acquisition returned a manager")

    user_task = asyncio.create_task(user())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        user_task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await user_task
        assert registry._entries == {}
        await registry.close()
    finally:
        finish.set()


@pytest.mark.unit
async def test_reaper_expires_idle_without_another_acquisition(config: Config) -> None:
    now = [0.0]
    expired = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        manager.close.side_effect = expired.set
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory, clock=lambda: now[0])
    async with anyio.create_task_group() as group:
        group.start_soon(registry.run_reaper, 0)
        async with registry.acquire("idle"):
            now[0] = 901
            await anyio.lowlevel.checkpoint()
            assert not expired.is_set()
        now[0] = 1801
        await anyio.to_thread.run_sync(expired.wait)
        assert expired.is_set()
        await registry.close()
    assert created[0].close.call_count == 1


@pytest.mark.unit
async def test_failed_eviction_quarantines_workspace_only(config: Config) -> None:
    now = [0.0]
    attempts: list[str] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        if concrete.workspace_id == "broken":

            def close_once() -> None:
                attempts.append("broken")
                if len(attempts) == 1:
                    raise OSError("storage unavailable")
                manager.closed = True

            manager.close.side_effect = close_once
        return manager

    registry = WorkspaceRegistry(config, factory, clock=lambda: now[0])
    async with registry.acquire("broken") as broken:
        pass
    now[0] = 901
    await registry.prune()
    assert attempts == ["broken"]
    with pytest.raises(RuntimeError, match="pending close"):
        async with registry.acquire("BROKEN"):
            pytest.fail("Quarantined manager was reused")
    async with registry.acquire("healthy") as healthy:
        assert healthy.config.workspace_id == "healthy"
    assert not was_closed(broken)
    await registry.close()
    assert attempts == ["broken", "broken"]
    assert was_closed(broken)
    assert was_closed(healthy)


@pytest.mark.unit
async def test_direct_cancel_during_eviction_waits_for_close(config: Config) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            started.set()
            assert finish.wait(timeout=5)
            manager.closed = True

        manager.close.side_effect = persist
        created.append(manager)
        return manager

    now = [0.0]
    registry = WorkspaceRegistry(config, factory, clock=lambda: now[0])
    async with registry.acquire("idle") as manager:
        pass
    now[0] = 901
    prune_task = asyncio.create_task(registry.prune())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        prune_task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await prune_task
        assert was_closed(manager)
        await registry.close()
        assert created[0].close.call_count == 1
    finally:
        finish.set()


@pytest.mark.unit
async def test_direct_cancel_during_failed_eviction_quarantines_for_retry(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    attempts = [0]

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            attempts[0] += 1
            if attempts[0] == 1:
                started.set()
                assert finish.wait(timeout=5)
                raise OSError("storage unavailable")
            manager.closed = True

        manager.close.side_effect = persist
        return manager

    now = [0.0]
    registry = WorkspaceRegistry(config, factory, clock=lambda: now[0])
    async with registry.acquire("broken") as manager:
        pass
    now[0] = 901
    prune_task = asyncio.create_task(registry.prune())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        prune_task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await prune_task
        with pytest.raises(RuntimeError, match="pending close"):
            async with registry.acquire("broken"):
                pytest.fail("Failed manager was reused")
        assert not was_closed(manager)
        await registry.close()
        assert was_closed(manager)
        assert attempts == [2]
    finally:
        finish.set()


@pytest.mark.unit
async def test_shutdown_retries_failed_closes_without_losing_managers(
    config: Config,
) -> None:
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        if concrete.workspace_id == "broken":
            attempts = [0]

            def close_once() -> None:
                attempts[0] += 1
                if attempts[0] == 1:
                    raise OSError("storage unavailable")
                manager.closed = True

            manager.close.side_effect = close_once
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    async with registry.acquire("broken"):
        pass
    async with registry.acquire("healthy"):
        pass
    with pytest.raises(ExceptionGroup, match="could not be closed"):
        await registry.close()
    assert not was_closed(created[0])
    assert was_closed(created[1])
    await registry.close()
    assert was_closed(created[0])
    assert created[0].close.call_count == 2
    assert created[1].close.call_count == 1


@pytest.mark.unit
async def test_concurrent_closers_report_persist_failure_and_can_retry(
    config: Config,
) -> None:
    acquired = anyio.Event()
    release = anyio.Event()
    attempts = [0]

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def close_once() -> None:
            attempts[0] += 1
            if attempts[0] == 1:
                raise OSError("storage unavailable")
            manager.closed = True

        manager.close.side_effect = close_once
        return manager

    registry = WorkspaceRegistry(config, factory)
    held: list[MemoryManager] = []

    async def user() -> None:
        async with registry.acquire("broken") as manager:
            held.append(manager)
            acquired.set()
            await release.wait()

    failures: list[ExceptionGroup] = []

    async def closer() -> None:
        try:
            await registry.close()
        except ExceptionGroup as error:
            failures.append(error)

    async with anyio.create_task_group() as group:
        group.start_soon(user)
        await acquired.wait()
        group.start_soon(closer)
        while not registry._closing:
            await anyio.lowlevel.checkpoint()
        group.start_soon(closer)
        await anyio.lowlevel.checkpoint()
        release.set()

    assert len(failures) == 2
    assert all("storage unavailable" in str(error.exceptions[0]) for error in failures)
    assert not was_closed(held[0])
    await registry.close()
    assert was_closed(held[0])
    assert attempts == [2]


@pytest.mark.unit
async def test_close_queued_behind_failed_persist_shares_attempt(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    second_entered = anyio.Event()
    attempts = [0]

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            attempts[0] += 1
            if attempts[0] == 1:
                started.set()
                assert finish.wait(timeout=5)
                raise OSError("storage unavailable")
            manager.closed = True

        manager.close.side_effect = persist
        return manager

    registry = WorkspaceRegistry(config, factory)
    async with registry.acquire("broken") as manager:
        pass

    async def closer(entered: anyio.Event | None = None) -> ExceptionGroup | None:
        if entered is not None:
            entered.set()
        try:
            await registry.close()
        except ExceptionGroup as error:
            return error
        return None

    first = asyncio.create_task(closer())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        second = asyncio.create_task(closer(second_entered))
        await second_entered.wait()
        await anyio.lowlevel.checkpoint()
        finish.set()
        first_failure = await first
        second_failure = await second
        assert isinstance(first_failure, ExceptionGroup)
        assert second_failure is first_failure
        assert attempts == [1]
        assert not was_closed(manager)
        await registry.close()
        assert was_closed(manager)
        assert attempts == [2]
    finally:
        finish.set()


@pytest.mark.unit
async def test_cancelled_shutdown_waits_for_manager_persistence(config: Config) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            started.set()
            assert finish.wait(timeout=5)
            manager.closed = True

        manager.close.side_effect = persist
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    async with registry.acquire("live") as manager:
        pass

    async with anyio.create_task_group() as group:
        group.start_soon(registry.close)
        assert await anyio.to_thread.run_sync(started.wait, 5)
        group.cancel_scope.cancel()
        finish.set()

    assert was_closed(manager)
    await registry.close()
    assert created[0].close.call_count == 1


@pytest.mark.unit
async def test_direct_cancel_during_shutdown_waits_for_persistence(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            started.set()
            assert finish.wait(timeout=5)
            manager.closed = True

        manager.close.side_effect = persist
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    async with registry.acquire("live") as manager:
        pass

    close_task = asyncio.create_task(registry.close())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        close_task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        assert was_closed(manager)
        with pytest.raises(RuntimeError, match="closed"):
            async with registry.acquire("live"):
                pytest.fail("Closing registry returned a manager")
        await registry.close()
        assert created[0].close.call_count == 1
    finally:
        finish.set()


@pytest.mark.unit
async def test_direct_cancel_during_failed_shutdown_reports_to_other_closer(
    config: Config,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    acquired = anyio.Event()
    release = anyio.Event()
    second_waiting = anyio.Event()
    attempts = [0]

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)

        def persist() -> None:
            attempts[0] += 1
            if attempts[0] == 1:
                started.set()
                assert finish.wait(timeout=5)
                raise OSError("storage unavailable")
            manager.closed = True

        manager.close.side_effect = persist
        return manager

    registry = WorkspaceRegistry(config, factory)
    held: list[MemoryManager] = []

    async def user() -> None:
        async with registry.acquire("broken") as manager:
            held.append(manager)
            acquired.set()
            await release.wait()

    user_task = asyncio.create_task(user())
    await acquired.wait()

    async def second_closer() -> None:
        second_waiting.set()
        await registry.close()

    first = asyncio.create_task(registry.close())
    try:
        while not registry._closing:
            await anyio.lowlevel.checkpoint()
        second = asyncio.create_task(second_closer())
        await second_waiting.wait()
        await anyio.lowlevel.checkpoint()
        release.set()
        assert await anyio.to_thread.run_sync(started.wait, 5)
        first.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(ExceptionGroup, match="could not be closed") as failure:
            await second
        assert "storage unavailable" in str(failure.value.exceptions[0])
        await user_task
        assert not was_closed(held[0])
        await registry.close()
        assert was_closed(held[0])
        assert attempts == [2]
    finally:
        release.set()
        finish.set()


@pytest.mark.unit
async def test_direct_cancel_of_pinned_user_drains_closing_registry(
    config: Config,
) -> None:
    created: list[MagicMock] = []

    def factory(concrete: Config) -> MagicMock:
        manager = fake_manager(concrete)
        created.append(manager)
        return manager

    registry = WorkspaceRegistry(config, factory)
    acquired = anyio.Event()
    held: list[MemoryManager] = []

    async def user() -> None:
        async with registry.acquire("live") as manager:
            held.append(manager)
            acquired.set()
            await anyio.sleep_forever()

    user_task = asyncio.create_task(user())
    try:
        await acquired.wait()
        close_task = asyncio.create_task(registry.close())
        while not registry._closing:
            await anyio.lowlevel.checkpoint()
        with pytest.raises(RuntimeError, match="closed"):
            async with registry.acquire("new"):
                pytest.fail("Closing registry accepted a new manager")
        user_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await user_task
        await close_task
        assert registry._entries == {}
        assert was_closed(held[0])
        await registry.close()
        assert created[0].close.call_count == 1
    finally:
        user_task.cancel()


class Pool:
    """Thread-safe factory and close recorder for resident-capacity tests."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.managers: dict[str, MagicMock] = {}
        self.alive = 0
        self.peak = 0
        self.build_started: dict[str, threading.Event] = {}
        self.build_gates: dict[str, threading.Event] = {}
        self.close_started: dict[str, threading.Event] = {}
        self.close_gates: dict[str, threading.Event] = {}
        self.close_failures: dict[str, int] = {}
        self.build_failures: set[str] = set()
        self._guard = threading.Lock()

    def hold_build(self, key: str) -> tuple[threading.Event, threading.Event]:
        started, gate = threading.Event(), threading.Event()
        self.build_started[key] = started
        self.build_gates[key] = gate
        return started, gate

    def hold_close(self, key: str) -> tuple[threading.Event, threading.Event]:
        started, gate = threading.Event(), threading.Event()
        self.close_started[key] = started
        self.close_gates[key] = gate
        return started, gate

    def factory(self, concrete: Config) -> MagicMock:
        key = concrete.workspace_id
        with self._guard:
            self.events.append(f"build:{key}")
            self.alive += 1
            self.peak = max(self.peak, self.alive)
        started = self.build_started.get(key)
        if started is not None:
            started.set()
        gate = self.build_gates.get(key)
        if gate is not None:
            assert gate.wait(timeout=5)
        if key in self.build_failures:
            with self._guard:
                self.alive -= 1
            raise OSError("constructor failed")
        manager = fake_manager(concrete)
        manager.close.side_effect = lambda: self._close(key, manager)
        with self._guard:
            self.managers[key] = manager
        return manager

    def _close(self, key: str, manager: MagicMock) -> None:
        started = self.close_started.get(key)
        if started is not None:
            started.set()
        gate = self.close_gates.get(key)
        if gate is not None:
            assert gate.wait(timeout=5)
        with self._guard:
            if self.close_failures.get(key, 0) > 0:
                self.close_failures[key] -= 1
                self.events.append(f"close-failed:{key}")
                raise OSError("storage unavailable")
            manager.closed = True
            self.alive -= 1
            self.events.append(f"close:{key}")


@pytest.mark.unit
@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "2"])
def test_max_resident_rejects_invalid_values(config: Config, limit: object) -> None:
    options: dict[str, Any] = {"max_resident": limit}
    with pytest.raises(ConfigurationError, match="max_resident") as failure:
        WorkspaceRegistry(config, fake_manager, **options)
    assert str(limit) not in str(failure.value)


@pytest.mark.unit
@pytest.mark.parametrize("limit", [None, 1, 3])
def test_max_resident_accepts_none_or_positive_int(
    config: Config, limit: int | None
) -> None:
    WorkspaceRegistry(config, fake_manager, max_resident=limit)


@pytest.mark.unit
async def test_capacity_error_is_prompt_and_content_free(config: Config) -> None:
    pool = Pool()
    registry = WorkspaceRegistry(config, pool.factory, max_resident=1)
    async with registry.acquire("tenant-red") as held:
        with anyio.fail_after(1):
            with pytest.raises(WorkspaceCapacityError) as failure:
                async with registry.acquire("tenant-blue"):
                    pytest.fail("Over-capacity acquisition returned a manager")
        assert pool.events == ["build:tenant-red"]
        assert "tenant-red" not in str(failure.value)
        assert "tenant-blue" not in str(failure.value)
        assert not was_closed(held)
    async with registry.acquire("tenant-blue") as admitted:
        assert admitted.config.workspace_id == "tenant-blue"
    assert pool.events == ["build:tenant-red", "close:tenant-red", "build:tenant-blue"]
    await registry.close()


@pytest.mark.unit
async def test_pressure_eviction_closes_idle_manager_before_building(
    config: Config,
) -> None:
    pool = Pool()
    started, gate = pool.hold_close("old")
    registry = WorkspaceRegistry(
        config, pool.factory, clock=lambda: 0.0, max_resident=1
    )
    async with registry.acquire("old"):
        pass

    async def newcomer() -> None:
        async with registry.acquire("new"):
            pass

    newcomer_task = asyncio.create_task(newcomer())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        assert pool.events == ["build:old"]
        assert not newcomer_task.done()
    finally:
        gate.set()
        await newcomer_task
        await registry.close()
    assert pool.events[:3] == ["build:old", "close:old", "build:new"]
    assert pool.managers["old"].close.call_count == 1


@pytest.mark.unit
async def test_pressure_eviction_closes_oldest_idle_first(config: Config) -> None:
    pool = Pool()
    now = [0.0]
    registry = WorkspaceRegistry(
        config, pool.factory, clock=lambda: now[0], max_resident=2
    )
    async with registry.acquire("old"):
        pass
    now[0] = 1
    async with registry.acquire("young"):
        pass
    now[0] = 2
    async with registry.acquire("third"):
        pass
    assert pool.events == ["build:old", "build:young", "close:old", "build:third"]
    assert not was_closed(pool.managers["young"])
    await registry.close()


@pytest.mark.unit
async def test_pressure_eviction_orders_by_idle_time_not_insertion(
    config: Config,
) -> None:
    pool = Pool()
    now = [0.0]
    registry = WorkspaceRegistry(
        config, pool.factory, clock=lambda: now[0], max_resident=2
    )
    async with registry.acquire("first"):
        now[0] = 1
        async with registry.acquire("second"):
            now[0] = 2
        now[0] = 3
    now[0] = 4
    async with registry.acquire("third"):
        pass
    assert pool.events == ["build:first", "build:second", "close:second", "build:third"]
    assert not was_closed(pool.managers["first"])
    await registry.close()


@pytest.mark.unit
async def test_same_key_reuse_never_needs_a_slot(config: Config) -> None:
    pool = Pool()
    registry = WorkspaceRegistry(config, pool.factory, max_resident=1)
    async with registry.acquire("Alpha") as first:
        async with registry.acquire("alpha") as same:
            async with registry.acquire("ALPHA") as again:
                assert same is first
                assert again is first
    assert pool.events == ["build:alpha"]
    await registry.close()


@pytest.mark.unit
async def test_concurrent_same_key_under_limit_builds_once(config: Config) -> None:
    pool = Pool()
    registry = WorkspaceRegistry(config, pool.factory, max_resident=1)
    release = anyio.Event()
    entered = anyio.Event()
    seen: list[MemoryManager] = []

    async def worker() -> None:
        async with registry.acquire("Shared") as manager:
            seen.append(manager)
            entered.set()
            await release.wait()

    async with anyio.create_task_group() as group:
        for _ in range(12):
            group.start_soon(worker)
        await entered.wait()
        release.set()
    assert pool.events == ["build:shared"]
    assert len(seen) == 12
    assert all(manager is pool.managers["shared"] for manager in seen)
    await registry.close()


@pytest.mark.unit
async def test_simultaneous_new_keys_never_exceed_the_limit(config: Config) -> None:
    pool = Pool()
    registry = WorkspaceRegistry(config, pool.factory, max_resident=2)
    release = anyio.Event()
    settled = anyio.Event()
    outcomes: list[str] = []

    def record(outcome: str) -> None:
        outcomes.append(outcome)
        if len(outcomes) == 5:
            settled.set()

    async def worker(index: int) -> None:
        try:
            async with registry.acquire(f"key-{index}"):
                record("held")
                await release.wait()
        except WorkspaceCapacityError:
            record("rejected")

    async with anyio.create_task_group() as group:
        for index in range(5):
            group.start_soon(worker, index)
        with anyio.fail_after(5):
            await settled.wait()
        assert outcomes.count("held") == 2
        assert outcomes.count("rejected") == 3
        assert len(pool.events) == 2
        release.set()
    assert pool.peak <= 2
    await registry.close()
    assert pool.alive == 0


@pytest.mark.unit
async def test_burst_over_idle_managers_stays_within_the_limit(config: Config) -> None:
    pool = Pool()
    now = [0.0]
    registry = WorkspaceRegistry(
        config, pool.factory, clock=lambda: now[0], max_resident=2
    )
    async with registry.acquire("a"):
        pass
    now[0] = 1
    async with registry.acquire("b"):
        pass

    async def worker(key: str) -> None:
        async with registry.acquire(key):
            pass

    async with anyio.create_task_group() as group:
        group.start_soon(worker, "c")
        group.start_soon(worker, "d")
    assert pool.peak <= 2
    assert pool.events.count("close:a") == 1
    assert pool.events.count("close:b") == 1
    assert "build:c" in pool.events
    assert "build:d" in pool.events
    assert not was_closed(pool.managers["c"])
    assert not was_closed(pool.managers["d"])
    await registry.close()
    assert pool.alive == 0
    assert all(manager.close.call_count == 1 for manager in pool.managers.values())


@pytest.mark.unit
async def test_unrelated_use_is_not_blocked_by_pressure_eviction(
    config: Config,
) -> None:
    pool = Pool()
    started, gate = pool.hold_close("a")
    now = [0.0]
    registry = WorkspaceRegistry(
        config, pool.factory, clock=lambda: now[0], max_resident=2
    )
    async with registry.acquire("a"):
        pass
    now[0] = 1
    async with registry.acquire("c"):
        pass

    async def newcomer() -> None:
        async with registry.acquire("b"):
            pass

    newcomer_task = asyncio.create_task(newcomer())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        with anyio.fail_after(1):
            async with registry.acquire("c") as resident:
                assert resident is pool.managers["c"]
        assert "build:b" not in pool.events
        assert not newcomer_task.done()
    finally:
        gate.set()
        await newcomer_task
        await registry.close()
    assert pool.events.index("close:a") < pool.events.index("build:b")
    assert pool.peak <= 2


@pytest.mark.unit
async def test_factory_failure_releases_its_reservation(config: Config) -> None:
    pool = Pool()
    pool.build_failures.add("bad")
    registry = WorkspaceRegistry(config, pool.factory, max_resident=1)
    with pytest.raises(OSError, match="constructor failed"):
        async with registry.acquire("bad"):
            pytest.fail("Failed construction returned a manager")
    async with registry.acquire("good") as manager:
        assert manager.config.workspace_id == "good"
    assert pool.events == ["build:bad", "build:good"]
    await registry.close()


@pytest.mark.unit
async def test_cancelled_construction_stays_counted_until_closed(
    config: Config,
) -> None:
    pool = Pool()
    started, gate = pool.hold_build("slow")
    registry = WorkspaceRegistry(config, pool.factory, max_resident=1)

    async def user() -> None:
        async with registry.acquire("slow"):
            pytest.fail("Cancelled acquisition returned a manager")

    user_task = asyncio.create_task(user())
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        user_task.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await user_task
        assert registry._entries["slow"].active == 0
        async with registry.acquire("other"):
            pass
        assert pool.events == ["build:slow", "close:slow", "build:other"]
        assert pool.managers["slow"].close.call_count == 1
        await registry.close()
    finally:
        gate.set()


@pytest.mark.unit
async def test_failed_pressure_close_keeps_the_slot_counted(config: Config) -> None:
    pool = Pool()
    pool.close_failures["old"] = 1
    registry = WorkspaceRegistry(config, pool.factory, max_resident=1)
    async with registry.acquire("old") as old:
        pass
    with pytest.raises(WorkspaceCapacityError) as failure:
        async with registry.acquire("new"):
            pytest.fail("Capacity was not freed")
    assert "old" not in str(failure.value)
    assert "storage unavailable" not in str(failure.value)
    assert pool.events == ["build:old", "close-failed:old"]
    with pytest.raises(RuntimeError, match="pending close"):
        async with registry.acquire("OLD"):
            pytest.fail("Quarantined manager was reused")
    assert not was_closed(old)
    await registry.close()
    assert was_closed(old)
    assert pool.managers["old"].close.call_count == 2
    assert pool.events == ["build:old", "close-failed:old", "close:old"]


@pytest.mark.unit
async def test_failed_pressure_close_moves_on_to_next_idle_manager(
    config: Config,
) -> None:
    pool = Pool()
    pool.close_failures["bad"] = 1
    now = [0.0]
    registry = WorkspaceRegistry(
        config, pool.factory, clock=lambda: now[0], max_resident=2
    )
    async with registry.acquire("bad") as bad:
        pass
    now[0] = 1
    async with registry.acquire("good"):
        pass
    async with registry.acquire("new"):
        pass
    assert pool.events == [
        "build:bad",
        "build:good",
        "close-failed:bad",
        "close:good",
        "build:new",
    ]
    assert pool.peak <= 2
    await registry.close()
    assert was_closed(bad)
    assert pool.alive == 0


@pytest.mark.unit
async def test_shutdown_overlapping_pressure_eviction_rejects_the_waiter(
    config: Config,
) -> None:
    pool = Pool()
    started, gate = pool.hold_close("old")
    registry = WorkspaceRegistry(config, pool.factory, max_resident=1)
    async with registry.acquire("old"):
        pass

    async def requester() -> None:
        with pytest.raises(
            RuntimeError, match="WorkspaceRegistry is closed"
        ) as failure:
            async with registry.acquire("new"):
                pytest.fail("Closing registry built a manager")
        assert not isinstance(failure.value, WorkspaceCapacityError)

    requester_task = asyncio.create_task(requester())
    close_task: asyncio.Task[None] | None = None
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        close_task = asyncio.create_task(registry.close())
        while not registry._closing:
            await anyio.lowlevel.checkpoint()
        assert not close_task.done()
    finally:
        gate.set()
        await requester_task
        if close_task is not None:
            await close_task
    assert pool.events == ["build:old", "close:old"]
    assert pool.managers["old"].close.call_count == 1


@pytest.mark.unit
async def test_eviction_in_flight_keeps_counting_against_limit(config: Config) -> None:
    pool = Pool()
    started, gate = pool.hold_close("a")
    checked = anyio.Event()
    observe_request = False

    def clock() -> float:
        if observe_request:
            checked.set()
        return 0.0

    registry = WorkspaceRegistry(config, pool.factory, clock=clock, max_resident=1)
    async with registry.acquire("a"):
        pass
    release = anyio.Event()
    settled = anyio.Event()
    held: list[str] = []
    rejected: list[str] = []

    def record_settled() -> None:
        if len(held) + len(rejected) == 2:
            settled.set()

    async def requester(key: str) -> None:
        try:
            async with registry.acquire(key):
                held.append(key)
                record_settled()
                await release.wait()
        except WorkspaceCapacityError:
            rejected.append(key)
            record_settled()

    tasks = [asyncio.create_task(requester("b"))]
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        observe_request = True
        tasks.append(asyncio.create_task(requester("c")))
        with anyio.fail_after(5):
            await checked.wait()
        # The clock handshake runs inside pruning under the acquisition lock.
        # Admission would already have reserved a build before yielding.
        assert registry._building == {}
        assert pool.events == ["build:a"]
        assert pool.alive == pool.peak == 1
        assert not tasks[1].done()
        assert held == rejected == []

        gate.set()
        with anyio.fail_after(5):
            await settled.wait()
        assert len(held) == len(rejected) == 1
        assert set(held + rejected) == {"b", "c"}
        assert pool.events == ["build:a", "close:a", f"build:{held[0]}"]
        assert f"build:{rejected[0]}" not in pool.events
        assert pool.alive == pool.peak == 1
    finally:
        gate.set()
        release.set()
        await asyncio.gather(*tasks)
        await registry.close()
    assert pool.alive == 0
    assert all(manager.close.call_count == 1 for manager in pool.managers.values())
    assert all(was_closed(manager) for manager in pool.managers.values())


@pytest.mark.unit
async def test_capacity_awaits_idle_ttl_eviction_already_in_flight(
    config: Config,
) -> None:
    pool = Pool()
    started, gate = pool.hold_close("a")
    now = [0.0]
    checked = anyio.Event()
    observe_request = False

    def clock() -> float:
        if observe_request:
            checked.set()
        return now[0]

    registry = WorkspaceRegistry(
        config, pool.factory, clock=clock, idle_ttl=900, max_resident=1
    )
    async with registry.acquire("a"):
        pass
    now[0] = 901.0
    admitted = anyio.Event()

    async def requester() -> None:
        async with registry.acquire("b") as manager:
            assert manager is pool.managers["b"]
            admitted.set()

    prune_task = asyncio.create_task(registry.prune())
    requester_task: asyncio.Task[None] | None = None
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        observe_request = True
        requester_task = asyncio.create_task(requester())
        with anyio.fail_after(5):
            await checked.wait()
        assert not requester_task.done()
        assert not admitted.is_set()
        assert pool.events == ["build:a"]
        assert registry._building == {}

        gate.set()
        await prune_task
        with anyio.fail_after(5):
            await requester_task
            assert admitted.is_set()
        assert pool.events == ["build:a", "close:a", "build:b"]
        assert pool.alive == pool.peak == 1
    finally:
        gate.set()
        await prune_task
        try:
            if requester_task is not None:
                await requester_task
        finally:
            await registry.close()
    assert pool.alive == 0
    assert all(was_closed(manager) for manager in pool.managers.values())


@pytest.mark.unit
async def test_waiter_notices_capacity_freed_by_another_eviction(
    config: Config,
) -> None:
    pool = Pool()
    a_started, a_gate = pool.hold_close("a")
    c_started, c_gate = pool.hold_build("c")
    now = [0.0]
    registry = WorkspaceRegistry(
        config, pool.factory, clock=lambda: now[0], idle_ttl=900, max_resident=2
    )
    async with registry.acquire("a"):
        pass
    now[0] = 1
    b_held = anyio.Event()
    b_release = anyio.Event()
    c_release = anyio.Event()

    async def holder() -> None:
        async with registry.acquire("b"):
            b_held.set()
            await b_release.wait()

    async def requester() -> None:
        async with registry.acquire("c"):
            await c_release.wait()

    holder_task = asyncio.create_task(holder())
    requester_task: asyncio.Task[None] | None = None
    try:
        await b_held.wait()
        requester_task = asyncio.create_task(requester())
        assert await anyio.to_thread.run_sync(a_started.wait, 5)
        b_release.set()
        await holder_task
        now[0] = 1000
        await registry.prune()
        assert pool.events == ["build:a", "build:b", "close:b"]
        assert await anyio.to_thread.run_sync(c_started.wait, 5)
        assert not was_closed(pool.managers["a"])
    finally:
        a_gate.set()
        c_gate.set()
        b_release.set()
        c_release.set()
        await holder_task
        if requester_task is not None:
            await requester_task
        await registry.close()
    assert pool.events[:4] == ["build:a", "build:b", "close:b", "build:c"]
    assert sorted(pool.events[4:]) == ["close:a", "close:c"]
    assert pool.alive == 0
    assert all(manager.close.call_count == 1 for manager in pool.managers.values())
    assert registry._quarantined == {}


@pytest.mark.unit
async def test_waiter_notices_capacity_freed_by_failed_build(config: Config) -> None:
    pool = Pool()
    pool.build_failures.add("x")
    x_started, x_gate = pool.hold_build("x")
    a_started, a_gate = pool.hold_close("a")
    c_started, c_gate = pool.hold_build("c")
    registry = WorkspaceRegistry(config, pool.factory, max_resident=2)
    async with registry.acquire("a"):
        pass

    async def failing_build() -> None:
        with pytest.raises(OSError, match="constructor failed"):
            async with registry.acquire("x"):
                pytest.fail("Failed construction returned a manager")

    async def requester() -> None:
        async with registry.acquire("c"):
            pass

    builder_task = asyncio.create_task(failing_build())
    requester_task: asyncio.Task[None] | None = None
    try:
        assert await anyio.to_thread.run_sync(x_started.wait, 5)
        requester_task = asyncio.create_task(requester())
        assert await anyio.to_thread.run_sync(a_started.wait, 5)
        x_gate.set()
        await builder_task
        assert await anyio.to_thread.run_sync(c_started.wait, 5)
        assert not was_closed(pool.managers["a"])
    finally:
        x_gate.set()
        a_gate.set()
        c_gate.set()
        await builder_task
        if requester_task is not None:
            await requester_task
        await registry.close()
    assert pool.alive == 0
    assert pool.events.count("close:a") == pool.events.count("close:c") == 1
    assert registry._quarantined == {}


@pytest.mark.unit
async def test_waiter_on_failing_in_flight_eviction_gets_capacity_error(
    config: Config,
) -> None:
    pool = Pool()
    started, gate = pool.hold_close("a")
    pool.close_failures["a"] = 1
    now = [0.0]
    checked = anyio.Event()
    observe_request = False

    def clock() -> float:
        if observe_request:
            checked.set()
        return now[0]

    registry = WorkspaceRegistry(
        config, pool.factory, clock=clock, idle_ttl=900, max_resident=1
    )
    async with registry.acquire("a"):
        pass
    now[0] = 901.0

    async def requester() -> None:
        with pytest.raises(WorkspaceCapacityError):
            async with registry.acquire("b"):
                pytest.fail("Capacity was not freed")

    prune_task = asyncio.create_task(registry.prune())
    requester_task: asyncio.Task[None] | None = None
    try:
        assert await anyio.to_thread.run_sync(started.wait, 5)
        observe_request = True
        requester_task = asyncio.create_task(requester())
        with anyio.fail_after(5):
            await checked.wait()
        assert not requester_task.done()
        gate.set()
        await prune_task
        await requester_task
        assert pool.events == ["build:a", "close-failed:a"]
    finally:
        gate.set()
        await prune_task
        if requester_task is not None:
            await requester_task
        await registry.close()
    assert pool.events == ["build:a", "close-failed:a", "close:a"]


class PinFailure(Exception):
    """Distinctive error raised by the patched WeMM pin."""


class WemmManager:
    """Stands in for `MemoryManager` while `workspace_registry.MemoryManager` is patched."""

    def __init__(self, config: Config, close_failures: int) -> None:
        self.config = config
        self.close_calls = 0
        self.close_failures = close_failures

    def close(self) -> None:
        self.close_calls += 1
        if self.close_failures > 0:
            self.close_failures -= 1
            raise OSError("storage unavailable")


class WemmWorld:
    """Patches the registry's WeMM pin so the first `pin_failures` pins raise."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, config: Config, pin_failures: int
    ) -> None:
        self.config = replace(config, embedder_provider=EmbedderProvider.WEMM)
        self.pin_failures = pin_failures
        self.close_failures = 0
        self.managers: list[WemmManager] = []
        self.pins: list[MagicMock] = []
        self.build_started = threading.Event()
        self.build_gate: threading.Event | None = None
        monkeypatch.setattr(workspace_registry, "MemoryManager", WemmManager)
        monkeypatch.setattr(workspace_registry, "wemm_config", lambda config: config)
        monkeypatch.setattr(workspace_registry, "acquire_wemm", self.acquire_pin)

    def factory(self, concrete: Config) -> MemoryManager:
        self.build_started.set()
        if self.build_gate is not None and len(self.managers) == 0:
            assert self.build_gate.wait(timeout=5)
        manager = WemmManager(concrete, self.close_failures)
        self.close_failures = 0
        self.managers.append(manager)
        return cast("MemoryManager", manager)

    def acquire_pin(self, _config: object) -> WeMMBorrow:
        if self.pin_failures > 0:
            self.pin_failures -= 1
            raise PinFailure("pin unavailable")
        pin = MagicMock(spec=WeMMBorrow)
        self.pins.append(pin)
        return pin

    def registry(self, **options: Any) -> WorkspaceRegistry:
        return WorkspaceRegistry(self.config, self.factory, **options)


@pytest.mark.unit
async def test_pin_failure_releases_reservation_and_closes_manager(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = WemmWorld(monkeypatch, config, pin_failures=1)
    registry = world.registry(max_resident=1)
    with pytest.raises(PinFailure, match="pin unavailable"):
        async with registry.acquire("one"):
            pytest.fail("Failed pin returned a manager")
    assert [manager.close_calls for manager in world.managers] == [1]
    assert registry._building == {}
    assert registry._evicting == {}
    async with registry.acquire("one") as rebuilt:
        assert rebuilt is world.managers[1]
    async with registry.acquire("two") as other:
        assert other is world.managers[2]
    await registry.close()
    assert [manager.close_calls for manager in world.managers] == [1, 1, 1]
    assert len(world.pins) == 1
    assert world.pins[0].close.call_count == 1


@pytest.mark.unit
async def test_same_key_waiter_builds_after_pin_failure(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = WemmWorld(monkeypatch, config, pin_failures=1)
    world.build_gate = threading.Event()
    checked = anyio.Event()
    observe_request = False

    def clock() -> float:
        if observe_request:
            checked.set()
        return 0.0

    registry = world.registry(clock=clock)

    async def request() -> MemoryManager:
        async with registry.acquire("shared") as manager:
            return manager

    failing = asyncio.create_task(request())
    waiter: asyncio.Task[MemoryManager] | None = None
    try:
        assert await anyio.to_thread.run_sync(world.build_started.wait, 5)
        observe_request = True
        waiter = asyncio.create_task(request())
        with anyio.fail_after(5):
            await checked.wait()
        world.build_gate.set()
        done, _ = await asyncio.wait({failing, waiter}, timeout=5)
        assert done == {failing, waiter}
        with pytest.raises(PinFailure):
            failing.result()
        assert waiter.result() is world.managers[1]
        assert world.managers[0].close_calls == 1
        assert registry._building == {}
    finally:
        world.build_gate.set()
        tasks = [failing] if waiter is None else [failing, waiter]
        _ = await asyncio.gather(*tasks, return_exceptions=True)
        await registry.close()
    assert [manager.close_calls for manager in world.managers] == [1, 1]


@pytest.mark.unit
async def test_pin_failure_with_failing_close_stays_counted_and_retried(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = WemmWorld(monkeypatch, config, pin_failures=1)
    world.close_failures = 1
    registry = world.registry(max_resident=1)
    with pytest.raises(PinFailure, match="pin unavailable"):
        async with registry.acquire("one"):
            pytest.fail("Failed pin returned a manager")
    assert [manager.close_calls for manager in world.managers] == [1]
    with pytest.raises(WorkspaceCapacityError):
        async with registry.acquire("two"):
            pytest.fail("Quarantined manager freed its slot")
    with pytest.raises(RuntimeError, match="pending close"):
        async with registry.acquire("one"):
            pytest.fail("Quarantined manager was reused")
    assert len(world.managers) == 1
    await registry.close()
    assert [manager.close_calls for manager in world.managers] == [2]
    await registry.close()
    assert world.managers[0].close_calls == 2
