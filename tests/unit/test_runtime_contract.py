"""Runtime contract for pinned FastMCP, Pydantic, ranx, and Portalocker."""

from __future__ import annotations

from importlib.metadata import metadata, version
import multiprocessing
import multiprocessing.synchronize
from pathlib import Path
import tomllib
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

APPROVED_PINS = {
    "fastmcp": "4.0.10",
    "pydantic": "2.14.0b2",
    "pydantic-core": "2.49.0",
    "ranx": "0.3.21",
    "portalocker": "4.4.0",
}


def _project_dependencies() -> list[str]:
    payload = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    project = payload["project"]
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    return [str(item) for item in dependencies]


def _hold_exclusive_lock(
    lock_path: str,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    import portalocker

    lock = portalocker.Lock(
        lock_path,
        timeout=5.0,
        flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
    )
    lock.acquire()
    try:
        ready.set()
        _ = release.wait(timeout=30.0)
    finally:
        lock.release()


def test_manifest_declares_approved_runtime_contract() -> None:
    dependencies = _project_dependencies()
    for name, pinned in APPROVED_PINS.items():
        expected = f"{name}=={pinned}"
        assert expected in dependencies, (
            f"pyproject.toml must declare exact pin {expected}"
        )


def test_installed_versions_match_approved_contract() -> None:
    for name, pinned in APPROVED_PINS.items():
        assert version(name) == pinned


def test_fastmcp_construction() -> None:
    from fastmcp import FastMCP

    server = FastMCP(name="reflectlog-runtime-contract")
    assert server is not None


def test_numba_scoring_annotations_import() -> None:
    """Numba inspects JIT signatures at import; NDArray must be a runtime name."""
    from reflectlog.utility.scoring import warmup_numba_functions

    warmup_numba_functions()


@runtime_checkable
class NamedProtocol(Protocol):
    name: str


class NamedModel(BaseModel):
    name: str


def test_pydantic_protocol_validation() -> None:
    model = NamedModel.model_validate({"name": "reflectlog"})
    assert model.name == "reflectlog"
    assert isinstance(model, NamedProtocol)
    with pytest.raises(ValidationError):
        _ = NamedModel.model_validate({"name": 1})


def test_ranx_metadata() -> None:
    info = metadata("ranx")
    assert info["Name"] == "ranx"
    assert info["Version"] == APPROVED_PINS["ranx"]


def test_portalocker_timeout_then_acquire_after_owner_exit(
    tmp_path: Path,
) -> None:
    import portalocker
    from portalocker.exceptions import AlreadyLocked

    lock_path = tmp_path / ".reflectlog.writer.lock"
    lock_path.touch()
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    child = ctx.Process(
        target=_hold_exclusive_lock,
        args=(str(lock_path), ready, release),
    )
    child.start()
    try:
        assert ready.wait(timeout=10.0)
        contended = portalocker.Lock(
            str(lock_path),
            timeout=0.2,
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        )
        with pytest.raises(AlreadyLocked):
            contended.acquire()
        assert lock_path.exists()
    finally:
        release.set()
        child.join(timeout=10.0)
        if child.is_alive():
            child.terminate()
            child.join(timeout=5.0)
        assert child.exitcode == 0

    winner = portalocker.Lock(
        str(lock_path),
        timeout=5.0,
        flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
    )
    winner.acquire()
    try:
        assert lock_path.exists()
    finally:
        winner.release()
    assert lock_path.exists()
