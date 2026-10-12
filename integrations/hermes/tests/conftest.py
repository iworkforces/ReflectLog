"""Load the plugin under a private package name without changing sys.path."""

from __future__ import annotations

from importlib import import_module, util
from pathlib import Path
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

PLUGIN_PACKAGE = "reflectlog_hermes_plugin"
PLUGIN_DIR = Path(__file__).resolve().parents[1]


def load_plugin_package(plugin_dir: Path, package_name: str) -> ModuleType:
    """Support the Wave 0 namespace and the later lazy package initializer."""
    initializer = plugin_dir / "__init__.py"
    spec = util.spec_from_file_location(
        package_name, initializer, submodule_search_locations=[str(plugin_dir)]
    )
    if initializer.is_file():
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the plugin package")
        package = util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    else:
        package = ModuleType(package_name)
        package.__path__ = [str(plugin_dir)]
        package.__package__ = package_name
        package.__spec__ = spec
        sys.modules[package_name] = package
    return package


@pytest.fixture(scope="session")
def plugin_import() -> Callable[[str], ModuleType]:
    """Import only the requested submodule; never import Hermes-bound layers."""
    _ = load_plugin_package(PLUGIN_DIR, PLUGIN_PACKAGE)

    def import_submodule(name: str) -> ModuleType:
        return import_module(f"{PLUGIN_PACKAGE}.{name}")

    return import_submodule


@pytest.fixture(scope="session")
def models(plugin_import: Callable[[str], ModuleType]) -> ModuleType:
    return plugin_import("core.models")


@pytest.fixture
def hermes_provider(plugin_import, tmp_path, monkeypatch):
    """Use the real Hermes boundary and the existing wire-level fake."""
    import httpx

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("REFLECTLOG_MCP_TOKEN", " qa-token \n")
    fake = plugin_import("tests.support.fake_server").FakeReflectLogServer()
    providers = []

    def create(settings=None, **context):
        plugin_import("core.config").save_settings(settings or {}, tmp_path)
        provider = plugin_import("provider").ReflectLogMemoryProvider(
            transport=httpx.MockTransport(fake)
        )
        provider.initialize("session", hermes_home=str(tmp_path), **context)
        providers.append(provider)
        return provider, fake

    yield create
    for provider in providers:
        provider.shutdown()
