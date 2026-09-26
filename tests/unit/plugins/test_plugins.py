"""Unit tests for reflectlog/plugins/ package.

Tests plugin discovery, loading, registry, and lifecycle management.
"""

import importlib
import pkgutil
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reflectlog.plugins.discovery import (
    CompositeDiscovery,
    DirectoryScanDiscovery,
    DiscoveredPlugin,
    EntryPointDiscovery,
    PluginDiscoverer,
    PluginDiscoveryStrategy,
    StaticRegistration,
    load_plugin,
)
from reflectlog.plugins.loading import (
    IPluginLifecycle,
    LifecycleHooks,
    PluginLoader,
)
from reflectlog.plugins.registry import (
    IPluggable,
    PluginCapability,
    PluginMetadata,
    PluginRegistry,
    PluginState,
    ToolRegistry,
    utc_now,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _DummyPlugin:
    """Simple plugin stub without lifecycle support."""

    pass


class _PluggablePlugin:
    """Plugin implementing IPluggable protocol."""

    @property
    def plugin_name(self) -> str:
        return "my_pluggable"

    @property
    def plugin_version(self) -> str:
        return "1.2.3"


class _LifecyclePlugin:
    """Plugin implementing IPluginLifecycle protocol."""

    def __init__(self) -> None:
        self.initialized = False
        self.activated = False
        self.deactivated = False
        self.cleaned_up = False

    async def initialize(self) -> None:
        self.initialized = True

    async def activate(self) -> None:
        self.activated = True

    async def deactivate(self) -> None:
        self.deactivated = True

    async def cleanup(self) -> None:
        self.cleaned_up = True


class _FailingLifecyclePlugin:
    """Plugin whose lifecycle methods raise."""

    async def initialize(self) -> None:
        raise RuntimeError("init boom")

    async def activate(self) -> None:
        raise RuntimeError("activate boom")

    async def deactivate(self) -> None:
        raise RuntimeError("deactivate boom")

    async def cleanup(self) -> None:
        raise RuntimeError("cleanup boom")


type TestPlugin = _DummyPlugin | _LifecyclePlugin | _FailingLifecyclePlugin


# ---------------------------------------------------------------------------
# registry.py — utc_now
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUtcNow:
    """Tests for utc_now helper."""

    def test_returns_timezone_aware_datetime(self) -> None:
        """utc_now should return a timezone-aware datetime."""
        now = utc_now()
        assert now.tzinfo is not None

    def test_returns_utc_timezone(self) -> None:
        """utc_now should use UTC timezone."""
        from datetime import UTC

        now = utc_now()
        assert now.tzinfo is UTC


# ---------------------------------------------------------------------------
# registry.py — PluginState
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginState:
    """Tests for PluginState enum."""

    def test_all_states_exist(self) -> None:
        """Verify all expected lifecycle states."""
        expected = {
            "DISCOVERED",
            "LOADED",
            "ACTIVATED",
            "DEACTIVATED",
            "UNLOADED",
            "ERROR",
        }
        assert {s.name for s in PluginState} == expected

    def test_state_values(self) -> None:
        """Verify state string values."""
        assert PluginState.DISCOVERED.value == "discovered"
        assert PluginState.ERROR.value == "error"


# ---------------------------------------------------------------------------
# registry.py — PluginCapability / PluginMetadata
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginMetadata:
    """Tests for PluginCapability and PluginMetadata dataclasses."""

    def test_capability_defaults(self) -> None:
        """PluginCapability defaults version to "0.0.0"."""
        cap = PluginCapability(name="search")
        assert cap.version == "0.0.0"

    def test_metadata_defaults(self) -> None:
        """PluginMetadata defaults should be sensible."""
        meta = PluginMetadata(name="test", version="1.0")
        assert meta.state == PluginState.DISCOVERED
        assert meta.error_message is None
        assert meta.loaded_at is None
        assert meta.activated_at is None
        assert meta.capabilities == []
        assert meta.dependencies == []

    def test_metadata_discovered_at_auto(self) -> None:
        """discovered_at should be auto-populated."""
        meta = PluginMetadata(name="t", version="0")
        assert meta.discovered_at is not None


# ---------------------------------------------------------------------------
# registry.py — IPluggable
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIPluggable:
    """Tests for IPluggable runtime-checkable protocol."""

    def test_pluggable_class_is_instance(self) -> None:
        """A class with plugin_name and plugin_version is IPluggable."""
        assert isinstance(_PluggablePlugin(), IPluggable)

    def test_plain_class_is_not_instance(self) -> None:
        """A plain class is not IPluggable."""
        assert not isinstance(_DummyPlugin(), IPluggable)


# ---------------------------------------------------------------------------
# registry.py — PluginRegistry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_register_auto_metadata(self) -> None:
        """Register with auto-generated metadata."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        plugin = _DummyPlugin()
        meta = reg.register(plugin)

        assert meta.name == "_DummyPlugin"
        assert meta.state == PluginState.LOADED
        assert meta.loaded_at is not None

    def test_register_pluggable_auto_metadata(self) -> None:
        """Register IPluggable — auto-metadata uses plugin_name/version."""
        reg: PluginRegistry[_PluggablePlugin] = PluginRegistry()
        plugin = _PluggablePlugin()
        meta = reg.register(plugin)

        assert meta.name == "my_pluggable"
        assert meta.version == "1.2.3"

    def test_register_custom_metadata(self) -> None:
        """Register with explicit metadata."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        custom_meta = PluginMetadata(name="custom", version="9.9.9")
        meta = reg.register(_DummyPlugin(), metadata=custom_meta)

        assert meta.name == "custom"
        assert meta.version == "9.9.9"
        assert meta.state == PluginState.LOADED

    def test_get_existing(self) -> None:
        """get() returns registered plugin."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        plugin = _DummyPlugin()
        reg.register(plugin, PluginMetadata(name="p", version="0"))
        assert reg.get("p") is plugin

    def test_get_missing_returns_none(self) -> None:
        """get() returns None for missing plugin."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        assert reg.get("nonexistent") is None

    def test_get_metadata(self) -> None:
        """get_metadata() returns metadata."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="p", version="1"))
        meta = reg.get_metadata("p")
        assert meta is not None
        assert meta.version == "1"

    def test_get_metadata_missing(self) -> None:
        """get_metadata() returns None for missing."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        assert reg.get_metadata("x") is None

    def test_unregister_existing(self) -> None:
        """unregister() removes plugin and returns True."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="p", version="0"))
        assert reg.unregister("p") is True
        assert reg.get("p") is None
        assert reg.count() == 0

    def test_unregister_missing(self) -> None:
        """unregister() returns False when not found."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        assert reg.unregister("missing") is False

    def test_list_all(self) -> None:
        """list_all() returns all registered names."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="a", version="0"))
        reg.register(_DummyPlugin(), PluginMetadata(name="b", version="0"))
        assert sorted(reg.list_all()) == ["a", "b"]

    def test_list_by_state(self) -> None:
        """list_by_state() filters correctly."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="a", version="0"))
        reg.register(_DummyPlugin(), PluginMetadata(name="b", version="0"))
        reg.activate("a")

        loaded = reg.list_by_state(PluginState.LOADED)
        activated = reg.list_by_state(PluginState.ACTIVATED)
        assert loaded == ["b"]
        assert activated == ["a"]

    def test_list_by_capability(self) -> None:
        """list_by_capability() returns plugins with matching capability."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        meta = PluginMetadata(
            name="search_plugin",
            version="1",
            capabilities=[PluginCapability(name="search")],
        )
        reg.register(_DummyPlugin(), meta)
        reg.register(_DummyPlugin(), PluginMetadata(name="other", version="0"))

        assert reg.list_by_capability("search") == ["search_plugin"]
        assert reg.list_by_capability("nonexistent") == []

    def test_list_by_type(self) -> None:
        """list_by_type() returns instances of the specified type."""
        reg: PluginRegistry[object] = PluginRegistry()
        dummy = _DummyPlugin()
        pluggable = _PluggablePlugin()
        reg.register(dummy, PluginMetadata(name="d", version="0"))
        reg.register(pluggable, PluginMetadata(name="p", version="0"))

        result = reg.list_by_type(_PluggablePlugin)
        assert len(result) == 1
        assert result[0] is pluggable

    def test_activate_success(self) -> None:
        """activate() transitions LOADED → ACTIVATED."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="p", version="0"))
        assert reg.activate("p") is True

        meta = reg.get_metadata("p")
        assert meta is not None
        assert meta.state == PluginState.ACTIVATED
        assert meta.activated_at is not None

    def test_activate_not_loaded_returns_false(self) -> None:
        """activate() returns False when state is not LOADED."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="p", version="0"))
        reg.activate("p")
        # Already ACTIVATED, should fail
        assert reg.activate("p") is False

    def test_activate_missing_returns_false(self) -> None:
        """activate() returns False when plugin not found."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        assert reg.activate("nonexistent") is False

    def test_deactivate_success(self) -> None:
        """deactivate() transitions ACTIVATED → DEACTIVATED."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="p", version="0"))
        reg.activate("p")
        assert reg.deactivate("p") is True

        meta = reg.get_metadata("p")
        assert meta is not None
        assert meta.state == PluginState.DEACTIVATED

    def test_deactivate_not_activated_returns_false(self) -> None:
        """deactivate() returns False when state is not ACTIVATED."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="p", version="0"))
        # State is LOADED, not ACTIVATED
        assert reg.deactivate("p") is False

    def test_deactivate_missing_returns_false(self) -> None:
        """deactivate() returns False when plugin not found."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        assert reg.deactivate("missing") is False

    def test_set_error_success(self) -> None:
        """set_error() sets ERROR state and message."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="p", version="0"))
        assert reg.set_error("p", "something broke") is True

        meta = reg.get_metadata("p")
        assert meta is not None
        assert meta.state == PluginState.ERROR
        assert meta.error_message == "something broke"

    def test_set_error_missing_returns_false(self) -> None:
        """set_error() returns False when plugin not found."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        assert reg.set_error("missing", "err") is False

    def test_clear(self) -> None:
        """clear() removes everything."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="a", version="0"))
        reg.register(_DummyPlugin(), PluginMetadata(name="b", version="0"))
        reg.clear()
        assert reg.count() == 0
        assert reg.list_all() == []

    def test_count(self) -> None:
        """count() returns number of registered plugins."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        assert reg.count() == 0
        reg.register(_DummyPlugin(), PluginMetadata(name="a", version="0"))
        assert reg.count() == 1

    def test_plugins_property_returns_copy(self) -> None:
        """plugins property returns a copy, not the internal dict."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="p", version="0"))
        copy = reg.plugins
        copy["injected"] = PluginMetadata(name="bad", version="0")
        assert "injected" not in reg.plugins

    def test_instances_property_returns_copy(self) -> None:
        """instances property returns a copy, not the internal dict."""
        reg: PluginRegistry[_DummyPlugin] = PluginRegistry()
        reg.register(_DummyPlugin(), PluginMetadata(name="p", version="0"))
        copy = reg.instances
        copy["injected"] = _DummyPlugin()
        assert "injected" not in reg.instances


# ---------------------------------------------------------------------------
# registry.py — ToolRegistry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self) -> None:
        """register_tool() creates metadata with tool capability."""
        reg: ToolRegistry[_DummyPlugin] = ToolRegistry()
        meta = reg.register_tool(
            _DummyPlugin(), name="my_tool", description="A tool", version="2.0"
        )
        assert meta.name == "my_tool"
        assert meta.version == "2.0"
        assert meta.description == "A tool"
        assert any(c.name == "tool" for c in meta.capabilities)

    def test_register_tool_duplicate_raises(self) -> None:
        """register_tool() raises ValueError for duplicate name."""
        reg: ToolRegistry[_DummyPlugin] = ToolRegistry()
        reg.register_tool(_DummyPlugin(), name="dup")
        with pytest.raises(ValueError, match="already registered"):
            reg.register_tool(_DummyPlugin(), name="dup")

    def test_unregister_tool(self) -> None:
        """unregister_tool() removes tool from both registries."""
        reg: ToolRegistry[_DummyPlugin] = ToolRegistry()
        reg.register_tool(_DummyPlugin(), name="t")
        assert reg.unregister_tool("t") is True
        assert reg.has_tool("t") is False
        assert reg.get("t") is None

    def test_unregister_tool_missing(self) -> None:
        """unregister_tool() returns False when tool not found."""
        reg: ToolRegistry[_DummyPlugin] = ToolRegistry()
        assert reg.unregister_tool("missing") is False

    def test_get_tool(self) -> None:
        """get_tool() returns the registered tool."""
        reg: ToolRegistry[_DummyPlugin] = ToolRegistry()
        plugin = _DummyPlugin()
        reg.register_tool(plugin, name="t")
        assert reg.get_tool("t") is plugin

    def test_get_tool_missing(self) -> None:
        """get_tool() returns None for unregistered tool."""
        reg: ToolRegistry[_DummyPlugin] = ToolRegistry()
        assert reg.get_tool("nope") is None

    def test_list_tool_names(self) -> None:
        """list_tool_names() returns sorted tool names."""
        reg: ToolRegistry[_DummyPlugin] = ToolRegistry()
        reg.register_tool(_DummyPlugin(), name="zebra")
        reg.register_tool(_DummyPlugin(), name="alpha")
        assert reg.list_tool_names() == ["alpha", "zebra"]

    def test_has_tool(self) -> None:
        """has_tool() returns True/False correctly."""
        reg: ToolRegistry[_DummyPlugin] = ToolRegistry()
        reg.register_tool(_DummyPlugin(), name="exists")
        assert reg.has_tool("exists") is True
        assert reg.has_tool("nope") is False


# ---------------------------------------------------------------------------
# discovery.py — DiscoveredPlugin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDiscoveredPlugin:
    """Tests for DiscoveredPlugin dataclass."""

    def test_defaults(self) -> None:
        """Verify default field values."""
        dp = DiscoveredPlugin(name="p", module_path="mod", class_name="Cls")
        assert dp.version == "0.0.0"
        assert dp.entry_point is None

    def test_with_entry_point(self) -> None:
        """entry_point can be set."""
        dp = DiscoveredPlugin(
            name="p",
            module_path="mod",
            class_name="Cls",
            entry_point="mod:Cls",
        )
        assert dp.entry_point == "mod:Cls"


# ---------------------------------------------------------------------------
# discovery.py — PluginDiscoveryStrategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginDiscoveryStrategy:
    """Tests for base PluginDiscoveryStrategy."""

    async def test_base_raises_not_implemented(self) -> None:
        """Base discover() raises NotImplementedError."""
        strategy: PluginDiscoveryStrategy[_DummyPlugin] = PluginDiscoveryStrategy()
        with pytest.raises(NotImplementedError):
            await strategy.discover()


# ---------------------------------------------------------------------------
# discovery.py — EntryPointDiscovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntryPointDiscovery:
    """Tests for EntryPointDiscovery."""

    async def test_discovers_from_entry_points(self) -> None:
        """Should parse entry points with colon-separated value."""
        mock_ep = MagicMock()
        mock_ep.name = "my_plugin"
        mock_ep.value = "some.module:MyClass"

        def entry_point_repr(_: MagicMock) -> str:
            return "my_plugin = some.module:MyClass"

        mock_ep.configure_mock(__str__=entry_point_repr)

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            strategy = EntryPointDiscovery(group="test.group", plugin_type=_DummyPlugin)
            result = await strategy.discover()

        assert len(result) == 1
        assert result[0].name == "my_plugin"
        assert result[0].module_path == "some.module"
        assert result[0].class_name == "MyClass"
        assert result[0].entry_point is not None

    async def test_discovers_module_only_entry_point(self) -> None:
        """Entry point without colon sets class_name to empty string."""
        mock_ep = MagicMock()
        mock_ep.name = "mod_plugin"
        mock_ep.value = "some.module"

        def entry_point_repr(_: MagicMock) -> str:
            return "mod_plugin = some.module"

        mock_ep.configure_mock(__str__=entry_point_repr)

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            strategy = EntryPointDiscovery(group="g", plugin_type=_DummyPlugin)
            result = await strategy.discover()

        assert len(result) == 1
        assert result[0].class_name == ""
        assert result[0].module_path == "some.module"

    async def test_type_error_fallback_with_select(self) -> None:
        """Handles TypeError with SelectableGroups that have select()."""
        mock_eps = MagicMock()
        mock_eps.select.return_value = []

        with patch(
            "importlib.metadata.entry_points",
            side_effect=[TypeError("bad"), mock_eps],
        ) as mock_fn:
            # First call raises TypeError; second call returns mock_eps
            mock_fn.side_effect = None
            mock_fn.side_effect = TypeError("bad")

            # Patch differently: the first call with group= raises,
            # second call without returns mock_eps
            strategy = EntryPointDiscovery(group="g", plugin_type=_DummyPlugin)

            # Patch at module level
            with patch(
                "reflectlog.plugins.discovery.importlib.metadata.entry_points"
            ) as inner_mock:
                inner_mock.side_effect = [TypeError("bad"), mock_eps]
                result = await strategy.discover()

            assert result == []

    async def test_type_error_fallback_without_select(self) -> None:
        """Handles TypeError fallback using list filtering."""
        mock_ep = MagicMock()
        mock_ep.name = "filtered"
        mock_ep.value = "mod:Cls"
        mock_ep.group = "my_group"

        def entry_point_repr(_: MagicMock) -> str:
            return "filtered = mod:Cls"

        mock_ep.configure_mock(__str__=entry_point_repr)

        # Result without select(), behaves like list
        eps_list = [mock_ep]

        with patch(
            "reflectlog.plugins.discovery.importlib.metadata.entry_points"
        ) as mock_fn:
            # First call (with group=) raises TypeError
            # Second call (without args) returns list
            mock_fn.side_effect = [TypeError("bad"), eps_list]

            strategy = EntryPointDiscovery(group="my_group", plugin_type=_DummyPlugin)
            result = await strategy.discover()

        assert len(result) == 1
        assert result[0].name == "filtered"

    async def test_empty_entry_points(self) -> None:
        """Should return empty list when no entry points found."""
        with patch(
            "reflectlog.plugins.discovery.importlib.metadata.entry_points",
            return_value=[],
        ):
            strategy = EntryPointDiscovery(group="empty", plugin_type=object)
            result = await strategy.discover()

        assert result == []


# ---------------------------------------------------------------------------
# discovery.py — DirectoryScanDiscovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDirectoryScanDiscovery:
    """Tests for DirectoryScanDiscovery."""

    async def test_discovers_subclasses_in_modules(self) -> None:
        """Should find subclasses of plugin_base_class in scanned modules."""

        class BasePlugin:
            pass

        class FoundPlugin(BasePlugin):
            pass

        # Create a fake module
        fake_module = types.ModuleType("fake_pkg.plugin_tool")
        fake_module.FoundPlugin = FoundPlugin  # type: ignore
        fake_module.BasePlugin = BasePlugin  # type: ignore  # should be skipped

        fake_pkg = types.ModuleType("fake_pkg")
        fake_pkg.__file__ = "/fake/fake_pkg/__init__.py"

        _real_import = importlib.import_module

        def _fake_import(name: str, *args: object, **kw: object) -> object:
            if name == "fake_pkg":
                return fake_pkg
            if name == "fake_pkg.plugin_tool":
                return fake_module
            return _real_import(name)

        with (
            patch.object(importlib, "import_module", side_effect=_fake_import),
            patch.object(
                pkgutil,
                "iter_modules",
                return_value=[(None, "fake_pkg.plugin_tool", False)],
            ),
        ):
            strategy = DirectoryScanDiscovery(
                package_names=["fake_pkg"],
                plugin_base_class=BasePlugin,
            )
            result = await strategy.discover()

        assert len(result) == 1
        assert result[0].class_name == "FoundPlugin"
        assert result[0].module_path == "fake_pkg.plugin_tool"

    async def test_skips_packages(self) -> None:
        """Should skip sub-packages (ispkg=True)."""

        class Base:
            pass

        fake_pkg = types.ModuleType("pkg")
        fake_pkg.__file__ = "/fake/pkg/__init__.py"

        _real_import = importlib.import_module

        def _fake_import(name: str, *args: object, **kw: object) -> object:
            if name == "pkg":
                return fake_pkg
            return _real_import(name)

        with (
            patch.object(importlib, "import_module", side_effect=_fake_import),
            patch.object(
                pkgutil,
                "iter_modules",
                return_value=[(None, "pkg.subpackage", True)],
            ),
        ):
            strategy = DirectoryScanDiscovery(
                package_names=["pkg"],
                plugin_base_class=Base,
            )
            result = await strategy.discover()

        assert result == []

    async def test_skips_namespace_packages(self) -> None:
        """Should skip namespace packages (__file__ is None)."""

        class Base:
            pass

        fake_pkg = types.ModuleType("ns_pkg")
        fake_pkg.__file__ = None  # namespace package

        _real_import = importlib.import_module

        def _fake_import(name: str, *args: object, **kw: object) -> object:
            if name == "ns_pkg":
                return fake_pkg
            return _real_import(name)

        with patch.object(importlib, "import_module", side_effect=_fake_import):
            strategy = DirectoryScanDiscovery(
                package_names=["ns_pkg"],
                plugin_base_class=Base,
            )
            result = await strategy.discover()

        assert result == []

    async def test_handles_import_error(self) -> None:
        """Should skip packages that raise ImportError."""

        class Base:
            pass

        _real_import = importlib.import_module

        def _fake_import(name: str, *args: object, **kw: object) -> object:
            if name == "nonexistent_pkg":
                raise ImportError("not found")
            return _real_import(name)

        with patch.object(importlib, "import_module", side_effect=_fake_import):
            strategy = DirectoryScanDiscovery(
                package_names=["nonexistent_pkg"],
                plugin_base_class=Base,
            )
            result = await strategy.discover()

        assert result == []

    async def test_skips_non_type_attributes(self) -> None:
        """Should ignore module attributes that are not types."""

        class Base:
            pass

        fake_module = types.ModuleType("pkg.plugin_x")
        fake_module.some_string = "not a class"  # type: ignore
        fake_module.some_int = 42  # type: ignore

        fake_pkg = types.ModuleType("pkg")
        fake_pkg.__file__ = "/fake/pkg/__init__.py"

        _real_import = importlib.import_module

        def _fake_import(name: str, *args: object, **kw: object) -> object:
            if name == "pkg":
                return fake_pkg
            if name == "pkg.plugin_x":
                return fake_module
            return _real_import(name)

        with (
            patch.object(importlib, "import_module", side_effect=_fake_import),
            patch.object(
                pkgutil,
                "iter_modules",
                return_value=[(None, "pkg.plugin_x", False)],
            ),
        ):
            strategy = DirectoryScanDiscovery(
                package_names=["pkg"],
                plugin_base_class=Base,
            )
            result = await strategy.discover()

        assert result == []


# ---------------------------------------------------------------------------
# discovery.py — StaticRegistration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStaticRegistration:
    """Tests for StaticRegistration."""

    async def test_returns_registered_plugins(self) -> None:
        """Should return a copy of registered plugins."""
        plugins = [
            DiscoveredPlugin(name="a", module_path="m", class_name="A"),
            DiscoveredPlugin(name="b", module_path="m", class_name="B"),
        ]
        strategy = StaticRegistration(plugins)
        result = await strategy.discover()

        assert len(result) == 2
        assert result[0].name == "a"
        assert result[1].name == "b"

    async def test_returns_copy(self) -> None:
        """Returned list should be a copy, not the original."""
        plugins = [
            DiscoveredPlugin(name="x", module_path="m", class_name="X"),
        ]
        strategy = StaticRegistration(plugins)
        result = await strategy.discover()
        result.append(DiscoveredPlugin(name="extra", module_path="m", class_name="E"))

        # Original should be unchanged
        result2 = await strategy.discover()
        assert len(result2) == 1

    async def test_empty_registration(self) -> None:
        """Empty list returns empty list."""
        strategy = StaticRegistration([])
        result = await strategy.discover()
        assert result == []


# ---------------------------------------------------------------------------
# discovery.py — CompositeDiscovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompositeDiscovery:
    """Tests for CompositeDiscovery."""

    async def test_combines_strategies(self) -> None:
        """Should combine results from all strategies."""
        s1 = StaticRegistration(
            [
                DiscoveredPlugin(name="a", module_path="m", class_name="A"),
            ]
        )
        s2 = StaticRegistration(
            [
                DiscoveredPlugin(name="b", module_path="m", class_name="B"),
            ]
        )
        composite = CompositeDiscovery(strategies=[s1, s2])
        result = await composite.discover()

        names = [p.name for p in result]
        assert "a" in names
        assert "b" in names

    async def test_deduplicates_by_name(self) -> None:
        """First discovery wins when duplicate names found."""
        s1 = StaticRegistration(
            [
                DiscoveredPlugin(
                    name="dup", module_path="m1", class_name="A", version="1.0"
                ),
            ]
        )
        s2 = StaticRegistration(
            [
                DiscoveredPlugin(
                    name="dup", module_path="m2", class_name="B", version="2.0"
                ),
            ]
        )
        composite = CompositeDiscovery(strategies=[s1, s2])
        result = await composite.discover()

        assert len(result) == 1
        assert result[0].module_path == "m1"  # first wins
        assert result[0].version == "1.0"

    async def test_empty_strategies(self) -> None:
        """No strategies means no results."""
        composite = CompositeDiscovery(strategies=[])
        result = await composite.discover()
        assert result == []


# ---------------------------------------------------------------------------
# discovery.py — load_plugin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadPlugin:
    """Tests for load_plugin() function."""

    async def test_load_with_class_name(self) -> None:
        """Should import module and instantiate class."""
        mock_module = MagicMock()
        mock_class = MagicMock(return_value="instance")
        mock_module.MyPlugin = mock_class

        with patch.object(importlib, "import_module", return_value=mock_module):
            dp = DiscoveredPlugin(
                name="p", module_path="some.mod", class_name="MyPlugin"
            )
            result = await load_plugin(dp)

        assert result == "instance"
        mock_class.assert_called_once()

    async def test_load_without_class_name(self) -> None:
        """Should return the module when no class_name."""
        mock_module = MagicMock()

        with patch.object(importlib, "import_module", return_value=mock_module):
            dp = DiscoveredPlugin(name="p", module_path="some.mod", class_name="")
            result = await load_plugin(dp)

        assert result is mock_module


# ---------------------------------------------------------------------------
# discovery.py — PluginDiscoverer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginDiscoverer:
    """Tests for PluginDiscoverer class."""

    async def test_discover_plugins(self) -> None:
        """discover_plugins() uses strategy and stores results."""
        plugins = [
            DiscoveredPlugin(name="a", module_path="m", class_name="A"),
        ]
        strategy = StaticRegistration(plugins)
        discoverer = PluginDiscoverer(strategy)

        result = await discoverer.discover_plugins()
        assert len(result) == 1
        assert discoverer.discovered_plugins == result

    async def test_load_plugin_by_name(self) -> None:
        """load_plugin() loads a previously discovered plugin."""
        dp = DiscoveredPlugin(name="p", module_path="some.mod", class_name="Cls")
        strategy = StaticRegistration([dp])
        discoverer = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()

        mock_module = MagicMock()
        mock_module.Cls = MagicMock(return_value="loaded_instance")

        with patch.object(importlib, "import_module", return_value=mock_module):
            instance = await discoverer.load_plugin("p")

        assert instance == "loaded_instance"
        assert "p" in discoverer.loaded_plugins

    async def test_load_plugin_already_loaded(self) -> None:
        """load_plugin() returns cached instance if already loaded."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        strategy = StaticRegistration([dp])
        discoverer = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()

        mock_module = MagicMock()
        mock_module.C = MagicMock(return_value="cached")

        with patch.object(importlib, "import_module", return_value=mock_module):
            first = await discoverer.load_plugin("p")
            second = await discoverer.load_plugin("p")

        assert first is second
        # import_module called only once
        assert mock_module.C.call_count == 1

    async def test_load_plugin_not_discovered(self) -> None:
        """load_plugin() returns None for undiscovered plugin."""
        strategy = StaticRegistration([])
        discoverer = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()

        result = await discoverer.load_plugin("missing")
        assert result is None

    async def test_load_all_plugins(self) -> None:
        """load_all_plugins() loads all discovered plugins."""
        plugins = [
            DiscoveredPlugin(name="a", module_path="m", class_name="A"),
            DiscoveredPlugin(name="b", module_path="m", class_name="B"),
        ]
        strategy = StaticRegistration(plugins)
        discoverer = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()

        mock_module = MagicMock()
        mock_module.A = MagicMock(return_value="inst_a")
        mock_module.B = MagicMock(return_value="inst_b")

        with patch.object(importlib, "import_module", return_value=mock_module):
            loaded = await discoverer.load_all_plugins()

        assert len(loaded) == 2

    async def test_discovered_plugins_returns_copy(self) -> None:
        """discovered_plugins property returns a copy."""
        plugins = [
            DiscoveredPlugin(name="x", module_path="m", class_name="X"),
        ]
        strategy = StaticRegistration(plugins)
        discoverer = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()

        copy = discoverer.discovered_plugins
        copy.append(DiscoveredPlugin(name="y", module_path="m", class_name="Y"))
        assert len(discoverer.discovered_plugins) == 1

    async def test_loaded_plugins_returns_copy(self) -> None:
        """loaded_plugins property returns a copy."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        strategy = StaticRegistration([dp])
        discoverer = PluginDiscoverer(strategy)
        await discoverer.discover_plugins()

        mock_module = MagicMock()
        mock_module.C = MagicMock(return_value="inst")

        with patch.object(importlib, "import_module", return_value=mock_module):
            await discoverer.load_plugin("p")

        copy = discoverer.loaded_plugins
        copy["injected"] = "bad"
        assert "injected" not in discoverer.loaded_plugins


# ---------------------------------------------------------------------------
# loading.py — LifecycleHooks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLifecycleHooks:
    """Tests for LifecycleHooks dataclass."""

    def test_defaults_none(self) -> None:
        """All hooks default to None."""
        hooks = LifecycleHooks()
        assert hooks.on_load is None
        assert hooks.on_initialize is None
        assert hooks.on_activate is None
        assert hooks.on_deactivate is None
        assert hooks.on_unload is None

    def test_custom_hooks(self) -> None:
        """Custom hooks can be set."""

        def cb(name: str) -> None:
            pass

        hooks = LifecycleHooks(on_load=cb, on_activate=cb)
        assert hooks.on_load is cb
        assert hooks.on_activate is cb


# ---------------------------------------------------------------------------
# loading.py — IPluginLifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIPluginLifecycle:
    """Tests for IPluginLifecycle protocol."""

    def test_lifecycle_plugin_is_instance(self) -> None:
        """A class with lifecycle methods passes isinstance check."""
        assert isinstance(_LifecyclePlugin(), IPluginLifecycle)

    def test_plain_class_is_not_instance(self) -> None:
        """A plain class does not implement IPluginLifecycle."""
        assert not isinstance(_DummyPlugin(), IPluginLifecycle)


# ---------------------------------------------------------------------------
# loading.py — PluginLoader
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginLoader:
    """Tests for PluginLoader."""

    def _make_loader(
        self,
        plugins: list[DiscoveredPlugin] | None = None,
        hooks: LifecycleHooks | None = None,
    ) -> PluginLoader[TestPlugin]:
        """Create a PluginLoader with static discovery."""
        if plugins is None:
            plugins = []
        strategy: StaticRegistration[TestPlugin] = StaticRegistration(plugins)
        registry: PluginRegistry[TestPlugin] = PluginRegistry()
        return PluginLoader[TestPlugin](
            discovery_strategy=strategy,
            registry=registry,
            hooks=hooks,
        )

    async def test_discover(self) -> None:
        """discover() delegates to discoverer."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        result = await loader.discover()
        assert len(result) == 1

    async def test_load_plugin_success(self) -> None:
        """load_plugin() loads and registers the plugin."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        mock_load = AsyncMock(return_value=_DummyPlugin())
        with patch(
            "reflectlog.plugins.discovery.load_plugin",
            new=mock_load,
        ):
            result = await loader.load_plugin("p")

        assert result is True
        assert loader.registry.count() == 1

    async def test_load_plugin_with_instance(self) -> None:
        """load_plugin() uses provided instance directly."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        instance = _DummyPlugin()
        result = await loader.load_plugin("p", instance=instance)
        assert result is True
        assert loader.registry.get("_DummyPlugin") is instance

    async def test_load_plugin_not_discovered(self) -> None:
        """load_plugin() returns False for undiscovered plugin."""
        loader = self._make_loader([])
        await loader.discover()
        result = await loader.load_plugin("missing")
        assert result is False

    async def test_load_plugin_import_error(self) -> None:
        """load_plugin() returns False and sets error on load failure."""
        dp = DiscoveredPlugin(name="p", module_path="bad.mod", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        mock_load = AsyncMock(side_effect=ImportError("module not found"))
        with patch(
            "reflectlog.plugins.discovery.load_plugin",
            new=mock_load,
        ):
            result = await loader.load_plugin("p")

        assert result is False

    async def test_load_plugin_calls_hook(self) -> None:
        """load_plugin() calls on_load hook."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        hook_calls: list[str] = []
        hooks = LifecycleHooks(on_load=lambda name: hook_calls.append(name))
        loader = self._make_loader([dp], hooks=hooks)
        await loader.discover()

        mock_load = AsyncMock(return_value=_DummyPlugin())
        with patch(
            "reflectlog.plugins.discovery.load_plugin",
            new=mock_load,
        ):
            await loader.load_plugin("p")

        assert "p" in hook_calls

    async def test_initialize_plugin_with_lifecycle(self) -> None:
        """initialize_plugin() calls initialize() on lifecycle plugins."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        lc_plugin = _LifecyclePlugin()
        await loader.load_plugin("p", instance=lc_plugin)

        # The plugin is registered under its class name
        plugin_name = loader.registry.list_all()[0]
        result = await loader.initialize_plugin(plugin_name)
        assert result is True
        assert lc_plugin.initialized is True

    async def test_initialize_plugin_without_lifecycle(self) -> None:
        """initialize_plugin() succeeds for non-lifecycle plugins."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        await loader.load_plugin("p", instance=_DummyPlugin())
        plugin_name = loader.registry.list_all()[0]
        result = await loader.initialize_plugin(plugin_name)
        assert result is True

    async def test_initialize_plugin_not_registered(self) -> None:
        """initialize_plugin() returns False for unregistered plugin."""
        loader = self._make_loader()
        result = await loader.initialize_plugin("missing")
        assert result is False

    async def test_initialize_plugin_failure(self) -> None:
        """initialize_plugin() returns False on lifecycle error."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        failing = _FailingLifecyclePlugin()
        await loader.load_plugin("p", instance=failing)
        plugin_name = loader.registry.list_all()[0]
        result = await loader.initialize_plugin(plugin_name)
        assert result is False

    async def test_initialize_plugin_calls_hook(self) -> None:
        """initialize_plugin() calls on_initialize hook."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        hook_calls: list[str] = []
        hooks = LifecycleHooks(on_initialize=lambda name: hook_calls.append(name))
        loader = self._make_loader([dp], hooks=hooks)
        await loader.discover()

        await loader.load_plugin("p", instance=_DummyPlugin())
        plugin_name = loader.registry.list_all()[0]
        await loader.initialize_plugin(plugin_name)
        assert plugin_name in hook_calls

    async def test_activate_plugin_success(self) -> None:
        """activate_plugin() activates a LOADED plugin."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        await loader.load_plugin("p", instance=_DummyPlugin())
        plugin_name = loader.registry.list_all()[0]
        result = await loader.activate_plugin(plugin_name)
        assert result is True

    async def test_activate_plugin_failure(self) -> None:
        """activate_plugin() returns False if registry rejects."""
        loader = self._make_loader()
        result = await loader.activate_plugin("missing")
        assert result is False

    async def test_activate_plugin_calls_hook(self) -> None:
        """activate_plugin() calls on_activate hook."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        hook_calls: list[str] = []
        hooks = LifecycleHooks(on_activate=lambda name: hook_calls.append(name))
        loader = self._make_loader([dp], hooks=hooks)
        await loader.discover()

        await loader.load_plugin("p", instance=_DummyPlugin())
        plugin_name = loader.registry.list_all()[0]
        await loader.activate_plugin(plugin_name)
        assert plugin_name in hook_calls

    async def test_deactivate_plugin_with_lifecycle(self) -> None:
        """deactivate_plugin() calls deactivate() on lifecycle plugins."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        lc_plugin = _LifecyclePlugin()
        await loader.load_plugin("p", instance=lc_plugin)
        plugin_name = loader.registry.list_all()[0]
        await loader.activate_plugin(plugin_name)

        result = await loader.deactivate_plugin(plugin_name)
        assert result is True
        assert lc_plugin.deactivated is True

    async def test_deactivate_plugin_not_registered(self) -> None:
        """deactivate_plugin() returns False for unregistered."""
        loader = self._make_loader()
        result = await loader.deactivate_plugin("missing")
        assert result is False

    async def test_deactivate_plugin_lifecycle_error_still_deactivates(
        self,
    ) -> None:
        """deactivate_plugin() continues even if lifecycle.deactivate() fails."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        failing = _FailingLifecyclePlugin()
        await loader.load_plugin("p", instance=failing)
        plugin_name = loader.registry.list_all()[0]
        await loader.activate_plugin(plugin_name)

        # Should still succeed (deactivate in registry) even if lifecycle fails
        result = await loader.deactivate_plugin(plugin_name)
        assert result is True

    async def test_deactivate_plugin_calls_hook(self) -> None:
        """deactivate_plugin() calls on_deactivate hook."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        hook_calls: list[str] = []
        hooks = LifecycleHooks(on_deactivate=lambda name: hook_calls.append(name))
        loader = self._make_loader([dp], hooks=hooks)
        await loader.discover()

        await loader.load_plugin("p", instance=_DummyPlugin())
        plugin_name = loader.registry.list_all()[0]
        await loader.activate_plugin(plugin_name)
        await loader.deactivate_plugin(plugin_name)
        assert plugin_name in hook_calls

    async def test_unload_plugin_with_lifecycle(self) -> None:
        """unload_plugin() calls cleanup() on lifecycle plugins."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        lc_plugin = _LifecyclePlugin()
        await loader.load_plugin("p", instance=lc_plugin)
        plugin_name = loader.registry.list_all()[0]

        result = await loader.unload_plugin(plugin_name)
        assert result is True
        assert lc_plugin.cleaned_up is True
        assert loader.registry.count() == 0

    async def test_unload_plugin_not_registered(self) -> None:
        """unload_plugin() returns False for unregistered."""
        loader = self._make_loader()
        result = await loader.unload_plugin("missing")
        assert result is False

    async def test_unload_plugin_cleanup_error_still_unloads(self) -> None:
        """unload_plugin() continues even if cleanup() fails."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        failing = _FailingLifecyclePlugin()
        await loader.load_plugin("p", instance=failing)
        plugin_name = loader.registry.list_all()[0]

        result = await loader.unload_plugin(plugin_name)
        assert result is True
        assert loader.registry.count() == 0

    async def test_unload_plugin_calls_hook(self) -> None:
        """unload_plugin() calls on_unload hook."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        hook_calls: list[str] = []
        hooks = LifecycleHooks(on_unload=lambda name: hook_calls.append(name))
        loader = self._make_loader([dp], hooks=hooks)
        await loader.discover()

        await loader.load_plugin("p", instance=_DummyPlugin())
        plugin_name = loader.registry.list_all()[0]
        await loader.unload_plugin(plugin_name)
        assert plugin_name in hook_calls

    async def test_load_all(self) -> None:
        """load_all() loads all discovered plugins."""
        plugins = [
            DiscoveredPlugin(name="a", module_path="m", class_name="A"),
            DiscoveredPlugin(name="b", module_path="m", class_name="B"),
        ]
        loader = self._make_loader(plugins)
        await loader.discover()

        mock_load = AsyncMock(return_value=_DummyPlugin())
        with patch(
            "reflectlog.plugins.discovery.load_plugin",
            new=mock_load,
        ):
            count = await loader.load_all()

        assert count == 2

    async def test_initialize_all(self) -> None:
        """initialize_all() initializes all LOADED plugins."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        await loader.load_plugin("p", instance=_DummyPlugin())
        count = await loader.initialize_all()
        assert count == 1

    async def test_activate_all(self) -> None:
        """activate_all() activates all registered plugins."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        await loader.load_plugin("p", instance=_DummyPlugin())
        count = await loader.activate_all()
        assert count == 1

    async def test_deactivate_all(self) -> None:
        """deactivate_all() deactivates all ACTIVATED plugins."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        await loader.load_plugin("p", instance=_DummyPlugin())
        plugin_name = loader.registry.list_all()[0]
        await loader.activate_plugin(plugin_name)

        count = await loader.deactivate_all()
        assert count == 1

    async def test_unload_all(self) -> None:
        """unload_all() unloads all registered plugins."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        await loader.load_plugin("p", instance=_DummyPlugin())
        count = await loader.unload_all()
        assert count == 1
        assert loader.registry.count() == 0

    async def test_shutdown(self) -> None:
        """shutdown() deactivates and unloads everything."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        loader = self._make_loader([dp])
        await loader.discover()

        lc_plugin = _LifecyclePlugin()
        await loader.load_plugin("p", instance=lc_plugin)
        plugin_name = loader.registry.list_all()[0]
        await loader.activate_plugin(plugin_name)

        await loader.shutdown()
        assert loader.registry.count() == 0
        assert lc_plugin.deactivated is True
        assert lc_plugin.cleaned_up is True

    def test_registry_property(self) -> None:
        """registry property returns the registry."""
        loader = self._make_loader()
        assert isinstance(loader.registry, PluginRegistry)

    def test_discoverer_property(self) -> None:
        """discoverer property returns the discoverer."""
        loader = self._make_loader()
        assert isinstance(loader.discoverer, PluginDiscoverer)


# ---------------------------------------------------------------------------
# __init__.py — Public API
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginsInit:
    """Tests for plugins package structure."""

    def test_key_classes_accessible_from_submodules(self) -> None:
        """Key classes accessible from direct submodule imports."""
        from reflectlog.plugins.discovery import (
            CompositeDiscovery,
            DirectoryScanDiscovery,
            DiscoveredPlugin,
            EntryPointDiscovery,
            PluginDiscoverer,
            PluginDiscoveryStrategy,
            StaticRegistration,
            load_plugin,
        )
        from reflectlog.plugins.loading import (
            IPluginLifecycle,
            LifecycleHooks,
            PluginLoader,
        )
        from reflectlog.plugins.registry import (
            IPluggable,
            PluginCapability,
            PluginMetadata,
            PluginRegistry,
            PluginState,
            ToolRegistry,
        )

        assert all(
            item is not None
            for item in (
                CompositeDiscovery,
                DirectoryScanDiscovery,
                DiscoveredPlugin,
                EntryPointDiscovery,
                PluginDiscoverer,
                PluginDiscoveryStrategy,
                StaticRegistration,
                load_plugin,
                IPluginLifecycle,
                LifecycleHooks,
                PluginLoader,
                IPluggable,
                PluginCapability,
                PluginMetadata,
                PluginRegistry,
                PluginState,
                ToolRegistry,
            )
        )


# ---------------------------------------------------------------------------
# Edge case: deactivate_plugin when registry.deactivate returns False
# (non-lifecycle plugin, not in ACTIVATED state)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPluginLoaderEdgeCases:
    """Edge case tests for PluginLoader."""

    async def test_deactivate_non_activated_plugin(self) -> None:
        """deactivate_plugin() returns False when plugin is LOADED (not ACTIVATED)."""
        dp = DiscoveredPlugin(name="p", module_path="m", class_name="C")
        strategy: StaticRegistration[object] = StaticRegistration([dp])
        registry: PluginRegistry[object] = PluginRegistry()
        loader = PluginLoader(
            discovery_strategy=strategy,
            registry=registry,
        )
        await loader.discover()

        await loader.load_plugin("p", instance=_DummyPlugin())
        plugin_name = loader.registry.list_all()[0]

        # Plugin is LOADED, not ACTIVATED. deactivate should fail.
        result = await loader.deactivate_plugin(plugin_name)
        assert result is False

    async def test_load_all_partial_failure(self) -> None:
        """load_all() counts only successful loads."""
        plugins = [
            DiscoveredPlugin(name="ok", module_path="m", class_name="OK"),
            DiscoveredPlugin(name="bad", module_path="m", class_name="Bad"),
        ]
        strategy: StaticRegistration[object] = StaticRegistration(plugins)
        registry: PluginRegistry[object] = PluginRegistry()
        loader = PluginLoader(
            discovery_strategy=strategy,
            registry=registry,
        )
        await loader.discover()

        call_count = 0

        async def _mock_load(plugin: DiscoveredPlugin) -> object:
            nonlocal call_count
            call_count += 1
            if plugin.name == "bad":
                raise ImportError("cannot load")
            return _DummyPlugin()

        with patch(
            "reflectlog.plugins.discovery.load_plugin",
            side_effect=_mock_load,
        ):
            count = await loader.load_all()

        assert count == 1  # only 'ok' succeeded

    async def test_shutdown_with_no_plugins(self) -> None:
        """shutdown() on empty loader is a no-op."""
        strategy: StaticRegistration[object] = StaticRegistration([])
        registry: PluginRegistry[object] = PluginRegistry()
        loader = PluginLoader(
            discovery_strategy=strategy,
            registry=registry,
        )
        # Should not raise
        await loader.shutdown()
        assert registry.count() == 0

    async def test_loader_default_hooks(self) -> None:
        """PluginLoader uses empty LifecycleHooks when none provided."""
        strategy: StaticRegistration[object] = StaticRegistration([])
        registry: PluginRegistry[object] = PluginRegistry()
        loader = PluginLoader(
            discovery_strategy=strategy,
            registry=registry,
            # No hooks
        )
        # The loader should have default (empty) hooks
        assert loader._hooks is not None
        assert loader._hooks.on_load is None
