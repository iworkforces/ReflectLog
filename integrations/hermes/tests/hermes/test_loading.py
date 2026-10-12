# pyright: basic
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from plugins.memory import load_memory_provider

PLUGIN = Path(__file__).resolve().parents[2]


def test_real_loader(tmp_path, monkeypatch):
    # Given a real user plugin under a temporary Hermes home.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    shutil.copytree(
        PLUGIN,
        tmp_path / "plugins" / "reflectlog",
        ignore=shutil.ignore_patterns(".venv", "tests", "*cache*", "uv.lock"),
    )
    # When the real pinned loader discovers it.
    provider = load_memory_provider("reflectlog")
    # Then the registered provider is found without initialization/network.
    assert provider is not None
    assert provider.name == "reflectlog"
    provider.shutdown()


def test_manifest_version():
    # Given independently parsed project and plugin metadata.
    project = tomllib.loads((PLUGIN / "pyproject.toml").read_text())
    # When the manifest is loaded.
    manifest = dict(
        line.split(": ", 1)
        for line in (PLUGIN / "plugin.yaml").read_text().splitlines()
        if line and not line.startswith(" ") and ": " in line
    )
    # Then the published versions agree.
    assert manifest["version"] == project["project"]["version"]


def test_package_import_is_lazy(tmp_path):
    # Given a clean process without prior dependency imports.
    script = f"import importlib.util,sys; s=importlib.util.spec_from_file_location('probe', {str(PLUGIN / '__init__.py')!r}, submodule_search_locations=[{str(PLUGIN)!r}]); m=importlib.util.module_from_spec(s); sys.modules['probe']=m; s.loader.exec_module(m); assert not any(n in sys.modules for n in ('agent','httpx'))"
    # When only the package initializer executes.
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    # Then neither Hermes nor the HTTP client is imported.
    assert result.returncode == 0, result.stderr


def test_real_cli_validate(tmp_path):
    # Given the installed Hermes CLI without a token or server.
    env = dict(os.environ, HERMES_HOME=str(tmp_path))
    env.pop("REFLECTLOG_MCP_TOKEN", None)
    # When its actual validator is invoked.
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("hermes")),
            "plugins",
            "validate",
            str(PLUGIN),
            "--json",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    # Then every catalog admission check passes.
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["ok"] is True
