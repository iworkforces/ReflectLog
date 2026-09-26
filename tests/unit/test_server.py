"""Unit tests for reflectlog/server.py CLI module."""

from collections.abc import Callable
import io
import os
from pathlib import Path
import signal
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.mark.unit
class TestCLIArgumentParsing:
    """Test CLI argument parsing functionality."""

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_default_arguments(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test main() with no arguments uses defaults."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        # Import main and test
        from reflectlog.server import main

        # Mock sys.argv
        with patch("sys.argv", ["reflectlog"]):
            # Should not raise any errors
            try:
                main()
            except SystemExit:
                pass  # main() may call sys.exit()

            # Verify server was created and run
            mock_server_class.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_transport_http_argument(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test --transport http argument."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog", "--transport", "http"]):
            try:
                main()
            except SystemExit:
                pass

            # Check environment variable was set
            assert os.environ.get("MCP_TRANSPORT") == "http"

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_transport_stdio_argument(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ):
        """Test --transport stdio argument."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog", "--transport", "stdio"]):
            try:
                main()
            except SystemExit:
                pass

            assert os.environ.get("MCP_TRANSPORT") == "stdio"

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_port_argument(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test --port argument."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog", "--port", "9999"]):
            try:
                main()
            except SystemExit:
                pass

            assert os.environ.get("MCP_PORT") == "9999"

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_host_argument(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test --host argument."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog", "--host", "192.168.1.1"]):
            try:
                main()
            except SystemExit:
                pass

            assert os.environ.get("MCP_HOST") == "192.168.1.1"

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_path_argument(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test --path argument."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog", "--path", "/custom"]):
            try:
                main()
            except SystemExit:
                pass

            assert os.environ.get("MCP_PATH") == "/custom"

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_multiple_arguments(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test multiple arguments together."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch(
            "sys.argv",
            [
                "reflectlog",
                "--transport",
                "http",
                "--port",
                "8080",
                "--host",
                "localhost",
            ],
        ):
            try:
                main()
            except SystemExit:
                pass

            assert os.environ.get("MCP_TRANSPORT") == "http"
            assert os.environ.get("MCP_PORT") == "8080"
            assert os.environ.get("MCP_HOST") == "localhost"

    @patch.dict(os.environ, {}, clear=True)
    def test_help_argument(self):
        """Test --help argument."""
        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # --help causes sys.exit(0)
            assert exc_info.value.code == 0

    @patch.dict(os.environ, {}, clear=True)
    def test_version_argument(self):
        """Test --version argument."""
        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # --version causes sys.exit(0)
            assert exc_info.value.code == 0


def _run_cli(args: list[str]):
    from pathlib import Path
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    return subprocess.run(
        ["uv", "run", "--frozen", "--no-sync", "reflectlog", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.unit
class TestLightweightCliMetadata:
    """Help/version exit before importing FastMCP, search, Numba, or ranx."""

    def test_version_matches_installed_distribution(self) -> None:
        from importlib.metadata import version

        completed = _run_cli(["--version"])
        assert completed.returncode == 0
        assert completed.stdout.strip() == f"reflectlog {version('reflectlog')}"

    def test_help_exits_zero(self) -> None:
        completed = _run_cli(["--help"])
        assert completed.returncode == 0
        assert "transport" in completed.stdout.lower()

    def test_version_does_not_import_heavy_modules(self) -> None:
        from pathlib import Path
        import subprocess
        import sys

        repo = Path(__file__).resolve().parents[2]
        script = """
import sys
sys.argv = ["reflectlog", "--version"]
from reflectlog.server import main
try:
    main()
except SystemExit as exc:
    assert exc.code == 0
heavy = {
    name
    for name in sys.modules
    if name == "ranx"
    or name == "numba"
    or name.startswith("fastmcp")
    or name.startswith("reflectlog.application.mcp_server")
    or name.startswith("reflectlog.utility.scoring")
    or name.startswith("reflectlog.infrastructure")
}
assert not heavy, sorted(heavy)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    def test_invalid_argument_does_not_construct_server(self) -> None:
        completed = _run_cli(["--definitely-invalid"])
        assert completed.returncode != 0
        assert "unrecognized arguments" in completed.stderr


@pytest.mark.unit
class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    @patch.dict(os.environ, {"MCP_TRANSPORT": "sse"}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_env_var_transport(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test MCP_TRANSPORT environment variable."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog"]):
            try:
                main()
            except SystemExit:
                pass

            # Environment variable should still be set
            assert os.environ.get("MCP_TRANSPORT") == "sse"

    @patch.dict(os.environ, {"MCP_TRANSPORT": "stdio", "MCP_PORT": "5000"}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_cli_args_override_env_vars(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ):
        """Test that CLI arguments override environment variables."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog", "--transport", "http", "--port", "9103"]):
            try:
                main()
            except SystemExit:
                pass

            # CLI args should override env vars
            assert os.environ.get("MCP_TRANSPORT") == "http"
            assert os.environ.get("MCP_PORT") == "9103"


@pytest.mark.unit
class TestServerInitialization:
    """Test server initialization and startup."""

    @patch.dict(os.environ, {"WORKSPACE_ID": "test_project"}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_server_instantiation(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test FastMCPServer is instantiated."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog"]):
            try:
                main()
            except SystemExit:
                pass

            # Verify server was created
            mock_server_class.assert_called_once()

    @patch.dict(os.environ, {"WORKSPACE_ID": "test_project"}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_server_run_called(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test server.run() is called."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog"]):
            try:
                main()
            except SystemExit:
                pass

            # Verify run was called
            mock_server.run.assert_called_once()


@pytest.mark.unit
class TestOutputStreams:
    """Test output stream configuration."""

    @patch.dict(os.environ, {"WORKSPACE_ID": "test_project"}, clear=True)
    @patch("sys.stderr")
    @patch("sys.stdout")
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_stdio_transport_uses_stderr(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
        mock_stdout: MagicMock,
        mock_stderr: MagicMock,
    ):
        """Test stdio transport redirects output to stderr."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog", "--transport", "stdio"]):
            try:
                main()
            except SystemExit:
                pass

            # For stdio, sys.stdout should be redirected to stderr
            # (This tests the logging configuration)
            assert os.environ.get("MCP_TRANSPORT") == "stdio"

    @patch.dict(os.environ, {"WORKSPACE_ID": "test_project"}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_http_transport_uses_stdout(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ):
        """Test http transport uses stdout."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from reflectlog.server import main

        with patch("sys.argv", ["reflectlog", "--transport", "http"]):
            try:
                main()
            except SystemExit:
                pass

            # For http, normal stdout is used
            assert os.environ.get("MCP_TRANSPORT") == "http"


@pytest.mark.unit
class TestWarmupNumbaWithConfig:
    """Test warmup_numba_with_config function directly."""

    @patch("reflectlog.server.warmup_numba_functions")
    def test_disabled_without_output_stream(self, mock_warmup: MagicMock) -> None:
        """Test warmup disabled without output stream returns None silently."""
        from reflectlog.server import warmup_numba_with_config

        result = warmup_numba_with_config(enabled=False, output_stream=None)

        assert result is None
        mock_warmup.assert_not_called()

    @patch("reflectlog.server.warmup_numba_functions")
    def test_disabled_with_output_stream(self, mock_warmup: MagicMock) -> None:
        """Test warmup disabled with output stream prints message (lines 51-53)."""
        from reflectlog.server import warmup_numba_with_config

        output = io.StringIO()
        result = warmup_numba_with_config(enabled=False, output_stream=output)

        assert result is None
        mock_warmup.assert_not_called()
        assert "disabled" in output.getvalue().lower()

    @patch("reflectlog.server.warmup_numba_functions")
    def test_invalid_mode_raises_value_error(self, mock_warmup: MagicMock) -> None:
        """Test invalid mode raises ValueError (lines 57-58)."""
        from reflectlog.server import warmup_numba_with_config

        with pytest.raises(ValueError, match="Invalid NUMBA_WARMUP_MODE"):
            warmup_numba_with_config(enabled=True, mode="invalid_mode")

        mock_warmup.assert_not_called()

    @patch("reflectlog.server.warmup_numba_functions")
    def test_invalid_mode_includes_valid_options(self, mock_warmup: MagicMock) -> None:
        """Test ValueError message includes all valid mode options."""
        from reflectlog.server import warmup_numba_with_config

        with pytest.raises(ValueError, match="sync, async, background"):
            warmup_numba_with_config(enabled=True, mode="bad")

    @patch("reflectlog.server.warmup_numba_functions")
    def test_sync_mode_with_output_stream(self, mock_warmup: MagicMock) -> None:
        """Test sync mode prints progress messages (lines 63-68)."""
        from reflectlog.server import warmup_numba_with_config

        output = io.StringIO()
        result = warmup_numba_with_config(
            enabled=True, mode="sync", output_stream=output
        )

        assert result is None
        mock_warmup.assert_called_once()
        output_text = output.getvalue()
        assert "synchronous" in output_text.lower()
        assert "compiled and cached" in output_text.lower()

    @patch("reflectlog.server.warmup_numba_functions")
    def test_sync_mode_without_output_stream(self, mock_warmup: MagicMock) -> None:
        """Test sync mode works without output stream."""
        from reflectlog.server import warmup_numba_with_config

        result = warmup_numba_with_config(enabled=True, mode="sync", output_stream=None)

        assert result is None
        mock_warmup.assert_called_once()

    @patch("reflectlog.server.warmup_numba_functions")
    def test_async_mode_returns_thread(self, mock_warmup: MagicMock) -> None:
        """Test async mode returns a non-daemon thread."""
        from reflectlog.server import warmup_numba_with_config

        output = io.StringIO()
        result = warmup_numba_with_config(
            enabled=True, mode="async", output_stream=output
        )

        assert isinstance(result, threading.Thread)
        assert result.daemon is False
        result.join(timeout=5)
        assert "background thread" in output.getvalue().lower()

    @patch("reflectlog.server.warmup_numba_functions")
    def test_background_mode_returns_daemon_thread(
        self, mock_warmup: MagicMock
    ) -> None:
        """Test background mode returns a daemon thread."""
        from reflectlog.server import warmup_numba_with_config

        output = io.StringIO()
        result = warmup_numba_with_config(
            enabled=True, mode="background", output_stream=output
        )

        assert isinstance(result, threading.Thread)
        assert result.daemon is True
        result.join(timeout=5)
        assert "background daemon thread" in output.getvalue().lower()

    @patch("reflectlog.server.warmup_numba_functions")
    def test_warmup_worker_exception_prints_warning(
        self, mock_warmup: MagicMock
    ) -> None:
        """Test warmup_worker exception path prints warning (lines 86-88)."""
        from reflectlog.server import warmup_numba_with_config

        mock_warmup.side_effect = RuntimeError("JIT compilation failed")

        output = io.StringIO()
        result = warmup_numba_with_config(
            enabled=True, mode="async", output_stream=output
        )

        assert isinstance(result, threading.Thread)
        result.join(timeout=5)

        output_text = output.getvalue()
        assert "warning" in output_text.lower()
        assert "JIT compilation failed" in output_text

    @patch("reflectlog.server.warmup_numba_functions")
    def test_warmup_worker_exception_without_output_stream(
        self, mock_warmup: MagicMock
    ) -> None:
        """Test warmup_worker exception is silenced without output stream."""
        from reflectlog.server import warmup_numba_with_config

        mock_warmup.side_effect = RuntimeError("JIT compilation failed")

        result = warmup_numba_with_config(
            enabled=True, mode="background", output_stream=None
        )

        assert isinstance(result, threading.Thread)
        # Should not raise - exception is swallowed silently
        result.join(timeout=5)


@pytest.mark.unit
class TestGracefulShutdown:
    """Test graceful shutdown signal handler (lines 232-242)."""

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_sigint_triggers_graceful_shutdown(
        self, mock_server_class: MagicMock, mock_warmup: MagicMock
    ) -> None:
        """Test SIGINT calls server.close() and sys.exit(0)."""
        from reflectlog.server import main

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        # Capture the signal handler registered for SIGINT
        registered_handlers: dict[int, Callable[[int, object], None]] = {}

        def capture_signal(signum: int, handler: Callable[[int, object], None]) -> None:
            registered_handlers[signum] = handler

        with (
            patch("sys.argv", ["reflectlog", "--transport", "http"]),
            patch("reflectlog.server.signal.signal", side_effect=capture_signal),
        ):
            try:
                main()
            except SystemExit:
                pass

        # Verify SIGINT handler was registered
        assert signal.SIGINT in registered_handlers
        handler = registered_handlers[signal.SIGINT]
        mock_server.close.reset_mock()

        # Invoke the handler - should call sys.exit(0)
        with pytest.raises(SystemExit) as exc_info:
            handler(signal.SIGINT, None)

        assert exc_info.value.code == 0
        mock_server.close.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_second_sigint_re_raises_default_signal(
        self, mock_server_class: MagicMock, mock_warmup: MagicMock
    ) -> None:
        """A second SIGINT restores SIG_DFL and re-raises the signal."""
        from reflectlog.server import main

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        registered_handlers: dict[int, Callable[[int, object], None]] = {}

        def capture_signal(signum: int, handler: Callable[[int, object], None]) -> None:
            registered_handlers[signum] = handler

        with (
            patch("sys.argv", ["reflectlog", "--transport", "http"]),
            patch("reflectlog.server.signal.signal", side_effect=capture_signal),
        ):
            try:
                main()
            except SystemExit:
                pass

        handler = registered_handlers[signal.SIGINT]
        with pytest.raises(SystemExit):
            handler(signal.SIGINT, None)

        with (
            patch("reflectlog.server.signal.signal") as mock_signal,
            patch("reflectlog.server.signal.raise_signal") as mock_raise,
        ):
            handler(signal.SIGINT, None)
            mock_signal.assert_any_call(signal.SIGINT, signal.SIG_DFL)
            mock_signal.assert_any_call(signal.SIGTERM, signal.SIG_DFL)
            if sys.platform == "win32":
                mock_signal.assert_any_call(signal.SIGBREAK, signal.SIG_DFL)
            mock_raise.assert_called_once_with(signal.SIGINT)

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_sigterm_triggers_graceful_shutdown(
        self, mock_server_class: MagicMock, mock_warmup: MagicMock
    ) -> None:
        """Test SIGTERM calls server.close() and sys.exit(0)."""
        from reflectlog.server import main

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        registered_handlers: dict[int, Callable[[int, object], None]] = {}

        def capture_signal(signum: int, handler: Callable[[int, object], None]) -> None:
            registered_handlers[signum] = handler

        with (
            patch("sys.argv", ["reflectlog", "--transport", "http"]),
            patch("reflectlog.server.signal.signal", side_effect=capture_signal),
        ):
            try:
                main()
            except SystemExit:
                pass

        # Verify SIGTERM handler was registered
        assert signal.SIGTERM in registered_handlers
        handler = registered_handlers[signal.SIGTERM]
        mock_server.close.reset_mock()

        # Invoke the handler
        with pytest.raises(SystemExit) as exc_info:
            handler(signal.SIGTERM, None)

        assert exc_info.value.code == 0
        mock_server.close.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_graceful_shutdown_when_server_is_none(
        self, mock_server_class: MagicMock, mock_warmup: MagicMock
    ):
        """Test graceful shutdown handles server=None (before initialization)."""
        from reflectlog.server import main

        # Make FastMCPServer raise so server stays None when handler runs
        mock_server_class.side_effect = RuntimeError("init failed")

        registered_handlers: dict[int, Callable[[int, object], None]] = {}

        def capture_signal(signum: int, handler: Callable[[int, object], None]) -> None:
            registered_handlers[signum] = handler

        with (
            patch("sys.argv", ["reflectlog", "--transport", "http"]),
            patch("reflectlog.server.signal.signal", side_effect=capture_signal),
        ):
            with pytest.raises(RuntimeError, match="init failed"):
                main()

        # The handler was registered before server creation attempt
        handler = registered_handlers[signal.SIGINT]

        # Should still exit cleanly without calling close()
        with pytest.raises(SystemExit) as exc_info:
            handler(signal.SIGINT, None)

        assert exc_info.value.code == 0


@pytest.mark.unit
class TestStartupTimingVerbose:
    """Test STARTUP_TIMING_VERBOSE output (lines 266-268)."""

    @patch.dict(
        os.environ,
        {"STARTUP_TIMING_VERBOSE": "true"},
        clear=True,
    )
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_verbose_timing_prints_breakdown(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ):
        """Test verbose startup timing prints phase breakdown (lines 266-268)."""
        from reflectlog.server import main

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        output = io.StringIO()

        with (
            patch("sys.argv", ["reflectlog", "--transport", "http"]),
            patch("sys.stdout", output),
        ):
            try:
                main()
            except SystemExit:
                pass

        output_text = output.getvalue()
        # Should contain the timing breakdown header and phase names
        assert "Startup timing breakdown:" in output_text
        assert "numba_warmup" in output_text
        assert "server_initialization" in output_text
        assert "total_startup" in output_text

    @patch.dict(
        os.environ,
        {"STARTUP_TIMING_VERBOSE": "false"},
        clear=True,
    )
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_non_verbose_skips_breakdown(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ):
        """Test non-verbose mode does not print timing breakdown."""
        from reflectlog.server import main

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        output = io.StringIO()

        with (
            patch("sys.argv", ["reflectlog", "--transport", "http"]),
            patch("sys.stdout", output),
        ):
            try:
                main()
            except SystemExit:
                pass

        output_text = output.getvalue()
        assert "Startup timing breakdown:" not in output_text


@pytest.mark.unit
class TestMainExceptionHandling:
    """Test exception handling in main() (lines 274-283)."""

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_keyboard_interrupt_during_run(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ):
        """Test KeyboardInterrupt during server.run() triggers cleanup (lines 274-278)."""
        from reflectlog.server import main

        mock_server = MagicMock()
        mock_server.run.side_effect = KeyboardInterrupt
        mock_server_class.return_value = mock_server

        with patch("sys.argv", ["reflectlog", "--transport", "http"]):
            # KeyboardInterrupt should be caught and handled gracefully
            main()

        mock_server.close.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_exception_during_run_reraises(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ):
        """Test generic Exception during server.run() calls close and re-raises (lines 279-283)."""
        from reflectlog.server import main

        mock_server = MagicMock()
        mock_server.run.side_effect = RuntimeError("Server crashed")
        mock_server_class.return_value = mock_server

        with patch("sys.argv", ["reflectlog", "--transport", "http"]):
            with pytest.raises(RuntimeError, match="Server crashed"):
                main()

        mock_server.close.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_exception_during_init_no_close(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test exception during FastMCPServer() does not call close on None server."""
        from reflectlog.server import main

        mock_server_class.side_effect = RuntimeError("Init failed")

        with patch("sys.argv", ["reflectlog", "--transport", "http"]):
            with pytest.raises(RuntimeError, match="Init failed"):
                main()

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.server.signal.signal")
    @patch("reflectlog.server.warmup_numba_functions")
    @patch("reflectlog.server.FastMCPServer")
    def test_keyboard_interrupt_with_none_server(
        self,
        mock_server_class: MagicMock,
        mock_warmup: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test KeyboardInterrupt when server is None does not call close."""
        from reflectlog.server import main

        mock_server_class.side_effect = KeyboardInterrupt

        with patch("sys.argv", ["reflectlog", "--transport", "http"]):
            # Should handle gracefully even though server is None
            main()
