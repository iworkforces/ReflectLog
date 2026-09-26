import argparse
import asyncio
import os
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING

from reflectlog.core.enums import TransportMode
from reflectlog.version import __version__

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import SupportsWrite

    from reflectlog.application.mcp_server import FastMCPServer as FastMCPServerCls

# Configure numba environment before the lazy scoring import in warmup.
for key, value in [
    ("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba_cache")),
    ("NUMBA_THREADING_LAYER", "workqueue"),
    ("NUMBA_DEBUG", "0"),
    ("NUMBA_FASTMATH", "1"),
]:
    _ = os.environ.setdefault(key, value)

# Populated on first real startup so --help/--version stay import-light.
# Tests continue to patch these module attributes.
FastMCPServer: type[FastMCPServerCls] | None = None
warmup_numba_functions: Callable[[], object] | None = None


def _server_cls() -> type[FastMCPServerCls]:
    global FastMCPServer
    loaded = FastMCPServer
    if loaded is None:
        from reflectlog.application.mcp_server import FastMCPServer as LoadedServer

        FastMCPServer = LoadedServer
        loaded = LoadedServer
    return loaded


def _call_warmup_numba_functions() -> object:
    global warmup_numba_functions
    loaded = warmup_numba_functions
    if loaded is None:
        from reflectlog.utility.scoring import (
            warmup_numba_functions as warmup,
        )

        warmup_numba_functions = warmup
        loaded = warmup
    return loaded()


def warmup_numba_with_config(
    enabled: bool = True,
    mode: str = "sync",
    output_stream: SupportsWrite[str] | None = None,
) -> threading.Thread | None:
    """Warm up numba JIT functions with configurable execution mode.

    Args:
        enabled: Whether to perform JIT warmup at all.
        mode: Execution mode - "sync" (default), "async" (background thread),
            or "background" (daemon thread).
        output_stream: Stream to print progress (stderr for stdio, stdout otherwise).

    Returns:
        Thread object if mode is "async" or "background", None otherwise.

    Raises:
        ValueError: If mode is not one of "sync", "async", or "background".
    """
    if not enabled:
        if output_stream:
            print("Numba JIT warmup disabled (NUMBA_WARMUP=false)", file=output_stream)
        return None

    valid_modes = ("sync", "async", "background")
    if mode not in valid_modes:
        modes_str = ", ".join(valid_modes)
        raise ValueError(
            f"Invalid NUMBA_WARMUP_MODE: '{mode}'. Valid options: {modes_str}"
        )

    if mode == "sync":
        if output_stream:
            print("Warming up numba JIT functions (synchronous)...", file=output_stream)
        _ = _call_warmup_numba_functions()
        if output_stream:
            print("Numba functions compiled and cached", file=output_stream)
        return None
    else:
        # async or background mode
        is_daemon = mode == "background"
        if output_stream:
            mode_desc = "background daemon thread" if is_daemon else "background thread"
            print(
                f"Warming up numba JIT functions ({mode_desc})...", file=output_stream
            )

        def warmup_worker() -> None:
            try:
                _ = _call_warmup_numba_functions()
                if output_stream:
                    print(
                        "Numba functions compiled and cached (background complete)",
                        file=output_stream,
                    )
            except Exception as e:
                if output_stream:
                    print(f"Numba warmup warning: {e}", file=output_stream)

        thread = threading.Thread(target=warmup_worker, daemon=is_daemon)
        thread.start()
        return thread


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        prog="reflectlog",
        description="MCP Server For Claude Code Project Memories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default stdio transport (for MCP clients)
  reflectlog

  # Run as HTTP server
  reflectlog --transport http --port 9103

  # Run with SSE transport
  reflectlog --transport sse --port 8080 --host 0.0.0.0

Workspace selection is required as workspace_id on every MCP tool call.
WORKSPACE_ID is ignored by the server.

Environment Variables:
  MCP_TRANSPORT   Override transport mode (stdio, http, sse, streamable-http)
  MCP_PORT        Override server port
  MCP_HOST        Override server host
  MCP_PATH        Override server path
        """,
    )

    _ = parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit",
    )

    _ = parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport protocol (default: stdio for tool usage, or from settings)",
    )

    _ = parser.add_argument(
        "--port",
        type=int,
        help="Server port for non-stdio transports (default: 9103)",
    )

    _ = parser.add_argument(
        "--host",
        type=str,
        help="Server host for non-stdio transports (default: 127.0.0.1)",
    )

    _ = parser.add_argument(
        "--path",
        type=str,
        help="Server path for non-stdio transports (default: /mcp)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the MCP server.

    Supports both CLI argument parsing and environment variable configuration.
    When run as a tool (via `reflectlog` command), defaults to stdio transport.

    Graceful shutdown is handled via signal handlers for SIGINT (Ctrl+C) and SIGTERM,
    ensuring all data from TantivyEngine and USearchEngine is persisted before exit.
    """
    args = parse_args()
    transport_mode = _apply_cli_env_vars(args)
    output_stream = sys.stderr if transport_mode == TransportMode.STDIO else sys.stdout

    print(
        "Starting ReflectLog - Project-based AI Agent Memories...",
        file=output_stream,
    )
    print(f"Version: {__version__}", file=output_stream)
    print(f"Transport: {transport_mode}", file=output_stream)

    startup_start_time = time.time()
    server: FastMCPServerCls | None = None
    startup_phases: dict[str, float] = {}
    try:
        started = _start_server(output_stream, startup_start_time, startup_phases)
        server = started
        extra_phases = _run_numba_warmup(output_stream)
        existing = started.startup_metrics
        merged: dict[str, float] = dict(startup_phases)
        if existing is not None:
            merged.update(existing)
        merged.update(extra_phases)
        started.set_startup_metrics(merged)
        _print_startup_timing(output_stream, merged)
        started.run()
        started.close()
    except KeyboardInterrupt, asyncio.CancelledError:
        if server is not None:
            server.close()
    except Exception:
        if server is not None:
            server.close()
        raise


def _apply_cli_env_vars(args: argparse.Namespace) -> str:
    """Set environment variables from CLI args and return transport mode."""
    if args.transport:
        os.environ["MCP_TRANSPORT"] = args.transport
    elif "MCP_TRANSPORT" not in os.environ:
        os.environ["MCP_TRANSPORT"] = TransportMode.STDIO

    if args.port:
        os.environ["MCP_PORT"] = str(args.port)
    if args.host:
        os.environ["MCP_HOST"] = args.host
    if args.path:
        os.environ["MCP_PATH"] = args.path

    return os.environ.get("MCP_TRANSPORT", TransportMode.STDIO)


def _print_startup_timing(
    output_stream: SupportsWrite[str],
    startup_phases: dict[str, float],
) -> None:
    """Print per-phase startup timings when STARTUP_TIMING_VERBOSE is set."""
    if os.environ.get("STARTUP_TIMING_VERBOSE", "false").lower() != "true":
        return
    print("Startup timing breakdown:", file=output_stream)
    for phase, duration in startup_phases.items():
        print(f"  {phase}: {duration * 1000:.1f}ms", file=output_stream)


def _run_numba_warmup(
    output_stream: SupportsWrite[str],
) -> dict[str, float]:
    """Run numba JIT warmup phase and return startup phases dict."""
    numba_warmup_enabled = os.environ.get("NUMBA_WARMUP", "true").lower() == "true"
    default_warmup_mode = "background"
    numba_warmup_mode = os.environ.get("NUMBA_WARMUP_MODE", default_warmup_mode).lower()

    startup_phases: dict[str, float] = {}
    numba_start = time.time()
    _ = warmup_numba_with_config(
        enabled=numba_warmup_enabled,
        mode=numba_warmup_mode,
        output_stream=output_stream,
    )
    startup_phases["numba_warmup"] = time.time() - numba_start
    return startup_phases


def _start_server(
    output_stream: SupportsWrite[str],
    startup_start_time: float,
    startup_phases: dict[str, float],
) -> FastMCPServerCls:
    """Initialize server with signal handlers and startup metrics.

    Registers SIGINT/SIGTERM handlers for graceful shutdown.
    """
    server: FastMCPServerCls | None = None

    shutting_down = {"value": False}

    def graceful_shutdown(signum: int, frame: object) -> None:
        """Signal handler for graceful shutdown."""
        if shutting_down["value"]:
            _ = signal.signal(signal.SIGINT, signal.SIG_DFL)
            _ = signal.signal(signal.SIGTERM, signal.SIG_DFL)
            if sys.platform == "win32":
                _ = signal.signal(signal.SIGBREAK, signal.SIG_DFL)
            signal.raise_signal(signum)
            return
        shutting_down["value"] = True
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(
            f"\nReceived {signal_name}, initiating graceful shutdown...",
            file=output_stream,
        )
        persist_ok = True
        if server is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                for task in asyncio.all_tasks(loop):
                    _ = task.cancel()
                return
            try:
                server.close()
            except Exception as exc:
                persist_ok = False
                print(f"Shutdown persist failed: {exc}", file=output_stream)
        if persist_ok:
            print("Shutdown complete.", file=output_stream)
            sys.exit(0)
        print("Shutdown incomplete; persist failed.", file=output_stream)
        sys.exit(1)

    _ = signal.signal(signal.SIGINT, graceful_shutdown)
    _ = signal.signal(signal.SIGTERM, graceful_shutdown)
    if sys.platform == "win32":
        _ = signal.signal(signal.SIGBREAK, graceful_shutdown)

    try:
        phase_start = time.time()
        started = _server_cls()()
        server = started
        startup_phases["server_initialization"] = time.time() - phase_start

        total_startup_time = time.time() - startup_start_time
        startup_phases["total_startup"] = total_startup_time

        print(
            f"Server startup completed in {total_startup_time * 1000:.1f}ms",
            file=output_stream,
        )

        started.set_startup_metrics(startup_phases)
        return started
    except KeyboardInterrupt:
        print("\nServer shutdown requested...", file=output_stream)
        if server is not None:
            server.close()
        raise
    except Exception as e:
        print(
            f"Error during server operation: {type(e).__name__}",
            file=output_stream,
        )
        if server is not None:
            server.close()
        raise


if __name__ == "__main__":
    main()
