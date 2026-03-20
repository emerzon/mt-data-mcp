"""Main entry point for the MCP server."""

import logging
import atexit
from typing import Literal, Optional, cast

from ..bootstrap.runtime import McpRuntimeSettings, apply_mcp_runtime_settings, load_mcp_runtime_settings
from ..bootstrap.tools import bootstrap_tools
from .config import load_environment

from ._mcp_instance import mcp
from . import _mcp_tools
from ._mcp_tools import (
    _TOOL_OBJECT_REGISTRY,
    _TOOL_REGISTRY,
    _coerce_bool,
    _coerce_float,
    _coerce_int,
    _coerce_kwargs_for_callable,
    _get_runtime_annotations,
    _get_runtime_signature,
    _unwrap_optional_annotation,
)
from .config import mt5_config
from .constants import SERVICE_NAME, TIMEFRAME_MAP, TIMEFRAME_SECONDS  # re-export for CLI/tests
from .server_entrypoints import (
    main_sse as _main_sse_entrypoint,
    main_stdio as _main_stdio_entrypoint,
    main_streamable_http as _main_streamable_http_entrypoint,
)
from ..utils.mt5 import mt5_connection, _ensure_symbol_ready

# Lightweight helpers used by tool modules
from ..utils.utils import _normalize_ohlcv_arg

_ORIG_TOOL_DECORATOR = _mcp_tools._ORIG_TOOL_DECORATOR


def _recording_tool_decorator(*dargs, **dkwargs):  # type: ignore[override]
    global _ORIG_TOOL_DECORATOR
    _mcp_tools._ORIG_TOOL_DECORATOR = _ORIG_TOOL_DECORATOR
    return _mcp_tools._recording_tool_decorator(*dargs, **dkwargs)


def get_tool_registry() -> dict[str, object]:
    bootstrap_tools()
    return _mcp_tools.get_tool_registry()


def get_tool_functions() -> dict[str, object]:
    bootstrap_tools()
    return _mcp_tools.get_tool_functions()



@atexit.register
def _disconnect_mt5():
    mt5_connection.disconnect()


def _resolve_transport(default: str = "sse") -> tuple[str, Optional[str]]:
    """Return the transport to use and optional mount path for SSE."""
    runtime = load_mcp_runtime_settings(default_transport=cast(Literal["stdio", "sse", "streamable-http"], default))
    mount_path = runtime.mount_path if runtime.transport == "sse" and runtime.mount_path not in ("", "/") else None
    return runtime.transport, mount_path


def main(
    *,
    transport: Optional[Literal["stdio", "sse", "streamable-http"]] = None,
    runtime_settings: Optional[McpRuntimeSettings] = None,
):
    """Main entry point for the MCP server"""
    load_environment()
    bootstrap_tools()
    runtime = runtime_settings or load_mcp_runtime_settings(transport_override=transport)
    apply_mcp_runtime_settings(mcp, runtime)
    settings = getattr(mcp, 'settings', None)
    if settings is not None:
        log_level = getattr(logging, str(getattr(settings, 'log_level', 'INFO')).upper(), logging.INFO)
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)
    transport_name = runtime.transport
    mount_path = runtime.mount_path if runtime.transport == "sse" and runtime.mount_path not in ("", "/") else None
    logger.info(f"Starting {SERVICE_NAME} server... transport={transport_name}")

    if transport_name == "sse" and settings is not None:
        base_path = str(getattr(settings, 'mount_path', '') or '').rstrip("/") or "/"
        logger.info(
            "SSE listening at http://%s:%s%s (event path %s, message path %s)",
            getattr(settings, 'host', '127.0.0.1'),
            getattr(settings, 'port', 8000),
            base_path,
            getattr(settings, 'sse_path', '/sse'),
            getattr(settings, 'message_path', '/message'),
        )

    run_fn = getattr(mcp, 'run', None)
    if run_fn is not None:
        transport_literal = cast(Literal['stdio', 'sse', 'streamable-http'], transport_name)
        run_fn(transport=transport_literal, mount_path=mount_path if transport_name == "sse" else None)


def main_stdio():
    """Entry point for stdio mode (forced)"""
    _main_stdio_entrypoint(main)


def main_sse():
    """Entry point for SSE mode (forced)"""
    _main_sse_entrypoint(main)


def main_streamable_http():
    """Entry point for streamable HTTP mode (forced)."""
    _main_streamable_http_entrypoint(main)


if __name__ == "__main__":
    main()
