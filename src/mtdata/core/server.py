"""Main entry point for the MCP server."""

import logging
import os
import atexit
from typing import Literal, Optional, cast

from ..bootstrap.tools import bootstrap_tools
from .config import load_environment

load_environment()

from ._mcp_instance import _apply_fastmcp_env_overrides, mcp
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
from ..utils.mt5 import mt5_connection, _auto_connect_wrapper, _ensure_symbol_ready

# Lightweight helpers used by tool modules
from ..utils.utils import _normalize_ohlcv_arg
_REEXPORTED_SYMBOLS = (
    mt5_config,
    TIMEFRAME_MAP,
    TIMEFRAME_SECONDS,
    _auto_connect_wrapper,
    _ensure_symbol_ready,
    _normalize_ohlcv_arg,
)

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
    transport = (os.getenv("MCP_TRANSPORT") or default).strip().lower()
    if transport not in ("stdio", "sse", "streamable-http"):
        transport = default
    mount_path = os.getenv("MCP_MOUNT_PATH")
    return transport, mount_path


def main():
    """Main entry point for the MCP server"""
    bootstrap_tools()
    settings = getattr(mcp, 'settings', None)
    if settings is not None:
        if not getattr(settings, "host", None):
            settings.host = os.getenv("FASTMCP_HOST", "127.0.0.1")
        log_level = getattr(logging, str(getattr(settings, 'log_level', 'INFO')).upper(), logging.INFO)
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)
    transport, mount_path = _resolve_transport()
    logger.info(f"Starting {SERVICE_NAME} server... transport={transport}")

    if transport == "sse" and settings is not None:
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
        transport_literal = cast(Literal['stdio', 'sse', 'streamable-http'], transport)
        run_fn(transport=transport_literal, mount_path=mount_path if transport == "sse" else None)


def main_stdio():
    """Entry point for stdio mode (forced)"""
    os.environ["MCP_TRANSPORT"] = "stdio"
    main()


def main_sse():
    """Entry point for SSE mode (forced)"""
    os.environ["MCP_TRANSPORT"] = "sse"
    main()


def main_streamable_http():
    """Entry point for streamable HTTP mode (forced)."""
    os.environ["MCP_TRANSPORT"] = "streamable-http"
    main()


if __name__ == "__main__":
    main()
