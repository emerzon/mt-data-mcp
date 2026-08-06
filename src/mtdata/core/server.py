"""Main entry point for the MCP server."""

import atexit
import logging
import os
from typing import Literal, Optional, cast

from ..bootstrap.runtime import (
    McpRuntimeSettings,
    apply_mcp_runtime_settings,
    load_mcp_runtime_settings,
)
from ..bootstrap.settings import load_environment
from ..bootstrap.tools import bootstrap_tools
from ..shared.constants import SERVICE_NAME
from ..utils.mt5 import mt5_connection
from ._mcp_instance import mcp


def _run_prefixed_sse(runtime: McpRuntimeSettings) -> None:
    """Mount the SSE app so advertised and routed message URLs agree."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount

    prefix = "/" + runtime.mount_path.strip("/")
    inner = mcp.sse_app(mount_path=prefix)
    app = Starlette(routes=[Mount(prefix, app=inner)])
    uvicorn.run(
        app,
        host=runtime.host,
        port=runtime.port,
        log_level=runtime.log_level.lower(),
    )


@atexit.register
def _disconnect_mt5():
    mt5_connection.disconnect()


def _warm_windows_joblib_cpu_cache() -> None:
    """Resolve joblib CPU topology before MCP tools enter worker threads."""
    if os.name != "nt":
        return

    try:
        import joblib

        joblib.cpu_count(only_physical_cores=True)
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not warm joblib CPU topology cache.",
            exc_info=True,
        )


def main(
    *,
    transport: Optional[Literal["stdio", "sse", "streamable-http"]] = None,
    runtime_settings: Optional[McpRuntimeSettings] = None,
):
    """Main entry point for the MCP server"""
    load_environment()
    runtime = runtime_settings or load_mcp_runtime_settings(transport_override=transport)
    # sklearn asks joblib for the physical CPU count before every first
    # KMeans fit. On Windows, resolving that count from an asyncio worker can
    # leave joblib waiting indefinitely on its PowerShell topology probe.
    _warm_windows_joblib_cpu_cache()
    bootstrap_tools()
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
    elif transport_name == "streamable-http" and settings is not None:
        logger.info(
            "Streamable HTTP listening at http://%s:%s%s",
            getattr(settings, 'host', '127.0.0.1'),
            getattr(settings, 'port', 8000),
            getattr(settings, 'streamable_http_path', runtime.mount_path),
        )

    run_fn = getattr(mcp, 'run', None)
    if run_fn is not None:
        if transport_name == "sse" and mount_path:
            _run_prefixed_sse(runtime)
            return
        transport_literal = cast(Literal['stdio', 'sse', 'streamable-http'], transport_name)
        run_fn(transport=transport_literal, mount_path=mount_path if transport_name == "sse" else None)


def main_stdio():
    """Entry point for stdio mode (forced)"""
    main(transport="stdio")


def main_sse():
    """Entry point for SSE mode (forced)"""
    main(transport="sse")


def main_streamable_http():
    """Entry point for streamable HTTP mode (forced)."""
    main(transport="streamable-http")


if __name__ == "__main__":
    main()
