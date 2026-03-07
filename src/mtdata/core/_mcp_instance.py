"""Leaf module that owns the FastMCP singleton."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from .constants import SERVICE_NAME
from ._mcp_tools import install_tool_registry


def _apply_fastmcp_env_overrides() -> None:
    """Bridge legacy MCP_* env vars to FastMCP's FASTMCP_* settings."""
    mapping = {
        "MCP_HOST": "FASTMCP_HOST",
        "MCP_PORT": "FASTMCP_PORT",
        "MCP_LOG_LEVEL": "FASTMCP_LOG_LEVEL",
        "MCP_MOUNT_PATH": "FASTMCP_MOUNT_PATH",
        "MCP_SSE_PATH": "FASTMCP_SSE_PATH",
        "MCP_MESSAGE_PATH": "FASTMCP_MESSAGE_PATH",
    }
    for legacy, target in mapping.items():
        val = os.getenv(legacy)
        if val and not os.getenv(target):
            os.environ[target] = val

    if not os.getenv("FASTMCP_HOST"):
        os.environ["FASTMCP_HOST"] = "0.0.0.0"


_apply_fastmcp_env_overrides()
mcp = FastMCP(SERVICE_NAME)
install_tool_registry(mcp)
