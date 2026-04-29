"""Leaf module that owns the FastMCP singleton."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ._mcp_tools import install_tool_registry
from ..shared.constants import SERVICE_NAME

mcp = FastMCP(SERVICE_NAME)
install_tool_registry(mcp)
