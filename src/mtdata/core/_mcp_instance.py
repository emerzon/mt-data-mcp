"""Leaf module that owns the FastMCP singleton."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .constants import SERVICE_NAME
from ._mcp_tools import install_tool_registry

mcp = FastMCP(SERVICE_NAME)
install_tool_registry(mcp)
