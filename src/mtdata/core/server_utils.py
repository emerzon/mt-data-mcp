"""Server utilities and helper functions."""

from typing import Any, Dict, Optional


def get_mcp_registry(mcp: Any) -> Optional[Dict[str, Any]]:
    """Return the MCP tool registry if available."""
    for attr in ("tools", "_tools", "registry", "tool_registry", "_tool_registry"):
        reg = getattr(mcp, attr, None)
        if reg and hasattr(reg, "items"):
            return reg
    return None
