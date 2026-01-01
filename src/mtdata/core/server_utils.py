"""Server utilities and helper functions."""
from typing import Any, Dict, Optional, Set

from ..utils.utils import _coerce_scalar, _normalize_ohlcv_arg


def coerce_scalar(s: str) -> str:
    """Try to coerce a scalar string to int or float; otherwise return original string."""
    return _coerce_scalar(s)


def normalize_ohlcv_arg(ohlcv: Optional[str]) -> Optional[Set[str]]:
    """Normalize user-provided OHLCV selection into a set of letters."""
    return _normalize_ohlcv_arg(ohlcv)


def get_mcp_registry(mcp: Any) -> Optional[Dict[str, Any]]:
    """Return the MCP tool registry if available."""
    for attr in ("tools", "_tools", "registry", "tool_registry", "_tool_registry"):
        reg = getattr(mcp, attr, None)
        if reg and hasattr(reg, "items"):
            return reg
    return None
