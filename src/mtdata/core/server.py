"""Main entry point for the MCP server."""

import logging
import atexit
from typing import Optional, Set

from mcp.server.fastmcp import FastMCP

from .config import mt5_config
from .constants import SERVICE_NAME, TIMEFRAME_MAP, TIMEFRAME_SECONDS  # re-export for CLI/tests
from ..utils.mt5 import mt5_connection, _auto_connect_wrapper, _ensure_symbol_ready

# Create MCP instance early to avoid circular imports when tools import `mcp`.
mcp = FastMCP(SERVICE_NAME)

# Lightweight helpers used by tool modules
def _coerce_scalar(s: str):
    """Try to coerce a scalar string to int or float; otherwise return original string."""
    try:
        if s is None:
            return s
        st = str(s).strip()
        if st == "":
            return st
        if st.isdigit() or (st.startswith('-') and st[1:].isdigit()):
            return int(st)
        v = float(st)
        return v
    except Exception:
        return s


def _normalize_ohlcv_arg(ohlcv: Optional[str]) -> Optional[Set[str]]:
    """Normalize user-provided OHLCV selection into a set of letters.

    Accepts forms like: 'close', 'price', 'ohlc', 'ohlcv', 'all', 'cl', 'OHLCV',
    or names 'open,high,low,close,volume'. Returns None when not specified.
    """
    if ohlcv is None:
        return None
    text = str(ohlcv).strip()
    if text == "":
        return None
    t = text.lower()
    if t in ("all", "ohlcv"):
        return {"O", "H", "L", "C", "V"}
    if t in ("ohlc",):
        return {"O", "H", "L", "C"}
    if t in ("price", "close"):
        return {"C"}
    # Compact letters like 'cl', 'oh', etc.
    if all(ch in "ohlcv" for ch in t):
        return {ch.upper() for ch in t}
    # Comma separated names
    parts = [p.strip().lower() for p in t.replace(";", ",").split(",") if p.strip() != ""]
    if not parts:
        return None
    mapping = {
        "o": "O", "open": "O",
        "h": "H", "high": "H",
        "l": "L", "low": "L",
        "c": "C", "close": "C", "price": "C",
        "v": "V", "vol": "V", "volume": "V", "tick_volume": "V",
    }
    out: Set[str] = set()
    for p in parts:
        key = mapping.get(p)
        if key:
            out.add(key)
    return out or None


# Import all tools so they are registered with the MCP server
# Augment FastMCP with a discoverable registry for the CLI
try:
    _ORIG_TOOL_DECORATOR = mcp.tool  # type: ignore[attr-defined]
except Exception:
    _ORIG_TOOL_DECORATOR = None
_TOOL_REGISTRY = {}

def _recording_tool_decorator(*dargs, **dkwargs):  # type: ignore[override]
    if _ORIG_TOOL_DECORATOR is None:
        # Fallback: no-op decorator
        def _noop(func):
            _TOOL_REGISTRY[getattr(func, '__name__', 'tool')] = func
            return func
        return _noop
    dec = _ORIG_TOOL_DECORATOR(*dargs, **dkwargs)
    def _wrap(func):
        res = dec(func)
        name = getattr(func, '__name__', None)
        if name:
            # Store the original callable for CLI discovery
            _TOOL_REGISTRY[str(name)] = func
        return res
    return _wrap

# Install wrapper and expose common registry fields for discovery
try:
    setattr(mcp, 'tool', _recording_tool_decorator)
    setattr(mcp, 'tools', _TOOL_REGISTRY)
    setattr(mcp, 'registry', _TOOL_REGISTRY)
    setattr(mcp, '_tools', _TOOL_REGISTRY)
    setattr(mcp, '_tool_registry', _TOOL_REGISTRY)
except Exception:
    pass

from .data import *
from .denoise import *
from .forecast import *
from .indicators import *
from .market_depth import *
from .patterns import *
from .pivot import *
from .simplify import *
from .symbols import *
from .regime import *
from .labels import *
from .report import *
from .trading import *


@atexit.register
def _disconnect_mt5():
    mt5_connection.disconnect()


def main():
    """Main entry point for the MCP server"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {SERVICE_NAME} server...")
    mcp.run()


if __name__ == "__main__":
    main()
