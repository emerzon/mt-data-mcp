"""Main entry point for the MCP server."""

import logging
import atexit
from typing import Optional, Set

from mcp.server.fastmcp import FastMCP
from functools import wraps as _wraps

from .config import mt5_config
from .constants import SERVICE_NAME, TIMEFRAME_MAP, TIMEFRAME_SECONDS  # re-export for CLI/tests
from .schema_attach import attach_schemas_to_tools
from .schema import get_shared_enum_lists
from ..utils.mt5 import mt5_connection, _auto_connect_wrapper, _ensure_symbol_ready

# Create MCP instance early to avoid circular imports when tools import `mcp`.
mcp = FastMCP(SERVICE_NAME)

# Lightweight helpers used by tool modules
from ..utils.utils import _coerce_scalar, _normalize_ohlcv_arg


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
    kwargs = dict(dkwargs)
    # Default to unstructured output so tools emit raw text content unless overridden
    structured_in_args = len(dargs) >= 5
    if not structured_in_args and 'structured_output' not in kwargs:
        kwargs['structured_output'] = False
    dec = _ORIG_TOOL_DECORATOR(*dargs, **kwargs)
    def _wrap(func):
        # Wrap the callable to ensure plain-text, minimal output for API calls
        try:
            from ..utils.minimal_output import format_result_minimal as _fmt_min, to_methods_availability_csv as _fmt_methods
        except Exception:
            _fmt_min = lambda x: str(x) if x is not None else ""  # fallback
            _fmt_methods = None  # type: ignore

        @_wraps(func)
        def _wrapped(*a, **kw):
            # Uniform input normalization for common structured args
            try:
                if 'denoise' in kw:
                    from ..utils.denoise import normalize_denoise_spec as _norm_dn  # type: ignore
                    kw['denoise'] = _norm_dn(kw.get('denoise'))
            except Exception:
                pass
            out = func(*a, **kw)
            try:
                # Special-case: compact method availability where applicable
                fname = getattr(func, '__name__', '')
                if fname in ('forecast_list_methods', 'denoise_list_methods') and isinstance(out, dict):
                    methods = out.get('methods') or []
                    if _fmt_methods:
                        s = _fmt_methods(methods)
                        if s:
                            return s
                return _fmt_min(out)
            except Exception:
                return str(out) if out is not None else ""

        res = dec(_wrapped)
        name = getattr(func, '__name__', None)
        if name:
            # Store the original callable for CLI discovery
            _TOOL_REGISTRY[str(name)] = _wrapped
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
from .causal import *
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

try:
    attach_schemas_to_tools(mcp, get_shared_enum_lists())
except Exception:
    pass



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
