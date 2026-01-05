"""Main entry point for the MCP server."""

import logging
import os
import atexit
import inspect
import math
import types
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Union, cast, get_args, get_origin

from mcp.server.fastmcp import FastMCP
from functools import wraps as _wraps

from .config import mt5_config
from .constants import SERVICE_NAME, TIMEFRAME_MAP, TIMEFRAME_SECONDS  # re-export for CLI/tests
from .schema_attach import attach_schemas_to_tools
from .schema import get_shared_enum_lists
from ..utils.mt5 import mt5_connection, _auto_connect_wrapper, _ensure_symbol_ready


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

    # Default to listen on all interfaces if no host is specified
    if not os.getenv("FASTMCP_HOST"):
        os.environ["FASTMCP_HOST"] = "0.0.0.0"

# Create MCP instance early to avoid circular imports when tools import `mcp`.
_apply_fastmcp_env_overrides()
mcp = FastMCP(SERVICE_NAME)

# Lightweight helpers used by tool modules
from ..utils.utils import _coerce_scalar, _normalize_ohlcv_arg


# Import all tools so they are registered with the MCP server
# Augment FastMCP with a discoverable registry for the CLI
try:
    _ORIG_TOOL_DECORATOR = mcp.tool  # type: ignore[attr-defined]
except Exception:
    _ORIG_TOOL_DECORATOR = None
_TOOL_REGISTRY: Dict[str, Any] = {}
_TOOL_OBJECT_REGISTRY: Dict[str, Any] = {}


def _unwrap_optional_annotation(annotation: Any) -> tuple[Any, bool]:
    if isinstance(annotation, str):
        cleaned = annotation.strip()
        scalar_map: dict[str, type] = {
            "bool": bool,
            "builtins.bool": bool,
            "int": int,
            "builtins.int": int,
            "float": float,
            "builtins.float": float,
        }

        # PEP 604 unions expressed as strings under `from __future__ import annotations`
        # e.g. "int | None".
        if "|" in cleaned:
            parts = [p.strip() for p in cleaned.split("|") if p.strip()]
            if any(p in ("None", "NoneType") for p in parts):
                non_none = [p for p in parts if p not in ("None", "NoneType")]
                if len(non_none) == 1:
                    mapped = scalar_map.get(non_none[0])
                    if mapped is not None:
                        return mapped, True

        # typing.Optional / typing.Union expressed as strings.
        for prefix in ("Optional[", "typing.Optional["):
            if cleaned.startswith(prefix) and cleaned.endswith("]"):
                inner = cleaned[len(prefix) : -1].strip()
                mapped = scalar_map.get(inner)
                if mapped is not None:
                    return mapped, True

        for prefix in ("Union[", "typing.Union["):
            if cleaned.startswith(prefix) and cleaned.endswith("]"):
                inner = cleaned[len(prefix) : -1]
                parts = [p.strip() for p in inner.split(",") if p.strip()]
                if any(p in ("None", "NoneType") for p in parts):
                    non_none = [p for p in parts if p not in ("None", "NoneType")]
                    if len(non_none) == 1:
                        mapped = scalar_map.get(non_none[0])
                        if mapped is not None:
                            return mapped, True

        mapped = scalar_map.get(cleaned)
        if mapped is not None:
            return mapped, False
        return annotation, False

    origin = get_origin(annotation)
    if origin in (Union, getattr(types, "UnionType", None)):
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            other = args[0] if args[1] is type(None) else args[1]
            return other, True
    return annotation, False


def _coerce_bool(value: Any, *, allow_none: bool, name: str) -> Any:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Invalid value for '{name}': expected boolean, got {value!r}")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("none", "null"):
            if allow_none:
                return None
            raise ValueError(f"Invalid value for '{name}': expected boolean, got {value!r}")
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
    raise ValueError(f"Invalid value for '{name}': expected boolean, got {value!r}")


def _coerce_int(value: Any, *, allow_none: bool, name: str) -> Any:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Invalid value for '{name}': expected integer, got {value!r}")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Invalid value for '{name}': expected integer, got {value!r}")
        if value.is_integer():
            return int(value)
        raise ValueError(f"Invalid value for '{name}': expected integer, got {value!r}")
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in ("none", "null"):
            if allow_none:
                return None
            raise ValueError(f"Invalid value for '{name}': expected integer, got {value!r}")
        coerced = _coerce_scalar(s)
        if isinstance(coerced, int) and not isinstance(coerced, bool):
            return coerced
        if isinstance(coerced, float) and math.isfinite(coerced) and coerced.is_integer():
            return int(coerced)
    raise ValueError(f"Invalid value for '{name}': expected integer, got {value!r}")


def _coerce_float(value: Any, *, allow_none: bool, name: str) -> Any:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")
        return out
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in ("none", "null"):
            if allow_none:
                return None
            raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")
        coerced = _coerce_scalar(s)
        if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
            out = float(coerced)
            if not math.isfinite(out):
                raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")
            return out
    raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")


def _coerce_kwargs_for_callable(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce common scalar string inputs (from MCP clients) based on annotations."""
    try:
        sig = inspect.signature(func)
    except Exception:
        return kwargs
    for param_name, param in sig.parameters.items():
        if param_name not in kwargs:
            continue
        ann = param.annotation
        if ann is inspect._empty:
            continue
        base_ann, allow_none = _unwrap_optional_annotation(ann)
        if base_ann is bool:
            kwargs[param_name] = _coerce_bool(kwargs.get(param_name), allow_none=allow_none, name=param_name)
        elif base_ann is int:
            kwargs[param_name] = _coerce_int(kwargs.get(param_name), allow_none=allow_none, name=param_name)
        elif base_ann is float:
            kwargs[param_name] = _coerce_float(kwargs.get(param_name), allow_none=allow_none, name=param_name)
    return kwargs


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

    def _sanitize_annotations(func):
        """Ensure annotations are classes to satisfy FastMCP's issubclass checks."""
        import inspect

        cleaned = {}
        ann = getattr(func, "__annotations__", {}) or {}
        for name, param in inspect.signature(func).parameters.items():
            value = ann.get(name, param.annotation)
            cleaned[name] = value if isinstance(value, type) else object
        if "return" in ann:
            cleaned["return"] = ann["return"] if isinstance(ann["return"], type) else object
        return cleaned

    def _wrap(func):
        # Wrap the callable to ensure plain-text, minimal output for API calls
        try:
            from ..utils.minimal_output import format_result_minimal as _fmt_min, to_methods_availability_toon as _fmt_methods
        except Exception:
            _fmt_min = lambda x: str(x) if x is not None else ""  # fallback
            _fmt_methods = None

        @_wraps(func)
        def _wrapped(*a, **kw):
            # Check for raw output flag (used by CLI to bypass formatting)
            raw_output = kw.pop('__cli_raw', False)

            try:
                _coerce_kwargs_for_callable(func, kw)

                # Uniform input normalization for common structured args
                try:
                    if 'denoise' in kw:
                        from ..utils.denoise import normalize_denoise_spec as _norm_dn  # type: ignore
                        kw['denoise'] = _norm_dn(kw.get('denoise'))
                except Exception:
                    pass

                out = func(*a, **kw)
            except Exception as exc:
                try:
                    logging.getLogger(__name__).exception("Tool '%s' failed", getattr(func, "__name__", "tool"))
                except Exception:
                    pass
                out = {"error": str(exc)}
            
            if raw_output:
                return out
                
            try:
                # Special-case: compact method availability where applicable
                fname = getattr(func, '__name__', '')
                if fname in ('forecast_list_methods', 'denoise_list_methods') and isinstance(out, dict):
                    methods_list = out.get('methods') or []
                    if _fmt_methods and isinstance(methods_list, list):
                        s = _fmt_methods(cast(List[Dict[str, Any]], methods_list))
                        if s:
                            return s
                return _fmt_min(out)
            except Exception:
                return str(out) if out is not None else ""

        try:
            import inspect

            cleaned = _sanitize_annotations(func)
            _wrapped.__annotations__ = cleaned
            # Override __signature__ so FastMCP sees sanitized annotations
            params = []
            for name, param in inspect.signature(func).parameters.items():
                params.append(
                    param.replace(annotation=cleaned.get(name))
                )
            return_ann = cleaned.get("return", inspect._empty)
            setattr(_wrapped, '__signature__', inspect.Signature(parameters=params, return_annotation=return_ann))
        except Exception:
            pass

        res = dec(_wrapped)
        name = getattr(func, '__name__', None)
        if name:
            # Store the original callable for CLI discovery
            _TOOL_REGISTRY[str(name)] = _wrapped
            try:
                _TOOL_OBJECT_REGISTRY[str(name)] = res
            except Exception:
                pass
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


def get_tool_registry() -> Dict[str, Any]:
    """Return the registered tool objects (preferred) or callables for CLI discovery."""
    if _TOOL_OBJECT_REGISTRY:
        return dict(_TOOL_OBJECT_REGISTRY)
    return dict(_TOOL_REGISTRY)


def get_tool_functions() -> Dict[str, Any]:
    """Return the registered tool callables (wrapped) for direct invocation."""
    return dict(_TOOL_REGISTRY)

from .data import *
from ..utils.denoise import denoise_list_methods
from .forecast import *
from .causal import *
from .indicators import *
from .market_depth import *
from .patterns import *
from .pivot import *
from .symbols import *
from .regime import *
from .labels import *
from .report import *
from .trading import *
from .temporal import *
from .finviz import *

try:
    attach_schemas_to_tools(mcp, get_shared_enum_lists())
except Exception:
    pass



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
    # Force listen on all interfaces
    settings = getattr(mcp, 'settings', None)
    if settings is not None:
        settings.host = "0.0.0.0"
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
            getattr(settings, 'host', '0.0.0.0'),
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


if __name__ == "__main__":
    main()
