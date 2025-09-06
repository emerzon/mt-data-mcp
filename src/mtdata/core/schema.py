"""
Shared JSON schema helpers for CLI/server tool inputs.

Provides reusable $defs such as TimeframeSpec and helpers to apply them
to per-tool parameter schemas.
"""
from typing import Dict, Any, Optional
import inspect
from typing import get_type_hints

from .constants import TIMEFRAME_MAP


def shared_defs() -> Dict[str, Any]:
    """Return shared $defs for input schemas (e.g., TimeframeSpec).

    Note: Additional shared enums (SimplifyMode, etc.) are injected by the server.
    """
    return {
        "TimeframeSpec": {
            "type": "string",
            "enum": sorted(TIMEFRAME_MAP.keys()),
            "description": "MetaTrader timeframe code (e.g., H1, M30, D1)",
        }
    }


def complex_defs() -> Dict[str, Any]:
    """Return complex reusable definitions for nested params.

    These use $ref to shared enums that the server injects (e.g., SimplifyMode).
    """
    return {
        "IndicatorSpec": {
            "type": "object",
            "properties": {
                "name": {"$ref": "#/$defs/IndicatorName"},
                "params": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["name"],
            "additionalProperties": False,
            "description": "Indicator name with optional numeric parameters.",
        },
        "DenoiseSpec": {
            "type": "object",
            "properties": {
                "method": {"$ref": "#/$defs/DenoiseMethod"},
                "params": {"type": "object", "description": "Method-specific parameters", "additionalProperties": True},
                "columns": {"type": "array", "items": {"type": "string"}},
                "when": {"$ref": "#/$defs/WhenSpec"},
                "causality": {"$ref": "#/$defs/CausalitySpec"},
                "keep_original": {"type": "boolean"},
                "suffix": {"type": "string"},
            },
            "required": ["method"],
            "additionalProperties": False,
            "description": "Denoise configuration for selected columns.",
        },
        "SimplifySpec": {
            "type": "object",
            "properties": {
                "mode": {"$ref": "#/$defs/SimplifyMode"},
                "method": {"$ref": "#/$defs/SimplifyMethod"},
                "points": {"type": "integer"},
                "ratio": {"type": "number"},
                "epsilon": {"type": "number"},
                "max_error": {"type": "number"},
                "segments": {"type": "integer"},
                "bucket_seconds": {"type": "integer"},
                "schema": {"oneOf": [
                    {"$ref": "#/$defs/EncodeSchema"},
                    {"$ref": "#/$defs/SymbolicSchema"}
                ]},
                "bits": {"type": "integer"},
                "as_chars": {"type": "boolean"},
                "alphabet": {"type": "string"},
                "scale": {"type": "number"},
                "zero_char": {"type": "string"},
                "algo": {"type": "string", "enum": ["zigzag","zz"]},
                "threshold_pct": {"type": "number"},
                "value_col": {"type": "string"},
                "paa": {"type": "integer"},
                "znorm": {"type": "boolean"},
            },
            "additionalProperties": False,
            "description": "Simplification/segmentation/encoding parameters.",
        },
        "VolatilityParams": {
            "type": "object",
            "properties": {
                "halflife": {"type": ["number", "null"]},
                "lambda_": {"type": ["number", "null"], "description": "EWMA smoothing factor"},
                "lookback": {"type": "integer"},
                "window": {"type": "integer"},
                "fit_bars": {"type": "integer"},
                "mean": {"type": "string", "enum": ["Zero", "Constant"]},
                "dist": {"type": "string", "enum": ["normal"]},
            },
            "additionalProperties": False,
            "description": "Volatility method parameters.",
        },
    }


def _ensure_defs(schema: Dict[str, Any]) -> Dict[str, Any]:
    if "$defs" not in schema or not isinstance(schema.get("$defs"), dict):
        schema["$defs"] = {}
    # Merge shared defs without overwriting existing keys
    defs = schema["$defs"]
    for k, v in shared_defs().items():
        defs.setdefault(k, v)
    return schema


def _parameters_obj(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Get or create the OpenAI/MCP-style parameters object inside a schema."""
    if not isinstance(schema.get("parameters"), dict):
        schema["parameters"] = {"type": "object", "properties": {}}
    params = schema["parameters"]
    if not isinstance(params.get("properties"), dict):
        params["properties"] = {}
    return params


def apply_timeframe_ref(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Replace simple timeframe property shapes with a $ref to TimeframeSpec.

    Looks for common parameter names and applies a $ref if present.
    """
    _ensure_defs(schema)
    params = _parameters_obj(schema)
    props = params["properties"]
    for key in ("timeframe", "target_timeframe", "source_timeframe"):
        if key in props and isinstance(props.get(key), dict):
            props[key] = {"$ref": "#/$defs/TimeframeSpec"}
    return schema


def build_minimal_schema(func_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimal parameters schema from a discovered function description.

    - Only includes parameter names and required flags.
    - Applies TimeframeSpec $ref to known timeframe param names.
    """
    schema: Dict[str, Any] = {"parameters": {"type": "object", "properties": {}, "required": []}}
    props = schema["parameters"]["properties"]
    req = schema["parameters"]["required"]
    for p in func_info.get("params", []):
        name = p.get("name")
        if not name:
            continue
        # Default to string for minimal typing; CLI does its own casting
        props[name] = {"type": "string"}
        if p.get("required"):
            req.append(name)
    _ensure_defs(schema)
    apply_timeframe_ref(schema)
    return schema


def enrich_schema_with_shared_defs(schema: Dict[str, Any], func_info: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure schema has $defs and timeframe refs. If empty, build minimal one."""
    if not isinstance(schema, dict) or not schema:
        schema = build_minimal_schema(func_info)
        return schema
    _ensure_defs(schema)
    apply_timeframe_ref(schema)
    return schema


def get_function_info(func: Any) -> Dict[str, Any]:
    """Extract minimal parameter info from a function for schema building."""
    # Introspect original function if wrapped
    try:
        target = inspect.unwrap(func)
    except Exception:
        target = func
    sig = inspect.signature(target)
    try:
        type_hints = get_type_hints(target)
    except Exception:
        type_hints = {}

    params = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        params.append({
            "name": name,
            "required": param.default == inspect._empty,  # type: ignore[attr-defined]
            "default": None if param.default == inspect._empty else param.default,  # type: ignore[attr-defined]
            "type": type_hints.get(name)
        })

    return {
        "name": getattr(target, "__name__", ""),
        "doc": inspect.getdoc(target) or "",
        "params": params,
    }
