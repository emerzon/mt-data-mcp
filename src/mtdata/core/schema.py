"""
Shared JSON schema helpers for CLI/server tool inputs.

Provides reusable $defs such as TimeframeSpec and helpers to apply them
to per-tool parameter schemas.
"""
from typing import Dict, Any, Optional, List, Tuple, Literal, Union
from typing_extensions import TypedDict
import inspect
from typing import get_type_hints, get_origin, get_args

from .constants import TIMEFRAME_MAP


PARAM_HINTS = {
    "direction": "Trade direction context (long/short). Determines which side is TP vs SL.",
    "symbol": "Trading symbol (e.g. EURUSD)",
    "timeframe": "MT5 timeframe (e.g. H1/M30/D1)",
    "limit": "Max rows/bars to return",
    "start": "Start time (YYYY-MM-DD or natural language)",
    "end": "End time (exclusive)",
    "ohlcv": "OHLCV column selector",
    "indicators": "Indicator specs list",
    "denoise": "Denoise spec JSON or preset",
    "simplify": "Simplify spec for downsampling",
    "method": "Method/algorithm name",
    "horizon": "Forecast horizon (bars)",
    "steps": "Backtest anchors or steps",
    "spacing": "Spacing between anchors",
    "alpha": "Confidence complement (e.g. 0.1)",
    "params": "Extra method parameters",
    "as_of": "Override reference time",
    "ci_alpha": "CI alpha (e.g. 0.1)",
    "features": "Feature set spec",
    "dimred_method": "Dimensionality reduction method",
    "dimred_params": "Dimensionality reduction params",
    "target_spec": "Target field spec",
    "quantity": "Quantity to analyze",
    "target": "Target series (price/return)",
    "points": "Target point count",
    "ratio": "Target ratio",
    "epsilon": "Tolerance value",
    "max_error": "Max approximation error",
    "segments": "Segment count",
    "bucket_seconds": "Bucket size (seconds)",
    "schema": "Encoding schema",
    "bits": "Bits per symbol",
    "paa": "PAA segments",
    "znorm": "Apply z-normalization",
    "threshold_pct": "Percent threshold",
    "value_col": "Column name",
    "lookback": "Lookback bars",
    "spacing_pct": "Spacing as percent",
    "volume": "Order volume (lots)",
    "type": "Order type",
    "price": "Price level",
    "stop_loss": "Stop-loss level",
    "take_profit": "Take-profit level",
    "id": "Entity identifier",
}


_TIMEFRAME_CHOICES = tuple(sorted(TIMEFRAME_MAP.keys()))
TimeframeLiteral = Literal[_TIMEFRAME_CHOICES]  # type: ignore

# Build a Literal for single OHLCV letters; the parameter will be a list of these
OhlcvCharLiteral = Literal['O', 'H', 'L', 'C', 'V']  # type: ignore

# ---- Technical Indicators (dynamic discovery and application) ----
try:
    from .indicators_docs import list_ta_indicators as _list_ta_indicators_docs
    _CATEGORY_CHOICES = sorted({it.get('category') for it in _list_ta_indicators_docs() if it.get('category')})
except Exception:
    _CATEGORY_CHOICES = []

if _CATEGORY_CHOICES:
    # Create a Literal type alias dynamically
    CategoryLiteral = Literal[tuple(_CATEGORY_CHOICES)]  # type: ignore
else:
    CategoryLiteral = str  # fallback

# Build indicator name Literal so details endpoint has enum name choices
try:
    from .indicators_docs import list_ta_indicators as _list_ta_indicators_docs
    _INDICATOR_NAME_CHOICES = sorted({it.get('name') for it in _list_ta_indicators_docs() if it.get('name')})
except Exception:
    _INDICATOR_NAME_CHOICES = []

if _INDICATOR_NAME_CHOICES:
    IndicatorNameLiteral = Literal[tuple(_INDICATOR_NAME_CHOICES)]  # type: ignore
else:
    IndicatorNameLiteral = str  # fallback

class IndicatorSpec(TypedDict, total=False):
    """Structured TI spec: name with optional numeric params.

    Note: 'name' accepts any string to allow compact forms like "rsi(20)".
    """
    name: str
    params: List[float]

# ---- Denoising (spec + application) ----
# Allowed denoising methods for first phase (no extra dependencies)
_DENOISE_METHODS = (
    "none",        # no-op
    "ema",         # exponential moving average
    "sma",         # simple moving average
    "median",      # rolling median
    "lowpass_fft", # zero-phase FFT low-pass
    "wavelet",     # wavelet shrinkage (PyWavelets optional)
    "emd",         # empirical mode decomposition (PyEMD optional)
    "eemd",        # ensemble EMD (PyEMD optional)
    "ceemdan",     # complementary EEMD with adaptive noise (PyEMD optional)
)

try:
    DenoiseMethodLiteral = Literal[_DENOISE_METHODS]  # type: ignore
except Exception:
    DenoiseMethodLiteral = str  # fallback for typing

class DenoiseSpec(TypedDict, total=False):
    method: DenoiseMethodLiteral  # type: ignore
    params: Dict[str, Any]
    columns: List[str]
    when: Literal['pre_ti', 'post_ti']  # type: ignore
    causality: Literal['causal', 'zero_phase']  # type: ignore
    keep_original: bool
    suffix: str

# ---- Simplify (schema for MCP) ----
_SIMPLIFY_MODES = (
    'select',        # pick representative existing rows
    'approximate',   # aggregate between selected rows
    'resample',      # time-bucket aggregation
    'encode',        # compact encodings (envelope, delta)
    'segment',       # swing points (e.g., ZigZag)
    'symbolic',      # SAX symbolic representation
)
_SIMPLIFY_METHODS = (
    'lttb', 'rdp', 'pla', 'apca'
)
try:
    SimplifyModeLiteral = Literal[_SIMPLIFY_MODES]  # type: ignore
except Exception:
    SimplifyModeLiteral = str
try:
    SimplifyMethodLiteral = Literal[_SIMPLIFY_METHODS]  # type: ignore
except Exception:
    SimplifyMethodLiteral = str
try:
    EncodeSchemaLiteral = Literal['envelope','delta']  # type: ignore
    SymbolicSchemaLiteral = Literal['sax']  # type: ignore
except Exception:
    EncodeSchemaLiteral = str
    SymbolicSchemaLiteral = str

class SimplifySpec(TypedDict, total=False):
    # Common
    mode: SimplifyModeLiteral  # type: ignore
    method: SimplifyMethodLiteral  # type: ignore
    points: int
    ratio: float
    # RDP/PLA/APCA specifics
    epsilon: float
    max_error: float
    segments: int
    # Resample
    bucket_seconds: int
    # Encode specifics
    schema: EncodeSchemaLiteral  # 'envelope' | 'delta' (or 'sax' when mode='symbolic')
    bits: int
    as_chars: bool
    alphabet: str
    scale: float
    zero_char: str
    # Segment specifics
    algo: Literal['zigzag','zz']  # type: ignore
    threshold_pct: float
    value_col: str
    # Symbolic specifics
    paa: int
    znorm: bool

# Volatility params (concise)
class VolatilityParams(TypedDict, total=False):
    # EWMA
    halflife: Optional[float]
    lambda_: Optional[float]  # use 'lambda_' to avoid reserved word in schema
    lookback: int
    # Parkinson/GK/RS
    window: int
    # GARCH
    fit_bars: int
    mean: Literal['Zero','Constant']  # type: ignore
    dist: Literal['normal']  # type: ignore

# ---- Pivot Point methods (enums) ----
_PIVOT_METHODS = (
    "classic",
    "fibonacci",
    "camarilla",
    "woodie",
    "demark",
)

try:
    PivotMethodLiteral = Literal[_PIVOT_METHODS]  # type: ignore
except Exception:
    PivotMethodLiteral = str  # fallback for typing

# ---- Fast Forecast methods (enums) ----
_FORECAST_METHODS = (
    "naive",
    "seasonal_naive",
    "drift",
    "theta",
    "fourier_ols",
    "ses",
    "holt",
    "holt_winters_add",
    "holt_winters_mul",
    "arima",
    "sarima",
    "mc_gbm",
    "hmm_mc",
    "nhits",
    "nbeatsx",
    "tft",
    "patchtst",
    "sf_autoarima",
    "sf_theta",
    "sf_autoets",
    "sf_seasonalnaive",
    "mlf_rf",
    "mlf_lightgbm",
    "chronos_bolt",
    "timesfm",
    "lag_llama",
    "moirai",
    "gt_deepar",
    "gt_sfeedforward",
    "gt_prophet",
    "gt_tft",
    "gt_wavenet",
    "gt_deepnpts",
    "gt_mqf2",
    "gt_npts",
    "ensemble",
)

try:
    ForecastMethodLiteral = Literal[_FORECAST_METHODS]  # type: ignore
except Exception:
    ForecastMethodLiteral = str  # fallback for typing


def shared_defs() -> Dict[str, Any]:
    """Return shared $defs for input schemas (e.g., TimeframeSpec).

    Note: Additional shared enums (SimplifyMode, etc.) are injected by the server.
    """
    return {
        "TimeframeSpec": {
            "type": "string",
            "enum": sorted(TIMEFRAME_MAP.keys()),
            "description": "MT5 timeframe code (e.g. H1/M30/D1)",
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
            "description": "Indicator name plus optional numeric params.",
        },
        "DenoiseSpec": {
            "type": "object",
            "properties": {
                "method": {"$ref": "#/$defs/DenoiseMethod"},
                "params": {"type": "object", "description": "Method-specific overrides", "additionalProperties": True},
                "columns": {"type": "array", "items": {"type": "string"}},
                "when": {"$ref": "#/$defs/WhenSpec"},
                "causality": {"$ref": "#/$defs/CausalitySpec"},
                "keep_original": {"type": "boolean"},
                "suffix": {"type": "string"},
            },
            "required": ["method"],
            "additionalProperties": False,
            "description": "Denoise spec: method plus optional columns/params.",
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
            "description": "Simplify/segment/encode spec for outputs.",
        },
        "VolatilityParams": {
            "type": "object",
            "properties": {
                "halflife": {"type": ["number", "null"]},
                "lambda_": {"type": ["number", "null"], "description": "EWMA smoothing weight"},
                "lookback": {"type": "integer"},
                "window": {"type": "integer"},
                "fit_bars": {"type": "integer"},
                "mean": {"type": "string", "enum": ["Zero", "Constant"]},
                "dist": {"type": "string", "enum": ["normal"]},
            },
            "additionalProperties": False,
            "description": "Volatility estimator parameters.",
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





def apply_param_hints(schema: Dict[str, Any]) -> Dict[str, Any]:
    params_obj = _parameters_obj(schema)
    props = params_obj.get("properties", {}) if isinstance(params_obj, dict) else {}
    for name, prop in list(props.items()):
        if not isinstance(prop, dict):
            continue
        hint = PARAM_HINTS.get(name)
        if hint and not prop.get("description"):
            prop["description"] = hint
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



def _allow_null(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of schema that also accepts null."""
    updated = dict(schema)
    schema_type = updated.get("type")
    if schema_type is None:
        if "oneOf" in updated:
            updated["oneOf"] = list(updated["oneOf"]) + [{"type": "null"}]
        elif "anyOf" in updated:
            updated["anyOf"] = list(updated["anyOf"]) + [{"type": "null"}]
        else:
            updated = {"anyOf": [schema, {"type": "null"}]}
        return updated
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            updated["type"] = schema_type + ["null"]
    else:
        if schema_type != "null":
            updated["type"] = [schema_type, "null"]
    return updated


_TYPED_DICT_REFS = {
    "IndicatorSpec": "#/$defs/IndicatorSpec",
    "DenoiseSpec": "#/$defs/DenoiseSpec",
    "SimplifySpec": "#/$defs/SimplifySpec",
    "VolatilityParams": "#/$defs/VolatilityParams",
}


def _type_hint_to_schema(type_hint: Any) -> Dict[str, Any]:
    """Convert a Python type hint to a minimal JSON Schema fragment."""
    if type_hint is None:
        return {"type": "string"}
    if type_hint is Any:  # allow arbitrary content
        return {}
    origin = get_origin(type_hint)
    if origin is Literal:
        literals = [lit for lit in get_args(type_hint) if lit is not None]
        if not literals:
            return {"type": "string"}
        literal_types = {type(lit) for lit in literals}
        if literal_types == {bool}:
            return {"type": "boolean"}
        if literal_types == {int}:
            return {"type": "integer", "enum": literals}
        if literal_types == {float}:
            return {"type": "number", "enum": literals}
        return {"type": "string", "enum": [str(lit) for lit in literals]}
    if origin is Union:
        args = list(get_args(type_hint))
        allow_null = False
        non_null_args = []
        for arg in args:
            if arg is type(None):
                allow_null = True
            else:
                non_null_args.append(arg)
        if not non_null_args:
            return {"type": "null"}
        if len(non_null_args) == 1:
            schema = _type_hint_to_schema(non_null_args[0])
        else:
            schema = {"oneOf": [_type_hint_to_schema(arg) for arg in non_null_args]}
        if allow_null:
            schema = _allow_null(schema)
        return schema
    if origin in (list, List, tuple, Tuple, set, frozenset):
        args = get_args(type_hint)
        item_type = args[0] if args else Any
        item_schema = _type_hint_to_schema(item_type)
        # Ensure items schema defaults to accepting any value if empty
        if not item_schema:
            item_schema = {}
        return {"type": "array", "items": item_schema}
    if origin in (dict, Dict):
        args = get_args(type_hint)
        value_type = args[1] if len(args) > 1 else Any
        value_schema = _type_hint_to_schema(value_type)
        if not value_schema:
            value_schema = {}
        return {"type": "object", "additionalProperties": value_schema or True}
    # Handle direct builtins and aliases
    if type_hint in (str, bytes):
        return {"type": "string"}
    if type_hint is int:
        return {"type": "integer"}
    if type_hint is float:
        return {"type": "number"}
    if type_hint is bool:
        return {"type": "boolean"}
    if type_hint is dict:
        return {"type": "object", "additionalProperties": True}
    if type_hint is list or type_hint is tuple:
        return {"type": "array"}
    ref_name = getattr(type_hint, "__name__", "")
    if ref_name in _TYPED_DICT_REFS:
        return {"$ref": _TYPED_DICT_REFS[ref_name]}
    annotations = getattr(type_hint, "__annotations__", None)
    if annotations and ref_name:
        ref = _TYPED_DICT_REFS.get(ref_name)
        if ref:
            return {"$ref": ref}
    return {"type": "string"}

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
        prop_schema = _type_hint_to_schema(p.get("type"))
        if not prop_schema:
            prop_schema = {"type": "string"}
        props[name] = prop_schema
        default_val = p.get("default")
        if default_val is not None:
            if isinstance(default_val, (str, int, float, bool, list, dict)):
                try:
                    props[name]["default"] = default_val
                except Exception:
                    pass
        if p.get("required"):
            req.append(name)
    _ensure_defs(schema)
    apply_timeframe_ref(schema)
    apply_param_hints(schema)
    return schema


def enrich_schema_with_shared_defs(schema: Dict[str, Any], func_info: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure schema has $defs and timeframe refs. If empty, build minimal one."""
    if not isinstance(schema, dict) or not schema:
        schema = build_minimal_schema(func_info)
        return schema
    _ensure_defs(schema)
    apply_timeframe_ref(schema)
    apply_param_hints(schema)
    return schema




def get_shared_enum_lists() -> Dict[str, List[str]]:
    """Return enum lists used to enrich schemas when attaching to tools."""
    enums: Dict[str, List[str]] = {
        "DENOISE_METHODS": list(_DENOISE_METHODS),
        "SIMPLIFY_MODES": list(_SIMPLIFY_MODES),
        "SIMPLIFY_METHODS": list(_SIMPLIFY_METHODS),
        "PIVOT_METHODS": list(_PIVOT_METHODS),
        "FORECAST_METHODS": list(_FORECAST_METHODS),
    }
    if _CATEGORY_CHOICES:
        enums["CATEGORY_CHOICES"] = list(_CATEGORY_CHOICES)
    if _INDICATOR_NAME_CHOICES:
        enums["INDICATOR_NAME_CHOICES"] = list(_INDICATOR_NAME_CHOICES)
    return enums


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
