"""
Shared schema defs and dynamic attachment of JSON Schemas to MCP tools.
Extracted from core.server to keep server thinner.
"""
import logging
from typing import Any, Callable, Dict

from .schema import (
    apply_param_hints as _apply_param_hints,
)
from .schema import (
    build_minimal_schema as _build_minimal_schema,
)
from .schema import (
    complex_defs as _complex_defs,
)
from .schema import (
    enrich_schema_with_shared_defs as _enrich_schema_with_shared_defs,
)
from .schema import (
    get_function_info as _get_function_info,
)
from .server_utils import get_mcp_registry

logger = logging.getLogger(__name__)

_BARRIER_PROB_METHODS = [
    "mc_gbm",
    "mc_gbm_bb",
    "hmm_mc",
    "garch",
    "bootstrap",
    "heston",
    "jump_diffusion",
    "closed_form",
    "auto",
]

_BARRIER_OPTIMIZE_METHODS = [
    "mc_gbm",
    "mc_gbm_bb",
    "hmm_mc",
    "garch",
    "bootstrap",
    "heston",
    "jump_diffusion",
    "auto",
]

_TRADE_PLACE_STRING_ORDER_TYPES = [
    "BUY",
    "SELL",
    "BUY_LIMIT",
    "BUY_STOP",
    "SELL_LIMIT",
    "SELL_STOP",
    "ORDER_TYPE_BUY",
    "ORDER_TYPE_SELL",
    "ORDER_TYPE_BUY_LIMIT",
    "ORDER_TYPE_BUY_STOP",
    "ORDER_TYPE_SELL_LIMIT",
    "ORDER_TYPE_SELL_STOP",
]

_SchemaPatcher = Callable[[Dict[str, Any]], None]


def _safe_schema_op(operation: str, func, default=None):
    try:
        return func()
    except Exception as exc:
        logger.debug("schema_attach operation=%s skipped: %s", operation, exc)
        return default


def server_shared_defs(shared_enums: Dict[str, Any]) -> Dict[str, Any]:
    """Build server-level $defs based on provided enum lists (avoids circular imports)."""
    defs: Dict[str, Any] = {}
    def _build_defs() -> None:
        defs.update({
            "OhlcvChar": {"type": "string", "enum": ["O","H","L","C","V"], "description": "OHLCV column code"},
            "DenoiseMethod": {"type": "string", "enum": list(shared_enums.get("DENOISE_METHODS", []))},
            "SimplifyMode": {"type": "string", "enum": list(shared_enums.get("SIMPLIFY_MODES", []))},
            "SimplifyMethod": {"type": "string", "enum": list(shared_enums.get("SIMPLIFY_METHODS", []))},
            "EncodeSchema": {"type": "string", "enum": ["envelope","delta"]},
            "SymbolicSchema": {"type": "string", "enum": ["sax"]},
            "PivotMethod": {"type": "string", "enum": list(shared_enums.get("PIVOT_METHODS", []))},
            "ForecastMethod": {"type": "string", "enum": list(shared_enums.get("FORECAST_METHODS", []))},
            "QuantitySpec": {"type": "string", "enum": ["price","return","volatility"]},
            "VolatilityMethod": {"type": "string", "enum": [
                "ewma","parkinson","gk","rs","yang_zhang","rolling_std",
                "garch","egarch","gjr_garch",
                "arima","sarima","ets","theta"
            ]},
            "WhenSpec": {"type": "string", "enum": ["pre_ti","post_ti"]},
            "CausalitySpec": {"type": "string", "enum": ["causal","zero_phase"]},
            "TargetSpec": {"type": "string", "enum": ["price","return"]},
        })
        if shared_enums.get("CATEGORY_CHOICES"):
            defs["IndicatorCategory"] = {"type": "string", "enum": list(shared_enums["CATEGORY_CHOICES"])}
        if shared_enums.get("INDICATOR_NAME_CHOICES"):
            defs["IndicatorName"] = {"type": "string", "enum": list(shared_enums["INDICATOR_NAME_CHOICES"])}
    _safe_schema_op("server_shared_defs", _build_defs)
    return defs


def _schema_params(schema: Dict[str, Any]) -> tuple[Dict[str, Any], set[str]]:
    params_obj = schema.get("parameters", {})
    if not isinstance(params_obj, dict):
        return {}, set()
    params = params_obj.get("properties", {})
    if not isinstance(params, dict):
        return {}, set()
    required_params = set(params_obj.get("required", []))
    return params, required_params


def _set_ref(
    params: Dict[str, Any],
    required_params: set[str],
    param_name: str,
    ref: str,
    *,
    allow_null: bool = False,
) -> None:
    if param_name not in params:
        return
    if allow_null and param_name not in required_params:
        params[param_name] = {"anyOf": [{"$ref": ref}, {"type": "null"}]}
        return
    params[param_name] = {"$ref": ref}


def _patch_forecast_generate_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    _set_ref(params, required_params, "quantity", "#/$defs/QuantitySpec")
    _set_ref(params, required_params, "denoise", "#/$defs/DenoiseSpec", allow_null=True)
    if "params" in params:
        params["params"] = {
            "type": "object",
            "additionalProperties": True,
        }


def _patch_indicators_list_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    if "IndicatorCategory" in schema.get("$defs", {}):
        _set_ref(params, required_params, "category", "#/$defs/IndicatorCategory")


def _patch_indicators_describe_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    if "IndicatorName" in schema.get("$defs", {}):
        _set_ref(params, required_params, "name", "#/$defs/IndicatorName")


def _patch_data_fetch_candles_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    if "indicators" in params:
        indicator_options = [
            {
                "type": "string",
                "description": "Compact indicator spec like 'rsi(14),ema(20),macd(12,26,9)'.",
            },
            {"type": "array", "items": {"$ref": "#/$defs/IndicatorSpec"}},
        ]
        if "indicators" not in required_params:
            indicator_options.append({"type": "null"})
        params["indicators"] = {"anyOf": indicator_options}
    _set_ref(params, required_params, "denoise", "#/$defs/DenoiseSpec", allow_null=True)
    _set_ref(params, required_params, "simplify", "#/$defs/SimplifySpec", allow_null=True)


def _patch_data_fetch_ticks_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    _set_ref(params, required_params, "simplify", "#/$defs/SimplifySpec", allow_null=True)


def _patch_forecast_barrier_prob_schema(schema: Dict[str, Any]) -> None:
    params, _required_params = _schema_params(schema)
    if "method" not in params:
        return
    params["method"] = {
        "type": "string",
        "enum": list(_BARRIER_PROB_METHODS),
        "description": "Barrier probability algorithm. Use a Monte Carlo engine, 'closed_form', or 'auto'.",
    }


def _patch_forecast_barrier_optimize_schema(schema: Dict[str, Any]) -> None:
    params, _required_params = _schema_params(schema)
    if "method" not in params:
        return
    params["method"] = {
        "type": "string",
        "enum": list(_BARRIER_OPTIMIZE_METHODS),
        "description": "Barrier simulation method for optimization.",
    }


def _patch_trade_place_schema(schema: Dict[str, Any]) -> None:
    params_obj = schema.get("parameters", {})
    if isinstance(params_obj, dict):
        params_obj["required"] = ["symbol", "volume", "order_type"]
    params, _required_params = _schema_params(schema)
    if "order_type" in params:
        params["order_type"] = {
            "anyOf": [
                {
                    "type": "string",
                    "enum": list(_TRADE_PLACE_STRING_ORDER_TYPES),
                },
                {
                    "type": "integer",
                    "enum": [0, 1, 2, 3, 4, 5],
                },
            ],
            "description": "Required. BUY/SELL (market by default; pending if price is provided), pending aliases, or MT5 numeric constants (0..5)."
        }
    if "expiration" in params:
        params["expiration"] = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "null"}
            ],
            "description": "Dateparser input, UTC epoch seconds, or GTC token."
        }


_TOOL_SCHEMA_PATCHERS: Dict[str, tuple[_SchemaPatcher, ...]] = {
    "forecast_generate": (_patch_forecast_generate_schema,),
    "indicators_list": (_patch_indicators_list_schema,),
    "indicators_describe": (_patch_indicators_describe_schema,),
    "data_fetch_candles": (_patch_data_fetch_candles_schema,),
    "data_fetch_ticks": (_patch_data_fetch_ticks_schema,),
    "forecast_barrier_prob": (_patch_forecast_barrier_prob_schema,),
    "forecast_barrier_optimize": (_patch_forecast_barrier_optimize_schema,),
    "trade_place": (_patch_trade_place_schema,),
}


def attach_schemas_to_tools(mcp: Any, shared_enums: Dict[str, Any]) -> None:
    """Attach enriched JSON Schemas to registered MCP tools on the given server."""
    try:
        registry = get_mcp_registry(mcp)
        if not registry:
            return
        shared_defs = server_shared_defs(shared_enums)
        for name, obj in list(registry.items()):
            func = None
            for attr in ("func", "function", "callable", "handler", "wrapped", "_func"):
                try:
                    val = getattr(obj, attr)
                    if callable(val):
                        func = val
                        break
                except Exception:
                    continue
            if func is None:
                func = obj if callable(obj) else None
            if not callable(func):
                continue
            info = _get_function_info(func)
            schema = _build_minimal_schema(info)
            schema = _enrich_schema_with_shared_defs(schema, info)
            def _update_defs() -> None:
                if "$defs" not in schema:
                    schema["$defs"] = {}
                schema["$defs"].update({k: v for k, v in shared_defs.items() if k not in schema["$defs"]})
                schema["$defs"].update({k: v for k, v in _complex_defs().items() if k not in schema["$defs"]})

            _safe_schema_op(f"update_defs:{name}", _update_defs)

            def _patch_params() -> None:
                for patcher in _TOOL_SCHEMA_PATCHERS.get(name, ()):
                    patcher(schema)
            _safe_schema_op(f"patch_params:{name}", _patch_params)
            _apply_param_hints(schema)
            _safe_schema_op(f"set_obj_schema:{name}", lambda: setattr(obj, "schema", schema))
            _safe_schema_op(f"set_func_schema:{name}", lambda: setattr(func, "schema", schema))
    except Exception:
        pass
