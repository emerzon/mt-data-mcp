"""
Shared schema defs and dynamic attachment of JSON Schemas to MCP tools.
Extracted from core.server to keep server thinner.
"""
import logging
from typing import Any, Dict
from .server_utils import get_mcp_registry

from .schema import (
    enrich_schema_with_shared_defs as _enrich_schema_with_shared_defs,
    build_minimal_schema as _build_minimal_schema,
    get_function_info as _get_function_info,
    complex_defs as _complex_defs,
    apply_param_hints as _apply_param_hints,
)

logger = logging.getLogger(__name__)


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
                params = schema.get("parameters", {}).get("properties", {})
                required_params = set(schema.get("parameters", {}).get("required", []))

                def _set_ref(param_name: str, ref: str, allow_null: bool = False) -> None:
                    if param_name not in params:
                        return
                    if allow_null and param_name not in required_params:
                        params[param_name] = {"anyOf": [{"$ref": ref}, {"type": "null"}]}
                    else:
                        params[param_name] = {"$ref": ref}

                if name == "forecast_generate":
                    _set_ref("quantity", "#/$defs/QuantitySpec")
                    _set_ref("denoise", "#/$defs/DenoiseSpec", allow_null=True)
                    if "params" in params:
                        params["params"] = {
                            "type": "object",
                            "additionalProperties": True,
                        }
                if name == "indicators_list" and "category" in params and "IndicatorCategory" in schema.get("$defs", {}):
                    _set_ref("category", "#/$defs/IndicatorCategory")
                if name == "indicators_describe" and "name" in params and "IndicatorName" in schema.get("$defs", {}):
                    _set_ref("name", "#/$defs/IndicatorName")
                if name == "data_fetch_candles":
                    if "indicators" in params:
                        params["indicators"] = {"type": "array", "items": {"$ref": "#/$defs/IndicatorSpec"}}
                    _set_ref("denoise", "#/$defs/DenoiseSpec", allow_null=True)
                    _set_ref("simplify", "#/$defs/SimplifySpec", allow_null=True)
                if name == "data_fetch_ticks":
                    _set_ref("simplify", "#/$defs/SimplifySpec", allow_null=True)
                if name == "forecast_barrier_prob":
                    if "method" in params:
                        params["method"] = {
                            "type": "string",
                            "enum": ["mc_gbm", "mc_gbm_bb", "hmm_mc", "garch", "bootstrap", "heston", "jump_diffusion", "closed_form", "auto"],
                            "description": "Barrier probability algorithm. Use a Monte Carlo engine, 'closed_form', or 'auto'.",
                        }
                if name == "forecast_barrier_optimize":
                    if "method" in params:
                        params["method"] = {
                            "type": "string",
                            "enum": ["mc_gbm", "mc_gbm_bb", "hmm_mc", "garch", "bootstrap", "heston", "jump_diffusion", "auto"],
                            "description": "Barrier simulation method for optimization.",
                        }
                # Trading schemas: add enums and param docs where helpful
                if name == "trade_place":
                    params_obj = schema.get("parameters", {})
                    if isinstance(params_obj, dict):
                        params_obj["required"] = ["symbol", "volume", "order_type"]
                    # Clarify acceptable order type values for orders
                    if "order_type" in params:
                        params["order_type"] = {
                            "anyOf": [
                                {
                                    "type": "string",
                                    "enum": [
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
                                    ],
                                },
                                {
                                    "type": "integer",
                                    "enum": [0, 1, 2, 3, 4, 5],
                                },
                            ],
                            "description": "Required. BUY/SELL (market by default; pending if price is provided), pending aliases, or MT5 numeric constants (0..5)."
                        }
                    # Document expiration flexibility
                    if "expiration" in params:
                        params["expiration"] = {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "null"}
                            ],
                            "description": "Dateparser input, UTC epoch seconds, or GTC token."
                        }
            _safe_schema_op(f"patch_params:{name}", _patch_params)
            _apply_param_hints(schema)
            _safe_schema_op(f"set_obj_schema:{name}", lambda: setattr(obj, "schema", schema))
            _safe_schema_op(f"set_func_schema:{name}", lambda: setattr(func, "schema", schema))
    except Exception:
        pass
