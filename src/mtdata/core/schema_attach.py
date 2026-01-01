"""
Shared schema defs and dynamic attachment of JSON Schemas to MCP tools.
Extracted from core.server to keep server thinner.
"""
from typing import Any, Dict
from .server_utils import get_mcp_registry

from .schema import (
    enrich_schema_with_shared_defs as _enrich_schema_with_shared_defs,
    build_minimal_schema as _build_minimal_schema,
    get_function_info as _get_function_info,
    complex_defs as _complex_defs,
    apply_param_hints as _apply_param_hints,
)


def server_shared_defs(shared_enums: Dict[str, Any]) -> Dict[str, Any]:
    """Build server-level $defs based on provided enum lists (avoids circular imports)."""
    defs: Dict[str, Any] = {}
    try:
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
    except Exception:
        pass
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
            try:
                if "$defs" not in schema:
                    schema["$defs"] = {}
                schema["$defs"].update({k: v for k, v in shared_defs.items() if k not in schema["$defs"]})
                schema["$defs"].update({k: v for k, v in _complex_defs().items() if k not in schema["$defs"]})
            except Exception:
                pass
            try:
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
                    if "model_params" in params:
                        params["model_params"] = {
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
                            "enum": ["mc", "closed_form", "auto"],
                            "description": "Barrier probability mode: 'mc' (Monte Carlo), 'closed_form', or 'auto' (MC with auto method).",
                        }
                    if "mc_method" in params:
                        params["mc_method"] = {
                            "type": "string",
                            "enum": ["mc_gbm", "mc_gbm_bb", "hmm_mc", "garch", "bootstrap", "heston", "jump_diffusion", "auto"],
                            "description": "Monte Carlo engine for barrier simulation.",
                        }
                if name == "forecast_barrier_optimize":
                    if "method" in params:
                        params["method"] = {
                            "type": "string",
                            "enum": ["mc_gbm", "mc_gbm_bb", "hmm_mc", "garch", "bootstrap", "heston", "jump_diffusion", "auto"],
                            "description": "Barrier simulation method for optimization.",
                        }
                # Trading schemas: add enums and param docs where helpful
                if name == "trading_place":
                    # Clarify acceptable order type values for orders
                    if "order_type" in params:
                        params["order_type"] = {
                            "type": "string",
                            "enum": [
                                "BUY", "SELL",
                                "BUYLIMIT", "BUYSTOP",
                                "SELLLIMIT", "SELLSTOP"
                            ],
                            "description": "BUY/SELL (market) or explicit pending type."
                        }
                    # Document expiration flexibility
                    if "expiration" in params:
                        params["expiration"] = {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "null"}
                            ],
                            "description": "Dateparser input, epoch seconds, or GTC token."
                        }
            except Exception:
                pass
            _apply_param_hints(schema)
            try:
                setattr(obj, "schema", schema)
            except Exception:
                pass
            try:
                setattr(func, "schema", schema)
            except Exception:
                pass
    except Exception:
        pass
