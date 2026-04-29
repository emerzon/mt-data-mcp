"""
Shared schema defs and dynamic attachment of JSON Schemas to MCP tools.
Extracted from core.server to keep server thinner.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, Iterable

from ..shared.parameter_contracts import (
    OUTPUT_EXTRA_FULL_ALIASES,
    OUTPUT_EXTRAS,
    REMOVED_PUBLIC_OUTPUT_PARAMS,
)
from ..shared.schema import (
    apply_param_hints as _apply_param_hints,
)
from ..shared.schema import (
    build_minimal_schema as _build_minimal_schema,
)
from ..shared.schema import (
    complex_defs as _complex_defs,
)
from ..shared.schema import (
    enrich_schema_with_shared_defs as _enrich_schema_with_shared_defs,
)
from ..shared.schema import (
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
            "OhlcvChar": {"type": "string", "enum": ["O", "H", "L", "C", "V"], "description": "OHLCV column code"},
            "DenoiseMethod": {"type": "string", "enum": list(shared_enums.get("DENOISE_METHODS", []))},
            "SimplifyMode": {"type": "string", "enum": list(shared_enums.get("SIMPLIFY_MODES", []))},
            "SimplifyMethod": {"type": "string", "enum": list(shared_enums.get("SIMPLIFY_METHODS", []))},
            "EncodeSchema": {"type": "string", "enum": ["envelope", "delta"]},
            "SymbolicSchema": {"type": "string", "enum": ["sax"]},
            "PivotMethod": {"type": "string", "enum": list(shared_enums.get("PIVOT_METHODS", []))},
            "ForecastMethod": {"type": "string", "enum": list(shared_enums.get("FORECAST_METHODS", []))},
            "QuantitySpec": {"type": "string", "enum": ["price", "return", "volatility"]},
            "VolatilityMethod": {"type": "string", "enum": [
                "ewma", "parkinson", "gk", "rs", "yang_zhang", "rolling_std",
                "garch", "egarch", "gjr_garch",
                "arima", "sarima", "ets", "theta",
            ]},
            "WhenSpec": {"type": "string", "enum": ["pre_ti", "post_ti"]},
            "CausalitySpec": {"type": "string", "enum": ["causal", "zero_phase"]},
            "TargetSpec": {"type": "string", "enum": ["price", "return"]},
        })
        if shared_enums.get("CATEGORY_CHOICES"):
            defs["IndicatorCategory"] = {"type": "string", "enum": list(shared_enums["CATEGORY_CHOICES"])}
        if shared_enums.get("INDICATOR_NAME_CHOICES"):
            defs["IndicatorName"] = {"type": "string", "enum": list(shared_enums["INDICATOR_NAME_CHOICES"])}

    _safe_schema_op("server_shared_defs", _build_defs)
    return defs


def _schema_obj(schema: Dict[str, Any]) -> Dict[str, Any]:
    params_obj = schema.get("parameters")
    if isinstance(params_obj, dict):
        return params_obj
    return schema if isinstance(schema, dict) else {}


def _schema_params(schema: Dict[str, Any]) -> tuple[Dict[str, Any], set[str]]:
    params_obj = _schema_obj(schema)
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


def _set_simplify_param(params: Dict[str, Any], required_params: set[str]) -> None:
    if "simplify" not in params:
        return
    options = [
        {"$ref": "#/$defs/SimplifySpec"},
        {"type": "boolean"},
        {"type": "string", "enum": ["on", "off", "auto"]},
    ]
    if "simplify" not in required_params:
        options.append({"type": "null"})
    params["simplify"] = {
        "description": (
            "Optional data reduction spec. Use a dict such as "
            "{'method': 'lttb', 'points': 100}; true, on, or auto enables "
            "default simplification; false or off disables it."
        ),
        "anyOf": options,
        "examples": [{"method": "lttb", "points": 100}, True, "off"],
    }


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
            {"type": "string"},
            {"type": "array", "items": {"$ref": "#/$defs/IndicatorSpec"}},
        ]
        if "indicators" not in required_params:
            indicator_options.append({"type": "null"})
        params["indicators"] = {"anyOf": indicator_options}
    _set_ref(params, required_params, "denoise", "#/$defs/DenoiseSpec", allow_null=True)
    _set_simplify_param(params, required_params)


def _patch_data_fetch_ticks_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    _set_simplify_param(params, required_params)


def _patch_forecast_barrier_prob_schema(schema: Dict[str, Any]) -> None:
    params, _required_params = _schema_params(schema)
    if "method" not in params:
        return
    params["method"] = {
        "type": "string",
        "enum": list(_BARRIER_PROB_METHODS),
        "description": "Barrier probability algorithm.",
    }


def _patch_labels_triple_barrier_schema(schema: Dict[str, Any]) -> None:
    # TP/SL unit-family exclusivity is already enforced by runtime validation.
    # The MCP/OpenAI tool-schema subset rejects top-level allOf/anyOf/not
    # combinators on input objects, so keep this attached schema flat.
    return


def _patch_forecast_barrier_optimize_schema(schema: Dict[str, Any]) -> None:
    params, _required_params = _schema_params(schema)
    if "method" not in params:
        return
    params["method"] = {
        "type": "string",
        "enum": list(_BARRIER_OPTIMIZE_METHODS),
        "description": "Barrier simulation method.",
    }


def _patch_trade_place_schema(schema: Dict[str, Any]) -> None:
    params_obj = _schema_obj(schema)
    if isinstance(params_obj, dict):
        params_obj["required"] = ["symbol", "volume", "order_type"]
    params, _required_params = _schema_params(schema)
    if "order_type" in params:
        params["order_type"] = {
            "type": "string",
            "enum": list(_TRADE_PLACE_STRING_ORDER_TYPES),
            "description": (
                "Canonical order type: BUY/SELL for market orders or "
                "BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP for pending orders."
            ),
        }
    if "expiration" in params:
        params["expiration"] = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "null"},
            ],
            "description": "Dateparser input, UTC epoch seconds, or GTC token.",
        }


def _patch_wait_event_schema(schema: Dict[str, Any]) -> None:
    from .data.requests import WaitEventRequest

    params, _required_params = _schema_params(schema)
    wait_event_schema = WaitEventRequest.model_json_schema()
    wait_event_props = wait_event_schema.get("properties")
    if not isinstance(wait_event_props, dict):
        return

    for field_name in ("watch_for", "end_on"):
        field_schema = wait_event_props.get(field_name)
        if isinstance(field_schema, dict):
            params[field_name] = copy.deepcopy(field_schema)

    defs = wait_event_schema.get("$defs")
    if isinstance(defs, dict):
        schema_defs = schema.setdefault("$defs", {})
        if isinstance(schema_defs, dict):
            schema_defs.update(copy.deepcopy(defs))


_TOOL_SCHEMA_PATCHERS: Dict[str, tuple[_SchemaPatcher, ...]] = {
    "forecast_generate": (_patch_forecast_generate_schema,),
    "indicators_list": (_patch_indicators_list_schema,),
    "indicators_describe": (_patch_indicators_describe_schema,),
    "data_fetch_candles": (_patch_data_fetch_candles_schema,),
    "data_fetch_ticks": (_patch_data_fetch_ticks_schema,),
    "forecast_barrier_prob": (_patch_forecast_barrier_prob_schema,),
    "labels_triple_barrier": (_patch_labels_triple_barrier_schema,),
    "forecast_barrier_optimize": (_patch_forecast_barrier_optimize_schema,),
    "trade_place": (_patch_trade_place_schema,),
    "wait_event": (_patch_wait_event_schema,),
}


def _iter_manager_tools(mcp: Any) -> Iterable[tuple[str, Any]]:
    manager = getattr(mcp, "_tool_manager", None)
    tools = getattr(manager, "_tools", None)
    if isinstance(tools, dict):
        return list(tools.items())
    return []


def _extract_callable(obj: Any) -> Any:
    for attr in ("func", "function", "callable", "handler", "wrapped", "_func", "fn"):
        try:
            val = getattr(obj, attr)
            if callable(val):
                return val
        except Exception:
            continue
    return obj if callable(obj) else None


def _merge_shared_defs(schema: Dict[str, Any], shared_defs: Dict[str, Any]) -> None:
    if "$defs" not in schema or not isinstance(schema.get("$defs"), dict):
        schema["$defs"] = {}
    schema["$defs"].update({k: v for k, v in shared_defs.items() if k not in schema["$defs"]})
    schema["$defs"].update({k: v for k, v in _complex_defs().items() if k not in schema["$defs"]})


def _summarize_description(text: str) -> str:
    for line in str(text or "").splitlines():
        compact = " ".join(line.split())
        if compact:
            return compact
    return ""


def _dedupe_union_options(options: list[Any]) -> list[Any]:
    seen: set[str] = set()
    deduped: list[Any] = []
    for option in options:
        key = repr(option)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    option_types = {
        opt.get("type")
        for opt in deduped
        if isinstance(opt, dict) and isinstance(opt.get("type"), str)
    }
    if "integer" in option_types and "number" in option_types:
        deduped = [
            opt
            for opt in deduped
            if not (isinstance(opt, dict) and opt.get("type") == "integer" and set(opt.keys()) == {"type"})
        ]
    return deduped


def _strip_schema_noise(value: Any, *, drop_descriptions: bool) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if key == "title":
                continue
            if key == "description" and drop_descriptions:
                continue
            if key == "default" and item is None:
                continue
            if key == "additionalProperties" and item is True:
                continue
            child = _strip_schema_noise(item, drop_descriptions=drop_descriptions)
            if key in {"anyOf", "oneOf"} and isinstance(child, list):
                child = _dedupe_union_options(child)
            cleaned[key] = child
        if isinstance(cleaned.get("required"), list) and not cleaned["required"]:
            cleaned.pop("required", None)
        return cleaned
    if isinstance(value, list):
        return [_strip_schema_noise(item, drop_descriptions=drop_descriptions) for item in value]
    return value


def _compact_optional_property_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(schema)
    for union_key in ("anyOf", "oneOf"):
        options = updated.get(union_key)
        if not isinstance(options, list):
            continue
        non_null = [
            option
            for option in options
            if not (isinstance(option, dict) and option.get("type") == "null" and set(option.keys()) == {"type"})
        ]
        non_null = _dedupe_union_options(non_null)
        if len(non_null) == len(options):
            continue
        if len(non_null) == 1:
            collapsed = dict(non_null[0]) if isinstance(non_null[0], dict) else non_null[0]
            if isinstance(collapsed, dict):
                for key, value in updated.items():
                    if key != union_key:
                        collapsed.setdefault(key, value)
            return collapsed if isinstance(collapsed, dict) else updated
        updated[union_key] = non_null
    return updated


def _compact_schema_shape(schema: Dict[str, Any]) -> Dict[str, Any]:
    compact = _strip_schema_noise(copy.deepcopy(schema), drop_descriptions=True)
    params_obj = _schema_obj(compact)
    props = params_obj.get("properties", {}) if isinstance(params_obj, dict) else {}
    required = set(params_obj.get("required", [])) if isinstance(params_obj, dict) else set()
    if isinstance(props, dict):
        for name, prop in list(props.items()):
            if isinstance(prop, dict) and name not in required:
                props[name] = _compact_optional_property_schema(prop)
    return compact


def _enforce_public_output_contract(schema: Dict[str, Any]) -> None:
    params_obj = _schema_obj(schema)
    props = params_obj.get("properties") if isinstance(params_obj, dict) else None
    if not isinstance(props, dict):
        return
    for name in REMOVED_PUBLIC_OUTPUT_PARAMS:
        props.pop(name, None)
    required = params_obj.get("required")
    if isinstance(required, list):
        params_obj["required"] = [name for name in required if name not in REMOVED_PUBLIC_OUTPUT_PARAMS]
    props.setdefault(
        "json",
        {
            "type": "boolean",
            "default": False,
            "description": "Return structured JSON instead of default TOON text.",
        },
    )
    props.setdefault(
        "extras",
        {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": sorted(OUTPUT_EXTRAS),
                    },
                },
                {"type": "string"},
            ],
            "description": (
                "Optional richer output sections. Use "
                f"{'/'.join(sorted(OUTPUT_EXTRA_FULL_ALIASES))} for every supported section."
            ),
        },
    )
    if isinstance(props.get("extras"), dict):
        props["extras"].setdefault(
            "description",
            (
                "Optional richer output sections. Use "
                f"{'/'.join(sorted(OUTPUT_EXTRA_FULL_ALIASES))} for every supported section."
            ),
        )


def _collect_schema_refs(value: Any, refs: set[str], *, skip_defs: bool) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$defs" and skip_defs:
                continue
            if key == "$ref" and isinstance(item, str) and item.startswith("#/$defs/"):
                refs.add(item.rsplit("/", 1)[-1])
                continue
            _collect_schema_refs(item, refs, skip_defs=skip_defs)
        return
    if isinstance(value, list):
        for item in value:
            _collect_schema_refs(item, refs, skip_defs=skip_defs)


def _prune_unused_defs(schema: Dict[str, Any]) -> Dict[str, Any]:
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return schema

    used: set[str] = set()
    _collect_schema_refs(schema, used, skip_defs=True)
    pending = list(used)
    while pending:
        ref_name = pending.pop()
        definition = defs.get(ref_name)
        if not isinstance(definition, dict):
            continue
        nested: set[str] = set()
        _collect_schema_refs(definition, nested, skip_defs=False)
        for child in sorted(nested - used):
            used.add(child)
            pending.append(child)

    if not used:
        schema.pop("$defs", None)
        return schema

    schema["$defs"] = {name: defs[name] for name in defs if name in used}
    return schema


def _build_internal_schema(public_schema: Dict[str, Any]) -> Dict[str, Any]:
    internal_schema: Dict[str, Any] = {
        "parameters": copy.deepcopy({k: v for k, v in public_schema.items() if k != "$defs"})
    }
    if isinstance(public_schema.get("$defs"), dict):
        internal_schema["$defs"] = copy.deepcopy(public_schema["$defs"])
    _apply_param_hints(internal_schema)
    return internal_schema


def attach_schemas_to_tools(mcp: Any, shared_enums: Dict[str, Any]) -> None:
    """Attach enriched JSON Schemas to registered MCP tools on the given server."""
    try:
        registry = get_mcp_registry(mcp) or {}
        manager_tools = dict(_iter_manager_tools(mcp))
        if not registry and not manager_tools:
            return

        shared_defs = server_shared_defs(shared_enums)
        all_names = sorted(set(registry.keys()) | set(manager_tools.keys()))

        for name in all_names:
            obj = registry.get(name)
            manager_tool = manager_tools.get(name)
            func = _extract_callable(obj) or _extract_callable(manager_tool)
            if not callable(func):
                continue

            info = _get_function_info(func)
            public_schema = getattr(manager_tool, "parameters", None)
            if not isinstance(public_schema, dict) or not public_schema:
                public_schema = _build_minimal_schema(info)
                public_schema = _enrich_schema_with_shared_defs(public_schema, info)
                public_schema = copy.deepcopy(_schema_obj(public_schema))
            else:
                public_schema = copy.deepcopy(public_schema)

            _safe_schema_op(
                f"update_public_defs:{name}",
                lambda: _merge_shared_defs(public_schema, shared_defs),
            )
            _safe_schema_op(
                f"patch_public_params:{name}",
                lambda: [patcher(public_schema) for patcher in _TOOL_SCHEMA_PATCHERS.get(name, ())],
            )
            _safe_schema_op(
                f"enforce_output_contract:{name}",
                lambda: _enforce_public_output_contract(public_schema),
            )

            public_schema = _compact_schema_shape(public_schema)
            public_schema = _prune_unused_defs(public_schema)
            internal_schema = _build_internal_schema(public_schema)

            concise_description = _summarize_description(
                str(getattr(manager_tool, "description", None) or info.get("doc") or "")
            )

            if manager_tool is not None:
                _safe_schema_op(
                    f"set_public_schema:{name}",
                    lambda: setattr(manager_tool, "parameters", copy.deepcopy(public_schema)),
                )
                if concise_description:
                    _safe_schema_op(
                        f"set_public_description:{name}",
                        lambda: setattr(manager_tool, "description", concise_description),
                    )

            if obj is not None:
                _safe_schema_op(
                    f"set_obj_schema:{name}",
                    lambda: setattr(obj, "schema", copy.deepcopy(internal_schema)),
                )
                if concise_description:
                    _safe_schema_op(
                        f"set_obj_description:{name}",
                        lambda: setattr(obj, "description", concise_description),
                    )

            _safe_schema_op(
                f"set_func_schema:{name}",
                lambda: setattr(func, "schema", copy.deepcopy(internal_schema)),
            )
            if concise_description:
                _safe_schema_op(
                    f"set_func_description:{name}",
                    lambda: setattr(func, "description", concise_description),
                )
    except Exception:
        pass
