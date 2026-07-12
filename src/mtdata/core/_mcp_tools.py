"""Shared FastMCP tool wrapping and registry helpers."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import types
from dataclasses import dataclass
from functools import wraps as _wraps
from typing import Any, Dict, List, Optional, Union, cast, get_args, get_origin

from pydantic import BaseModel

from ..shared.annotations import get_runtime_annotations, get_runtime_signature
from ..shared.parameter_contracts import (
    OUTPUT_EXTRA_FULL_ALIASES,
    OUTPUT_EXTRAS,
    PUBLIC_OUTPUT_PARAMS,
)
from ..utils.utils import UNPARSED_BOOL, coerce_scalar, parse_bool_like
from .error_envelope import (
    build_error_payload,
    log_transport_exception,
    normalize_error_payload,
)
from .output_contract import (
    attach_success_guidance,
    apply_output_verbosity,
    normalize_output_extras,
    resolve_output_contract,
)

_ORIG_TOOL_DECORATOR: Any = None
_REGISTRY_UNSET = object()
_TOOL_TIMEOUT_SECONDS = 120
_NO_TIMEOUT_TOOLS = frozenset(
    {
        "wait_event",
        "forecast_optimize_hints",
        "forecast_tune_genetic",
        "forecast_tune_optuna",
    }
)
_PUBLIC_OUTPUT_PARAMS = PUBLIC_OUTPUT_PARAMS
_MARKET_DEPTH_FETCH_ENV = "MTDATA_ENABLE_MARKET_DEPTH_FETCH"
logger = logging.getLogger(__name__)


@dataclass
class _ToolRegistration:
    function: Any = _REGISTRY_UNSET
    tool_object: Any = _REGISTRY_UNSET


_TOOL_METADATA_REGISTRY: Dict[str, _ToolRegistration] = {}


def get_mcp_registry(mcp: Any) -> Optional[Dict[str, Any]]:
    """Return the MCP tool registry if available."""
    for attr in ("tools", "_tools", "registry", "tool_registry", "_tool_registry"):
        reg = getattr(mcp, attr, None)
        if reg and hasattr(reg, "items"):
            return reg
    return None


def _project_tool_registry(field: str) -> Dict[str, Any]:
    projected: Dict[str, Any] = {}
    for name, entry in _TOOL_METADATA_REGISTRY.items():
        value = getattr(entry, field, _REGISTRY_UNSET)
        if value is not _REGISTRY_UNSET:
            projected[name] = value
    return projected


def _replace_dict_contents(target: Dict[str, Any], data: Dict[str, Any]) -> None:
    dict.clear(target)
    dict.update(target, data)


def _sync_tool_registry_views() -> None:
    _replace_dict_contents(_TOOL_REGISTRY, _project_tool_registry("function"))
    _replace_dict_contents(_TOOL_OBJECT_REGISTRY, _project_tool_registry("tool_object"))


def _upsert_tool_registration(
    name: Any,
    *,
    function: Any = _REGISTRY_UNSET,
    tool_object: Any = _REGISTRY_UNSET,
) -> None:
    key = str(name)
    entry = _TOOL_METADATA_REGISTRY.get(key)
    if entry is None:
        entry = _ToolRegistration()
        _TOOL_METADATA_REGISTRY[key] = entry
    if function is not _REGISTRY_UNSET:
        entry.function = function
    if tool_object is not _REGISTRY_UNSET:
        entry.tool_object = tool_object
    _sync_tool_registry_views()


def _remove_tool_registration_field(name: Any, field: str, default: Any = _REGISTRY_UNSET) -> Any:
    key = str(name)
    entry = _TOOL_METADATA_REGISTRY.get(key)
    if entry is None:
        if default is _REGISTRY_UNSET:
            raise KeyError(key)
        return default

    value = getattr(entry, field, _REGISTRY_UNSET)
    if value is _REGISTRY_UNSET:
        if default is _REGISTRY_UNSET:
            raise KeyError(key)
        return default

    setattr(entry, field, _REGISTRY_UNSET)
    if entry.function is _REGISTRY_UNSET and entry.tool_object is _REGISTRY_UNSET:
        _TOOL_METADATA_REGISTRY.pop(key, None)
    _sync_tool_registry_views()
    return value


def _clear_tool_registration_field(field: str) -> None:
    if not _TOOL_METADATA_REGISTRY:
        _sync_tool_registry_views()
        return

    for key, entry in list(_TOOL_METADATA_REGISTRY.items()):
        setattr(entry, field, _REGISTRY_UNSET)
        if entry.function is _REGISTRY_UNSET and entry.tool_object is _REGISTRY_UNSET:
            _TOOL_METADATA_REGISTRY.pop(key, None)
    _sync_tool_registry_views()


class _ToolRegistryView(dict):
    def __init__(self, field: str) -> None:
        super().__init__()
        self._field = field

    def __setitem__(self, key: Any, value: Any) -> None:
        _upsert_tool_registration(key, **{self._field: value})

    def __delitem__(self, key: Any) -> None:
        _remove_tool_registration_field(key, self._field)

    def pop(self, key: Any, default: Any = _REGISTRY_UNSET) -> Any:
        return _remove_tool_registration_field(key, self._field, default)

    def clear(self) -> None:
        _clear_tool_registration_field(self._field)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        existing = dict.get(self, key, _REGISTRY_UNSET)
        if existing is not _REGISTRY_UNSET:
            return existing
        _upsert_tool_registration(key, **{self._field: default})
        return default

    def update(self, *args: Any, **kwargs: Any) -> None:
        merged = dict(*args, **kwargs)
        if not merged:
            return
        for key, value in merged.items():
            entry = _TOOL_METADATA_REGISTRY.get(str(key))
            if entry is None:
                entry = _ToolRegistration()
                _TOOL_METADATA_REGISTRY[str(key)] = entry
            setattr(entry, self._field, value)
        _sync_tool_registry_views()

    def popitem(self) -> tuple[Any, Any]:
        key, value = dict.popitem(self)
        entry = _TOOL_METADATA_REGISTRY.get(str(key))
        if entry is not None:
            setattr(entry, self._field, _REGISTRY_UNSET)
            if entry.function is _REGISTRY_UNSET and entry.tool_object is _REGISTRY_UNSET:
                _TOOL_METADATA_REGISTRY.pop(str(key), None)
        _sync_tool_registry_views()
        return key, value


_TOOL_REGISTRY: Dict[str, Any] = _ToolRegistryView("function")
_TOOL_OBJECT_REGISTRY: Dict[str, Any] = _ToolRegistryView("tool_object")


def _tool_catalog_category(name: str, func: Any) -> str:
    module = str(getattr(func, "__module__", "") or "")
    if name.startswith("trade_") or ".trading" in module:
        return "trading"
    if name.startswith("forecast_") or name.startswith("strategy_"):
        return "forecast"
    if name.startswith("finviz_") or name in {
        "market_depth_fetch",
        "market_scan",
        "market_ticker",
        "market_status",
    }:
        return "market"
    if name.startswith("symbols_"):
        return "symbols"
    if name.startswith("data_") or name == "wait_event":
        return "data"
    if name.startswith("patterns_") or name.startswith("regime_"):
        return "pattern_regime"
    if name.startswith("options_"):
        return "options"
    if name.startswith("report_"):
        return "report"
    if name.startswith("denoise_") or name.startswith("indicators_"):
        return "methods"
    if name in {"pivot_compute_points", "support_resistance_levels", "temporal_analyze"}:
        return "analysis"
    return "research"


def _tool_catalog_description(func: Any) -> str:
    target = getattr(func, "__wrapped__", func)
    doc = inspect.getdoc(target) or inspect.getdoc(func) or ""
    for line in doc.splitlines():
        text = line.strip()
        if text:
            return text
    return ""


def _tool_catalog_parameters(func: Any) -> Dict[str, str]:
    target = getattr(func, "__wrapped__", func)
    try:
        signature = get_runtime_signature(target)
    except Exception:
        return {}
    params = list(signature.parameters.values())
    if len(params) == 1:
        annotation = params[0].annotation
        try:
            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                return {
                    name: "required" if field.is_required() else "optional"
                    for name, field in annotation.model_fields.items()
                }
        except Exception as exc:
            logger.exception(
                "Failed to attach MCP signature for tool %s: %s",
                getattr(func, "__name__", "tool"),
                exc,
            )
    out: Dict[str, str] = {}
    for param in params:
        if param.name.startswith("__"):
            continue
        out[param.name] = "required" if param.default is inspect._empty else "optional"
    return out


def _market_depth_fetch_catalog_state() -> Dict[str, Any]:
    enabled = str(os.getenv(_MARKET_DEPTH_FETCH_ENV) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    out: Dict[str, Any] = {
        "enabled": enabled,
        "enable_env": _MARKET_DEPTH_FETCH_ENV,
    }
    if not enabled:
        out.update(
            {
                "status": "disabled",
                "why_disabled": "Requires broker Level 2/DOM support and is off by default.",
                "recommended_alternative": "market_ticker",
            }
        )
    return out


def _market_depth_fetch_catalog_row(*, detail_mode: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": "market_depth_fetch",
        "category": "market",
        "description": (
            "Return DOM/order-book depth when explicitly enabled and supported by the broker."
        ),
    }
    row.update(_market_depth_fetch_catalog_state())
    if detail_mode in {"standard", "full"}:
        row["parameters"] = {
            "symbol": "required",
            "spread": "optional",
            "require_dom": "optional",
        }
    if detail_mode == "full":
        row["module"] = "mtdata.core.market_depth"
    return row


def registered_tool_catalog(*, detail: str = "compact") -> Dict[str, Any]:
    """Return a generated catalog of registered mtdata tools."""
    from .output_contract import related_tools_for

    requested_detail = str(detail or "compact").strip().lower()
    detail_mode = requested_detail if requested_detail in {"compact", "standard", "full"} else "compact"
    tools = []
    categories: Dict[str, List[str]] = {}
    seen: set[str] = set()
    for name in sorted(_TOOL_METADATA_REGISTRY):
        entry = _TOOL_METADATA_REGISTRY[name]
        func = entry.function
        if func is _REGISTRY_UNSET:
            continue
        seen.add(name)
        category = _tool_catalog_category(name, func)
        categories.setdefault(category, []).append(name)
        row: Dict[str, Any] = {
            "name": name,
            "category": category,
            "description": _tool_catalog_description(func),
        }
        related = related_tools_for(name)
        if related:
            row["related_tools"] = related
        if name == "market_depth_fetch":
            row.update(_market_depth_fetch_catalog_state())
        if detail_mode in {"standard", "full"}:
            row["parameters"] = _tool_catalog_parameters(func)
        if detail_mode == "full":
            row["module"] = str(getattr(func, "__module__", "") or "")
        tools.append(row)
    if "market_depth_fetch" not in seen:
        row = _market_depth_fetch_catalog_row(detail_mode=detail_mode)
        tools.append(row)
        categories.setdefault("market", []).append("market_depth_fetch")
    return {
        "success": True,
        "detail": detail_mode,
        "count": len(tools),
        "categories": categories,
        "output_extras": {
            "accepted": sorted(OUTPUT_EXTRAS),
            "full_aliases": sorted(OUTPUT_EXTRA_FULL_ALIASES),
            "support": "best_effort_by_tool",
        },
        "tools": tools,
    }


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

        if "|" in cleaned:
            parts = [p.strip() for p in cleaned.split("|") if p.strip()]
            if any(p in ("None", "NoneType") for p in parts):
                non_none = [p for p in parts if p not in ("None", "NoneType")]
                if len(non_none) == 1:
                    mapped = scalar_map.get(non_none[0])
                    if mapped is not None:
                        return mapped, True

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
    parsed = parse_bool_like(value, allow_none=allow_none)
    if parsed is UNPARSED_BOOL:
        raise ValueError(f"Invalid value for '{name}': expected boolean, got {value!r}")
    return parsed


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
        coerced = coerce_scalar(s)
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
        coerced = coerce_scalar(s)
        if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
            out = float(coerced)
            if not math.isfinite(out):
                raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")
            return out
    raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")


def _get_pydantic_model_fields(model_type: Any) -> tuple[Dict[str, Any], bool]:
    if not isinstance(model_type, type):
        return {}, False
    try:
        if not issubclass(model_type, BaseModel):
            return {}, False
    except TypeError:
        return {}, False

    model_fields = getattr(model_type, "model_fields", None)
    if isinstance(model_fields, dict):
        return model_fields, True

    return {}, False


def _coerce_kwargs_for_callable(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce common scalar string inputs (from MCP clients) based on annotations."""
    try:
        sig = get_runtime_signature(func)
    except Exception:
        return kwargs
    for param_name, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect._empty or param_name in kwargs:
            continue
        base_ann, allow_none = _unwrap_optional_annotation(ann)
        if not (isinstance(base_ann, type) and issubclass(base_ann, BaseModel)):
            continue
        try:
            model_fields, _ = _get_pydantic_model_fields(base_ann)
            field_names = set(model_fields.keys())
        except Exception:
            field_names = set()
        if not field_names:
            continue
        payload = {key: kwargs.pop(key) for key in list(kwargs.keys()) if key in field_names}
        if not payload and allow_none:
            continue
        if not payload and param.default is not inspect._empty:
            continue
        model_validate = getattr(base_ann, "model_validate", None)
        if callable(model_validate):
            kwargs[param_name] = model_validate(payload)
        else:
            kwargs[param_name] = base_ann.parse_obj(payload)
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
        elif isinstance(base_ann, type) and issubclass(base_ann, BaseModel):
            value = kwargs.get(param_name)
            if value is None and allow_none:
                continue
            if isinstance(value, base_ann):
                continue
            if isinstance(value, dict):
                model_validate = getattr(base_ann, "model_validate", None)
                if callable(model_validate):
                    kwargs[param_name] = model_validate(value)
                else:
                    kwargs[param_name] = base_ann.parse_obj(value)
    return kwargs


def _request_model_signature_fields(func: Any) -> List[inspect.Parameter]:
    """Flatten a single request-model parameter into top-level keyword params."""
    try:
        sig = get_runtime_signature(func)
    except Exception:
        return []

    params = list(sig.parameters.values())
    if len(params) != 1:
        return []

    request_param = params[0]
    base_ann, _ = _unwrap_optional_annotation(request_param.annotation)
    if not (isinstance(base_ann, type) and issubclass(base_ann, BaseModel)):
        return []

    model_fields, modern_fields = _get_pydantic_model_fields(base_ann)
    if model_fields and modern_fields:
        flattened: List[inspect.Parameter] = []
        for field_name, field in model_fields.items():
            annotation = inspect._empty
            rebuild_annotation = getattr(field, "rebuild_annotation", None)
            if callable(rebuild_annotation):
                try:
                    annotation = rebuild_annotation()
                except Exception:
                    annotation = inspect._empty
            if annotation is inspect._empty:
                annotation = getattr(field, "annotation", inspect._empty)
            is_required = bool(getattr(field, "is_required", lambda: False)())
            default = inspect._empty if is_required else _signature_default_for_model_field(field)
            flattened.append(
                inspect.Parameter(
                    field_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
        return flattened

    return []


def _signature_default_for_model_field(field: Any) -> Any:
    factory = getattr(field, "default_factory", None)
    if callable(factory):
        try:
            return factory()
        except Exception:
            return None
    default = getattr(field, "default", inspect._empty)
    if default is inspect._empty:
        return None
    if type(default).__name__ == "PydanticUndefinedType":
        return None
    return default


def _normalize_exposed_annotation(annotation: Any) -> Any:
    """Keep rich typing metadata for FastMCP schema generation when possible."""
    if annotation is inspect._empty:
        return object
    # Unresolved string annotations are safer to downcast than to expose
    # directly to FastMCP/Pydantic.
    if isinstance(annotation, str):
        return object
    return annotation


def _append_public_output_params(params: List[inspect.Parameter]) -> List[inspect.Parameter]:
    names = {param.name for param in params}
    out = list(params)
    if "json" not in names:
        out.append(
            inspect.Parameter(
                "json",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool,
            )
        )
    if "extras" not in names:
        out.append(
            inspect.Parameter(
                "extras",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Union[str, List[str], None],
            )
        )
    if "fields" not in names:
        out.append(
            inspect.Parameter(
                "fields",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Union[str, List[str], None],
            )
        )
    return out


_FIELD_SELECTION_META_KEYS = frozenset(
    {
        "success",
        "error",
        "error_code",
        "request_id",
        "symbol",
        "symbols",
        "timeframe",
        "detail",
        "count",
        "total",
        "truncated",
    }
)


def _normalize_output_fields(value: Any) -> tuple[str, ...]:
    if value in (None, False, ""):
        return ()
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_items = list(value)
    else:
        raw_items = [value]
    fields: list[str] = []
    for item in raw_items:
        field = str(item or "").strip()
        if field and field not in fields:
            fields.append(field)
    return tuple(fields)


def _filter_output_fields(
    value: Any,
    wanted: set[str],
    *,
    preserve_meta: bool,
) -> tuple[Any, bool]:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        matched = False
        for key, subvalue in value.items():
            field = str(key)
            if field in wanted:
                out[key] = subvalue
                matched = True
                continue
            if preserve_meta and field in _FIELD_SELECTION_META_KEYS:
                out[key] = subvalue
                continue
            filtered, submatched = _filter_output_fields(
                subvalue,
                wanted,
                preserve_meta=False,
            )
            if submatched:
                out[key] = filtered
                matched = True
        return out, matched
    if isinstance(value, list):
        out_items = []
        matched = False
        for item in value:
            filtered, submatched = _filter_output_fields(
                item,
                wanted,
                preserve_meta=False,
            )
            if submatched:
                out_items.append(filtered)
                matched = True
        return out_items, matched
    if isinstance(value, tuple):
        filtered_items = []
        matched = False
        for item in value:
            filtered, submatched = _filter_output_fields(
                item,
                wanted,
                preserve_meta=False,
            )
            if submatched:
                filtered_items.append(filtered)
                matched = True
        return tuple(filtered_items), matched
    return value, False


def _select_output_fields(value: Any, fields: Any) -> Any:
    requested = _normalize_output_fields(fields)
    if not requested or not isinstance(value, dict):
        return value
    wanted = set(requested)
    filtered, matched = _filter_output_fields(
        value,
        wanted,
        preserve_meta=True,
    )
    if matched:
        return filtered
    if value.get("error"):
        return value
    return {
        "success": False,
        "error_code": "invalid_output_fields",
        "error": "No requested output fields were present.",
        "requested_fields": list(requested),
        "available_fields": sorted(str(key) for key in value),
    }


def _callable_accepts_kwarg(func: Any, name: str) -> bool:
    try:
        sig = get_runtime_signature(func)
    except Exception:
        return False

    if name in sig.parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def _callable_exposes_kwarg(func: Any, name: str) -> bool:
    if _callable_accepts_kwarg(func, name):
        return True
    return any(param.name == name for param in _request_model_signature_fields(func))


def _recording_tool_decorator(*dargs, **dkwargs):  # type: ignore[override]  # noqa: C901
    if _ORIG_TOOL_DECORATOR is None:
        def _noop(func):
            _upsert_tool_registration(getattr(func, "__name__", "tool"), function=func)
            return func

        return _noop
    kwargs = dict(dkwargs)
    structured_in_args = len(dargs) >= 5
    if not structured_in_args and "structured_output" not in kwargs:
        kwargs["structured_output"] = False
    dec = _ORIG_TOOL_DECORATOR(*dargs, **kwargs)

    def _sanitize_annotations(func):
        flattened_params = _request_model_signature_fields(func)
        if flattened_params:
            cleaned = {
                param.name: (
                    _normalize_exposed_annotation(param.annotation)
                )
                for param in flattened_params
            }
            ann = get_runtime_annotations(func)
            if "return" in ann:
                cleaned["return"] = _normalize_exposed_annotation(ann["return"])
            return cleaned
        cleaned = {}
        ann = get_runtime_annotations(func)
        sig = get_runtime_signature(func)
        for name, param in sig.parameters.items():
            value = ann.get(name, param.annotation)
            cleaned[name] = _normalize_exposed_annotation(value)
        if "return" in ann:
            cleaned["return"] = _normalize_exposed_annotation(ann["return"])
        return cleaned

    def _wrap(func):  # noqa: C901
        from ..utils.minimal_output import format_result_minimal as _fmt_min
        from ..utils.minimal_output import (
            to_methods_availability_toon as _fmt_methods,
        )

        @_wraps(func)
        def _wrapped(*a, **kw):
            raw_output = kw.pop("__cli_raw", False)
            precision = kw.pop("precision", None)
            json_output = kw.pop("json", False)
            extras = kw.pop("extras", None)
            fields = kw.pop("fields", None)
            contract_state = resolve_output_contract({})

            try:
                normalized_extras = normalize_output_extras(extras)
                if normalized_extras and "detail" not in kw and _callable_exposes_kwarg(func, "detail"):
                    kw["detail"] = "full"
                _coerce_kwargs_for_callable(func, kw)
                contract_state = resolve_output_contract(
                    kw,
                    json=json_output,
                    extras=normalized_extras,
                )
                try:
                    if "denoise" in kw:
                        from ..utils.denoise import (
                            normalize_denoise_spec as _norm_dn,  # type: ignore
                        )

                        kw["denoise"] = _norm_dn(kw.get("denoise"))
                except Exception:
                    pass

                out = func(*a, **kw)
            except Exception as exc:
                request_id = None
                try:
                    request_id = build_error_payload(
                        str(exc),
                        code="tool_execution_error",
                        operation=getattr(func, "__name__", "tool"),
                        details={"tool": getattr(func, "__name__", "tool")},
                    )["request_id"]
                    log_transport_exception(
                        logging.getLogger(__name__),
                        transport="mcp",
                        operation=getattr(func, "__name__", "tool"),
                        request_id=request_id,
                        exc=exc,
                    )
                except Exception:
                    pass
                out = build_error_payload(
                    str(exc),
                    code="tool_execution_error",
                    request_id=request_id,
                    operation=getattr(func, "__name__", "tool"),
                    details={"tool": getattr(func, "__name__", "tool")},
                )

            if isinstance(out, dict):
                out = normalize_error_payload(
                    out,
                    default_code="tool_error",
                    operation=getattr(func, "__name__", "tool"),
                )

            if raw_output:
                return out

            fname = getattr(func, "__name__", "")
            public_out = out
            if isinstance(public_out, dict):
                if fname.strip().lower() == "news":
                    from .news import normalize_news_output

                    public_out = normalize_news_output(
                        public_out,
                        detail=contract_state.detail,
                    )
                if "guidance" in contract_state.extras:
                    public_out = attach_success_guidance(
                        public_out,
                        tool_name=fname,
                    )
                public_out = apply_output_verbosity(
                    public_out,
                    tool_name=fname,
                    detail=contract_state.shape_detail,
                )
                public_out = _select_output_fields(public_out, fields)

            if contract_state.json:
                return public_out

            try:
                if (
                    fname in ("forecast_list_methods", "denoise_list_methods")
                    and isinstance(public_out, dict)
                    and not contract_state.verbose
                ):
                    methods_list = public_out.get("methods") or []
                    if _fmt_methods and isinstance(methods_list, list):
                        s = _fmt_methods(cast(List[Dict[str, Any]], methods_list))
                        if s:
                            return s
                return _fmt_min(
                    public_out,
                    verbose=contract_state.verbose,
                    precision=precision,
                    tool_name=fname,
                )
            except Exception:
                return str(out) if out is not None else ""

        try:
            cleaned = _sanitize_annotations(func)
            _wrapped.__annotations__ = cleaned
            params = _request_model_signature_fields(func)
            if not params:
                sig = get_runtime_signature(func)
                for name, param in sig.parameters.items():
                    if param.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue
                    params.append(param.replace(annotation=cleaned.get(name)))
            params = _append_public_output_params(params)
            _wrapped.__annotations__ = cleaned
            return_ann = cleaned.get("return", inspect._empty)
            _wrapped.__signature__ = inspect.Signature(parameters=params, return_annotation=return_ann)
        except Exception as exc:
            logger.exception(
                "Failed to attach async MCP signature for tool %s: %s",
                getattr(func, "__name__", "tool"),
                exc,
            )

        # Register an async wrapper with FastMCP so sync tool execution does not
        # block the event loop while the underlying work runs in a worker thread.
        # A generous timeout prevents indefinite hangs when the underlying MT5
        # COM bridge deadlocks under concurrent access.
        @_wraps(func)
        async def _async_wrapped(*a, **kw):
            _tool_name = getattr(func, "__name__", "tool")
            if _tool_name in _NO_TIMEOUT_TOOLS:
                return await asyncio.to_thread(_wrapped, *a, **kw)
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_wrapped, *a, **kw),
                    timeout=_TOOL_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                return build_error_payload(
                    f"{_tool_name} timed out after {_TOOL_TIMEOUT_SECONDS}s",
                    code="tool_timeout",
                    operation=_tool_name,
                    details={"tool": _tool_name, "timeout_seconds": _TOOL_TIMEOUT_SECONDS},
                )

        try:
            _async_wrapped.__annotations__ = getattr(_wrapped, "__annotations__", {})
            _sig = getattr(_wrapped, "__signature__", None)
            if _sig is not None:
                _async_wrapped.__signature__ = _sig
        except Exception as exc:
            logger.exception(
                "Failed to attach MCP metadata for tool %s: %s",
                getattr(func, "__name__", "tool"),
                exc,
            )

        res = dec(_async_wrapped)
        name = getattr(func, "__name__", None)
        try:
            _wrapped._mcp_async_wrapper = _async_wrapped
            _wrapped._mcp_tool_object = res
        except Exception:
            pass
        if name:
            _upsert_tool_registration(name, function=_wrapped, tool_object=res)
        return _wrapped

    return _wrap


def install_tool_registry(mcp_obj: Any) -> None:
    """Install the wrapped tool decorator and registry attributes on an MCP instance."""
    global _ORIG_TOOL_DECORATOR
    if _ORIG_TOOL_DECORATOR is None:
        try:
            _ORIG_TOOL_DECORATOR = mcp_obj.tool  # type: ignore[attr-defined]
        except Exception:
            _ORIG_TOOL_DECORATOR = None
    try:
        mcp_obj.tool = _recording_tool_decorator
        mcp_obj.tools = _TOOL_REGISTRY
        mcp_obj.registry = _TOOL_REGISTRY
        mcp_obj._tools = _TOOL_REGISTRY
        mcp_obj._tool_registry = _TOOL_REGISTRY
    except Exception:
        pass


def unregister_tool(name: str, *, mcp_obj: Any = None) -> None:
    """Remove a tool from mtdata and FastMCP registries when a feature gate is off."""
    _remove_tool_registration_field(name, "function", default=None)
    _remove_tool_registration_field(name, "tool_object", default=None)
    if mcp_obj is None:
        return
    try:
        remove_tool = getattr(mcp_obj, "remove_tool", None)
        if callable(remove_tool):
            remove_tool(name)
            return
    except Exception:
        pass
    try:
        manager = getattr(mcp_obj, "_tool_manager", None)
        remove_tool = getattr(manager, "remove_tool", None)
        if callable(remove_tool):
            remove_tool(name)
    except Exception:
        pass


def get_tool_registry() -> Dict[str, Any]:
    tool_objects = _project_tool_registry("tool_object")
    if tool_objects:
        return tool_objects
    return _project_tool_registry("function")


def get_tool_functions() -> Dict[str, Any]:
    return _project_tool_registry("function")
