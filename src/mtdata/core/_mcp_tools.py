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
from ..shared.tool_categories import tool_catalog_category
from ..utils.coercion import UNPARSED_BOOL, coerce_scalar, parse_bool_like
from .error_envelope import (
    build_error_payload,
    log_transport_exception,
    normalize_error_payload,
)
from .output_contract import (
    OutputContractState,
    apply_output_verbosity,
    attach_success_guidance,
    resolve_output_contract,
)
from .request_context import ensure_request_id_scope

_ORIG_TOOL_DECORATOR: Any = None
_REGISTRY_UNSET = object()
_PUBLIC_OUTPUT_PARAMS = PUBLIC_OUTPUT_PARAMS
_MARKET_DEPTH_FETCH_ENV = "MTDATA_ENABLE_MARKET_DEPTH_FETCH"
_TOOL_CATALOG_SCHEMA_VERSION = "1.0"
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
    return tool_catalog_category(name, module=module)


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


def _tool_catalog_input_schema(name: str, func: Any) -> Dict[str, Any]:
    from .schema_attach import get_public_tool_schema

    schema = get_public_tool_schema(name)
    if schema:
        return schema

    # Gated tools are intentionally absent from the public MCP registry. Build
    # their discoverability schema from the callable when they appear in the
    # catalog as disabled rows.
    from ..shared.schema import build_minimal_schema, get_function_info

    fallback = build_minimal_schema(get_function_info(func))
    parameters = fallback.get("parameters")
    return parameters if isinstance(parameters, dict) else fallback


def _tool_catalog_schema_value_format(
    property_schema: Dict[str, Any],
    *,
    definitions: Dict[str, Any],
) -> str:
    ref = property_schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        resolved = definitions.get(ref.rsplit("/", 1)[-1])
        if isinstance(resolved, dict):
            return _tool_catalog_schema_value_format(
                resolved,
                definitions=definitions,
            )
    for branch_key in ("anyOf", "oneOf"):
        branches = property_schema.get(branch_key)
        if isinstance(branches, list):
            formats = {
                _tool_catalog_schema_value_format(branch, definitions=definitions)
                for branch in branches
                if isinstance(branch, dict) and branch.get("type") != "null"
            }
            if len(formats) == 1:
                return formats.pop()
    schema_type = property_schema.get("type")
    if schema_type == "object" or isinstance(property_schema.get("properties"), dict):
        return "json_object"
    if schema_type == "array":
        return "repeatable_values"
    if schema_type == "boolean":
        return "boolean"
    return "scalar"


def _tool_catalog_cli_binding(
    tool_name: str,
    parameter_name: str,
    *,
    index: int,
    required: bool,
    property_schema: Dict[str, Any],
    definitions: Dict[str, Any],
) -> Dict[str, Any]:
    from .cli.parsing.discovery import (
        _NAMED_ONLY_REQUIRED_PARAMS,
        _OPTIONAL_POSITIONAL_PARAMS,
        should_expose_cli_param,
    )

    key = (tool_name, parameter_name)
    exposed = should_expose_cli_param(
        cmd_name=tool_name,
        param_name=parameter_name,
    )
    first_required = required and index == 0 and key not in _NAMED_ONLY_REQUIRED_PARAMS
    symbol_alias = first_required and parameter_name in {"symbol", "symbols"}
    positional = first_required or key in _OPTIONAL_POSITIONAL_PARAMS
    option = exposed and (not first_required or symbol_alias)
    forms: List[Dict[str, str]] = []
    if positional:
        forms.append(
            {
                "kind": "positional",
                "token": parameter_name.upper(),
            }
        )
    if option:
        forms.append(
            {
                "kind": "option",
                "token": f"--{parameter_name.replace('_', '-')}",
            }
        )
    binding: Dict[str, Any] = {
        "available": exposed,
        "forms": forms,
        "value_format": _tool_catalog_schema_value_format(
            property_schema,
            definitions=definitions,
        ),
    }
    if (
        binding["value_format"] == "boolean"
        and option
        and parameter_name != "json"
    ):
        binding["negated_option"] = f"--no-{parameter_name.replace('_', '-')}"
    return binding


def _tool_catalog_full_parameters(
    tool_name: str,
    input_schema: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    definitions = input_schema.get("$defs")
    if not isinstance(definitions, dict):
        definitions = {}
    required_names = {
        str(item) for item in input_schema.get("required", []) if item is not None
    }
    parameters: Dict[str, Dict[str, Any]] = {}
    for index, (name, raw_schema) in enumerate(properties.items()):
        property_schema = dict(raw_schema) if isinstance(raw_schema, dict) else {}
        is_required = str(name) in required_names
        property_schema["required"] = is_required
        if not is_required and "default" not in property_schema:
            property_schema["default"] = None
        description = str(property_schema.get("description") or "").strip()
        if not description or description.startswith("Value for "):
            from .cli.parsing.discovery import _COMMAND_PARAM_HELP_OVERRIDES

            description = _COMMAND_PARAM_HELP_OVERRIDES.get(
                (str(tool_name), str(name)),
                f"Input parameter '{name}' for {tool_name}.",
            )
            property_schema["description"] = description
        property_schema["cli"] = _tool_catalog_cli_binding(
            tool_name,
            str(name),
            index=index,
            required=is_required,
            property_schema=property_schema,
            definitions=definitions,
        )
        parameters[str(name)] = property_schema
    return parameters


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
    if detail_mode == "standard":
        row["parameters"] = {
            "symbol": "required",
            "spread": "optional",
            "require_dom": "optional",
        }
    if detail_mode == "full":
        from .market_depth import market_depth_fetch

        input_schema = _tool_catalog_input_schema(
            "market_depth_fetch",
            market_depth_fetch,
        )
        row["schema_version"] = _TOOL_CATALOG_SCHEMA_VERSION
        row["input_schema"] = input_schema
        row["parameters"] = _tool_catalog_full_parameters(
            "market_depth_fetch",
            input_schema,
        )
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
        if detail_mode == "standard":
            row["parameters"] = _tool_catalog_parameters(func)
        if detail_mode == "full":
            input_schema = _tool_catalog_input_schema(name, func)
            row["schema_version"] = _TOOL_CATALOG_SCHEMA_VERSION
            row["input_schema"] = input_schema
            row["parameters"] = _tool_catalog_full_parameters(
                name,
                input_schema,
            )
            row["module"] = str(getattr(func, "__module__", "") or "")
        tools.append(row)
    if "market_depth_fetch" not in seen:
        row = _market_depth_fetch_catalog_row(detail_mode=detail_mode)
        tools.append(row)
        categories.setdefault("market", []).append("market_depth_fetch")
    return {
        "success": True,
        "schema_version": _TOOL_CATALOG_SCHEMA_VERSION,
        "parameter_schema": {
            "available_in_detail": "full",
            "format": "JSON Schema Draft 2020-12 with CLI bindings",
        },
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
    if "output_fields" not in names:
        out.append(
            inspect.Parameter(
                "output_fields",
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
        "pagination",
        "warnings",
        "history_window_truncated",
        "history_window_limit_days",
        "history_window_floor",
        "effective_start",
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
            if field == "units":
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


def _filter_output_path(value: Any, path: tuple[str, ...]) -> tuple[Any, bool]:
    if not path:
        return value, True
    if isinstance(value, dict):
        key = path[0]
        if key not in value:
            return {}, False
        filtered, matched = _filter_output_path(value[key], path[1:])
        return ({key: filtered}, True) if matched else ({}, False)
    if isinstance(value, list):
        items = []
        for item in value:
            filtered, matched = _filter_output_path(item, path)
            if matched:
                items.append(filtered)
        return items, bool(items)
    if isinstance(value, tuple):
        items = []
        for item in value:
            filtered, matched = _filter_output_path(item, path)
            if matched:
                items.append(filtered)
        return tuple(items), bool(items)
    return value, False


def _merge_output_field_selection(left: Any, right: Any) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        out = dict(left)
        for key, value in right.items():
            out[key] = (
                _merge_output_field_selection(out[key], value)
                if key in out
                else value
            )
        return out
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            _merge_output_field_selection(a, b)
            for a, b in zip(left, right)
        ]
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return tuple(
            _merge_output_field_selection(a, b)
            for a, b in zip(left, right)
        )
    return right


def _row_collection_names(value: Dict[str, Any]) -> list[str]:
    names: list[str] = []
    row_key = value.get("row_key")
    if isinstance(row_key, str) and isinstance(value.get(row_key), list):
        names.append(row_key)
    for name in ("row_keys",):
        extra = value.get(name)
        if isinstance(extra, list):
            for item in extra:
                if isinstance(item, str) and item not in names and isinstance(
                    value.get(item), list
                ):
                    names.append(item)
    for name in ("data", "items", "deals", "orders"):
        if name not in names and isinstance(value.get(name), list):
            names.append(name)
    return names


def _project_row_collection_field(
    value: Dict[str, Any],
    field: str,
) -> tuple[Dict[str, Any], bool]:
    for name in _row_collection_names(value):
        rows = value.get(name)
        if not isinstance(rows, list):
            continue
        if not any(isinstance(row, dict) and field in row for row in rows):
            continue
        projected = []
        for row in rows:
            if isinstance(row, dict) and field in row:
                projected.append({field: row[field]})
            elif isinstance(row, dict):
                projected.append({})
            else:
                projected.append(row)
        return {name: projected}, True
    return {}, False


def _project_forecast_alias_field(
    value: Dict[str, Any],
    field: str,
) -> tuple[Dict[str, Any], bool]:
    """Resolve canonical forecast arrays from compact forecast rows."""
    rows = value.get("forecast")
    if not isinstance(rows, list) or not rows:
        return {}, False
    aliases = {
        "forecast_time": ("time",),
        "forecast_return": ("return",),
        "lower_price": ("lower_price",),
        "upper_price": ("upper_price",),
        "lower_return": ("lower_return",),
        "upper_return": ("upper_return",),
    }
    candidates = aliases.get(field)
    if field == "forecast_price":
        quantity = str(value.get("quantity") or "").strip().lower()
        if quantity == "volatility":
            return {}, False
        candidates = ("price",) if quantity == "return" else ("price", "value")
    if candidates is None:
        return {}, False
    projected: list[Any] = []
    matched = False
    for row in rows:
        if not isinstance(row, dict):
            projected.append(None)
            continue
        row_value = None
        for candidate in candidates:
            if candidate in row:
                row_value = row[candidate]
                matched = True
                break
        projected.append(row_value)
    return ({field: projected}, True) if matched else ({}, False)


def _select_output_fields(value: Any, fields: Any) -> Any:
    requested = _normalize_output_fields(fields)
    if not requested or not isinstance(value, dict):
        return value
    selected = {
        key: subvalue
        for key, subvalue in value.items()
        if key in _FIELD_SELECTION_META_KEYS
    }
    unresolved: list[str] = []
    for requested_field in requested:
        if "." in requested_field:
            filtered, matched = _filter_output_path(
                value,
                tuple(part for part in requested_field.split(".") if part),
            )
        elif requested_field in value:
            filtered, matched = {requested_field: value[requested_field]}, True
        else:
            filtered, matched = _project_forecast_alias_field(
                value,
                requested_field,
            )
            if not matched:
                filtered, matched = _project_row_collection_field(
                    value,
                    requested_field,
                )
            if not matched:
                filtered, matched = {}, requested_field in {
                    "error",
                    "error_code",
                    "remediation",
                    "documentation",
                }
        if not matched:
            unresolved.append(requested_field)
            continue
        selected = _merge_output_field_selection(selected, filtered)
    # Optional error-envelope fields may be absent on success. Other missing
    # paths are surfaced so projection typos cannot silently discard data.
    if unresolved:
        selected["unresolved_output_fields"] = unresolved
        selected["valid_output_fields"] = sorted(
            str(key)
            for key in value
            if key not in _FIELD_SELECTION_META_KEYS
        )
    return selected


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


def _update_supplied_request_model_field(
    func: Any,
    kwargs: Dict[str, Any],
    name: str,
    value: Any,
) -> bool:
    """Update a flattened field when the caller supplied its request model."""
    try:
        sig = get_runtime_signature(func)
    except Exception:
        return False
    for param_name, param in sig.parameters.items():
        if param_name not in kwargs:
            continue
        base_ann, _ = _unwrap_optional_annotation(param.annotation)
        model_fields, _ = _get_pydantic_model_fields(base_ann)
        if name not in model_fields:
            continue
        request = kwargs[param_name]
        if isinstance(request, BaseModel):
            kwargs[param_name] = request.model_copy(update={name: value})
            return True
        if isinstance(request, dict):
            kwargs[param_name] = {**request, name: value}
            return True
    return False


def _prepare_public_tool_call(
    func: Any,
    kwargs: Dict[str, Any],
    *,
    json_output: Any = False,
) -> OutputContractState:
    """Apply shared public output arguments before invoking a raw tool callable."""
    explicit_detail = kwargs.get("detail", _REGISTRY_UNSET)
    if (
        explicit_detail is not _REGISTRY_UNSET
        and not _callable_accepts_kwarg(func, "detail")
        and _callable_exposes_kwarg(func, "detail")
    ):
        detail_value = kwargs.pop("detail")
        if not _update_supplied_request_model_field(func, kwargs, "detail", detail_value):
            kwargs["detail"] = detail_value
    _coerce_kwargs_for_callable(func, kwargs)
    contract_source: Any = kwargs
    for value in kwargs.values():
        if isinstance(value, BaseModel) and hasattr(value, "detail"):
            contract_source = value
            break
    contract_kwargs: Dict[str, Any] = {"json": json_output}
    if explicit_detail is not _REGISTRY_UNSET:
        contract_kwargs["detail"] = explicit_detail
    return resolve_output_contract(contract_source, **contract_kwargs)


def _shape_public_tool_output(
    result: Any,
    *,
    tool_name: str,
    contract_state: OutputContractState,
    output_fields: Any = None,
) -> Any:
    """Apply shared structured-output shaping used by public transports."""
    if not isinstance(result, dict):
        return result
    public_out = result
    if tool_name.strip().lower() == "news":
        from .news import normalize_news_output

        public_out = normalize_news_output(
            public_out,
            detail=contract_state.detail,
        )
    if contract_state.detail == "full":
        public_out = attach_success_guidance(public_out, tool_name=tool_name)
    public_out = apply_output_verbosity(
        public_out,
        tool_name=tool_name,
        detail=contract_state.shape_detail,
    )
    return _select_output_fields(public_out, output_fields)


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

        def _invoke_wrapped(*a, **kw):
            raw_output = kw.pop("__cli_raw", False)
            precision = kw.pop("precision", None)
            json_output = kw.pop("json", False)
            output_fields = kw.pop("output_fields", None)
            # Resolve the requested representation before any fallible argument
            # normalization so wrapper-generated errors keep the same contract.
            contract_state = resolve_output_contract({}, json=json_output)

            try:
                contract_state = _prepare_public_tool_call(
                    func,
                    kw,
                    json_output=json_output,
                )
                if "denoise" in kw:
                    from ..utils.denoise import (
                        normalize_denoise_spec as _norm_dn,  # type: ignore
                    )

                    kw["denoise"] = _norm_dn(kw.get("denoise"))

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

            if raw_output and isinstance(out, dict) and contract_state.detail == "full":
                out = attach_success_guidance(
                    out,
                    tool_name=getattr(func, "__name__", ""),
                )
            if raw_output:
                return out

            fname = getattr(func, "__name__", "")
            public_out = _shape_public_tool_output(
                out,
                tool_name=fname,
                contract_state=contract_state,
                output_fields=output_fields,
            )

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

        @_wraps(func)
        def _wrapped(*a, **kw):
            with ensure_request_id_scope():
                return _invoke_wrapped(*a, **kw)

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
        # Keep the transport attached until completion: Python cannot safely
        # cancel a running worker thread, so returning a timeout would falsely
        # imply that broker or analysis work had stopped.
        @_wraps(func)
        async def _async_wrapped(*a, **kw):
            worker = asyncio.create_task(asyncio.to_thread(_wrapped, *a, **kw))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # The thread cannot be stopped safely. Keep this handler
                # attached until the operation reaches a terminal state so a
                # cancellation acknowledgement cannot imply that a mutating
                # broker call was aborted.
                await worker
                raise

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
