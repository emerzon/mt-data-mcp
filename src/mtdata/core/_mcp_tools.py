"""Shared FastMCP tool wrapping and registry helpers."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import types
from functools import wraps as _wraps
from typing import Any, Dict, List, Union, cast, get_args, get_origin

from pydantic import BaseModel

from .error_envelope import build_error_payload, log_transport_exception, normalize_error_payload
from ..utils.utils import _coerce_scalar

try:
    import annotationlib
except Exception:  # pragma: no cover - Python 3.14+ should provide this
    annotationlib = None

_ORIG_TOOL_DECORATOR: Any = None
_TOOL_REGISTRY: Dict[str, Any] = {}
_TOOL_OBJECT_REGISTRY: Dict[str, Any] = {}
_ANNOTATION_VALUE_FORMAT = getattr(getattr(annotationlib, "Format", None), "VALUE", None)


def _get_runtime_signature(obj: Any) -> inspect.Signature:
    """Resolve a signature with evaluated annotations when available."""
    if _ANNOTATION_VALUE_FORMAT is not None:
        try:
            return inspect.signature(obj, eval_str=True, annotation_format=_ANNOTATION_VALUE_FORMAT)
        except Exception:
            pass
    return inspect.signature(obj)


def _get_runtime_annotations(obj: Any) -> Dict[str, Any]:
    """Resolve runtime annotations using the 3.14 annotation API when available."""
    if annotationlib is not None and _ANNOTATION_VALUE_FORMAT is not None:
        try:
            resolved = annotationlib.get_annotations(obj, eval_str=True, format=_ANNOTATION_VALUE_FORMAT)
            if isinstance(resolved, dict):
                return resolved
        except Exception:
            pass
    raw = getattr(obj, "__annotations__", None)
    return raw if isinstance(raw, dict) else {}


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
        sig = _get_runtime_signature(func)
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
            model_fields = getattr(base_ann, "model_fields", None)
            if isinstance(model_fields, dict):
                field_names = set(model_fields.keys())
            else:
                legacy_fields = getattr(base_ann, "__fields__", None)
                field_names = set(legacy_fields.keys()) if isinstance(legacy_fields, dict) else set()
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
        sig = _get_runtime_signature(func)
    except Exception:
        return []

    params = list(sig.parameters.values())
    if len(params) != 1:
        return []

    request_param = params[0]
    base_ann, _ = _unwrap_optional_annotation(request_param.annotation)
    if not (isinstance(base_ann, type) and issubclass(base_ann, BaseModel)):
        return []

    model_fields = getattr(base_ann, "model_fields", None)
    if isinstance(model_fields, dict):
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

    legacy_fields = getattr(base_ann, "__fields__", None)
    if isinstance(legacy_fields, dict):
        flattened = []
        for field_name, field in legacy_fields.items():
            annotation = getattr(field, "outer_type_", getattr(field, "type_", inspect._empty))
            is_required = bool(getattr(field, "required", False))
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


def _recording_tool_decorator(*dargs, **dkwargs):  # type: ignore[override]
    if _ORIG_TOOL_DECORATOR is None:
        def _noop(func):
            _TOOL_REGISTRY[getattr(func, "__name__", "tool")] = func
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
                    param.annotation
                    if param.annotation is not inspect._empty
                    else object
                )
                for param in flattened_params
            }
            ann = _get_runtime_annotations(func)
            if "return" in ann:
                cleaned["return"] = ann["return"] if ann["return"] is not inspect._empty else object
            return cleaned
        cleaned = {}
        ann = _get_runtime_annotations(func)
        sig = _get_runtime_signature(func)
        for name, param in sig.parameters.items():
            value = ann.get(name, param.annotation)
            cleaned[name] = value if isinstance(value, type) else object
        if "return" in ann:
            cleaned["return"] = ann["return"] if isinstance(ann["return"], type) else object
        return cleaned

    def _wrap(func):
        try:
            from ..utils.minimal_output import format_result_minimal as _fmt_min, to_methods_availability_toon as _fmt_methods
        except Exception:
            _fmt_min = lambda x, **_: str(x) if x is not None else ""
            _fmt_methods = None

        @_wraps(func)
        def _wrapped(*a, **kw):
            raw_output = kw.pop("__cli_raw", False)

            try:
                _coerce_kwargs_for_callable(func, kw)
                try:
                    if "denoise" in kw:
                        from ..utils.denoise import normalize_denoise_spec as _norm_dn  # type: ignore

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

            try:
                fname = getattr(func, "__name__", "")
                if fname in ("forecast_list_methods", "denoise_list_methods") and isinstance(out, dict):
                    methods_list = out.get("methods") or []
                    if _fmt_methods and isinstance(methods_list, list):
                        s = _fmt_methods(cast(List[Dict[str, Any]], methods_list))
                        if s:
                            return s
                simplify_numbers = not str(fname).startswith("trade_")
                return _fmt_min(
                    out,
                    verbose=False,
                    simplify_numbers=simplify_numbers,
                    tool_name=fname,
                )
            except Exception:
                return str(out) if out is not None else ""

        try:
            cleaned = _sanitize_annotations(func)
            _wrapped.__annotations__ = cleaned
            params = _request_model_signature_fields(func)
            if not params:
                sig = _get_runtime_signature(func)
                for name, param in sig.parameters.items():
                    params.append(param.replace(annotation=cleaned.get(name)))
            return_ann = cleaned.get("return", inspect._empty)
            setattr(_wrapped, "__signature__", inspect.Signature(parameters=params, return_annotation=return_ann))
        except Exception:
            pass

        # Register an async wrapper with FastMCP so sync tool execution does not
        # block the event loop while the underlying work runs in a worker thread.
        @_wraps(func)
        async def _async_wrapped(*a, **kw):
            return await asyncio.to_thread(_wrapped, *a, **kw)

        try:
            _async_wrapped.__annotations__ = getattr(_wrapped, "__annotations__", {})
            _sig = getattr(_wrapped, "__signature__", None)
            if _sig is not None:
                _async_wrapped.__signature__ = _sig
        except Exception:
            pass

        res = dec(_async_wrapped)
        name = getattr(func, "__name__", None)
        if name:
            _TOOL_REGISTRY[str(name)] = _wrapped
            try:
                _TOOL_OBJECT_REGISTRY[str(name)] = res
            except Exception:
                pass
        return res

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
        setattr(mcp_obj, "tool", _recording_tool_decorator)
        setattr(mcp_obj, "tools", _TOOL_REGISTRY)
        setattr(mcp_obj, "registry", _TOOL_REGISTRY)
        setattr(mcp_obj, "_tools", _TOOL_REGISTRY)
        setattr(mcp_obj, "_tool_registry", _TOOL_REGISTRY)
    except Exception:
        pass


def get_tool_registry() -> Dict[str, Any]:
    if _TOOL_OBJECT_REGISTRY:
        return dict(_TOOL_OBJECT_REGISTRY)
    return dict(_TOOL_REGISTRY)


def get_tool_functions() -> Dict[str, Any]:
    return dict(_TOOL_REGISTRY)
