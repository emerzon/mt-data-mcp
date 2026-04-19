from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from .exceptions import ForecastError
from .forecast_registry import get_forecast_methods_data
from .interface import ForecastCapabilityDescriptor
from .registry import ForecastRegistry

_PRETRAINED_METHODS = {"chronos2", "chronos_bolt", "timesfm", "lag_llama"}


def _method_categories(method_data: Dict[str, Any]) -> Dict[str, str]:
    categories_raw = method_data.get("categories") if isinstance(method_data, dict) else None
    mapping: Dict[str, str] = {}
    if not isinstance(categories_raw, dict):
        return mapping
    for category, method_names in categories_raw.items():
        if not isinstance(method_names, list):
            continue
        for method_name in method_names:
            if method_name is None:
                continue
            mapping[str(method_name)] = str(category)
    return mapping


def _namespace_for_method(method_name: str, category: str) -> str:
    method_norm = str(method_name or "").strip()
    category_norm = str(category or "").strip().lower()
    if method_norm.startswith("sf_") or category_norm == "statsforecast":
        return "statsforecast"
    if method_norm.startswith("skt_") or category_norm == "sktime":
        return "sktime"
    if method_norm in _PRETRAINED_METHODS or category_norm == "pretrained":
        return "pretrained"
    if method_norm.startswith("mlf_") or category_norm in {"mlforecast", "ml", "machine_learning"}:
        return "mlforecast"
    return "native"


def _concept_for_method(method_name: str, namespace: str) -> str:
    method_norm = str(method_name or "").strip()
    if namespace == "statsforecast" and method_norm == "statsforecast":
        return "adapter"
    if namespace == "statsforecast" and method_norm.startswith("sf_"):
        return method_norm[3:]
    if namespace == "sktime" and method_norm == "sktime":
        return "adapter"
    if namespace == "sktime" and method_norm.startswith("skt_"):
        return method_norm[4:]
    if namespace == "mlforecast" and method_norm == "mlforecast":
        return "adapter"
    if namespace == "mlforecast" and method_norm.startswith("mlf_"):
        return method_norm[4:]
    return method_norm


def _resolve_method_capability_attr(raw_value: Any, method_name: str) -> Any:
    if isinstance(raw_value, dict):
        if method_name in raw_value:
            return raw_value[method_name]
        method_name_lower = method_name.lower()
        for key in (method_name_lower, "*", "default"):
            if key in raw_value:
                return raw_value[key]
        return None
    return raw_value


def _get_method_class_attrs(method_name: str) -> Dict[str, Any]:
    try:
        cls = ForecastRegistry.get_class(method_name)
    except Exception:
        return {}
    attrs = {
        "execution_library": _resolve_method_capability_attr(
            getattr(cls, "CAPABILITY_EXECUTION_LIBRARY", None),
            method_name,
        ),
        "adapter_method": _resolve_method_capability_attr(
            getattr(cls, "CAPABILITY_ADAPTER_METHOD", None),
            method_name,
        ),
        "selector_key": _resolve_method_capability_attr(
            getattr(cls, "CAPABILITY_SELECTOR_KEY", None),
            method_name,
        ),
        "selector_mode": _resolve_method_capability_attr(
            getattr(cls, "CAPABILITY_SELECTOR_MODE", None),
            method_name,
        ),
        "selector_value": _resolve_method_capability_attr(
            getattr(cls, "CAPABILITY_SELECTOR_VALUE", None),
            method_name,
        ),
        "concept": _resolve_method_capability_attr(
            getattr(cls, "CAPABILITY_CONCEPT", None),
            method_name,
        ),
        "display_name": _resolve_method_capability_attr(
            getattr(cls, "CAPABILITY_DISPLAY_NAME", None),
            method_name,
        ),
        "aliases": _resolve_method_capability_attr(
            getattr(cls, "CAPABILITY_ALIASES", ()) or (),
            method_name,
        ),
        "requires": _resolve_method_capability_attr(
            getattr(cls, "CAPABILITY_REQUIRES", None),
            method_name,
        ),
        "params": _resolve_method_capability_attr(
            getattr(cls, "CAPABILITY_PARAMS", None),
            method_name,
        ),
        "notes": _resolve_method_capability_attr(
            getattr(cls, "CAPABILITY_NOTES", None),
            method_name,
        ),
    }
    return attrs


def _registered_descriptor(method_name: str, row: Dict[str, Any], category: str) -> ForecastCapabilityDescriptor:
    attrs = _get_method_class_attrs(method_name)
    namespace = _namespace_for_method(method_name, category)
    concept = str(attrs.get("concept") or _concept_for_method(method_name, namespace))
    library = str(attrs.get("execution_library") or namespace)
    selector_key = attrs.get("selector_key")
    if attrs.get("adapter_method"):
        adapter_method = str(attrs["adapter_method"])
    elif selector_key and namespace in {"statsforecast", "sktime", "mlforecast"}:
        adapter_method = namespace
    else:
        adapter_method = namespace if concept == "adapter" else method_name
    selector_mode = str(attrs.get("selector_mode") or ("method" if adapter_method == method_name else "parameter"))
    selector_value = attrs.get("selector_value")
    display_name = str(attrs.get("display_name") or selector_value or method_name)
    requires = attrs.get("requires")
    if requires is None:
        requires = row.get("requires") or []
    params = attrs.get("params")
    if params is None:
        params = row.get("params") or []
    capability_id = f"{namespace}:{concept}"
    return ForecastCapabilityDescriptor(
        capability_id=capability_id,
        method=method_name,
        adapter_method=adapter_method,
        library=library,
        namespace=namespace,
        concept=concept,
        display_name=display_name,
        category=category or str(row.get("category") or "unknown"),
        description=str(row.get("description") or ""),
        available=bool(row.get("available", True)),
        requires=tuple(str(req) for req in requires if str(req).strip()),
        supports=dict(row.get("supports") or {}),
        params=tuple(dict(param) for param in params if isinstance(param, dict)),
        aliases=tuple(str(alias) for alias in attrs.get("aliases", ()) if str(alias).strip()),
        selector_key=str(selector_key) if selector_key else None,
        selector_value=str(selector_value) if selector_value is not None else None,
        selector_mode=selector_mode,
        source="registry",
        notes=attrs.get("notes"),
    )


def get_registered_capabilities() -> List[Dict[str, Any]]:
    method_data = get_forecast_methods_data()
    methods = method_data.get("methods") if isinstance(method_data, dict) else None
    if not isinstance(methods, list):
        return []
    categories = _method_categories(method_data)
    descriptors: List[Dict[str, Any]] = []
    for row in methods:
        if not isinstance(row, dict):
            continue
        method_name = str(row.get("method") or "").strip()
        if not method_name:
            continue
        category = str(row.get("category") or categories.get(method_name) or "other")
        descriptors.append(_registered_descriptor(method_name, row, category).to_record())
    return descriptors


def _normalize_concept(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum()) or str(name).lower()


def _statsforecast_capabilities() -> List[Dict[str, Any]]:
    try:
        try:
            models_mod = importlib.import_module("statsforecast.models")
        except Exception:
            statsforecast_mod = importlib.import_module("statsforecast")
            models_mod = getattr(statsforecast_mod, "models", None)
            if models_mod is None:
                return []
    except Exception:
        return []
    capabilities: List[Dict[str, Any]] = []
    for attr in dir(models_mod):
        if attr.startswith("_"):
            continue
        obj = getattr(models_mod, attr, None)
        if not isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != getattr(models_mod, "__name__", None):
            continue
        if not any(callable(getattr(obj, name, None)) for name in ("fit", "forecast", "predict")):
            continue
        descriptor = ForecastCapabilityDescriptor(
            capability_id=f"statsforecast:{_normalize_concept(attr)}",
            method=attr,
            adapter_method="statsforecast",
            library="statsforecast",
            namespace="statsforecast",
            concept=_normalize_concept(attr),
            display_name=attr,
            category="statsforecast",
            description=f"StatsForecast model {attr}.",
            available=True,
            supports={"price": True, "return": True, "volatility": True, "ci": True},
            selector_key="model_name",
            selector_value=attr,
            selector_mode="class_name",
            source="library_discovery",
        )
        capabilities.append(descriptor.to_record())
    return sorted(capabilities, key=lambda row: str(row.get("display_name")))


def _sktime_capabilities(
    *,
    discover_sktime_forecasters: Optional[Callable[[], Dict[str, Tuple[str, str]]]] = None,
) -> List[Dict[str, Any]]:
    discover = discover_sktime_forecasters
    if discover is None:
        from .use_cases import _discover_sktime_forecasters as discover
    try:
        mapping = discover() or {}
    except Exception:
        return []
    capabilities: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for class_name, dotted_path in sorted({value for value in mapping.values()}, key=lambda item: item[0]):
        capability_id = f"sktime:{_normalize_concept(class_name)}"
        if capability_id in seen:
            continue
        seen.add(capability_id)
        descriptor = ForecastCapabilityDescriptor(
            capability_id=capability_id,
            method=class_name,
            adapter_method="sktime",
            library="sktime",
            namespace="sktime",
            concept=_normalize_concept(class_name),
            display_name=class_name,
            category="sktime",
            description=f"sktime forecaster {class_name}.",
            available=True,
            supports={"price": True, "return": True, "volatility": True, "ci": True},
            aliases=(class_name,),
            selector_key="estimator",
            selector_value=dotted_path,
            selector_mode="dotted_path",
            source="library_discovery",
        )
        capabilities.append(descriptor.to_record())
    return capabilities


def _pretrained_capabilities() -> List[Dict[str, Any]]:
    rows = [row for row in get_registered_capabilities() if str(row.get("namespace")) == "pretrained"]
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("display_name") or row.get("method")),
            str(row.get("method")),
        ),
    )


def _native_capabilities() -> List[Dict[str, Any]]:
    rows = [row for row in get_registered_capabilities() if str(row.get("namespace")) == "native"]
    return sorted(rows, key=lambda row: str(row.get("method")))


def _mlforecast_capabilities() -> List[Dict[str, Any]]:
    rows = [row for row in get_registered_capabilities() if str(row.get("method")) == "mlforecast"]
    if rows:
        return rows
    descriptor = ForecastCapabilityDescriptor(
        capability_id="mlforecast:adapter",
        method="mlforecast",
        adapter_method="mlforecast",
        library="mlforecast",
        namespace="mlforecast",
        concept="adapter",
        display_name="mlforecast",
        category="machine_learning",
        description="Generic MLForecast adapter for dotted estimator classes.",
        available=True,
        supports={"price": True, "return": True, "volatility": True, "ci": False},
        selector_key="model",
        selector_mode="dotted_class",
        source="library_discovery",
    )
    return [descriptor.to_record()]


def get_library_capabilities(
    library: str,
    *,
    discover_sktime_forecasters: Optional[Callable[[], Dict[str, Tuple[str, str]]]] = None,
) -> List[Dict[str, Any]]:
    library_norm = str(library or "").strip().lower()
    if library_norm == "native":
        return _native_capabilities()
    if library_norm == "statsforecast":
        return _statsforecast_capabilities()
    if library_norm == "sktime":
        return _sktime_capabilities(discover_sktime_forecasters=discover_sktime_forecasters)
    if library_norm == "pretrained":
        return _pretrained_capabilities()
    if library_norm == "mlforecast":
        return _mlforecast_capabilities()
    return []


def resolve_capability_request(
    *,
    library: Optional[str],
    method: Optional[str],
    params: Optional[Dict[str, Any]] = None,
    discover_sktime_forecasters: Optional[Callable[[], Dict[str, Tuple[str, str]]]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    library_norm = str(library or "native").strip().lower() or "native"
    method_norm = str(method or "").strip()
    params_out = dict(params or {})
    if not method_norm:
        return library_norm, method_norm, params_out

    if ":" not in method_norm:
        if library_norm in {"statsforecast", "sktime", "mlforecast", "pretrained"}:
            capabilities = [
                row
                for row in get_registered_capabilities()
                if str(row.get("namespace") or "").strip().lower() == library_norm
            ]
            requested_method = method_norm.lower()
            match = next(
                (
                    row
                    for row in capabilities
                    if str(row.get("method") or "").strip().lower() == requested_method
                ),
                None,
            )
            if isinstance(match, dict):
                execution = match.get("execution") if isinstance(match.get("execution"), dict) else {}
                resolved_library = str(execution.get("library") or library_norm)
                resolved_method = str(
                    execution.get("method") or match.get("adapter_method") or method_norm
                )
                selector_params = (
                    execution.get("params") if isinstance(execution.get("params"), dict) else {}
                )
                merged_params = dict(params_out)
                merged_params.update(selector_params)
                return resolved_library, resolved_method, merged_params
        return library_norm, method_norm, params_out

    namespace = method_norm.split(":", 1)[0].strip().lower()
    if not namespace:
        return library_norm, method_norm, params_out

    capabilities: List[Dict[str, Any]] = []
    if namespace == "native":
        capabilities = get_registered_capabilities()
    else:
        capabilities = get_library_capabilities(
            namespace,
            discover_sktime_forecasters=discover_sktime_forecasters,
        )
        if namespace in {"statsforecast", "sktime", "mlforecast", "pretrained"}:
            capabilities = list(capabilities) + [
                row
                for row in get_registered_capabilities()
                if str(row.get("namespace")) == namespace
            ]

    requested_id = method_norm.lower()
    match = next(
        (
            row
            for row in capabilities
            if str(row.get("capability_id", "")).lower() == requested_id
        ),
        None,
    )
    if not isinstance(match, dict):
        raise ForecastError(f"Unknown forecast capability '{method_norm}'")

    execution = match.get("execution") if isinstance(match.get("execution"), dict) else {}
    resolved_library = str(execution.get("library") or namespace or library_norm)
    resolved_method = str(execution.get("method") or match.get("adapter_method") or method_norm)
    selector_params = execution.get("params") if isinstance(execution.get("params"), dict) else {}
    merged_params = dict(params_out)
    merged_params.update(selector_params)
    return resolved_library, resolved_method, merged_params
