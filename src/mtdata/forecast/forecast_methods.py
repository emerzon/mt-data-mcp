"""
Forecast method definitions and metadata (registry-derived).
"""

from importlib import import_module
from typing import Any, Dict, List, Optional

from .forecast_registry import (
    DEFAULT_METHOD_SUPPORTS as _DEFAULT_METHOD_SUPPORTS,
)
from .forecast_registry import (
    get_forecast_methods_data as _registry_methods_data,
)


def _forecast_capabilities_module():
    return import_module("mtdata.forecast.capabilities")


def _get_registered_capabilities() -> List[Dict[str, Any]]:
    return _forecast_capabilities_module().get_registered_capabilities()


def get_forecast_methods_data() -> Dict[str, Any]:
    """Get comprehensive data about available forecast methods."""
    return _registry_methods_data()


def _build_method_category_lookup(method_data: Dict[str, Any]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    categories = method_data.get("categories") if isinstance(method_data.get("categories"), dict) else {}
    for category, methods in categories.items():
        if not isinstance(methods, list):
            continue
        for method in methods:
            if method in (None, ""):
                continue
            lookup[str(method)] = str(category)

    methods = method_data.get("methods")
    if isinstance(methods, list):
        for method_def in methods:
            if not isinstance(method_def, dict):
                continue
            method_name = str(method_def.get("method") or "").strip()
            if not method_name:
                continue
            category = str(method_def.get("category") or lookup.get(method_name) or "other")
            lookup[method_name] = category
    return lookup


def _enrich_method_definition(
    method_def: Dict[str, Any],
    *,
    capability_by_method: Dict[str, Dict[str, Any]],
    method_to_category: Dict[str, str],
) -> Dict[str, Any]:
    method_name = str(method_def.get("method") or "").strip()
    row = dict(method_def)
    if not method_name:
        return row

    category = str(row.get("category") or method_to_category.get(method_name) or "other")
    capability = capability_by_method.get(method_name, {})
    namespace = str(capability.get("namespace") or "native")
    concept = str(capability.get("concept") or method_name)
    adapter_method = str(capability.get("adapter_method") or method_name)
    method_id = str(capability.get("capability_id") or f"native:{method_name}")
    selector = capability.get("selector")
    execution = capability.get("execution")
    supports = capability.get("supports", row.get("supports"))
    aliases = capability.get("aliases")

    row["category"] = category
    row["namespace"] = namespace
    row["concept"] = concept
    row["method_id"] = method_id
    row["capability_id"] = str(capability.get("capability_id") or method_id)
    row["adapter_method"] = adapter_method
    row["selector"] = dict(selector) if isinstance(selector, dict) else {"mode": "method"}
    row["execution"] = (
        dict(execution)
        if isinstance(execution, dict)
        else {"library": namespace, "method": adapter_method}
    )
    if isinstance(supports, dict):
        row["supports"] = dict(supports)
        if isinstance(supports.get("ci"), bool):
            row["supports_ci"] = bool(supports.get("ci"))
    elif isinstance(row.get("supports_ci"), bool):
        row["supports_ci"] = bool(row.get("supports_ci"))
    row["display_name"] = str(capability.get("display_name") or method_name)
    row["aliases"] = (
        [str(alias) for alias in aliases if str(alias).strip()]
        if isinstance(aliases, (list, tuple, set))
        else []
    )
    row["source"] = str(capability.get("source") or "registry")
    return row


def _find_method_definition_in(
    method: str,
    methods: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for method_def in methods:
        if isinstance(method_def, dict) and method_def.get("method") == method:
            return method_def
    return None


def get_forecast_methods_snapshot(
    *,
    method_data: Optional[Dict[str, Any]] = None,
    capabilities: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return registry-backed methods enriched with capability metadata."""
    data = method_data if isinstance(method_data, dict) else get_forecast_methods_data()
    if not isinstance(data, dict):
        data = {}

    method_to_category = _build_method_category_lookup(data)
    methods = data.get("methods")
    methods_valid = isinstance(methods, list) and all(isinstance(item, dict) for item in methods)
    if not methods_valid:
        return {
            "data": data,
            "methods": [],
            "method_to_category": method_to_category,
            "methods_valid": False,
        }

    capability_rows = capabilities if isinstance(capabilities, list) else _get_registered_capabilities()
    capability_by_method = {
        str(row.get("method")): row
        for row in capability_rows
        if isinstance(row, dict) and row.get("method")
    }
    return {
        "data": data,
        "methods": [
            _enrich_method_definition(
                method_def,
                capability_by_method=capability_by_method,
                method_to_category=method_to_category,
            )
            for method_def in methods
        ],
        "method_to_category": method_to_category,
        "methods_valid": True,
    }


def get_forecast_method_names() -> tuple[str, ...]:
    """Return forecast method names from the current registry-derived catalog."""
    methods = get_forecast_methods_snapshot().get("methods", [])
    if not isinstance(methods, list):
        return ()
    return tuple(
        str(method_def.get("method"))
        for method_def in methods
        if isinstance(method_def, dict) and method_def.get("method")
    )


def _find_method_definition(method: str) -> Optional[Dict[str, Any]]:
    methods = get_forecast_methods_snapshot().get("methods", [])
    if not isinstance(methods, list):
        return None
    return _find_method_definition_in(method, methods)


def get_method_category(method: str) -> str:
    """Get the category of a forecast method."""
    snapshot = get_forecast_methods_snapshot()
    methods = snapshot.get("methods", [])
    method_def = _find_method_definition_in(method, methods) if isinstance(methods, list) else None
    if isinstance(method_def, dict):
        category = method_def.get("category")
        if isinstance(category, str) and category.strip():
            return category
    method_to_category = snapshot.get("method_to_category")
    if isinstance(method_to_category, dict):
        category = method_to_category.get(method)
        if isinstance(category, str) and category.strip():
            return category
    return "unknown"


def get_method_requirements(method: str) -> List[str]:
    """Get the list of required packages for a method."""
    method_def = _find_method_definition(method)
    if isinstance(method_def, dict):
        requires = method_def.get("requires")
        if isinstance(requires, list):
            return requires
    return []


def get_method_supports(method: str) -> Dict[str, bool]:
    """Get the supported data types and features for a method."""
    method_def = _find_method_definition(method)
    if isinstance(method_def, dict):
        supports = method_def.get("supports")
        if isinstance(supports, dict):
            return supports
    return {key: False for key in _DEFAULT_METHOD_SUPPORTS}


def validate_method_params(method: str, params: Dict[str, Any]) -> List[str]:
    """Validate method parameters and return list of errors."""
    errors = []

    method_def = _find_method_definition(method)
    if not method_def:
        errors.append(f"Unknown method: {method}")
        return errors

    # Check parameter types
    for param_def in method_def.get("params", []):
        param_name = param_def["name"]
        param_type = param_def["type"]

        if param_name in params:
            param_value = params[param_name]

            # Type validation
            if param_type == "int":
                try:
                    int(param_value)
                except (ValueError, TypeError):
                    errors.append(f"Parameter '{param_name}' should be an integer")
            elif param_type == "float":
                try:
                    float(param_value)
                except (ValueError, TypeError):
                    errors.append(f"Parameter '{param_name}' should be a float")
            elif param_type == "bool":
                if not isinstance(param_value, bool):
                    errors.append(f"Parameter '{param_name}' should be a boolean")
            elif param_type == "tuple":
                if not isinstance(param_value, (list, tuple)):
                    errors.append(f"Parameter '{param_name}' should be a tuple or list")
                else:
                    expected_len = None
                    try:
                        if "tuple_length" in param_def:
                            expected_len = int(param_def["tuple_length"])
                        elif "length" in param_def:
                            expected_len = int(param_def["length"])
                    except Exception:
                        expected_len = None
                    if expected_len is None:
                        expected_len = 4 if param_name == "seasonal_order" else 3
                    if len(param_value) != expected_len:
                        errors.append(f"Parameter '{param_name}' should have {expected_len} elements")

    return errors
