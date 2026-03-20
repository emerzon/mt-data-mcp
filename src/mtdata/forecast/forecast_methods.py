"""
Forecast method definitions and metadata (registry-derived).
"""

from typing import Any, Dict, List, Optional

from .forecast_registry import (
    DEFAULT_METHOD_SUPPORTS as _DEFAULT_METHOD_SUPPORTS,
    _find_method_definition as _registry_find_method_definition,
    get_forecast_methods_data as _registry_methods_data,
)

# Supported forecast methods (derived from registry to avoid drift)
_METHOD_DATA = _registry_methods_data()
FORECAST_METHODS = tuple(
    m.get("method") for m in _METHOD_DATA.get("methods", []) if m.get("method")
)


def get_forecast_methods_data() -> Dict[str, Any]:
    """Get comprehensive data about available forecast methods."""
    return _registry_methods_data()


def _find_method_definition(method: str) -> Optional[Dict[str, Any]]:
    return _registry_find_method_definition(method, get_forecast_methods_data())


def get_method_category(method: str) -> str:
    """Get the category of a forecast method."""
    method_data = get_forecast_methods_data()
    method_def = _registry_find_method_definition(method, method_data)
    if isinstance(method_def, dict):
        category = method_def.get("category")
        if isinstance(category, str) and category.strip():
            return category
    categories = method_data.get("categories") or {}
    for cat, methods in categories.items():
        if method in methods:
            return cat
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
                        errors.append(
                            f"Parameter '{param_name}' should have {expected_len} elements"
                        )

    return errors
