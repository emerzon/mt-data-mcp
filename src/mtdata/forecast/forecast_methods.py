"""
Forecast method definitions and metadata (registry-derived).
"""

from typing import Any, Dict, List

from .forecast_registry import get_forecast_methods_data as _registry_methods_data

# Local fallbacks for typing aliases (avoid import cycle)
try:
    from mtdata.core.server import ForecastMethodLiteral, TimeframeLiteral  # type: ignore
except Exception:  # runtime fallback
    ForecastMethodLiteral = str
    TimeframeLiteral = str

# Supported forecast methods (derived from registry to avoid drift)
_METHOD_DATA = _registry_methods_data()
FORECAST_METHODS = tuple(
    m.get("method") for m in _METHOD_DATA.get("methods", []) if m.get("method")
)


def get_forecast_methods_data() -> Dict[str, Any]:
    """Get comprehensive data about available forecast methods."""
    return _registry_methods_data()


def get_method_category(method: str) -> str:
    """Get the category of a forecast method."""
    method_data = get_forecast_methods_data()
    categories = method_data.get("categories") or {}
    for cat, methods in categories.items():
        if method in methods:
            return cat
    return "unknown"


def get_method_requirements(method: str) -> List[str]:
    """Get the list of required packages for a method."""
    method_data = get_forecast_methods_data()
    for m in method_data["methods"]:
        if m["method"] == method:
            return m["requires"]
    return []


def get_method_supports(method: str) -> Dict[str, bool]:
    """Get the supported data types and features for a method."""
    method_data = get_forecast_methods_data()
    for m in method_data["methods"]:
        if m["method"] == method:
            return m["supports"]
    return {"price": False, "return": False, "volatility": False, "ci": False}


def validate_method_params(method: str, params: Dict[str, Any]) -> List[str]:
    """Validate method parameters and return list of errors."""
    errors = []

    # Get method definition
    method_data = get_forecast_methods_data()
    method_def = None
    for m in method_data["methods"]:
        if m["method"] == method:
            method_def = m
            break

    if not method_def:
        errors.append(f"Unknown method: {method}")
        return errors

    # Check parameter types
    for param_def in method_def["params"]:
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
                elif len(param_value) != 3:  # Assuming (p,d,q) order
                    errors.append(f"Parameter '{param_name}' should have 3 elements")

    return errors
