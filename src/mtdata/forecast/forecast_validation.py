"""
Forecast validation utilities and error handling.
"""

import difflib
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

from ..shared.constants import TIMEFRAME_MAP
from ..shared.schema import DenoiseSpec, ForecastMethodLiteral, TimeframeLiteral
from ..shared.validators import invalid_timeframe_error
from .forecast_methods import (
    get_forecast_method_names,
    get_forecast_methods_snapshot,
    get_method_requirements,
    validate_method_params,
)


class ForecastValidationError(Exception):
    """Custom exception for forecast validation errors."""
    pass


def _normalize_method_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _method_family(value: Any) -> Optional[str]:
    normalized = _normalize_method_text(value)
    if not normalized:
        return None
    for prefix in ("chronos", "timesfm", "sf_", "skt_", "mlf_"):
        if normalized.startswith(prefix):
            return prefix.rstrip("_")
    return None


def _method_description_map() -> Dict[str, str]:
    snapshot = get_forecast_methods_snapshot()
    methods = snapshot.get("methods", [])
    if not isinstance(methods, list):
        return {}
    out: Dict[str, str] = {}
    for item in methods:
        if not isinstance(item, dict):
            continue
        name = str(item.get("method") or "").strip()
        if not name:
            continue
        description = str(item.get("description") or item.get("display_name") or "").strip()
        if description:
            out[name] = description.splitlines()[0].strip()
    return out


def suggest_forecast_methods(method: Any, valid_methods: List[str], limit: int = 5) -> List[str]:
    normalized_needle = _normalize_method_text(method)
    if not normalized_needle:
        return []
    family = _method_family(method)
    candidates = [
        str(candidate).strip()
        for candidate in valid_methods
        if str(candidate).strip()
    ]
    if family:
        same_family = [
            candidate
            for candidate in candidates
            if _method_family(candidate) == family
        ]
        if same_family:
            candidates = same_family
    ranked: List[str] = []
    normalized_to_name: Dict[str, str] = {}
    for name in candidates:
        lowered = _normalize_method_text(name)
        normalized_to_name[lowered] = name
        concepts = {
            lowered,
            lowered.removeprefix("sf_"),
            lowered.removeprefix("skt_"),
            lowered.removeprefix("mlf_"),
        }
        if any(
            normalized_needle == concept or normalized_needle in concept
            for concept in concepts
            if concept
        ):
            if name not in ranked:
                ranked.append(name)
    fuzzy = difflib.get_close_matches(
        normalized_needle,
        list(normalized_to_name),
        n=limit,
        cutoff=0.6,
    )
    for normalized in fuzzy:
        name = normalized_to_name.get(normalized, normalized)
        if name not in ranked:
            ranked.append(name)
    return ranked[:limit]


def format_invalid_method_error(method: Any, valid_methods: List[str]) -> str:
    suggestions = suggest_forecast_methods(method, valid_methods)
    message = f"Invalid method: {method!s}."
    if suggestions:
        descriptions = _method_description_map()
        suggestion_text = []
        for name in suggestions:
            description = descriptions.get(name)
            if description:
                suggestion_text.append(f"{name} ({description})")
            else:
                suggestion_text.append(name)
        message += f" Did you mean: {'; '.join(suggestion_text)}?"
    message += " Run forecast_list_methods for the full catalog."
    return message


def validate_timeframe(timeframe: TimeframeLiteral) -> List[str]:
    """Validate timeframe parameter."""
    errors = []
    if timeframe not in TIMEFRAME_MAP:
        errors.append(invalid_timeframe_error(timeframe, TIMEFRAME_MAP))
    return errors


def validate_method(method: ForecastMethodLiteral) -> List[str]:
    """Validate forecast method parameter."""
    errors = []
    method_l = str(method).lower().strip()
    valid_methods = list(get_forecast_method_names())
    if method_l not in valid_methods:
        errors.append(format_invalid_method_error(method, valid_methods))
    return errors


def validate_horizon(horizon: int) -> List[str]:
    """Validate horizon parameter."""
    errors = []
    if not isinstance(horizon, int) or horizon <= 0:
        errors.append("Horizon must be a positive integer")
    elif horizon > 1000:  # Reasonable upper limit
        errors.append("Horizon too large (maximum: 1000)")
    return errors


def validate_lookback(lookback: Optional[int]) -> List[str]:
    """Validate lookback parameter."""
    errors = []
    if lookback is not None:
        if not isinstance(lookback, int) or lookback <= 0:
            errors.append("Lookback must be a positive integer if specified")
        elif lookback > 10000:  # Reasonable upper limit
            errors.append("Lookback too large (maximum: 10000)")
    return errors


def validate_ci_alpha(ci_alpha: Optional[float]) -> List[str]:
    """Validate confidence interval alpha parameter."""
    errors = []
    if ci_alpha is not None:
        if not isinstance(ci_alpha, (int, float)):
            errors.append("ci_alpha must be a number if specified")
        elif not (0 < ci_alpha < 1):
            errors.append("ci_alpha must be between 0 and 1 (exclusive)")
    return errors


def validate_quantity_method_combination(quantity: str, method: str) -> List[str]:
    """Validate quantity and method compatibility."""
    errors = []
    quantity_l = str(quantity).lower().strip()
    method_l = str(method).lower().strip()

    # Check volatility method usage
    if quantity_l == 'volatility' or method_l.startswith('vol_'):
        errors.append("Use forecast_volatility for volatility models")

    # Validate quantity values
    if quantity_l not in ['price', 'return', 'volatility']:
        errors.append(f"Invalid quantity: {quantity}. Must be 'price', 'return', or 'volatility'")

    return errors


def validate_denoise_spec(denoise: Optional[DenoiseSpec]) -> List[str]:
    """Validate denoise specification."""
    errors = []
    if denoise is None:
        return errors

    if not isinstance(denoise, dict):
        errors.append("denoise must be a dictionary")
        return errors

    method = ""
    method_supports: Dict[str, Any] = {}
    try:
        from ..utils.denoise import get_denoise_methods_data as _get_denoise_methods_data

        methods_data = _get_denoise_methods_data()
        methods = methods_data.get("methods") if isinstance(methods_data, dict) else None
        if isinstance(methods, list):
            method_supports = {
                str(entry.get("method")).lower(): dict(entry)
                for entry in methods
                if isinstance(entry, dict) and entry.get("method")
            }
    except Exception:
        method_supports = {}

    # Validate required fields
    if 'method' not in denoise:
        errors.append("denoise must specify a 'method'")
    else:
        valid_methods = sorted(method_supports) or ['none', 'ema', 'sma', 'median', 'lowpass_fft', 'wavelet', 'emd', 'eemd', 'ceemdan']
        method = str(denoise['method']).lower()
        if method not in valid_methods:
            errors.append(f"Invalid denoise method: {method}. Valid options: {valid_methods}")

    # Validate causality
    if 'causality' in denoise:
        causality = str(denoise['causality']).lower()
        if causality not in ['causal', 'zero_phase']:
            errors.append(f"Invalid denoise causality: {causality}. Must be 'causal' or 'zero_phase'")
        elif method and method in method_supports:
            supported = method_supports[method].get("supports", {}).get("causality") or []
            if causality not in supported:
                errors.append(
                    f"Invalid denoise causality for {method}: {causality}. "
                    f"Supported values: {supported}"
                )

    # Validate when parameter
    if 'when' in denoise:
        when = str(denoise['when']).lower()
        if when not in ['pre_ti', 'post_ti']:
            errors.append(f"Invalid denoise when: {when}. Must be 'pre_ti' or 'post_ti'")

    return errors


def validate_features_spec(features: Optional[Dict[str, Any]]) -> List[str]:
    """Validate features specification."""
    errors = []
    if features is None:
        return errors

    if not isinstance(features, dict):
        errors.append("features must be a dictionary")
        return errors

    # Validate technical indicators specification
    if 'ti' in features:
        ti_spec = features['ti']
        if not isinstance(ti_spec, (str, list, dict)):
            errors.append("features.ti must be a string, list, or dictionary")

    # Validate exogenous variables specification
    if 'exog' in features:
        exog = features['exog']
        if not isinstance(exog, (str, list)):
            errors.append("features.exog must be a string or list")
        elif isinstance(exog, str):
            # Basic validation of comma-separated string
            if not exog.strip():
                errors.append("features.exog string cannot be empty")

    return errors


def validate_dimred_spec(dimred_method: Optional[str], dimred_params: Optional[Dict[str, Any]]) -> List[str]:
    """Validate dimensionality reduction specification."""
    errors = []

    if dimred_method is None:
        return errors

    valid_methods = ['pca', 'tsne', 'selectkbest']
    if str(dimred_method).lower() not in valid_methods:
        errors.append(f"Invalid dimred_method: {dimred_method}. Valid options: {valid_methods}")

    if dimred_params is not None:
        if not isinstance(dimred_params, dict):
            errors.append("dimred_params must be a dictionary")
        else:
            # Validate specific parameters for each method
            method = str(dimred_method).lower()
            if method == 'pca':
                if 'n_components' in dimred_params:
                    n_components = dimred_params['n_components']
                    if not isinstance(n_components, int) or n_components <= 0:
                        errors.append("dimred_params.n_components must be a positive integer")
            elif method == 'tsne':
                if 'n_components' in dimred_params:
                    n_components = dimred_params['n_components']
                    if n_components not in [2, 3]:
                        errors.append("dimred_params.n_components for tsne must be 2 or 3")
            elif method == 'selectkbest':
                if 'k' in dimred_params:
                    k = dimred_params['k']
                    if not isinstance(k, int) or k <= 0:
                        errors.append("dimred_params.k must be a positive integer")

    return errors


def validate_target_spec(target_spec: Optional[Dict[str, Any]]) -> List[str]:
    """Validate target specification."""
    errors = []
    if target_spec is None:
        return errors

    if not isinstance(target_spec, dict):
        errors.append("target_spec must be a dictionary")
        return errors

    # Validate column/base field
    if 'column' in target_spec:
        column = target_spec['column']
        if not isinstance(column, str) or not column.strip():
            errors.append("target_spec.column must be a non-empty string")
    if 'base' in target_spec:
        base = target_spec['base']
        if not isinstance(base, str) or not base.strip():
            errors.append("target_spec.base must be a non-empty string")

    # Validate transform field
    if 'transform' in target_spec:
        valid_transforms = ['none', 'return', 'log_return', 'diff', 'pct_change', 'log', 'pct']
        transform = str(target_spec['transform']).lower()
        if transform not in valid_transforms:
            errors.append(f"Invalid target_spec.transform: {transform}. Valid options: {valid_transforms}")

    return errors


def validate_data_sufficiency(df: pd.DataFrame, base_col: str, min_points: int = 3) -> List[str]:
    """Validate that sufficient data is available for forecasting."""
    errors = []

    if len(df) < min_points:
        errors.append(f"Insufficient data: only {len(df)} bars available, minimum {min_points} required")

    if base_col not in df.columns:
        errors.append(f"Base column '{base_col}' not found in data")
    else:
        # Check for sufficient non-NaN values
        non_nan_count = df[base_col].notna().sum()
        if non_nan_count < min_points:
            errors.append(f"Insufficient non-NaN values in column '{base_col}': {non_nan_count} valid points")

    return errors


def validate_seasonality_for_method(method: str, seasonality: Optional[int]) -> List[str]:
    """Validate seasonality parameter for seasonal methods."""
    errors = []
    method_l = str(method).lower().strip()

    if method_l == 'seasonal_naive':
        if seasonality is None or seasonality <= 0:
            errors.append("seasonal_naive requires a positive 'seasonality' parameter")
    elif method_l in ['holt_winters_add', 'holt_winters_mul', 'sarima', 'fourier_ols']:
        if seasonality is not None and seasonality <= 0:
            errors.append(f"{method} requires a positive 'seasonality' if specified")

    return errors


def check_method_dependencies(method: str) -> List[str]:
    """Check if required dependencies are installed for a method."""
    errors = []
    requirements = get_method_requirements(method)

    for package in requirements:
        try:
            __import__(package)
        except ImportError:
            errors.append(f"Missing required package: {package}")

    return errors


def validate_forecast_request(
    symbol: str,
    timeframe: TimeframeLiteral,
    method: ForecastMethodLiteral,
    horizon: int,
    lookback: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    ci_alpha: Optional[float] = 0.05,
    quantity: Literal['price', 'return', 'volatility'] = 'price',
    denoise: Optional[DenoiseSpec] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    target_spec: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Comprehensive validation of forecast request parameters."""
    errors = []

    # Basic parameter validation
    errors.extend(validate_timeframe(timeframe))
    errors.extend(validate_method(method))
    errors.extend(validate_horizon(horizon))
    errors.extend(validate_lookback(lookback))
    errors.extend(validate_ci_alpha(ci_alpha))
    errors.extend(validate_quantity_method_combination(quantity, method))

    # Advanced parameter validation
    errors.extend(validate_denoise_spec(denoise))
    errors.extend(validate_features_spec(features))
    errors.extend(validate_dimred_spec(dimred_method, dimred_params))
    errors.extend(validate_target_spec(target_spec))

    # Method-specific validation
    if params:
        p = params if isinstance(params, dict) else {}
        seasonality = p.get('seasonality')
        if seasonality is not None:
            try:
                seasonality = int(seasonality)
            except (ValueError, TypeError):
                seasonality = None
        errors.extend(validate_seasonality_for_method(method, seasonality))
        errors.extend(validate_method_params(method, p))

    # Check dependencies
    errors.extend(check_method_dependencies(method))

    return errors


def create_error_response(errors: List[str]) -> Dict[str, Any]:
    """Create a standardized error response."""
    if not errors:
        return {"success": True}

    return {
        "error": "; ".join(errors),
        "validation_errors": errors,
        "error_count": len(errors)
    }


def safe_cast_numeric(value: Any, param_name: str) -> Union[int, float, str]:
    """Safely cast a value to numeric type or return original."""
    if value is None:
        return None

    try:
        # Try int first
        return int(value)
    except (ValueError, TypeError):
        try:
            # Try float
            return float(value)
        except (ValueError, TypeError):
            # Return original if can't cast
            return value


def sanitize_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize and cast forecast parameters."""
    if params is None:
        return {}

    sanitized = {}
    for key, value in params.items():
        sanitized[key] = safe_cast_numeric(value, key)

    return sanitized
