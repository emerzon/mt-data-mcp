"""
Forecast validation utilities and error handling.
"""

from typing import Any, Dict, Optional, List, Literal, Union
import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to path for relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mtdata.core.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from mtdata.forecast.forecast_methods import FORECAST_METHODS, validate_method_params, get_method_requirements

# Local fallbacks for typing aliases (avoid import cycle)
try:
    from mtdata.core.server import ForecastMethodLiteral, TimeframeLiteral, DenoiseSpec  # type: ignore
except Exception:  # runtime fallback
    ForecastMethodLiteral = str
    TimeframeLiteral = str
    DenoiseSpec = Dict[str, Any]


class ForecastValidationError(Exception):
    """Custom exception for forecast validation errors."""
    pass


def validate_timeframe(timeframe: TimeframeLiteral) -> List[str]:
    """Validate timeframe parameter."""
    errors = []
    if timeframe not in TIMEFRAME_MAP:
        errors.append(f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}")
    return errors


def validate_method(method: ForecastMethodLiteral) -> List[str]:
    """Validate forecast method parameter."""
    errors = []
    method_l = str(method).lower().strip()
    if method_l not in FORECAST_METHODS:
        errors.append(f"Invalid method: {method}. Valid options: {list(FORECAST_METHODS)}")
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


def validate_quantity_target_combination(quantity: str, target: str, method: str) -> List[str]:
    """Validate quantity and target parameter combination."""
    errors = []
    quantity_l = str(quantity).lower().strip()
    target_l = str(target).lower().strip()
    method_l = str(method).lower().strip()

    # Check volatility method usage
    if quantity_l == 'volatility' or method_l.startswith('vol_'):
        errors.append("Use forecast_volatility for volatility models")

    # Validate quantity values
    if quantity_l not in ['price', 'return', 'volatility']:
        errors.append(f"Invalid quantity: {quantity}. Must be 'price', 'return', or 'volatility'")

    # Validate target values
    if target_l not in ['price', 'return']:
        errors.append(f"Invalid target: {target}. Must be 'price' or 'return'")

    return errors


def validate_denoise_spec(denoise: Optional[DenoiseSpec]) -> List[str]:
    """Validate denoise specification."""
    errors = []
    if denoise is None:
        return errors

    if not isinstance(denoise, dict):
        errors.append("denoise must be a dictionary")
        return errors

    # Validate required fields
    if 'method' not in denoise:
        errors.append("denoise must specify a 'method'")
    else:
        valid_methods = ['none', 'ema', 'sma', 'median', 'lowpass_fft', 'wavelet', 'emd', 'eemd', 'ceemdan']
        method = str(denoise['method']).lower()
        if method not in valid_methods:
            errors.append(f"Invalid denoise method: {method}. Valid options: {valid_methods}")

    # Validate causality
    if 'causality' in denoise:
        causality = str(denoise['causality']).lower()
        if causality not in ['causal', 'zero_phase']:
            errors.append(f"Invalid denoise causality: {causality}. Must be 'causal' or 'zero_phase'")

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

    # Validate column field
    if 'column' in target_spec:
        column = target_spec['column']
        if not isinstance(column, str) or not column.strip():
            errors.append("target_spec.column must be a non-empty string")

    # Validate transform field
    if 'transform' in target_spec:
        valid_transforms = ['log', 'diff', 'pct']
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
    target: Literal['price', 'return'] = 'price',
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
    errors.extend(validate_quantity_target_combination(quantity, target, method))

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