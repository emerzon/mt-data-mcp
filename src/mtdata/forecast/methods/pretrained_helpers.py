"""
Helper functions for pretrained forecast methods to eliminate DRY violations.

This module provides common functionality used across multiple pretrained
forecasting models to reduce code duplication and improve maintainability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def extract_context_window(
    series: np.ndarray,
    context_length: int,
    n: int,
    dtype: np.dtype = np.float64
) -> np.ndarray:
    """
    Extract context window from series with consistent logic.
    
    Args:
        series: Input time series data
        context_length: Maximum context length (0 or negative means use full series)
        n: Length of series
        dtype: NumPy dtype for output array
        
    Returns:
        Context window as numpy array
    """
    if context_length and context_length > 0:
        context = series[-int(min(n, context_length)) :]
    else:
        context = series
    
    return np.asarray(context, dtype=dtype)


def validate_and_clean_data(
    data: np.ndarray,
    method_name: str = "forecast"
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Validate and clean time series data by handling NaN values.
    
    Args:
        data: Input time series data
        method_name: Name of the forecasting method for error messages
        
    Returns:
        Tuple of (cleaned_data, error_message)
        If validation fails, returns (empty_array, error_message)
    """
    if len(data) == 0:
        return np.array([]), f"{method_name} error: empty context data"
    
    # Check for all NaN values
    if not np.any(np.isfinite(data)):
        return np.array([]), f"{method_name} error: context contains no finite values"
    
    # Clean the data - replace NaN with mean of finite values
    finite_mean = np.nanmean(data[np.isfinite(data)])
    if not np.isfinite(finite_mean):
        finite_mean = 0.0
    
    cleaned_data = np.nan_to_num(data, nan=float(finite_mean))
    
    # Validate cleaned data
    if not np.any(np.isfinite(cleaned_data)):
        return np.array([]), f"{method_name} error: cleaned context contains no finite values"
    
    return cleaned_data, None


def extract_forecast_values(
    forecast_obj: Any,
    fh: int,
    method_name: str = "forecast",
    do_mean: bool = True,
    do_median: bool = True
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Extract point forecast values from forecast object with fallback logic.
    
    Args:
        forecast_obj: Forecast object with mean, median, or samples attributes
        fh: Forecast horizon
        method_name: Name of the forecasting method for error messages
        do_mean: Whether to prefer mean estimate
        do_median: Whether to fallback to median estimate
        
    Returns:
        Tuple of (forecast_values, error_message)
    """
    f_vals = None
    
    try:
        # Try mean first if requested
        if do_mean and hasattr(forecast_obj, 'mean') and forecast_obj.mean is not None:
            f_vals = np.asarray(forecast_obj.mean, dtype=float)
        # Try median if requested and mean not available
        elif do_median and hasattr(forecast_obj, 'median') and forecast_obj.median is not None:
            f_vals = np.asarray(forecast_obj.median, dtype=float)
        # Fallback to samples if available
        elif hasattr(forecast_obj, 'samples') and forecast_obj.samples is not None:
            samples = np.asarray(forecast_obj.samples, dtype=float)
            if samples.ndim >= 2:
                f_vals = np.mean(samples, axis=0)
            else:
                f_vals = samples
        else:
            return None, f"{method_name} error: no forecast values available (no mean, median, or samples)"
    except Exception as extract_ex:
        return None, f"{method_name} error: failed to extract forecast values: {extract_ex}"
    
    # Validate extracted forecast
    if f_vals is None or len(f_vals) == 0:
        return None, f"{method_name} error: extracted forecast is empty or invalid"
    
    return f_vals, None


def adjust_forecast_length(
    values: np.ndarray,
    fh: int,
    method_name: str = "forecast"
) -> np.ndarray:
    """
    Adjust forecast length to match requested horizon by padding or truncating.
    
    Args:
        values: Forecast values array
        fh: Desired forecast horizon
        method_name: Name of the forecasting method for error messages
        
    Returns:
        Adjusted forecast values array
    """
    if len(values) < fh:
        # Pad with last value if forecast is shorter than requested
        return np.pad(values, (0, fh - len(values)), mode='edge')
    elif len(values) > fh:
        # Truncate if forecast is longer than requested
        return values[:fh]
    return values


def extract_quantiles_from_forecast(
    forecast_obj: Any,
    quantiles: Optional[List[Union[str, float, int]]],
    fh: int,
    method_name: str = "forecast"
) -> Dict[str, List[float]]:
    """
    Extract quantile forecasts from forecast object.
    
    Args:
        forecast_obj: Forecast object with quantile method
        quantiles: List of quantile levels to extract
        fh: Forecast horizon
        method_name: Name of the forecasting method for error messages
        
    Returns:
        Dictionary mapping quantile strings to forecast value lists
    """
    fq: Dict[str, List[float]] = {}
    
    if quantiles and hasattr(forecast_obj, 'quantile'):
        for q in quantiles:
            try:
                qf = float(q)
                q_value = forecast_obj.quantile(qf)
                if q_value is not None:
                    q_array = np.asarray(q_value, dtype=float)
                    # Adjust length to match forecast horizon
                    if len(q_array) < fh:
                        q_array = np.pad(q_array, (0, fh - len(q_array)), mode='edge')
                    elif len(q_array) > fh:
                        q_array = q_array[:fh]
                    fq[str(qf)] = q_array.tolist()
            except Exception:
                continue
    
    return fq


def process_quantile_levels(
    quantiles: Optional[List[Union[str, float, int]]],
    method_name: str = "forecast"
) -> Optional[List[float]]:
    """
    Process and validate quantile levels.
    
    Args:
        quantiles: Input quantiles list
        method_name: Name of the forecasting method for error messages
        
    Returns:
        List of valid quantile levels or None
    """
    if quantiles is None:
        return None
    
    if not isinstance(quantiles, (list, tuple)):
        return None
    
    try:
        q_levels = [float(q) for q in quantiles if q is not None]
        return q_levels if q_levels else None
    except Exception:
        return None


def build_params_used(
    base_params: Dict[str, Any],
    quantiles_dict: Optional[Dict[str, List[float]]] = None,
    context_length: Optional[int] = None,
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build standardized params_used dictionary for return values.
    
    Args:
        base_params: Base parameters dictionary
        quantiles_dict: Dictionary of quantile forecasts
        context_length: Context length used
        additional_params: Additional parameters to include
        
    Returns:
        Standardized params_used dictionary
    """
    params_used = base_params.copy()
    
    if context_length is not None:
        params_used['context_length'] = context_length
    
    if quantiles_dict:
        params_used['quantiles'] = sorted(list(quantiles_dict.keys()), key=lambda x: float(x))
    
    if additional_params:
        params_used.update(additional_params)
    
    return params_used


def safe_import_modules(
    required_modules: List[str],
    method_name: str = "forecast",
    fallback_imports: Optional[Dict[str, str]] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Safely import required modules with error handling.
    
    Args:
        required_modules: List of module names to import
        method_name: Name of the forecasting method for error messages
        fallback_imports: Dict of {alias: module_name} for fallback imports
        
    Returns:
        Tuple of (imported_modules_dict, error_message)
    """
    imported_modules = {}
    
    try:
        for module_name in required_modules:
            try:
                # Try direct import first
                module = __import__(module_name, fromlist=[''])
                imported_modules[module_name] = module
            except ImportError:
                # Try fallback import if specified
                if fallback_imports and module_name in fallback_imports:
                    fallback_name = fallback_imports[module_name]
                    module = __import__(fallback_name, fromlist=[''])
                    imported_modules[module_name] = module
                else:
                    raise
                    
    except Exception as ex:
        return None, f"{method_name} requires {', '.join(required_modules)}: {ex}"
    
    return imported_modules, None


def validate_required_params(
    params: Dict[str, Any],
    required_keys: List[str],
    method_name: str = "forecast"
) -> Tuple[bool, Optional[str]]:
    """
    Validate that required parameters are present.
    
    Args:
        params: Parameters dictionary
        required_keys: List of required parameter keys
        method_name: Name of the forecasting method for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_keys = []
    for key in required_keys:
        if key not in params or params[key] is None:
            missing_keys.append(key)
    
    if missing_keys:
        return False, f"{method_name} requires parameters: {', '.join(missing_keys)}"
    
    return True, None


def create_return_tuple(
    f_vals: Optional[np.ndarray],
    fq: Optional[Dict[str, List[float]]],
    params_used: Dict[str, Any],
    error: Optional[str]
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """
    Create standardized return tuple for forecast functions.
    
    Args:
        f_vals: Forecast values
        fq: Quantile forecasts
        params_used: Parameters used
        error: Error message
        
    Returns:
        Standardized return tuple
    """
    return (f_vals, fq, params_used, error)