"""
Helper functions for pretrained forecast methods to eliminate DRY violations.

This module provides common functionality used across multiple pretrained
forecasting models to reduce code duplication and improve maintainability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..common import edge_pad_to_length as _edge_pad_to_length


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
    return _edge_pad_to_length(values, int(fh))


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
                    q_array = _edge_pad_to_length(q_array, int(fh))
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


