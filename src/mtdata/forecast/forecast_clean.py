"""
Simplified forecast execution module.

This replaces the massive monolithic forecast.py with a clean, focused implementation.
"""

from typing import Any, Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import warnings

from ..core.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..utils.mt5 import _mt5_epoch_to_utc, _mt5_copy_rates_from, _ensure_symbol_ready
from ..utils.utils import (
    _parse_start_datetime,
    _format_time_minimal,
    to_float_np,
    parse_kv_or_json as _parse_kv_or_json,
)

# Import our new modular components
from .forecast_registry import get_forecast_methods_data
from .forecast_preprocessing import (
    _default_seasonality_period, 
    _next_times_from_last,
    prepare_features,
    apply_preprocessing
)
from .method_adapters import get_method_adapter
from .common import fetch_history as _fetch_history

# Import only availability flags we need
from .forecast_registry import (
    _SM_ETS_AVAILABLE, _SM_SARIMAX_AVAILABLE, _NF_AVAILABLE, 
    _MLF_AVAILABLE, _SF_AVAILABLE, _LGB_AVAILABLE,
    _CHRONOS_AVAILABLE, _TIMESFM_AVAILABLE, _LAG_LLAMA_AVAILABLE
)

# Type aliases to avoid import cycles
try:
    from ..core.server import ForecastMethodLiteral, TimeframeLiteral, DenoiseSpec  # type: ignore
except Exception:
    ForecastMethodLiteral = str
    TimeframeLiteral = str 
    DenoiseSpec = Dict[str, Any]


def forecast(
    symbol: str,
    timeframe: str = "H1",
    method: str = "theta", 
    horizon: int = 12,
    lookback: Optional[int] = None,
    as_of: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    ci_alpha: Optional[float] = 0.05,
    quantity: str = 'price',
    target: str = 'price', 
    denoise: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    timezone: str = 'UTC'
) -> Dict[str, Any]:
    """
    Clean, simplified forecast function using modular components.
    
    This replaces the massive monolithic implementation with a focused approach.
    """
    try:
        # Validate basic inputs
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}"}
        
        method_l = method.lower().strip()
        
        # Check method availability using registry
        methods_data = get_forecast_methods_data()
        method_info = None
        for m in methods_data["methods"]:
            if m["method"] == method_l:
                method_info = m
                break
        
        if not method_info:
            return {"error": f"Unknown method: {method}"}
        
        if not method_info["available"]:
            missing = ", ".join(method_info["requires"])
            return {"error": f"{method_l} requires: {missing}"}
        
        # Fetch historical data
        try:
            result = _fetch_history(
                symbol=symbol,
                timeframe=timeframe,
                limit=lookback or 500,
                as_of=as_of,
                timezone=timezone
            )
            if "error" in result:
                return result
            
            df = result["df"]
            last_epoch = result["last_epoch"]
            
        except Exception as ex:
            return {"error": f"Data fetch failed: {ex}"}
        
        if len(df) < 10:
            return {"error": f"Insufficient data: {len(df)} bars"}
        
        # Apply preprocessing
        base_col = apply_preprocessing(df, quantity, target, denoise)
        if base_col not in df.columns:
            return {"error": f"Column {base_col} not found after preprocessing"}
        
        # Convert to numpy and handle transformations
        series = to_float_np(df[base_col])
        if quantity == 'return':
            if len(series) < 2:
                return {"error": "Insufficient data for returns"}
            series = np.diff(np.log(series))
        elif quantity == 'volatility':
            if len(series) < 30:
                return {"error": "Insufficient data for volatility"}
            returns = np.diff(np.log(series))
            # Simple rolling volatility estimation
            window = min(30, len(returns) // 2)
            vol_series = []
            for i in range(window, len(returns)):
                vol_series.append(np.std(returns[i-window:i]) * np.sqrt(252))
            series = np.array(vol_series)
        
        if len(series) < max(3, horizon):
            return {"error": f"Insufficient data for method {method_l}"}
        
        # Prepare parameters
        p = _parse_kv_or_json(params) if params else {}
        m = p.get('seasonality') or _default_seasonality_period(timeframe)
        
        # Generate future timestamps
        tf_secs = TIMEFRAME_SECONDS[timeframe]
        future_times = _next_times_from_last(last_epoch, tf_secs, horizon)
        
        # Prepare features if specified
        exog_used, exog_future, include_cols, ti_cols = prepare_features(
            df, features, future_times, horizon
        )
        
        # Execute forecast using appropriate adapter
        try:
            adapter = get_method_adapter(method_l)
            f_vals, params_used = adapter.execute(
                series=series,
                fh=horizon,
                params=p,
                timeframe=timeframe,
                m=m,
                exog_used=exog_used,
                exog_future=exog_future,
                future_times=future_times
            )
        except Exception as ex:
            return {"error": f"{method_l} execution failed: {ex}"}
        
        # Post-process results
        if quantity == 'return' and target == 'price':
            # Convert returns back to prices
            last_price = float(df[base_col].iloc[-1])
            f_vals = last_price * np.exp(np.cumsum(f_vals))
        
        # Format output
        forecast_dict = {}
        for i, (t, v) in enumerate(zip(future_times, f_vals)):
            forecast_dict[_format_time_minimal(t)] = float(v)
        
        # Build response
        response = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe, 
            "method": method_l,
            "horizon": horizon,
            "quantity": quantity,
            "target": target,
            "forecast": forecast_dict,
            "params_used": params_used,
            "data_points": len(series),
            "last_value": float(series[-1]) if len(series) > 0 else None,
            "seasonality": m
        }
        
        if include_cols:
            response["features_included"] = include_cols
        if ti_cols:
            response["indicators_added"] = ti_cols
        
        return response
        
    except Exception as ex:
        return {"error": f"Forecast failed: {ex}"}


# Re-export the registry function for backward compatibility
__all__ = [
    'forecast',
    'get_forecast_methods_data'
]
