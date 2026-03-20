from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import math
import numpy as np
import pandas as pd

from ..interface import ForecastMethod, ForecastResult
from ..registry import ForecastRegistry

class ClassicalMethod(ForecastMethod):
    """Base class for classical methods."""
    
    @property
    def category(self) -> str:
        return "classical"
        
    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": False}

@ForecastRegistry.register("naive")
class NaiveMethod(ClassicalMethod):
    PARAMS: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "naive"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        last_val = float(series.iloc[-1])
        f_vals = np.full(int(horizon), last_val, dtype=float)
        return ForecastResult(forecast=f_vals, params_used={})

@ForecastRegistry.register("drift")
class DriftMethod(ClassicalMethod):
    PARAMS: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "drift"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        vals = series.values
        n = int(vals.size)
        slope = (float(vals[-1]) - float(vals[0])) / float(max(1, n - 1))
        f_vals = float(vals[-1]) + slope * np.arange(1, int(horizon) + 1, dtype=float)
        return ForecastResult(forecast=f_vals, params_used={"slope": slope})

@ForecastRegistry.register("seasonal_naive")
class SeasonalNaiveMethod(ClassicalMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "seasonality", "type": "int", "description": "Seasonal period (m)."},
    ]

    @property
    def name(self) -> str:
        return "seasonal_naive"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        m = int(seasonality)
        if m <= 0 or len(series) < m:
            raise ValueError("Insufficient data for seasonal_naive")
        
        last_season = series.values[-m:]
        reps = int(math.ceil(int(horizon) / float(m)))
        f_vals = np.tile(last_season, reps)[: int(horizon)]
        return ForecastResult(forecast=f_vals, params_used={"m": m})

@ForecastRegistry.register("theta")
class ThetaMethod(ClassicalMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "alpha", "type": "float", "description": "SES smoothing factor (default: 0.2)."},
    ]

    @property
    def name(self) -> str:
        return "theta"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        vals = series.values
        n = int(vals.size)
        alpha = float(params.get('alpha', 0.2))
        
        tt = np.arange(1, n + 1, dtype=float)
        A = np.vstack([np.ones(n), tt]).T
        coef, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        trend_future = a + b * (tt[-1] + np.arange(1, int(horizon) + 1, dtype=float))
        
        level = float(vals[0])
        for v in vals[1:]:
            level = alpha * float(v) + (1.0 - alpha) * level
        ses_future = np.full(int(horizon), level, dtype=float)
        
        f_vals = 0.5 * (trend_future + ses_future)
        return ForecastResult(forecast=f_vals, params_used={"alpha": alpha, "trend_slope": b})

@ForecastRegistry.register("fourier_ols")
class FourierOLSMethod(ClassicalMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "seasonality", "type": "int", "description": "Seasonal period (m)."},
        {"name": "terms", "type": "int", "description": "Number of Fourier harmonics (default: 3)."},
        {"name": "trend", "type": "bool", "description": "Include linear trend (default: True)."},
    ]

    @property
    def name(self) -> str:
        return "fourier_ols"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        vals = series.values
        n = int(vals.size)
        m_eff = int(seasonality) if seasonality > 0 else 0
        K = params.get('terms')
        trend = params.get('trend', True)
        
        if K is None:
            K_eff = min(3, max(1, (m_eff // 2) if m_eff else 2))
        else:
            K_eff = int(K)
            
        tt = np.arange(1, n + 1, dtype=float)
        X_list = [np.ones(n)]
        if trend:
            X_list.append(tt)
        for k in range(1, K_eff + 1):
            w = 2.0 * math.pi * k / float(m_eff if m_eff else max(2, n))
            X_list.append(np.sin(w * tt))
            X_list.append(np.cos(w * tt))
        X = np.vstack(X_list).T
        
        coef, _, _, _ = np.linalg.lstsq(X, vals, rcond=None)
        
        tt_f = tt[-1] + np.arange(1, int(horizon) + 1, dtype=float)
        Xf_list = [np.ones(int(horizon))]
        if trend:
            Xf_list.append(tt_f)
        for k in range(1, K_eff + 1):
            w = 2.0 * math.pi * k / float(m_eff if m_eff else max(2, n))
            Xf_list.append(np.sin(w * tt_f))
            Xf_list.append(np.cos(w * tt_f))
        Xf = np.vstack(Xf_list).T
        
        f_vals = Xf @ coef
        return ForecastResult(
            forecast=f_vals.astype(float, copy=False), 
            params_used={"m": m_eff, "K": K_eff, "trend": bool(trend)}
        )

# Backward compatibility wrappers
def forecast_naive(series: np.ndarray, fh: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    res = ForecastRegistry.get("naive").forecast(pd.Series(series), fh, 0, {})
    return res.forecast, res.params_used

def forecast_drift(series: np.ndarray, fh: int, n: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Note: original drift had 'n' param for slope calculation window, but implementation used full series if n is None.
    # The new implementation uses full series. If 'n' was used to slice input, we should slice before calling.
    # The original implementation: slope = (last - first) / (n-1). If n was passed, it implied using only last n points?
    # Actually original implementation: if n is None: n = series.size. slope = (series[-1] - series[0]) / (n-1).
    # It seems it always used first and last point of the PASSED series. 
    # So 'n' argument in original function was redundant if it just meant series size.
    res = ForecastRegistry.get("drift").forecast(pd.Series(series), fh, 0, {})
    return res.forecast, res.params_used

def forecast_seasonal_naive(series: np.ndarray, fh: int, m: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    res = ForecastRegistry.get("seasonal_naive").forecast(pd.Series(series), fh, m, {})
    return res.forecast, res.params_used

def forecast_theta(series: np.ndarray, fh: int, alpha: float = 0.2) -> Tuple[np.ndarray, Dict[str, Any]]:
    res = ForecastRegistry.get("theta").forecast(pd.Series(series), fh, 0, {"alpha": alpha})
    return res.forecast, res.params_used

def forecast_fourier_ols(series: np.ndarray, fh: int, m: Optional[int], K: Optional[int], trend: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    res = ForecastRegistry.get("fourier_ols").forecast(pd.Series(series), fh, m or 0, {"terms": K, "trend": trend})
    return res.forecast, res.params_used
