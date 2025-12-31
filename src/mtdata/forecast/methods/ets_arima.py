from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import warnings
import numpy as np
import pandas as pd

from ..interface import ForecastMethod, ForecastResult
from ..registry import ForecastRegistry

try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing as _SES, ExponentialSmoothing as _ETS  # type: ignore
    _SM_ETS_AVAILABLE = True
except Exception:
    _SM_ETS_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX  # type: ignore
    _SM_SARIMAX_AVAILABLE = True
except Exception:
    _SM_SARIMAX_AVAILABLE = False

class ETSArimaMethod(ForecastMethod):
    """Base class for ETS and ARIMA methods."""
    
    @property
    def category(self) -> str:
        return "ets_arima"
        
    @property
    def required_packages(self) -> List[str]:
        return ["statsmodels"]
        
    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": True}

@ForecastRegistry.register("ses")
class SESMethod(ETSArimaMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "alpha", "type": "float|null", "description": "Smoothing level (auto if omitted)."},
    ]

    @property
    def name(self) -> str:
        return "ses"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        if not _SM_ETS_AVAILABLE:
            raise RuntimeError("SES requires statsmodels")

        vals = series.values
        alpha = params.get('alpha')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if alpha is None:
                res = _SES(vals, initialization_method='heuristic').fit(optimized=True)
            else:
                res = _SES(vals, initialization_method='heuristic').fit(smoothing_level=float(alpha), optimized=False)
                
        f_vals = np.asarray(res.forecast(int(horizon)), dtype=float)
        
        # Recover effective alpha
        alpha_used = alpha
        try:
            par = getattr(res, 'params', None)
            if par is not None and hasattr(par, 'get'):
                alpha_used = par.get('smoothing_level', alpha)
        except Exception:
            pass
            
        return ForecastResult(forecast=f_vals, params_used={"alpha": alpha_used})

@ForecastRegistry.register("holt")
class HoltMethod(ETSArimaMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "alpha", "type": "float|null", "description": "Level smoothing (auto if omitted)."},
        {"name": "beta", "type": "float|null", "description": "Trend smoothing (auto if omitted)."},
        {"name": "damped", "type": "bool", "description": "Use damped trend (default: False)."},
    ]

    @property
    def name(self) -> str:
        return "holt"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        if not _SM_ETS_AVAILABLE:
            raise RuntimeError("Holt requires statsmodels")

        vals = series.values
        damped = bool(params.get('damped', False))
        alpha = params.get('alpha')
        beta = params.get('beta')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _ETS(vals, trend='add', damped_trend=damped, initialization_method='heuristic')
            use_manual = alpha is not None or beta is not None
            if use_manual:
                res = model.fit(
                    optimized=False,
                    smoothing_level=None if alpha is None else float(alpha),
                    smoothing_trend=None if beta is None else float(beta),
                )
            else:
                res = model.fit(optimized=True)
            
        f_vals = np.asarray(res.forecast(int(horizon)), dtype=float)
        params_used = {"damped": damped}
        if alpha is not None:
            params_used["alpha"] = float(alpha)
        if beta is not None:
            params_used["beta"] = float(beta)
        return ForecastResult(forecast=f_vals, params_used=params_used)

@ForecastRegistry.register("holt_winters_add")
class HoltWintersAddMethod(ETSArimaMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "seasonality", "type": "int", "description": "Seasonal period (m)."},
        {"name": "alpha", "type": "float|null", "description": "Level smoothing (auto if omitted)."},
        {"name": "beta", "type": "float|null", "description": "Trend smoothing (auto if omitted)."},
        {"name": "gamma", "type": "float|null", "description": "Seasonal smoothing (auto if omitted)."},
        {"name": "damped", "type": "bool", "description": "Use damped trend (default: False)."},
    ]

    @property
    def name(self) -> str:
        return "holt_winters_add"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        return self._forecast_hw(series, horizon, seasonality, params, 'add')

    def _forecast_hw(self, series, horizon, seasonality, params, seasonal_type):
        if not _SM_ETS_AVAILABLE:
            raise RuntimeError("Holt-Winters requires statsmodels")

        m = int(seasonality)
        if m <= 0:
            raise ValueError("Holt-Winters requires positive seasonality")
            
        vals = series.values
        damped = bool(params.get('damped', False))
        alpha = params.get('alpha')
        beta = params.get('beta')
        gamma = params.get('gamma')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _ETS(vals, trend='add', seasonal=seasonal_type, seasonal_periods=m, damped_trend=damped, initialization_method='heuristic')
            use_manual = alpha is not None or beta is not None or gamma is not None
            if use_manual:
                res = model.fit(
                    optimized=False,
                    smoothing_level=None if alpha is None else float(alpha),
                    smoothing_trend=None if beta is None else float(beta),
                    smoothing_seasonal=None if gamma is None else float(gamma),
                )
            else:
                res = model.fit(optimized=True)
            
        f_vals = np.asarray(res.forecast(int(horizon)), dtype=float)
        params_used = {"seasonal": seasonal_type, "m": m, "damped": damped}
        if alpha is not None:
            params_used["alpha"] = float(alpha)
        if beta is not None:
            params_used["beta"] = float(beta)
        if gamma is not None:
            params_used["gamma"] = float(gamma)
        return ForecastResult(forecast=f_vals, params_used=params_used)

@ForecastRegistry.register("holt_winters_mul")
class HoltWintersMulMethod(HoltWintersAddMethod):
    PARAMS: List[Dict[str, Any]] = HoltWintersAddMethod.PARAMS

    @property
    def name(self) -> str:
        return "holt_winters_mul"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        return self._forecast_hw(series, horizon, seasonality, params, 'mul')


@ForecastRegistry.register("ets")
class ETSMethod(ETSArimaMethod):
    """Generic exponential smoothing (ETS) with optional trend/seasonality."""

    PARAMS: List[Dict[str, Any]] = [
        {"name": "seasonality", "type": "int", "description": "Seasonal period (m)."},
        {"name": "trend", "type": "str|null", "description": "Trend: add|mul|null (default: add)."},
        {
            "name": "seasonal",
            "type": "str|null",
            "description": "Seasonal: add|mul|null|auto (default: auto).",
        },
        {"name": "damped", "type": "bool", "description": "Use damped trend (default: False)."},
        {"name": "alpha", "type": "float|null", "description": "Level smoothing (auto if omitted)."},
        {"name": "beta", "type": "float|null", "description": "Trend smoothing (auto if omitted)."},
        {"name": "gamma", "type": "float|null", "description": "Seasonal smoothing (auto if omitted)."},
    ]

    @property
    def name(self) -> str:
        return "ets"

    @staticmethod
    def _norm_component(val: Any, *, allow_auto: bool = False) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip().lower()
        if not s or s in {"none", "null", "nil"}:
            return None
        if allow_auto and s == "auto":
            return "auto"
        if s in {"add", "a", "additive"}:
            return "add"
        if s in {"mul", "m", "multiplicative"}:
            return "mul"
        raise ValueError(
            f"Invalid ETS component: {val!r} (use add|mul|null{'|auto' if allow_auto else ''})"
        )

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        if not _SM_ETS_AVAILABLE:
            raise RuntimeError("ETS requires statsmodels")

        vals = series.values
        m = int(seasonality or 0)

        trend = self._norm_component(params.get("trend", "add"))
        seasonal_raw = self._norm_component(params.get("seasonal", "auto"), allow_auto=True)
        if seasonal_raw == "auto":
            seasonal = "add" if m >= 2 else None
        else:
            seasonal = seasonal_raw

        if seasonal is not None and m < 2:
            raise ValueError("ETS seasonal component requires seasonality >= 2")

        damped = bool(params.get("damped", False))
        if trend is None:
            damped = False

        alpha = params.get("alpha")
        beta = params.get("beta")
        gamma = params.get("gamma")
        use_manual = alpha is not None or beta is not None or gamma is not None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _ETS(
                vals,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=m if seasonal is not None else None,
                damped_trend=damped,
                initialization_method="heuristic",
            )
            if use_manual:
                res = model.fit(
                    optimized=False,
                    smoothing_level=None if alpha is None else float(alpha),
                    smoothing_trend=None if beta is None else float(beta),
                    smoothing_seasonal=None if gamma is None else float(gamma),
                )
            else:
                res = model.fit(optimized=True)

        f_vals = np.asarray(res.forecast(int(horizon)), dtype=float)

        params_used: Dict[str, Any] = {
            "trend": trend,
            "seasonal": seasonal,
            "m": m if seasonal is not None else 0,
            "damped": damped,
        }
        try:
            par = getattr(res, "params", None)
            if par is not None and hasattr(par, "get"):
                for key, out_key in (
                    ("smoothing_level", "alpha"),
                    ("smoothing_trend", "beta"),
                    ("smoothing_seasonal", "gamma"),
                ):
                    v = par.get(key)
                    if v is not None:
                        params_used[out_key] = float(v)
        except Exception:
            pass

        return ForecastResult(forecast=f_vals, params_used=params_used)


@ForecastRegistry.register("arima")
class ARIMAMethod(ETSArimaMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "order", "type": "tuple", "description": "(p,d,q) order (optional)."},
        {"name": "p", "type": "int", "description": "AR order (default: 1)."},
        {"name": "d", "type": "int", "description": "Differencing order (default: 1)."},
        {"name": "q", "type": "int", "description": "MA order (default: 1)."},
        {"name": "trend", "type": "str", "description": "Trend spec (default: c)."},
        {"name": "alpha", "type": "float", "description": "CI alpha (default: 0.05)."},
    ]

    @property
    def name(self) -> str:
        return "arima"
        
    @property
    def category(self) -> str:
        return "arima"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        return self._forecast_sarimax(series, horizon, seasonality, params, seasonal=False, exog_future=exog_future, **kwargs)

    def _forecast_sarimax(self, series, horizon, seasonality, params, seasonal, exog_future=None, **kwargs):
        if not _SM_SARIMAX_AVAILABLE:
            raise RuntimeError("SARIMAX requires statsmodels")

        vals = series.values.astype(float)
        if params.get('order') is not None:
            order = params.get('order')
        else:
            p = int(params.get('p', 1))
            d = int(params.get('d', 1))
            q = int(params.get('q', 1))
            order = (p, d, q)

        if params.get('seasonal_order') is not None:
            seasonal_order = params.get('seasonal_order')
        else:
            P = int(params.get('P', 0))
            D = int(params.get('D', 0))
            Q = int(params.get('Q', 0))
            seasonal_order = (P, D, Q, int(seasonality or 0))
        if seasonal and seasonality > 1 and seasonal_order == (0, 0, 0, 0):
             # Auto-guess seasonal order if not provided but requested
             seasonal_order = (0, 1, 1, seasonality)
             
        trend = params.get('trend', 'c')
        ci_alpha = params.get('alpha', 0.05)
        
        exog_used = kwargs.get('exog_used')
        exog_future_arr = kwargs.get('exog_future') # This might come from kwargs or explicit arg
        
        # If exog_future was passed as explicit arg, use it (it might be DataFrame)
        # The interface defines exog_future as Optional[pd.DataFrame]
        # But legacy wrapper passes numpy array.
        # We need to handle both.
        
        exog_u = exog_used
        exog_f = exog_future_arr if exog_future_arr is not None else exog_future
        
        if isinstance(exog_f, pd.DataFrame):
            exog_f = exog_f.values
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _SARIMAX(
                vals,
                order=order,
                seasonal_order=seasonal_order,
                trend=str(trend),
                enforce_stationarity=True,
                enforce_invertibility=True,
                exog=exog_u
            )
            res = model.fit(method='lbfgs', disp=False, maxiter=100)
            
            if exog_f is not None:
                pred = res.get_forecast(steps=int(horizon), exog=exog_f)
            else:
                pred = res.get_forecast(steps=int(horizon))
                
        pm = pred.predicted_mean
        f_vals = np.asarray(pm, dtype=float)
        
        ci = None
        try:
            _alpha = float(ci_alpha) if ci_alpha is not None else 0.05
            ci_df = pred.conf_int(alpha=_alpha)
            ci_arr = np.asarray(ci_df)
            if ci_arr.ndim == 2 and ci_arr.shape[1] >= 2:
                ci = (ci_arr[:, 0], ci_arr[:, 1])
        except Exception:
            pass
            
        params_used = {"order": tuple(order), "seasonal_order": tuple(seasonal_order), "trend": str(trend)}
        if exog_u is not None:
            params_used["exog"] = {"n_features": int(exog_u.shape[1])}
            
        return ForecastResult(forecast=f_vals, ci_values=ci, params_used=params_used)

@ForecastRegistry.register("sarima")
class SARIMAMethod(ARIMAMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "order", "type": "tuple", "description": "(p,d,q) order (optional)."},
        {"name": "seasonal_order", "type": "tuple", "description": "(P,D,Q,m) order (optional)."},
        {"name": "p", "type": "int", "description": "AR order (default: 1)."},
        {"name": "d", "type": "int", "description": "Differencing order (default: 1)."},
        {"name": "q", "type": "int", "description": "MA order (default: 1)."},
        {"name": "P", "type": "int", "description": "Seasonal AR order (default: 0)."},
        {"name": "D", "type": "int", "description": "Seasonal differencing order (default: 0)."},
        {"name": "Q", "type": "int", "description": "Seasonal MA order (default: 0)."},
        {"name": "seasonality", "type": "int", "description": "Seasonal period (m)."},
        {"name": "trend", "type": "str", "description": "Trend spec (default: c)."},
        {"name": "alpha", "type": "float", "description": "CI alpha (default: 0.05)."},
    ]

    @property
    def name(self) -> str:
        return "sarima"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        return self._forecast_sarimax(series, horizon, seasonality, params, seasonal=True, exog_future=exog_future, **kwargs)


# Backward compatibility wrappers
def forecast_ses(series: np.ndarray, fh: int, alpha: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    res = ForecastRegistry.get("ses").forecast(pd.Series(series), fh, 0, {"alpha": alpha})
    # Note: original returned fitted values as 3rd element. New interface doesn't strictly require it but we can add to metadata if needed.
    # For now, returning None for fitted to match signature
    return res.forecast, res.params_used, None

def forecast_holt(series: np.ndarray, fh: int, damped: bool = True) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    res = ForecastRegistry.get("holt").forecast(pd.Series(series), fh, 0, {"damped": damped})
    return res.forecast, res.params_used, None

def forecast_holt_winters(series: np.ndarray, fh: int, m: int, seasonal: str = 'add') -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    method_name = "holt_winters_add" if seasonal == 'add' else "holt_winters_mul"
    res = ForecastRegistry.get(method_name).forecast(pd.Series(series), fh, m, {"damped": False}) # Original wrapper didn't expose damped param?
    return res.forecast, res.params_used, None

def forecast_sarimax(
    series: np.ndarray,
    fh: int,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str = 'c',
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    ci_alpha: Optional[float] = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any], Optional[Tuple[np.ndarray, np.ndarray]]]:
    
    # Determine if it's ARIMA or SARIMA based on seasonal_order
    method_name = "sarima" if sum(seasonal_order) > 0 else "arima"
    
    params = {
        "order": order,
        "seasonal_order": seasonal_order,
        "trend": trend,
        "alpha": ci_alpha
    }
    
    res = ForecastRegistry.get(method_name).forecast(
        pd.Series(series), 
        fh, 
        seasonal_order[3] if len(seasonal_order) > 3 else 0, 
        params, 
        exog_used=exog_used, 
        exog_future=exog_future
    )
    
    return res.forecast, res.params_used, res.ci_values

