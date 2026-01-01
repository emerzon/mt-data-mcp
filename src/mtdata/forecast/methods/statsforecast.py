from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import json
import warnings
import numpy as np
import pandas as pd
import inspect

from ..interface import ForecastMethod, ForecastResult
from ..registry import ForecastRegistry


def _coerce_param_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _coerce_param_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        coerced = [_coerce_param_value(v) for v in value]
        return tuple(coerced) if isinstance(value, tuple) else coerced
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return _coerce_param_value(json.loads(s))
            except Exception:
                pass
        try:
            return int(s)
        except (TypeError, ValueError):
            pass
        try:
            return float(s)
        except (TypeError, ValueError):
            pass
    return value


def _coerce_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        out[k] = _coerce_param_value(v)
    return out

class StatsForecastMethod(ForecastMethod):
    """Base class for StatsForecast methods."""
    
    @property
    def category(self) -> str:
        return "statsforecast"
        
    @property
    def required_packages(self) -> List[str]:
        return ["statsforecast"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": True}

    def _get_model(self, seasonality: int, params: Dict[str, Any]):
        raise NotImplementedError

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"pkg_resources is deprecated as an API\..*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"Deprecated call to `pkg_resources\.declare_namespace\(.*\)`\..*",
                    category=DeprecationWarning,
                )
                from statsforecast import StatsForecast  # type: ignore
        except ImportError as ex:
            raise RuntimeError(f"Failed to import statsforecast: {ex}") from ex

        # Build single-series training dataframe
        from ..common import _create_training_dataframes, _extract_forecast_values
        
        exog_used = kwargs.get('exog_used')
        exog_future_arr = kwargs.get('exog_future')
        
        Y_df, X_df, Xf_df = _create_training_dataframes(series.values, horizon, exog_used, exog_future_arr)

        clean_params = _coerce_params(params)
        model = self._get_model(seasonality, clean_params)
        
        ci_alpha = kwargs.get('ci_alpha')
        level = None
        if ci_alpha is not None:
            level = [int((1 - float(ci_alpha)) * 100)]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sf = StatsForecast(models=[model], freq=1) # freq=1 for integer index fallback
                if X_df is not None:
                    sf.fit(Y_df, X_df=X_df)
                else:
                    sf.fit(Y_df)
                
                if Xf_df is not None:
                    Yf = sf.predict(h=int(horizon), X_df=Xf_df, level=level)
                else:
                    Yf = sf.predict(h=int(horizon), level=level)
            
            try:
                Yf = Yf[Yf['unique_id'] == 'ts']
            except Exception:
                pass
            
            # Extract values
            f_vals = _extract_forecast_values(Yf, horizon, f"StatsForecast {self.name}")
            
            # CI extraction
            ci_values = None
            if level:
                lev_val = level[0]
                cols = Yf.columns
                lo_col = None
                hi_col = None
                for c in cols:
                    if str(c).endswith(f'-lo-{lev_val}'):
                        lo_col = c
                    elif str(c).endswith(f'-hi-{lev_val}'):
                        hi_col = c
                
                if lo_col and hi_col:
                    lo_vals = Yf[lo_col].values
                    hi_vals = Yf[hi_col].values
                    # Ensure length matches horizon
                    if len(lo_vals) >= horizon:
                        lo_vals = lo_vals[:horizon]
                        hi_vals = hi_vals[:horizon]
                    else:
                        # Pad if needed (unlikely for forecast)
                        pad_width = horizon - len(lo_vals)
                        lo_vals = np.pad(lo_vals, (0, pad_width), mode='edge')
                        hi_vals = np.pad(hi_vals, (0, pad_width), mode='edge')
                        
                    ci_values = np.vstack([
                        lo_vals.astype(float),
                        hi_vals.astype(float),
                    ])

            # Filter out internal context params and build clean params_used
            internal_keys = {'symbol', 'timeframe', 'as_of', 'exog_used', 'exog_future', 'seasonality'}
            clean_params = {k: v for k, v in clean_params.items() if k not in internal_keys}
            params_used = {"seasonality": seasonality, **clean_params}
            
            return ForecastResult(
                forecast=f_vals,
                ci_values=ci_values,
                params_used=params_used
            )
            
        except Exception as ex:
            raise RuntimeError(f"StatsForecast {self.name} error: {ex}")

@ForecastRegistry.register("statsforecast")
class GenericStatsForecastMethod(StatsForecastMethod):
    """Generic wrapper for any StatsForecast model."""

    PARAMS: List[Dict[str, Any]] = [
        {"name": "model_name", "type": "str", "description": "StatsForecast model class name."},
        {"name": "season_length", "type": "int", "description": "Season length (auto if omitted)."},
    ]
    
    @property
    def name(self) -> str:
        return "statsforecast"
        
    def _get_model(self, seasonality: int, params: Dict[str, Any]):
        model_name = params.get('model_name') or params.get('model') or 'autoarima'

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"pkg_resources is deprecated as an API\..*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"Deprecated call to `pkg_resources\.declare_namespace\(.*\)`\..*",
                    category=DeprecationWarning,
                )
                from statsforecast import models  # type: ignore
        except ImportError as ex:
            raise RuntimeError(f"Failed to import statsforecast models: {ex}") from ex
        
        # Handle case-insensitive lookup (prefer classes to avoid function collisions)
        public_names = [m for m in dir(models) if not m.startswith('_')]
        class_names = [m for m in public_names if inspect.isclass(getattr(models, m, None))]
        available = {m.lower(): m for m in class_names}
        target = str(model_name).lower()
        
        if target not in available:
            raise ValueError(f"Unknown StatsForecast model: {model_name}. Available: {list(available.keys())}")
            
        model_cls = getattr(models, available[target])
        
        # Filter params for the model constructor
        try:
            sig = inspect.signature(model_cls)
            valid_params = set(sig.parameters.keys())
        except ValueError:
            valid_params = set()
            
        model_params = {k: v for k, v in params.items() if k in valid_params}
        
        # Inject seasonality if applicable
        if 'season_length' in valid_params and 'season_length' not in model_params:
            model_params['season_length'] = max(1, seasonality)
        if 'period' in valid_params and 'period' not in model_params:
            model_params['period'] = max(1, seasonality)
        if 'periods' in valid_params and 'periods' not in model_params:
            model_params['periods'] = [max(1, seasonality)]
            
        return model_cls(**model_params)

_SF_MODEL_CLASS_NAMES: Tuple[str, ...] = (
    # Keep this list lightweight (no import-time statsforecast dependency).
    # These correspond to common classes under statsforecast.models.
    "ADIDA",
    "ARCH",
    "ARIMA",
    "AutoARIMA",
    "AutoCES",
    "AutoETS",
    "AutoMFLES",
    "AutoRegressive",
    "AutoTBATS",
    "AutoTheta",
    "ConstantModel",
    "CrostonClassic",
    "CrostonOptimized",
    "CrostonSBA",
    "DynamicOptimizedTheta",
    "DynamicTheta",
    "GARCH",
    "HistoricAverage",
    "Holt",
    "HoltWinters",
    "IMAPA",
    "MFLES",
    "MSTL",
    "NaNModel",
    "Naive",
    "OptimizedTheta",
    "RandomWalkWithDrift",
    "SeasonalExponentialSmoothing",
    "SeasonalExponentialSmoothingOptimized",
    "SeasonalNaive",
    "SeasonalWindowAverage",
    "SimpleExponentialSmoothing",
    "SimpleExponentialSmoothingOptimized",
    "SklearnModel",
    "TBATS",
    "TSB",
    "Theta",
    "WindowAverage",
    "ZeroModel",
)


def _build_sf_alias_class(model_name: str, alias: str):
    class _StatsForecastAlias(GenericStatsForecastMethod):
        @property
        def name(self) -> str:
            return alias

        def _get_model(self, seasonality: int, params: Dict[str, Any]):
            p = dict(params or {})
            p["model_name"] = model_name
            return super()._get_model(seasonality, p)

    _StatsForecastAlias.__name__ = f"SF_{model_name}"
    _StatsForecastAlias.__qualname__ = _StatsForecastAlias.__name__
    _StatsForecastAlias.__doc__ = (
        f"StatsForecast {model_name} (alias; equivalent to method='statsforecast' "
        f"with params.model_name='{model_name}')."
    )
    return _StatsForecastAlias


def _register_statsforecast_aliases() -> None:
    for model_name in _SF_MODEL_CLASS_NAMES:
        alias = f"sf_{str(model_name).lower()}"
        try:
            cls = _build_sf_alias_class(model_name, alias)
            ForecastRegistry.register(alias)(cls)
        except Exception:
            continue
    # Compatibility alias: users often try sf_ets (AutoETS)
    try:
        cls = _build_sf_alias_class("AutoETS", "sf_ets")
        ForecastRegistry.register("sf_ets")(cls)
    except Exception:
        pass


_register_statsforecast_aliases()

# Backward compatibility wrapper
def forecast_statsforecast(
    *,
    method: str,
    series: np.ndarray,
    fh: int,
    timeframe: str,
    m_eff: int,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[list] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Legacy wrapper for StatsForecast methods."""
    forecaster = ForecastRegistry.get(method)
    # Convert numpy series to pandas Series for the interface
    # (Though our implementation converts it back to numpy for _create_training_dataframes, 
    # the interface expects pd.Series)
    s_pd = pd.Series(series)
    
    result = forecaster.forecast(
        series=s_pd,
        horizon=fh,
        seasonality=m_eff,
        params={},
        exog_used=exog_used,
        exog_future=exog_future
    )
    return result.forecast, result.params_used
