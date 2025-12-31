from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import warnings
import numpy as np
import pandas as pd

from ..interface import ForecastMethod, ForecastResult
from ..registry import ForecastRegistry

try:
    import importlib.util as _importlib_util
    _HAS_SKTIME = _importlib_util.find_spec('sktime') is not None
except Exception:
    _HAS_SKTIME = False

_SKTIME_IMPORT_ERROR = "sktime is not installed; install it to enable sktime-based forecast methods."

class SktimeMethod(ForecastMethod):
    """Base class for Sktime methods."""
    
    @property
    def category(self) -> str:
        return "sktime"
        
    @property
    def required_packages(self) -> List[str]:
        return ["sktime"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": True}

    def _get_estimator(self, seasonality: int, params: Dict[str, Any]):
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
        if not _HAS_SKTIME:
            raise RuntimeError(_SKTIME_IMPORT_ERROR)
        try:
            import sktime
            from sktime.forecasting.base import BaseForecaster
        except ImportError as ex:
            raise RuntimeError(f"Failed to import sktime: {ex}")

        # Prepare data
        # Sktime expects pandas Series/DataFrame with PeriodIndex or DatetimeIndex
        # Our series usually has DatetimeIndex from the engine
        
        y = series.copy()
        # Ensure frequency is set if missing (DatetimeIndex / PeriodIndex only).
        if not isinstance(y.index, pd.RangeIndex) and getattr(y.index, "freq", None) is None:
            try:
                y.index.freq = pd.infer_freq(y.index)
            except Exception:
                pass
                
        # If inference failed, we might need to use integer index or period index
        # For simplicity, let's assume the engine provides a good index or we fallback to RangeIndex
        if not isinstance(y.index, pd.RangeIndex) and getattr(y.index, "freq", None) is None:
            y = y.reset_index(drop=True)

        estimator = self._get_estimator(seasonality, params)
        
        # Exogenous variables
        X = kwargs.get('exog_used')
        X_future = kwargs.get('exog_future')
        
        # Convert numpy exog to pandas if needed
        if isinstance(X, np.ndarray):
             X = pd.DataFrame(X, index=y.index)
        
        fh = np.arange(1, horizon + 1)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if X is not None:
                    estimator.fit(y, X=X)
                else:
                    estimator.fit(y)
                    
                if X_future is not None:
                    # Ensure X_future has correct index
                    # This is tricky without knowing the future dates exactly here if using DatetimeIndex
                    # But engine passes exog_future as numpy array usually.
                    # We might need to reconstruct index.
                    # For now, let's assume if X was numpy, X_future is too, and we need to match length
                    if isinstance(X_future, np.ndarray):
                        # We need to create an index for X_future
                        if isinstance(y.index, pd.RangeIndex):
                            start = y.index[-1] + 1
                            idx = pd.RangeIndex(start, start + horizon)
                            X_future = pd.DataFrame(X_future, index=idx)
                        elif isinstance(y.index, pd.DatetimeIndex):
                            freq = y.index.freq or pd.infer_freq(y.index)
                            if freq is not None:
                                try:
                                    offset = pd.tseries.frequencies.to_offset(freq)
                                    start = y.index[-1] + offset
                                    idx = pd.date_range(start=start, periods=horizon, freq=offset)
                                    X_future = pd.DataFrame(X_future, index=idx)
                                except Exception:
                                    pass
                        elif isinstance(y.index, pd.PeriodIndex):
                            try:
                                freq = y.index.freq
                                if freq is not None:
                                    start = y.index[-1] + 1
                                    idx = pd.period_range(start=start, periods=horizon, freq=freq)
                                    X_future = pd.DataFrame(X_future, index=idx)
                            except Exception:
                                pass
                        
                    y_pred = estimator.predict(fh=fh, X=X_future)
                else:
                    y_pred = estimator.predict(fh=fh)
            
            # Extract values
            if isinstance(y_pred, pd.Series):
                f_vals = y_pred.values
            elif isinstance(y_pred, pd.DataFrame):
                f_vals = y_pred.iloc[:, 0].values
            else:
                f_vals = np.array(y_pred)
                
            # CI extraction
            ci_values = None
            ci_alpha = kwargs.get('ci_alpha')
            if ci_alpha is not None:
                try:
                    # sktime predict_interval returns DataFrame with MultiIndex columns (coverage, lower/upper)
                    # coverage is 1 - alpha? No, coverage is e.g. 0.9 for alpha 0.1
                    coverage = 1.0 - float(ci_alpha)
                    intervals = estimator.predict_interval(fh=fh, X=X_future, coverage=coverage)
                    # intervals columns: (var_name, coverage, 'lower'/'upper')
                    # We assume univariate
                    cols = intervals.columns
                    # We want the coverage we asked for
                    # cols levels: 0=var, 1=coverage, 2=direction
                    
                    # Flatten or find correct cols
                    # Example col: ('y', 0.9, 'lower')
                    
                    # Let's try to find them dynamically
                    lo_vals = None
                    hi_vals = None
                    
                    for col in cols:
                        # col is a tuple
                        if len(col) >= 3:
                            cov = col[1]
                            direction = col[2]
                            if abs(cov - coverage) < 1e-6:
                                if direction == 'lower':
                                    lo_vals = intervals[col].values
                                elif direction == 'upper':
                                    hi_vals = intervals[col].values
                    
                    if lo_vals is not None and hi_vals is not None:
                         ci_values = (lo_vals.astype(float), hi_vals.astype(float))
                         
                except Exception:
                    pass

            return ForecastResult(
                forecast=f_vals,
                ci_values=ci_values,
                params_used={"seasonality": seasonality, **params}
            )
            
        except Exception as ex:
            raise RuntimeError(f"Sktime {self.name} error: {ex}")

@ForecastRegistry.register("sktime")
class GenericSktimeMethod(SktimeMethod):
    """Generic wrapper for any Sktime estimator."""

    PARAMS: List[Dict[str, Any]] = [
        {"name": "estimator", "type": "str", "description": "Fully qualified class path."},
        {"name": "estimator_params", "type": "dict", "description": "Constructor kwargs for estimator."},
        {"name": "seasonality", "type": "int|null", "description": "Seasonal period (sp) if supported."},
    ]
    
    @property
    def name(self) -> str:
        return "sktime"
        
    def _get_estimator(self, seasonality: int, params: Dict[str, Any]):
        if not _HAS_SKTIME:
            raise RuntimeError(_SKTIME_IMPORT_ERROR)
        estimator_path = params.get('estimator')
        if not estimator_path:
            # Default to a robust, commonly available estimator.
            estimator_path = "sktime.forecasting.theta.ThetaForecaster"
            
        # Import dynamically
        try:
            module_path, class_name = estimator_path.rsplit('.', 1)
            import importlib
            module = importlib.import_module(module_path)
            estimator_cls = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
             raise ValueError(f"Could not import sktime estimator '{estimator_path}': {e}")
             
        # Filter params
        import inspect
        try:
            sig = inspect.signature(estimator_cls)
            valid_params = set(sig.parameters.keys())
        except ValueError:
            valid_params = set()
            
        est_params = {k: v for k, v in params.items() if k in valid_params}
        
        # Inject seasonality (sp) if applicable
        if 'sp' in valid_params and 'sp' not in est_params:
            est_params['sp'] = max(1, seasonality)
            
        return estimator_cls(**est_params)


@ForecastRegistry.register("skt_theta")
class SktThetaMethod(GenericSktimeMethod):
    """Alias for `sktime` using `sktime.forecasting.theta.ThetaForecaster`."""

    @property
    def name(self) -> str:
        return "skt_theta"

    def _get_estimator(self, seasonality: int, params: Dict[str, Any]):
        p = dict(params or {})
        p.setdefault("estimator", "sktime.forecasting.theta.ThetaForecaster")
        return super()._get_estimator(seasonality, p)


@ForecastRegistry.register("skt_naive")
class SktNaiveMethod(GenericSktimeMethod):
    """Alias for `sktime` using `sktime.forecasting.naive.NaiveForecaster`."""

    @property
    def name(self) -> str:
        return "skt_naive"

    def _get_estimator(self, seasonality: int, params: Dict[str, Any]):
        p = dict(params or {})
        p.setdefault("estimator", "sktime.forecasting.naive.NaiveForecaster")
        p.setdefault("strategy", "last")
        return super()._get_estimator(seasonality, p)


@ForecastRegistry.register("skt_autoets")
class SktAutoETSMethod(GenericSktimeMethod):
    """Alias for `sktime` using `sktime.forecasting.ets.AutoETS`."""

    @property
    def name(self) -> str:
        return "skt_autoets"

    def _get_estimator(self, seasonality: int, params: Dict[str, Any]):
        p = dict(params or {})
        p.setdefault("estimator", "sktime.forecasting.ets.AutoETS")
        return super()._get_estimator(seasonality, p)
