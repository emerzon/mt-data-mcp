from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..common import build_ci_diagnostics as _build_ci_diagnostics
from ..interface import CancelToken, ForecastMethod, ForecastResult, ProgressReporter, TrainResult
from ..forecast_registry import ForecastRegistry

try:
    import importlib.util as _importlib_util
    _HAS_SKTIME = _importlib_util.find_spec('sktime') is not None
except Exception:
    _HAS_SKTIME = False

_SKTIME_IMPORT_ERROR = "sktime is not installed; install it to enable sktime-based forecast methods."
logger = logging.getLogger(__name__)

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

    # ------------------------------------------------------------------
    # Train / predict lifecycle
    # ------------------------------------------------------------------

    @property
    def supports_training(self) -> bool:
        return True

    @property
    def training_category(self):
        return "fast"

    @property
    def train_supports_cancel(self) -> bool:
        return True

    @property
    def train_supports_progress(self) -> bool:
        return True

    def train(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        progress_callback=None,
        cancel_token: Optional[CancelToken] = None,
        exog=None,
        **kwargs,
    ) -> TrainResult:
        if not _HAS_SKTIME:
            raise RuntimeError(_SKTIME_IMPORT_ERROR)
        reporter = ProgressReporter(progress_callback, total_steps=3)
        reporter.stage(0, "Preparing sktime training data", force=True)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        y = series.copy()
        if not isinstance(y.index, pd.RangeIndex) and getattr(y.index, "freq", None) is None:
            try:
                y.index.freq = pd.infer_freq(y.index)
            except Exception:
                pass
        if not isinstance(y.index, pd.RangeIndex) and getattr(y.index, "freq", None) is None:
            y = y.reset_index(drop=True)

        estimator = self._get_estimator(seasonality, params)
        reporter.stage(1, "Fitting sktime estimator", force=True)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        X = exog if exog is not None else params.get('exog_used')
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, index=y.index)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if X is not None:
                estimator.fit(y, X=X)
            else:
                estimator.fit(y)

        artifact_bytes = self.serialize_artifact(estimator)
        reporter.stage(3, "Training complete", force=True)
        return TrainResult(
            artifact_bytes=artifact_bytes,
            params_used={"seasonality": seasonality, **params},
        )

    def predict_with_model(
        self,
        model,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        exog_future=None,
        **kwargs,
    ) -> ForecastResult:
        if not _HAS_SKTIME:
            raise RuntimeError(_SKTIME_IMPORT_ERROR)

        estimator = model  # deserialized sktime estimator
        fh = np.arange(1, horizon + 1)

        X_future = kwargs.get('exog_future')
        if X_future is None:
            X_future = exog_future if exog_future is not None else params.get('exog_future')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if X_future is not None:
                y_pred = estimator.predict(fh=fh, X=X_future)
            else:
                y_pred = estimator.predict(fh=fh)

        if isinstance(y_pred, pd.Series):
            f_vals = y_pred.values
        elif isinstance(y_pred, pd.DataFrame):
            f_vals = y_pred.iloc[:, 0].values
        else:
            f_vals = np.array(y_pred)

        return ForecastResult(
            forecast=f_vals,
            params_used={"seasonality": seasonality, **params},
        )

    def forecast(  # noqa: C901
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
        if X is None:
            X = params.get('exog_used')
        X_future = kwargs.get('exog_future')
        if X_future is None:
            X_future = exog_future if exog_future is not None else params.get('exog_future')
        
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
            metadata: Dict[str, Any] = {}
            ci_alpha = kwargs.get('ci_alpha', params.get('ci_alpha'))
            if ci_alpha is not None:
                ci_alpha_value: Optional[float] = None
                try:
                    # sktime predict_interval returns DataFrame with MultiIndex columns (coverage, lower/upper)
                    # coverage is 1 - alpha? No, coverage is e.g. 0.9 for alpha 0.1
                    ci_alpha_value = float(ci_alpha)
                    coverage = 1.0 - ci_alpha_value
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
                    interval_columns = [str(col) for col in cols]
                    
                    for col in cols:
                        # col is a tuple
                        if isinstance(col, tuple) and len(col) >= 3:
                            try:
                                cov = float(col[1])
                            except (TypeError, ValueError):
                                continue
                            direction = col[2]
                            if np.isclose(cov, coverage, atol=1e-3, rtol=0.0):
                                if direction == 'lower':
                                    lo_vals = intervals[col].values
                                elif direction == 'upper':
                                    hi_vals = intervals[col].values
                    
                    if lo_vals is not None and hi_vals is not None:
                        ci_values = (lo_vals.astype(float), hi_vals.astype(float))
                        metadata = _build_ci_diagnostics(
                            provider=self.name,
                            requested=True,
                            available=True,
                            status="available",
                            alpha=ci_alpha_value,
                            coverage=coverage,
                        )
                    else:
                        warning_text = (
                            f"Sktime {self.name} did not return matching interval columns for coverage "
                            f"{coverage:g}; returning point forecast only."
                        )
                        logger.warning("%s Columns=%s", warning_text, interval_columns)
                        metadata = _build_ci_diagnostics(
                            provider=self.name,
                            requested=True,
                            available=False,
                            status="unavailable",
                            alpha=ci_alpha_value,
                            coverage=coverage,
                            warning=warning_text,
                            interval_columns=interval_columns,
                        )

                except Exception as ex:
                    warning_text = (
                        f"Sktime {self.name} confidence interval extraction failed: {ex}. "
                        "Returning point forecast only."
                    )
                    logger.warning("%s", warning_text)
                    metadata = _build_ci_diagnostics(
                        provider=self.name,
                        requested=True,
                        available=False,
                        status="error",
                        alpha=ci_alpha_value,
                        warning=warning_text,
                        error=str(ex),
                        error_type=type(ex).__name__,
                    )

            return ForecastResult(
                forecast=f_vals,
                ci_values=ci_values,
                params_used={"seasonality": seasonality, **params},
                metadata=metadata or None,
            )
            
        except Exception as ex:
            raise RuntimeError(f"Sktime {self.name} error: {ex}")

@ForecastRegistry.register("sktime")
class GenericSktimeMethod(SktimeMethod):
    """Generic wrapper for any Sktime estimator."""

    CAPABILITY_EXECUTION_LIBRARY = "sktime"
    CAPABILITY_SELECTOR_KEY = "estimator"
    CAPABILITY_SELECTOR_MODE = "dotted_path"

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

    CAPABILITY_SELECTOR_VALUE = "sktime.forecasting.theta.ThetaForecaster"
    CAPABILITY_ALIASES = ("ThetaForecaster", "theta")

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

    CAPABILITY_SELECTOR_VALUE = "sktime.forecasting.naive.NaiveForecaster"
    CAPABILITY_ALIASES = ("NaiveForecaster", "naive")

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

    CAPABILITY_SELECTOR_VALUE = "sktime.forecasting.ets.AutoETS"
    CAPABILITY_ALIASES = ("AutoETS", "autoets")

    @property
    def name(self) -> str:
        return "skt_autoets"

    def _get_estimator(self, seasonality: int, params: Dict[str, Any]):
        p = dict(params or {})
        p.setdefault("estimator", "sktime.forecasting.ets.AutoETS")
        return super()._get_estimator(seasonality, p)
