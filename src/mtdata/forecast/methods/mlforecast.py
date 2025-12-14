from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import warnings
import numpy as np
import pandas as pd

from ..interface import ForecastMethod, ForecastResult
from ..registry import ForecastRegistry

class MLForecastMethod(ForecastMethod):
    """Base class for MLForecast methods."""
    
    @property
    def category(self) -> str:
        return "machine_learning"
        
    @property
    def required_packages(self) -> List[str]:
        return ["mlforecast"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": True}

    def _get_model(self, params: Dict[str, Any]):
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
            from mlforecast import MLForecast
            import pandas as pd
        except ImportError as ex:
            raise RuntimeError(f"Failed to import mlforecast: {ex}")

        # Build single-series training dataframe
        from ..common import _create_training_dataframes, _extract_forecast_values
        
        exog_used = kwargs.get('exog_used')
        exog_future_arr = kwargs.get('exog_future')
        
        Y_df, X_df, Xf_df = _create_training_dataframes(series.values, horizon, exog_used, exog_future_arr)

        model = self._get_model(params)
        lags = params.get('lags')
        if not lags:
            # Provide a safe default lag set so the method works out-of-the-box.
            base = int(seasonality) if seasonality and int(seasonality) > 0 else 24
            max_lag = int(min(30, max(1, base)))
            lags = list(range(1, max_lag + 1))
            params = dict(params or {})
            params["lags"] = lags
        rolling_agg = params.get('rolling_agg')
        
        try:
            # Pass lags to constructor
            # Use freq=1 because _create_training_dataframes uses integer index
            mlf = MLForecast(models=[model], freq=1, lags=lags)
            
            # rolling_agg support requires window_ops or similar, disabling for now to fix basic functionality
            # if rolling_agg in {'mean', 'min', 'max', 'std'} and lags:
            #     for w in sorted(set([x for x in lags if x and x > 1])):
            #         mlf = mlf.add_rolling_windows(rolling_features={rolling_agg: [w]})
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlf.fit(Y_df)
            
            if Xf_df is not None:
                Yf = mlf.predict(h=int(horizon), X_df=Xf_df)
            else:
                Yf = mlf.predict(h=int(horizon))
            
            try:
                Yf = Yf[Yf['unique_id'] == 'ts']
            except Exception:
                pass
            
            f_vals = _extract_forecast_values(Yf, horizon, self.name)
            
            # Filter out internal context params
            internal_keys = {'symbol', 'timeframe', 'as_of', 'exog_used', 'exog_future'}
            clean_params = {k: v for k, v in params.items() if k not in internal_keys}
            
            return ForecastResult(
                forecast=f_vals,
                ci_values=None,
                params_used=clean_params
            )
            
        except Exception as ex:
            raise RuntimeError(f"{self.name} error: {ex}")

@ForecastRegistry.register("mlf_rf")
class MLFRandomForest(MLForecastMethod):
    @property
    def name(self) -> str:
        return "mlf_rf"
        
    @property
    def required_packages(self) -> List[str]:
        return ["mlforecast", "scikit-learn"]

    def _get_model(self, params: Dict[str, Any]):
        from sklearn.ensemble import RandomForestRegressor
        n_estimators = int(params.get('n_estimators', 200))
        max_depth = params.get('max_depth')
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

@ForecastRegistry.register("mlf_lightgbm")
class MLFLightGBM(MLForecastMethod):
    @property
    def name(self) -> str:
        return "mlf_lightgbm"
        
    @property
    def required_packages(self) -> List[str]:
        return ["mlforecast", "lightgbm"]

    def _get_model(self, params: Dict[str, Any]):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=int(params.get('n_estimators', 200)),
            learning_rate=float(params.get('learning_rate', 0.05)),
            num_leaves=int(params.get('num_leaves', 31)),
            max_depth=int(params.get('max_depth', -1)),
            random_state=42
        )

@ForecastRegistry.register("mlforecast")
class GenericMLForecastMethod(MLForecastMethod):
    """Generic wrapper for any MLForecast compatible model."""
    
    @property
    def name(self) -> str:
        return "mlforecast"
        
    def _get_model(self, params: Dict[str, Any]):
        model_path = params.get('model')
        if not model_path:
            raise ValueError("GenericMLForecastMethod requires 'model' (dotted path) in params")
            
        # Import dynamically
        try:
            module_path, class_name = model_path.rsplit('.', 1)
            import importlib
            module = importlib.import_module(module_path)
            model_cls = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
             raise ValueError(f"Could not import ML model '{model_path}': {e}")
             
        # Filter params
        import inspect
        try:
            sig = inspect.signature(model_cls)
            valid_params = set(sig.parameters.keys())
        except ValueError:
            valid_params = set()
            
        model_params = {k: v for k, v in params.items() if k in valid_params}
        
        return model_cls(**model_params)

# Backward compatibility wrappers
def forecast_mlf_rf(
    *,
    series: np.ndarray,
    fh: int,
    timeframe: str,
    lags: Optional[list] = None,
    rolling_agg: Optional[str] = None,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[list] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    forecaster = ForecastRegistry.get("mlf_rf")
    s_pd = pd.Series(series)
    params = {'lags': lags, 'rolling_agg': rolling_agg}
    result = forecaster.forecast(s_pd, fh, 0, params, exog_used=exog_used, exog_future=exog_future)
    return result.forecast, result.params_used

def forecast_mlf_lightgbm(
    *,
    series: np.ndarray,
    fh: int,
    timeframe: str,
    lags: Optional[list] = None,
    rolling_agg: Optional[str] = None,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[list] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    forecaster = ForecastRegistry.get("mlf_lightgbm")
    s_pd = pd.Series(series)
    params = {
        'lags': lags, 
        'rolling_agg': rolling_agg,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth
    }
    result = forecaster.forecast(s_pd, fh, 0, params, exog_used=exog_used, exog_future=exog_future)
    return result.forecast, result.params_used
