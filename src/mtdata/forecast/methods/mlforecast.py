from __future__ import annotations

import importlib
import inspect
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..interface import ForecastMethod, ForecastResult
from ..registry import ForecastRegistry

_GENERIC_MLFORECAST_ALLOWED_MODELS = {
    "catboost.CatBoostRegressor",
    "lightgbm.LGBMRegressor",
    "sklearn.ensemble.AdaBoostRegressor",
    "sklearn.ensemble.ExtraTreesRegressor",
    "sklearn.ensemble.GradientBoostingRegressor",
    "sklearn.ensemble.HistGradientBoostingRegressor",
    "sklearn.ensemble.RandomForestRegressor",
    "sklearn.linear_model.ElasticNet",
    "sklearn.linear_model.Lasso",
    "sklearn.linear_model.LinearRegression",
    "sklearn.linear_model.Ridge",
    "sklearn.neighbors.KNeighborsRegressor",
    "sklearn.neural_network.MLPRegressor",
    "sklearn.svm.LinearSVR",
    "sklearn.svm.SVR",
    "sklearn.tree.DecisionTreeRegressor",
    "sklearn.tree.ExtraTreeRegressor",
    "xgboost.XGBRegressor",
}


def _import_model_module(module_path: str):
    try:
        return importlib.import_module(module_path)
    except ImportError:
        module = sys.modules.get(module_path)
        if module is None or getattr(module, "__spec__", None) is not None:
            raise
        return module

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
        return {"price": True, "return": True, "volatility": True, "ci": False}

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
        except ImportError as ex:
            raise RuntimeError(f"Failed to import mlforecast: {ex}")

        # Build single-series training dataframe
        from ..common import _create_training_dataframes, _extract_forecast_values
        
        exog_used = kwargs.get('exog_used')
        if exog_used is None:
            exog_used = params.get('exog_used')
        exog_future_arr = kwargs.get('exog_future')
        if exog_future_arr is None:
            exog_future_arr = exog_future if exog_future is not None else params.get('exog_future')
        
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
        rolling_agg = params.get("rolling_agg")
        
        try:
            if rolling_agg is not None and str(rolling_agg).strip():
                raise RuntimeError(
                    "rolling_agg is not supported for mlforecast methods."
                )
            # Pass lags to constructor
            # Use freq=1 because _create_training_dataframes uses integer index
            mlf = MLForecast(models=[model], freq=1, lags=lags)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if X_df is not None:
                    mlf.fit(Y_df, X_df=X_df)
                else:
                    mlf.fit(Y_df)
            
            if Xf_df is not None:
                Yf = mlf.predict(h=int(horizon), X_df=Xf_df)
            else:
                Yf = mlf.predict(h=int(horizon))
            
            if 'unique_id' not in Yf.columns:
                raise RuntimeError("mlforecast output missing unique_id column")
            Yf = Yf[Yf['unique_id'] == 'ts']
            if Yf.empty:
                raise RuntimeError("mlforecast output missing rows for unique_id='ts'")
            
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
    CAPABILITY_EXECUTION_LIBRARY = "native"
    CAPABILITY_CONCEPT = "rf"
    CAPABILITY_DISPLAY_NAME = "RandomForestRegressor"
    CAPABILITY_ALIASES = ("randomforest", "random_forest")

    PARAMS: List[Dict[str, Any]] = [
        {"name": "n_estimators", "type": "int", "description": "Number of trees (default: 200)."},
        {"name": "max_depth", "type": "int|null", "description": "Maximum depth (default: None)."},
        {"name": "lags", "type": "list", "description": "Lag features to use (auto if omitted)."},
    ]

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
    CAPABILITY_EXECUTION_LIBRARY = "native"
    CAPABILITY_CONCEPT = "lightgbm"
    CAPABILITY_DISPLAY_NAME = "LGBMRegressor"
    CAPABILITY_ALIASES = ("lgbm", "light_gbm")

    PARAMS: List[Dict[str, Any]] = [
        {"name": "n_estimators", "type": "int", "description": "Number of trees (default: 200)."},
        {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 0.05)."},
        {"name": "num_leaves", "type": "int", "description": "Number of leaves (default: 31)."},
        {"name": "max_depth", "type": "int", "description": "Maximum depth (default: -1)."},
        {"name": "lags", "type": "list", "description": "Lag features to use (auto if omitted)."},
    ]

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

    CAPABILITY_EXECUTION_LIBRARY = "mlforecast"
    CAPABILITY_SELECTOR_KEY = "model"
    CAPABILITY_SELECTOR_MODE = "dotted_class"

    PARAMS: List[Dict[str, Any]] = [
        {"name": "model", "type": "str", "description": "Approved dotted class path for ML model."},
        {"name": "lags", "type": "list", "description": "Lag features to use (auto if omitted)."},
    ]
    
    @property
    def name(self) -> str:
        return "mlforecast"
        
    def _get_model(self, params: Dict[str, Any]):
        model_path = str(params.get('model') or "").strip()
        if not model_path:
            raise ValueError("GenericMLForecastMethod requires 'model' (dotted path) in params")

        if model_path not in _GENERIC_MLFORECAST_ALLOWED_MODELS:
            allowed = ", ".join(sorted(_GENERIC_MLFORECAST_ALLOWED_MODELS))
            raise ValueError(
                f"Model '{model_path}' is not allowed for GenericMLForecastMethod. "
                f"Allowed models: {allowed}"
            )

        try:
            module_path, class_name = model_path.rsplit('.', 1)
            module = _import_model_module(module_path)
            model_cls = getattr(module, class_name)
            if not inspect.isclass(model_cls):
                raise TypeError(f"{model_path} did not resolve to a class")
        except (TypeError, ValueError, ImportError, AttributeError) as e:
            raise ValueError(f"Could not import ML model '{model_path}': {e}")

        try:
            sig = inspect.signature(model_cls)
            valid_params = set(sig.parameters.keys())
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Could not inspect constructor for ML model '{model_path}': {e}"
            )

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
