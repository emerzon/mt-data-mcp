"""
Method adapters for different forecasting approaches.

Provides a clean adapter pattern to isolate method implementations.
"""

from typing import Any, Dict, Tuple
import numpy as np

# Import implementations
from .methods.classical import (
    forecast_naive as _naive_impl,
    forecast_drift as _drift_impl, 
    forecast_seasonal_naive as _snaive_impl,
    forecast_theta as _theta_impl,
    forecast_fourier_ols as _fourier_impl,
)
from .methods.ets_arima import (
    forecast_ses as _ses_impl,
    forecast_holt as _holt_impl,
    forecast_holt_winters as _hw_impl,
    forecast_sarimax as _sarimax_impl,
)
from .methods.neural import forecast_neural as _neural_impl
from .methods.statsforecast import forecast_statsforecast as _sf_impl
from .methods.mlforecast import forecast_mlf_rf as _mlf_rf_impl, forecast_mlf_lightgbm as _mlf_lgb_impl
from .methods.pretrained import (
    forecast_chronos_bolt as _chronos_bolt_impl,
    forecast_timesfm as _timesfm_impl,
    forecast_lag_llama as _lag_llama_impl,
)
from .monte_carlo import simulate_gbm_mc as _simulate_gbm_mc, simulate_hmm_mc as _simulate_hmm_mc


class ForecastMethodAdapter:
    """Base adapter for forecast methods."""
    
    def __init__(self, method: str):
        self.method = method.lower().strip()
    
    def execute(
        self, 
        series: np.ndarray,
        fh: int,
        params: Dict[str, Any],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute the forecast method."""
        raise NotImplementedError
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parameters."""
        return params


class ClassicalMethodAdapter(ForecastMethodAdapter):
    """Adapter for classical forecasting methods."""
    
    def execute(
        self, 
        series: np.ndarray,
        fh: int, 
        params: Dict[str, Any],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        if self.method == 'naive':
            return _naive_impl(series, fh)
            
        elif self.method == 'drift':
            n = params.get('n')
            return _drift_impl(series, fh, n)
            
        elif self.method == 'seasonal_naive':
            m = params.get('seasonality', kwargs.get('m', 0))
            return _snaive_impl(series, fh, m)
            
        elif self.method == 'theta':
            alpha = float(params.get('alpha', 0.2))
            return _theta_impl(series, fh, alpha)
            
        elif self.method == 'fourier_ols':
            m = params.get('seasonality', kwargs.get('m', 0))
            K = int(params.get('K', 3))
            trend = bool(params.get('trend', True))
            return _fourier_impl(series, fh, m, K, trend)
            
        else:
            raise ValueError(f"Unknown classical method: {self.method}")


class ETSMethodAdapter(ForecastMethodAdapter):
    """Adapter for ETS/Holt-Winters methods."""
    
    def execute(
        self, 
        series: np.ndarray,
        fh: int,
        params: Dict[str, Any], 
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        if self.method == 'ses':
            alpha = params.get('alpha')
            return _ses_impl(series, fh, alpha)
            
        elif self.method == 'holt':
            damped = bool(params.get('damped', True))
            return _holt_impl(series, fh, damped)
            
        elif self.method in ('holt_winters_add', 'holt_winters_mul'):
            m = params.get('seasonality', kwargs.get('m', 0))
            seasonal = 'add' if self.method == 'holt_winters_add' else 'mul'
            return _hw_impl(series, fh, m, seasonal)
            
        elif self.method in ('arima', 'sarima'):
            return self._execute_arima(series, fh, params, kwargs)
            
        else:
            raise ValueError(f"Unknown ETS method: {self.method}")
    
    def _execute_arima(
        self, 
        series: np.ndarray, 
        fh: int, 
        params: Dict[str, Any], 
        kwargs: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute ARIMA/SARIMA methods."""
        p = int(params.get('p', 1))
        d = int(params.get('d', 0)) 
        q = int(params.get('q', 1))
        trend = params.get('trend', 'c')
        
        if self.method == 'sarima':
            P = int(params.get('P', 0))
            D = int(params.get('D', 1))
            Q = int(params.get('Q', 0))
            m = params.get('seasonality', kwargs.get('m', 0))
            return _sarimax_impl(
                series=series, fh=fh, p=p, d=d, q=q, P=P, D=D, Q=Q, 
                m=m, trend=trend, exog_used=kwargs.get('exog_used'),
                exog_future=kwargs.get('exog_future')
            )
        else:
            return _sarimax_impl(
                series=series, fh=fh, p=p, d=d, q=q, P=0, D=0, Q=0,
                m=0, trend=trend, exog_used=kwargs.get('exog_used'),
                exog_future=kwargs.get('exog_future')
            )


class MonteCarloMethodAdapter(ForecastMethodAdapter):
    """Adapter for Monte Carlo simulation methods."""
    
    def execute(
        self,
        series: np.ndarray,
        fh: int,
        params: Dict[str, Any],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        if self.method == 'mc_gbm':
            n_sims = int(params.get('n_sims', 500))
            seed = int(params.get('seed', 42))
            paths, mu, sigma = _simulate_gbm_mc(series, fh, n_sims, seed)
            f_vals = np.mean(paths, axis=0)
            return f_vals, {"mu": mu, "sigma": sigma, "n_sims": n_sims}
            
        elif self.method == 'hmm_mc':
            n_states = int(params.get('n_states', 2))
            n_sims = int(params.get('n_sims', 500))
            seed = int(params.get('seed', 42))
            paths, model_params = _simulate_hmm_mc(series, fh, n_states, n_sims, seed)
            f_vals = np.mean(paths, axis=0)
            return f_vals, {"n_states": n_states, "n_sims": n_sims, **model_params}
            
        else:
            raise ValueError(f"Unknown Monte Carlo method: {self.method}")


class NeuralMethodAdapter(ForecastMethodAdapter):
    """Adapter for neural network methods."""
    
    def execute(
        self,
        series: np.ndarray, 
        fh: int,
        params: Dict[str, Any],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        return _neural_impl(
            method=self.method,
            series=series,
            fh=fh,
            timeframe=kwargs.get('timeframe', 'H1'),
            m_eff=kwargs.get('m', 0),
            params=params,
            exog_used=kwargs.get('exog_used'),
            exog_future=kwargs.get('exog_future'),
            future_times=kwargs.get('future_times')
        )


class StatisticalMethodAdapter(ForecastMethodAdapter):
    """Adapter for StatsForecast methods."""
    
    def execute(
        self,
        series: np.ndarray,
        fh: int, 
        params: Dict[str, Any],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        return _sf_impl(
            method=self.method,
            series=series,
            fh=fh,
            timeframe=kwargs.get('timeframe', 'H1'),
            m_eff=int(params.get('seasonality', kwargs.get('m', 0)) or kwargs.get('m', 0)),
            exog_used=kwargs.get('exog_used'),
            exog_future=kwargs.get('exog_future'),
            future_times=kwargs.get('future_times')
        )


class MLMethodAdapter(ForecastMethodAdapter):
    """Adapter for machine learning methods."""
    
    def execute(
        self,
        series: np.ndarray,
        fh: int,
        params: Dict[str, Any], 
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        # Prepare lags
        lags_in = params.get('lags', 'auto')
        m = kwargs.get('m', 0)
        if lags_in == 'auto' or lags_in is None:
            base_lags = [1, 2, 3, 4, 5]
            if m and m > 0:
                base_lags += [m]
            lags = sorted(set([int(abs(x)) for x in base_lags if int(abs(x)) > 0]))
        else:
            try:
                lags = [int(v) for v in lags_in]
            except Exception:
                lags = [1, 2, 3, 4, 5]
        
        roll = str(params.get('rolling_agg', 'mean')).lower() if params.get('rolling_agg') is not None else None
        
        if self.method == 'mlf_rf':
            return _mlf_rf_impl(
                series=series, fh=fh, timeframe=kwargs.get('timeframe', 'H1'),
                lags=lags, rolling_agg=roll,
                exog_used=kwargs.get('exog_used'),
                exog_future=kwargs.get('exog_future'),
                future_times=kwargs.get('future_times')
            )
            
        elif self.method == 'mlf_lightgbm':
            n_estimators = int(params.get('n_estimators', 200))
            lr = float(params.get('learning_rate', 0.05))
            num_leaves = int(params.get('num_leaves', 31))
            max_depth = int(params.get('max_depth', -1))
            
            return _mlf_lgb_impl(
                series=series, fh=fh, timeframe=kwargs.get('timeframe', 'H1'),
                lags=lags, rolling_agg=roll,
                n_estimators=n_estimators, learning_rate=lr,
                num_leaves=num_leaves, max_depth=max_depth,
                exog_used=kwargs.get('exog_used'),
                exog_future=kwargs.get('exog_future'),
                future_times=kwargs.get('future_times')
            )
            
        else:
            raise ValueError(f"Unknown ML method: {self.method}")


class PretrainedMethodAdapter(ForecastMethodAdapter):
    """Adapter for pre-trained model methods."""
    
    def execute(
        self,
        series: np.ndarray,
        fh: int,
        params: Dict[str, Any],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        if self.method in ('chronos_bolt', 'chronos2'):
            f_vals, f_quantiles, params_used, error = _chronos_bolt_impl(
                series=series, fh=fh, params=params, n=len(series)
            )
            if error:
                raise RuntimeError(error)
            return f_vals, params_used
            
        elif self.method == 'timesfm':
            model_name = params.get('model_name') or "google/timesfm-1.0-200m"
            if not model_name:
                raise ValueError(f"{self.method} requires params.model_name with a valid HF repo id")

            return _timesfm_impl(
                series=series, fh=fh, params=params, n=fh
            )

        elif self.method == 'lag_llama':
            return _lag_llama_impl(
                series=series, fh=fh, params=params, n=fh
            )
            
        else:
            raise ValueError(f"Unknown pretrained method: {self.method}")


class SktimeMethodAdapter(ForecastMethodAdapter):
    """Adapter for sktime BaseForecaster models via dynamic import.

    Usage (params):
      - estimator: fully qualified class path, e.g.,
          'sktime.forecasting.naive.NaiveForecaster'
          'sktime.forecasting.theta.ThetaForecaster'
          'sktime.forecasting.arima.ARIMA'
          'sktime.forecasting.ets.AutoETS'
      - estimator_params: dict passed to estimator constructor
      - seasonality (optional): if provided and estimator supports 'sp', inject sp
    """

    def execute(
        self,
        series: np.ndarray,
        fh: int,
        params: Dict[str, Any],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            import importlib
            import importlib.util as _importlib_util  # noqa: F401
            import pandas as pd
        except Exception as e:
            raise RuntimeError(f"Required dependencies missing for sktime adapter: {e}")

        # Resolve estimator (support convenience aliases)
        est_path = str(params.get('estimator') or '').strip()
        if not est_path:
            alias_map = {
                'sktime': ('sktime.forecasting.naive.NaiveForecaster', {'strategy': 'last'}),
                'skt_naive': ('sktime.forecasting.naive.NaiveForecaster', {'strategy': 'last'}),
                'skt_snaive': ('sktime.forecasting.naive.NaiveForecaster', {'strategy': 'last'}),  # sp injected below
                'skt_theta': ('sktime.forecasting.theta.ThetaForecaster', {}),
                'skt_autoets': ('sktime.forecasting.ets.AutoETS', {}),
                'skt_arima': ('sktime.forecasting.arima.ARIMA', {}),
                'skt_autoarima': ('sktime.forecasting.arima.AutoARIMA', {}),
            }
            cls, defaults = alias_map.get(self.method, alias_map['sktime'])
            est_path = cls
            # Merge defaults unless user provided overrides
            ep = params.get('estimator_params')
            if not ep:
                params['estimator_params'] = defaults
        try:
            module_name, class_name = est_path.rsplit('.', 1)
            mod = importlib.import_module(module_name)
            Estimator = getattr(mod, class_name)
        except Exception as e:
            raise RuntimeError(f"Failed to import sktime estimator '{est_path}': {e}")

        est_kwargs_in = params.get('estimator_params')
        if isinstance(est_kwargs_in, dict):
            est_kwargs = dict(est_kwargs_in)
        else:
            # Allow JSON string from UI
            try:
                import json as _json
                est_kwargs = dict(_json.loads(str(est_kwargs_in))) if est_kwargs_in else {}
            except Exception:
                est_kwargs = {}

        # Inject seasonal period if available and not explicitly set
        m = params.get('seasonality') or kwargs.get('m')
        if m and 'sp' not in est_kwargs:
            est_kwargs['sp'] = int(m)

        # Build sktime forecaster
        try:
            forecaster = Estimator(**est_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to construct {Estimator.__name__} with params {est_kwargs}: {e}")

        # Prepare y and fh
        y = pd.Series(np.asarray(series, dtype=float))
        fh_idx = list(range(1, int(fh) + 1))

        # Optional exogenous variables
        X = kwargs.get('exog_used')
        X_future = kwargs.get('exog_future')
        try:
            forecaster.fit(y, X=X)
            y_pred = forecaster.predict(fh_idx, X=X_future)
        except Exception as e:
            raise RuntimeError(f"sktime forecasting failed: {e}")

        f_vals = np.asarray(y_pred, dtype=float)
        return f_vals, {
            'estimator': est_path,
            'estimator_params': est_kwargs,
        }


def get_method_adapter(method: str) -> ForecastMethodAdapter:
    """Factory function to get the appropriate adapter for a method."""
    method_l = method.lower().strip()
    
    # Classical methods
    if method_l in ('naive', 'drift', 'seasonal_naive', 'theta', 'fourier_ols'):
        return ClassicalMethodAdapter(method_l)
    
    # ETS/ARIMA methods
    elif method_l in ('ses', 'holt', 'holt_winters_add', 'holt_winters_mul', 'arima', 'sarima'):
        return ETSMethodAdapter(method_l)
    
    # Monte Carlo methods
    elif method_l in ('mc_gbm', 'hmm_mc'):
        return MonteCarloMethodAdapter(method_l)
    
    # Neural methods
    elif method_l in ('nhits', 'nbeatsx', 'tft', 'patchtst'):
        return NeuralMethodAdapter(method_l)
    
    # Statistical methods
    elif method_l.startswith('sf_'):
        return StatisticalMethodAdapter(method_l)
    
    # ML methods
    elif method_l in ('mlf_rf', 'mlf_lightgbm'):
        return MLMethodAdapter(method_l)
    
    # Pretrained methods
    elif method_l in ('chronos_bolt', 'chronos2', 'timesfm', 'lag_llama'):
        return PretrainedMethodAdapter(method_l)
    
    # sktime generic adapter
    elif method_l in ('sktime',):
        return SktimeMethodAdapter(method_l)
    
    else:
        raise ValueError(f"Unknown forecast method: {method}")


__all__ = [
    'ForecastMethodAdapter',
    'ClassicalMethodAdapter', 
    'ETSMethodAdapter',
    'MonteCarloMethodAdapter',
    'NeuralMethodAdapter',
    'StatisticalMethodAdapter',
    'MLMethodAdapter',
    'PretrainedMethodAdapter',
    'SktimeMethodAdapter',
    'get_method_adapter'
]
