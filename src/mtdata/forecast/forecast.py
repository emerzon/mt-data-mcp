from typing import Any, Dict, Optional, Literal
import os

# Adopt upcoming StatsForecast DataFrame format to avoid repeated warnings
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

from ..core.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..core.schema import ForecastMethodLiteral, TimeframeLiteral, DenoiseSpec
from .exceptions import ForecastError
from .common import fetch_history as _fetch_history
from .forecast_methods import get_forecast_methods_data
from .forecast_preprocessing import _create_dimred_reducer

_FORECAST_METHODS_EXPORT = get_forecast_methods_data

# Removed unused imports of specific method implementations
# Logic is now handled by forecast_engine via registry

# Optional availability flags and lazy imports following server logic
# (Kept for backward compatibility if anything relies on these flags, though mostly unused now)
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing as _SES, ExponentialSmoothing as _ETS  # type: ignore
    _SM_ETS_AVAILABLE = True
except Exception:
    _SM_ETS_AVAILABLE = False
    _SES = _ETS = None  # type: ignore
# ... (other availability checks can remain or be cleaned up, keeping for safety) ...


def forecast(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    method: ForecastMethodLiteral = "theta",
    horizon: int = 12,
    lookback: Optional[int] = None,
    as_of: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    ci_alpha: Optional[float] = 0.05,
    quantity: Literal['price','return','volatility'] = 'price',  # type: ignore
    target: Literal['price','return'] = 'price',  # deprecated in favor of quantity for modeling scale
    denoise: Optional[DenoiseSpec] = None,
    # Feature engineering for exogenous/multivariate models
    features: Optional[Dict[str, Any]] = None,
    # Optional dimensionality reduction across feature columns (overrides features.dimred_* if set)
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    # Custom target specification (base column/alias, transform, and horizon aggregation)
    target_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fast forecasts for the next `horizon` bars using lightweight methods.
    Parameters: symbol, timeframe, method, horizon, lookback?, as_of?, params?, ci_alpha?, target, denoise?

    Methods: naive, seasonal_naive, drift, theta, fourier_ols, ses, holt, holt_winters_add, holt_winters_mul, arima, sarima.
    
    - `params`: method-specific settings; use `seasonality` inside params when needed (auto if omitted).
    - `target`: 'price' or 'return' (log-return). Price forecasts operate on close prices.
    - `ci_alpha`: confidence level (e.g., 0.05). Set to null to disable intervals.
    - `features`: Dict or "key=value" string for feature engineering.
        - `include`: List of columns to include (e.g., "open,high").
        - `future_covariates`: List of date-based features to generate for future horizon.
          Supported tokens: `hour`, `dow` (day of week), `month`, `day`, `doy` (day of year), 
          `week`, `minute`, `mod` (minute of day), `is_weekend`, `is_holiday`.
          For `is_holiday`, specify `country` in features (default: US).
        - `dimred_method`: Dimensionality reduction method (e.g., "pca").
    """
    try:
        method_l = str(method).lower().strip()
        quantity_l = str(quantity).lower().strip()

        if quantity_l == 'volatility' or method_l.startswith('vol_'):
            from .volatility import forecast_volatility
            return forecast_volatility(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                method=method,
                params=params,
                as_of=as_of
            )

        from .forecast_engine import forecast_engine

        return forecast_engine(
            symbol=symbol,
            timeframe=timeframe,
            method=method,
            horizon=horizon,
            lookback=lookback,
            as_of=as_of,
            params=params,
            ci_alpha=ci_alpha,
            quantity=quantity,
            target=target,
            denoise=denoise,
            features=features,
            dimred_method=dimred_method,
            dimred_params=dimred_params,
            target_spec=target_spec,
        )
    except ForecastError:
        raise
    except Exception as exc:
        raise ForecastError(str(exc)) from exc
