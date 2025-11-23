from typing import Any, Dict, Optional, List, Literal
from datetime import datetime
import os
import json
import math
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import warnings

# Adopt upcoming StatsForecast DataFrame format to avoid repeated warnings
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

from ..core.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..utils.mt5 import _mt5_epoch_to_utc, _mt5_copy_rates_from, _ensure_symbol_ready
from ..utils.utils import (
    _parse_start_datetime as _parse_start_datetime_util,
    _format_time_minimal as _format_time_minimal_util,
    _format_time_minimal_local as _format_time_minimal_local_util,
    _use_client_tz as _use_client_tz_util,
)
from ..utils.indicators import _parse_ti_specs as _parse_ti_specs_util, _apply_ta_indicators as _apply_ta_indicators_util
from ..utils.denoise import _apply_denoise, normalize_denoise_spec as _normalize_denoise_spec
from .common import (
    parse_kv_or_json as _parse_kv_or_json,
    fetch_history as _fetch_history,
)
# Removed invalid import: from .registry import get_forecast_methods_data
from .helpers import (
    default_seasonality_period as _default_seasonality_period,
    next_times_from_last as _next_times_from_last,
    pd_freq_from_timeframe as _pd_freq_from_timeframe,
)
from .target_builder import build_target_series, aggregate_horizon_target

# Removed unused imports of specific method implementations
# Logic is now handled by forecast_engine via registry

# Local fallbacks for typing aliases used in signatures (avoid import cycle)
try:
    from ..core.server import ForecastMethodLiteral, TimeframeLiteral, DenoiseSpec  # type: ignore
except Exception:  # runtime fallback
    ForecastMethodLiteral = str  # type: ignore
    TimeframeLiteral = str  # type: ignore
    DenoiseSpec = Dict[str, Any]  # type: ignore

# Optional availability flags and lazy imports following server logic
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing as _SES, ExponentialSmoothing as _ETS  # type: ignore
    _SM_ETS_AVAILABLE = True
except Exception:
    _SM_ETS_AVAILABLE = False
    _SES = _ETS = None  # type: ignore
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX  # type: ignore
    _SM_SARIMAX_AVAILABLE = True
except Exception:
    _SM_SARIMAX_AVAILABLE = False
    _SARIMAX = None  # type: ignore
try:
    import importlib.util as _importlib_util  # type: ignore
    _NF_AVAILABLE = _importlib_util.find_spec("neuralforecast") is not None
except Exception:
    _NF_AVAILABLE = False
try:
    import importlib.util as _importlib_util2  # type: ignore
    _SF_AVAILABLE = _importlib_util2.find_spec("statsforecast") is not None
except Exception:
    _SF_AVAILABLE = False
try:
    import importlib.util as _importlib_util3  # type: ignore
    _MLF_AVAILABLE = _importlib_util3.find_spec("mlforecast") is not None
except Exception:
    _MLF_AVAILABLE = False
try:
    import importlib.util as _importlib_util4  # type: ignore
    _LGB_AVAILABLE = _importlib_util4.find_spec("lightgbm") is not None
except Exception:
    _LGB_AVAILABLE = False
try:
    import importlib.util as _importlib_util5  # type: ignore
    # Chronos available only when native package is installed
    _CHRONOS_AVAILABLE = (_importlib_util5.find_spec("chronos") is not None)
except Exception:
    _CHRONOS_AVAILABLE = False
try:
    import importlib.util as _importlib_util6  # type: ignore
    # Consider TimesFM available only when the native package is installed
    _TIMESFM_AVAILABLE = (_importlib_util6.find_spec("timesfm") is not None)
except Exception:
    _TIMESFM_AVAILABLE = False
try:
    import importlib.util as _importlib_util7  # type: ignore
    _LAG_LLAMA_AVAILABLE = (_importlib_util7.find_spec("lag_llama") is not None)
except Exception:
    _LAG_LLAMA_AVAILABLE = False
try:
    import importlib.util as _importlib_util8  # type: ignore
    _MOIRAI_AVAILABLE = (_importlib_util8.find_spec("uni2ts") is not None)
except Exception:
    _MOIRAI_AVAILABLE = False


def get_forecast_methods_data() -> Dict[str, Any]:
    """Return metadata about available forecast methods and their requirements."""
    methods: List[Dict[str, Any]] = []

    def add(method: str, description: str, params: List[Dict[str, Any]], requires: List[str], supports: Dict[str, bool]) -> None:
        available = True
        reqs = list(requires)
        if method in ("ses", "holt", "holt_winters_add", "holt_winters_mul") and not _SM_ETS_AVAILABLE:
            available = False; reqs.append("statsmodels")
        if method in ("arima", "sarima") and not _SM_SARIMAX_AVAILABLE:
            available = False; reqs.append("statsmodels")
        if method in ("nhits", "nbeatsx", "tft", "patchtst") and not _NF_AVAILABLE:
            available = False; reqs.append("neuralforecast[torch]")
        if method.startswith("sf_") and not _SF_AVAILABLE:
            available = False; reqs.append("statsforecast")
        if method == "mlf_rf" and not _MLF_AVAILABLE:
            available = False; reqs.append("mlforecast, scikit-learn")
        if method == "mlf_lightgbm" and (not _MLF_AVAILABLE or not _LGB_AVAILABLE):
            available = False; reqs.append("mlforecast, lightgbm")
        if method == "chronos_bolt" and not _CHRONOS_AVAILABLE:
            available = False; reqs.append("chronos-forecasting")
        if method == "timesfm" and not _TIMESFM_AVAILABLE:
            available = False; reqs.append("timesfm")
        if method == "lag_llama" and not _LAG_LLAMA_AVAILABLE:
            available = False; reqs.append("lag-llama, gluonts, torch")
        if method == "moirai" and not _MOIRAI_AVAILABLE:
            available = False; reqs.append("uni2ts, torch")
        # ensemble is not wired yet in this repo
        if method == "ensemble":
            available = False; reqs.append("not implemented")
        methods.append({
            "method": method,
            "available": bool(available),
            "requires": sorted(set(reqs)),
            "description": description,
            "params": params,
            "supports": supports,
        })

    # Baselines
    add("naive", "Repeat last value forward.", [], [], {"price": True, "return": True, "ci": True})
    add("seasonal_naive", "Repeat last seasonal value (period m).", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Seasonal period. Auto by timeframe if omitted."},
    ], [], {"price": True, "return": True, "ci": True})
    add("drift", "Line from first to last with constant slope.", [], [], {"price": True, "return": True, "ci": True})
    add("theta", "Theta method (SES + trend).", [
        {"name": "alpha", "type": "float", "default": 0.2, "description": "SES smoothing factor for theta blend."},
    ], [], {"price": True, "return": True, "ci": True})
    add("fourier_ols", "Fourier series regression with optional trend.", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Seasonal period m."},
        {"name": "K", "type": "int", "default": 3, "description": "Number of Fourier harmonics."},
        {"name": "trend", "type": "bool", "default": True, "description": "Include linear trend."},
    ], [], {"price": True, "return": True, "ci": True})

    # ETS / Holt-Winters / ARIMA family
    add("ses", "Simple Exponential Smoothing (statsmodels).", [
        {"name": "alpha", "type": "float|null", "default": None, "description": "If None, estimated by MLE."},
    ], [], {"price": True, "return": True, "ci": True})
    add("holt", "Holt’s linear trend (statsmodels).", [
        {"name": "damped", "type": "bool", "default": True, "description": "Use damped trend variant."},
    ], [], {"price": True, "return": True, "ci": True})
    add("holt_winters_add", "Additive Holt-Winters (statsmodels).", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Seasonal period m."},
    ], [], {"price": True, "return": True, "ci": True})
    add("holt_winters_mul", "Multiplicative Holt-Winters (statsmodels).", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Seasonal period m."},
    ], [], {"price": True, "return": True, "ci": True})
    add("arima", "Non-seasonal ARIMA via SARIMAX.", [
        {"name": "p", "type": "int", "default": 1}, {"name": "d", "type": "int", "default": 0}, {"name": "q", "type": "int", "default": 1},
        {"name": "trend", "type": "str", "default": "c"},
    ], [], {"price": True, "return": True, "ci": True})
    add("sarima", "Seasonal ARIMA via SARIMAX.", [
        {"name": "p", "type": "int", "default": 1}, {"name": "d", "type": "int", "default": 0}, {"name": "q", "type": "int", "default": 1},
        {"name": "P", "type": "int", "default": 0}, {"name": "D", "type": "int", "default": 1}, {"name": "Q", "type": "int", "default": 0},
        {"name": "seasonality", "type": "int", "default": None},
        {"name": "trend", "type": "str", "default": "c"},
    ], [], {"price": True, "return": True, "ci": True})

    # Monte Carlo
    add("mc_gbm", "Monte Carlo with GBM calibrated from log-returns.", [
        {"name": "n_sims", "type": "int", "default": 500},
        {"name": "seed", "type": "int", "default": 42},
    ], [], {"price": True, "return": True, "ci": True})
    add("hmm_mc", "Monte Carlo with Gaussian HMM regimes over returns.", [
        {"name": "n_states", "type": "int", "default": 2},
        {"name": "n_sims", "type": "int", "default": 500},
        {"name": "seed", "type": "int", "default": 42},
    ], [], {"price": True, "return": True, "ci": True})

    # Optional ecosystems
    add("nhits", "NeuralForecast NHITS (PyTorch).", [], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": False})
    add("nbeatsx", "NeuralForecast NBEATSx (PyTorch).", [], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": False})
    add("tft", "NeuralForecast TFT (PyTorch).", [], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": False})
    add("patchtst", "NeuralForecast PatchTST (PyTorch).", [], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": False})
    add("sf_autoarima", "StatsForecast AutoARIMA.", [], ["statsforecast"], {"price": True, "return": True, "ci": False})
    add("sf_theta", "StatsForecast Theta.", [], ["statsforecast"], {"price": True, "return": True, "ci": False})
    add("sf_autoets", "StatsForecast AutoETS.", [], ["statsforecast"], {"price": True, "return": True, "ci": False})
    add("sf_seasonalnaive", "StatsForecast SeasonalNaive.", [], ["statsforecast"], {"price": True, "return": True, "ci": False})
    add("mlf_rf", "MLForecast RandomForest.", [], ["mlforecast", "scikit-learn"], {"price": True, "return": True, "ci": False})
    add("mlf_lightgbm", "MLForecast LightGBM.", [], ["mlforecast", "lightgbm"], {"price": True, "return": True, "ci": False})
    add("chronos_bolt", "Chronos-2 foundation model (alias: chronos2). Upstream model supports cross-learning, multivariate, and covariates — adapter currently uses univariate target only.", [
        {"name": "model_name", "type": "str", "description": "Hugging Face model id (default: amazon/chronos-2)"},
        {"name": "context_length", "type": "int", "description": "Context window length (auto if omitted)"},
        {"name": "quantiles", "type": "list", "description": "Quantile levels to return (default: [0.5])"},
        {"name": "device_map", "type": "str", "description": "Device placement (default: auto)"},
    ], ["chronos-forecasting>=2.0.0"], {"price": True, "return": True, "ci": False})
    add("chronos2", "Chronos-2 foundation model (preferred name; same as chronos_bolt). Upstream model supports cross-learning, multivariate, and covariates — adapter currently uses univariate target only.", [
        {"name": "model_name", "type": "str", "description": "Hugging Face model id (default: amazon/chronos-2)"},
        {"name": "context_length", "type": "int", "description": "Context window length (auto if omitted)"},
        {"name": "quantiles", "type": "list", "description": "Quantile levels to return (default: [0.5])"},
        {"name": "device_map", "type": "str", "description": "Device placement (default: auto)"},
    ], ["chronos-forecasting>=2.0.0"], {"price": True, "return": True, "ci": False})
    add("timesfm", "Google TimesFM (native package).", [], ["timesfm"], {"price": True, "return": True, "ci": False})
    add("lag_llama", "Lag-Llama (native estimator).", [], ["lag-llama", "gluonts", "torch"], {"price": True, "return": True, "ci": False})
    add("moirai", "Salesforce Moirai (uni2ts).", [
        {"name": "variant", "type": "str", "description": "Model variant (default: moirai-1.1-R-large, other options: moirai-1.1-R-small, moirai-1.1-R-base, 1.0-R-small, 1.0-R-base, 1.0-R-large)"},
        {"name": "context_length", "type": "int", "description": "Context window length (auto if omitted)"},
        {"name": "device", "type": "str", "description": "Torch device string (e.g., 'cpu', 'cuda:0')"},
        {"name": "quantiles", "type": "list", "description": "Quantile levels to return (default: [0.5])"},
        {"name": "do_mean", "type": "bool", "description": "Return mean estimate if available (default: True)"},
        {"name": "do_median", "type": "bool", "description": "Fallback to median if mean unavailable (default: True)"},
    ], ["uni2ts", "torch"], {"price": True, "return": True, "ci": False})

    # GluonTS Torch quick-train models
    add("gt_deepar", "GluonTS DeepAR (quick train)", [
        {"name": "context_length", "type": "int", "description": "Input window length (default: min(64,n))"},
        {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
        {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
        {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
        {"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"},
    ], ["gluonts", "torch"], {"price": True, "return": True, "ci": True})

    add("gt_sfeedforward", "GluonTS SimpleFeedForward (quick train)", [
        {"name": "context_length", "type": "int", "description": "Input window length (default: min(64,n))"},
        {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
        {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
        {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
        {"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"},
    ], ["gluonts", "torch"], {"price": True, "return": True, "ci": True})

    add("gt_prophet", "GluonTS Prophet wrapper", [
        {"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"},
        {"name": "prophet_params", "type": "dict", "description": "Passed to ProphetPredictor (growth, seasonality_mode, ...)"},
    ], ["gluonts", "prophet"], {"price": True, "return": True, "ci": True})

    add("gt_tft", "GluonTS Temporal Fusion Transformer (quick train)", [
        {"name": "context_length", "type": "int", "description": "Input window length (default: min(128,n))"},
        {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
        {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
        {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
        {"name": "hidden_size", "type": "int", "description": "Model width (default: 64)"},
        {"name": "dropout", "type": "float", "description": "Dropout (default: 0.1)"},
        {"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"},
    ], ["gluonts", "torch"], {"price": True, "return": True, "ci": True})

    add("gt_wavenet", "GluonTS WaveNet (quick train)", [
        {"name": "context_length", "type": "int", "description": "Input window length (default: min(128,n))"},
        {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
        {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
        {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
        {"name": "dilation_depth", "type": "int", "description": "Dilation depth (default: 5)"},
        {"name": "num_stacks", "type": "int", "description": "WaveNet stacks (default: 1)"},
        {"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"},
    ], ["gluonts", "torch"], {"price": True, "return": True, "ci": True})

    add("gt_deepnpts", "GluonTS DeepNPTS (quick train)", [
        {"name": "context_length", "type": "int", "description": "Input window length (default: min(128,n))"},
        {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
        {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
        {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
        {"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"},
    ], ["gluonts", "torch"], {"price": True, "return": True, "ci": True})

    add("gt_mqf2", "GluonTS MQF2 (quick train, quantiles)", [
        {"name": "context_length", "type": "int", "description": "Input window length (default: min(128,n))"},
        {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
        {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
        {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
        {"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"},
        {"name": "quantiles", "type": "list", "description": "Quantiles to return (e.g., [0.05,0.5,0.95])"},
    ], ["gluonts", "torch", "cpflows"], {"price": True, "return": True, "ci": True})
    add("gt_npts", "GluonTS NPTS (Non-Parametric Time Series)", [
        {"name": "context_length", "type": "int", "description": "Input window length (default: min(128,n))"},
        {"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"},
    ], ["gluonts"], {"price": True, "return": True, "ci": True})
    add("ensemble", "Adaptive ensemble (average, Bayesian model averaging, stacking).",
        [{"name": "methods", "type": "list", "description": "Component methods (default: naive,theta,fourier_ols)"},
         {"name": "mode", "type": "str", "description": "average|bma|stacking (default: average)"},
         {"name": "weights", "type": "list", "description": "Manual weights for average mode"},
         {"name": "cv_points", "type": "int", "description": "Walk-forward anchors for weighting"},
         {"name": "min_train_size", "type": "int", "description": "Minimum observations per anchor"},
         {"name": "method_params", "type": "dict", "description": "Per-method parameter overrides"},
         {"name": "expose_components", "type": "bool", "description": "Include component forecasts (default: True)"}],
        [], {"price": True, "return": True, "ci": False})

    return {"success": True, "schema_version": 1, "methods": methods}


def _default_seasonality_period(timeframe: str) -> int:
    from .common import default_seasonality
    return int(default_seasonality(timeframe))


def _next_times_from_last(last_epoch: float, tf_secs: int, horizon: int) -> List[float]:
    from .common import next_times_from_last
    return next_times_from_last(last_epoch, tf_secs, horizon)


def _pd_freq_from_timeframe(tf: str) -> str:
    from .common import pd_freq_from_timeframe
    return pd_freq_from_timeframe(tf)


_FORECAST_METHODS = (
    "naive",
    "seasonal_naive",
    "drift",
    "theta",
    "fourier_ols",
    "ses",
    "holt",
    "holt_winters_add",
    "holt_winters_mul",
    "arima",
    "sarima",
    "mc_gbm",
    "hmm_mc",
    "nhits",
    "nbeatsx",
    "tft",
    "patchtst",
    "sf_autoarima",
    "sf_theta",
    "sf_autoets",
    "sf_seasonalnaive",
    "mlf_rf",
    "mlf_lightgbm",
    "chronos_bolt",
    "chronos2",
    "timesfm",
    "lag_llama",
    "moirai",
    "gt_deepar",
    "gt_sfeedforward",
    "gt_prophet",
    "gt_tft",
    "gt_wavenet",
    "gt_deepnpts",
    "gt_mqf2",
    "gt_npts",
    "ensemble",
)


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
    """
    try:
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}

        method_l = str(method).lower().strip()
        quantity_l = str(quantity).lower().strip()
        if method_l not in _FORECAST_METHODS:
            return {"error": f"Invalid method: {method}. Valid options: {list(_FORECAST_METHODS)}"}

        # Volatility models have a dedicated endpoint; keep forecast focused on price/return
        if quantity_l == 'volatility' or method_l.startswith('vol_'):
            return {"error": "Use forecast_volatility for volatility models"}

        # Parse method params via shared helper
        from .common import parse_kv_or_json as _parse_kv_or_json  # local import to avoid cycles
        p = _parse_kv_or_json(params)
        # Prefer explicit seasonality inside params; otherwise auto by timeframe
        m = int(p.get('seasonality')) if p.get('seasonality') is not None else _default_seasonality_period(timeframe)
        if method_l == 'seasonal_naive' and (not m or m <= 0):
            return {"error": "seasonal_naive requires a positive 'seasonality' in params or auto period"}

        # Determine lookback bars to fetch (robust to string input)
        lb = None
        try:
            if lookback is not None:
                lb = int(lookback)  # CLI may pass strings; coerce
        except Exception:
            lb = None
        if lb is not None and lb > 0:
            need = int(lb) + 2
        else:
            if method_l == 'seasonal_naive':
                need = max(3 * m, int(horizon) + m + 2)
            elif method_l in ('theta', 'fourier_ols'):
                need = max(300, int(horizon) + (2 * m if m else 50))
            else:  # naive, drift and others
                need = max(100, int(horizon) + 10)

        # Fetch via shared helper (normalizes UTC time and drops live last bar)
        _info_before = mt5.symbol_info(symbol)
        try:
            df = _fetch_history(symbol, timeframe, int(need), as_of)
        except Exception as ex:
            return {"error": str(ex)}
        if len(df) < 3:
            return {"error": "Not enough closed bars to compute forecast"}

        # Optionally denoise
        base_col = 'close'
        dn_spec_used = None
        if denoise:
            try:
                _dn = _normalize_denoise_spec(denoise, default_when='pre_ti')
            except Exception:
                _dn = None
            added = _apply_denoise(df, _dn, default_when='pre_ti') if _dn else []
            dn_spec_used = _dn
            if len(added) > 0 and f"{base_col}_dn" in added:
                base_col = f"{base_col}_dn"

        # Build target series: support custom target_spec or legacy target/quantity
        t = np.arange(1, len(df) + 1, dtype=float)
        last_time = float(df['time'].iloc[-1])
        future_times = _next_times_from_last(last_time, int(tf_secs), int(horizon))

        __stage = 'target_build'
        custom_target_mode = False
        target_info: Dict[str, Any] = {}
        # Helper to resolve alias base columns
        def _alias_base(arrs: Dict[str, np.ndarray], name: str) -> Optional[np.ndarray]:
            nm = name.strip().lower()
            if nm in ('typical','tp'):
                if all(k in arrs for k in ('high','low','close')):
                    return (arrs['high'] + arrs['low'] + arrs['close']) / 3.0
                return None
            if nm in ('hl2',):
                if all(k in arrs for k in ('high','low')):
                    return (arrs['high'] + arrs['low']) / 2.0
                return None
            if nm in ('ohlc4','ha_close','haclose'):
                if all(k in arrs for k in ('open','high','low','close')):
                    return (arrs['open'] + arrs['high'] + arrs['low'] + arrs['close']) / 4.0
                return None
            return None

        # Resolve base and transform from target_spec when provided
        if target_spec and isinstance(target_spec, dict):
            custom_target_mode = True
            ts = dict(target_spec)
            # Compute indicators if requested so 'base' can reference them
            ts_inds = ts.get('indicators')
            if ts_inds:
                try:
                    specs = _parse_ti_specs_util(str(ts_inds)) if isinstance(ts_inds, str) else ts_inds
                    _apply_ta_indicators_util(df, specs, default_when='pre_ti')
                except Exception:
                    pass
            base_name = str(ts.get('base', base_col))
            # Resolve base series
            if base_name in df.columns:
                y_base = df[base_name].astype(float).to_numpy()
            else:
                # Attempt alias
                arrs = {c: df[c].astype(float).to_numpy() for c in df.columns if c in ('open','high','low','close','volume')}
                y_alias = _alias_base(arrs, base_name)
                if y_alias is None:
                    # Fallback to default base_col
                    y_base = df[base_col].astype(float).to_numpy()
                else:
                    y_base = np.asarray(y_alias, dtype=float)
            target_info['base'] = base_name
            # Transform
            transform = str(ts.get('transform', 'none')).lower()
            k_trans = int(ts.get('k', 1)) if ts.get('k') is not None else 1
            if transform in ('return','log_return','diff','pct_change'):
                # general k-step transform
                if k_trans < 1:
                    k_trans = 1
                if transform == 'log_return':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y_shift = np.roll(np.log(np.maximum(y_base, 1e-12)), k_trans)
                        series = np.log(np.maximum(y_base, 1e-12)) - y_shift
                elif transform == 'return' or transform == 'pct_change':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y_shift = np.roll(y_base, k_trans)
                        series = (y_base - y_shift) / np.where(np.abs(y_shift) > 1e-12, y_shift, 1.0)
                    if transform == 'pct_change':
                        series = 100.0 * series
                else:  # diff
                    y_shift = np.roll(y_base, k_trans)
                    series = y_base - y_shift
                # Drop first k rows for valid transform
                series = np.asarray(series[k_trans:], dtype=float)
                series = series[np.isfinite(series)]
                if series.size < 5:
                    return {"error": "Not enough data for transformed target"}
                target_info['transform'] = transform
                target_info['k'] = k_trans
            else:
                series = np.asarray(y_base, dtype=float)
                series = series[np.isfinite(series)]
                if series.size < 3:
                    return {"error": "Not enough data for target"}
                target_info['transform'] = 'none'
            # Since custom target can be any series, skip legacy price/return mapping
            use_returns = False
            origin_price = float('nan')
        else:
            # Legacy target behavior: price vs return on close
            y = df[base_col].astype(float).to_numpy()
            # Decide modeling scale for price/return
            use_returns = (quantity_l == 'return') or (str(target).lower() == 'return')
            if use_returns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    x = np.diff(np.log(np.maximum(y, 1e-12)))
                x = x[np.isfinite(x)]
                if x.size < 5:
                    return {"error": "Not enough data to compute return-based forecast"}
                series = x
                origin_price = float(y[-1])
            else:
                series = y
                origin_price = float(y[-1])

        # Ensure finite numeric series for modeling
        series = np.asarray(series, dtype=float)
        series = series[np.isfinite(series)]
        n = len(series)
        if n < 3:
            return {"error": "Series too short for forecasting"}

        # ---- Optional feature engineering for exogenous models ----
        exog_used: Optional[np.ndarray] = None
        exog_future: Optional[np.ndarray] = None
        feat_info: Dict[str, Any] = {}
        __stage = 'features_start'
        if features:
            try:
                # Accept dict, JSON string, or key=value pairs
                if isinstance(features, dict):
                    fcfg = dict(features)
                elif isinstance(features, str):
                    s = features.strip()
                    if (s.startswith('{') and s.endswith('}')):
                        try:
                            fcfg = json.loads(s)
                        except Exception:
                            # Fallback: parse colon/equals pairs inside braces
                            fcfg = {}
                            toks = [tok for tok in s.strip().strip('{}').split() if tok]
                            i = 0
                            while i < len(toks):
                                tok = toks[i].strip().strip(',')
                                if not tok:
                                    i += 1; continue
                                if '=' in tok:
                                    k, v = tok.split('=', 1)
                                    fcfg[k.strip()] = v.strip().strip(',')
                                    i += 1; continue
                                if tok.endswith(':'):
                                    key = tok[:-1].strip()
                                    val = ''
                                    if i + 1 < len(toks):
                                        val = toks[i+1].strip().strip(',')
                                        i += 2
                                    else:
                                        i += 1
                                    fcfg[key] = val
                                    continue
                                i += 1
                    else:
                        # Parse k=v or k: v pairs split on whitespace
                        fcfg = {}
                        toks = [tok for tok in s.split() if tok]
                        i = 0
                        while i < len(toks):
                            tok = toks[i].strip().strip(',')
                            if '=' in tok:
                                k, v = tok.split('=', 1)
                                fcfg[k.strip()] = v.strip()
                                i += 1; continue
                            if tok.endswith(':'):
                                key = tok[:-1].strip()
                                val = ''
                                if i + 1 < len(toks):
                                    val = toks[i+1].strip().strip(',')
                                    i += 2
                                else:
                                    i += 1
                                fcfg[key] = val
                                continue
                            i += 1
                else:
                    fcfg = {}
                include = fcfg.get('include', 'ohlcv')
                include_cols: list[str] = []
                if isinstance(include, str):
                    inc = include.strip().lower()
                    if inc == 'ohlcv':
                        for col in ('open','high','low','volume','tick_volume','real_volume'):
                            if col in df.columns:
                                include_cols.append(col)
                    else:
                        # comma/space separated list
                        toks = [tok.strip() for tok in include.replace(',', ' ').split() if tok.strip()]
                        for tok in toks:
                            if tok in df.columns and tok not in ('time','close'):
                                include_cols.append(tok)
                elif isinstance(include, (list, tuple)):
                    for tok in include:
                        s = str(tok).strip()
                        if s in df.columns and s not in ('time','close'):
                            include_cols.append(s)
                # Indicators (add columns)
                __stage = 'features_indicators'
                ind_specs = fcfg.get('indicators')
                if ind_specs:
                    try:
                        specs = _parse_ti_specs_util(str(ind_specs)) if isinstance(ind_specs, str) else ind_specs
                        _apply_ta_indicators_util(df, specs, default_when='pre_ti')
                    except Exception:
                        pass
                # Add any newly created indicator columns (heuristic: non-time, non-OHLCV)
                __stage = 'features_collect'
                ti_cols = []
                for c in df.columns:
                    if c in ('time','open','high','low','close','volume','tick_volume','real_volume'):
                        continue
                    if df[c].dtype.kind in ('f','i'):
                        ti_cols.append(c)
                # Calendar/future-known covariates (hour, dow, fourier:P)
                cal_cols: list[str] = []
                cal_train: Optional[np.ndarray] = None
                cal_future: Optional[np.ndarray] = None
                fut_cov = fcfg.get('future_covariates')
                if fut_cov:
                    tokens: list[str] = []
                    if isinstance(fut_cov, str):
                        tokens = [tok.strip() for tok in fut_cov.replace(',', ' ').split() if tok.strip()]
                    elif isinstance(fut_cov, (list, tuple)):
                        tokens = [str(tok).strip() for tok in fut_cov]
                    t_train = df['time'].astype(float).to_numpy()
                    t_future = np.asarray(future_times, dtype=float)
                    tr_list: list[np.ndarray] = []
                    tf_list: list[np.ndarray] = []
                    for tok in tokens:
                        tl = tok.lower()
                        if tl.startswith('fourier:'):
                            try:
                                per = int(tl.split(':',1)[1])
                            except Exception:
                                per = 24
                            w = 2.0 * math.pi / float(max(1, per))
                            idx_tr = np.arange(t_train.size, dtype=float)
                            idx_tf = np.arange(t_future.size, dtype=float)
                            tr_list.append(np.sin(w * idx_tr)); cal_cols.append(f'fx_sin_{per}')
                            tr_list.append(np.cos(w * idx_tr)); cal_cols.append(f'fx_cos_{per}')
                            tf_list.append(np.sin(w * idx_tf));
                            tf_list.append(np.cos(w * idx_tf));
                        elif tl in ('hour','hr'):
                            try:
                                hrs_tr = pd.to_datetime(t_train, unit='s', utc=True).hour.to_numpy()
                            except Exception:
                                hrs_tr = (np.arange(t_train.size) % 24)
                            try:
                                hrs_tf = pd.to_datetime(t_future, unit='s', utc=True).hour.to_numpy()
                            except Exception:
                                hrs_tf = (np.arange(t_future.size) % 24)
                            w = 2.0 * math.pi / 24.0
                            tr_list.append(np.sin(w * hrs_tr)); cal_cols.append('hr_sin')
                            tr_list.append(np.cos(w * hrs_tr)); cal_cols.append('hr_cos')
                            tf_list.append(np.sin(w * hrs_tf));
                            tf_list.append(np.cos(w * hrs_tf));
                        elif tl in ('dow','wday','weekday'):
                            try:
                                d_tr = pd.to_datetime(t_train, unit='s', utc=True).weekday.to_numpy()
                            except Exception:
                                d_tr = (np.arange(t_train.size) % 7)
                            try:
                                d_tf = pd.to_datetime(t_future, unit='s', utc=True).weekday.to_numpy()
                            except Exception:
                                d_tf = (np.arange(t_future.size) % 7)
                            w = 2.0 * math.pi / 7.0
                            tr_list.append(np.sin(w * d_tr)); cal_cols.append('dow_sin')
                            tr_list.append(np.cos(w * d_tr)); cal_cols.append('dow_cos')
                            tf_list.append(np.sin(w * d_tf));
                            tf_list.append(np.cos(w * d_tf));
                    if tr_list:
                        cal_train = np.vstack(tr_list).T.astype(float)
                        cal_future = np.vstack(tf_list).T.astype(float)
                sel_cols = sorted(set(include_cols + ti_cols))
                __stage = 'features_matrix'
                if sel_cols:
                    X = df[sel_cols].astype(float).copy()
                    # Fill missing values conservatively (ffill then bfill)
                    X = X.replace([np.inf, -np.inf], np.nan)
                    X = X.ffill().bfill()
                    X_arr = X.to_numpy(dtype=float)
                    # Dimensionality reduction across feature columns
                    dr_method = (fcfg.get('dimred_method') or dimred_method)
                    dr_params = fcfg.get('dimred_params') or dimred_params
                    if dr_method and str(dr_method).lower() not in ('', 'none'):
                        try:
                            reducer, _ = _create_dimred_reducer(dr_method, dr_params)
                            X_red = reducer.fit_transform(X_arr)
                            exog = np.asarray(X_red, dtype=float)
                            feat_info['dimred_method'] = str(dr_method)
                            if isinstance(dr_params, dict):
                                feat_info['dimred_params'] = dr_params
                            elif dr_params is None:
                                feat_info['dimred_params'] = {}
                            else:
                                feat_info['dimred_params'] = {"raw": str(dr_params)}
                            feat_info['dimred_n_features'] = int(exog.shape[1])
                        except Exception as _ex:
                            # Fallback to raw features on failure
                            exog = X_arr
                            feat_info['dimred_error'] = str(_ex)
                    else:
                        exog = X_arr
                    # Append calendar features
                    if cal_train is not None:
                        exog = np.hstack([exog, cal_train]) if exog.size else cal_train
                    # Align with return series if needed
                    if (quantity_l == 'return') or (str(target).lower() == 'return'):
                        exog = exog[1:]
                    # Build future exog by holding the last observed row (default policy)
                    if exog.shape[0] >= 1:
                        last_row = exog[-1]
                        exog_f = np.tile(last_row.reshape(1, -1), (int(horizon), 1))
                    else:
                        exog_f = None
                    if exog_f is not None and cal_future is not None:
                        exog_f = np.hstack([exog_f, cal_future])
                    exog_used = exog
                    exog_future = exog_f
                    feat_info['selected_columns'] = sel_cols + cal_cols
                    feat_info['n_features'] = int(exog_used.shape[1]) if exog_used is not None else 0
                else:
                    feat_info['selected_columns'] = []
            except Exception as _ex:
                feat_info['error'] = f"feature_build_error: {str(_ex)}"
                __stage = 'features_error'

        # Volatility branch: compute and return volatility metrics
        __stage = 'quantity_branch'
        if quantity_l == 'volatility':
            mt5_tf = TIMEFRAME_MAP[timeframe]
            # Use the dedicated volatility engine
            from .volatility import forecast_volatility
            return forecast_volatility(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                method=method,
                params=params,
                as_of=as_of
            )

        # Use the unified forecast engine
        from .forecast_engine import forecast_engine
        
        # Map legacy arguments to engine arguments
        engine_params = params or {}
        
        # Call engine
        result = forecast_engine(
            symbol=symbol,
            timeframe=timeframe,
            method=method,
            horizon=horizon,
            lookback=lookback,
            as_of=as_of,
            params=engine_params,
            ci_alpha=ci_alpha,
            quantity=quantity,
            target=target,
            denoise=denoise,
            features=features,
            dimred_method=dimred_method,
            dimred_params=dimred_params,
            target_spec=target_spec
        )
        
        if "error" in result:
            return result
            
        # Add legacy fields if missing (for backward compatibility with CLI/API consumers)
        if "forecast" not in result and "forecast_price" in result:
            result["forecast"] = result["forecast_price"]
            
        return result

    except Exception as e:
        import traceback
        return {"error": f"Forecast failed: {str(e)}", "traceback": traceback.format_exc()}
