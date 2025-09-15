from typing import Any, Dict, Optional, List, Literal
from datetime import datetime
import os
import json
import math
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import warnings

from ..core.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..utils.mt5 import _mt5_epoch_to_utc, _mt5_copy_rates_from, _ensure_symbol_ready
from ..utils.utils import (
    _parse_start_datetime as _parse_start_datetime_util,
    _format_time_minimal as _format_time_minimal_util,
    _format_time_minimal_local as _format_time_minimal_local_util,
    _use_client_tz as _use_client_tz_util,
)
from ..utils.indicators import _parse_ti_specs as _parse_ti_specs_util, _apply_ta_indicators as _apply_ta_indicators_util
from ..utils.denoise import _apply_denoise
from .common import (
    parse_kv_or_json as _parse_kv_or_json,
    fetch_history as _fetch_history,
)

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
    _CHRONOS_AVAILABLE = (_importlib_util5.find_spec("chronos") is not None) or (_importlib_util5.find_spec("transformers") is not None)
except Exception:
    _CHRONOS_AVAILABLE = False
try:
    import importlib.util as _importlib_util6  # type: ignore
    _TIMESFM_AVAILABLE = (_importlib_util6.find_spec("timesfm") is not None) or (_importlib_util6.find_spec("transformers") is not None)
except Exception:
    _TIMESFM_AVAILABLE = False
try:
    import importlib.util as _importlib_util7  # type: ignore
    _LAG_LLAMA_AVAILABLE = (_importlib_util7.find_spec("lag_llama") is not None) or (_importlib_util7.find_spec("transformers") is not None)
except Exception:
    _LAG_LLAMA_AVAILABLE = False


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
    "timesfm",
    "lag_llama",
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
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Fast forecasts for the next `horizon` bars using lightweight methods.
    Parameters: symbol, timeframe, method, horizon, lookback?, as_of?, params?, ci_alpha?, target, denoise?, timezone

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
        if denoise:
            added = _apply_denoise(df, denoise, default_when='pre_ti')
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
            tf_secs = TIMEFRAME_SECONDS.get(timeframe)
            if not tf_secs:
                return {"error": f"Unsupported timeframe seconds for {timeframe}"}
            if isinstance(params, dict):
                p = dict(params)
            elif isinstance(params, str):
                s = params.strip()
                if (s.startswith('{') and s.endswith('}')):
                    try:
                        p = json.loads(s)
                    except Exception:
                        p = {}
                        toks = [tok for tok in s.strip().strip('{}').split() if tok]
                        i = 0
                        while i < len(toks):
                            tok = toks[i].strip().strip(',')
                            if not tok:
                                i += 1; continue
                            if '=' in tok:
                                k, v = tok.split('=', 1)
                                p[k.strip()] = v.strip().strip(',')
                                i += 1; continue
                            if tok.endswith(':'):
                                key = tok[:-1].strip()
                                val = ''
                                if i + 1 < len(toks):
                                    val = toks[i+1].strip().strip(',')
                                    i += 2
                                else:
                                    i += 1
                                p[key] = val
                                continue
                            i += 1
                else:
                    p = {}
                    for tok in s.split():
                        t = tok.strip().strip(',')
                        if '=' in t:
                            k, v = t.split('=', 1)
                            p[k.strip()] = v.strip()
            else:
                p = {}
            if method_l == 'vol_ewma':
                look = int(p.get('lookback', 1500))
                halflife = p.get('halflife'); lam = p.get('lambda_', 0.94)
                with np.errstate(divide='ignore', invalid='ignore'):
                    r = np.diff(np.log(np.maximum(df[base_col].astype(float).to_numpy(), 1e-12)))
                r = r[np.isfinite(r)]
                if r.size < max(look, 10):
                    return {"error": "Not enough data for EWMA volatility"}
                if halflife is not None:
                    try:
                        lam = 1.0 - math.log(2.0) / float(halflife)
                    except Exception:
                        lam = 0.94
                lam = float(lam)
                w = np.power(lam, np.arange(look-1, -1, -1, dtype=float))
                w = w / float(np.sum(w))
                tail = r[-look:]
                sigma2 = float(np.sum(w * (tail * tail)))
                sigma_bar = math.sqrt(max(0.0, sigma2))
                horizon_sigma = float(sigma_bar * math.sqrt(max(1, int(horizon))))
                # Annualization: bars per year based on timeframe seconds
                bpy = float(365.0*24.0*3600.0/float(tf_secs))
                sigma_annual = float(sigma_bar * math.sqrt(bpy))
                horizon_sigma_annual = float(horizon_sigma * math.sqrt(bpy / max(1, int(horizon))))
                return {
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "quantity": "volatility",
                    "method": method_l,
                    "horizon": int(horizon),
                    "sigma_bar_return": sigma_bar,
                    "sigma_annual_return": sigma_annual,
                    "horizon_sigma_return": horizon_sigma,
                    "horizon_sigma_annual": horizon_sigma_annual,
                    "params_used": {"lookback": look, "lambda_": lam},
                }
            elif method_l in ('vol_parkinson','vol_gk','vol_rs'):
                window = int(p.get('window', 20))
                o = df['open'].astype(float).to_numpy(); h = df['high'].astype(float).to_numpy(); l = df['low'].astype(float).to_numpy(); c = df[base_col].astype(float).to_numpy()
                if o.size < window + 2:
                    return {"error": "Not enough OHLC bars for range volatility"}
                if method_l == 'vol_parkinson':
                    v = _parkinson_sigma_sq(h, l)
                elif method_l == 'vol_gk':
                    v = _garman_klass_sigma_sq(o, h, l, c)
                else:
                    v = _rogers_satchell_sigma_sq(o, h, l, c)
                # rolling mean of variance
                vv = pd.Series(v).rolling(window=window, min_periods=window).mean().to_numpy()
                sigma2 = float(vv[-1]) if np.isfinite(vv[-1]) else float(np.nanmean(vv[-window:]))
                sigma_bar = math.sqrt(max(0.0, sigma2))
                horizon_sigma = float(sigma_bar * math.sqrt(max(1, int(horizon))))
                bpy = float(365.0*24.0*3600.0/float(tf_secs))
                sigma_annual = float(sigma_bar * math.sqrt(bpy))
                horizon_sigma_annual = float(horizon_sigma * math.sqrt(bpy / max(1, int(horizon))))
                return {
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "quantity": "volatility",
                    "method": method_l,
                    "horizon": int(horizon),
                    "sigma_bar_return": sigma_bar,
                    "sigma_annual_return": sigma_annual,
                    "horizon_sigma_return": horizon_sigma,
                    "horizon_sigma_annual": horizon_sigma_annual,
                    "params_used": {"window": int(window)},
                }
            elif method_l == 'vol_garch':
                if not _ARCH_AVAILABLE:
                    return {"error": "vol_garch requires 'arch' package."}
                fit_bars = int(p.get('fit_bars', 2000)); mean_model = str(p.get('mean', 'Zero')).lower(); dist = str(p.get('dist', 'normal'))
                with np.errstate(divide='ignore', invalid='ignore'):
                    r = 100.0 * np.diff(np.log(np.maximum(df[base_col].astype(float).to_numpy(), 1e-12)))
                r = r[np.isfinite(r)]
                if r.size < max(100, fit_bars):
                    return {"error": "Not enough data to fit GARCH"}
                try:
                    am = _arch_model(r[-fit_bars:], mean=mean_model if mean_model in ('zero','constant') else 'zero', vol='GARCH', p=1, q=1, dist=dist)
                    res = am.fit(disp='off')
                    fc = res.forecast(horizon=max(1, int(horizon)), reindex=False)
                    variances = fc.variance.values[-1]
                    sigma_bar = float(math.sqrt(max(0.0, float(variances[0])))) / 100.0
                    horizon_sigma = float(math.sqrt(max(0.0, float(np.sum(variances))))) / 100.0
                    bpy = float(365.0*24.0*3600.0/float(tf_secs))
                    sigma_annual = float(sigma_bar * math.sqrt(bpy))
                    horizon_sigma_annual = float(horizon_sigma * math.sqrt(bpy / max(1, int(horizon))))
                    return {
                        "success": True,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "quantity": "volatility",
                        "method": method_l,
                        "horizon": int(horizon),
                        "sigma_bar_return": sigma_bar,
                        "sigma_annual_return": sigma_annual,
                        "horizon_sigma_return": horizon_sigma,
                        "horizon_sigma_annual": horizon_sigma_annual,
                        "params_used": {"fit_bars": int(fit_bars), "mean": mean_model, "dist": dist},
                    }
                except Exception as ex:
                    return {"error": f"GARCH error: {ex}"}
            else:
                return {"error": f"Unknown volatility method: {method_l}"}

        # Fit/forecast by method (price/return)
        __stage = f'method_{method_l}_fit'
        fh = int(horizon)
        f_vals = np.zeros(fh, dtype=float)
        pre_ci: Optional[Tuple[np.ndarray, np.ndarray]] = None
        model_fitted: Optional[np.ndarray] = None
        params_used: Dict[str, Any] = {}

        if method_l == 'naive':
            last_val = float(series[-1])
            f_vals[:] = last_val
            params_used = {}

        elif method_l == 'drift':
            # Classic drift: y_{T+h} = y_T + h*(y_T - y_1)/(T-1)
            slope = (float(series[-1]) - float(series[0])) / float(max(1, n - 1))
            f_vals = float(series[-1]) + slope * np.arange(1, fh + 1, dtype=float)
            params_used = {"slope": slope}

        elif method_l == 'seasonal_naive':
            m_eff = int(p.get('seasonality', m) or m)
            if m_eff <= 0 or n < m_eff:
                return {"error": f"Insufficient data for seasonal_naive (m={m_eff})"}
            last_season = series[-m_eff:]
            reps = int(np.ceil(fh / float(m_eff)))
            f_vals = np.tile(last_season, reps)[:fh]
            params_used = {"m": m_eff}

        elif method_l == 'theta':
            # Combine linear trend extrapolation with simple exponential smoothing (fast, fixed alpha)
            alpha = float(p.get('alpha', 0.2))
            # Linear trend via least squares on original series index
            tt = np.arange(1, n + 1, dtype=float)
            A = np.vstack([np.ones(n), tt]).T
            coef, _, _, _ = np.linalg.lstsq(A, series, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            trend_future = a + b * (tt[-1] + np.arange(1, fh + 1, dtype=float))
            # SES on series
            level = float(series[0])
            for v in series[1:]:
                level = alpha * float(v) + (1.0 - alpha) * level
            ses_future = np.full(fh, level, dtype=float)
            f_vals = 0.5 * (trend_future + ses_future)
            params_used = {"alpha": alpha, "trend_slope": b}

        elif method_l == 'fourier_ols':
            m_eff = int(p.get('seasonality', m) or m)
            K = int(p.get('K', min(3, max(1, (m_eff // 2) if m_eff else 2))))
            use_trend = bool(p.get('trend', True))
            tt = np.arange(1, n + 1, dtype=float)
            X_list = [np.ones(n)]
            if use_trend:
                X_list.append(tt)
            for k in range(1, K + 1):
                w = 2.0 * math.pi * k / float(m_eff if m_eff else max(2, n))
                X_list.append(np.sin(w * tt))
                X_list.append(np.cos(w * tt))
            X = np.vstack(X_list).T
            coef, _, _, _ = np.linalg.lstsq(X, series, rcond=None)
            # Future design
            tt_f = tt[-1] + np.arange(1, fh + 1, dtype=float)
            Xf_list = [np.ones(fh)]
            if use_trend:
                Xf_list.append(tt_f)
            for k in range(1, K + 1):
                w = 2.0 * math.pi * k / float(m_eff if m_eff else max(2, n))
                Xf_list.append(np.sin(w * tt_f))
                Xf_list.append(np.cos(w * tt_f))
            Xf = np.vstack(Xf_list).T
            f_vals = Xf @ coef
            params_used = {"m": m_eff, "K": K, "trend": use_trend}

        elif method_l == 'ses':
            if not _SM_ETS_AVAILABLE:
                return {"error": "SES requires statsmodels. Please install 'statsmodels'."}
            alpha = p.get('alpha')
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if alpha is None:
                        res = _SES(series, initialization_method='heuristic').fit(optimized=True)
                    else:
                        res = _SES(series, initialization_method='heuristic').fit(smoothing_level=float(alpha), optimized=False)
                f_vals = res.forecast(fh)
                f_vals = np.asarray(f_vals, dtype=float)
                try:
                    model_fitted = np.asarray(res.fittedvalues, dtype=float)
                except Exception:
                    model_fitted = None
                alpha_used = None
                try:
                    par = getattr(res, 'params', None)
                    if par is not None:
                        # pandas Series or dict-like
                        if hasattr(par, 'get'):
                            val = par.get('smoothing_level', None)
                            if val is None:
                                # fall back to first element if available
                                try:
                                    val = float(par.iloc[0]) if hasattr(par, 'iloc') else float(par[0])
                                except Exception:
                                    val = None
                            alpha_used = val
                        else:
                            # array-like
                            try:
                                alpha_used = float(par[0]) if len(par) > 0 else None
                            except Exception:
                                alpha_used = None
                    if alpha_used is None:
                        mv = getattr(res.model, 'smoothing_level', None)
                        alpha_used = mv if mv is not None else alpha
                except Exception:
                    alpha_used = alpha
                params_used = {"alpha": _to_float_or_nan(alpha_used)}
            except Exception as ex:
                return {"error": f"SES fitting error: {ex}"}

        elif method_l == 'holt':
            if not _SM_ETS_AVAILABLE:
                return {"error": "Holt requires statsmodels. Please install 'statsmodels'."}
            damped = bool(p.get('damped', True))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = _ETS(series, trend='add', damped_trend=damped, initialization_method='heuristic')
                    res = model.fit(optimized=True)
                f_vals = res.forecast(fh)
                f_vals = np.asarray(f_vals, dtype=float)
                try:
                    model_fitted = np.asarray(res.fittedvalues, dtype=float)
                except Exception:
                    model_fitted = None
                params_used = {"damped": damped}
            except Exception as ex:
                return {"error": f"Holt fitting error: {ex}"}

        elif method_l in ('holt_winters_add', 'holt_winters_mul'):
            if not _SM_ETS_AVAILABLE:
                return {"error": "Holt-Winters requires statsmodels. Please install 'statsmodels'."}
            m_eff = int(p.get('seasonality', m) or m)
            if m_eff <= 0:
                return {"error": "Holt-Winters requires a positive seasonality_period"}
            seasonal = 'add' if method_l == 'holt_winters_add' else 'mul'
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = _ETS(series, trend='add', seasonal=seasonal, seasonal_periods=m_eff, initialization_method='heuristic')
                    res = model.fit(optimized=True)
                f_vals = res.forecast(fh)
                f_vals = np.asarray(f_vals, dtype=float)
                try:
                    model_fitted = np.asarray(res.fittedvalues, dtype=float)
                except Exception:
                    model_fitted = None
                params_used = {"seasonal": seasonal, "m": m_eff}
            except Exception as ex:
                return {"error": f"Holt-Winters fitting error: {ex}"}

        elif method_l in ('arima', 'sarima'):
            if not _SM_SARIMAX_AVAILABLE:
                return {"error": "ARIMA/SARIMA require statsmodels. Please install 'statsmodels'."}
            # Defaults: price: d=1, returns: d=0
            d_default = 0 if use_returns else 1
            p_ord = int(p.get('p', 1)); d_ord = int(p.get('d', d_default)); q_ord = int(p.get('q', 1))
            if method_l == 'sarima':
                m_eff = int(p.get('seasonality', m) or m)
                P = int(p.get('P', 0)); D = int(p.get('D', 1 if not use_returns else 0)); Q = int(p.get('Q', 0))
                # SARIMAX requires seasonal period >= 2; fall back to non-seasonal if < 2
                if m_eff is None or m_eff < 2:
                    seas = (0, 0, 0, 0)
                else:
                    seas = (P, D, Q, int(m_eff))
            else:
                seas = (0, 0, 0, 0)
            trend = str(p.get('trend', 'c'))  # 'n' or 'c'
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    endog = pd.Series(series.astype(float))
                    model = _SARIMAX(
                        endog,
                        order=(p_ord, d_ord, q_ord),
                        seasonal_order=seas,
                        trend=trend,
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                        exog=exog_used if exog_used is not None else None,
                    )
                    res = model.fit(method='lbfgs', disp=False, maxiter=100)
                    if exog_future is not None:
                        pred = res.get_forecast(steps=fh, exog=exog_future)
                    else:
                        pred = res.get_forecast(steps=fh)
                f_vals = pred.predicted_mean.to_numpy()
                ci = None
                try:
                    # Use configured CI alpha if provided; default to 0.05
                    _alpha = float(ci_alpha) if ci_alpha is not None else 0.05
                    ci_df = pred.conf_int(alpha=_alpha)
                    ci = (ci_df.iloc[:, 0].to_numpy(), ci_df.iloc[:, 1].to_numpy())
                except Exception:
                    ci = None
                params_used = {"order": (p_ord, d_ord, q_ord), "seasonal_order": seas if method_l=='sarima' else (0,0,0,0), "trend": trend}
                if exog_used is not None:
                    params_used["exog_features"] = {"n_features": int(exog_used.shape[1]), **feat_info}
                if ci is not None:
                    pre_ci = ci
            except Exception as ex:
                return {"error": f"SARIMAX fitting error: {ex}"}

        elif method_l in ('nhits', 'nbeatsx', 'tft', 'patchtst'):
            # Deep learning via Nixtla NeuralForecast (optional dependency)
            if not _NF_AVAILABLE:
                return {"error": f"{method_l.upper()} requires 'neuralforecast' (and PyTorch). Install: pip install neuralforecast[torch]"}
            try:
                from neuralforecast import NeuralForecast as _NeuralForecast  # type: ignore
                # Try to import model classes individually; some may be missing depending on version
                try:
                    from neuralforecast.models import NHITS as _NF_NHITS  # type: ignore
                except Exception:
                    _NF_NHITS = None  # type: ignore
                try:
                    from neuralforecast.models import NBEATSx as _NF_NBEATSX  # type: ignore
                except Exception:
                    _NF_NBEATSX = None  # type: ignore
                try:
                    from neuralforecast.models import TFT as _NF_TFT  # type: ignore
                except Exception:
                    _NF_TFT = None  # type: ignore
                try:
                    from neuralforecast.models import PatchTST as _NF_PATCHTST  # type: ignore
                except Exception:
                    _NF_PATCHTST = None  # type: ignore
                import pandas as _pd
            except Exception as ex:
                return {"error": f"Failed to import neuralforecast: {ex}"}

            # Resolve model class based on method
            model_class = None
            if method_l == 'nhits':
                model_class = _NF_NHITS
            elif method_l == 'nbeatsx':
                model_class = _NF_NBEATSX
            elif method_l == 'tft':
                model_class = _NF_TFT
            elif method_l == 'patchtst':
                model_class = _NF_PATCHTST
            if model_class is None:
                return {"error": f"Model '{method_l}' not available in installed neuralforecast version"}

            # Training setup
            max_epochs = int(p.get('max_epochs', 50))
            batch_size = int(p.get('batch_size', 32))
            lr = p.get('learning_rate', None)
            # Choose input_size: prefer user param; else modest window leveraging seasonality if known
            h = int(fh)
            if p.get('input_size') is not None:
                requested = int(p['input_size'])
                # Cap input to available length so we have at least some windows; prefer n - h when possible
                if n > h:
                    max_input = max(8, n - h)
                    input_size = int(min(requested, max_input))
                else:
                    input_size = int(max(2, min(requested, n)))
            else:
                base = max(64, (m * 3) if m and m > 0 else 96)
                input_size = int(min(n, base))
            # Build single-series training dataframe
            try:
                # Align timestamps to target series length (returns drop first bar)
                if use_returns:
                    ts_train = _pd.to_datetime(df['time'].iloc[1:].astype(float), unit='s', utc=True)
                else:
                    ts_train = _pd.to_datetime(df['time'].astype(float), unit='s', utc=True)
                Y_df = _pd.DataFrame({
                    'unique_id': ['ts'] * int(len(series)),
                    'ds': _pd.Index(ts_train).to_pydatetime(),
                    'y': series.astype(float),
                })
            except Exception as ex:
                return {"error": f"Failed to build training frame for {method_l.upper()}: {ex}"}

            # Build model kwargs with compatibility across NF versions (max_steps vs max_epochs)
            steps = int(p.get('max_steps', p.get('max_epochs', 50)))
            try:
                from .common import nf_setup_and_predict as _nf_setup_and_predict
                Yf = _nf_setup_and_predict(
                    model_class=model_class,
                    fh=int(fh),
                    timeframe=timeframe,
                    Y_df=Y_df,
                    input_size=int(input_size),
                    batch_size=int(batch_size),
                    steps=int(steps),
                    learning_rate=float(lr) if lr is not None else None,
                    exog_used=exog_used,
                    exog_future=exog_future,
                    future_times=future_times,
                )
                try:
                    Yf = Yf[Yf['unique_id'] == 'ts']
                except Exception:
                    pass
                # Prefer standard 'y_hat' if present; else first non-meta column
                pred_col = None
                for c in list(Yf.columns):
                    if c not in ('unique_id', 'ds', 'y'):
                        pred_col = c
                        if c == 'y_hat':
                            break
                if pred_col is None:
                    return {"error": f"{method_l.upper()} prediction columns not found"}
                vals = np.asarray(Yf[pred_col].to_numpy(), dtype=float)
                f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
                params_used = {
                    'max_epochs': int(max_epochs),
                    'input_size': int(input_size),
                    'batch_size': int(batch_size),
                }
            except Exception as ex:
                return {"error": f"{method_l.upper()} fitting/prediction error: {ex}"}

        elif method_l in ('sf_autoarima', 'sf_theta', 'sf_autoets', 'sf_seasonalnaive'):
            if not _SF_AVAILABLE:
                return {"error": f"{method_l} requires 'statsforecast'. Install: pip install statsforecast"}
            try:
                from statsforecast import StatsForecast as _StatsForecast  # type: ignore
                from statsforecast.models import AutoARIMA as _SF_AutoARIMA, Theta as _SF_Theta, AutoETS as _SF_AutoETS, SeasonalNaive as _SF_SeasonalNaive  # type: ignore
                import pandas as _pd
            except Exception as ex:
                return {"error": f"Failed to import statsforecast: {ex}"}
            # Build training frame
            try:
                if use_returns:
                    ts_train = _pd.to_datetime(df['time'].iloc[1:].astype(float), unit='s', utc=True)
                else:
                    ts_train = _pd.to_datetime(df['time'].astype(float), unit='s', utc=True)
                Y_df = _pd.DataFrame({
                    'unique_id': ['ts'] * int(len(series)),
                    'ds': _pd.Index(ts_train).to_pydatetime(),
                    'y': series.astype(float),
                })
                # Optional exogenous covariates
                X_df = None
                Xf_df = None
                if exog_used is not None and isinstance(exog_used, np.ndarray) and exog_used.size:
                    cols = [f'x{i}' for i in range(exog_used.shape[1])]
                    X_df = _pd.DataFrame({'unique_id': ['ts'] * int(len(series)), 'ds': _pd.Index(ts_train).to_pydatetime()})
                    for j, cname in enumerate(cols):
                        X_df[cname] = exog_used[:, j]
                    # Future exog
                    if exog_future is not None and isinstance(exog_future, np.ndarray) and exog_future.size:
                        ds_f = _pd.to_datetime(_pd.Series(future_times), unit='s', utc=True)
                        Xf_df = _pd.DataFrame({'unique_id': ['ts'] * int(len(ds_f)), 'ds': _pd.Index(ds_f).to_pydatetime()})
                        for j, cname in enumerate(cols):
                            Xf_df[cname] = exog_future[:, j]
            except Exception as ex:
                return {"error": f"Failed to build training frame for {method_l}: {ex}"}
            m_eff = int(p.get('seasonality', m) or m)
            if method_l == 'sf_autoarima':
                stepwise = bool(p.get('stepwise', True))
                d_ord = p.get('d'); D_ord = p.get('D')
                model = _SF_AutoARIMA(season_length=max(1, m_eff or 1), stepwise=stepwise, d=d_ord, D=D_ord)
                params_used = {"seasonality": m_eff, "stepwise": stepwise, "d": d_ord, "D": D_ord}
            elif method_l == 'sf_theta':
                model = _SF_Theta(season_length=max(1, m_eff or 1))
                params_used = {"seasonality": m_eff}
            elif method_l == 'sf_autoets':
                model = _SF_AutoETS(season_length=max(1, m_eff or 1))
                params_used = {"seasonality": m_eff}
            else:  # sf_seasonalnaive
                model = _SF_SeasonalNaive(season_length=max(1, m_eff or 1))
                params_used = {"seasonality": m_eff}
            try:
                sf = _StatsForecast(models=[model], freq=_pd_freq_from_timeframe(timeframe))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if X_df is not None:
                        sf.fit(Y_df, X_df=X_df)
                    else:
                        sf.fit(Y_df)
                if Xf_df is not None:
                    Yf = sf.predict(h=int(fh), X_df=Xf_df)
                else:
                    Yf = sf.predict(h=int(fh))
                try:
                    Yf = Yf[Yf['unique_id'] == 'ts']
                except Exception:
                    pass
                pred_col = None
                for c in list(Yf.columns):
                    if c not in ('unique_id', 'ds', 'y'):
                        pred_col = c
                        if c == 'y':  # some versions may return 'y'
                            break
                if pred_col is None:
                    # Fallback: try 'y' directly
                    pred_col = 'y' if 'y' in Yf.columns else None
                if pred_col is None:
                    return {"error": f"StatsForecast prediction columns not found"}
                vals = np.asarray(Yf[pred_col].to_numpy(), dtype=float)
                f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
            except Exception as ex:
                return {"error": f"StatsForecast {method_l} error: {ex}"}

        elif method_l == 'mlf_rf':
            if not _MLF_AVAILABLE:
                return {"error": "mlf_rf requires 'mlforecast'. Install: pip install mlforecast scikit-learn"}
            try:
                from mlforecast import MLForecast as _MLForecast  # type: ignore
                from sklearn.ensemble import RandomForestRegressor as _RF  # type: ignore
                from utilsforecast.features import time_features as _time_features  # type: ignore
                import pandas as _pd
            except Exception as ex:
                return {"error": f"Failed to import mlforecast/sklearn: {ex}"}
            # Prepare features config
            lags_in = p.get('lags', 'auto')
            if lags_in == 'auto' or lags_in is None:
                # Use short and seasonal lags when available
                base_lags = [1, 2, 3, 4, 5]
                if m and m > 0:
                    base_lags += [m]
                lags = sorted(set([int(abs(x)) for x in base_lags if int(abs(x)) > 0]))
            else:
                try:
                    lags = [int(v) for v in lags_in]
                except Exception:
                    lags = [1, 2, 3, 4, 5]
            roll = str(p.get('rolling_agg', 'mean')).lower() if p.get('rolling_agg', None) is not None else None
            n_estimators = int(p.get('n_estimators', 200))
            max_depth = p.get('max_depth', None)

            try:
                if use_returns:
                    ts_train = _pd.to_datetime(df['time'].iloc[1:].astype(float), unit='s', utc=True)
                else:
                    ts_train = _pd.to_datetime(df['time'].astype(float), unit='s', utc=True)
                Y_df = _pd.DataFrame({
                    'unique_id': ['ts'] * int(len(series)),
                    'ds': _pd.Index(ts_train).to_pydatetime(),
                    'y': series.astype(float),
                })
            except Exception as ex:
                return {"error": f"Failed to build training frame for mlf_rf: {ex}"}

            rf = _RF(n_estimators=n_estimators, max_depth=None if max_depth in (None, 'None') else int(max_depth), random_state=42)
            try:
                # Attach exogenous columns if present
                Xf_df = None
                if exog_used is not None and isinstance(exog_used, np.ndarray) and exog_used.size:
                    cols = [f'x{i}' for i in range(exog_used.shape[1])]
                    for j, cname in enumerate(cols):
                        Y_df[cname] = exog_used[:, j]
                    if exog_future is not None and isinstance(exog_future, np.ndarray) and exog_future.size:
                        ds_f = _pd.to_datetime(_pd.Series(future_times), unit='s', utc=True)
                        Xf_df = _pd.DataFrame({'unique_id': ['ts'] * int(len(ds_f)), 'ds': _pd.Index(ds_f).to_pydatetime()})
                        for j, cname in enumerate(cols):
                            Xf_df[cname] = exog_future[:, j]
                mlf = _MLForecast(models=[rf], freq=_pd_freq_from_timeframe(timeframe))
                # Set lags and optional rolling aggregates
                mlf = mlf.add_lags(lags)
                if roll in {'mean', 'min', 'max', 'std'}:
                    # Add simple rolling window features for each lag window
                    for w in sorted(set([x for x in lags if x > 1])):
                        mlf = mlf.add_rolling_windows(rolling_features={roll: [w]})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mlf.fit(Y_df)
                if Xf_df is not None:
                    Yf = mlf.predict(h=int(fh), X_df=Xf_df)
                else:
                    Yf = mlf.predict(h=int(fh))
                try:
                    Yf = Yf[Yf['unique_id'] == 'ts']
                except Exception:
                    pass
                # mlforecast usually returns column named after target 'y'
                pred_col = 'y' if 'y' in Yf.columns else None
                if pred_col is None:
                    # fallback to first non-meta
                    for c in list(Yf.columns):
                        if c not in ('unique_id', 'ds'):
                            pred_col = c
                            break
                if pred_col is None:
                    return {"error": "mlf_rf prediction columns not found"}
                vals = np.asarray(Yf[pred_col].to_numpy(), dtype=float)
                f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
                params_used = {
                    'lags': lags,
                    'rolling_agg': roll,
                    'n_estimators': n_estimators,
                    'max_depth': None if max_depth in (None, 'None') else int(max_depth),
                }
            except Exception as ex:
                return {"error": f"mlf_rf error: {ex}"}

        elif method_l == 'mlf_lightgbm':
            if not _MLF_AVAILABLE:
                return {"error": "mlf_lightgbm requires 'mlforecast'. Install: pip install mlforecast"}
            if not _LGB_AVAILABLE:
                return {"error": "mlf_lightgbm requires 'lightgbm'. Install: pip install lightgbm"}
            try:
                from mlforecast import MLForecast as _MLForecast  # type: ignore
                from lightgbm import LGBMRegressor as _LGBM  # type: ignore
                import pandas as _pd
            except Exception as ex:
                return {"error": f"Failed to import mlforecast/lightgbm: {ex}"}
            # Prepare features config
            lags_in = p.get('lags', 'auto')
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
            roll = str(p.get('rolling_agg', 'mean')).lower() if p.get('rolling_agg', None) is not None else None
            n_estimators = int(p.get('n_estimators', 200))
            lr = float(p.get('learning_rate', 0.05))
            num_leaves = int(p.get('num_leaves', 31))
            max_depth = int(p.get('max_depth', -1))

            try:
                if use_returns:
                    ts_train = _pd.to_datetime(df['time'].iloc[1:].astype(float), unit='s', utc=True)
                else:
                    ts_train = _pd.to_datetime(df['time'].astype(float), unit='s', utc=True)
                Y_df = _pd.DataFrame({
                    'unique_id': ['ts'] * int(len(series)),
                    'ds': _pd.Index(ts_train).to_pydatetime(),
                    'y': series.astype(float),
                })
            except Exception as ex:
                return {"error": f"Failed to build training frame for mlf_lightgbm: {ex}"}

            lgbm = _LGBM(n_estimators=n_estimators, learning_rate=lr, num_leaves=num_leaves, max_depth=max_depth, random_state=42)
            try:
                # Attach exogenous columns if present
                Xf_df = None
                if exog_used is not None and isinstance(exog_used, np.ndarray) and exog_used.size:
                    cols = [f'x{i}' for i in range(exog_used.shape[1])]
                    for j, cname in enumerate(cols):
                        Y_df[cname] = exog_used[:, j]
                    if exog_future is not None and isinstance(exog_future, np.ndarray) and exog_future.size:
                        ds_f = _pd.to_datetime(_pd.Series(future_times), unit='s', utc=True)
                        Xf_df = _pd.DataFrame({'unique_id': ['ts'] * int(len(ds_f)), 'ds': _pd.Index(ds_f).to_pydatetime()})
                        for j, cname in enumerate(cols):
                            Xf_df[cname] = exog_future[:, j]
                mlf = _MLForecast(models=[lgbm], freq=_pd_freq_from_timeframe(timeframe))
                mlf = mlf.add_lags(lags)
                if roll in {'mean', 'min', 'max', 'std'}:
                    for w in sorted(set([x for x in lags if x > 1])):
                        mlf = mlf.add_rolling_windows(rolling_features={roll: [w]})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mlf.fit(Y_df)
                if Xf_df is not None:
                    Yf = mlf.predict(h=int(fh), X_df=Xf_df)
                else:
                    Yf = mlf.predict(h=int(fh))
                try:
                    Yf = Yf[Yf['unique_id'] == 'ts']
                except Exception:
                    pass
                pred_col = 'y' if 'y' in Yf.columns else None
                if pred_col is None:
                    for c in list(Yf.columns):
                        if c not in ('unique_id', 'ds'):
                            pred_col = c
                            break
                if pred_col is None:
                    return {"error": "mlf_lightgbm prediction columns not found"}
                vals = np.asarray(Yf[pred_col].to_numpy(), dtype=float)
                f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
                params_used = {
                    'lags': lags,
                    'rolling_agg': roll,
                    'n_estimators': n_estimators,
                    'learning_rate': lr,
                    'num_leaves': num_leaves,
                    'max_depth': max_depth,
                }
            except Exception as ex:
                return {"error": f"mlf_lightgbm error: {ex}"}

        elif method_l == 'chronos_bolt':
            if not _CHRONOS_AVAILABLE:
                return {"error": "chronos_bolt requires 'chronos' or 'transformers' with a supported model. Try: pip install chronos-forecasting or pip install transformers torch accelerate"}
            model_name = str(p.get('model_name', 'amazon/chronos-bolt-base'))
            ctx_len = int(p.get('context_length', 0) or 0)
            device = p.get('device')
            device_map = p.get('device_map', 'auto')
            quantization = str(p.get('quantization')) if p.get('quantization') is not None else None
            quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None
            revision = p.get('revision')
            trust_remote_code = bool(p.get('trust_remote_code', False))
            # Select context window
            if ctx_len and ctx_len > 0:
                context = series[-int(min(n, ctx_len)) :]
            else:
                context = series
            f_vals = None
            last_err = None
            fq: Dict[str, List[float]] = {}
            # Try native ChronosPipeline first
            try:
                from chronos import ChronosPipeline  # type: ignore
                _kwargs: Dict[str, Any] = {}
                if quantization:
                    if quantization.lower() in ('int8', '8bit', 'bnb.int8'):
                        _kwargs['load_in_8bit'] = True
                    elif quantization.lower() in ('int4', '4bit', 'bnb.int4'):
                        _kwargs['load_in_4bit'] = True
                if revision:
                    _kwargs['revision'] = revision
                pipe = ChronosPipeline.from_pretrained(model_name, device_map=device_map, **_kwargs)  # type: ignore[arg-type]
                if quantiles:
                    yhat = pipe.predict(context=context, prediction_length=int(fh), quantiles=list(quantiles))  # type: ignore[call-arg]
                    # yhat could be dict quantile->list
                    if isinstance(yhat, dict):
                        for q, arr in yhat.items():
                            try:
                                qf = float(q)
                            except Exception:
                                continue
                            fq[str(qf)] = [float(v) for v in np.asarray(arr, dtype=float)[:fh].tolist()]
                        # choose median or first quantile as point
                        if '0.5' in fq:
                            f_vals = np.asarray(fq['0.5'], dtype=float)
                    else:
                        vals = np.asarray(yhat, dtype=float)
                        f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
                else:
                    yhat = pipe.predict(context=context, prediction_length=int(fh))  # type: ignore[call-arg]
                    vals = np.asarray(yhat, dtype=float)
                    f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
            except Exception as ex1:
                last_err = ex1
                try:
                    # Fallback via Transformers pipeline API
                    from transformers import pipeline as _hf_pipeline  # type: ignore
                    _pipe_kwargs: Dict[str, Any] = {'model': model_name}
                    if device and str(device).lower() != 'auto':
                        _pipe_kwargs['device'] = device
                    else:
                        _pipe_kwargs['device_map'] = device_map
                    if revision:
                        _pipe_kwargs['revision'] = revision
                    _model_kwargs: Dict[str, Any] = {}
                    if quantization:
                        if quantization.lower() in ('int8', '8bit', 'bnb.int8'):
                            _model_kwargs['load_in_8bit'] = True
                        elif quantization.lower() in ('int4', '4bit', 'bnb.int4'):
                            _model_kwargs['load_in_4bit'] = True
                    if trust_remote_code:
                        _model_kwargs['trust_remote_code'] = True
                    if _model_kwargs:
                        _pipe_kwargs['model_kwargs'] = _model_kwargs
                    hf = _hf_pipeline("time-series-forecasting", **_pipe_kwargs)  # type: ignore[call-arg]
                    call_kwargs: Dict[str, Any] = {'prediction_length': int(fh)}
                    if quantiles:
                        call_kwargs['quantiles'] = list(quantiles)
                    yhat = hf(context, **call_kwargs)  # type: ignore[call-arg]
                    # yhat may be list or dict depending on version
                    if isinstance(yhat, dict):
                        # Expect possible 'forecast' for point and 'quantiles' mapping
                        qmap = yhat.get('quantiles')
                        if isinstance(qmap, dict):
                            for q, arrq in qmap.items():
                                try:
                                    qf = float(q)
                                except Exception:
                                    continue
                                fq[str(qf)] = [float(v) for v in np.asarray(arrq, dtype=float)[:fh].tolist()]
                        arr = yhat.get('forecast') or yhat.get('yhat') or yhat.get('y_hat') or yhat.get('y')
                    else:
                        arr = yhat
                    vals = np.asarray(arr, dtype=float)
                    f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
                except Exception as ex2:
                    return {"error": f"Chronos-Bolt inference error: {ex2 if ex2 else last_err}"}
            params_used = {
                'model_name': model_name,
                'context_length': int(ctx_len) if ctx_len else int(n),
                'device': device,
                'device_map': device_map,
                'quantization': quantization,
                'revision': revision,
                'trust_remote_code': trust_remote_code,
            }
            if fq:
                # attach quantiles for downstream consumers
                params_used['quantiles'] = sorted(list(fq.keys()), key=lambda x: float(x))
                # Stash on locals to use when mapping to prices below
                forecast_quantiles = fq  # type: ignore[name-defined]

        elif method_l in ('timesfm', 'lag_llama'):
            # Generic HF pipeline adapter; try native libs if present
            model_name = p.get('model_name') or ("google/timesfm-1.0-200m" if method_l == 'timesfm' else None)
            if not model_name:
                return {"error": f"{method_l} requires params.model_name with a valid HF repo id"}
            ctx_len = int(p.get('context_length', 0) or 0)
            device = p.get('device')
            device_map = p.get('device_map', 'auto')
            quantization = str(p.get('quantization')) if p.get('quantization') is not None else None
            quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None
            revision = p.get('revision')
            trust_remote_code = bool(p.get('trust_remote_code', False))
            if ctx_len and ctx_len > 0:
                context = series[-int(min(n, ctx_len)) :]
            else:
                context = series
            f_vals = None
            last_err = None
            fq: Dict[str, List[float]] = {}
            # Try native packages if available
            if method_l == 'timesfm':
                try:
                    import timesfm as _timesfm  # type: ignore
                    # Heuristic: try a from_pretrained + predict API similar to Chronos
                    try:
                        kw = {}
                        if revision:
                            kw['revision'] = revision
                        mdl = getattr(_timesfm, 'TimesFm').from_pretrained(model_name, **kw)  # type: ignore[attr-defined]
                        if quantiles:
                            yhat = mdl.predict(context=context, prediction_length=int(fh), quantiles=list(quantiles))  # type: ignore[call-arg]
                            if isinstance(yhat, dict):
                                for q, arr in yhat.items():
                                    try:
                                        qf = float(q)
                                    except Exception:
                                        continue
                                    fq[str(qf)] = [float(v) for v in np.asarray(arr, dtype=float)[:fh].tolist()]
                                if '0.5' in fq:
                                    f_vals = np.asarray(fq['0.5'], dtype=float)
                        else:
                            yhat = mdl.predict(context=context, prediction_length=int(fh))  # type: ignore[call-arg]
                            vals = np.asarray(yhat, dtype=float)
                            f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
                    except Exception as _:
                        pass
                except Exception as ex:
                    last_err = ex
            # Fallback to Transformers pipeline
            if f_vals is None:
                try:
                    from transformers import pipeline as _hf_pipeline  # type: ignore
                    _pipe_kwargs: Dict[str, Any] = {'model': model_name}
                    if device and str(device).lower() != 'auto':
                        _pipe_kwargs['device'] = device
                    else:
                        _pipe_kwargs['device_map'] = device_map
                    if revision:
                        _pipe_kwargs['revision'] = revision
                    _model_kwargs: Dict[str, Any] = {}
                    if quantization:
                        if quantization.lower() in ('int8', '8bit', 'bnb.int8'):
                            _model_kwargs['load_in_8bit'] = True
                        elif quantization.lower() in ('int4', '4bit', 'bnb.int4'):
                            _model_kwargs['load_in_4bit'] = True
                    if trust_remote_code:
                        _model_kwargs['trust_remote_code'] = True
                    if _model_kwargs:
                        _pipe_kwargs['model_kwargs'] = _model_kwargs
                    hf = _hf_pipeline("time-series-forecasting", **_pipe_kwargs)  # type: ignore[call-arg]
                    call_kwargs: Dict[str, Any] = {'prediction_length': int(fh)}
                    if quantiles:
                        call_kwargs['quantiles'] = list(quantiles)
                    yhat = hf(context, **call_kwargs)  # type: ignore[call-arg]
                    arr = yhat.get('forecast') if isinstance(yhat, dict) else yhat
                    qmap = yhat.get('quantiles') if isinstance(yhat, dict) else None
                    if isinstance(qmap, dict):
                        for q, arrq in qmap.items():
                            try:
                                qf = float(q)
                            except Exception:
                                continue
                            fq[str(qf)] = [float(v) for v in np.asarray(arrq, dtype=float)[:fh].tolist()]
                    vals = np.asarray(arr, dtype=float)
                    f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
                except Exception as ex2:
                    return {"error": f"{method_l} inference error: {ex2 if ex2 else last_err}"}
            params_used = {
                'model_name': str(model_name),
                'context_length': int(ctx_len) if ctx_len else int(n),
                'device': device,
                'device_map': device_map,
                'quantization': quantization,
                'revision': revision,
                'trust_remote_code': trust_remote_code,
            }
            if fq:
                params_used['quantiles'] = sorted(list(fq.keys()), key=lambda x: float(x))
                forecast_quantiles = fq  # type: ignore[name-defined]

        # Compute residual scale for intervals (on modeling scale)
        lower = upper = None
        do_ci = (ci_alpha is not None)
        _alpha = float(ci_alpha) if ci_alpha is not None else 0.05
        if do_ci:
            try:
                # Prefer model-provided intervals if available (e.g., SARIMAX)
                if pre_ci is not None:
                    lo, hi = pre_ci
                    lower = np.asarray(lo, dtype=float)
                    upper = np.asarray(hi, dtype=float)
                # Else compute from in-sample residuals
                elif method_l == 'naive':
                    fitted = np.roll(series, 1)[1:]
                    resid = series[1:] - fitted
                elif method_l == 'drift':
                    slope = (float(series[-1]) - float(series[0])) / float(max(1, n - 1))
                    fitted = series[:-1] + slope  # 1-step ahead approx
                    resid = series[1:] - fitted
                elif method_l == 'seasonal_naive':
                    m_eff = int(params_used.get('m', m) or m)
                    if n > m_eff:
                        resid = series[m_eff:] - series[:-m_eff]
                    else:
                        resid = series - np.mean(series)
                elif method_l == 'theta':
                    alpha = float(params_used.get('alpha', 0.2))
                    tt = np.arange(1, n + 1, dtype=float)
                    A = np.vstack([np.ones(n), tt]).T
                    coef, _, _, _ = np.linalg.lstsq(A, series, rcond=None)
                    a, b = float(coef[0]), float(coef[1])
                    trend = a + b * tt
                    level = float(series[0])
                    fitted_ses = [level]
                    for v in series[1:]:
                        level = alpha * float(v) + (1.0 - alpha) * level
                        fitted_ses.append(level)
                    fitted_theta = 0.5 * (trend + np.array(fitted_ses))
                    resid = series - fitted_theta
                elif method_l in ('ses','holt','holt_winters_add','holt_winters_mul') and model_fitted is not None:
                    fitted = model_fitted
                    if fitted.shape[0] > n:
                        fitted = fitted[-n:]
                    elif fitted.shape[0] < n:
                        # pad with last fitted
                        last = fitted[-1] if fitted.size > 0 else float('nan')
                        fitted = np.pad(fitted, (n - fitted.shape[0], 0), mode='edge') if fitted.size > 0 else np.full(n, last)
                    resid = series - fitted
                else:  # fourier_ols fallback
                    tt = np.arange(1, n + 1, dtype=float)
                    m_eff = int(params_used.get('m', m) or m)
                    K = int(params_used.get('K', min(3, (m_eff // 2) if m_eff else 2)))
                    use_trend = bool(params_used.get('trend', True))
                    X_list = [np.ones(n)]
                    if use_trend:
                        X_list.append(tt)
                    for k in range(1, K + 1):
                        w = 2.0 * math.pi * k / float(m_eff if m_eff else max(2, n))
                        X_list.append(np.sin(w * tt))
                        X_list.append(np.cos(w * tt))
                    X = np.vstack(X_list).T
                    coef, _, _, _ = np.linalg.lstsq(X, series, rcond=None)
                    fitted = X @ coef
                    resid = series - fitted
                if pre_ci is None:
                    resid = resid[np.isfinite(resid)]
                    sigma = float(np.std(resid, ddof=1)) if resid.size >= 3 else float('nan')
                    from scipy.stats import norm  # optional if available
                    try:
                        z = float(norm.ppf(1.0 - _alpha / 2.0))
                    except Exception:
                        z = 1.96
                    lower = f_vals - z * sigma
                    upper = f_vals + z * sigma
            except Exception:
                do_ci = False

        # Map back to price if legacy target was returns (custom targets skip mapping)
        if (not custom_target_mode) and use_returns:
            # Compose price path multiplicatively from origin_price
            price_path = np.empty(fh, dtype=float)
            price_path[0] = origin_price * math.exp(float(f_vals[0]))
            for i in range(1, fh):
                price_path[i] = price_path[i-1] * math.exp(float(f_vals[i]))
            out_forecast_price = price_path
            if do_ci and lower is not None and upper is not None:
                # Convert return bands to price bands via lognormal mapping per-step
                lower_price = np.empty(fh, dtype=float)
                upper_price = np.empty(fh, dtype=float)
                lower_price[0] = origin_price * math.exp(float(lower[0]))
                upper_price[0] = origin_price * math.exp(float(upper[0]))
                for i in range(1, fh):
                    lower_price[i] = lower_price[i-1] * math.exp(float(lower[i]))
                    upper_price[i] = upper_price[i-1] * math.exp(float(upper[i]))
            else:
                lower_price = upper_price = None
        else:
            out_forecast_price = f_vals
            lower_price = lower
            upper_price = upper

        # If model produced quantile forecasts, map them to price space if needed and attach
        forecast_quantiles_price: Optional[Dict[str, List[float]]] = None
        try:
            if 'forecast_quantiles' in locals() and isinstance(forecast_quantiles, dict):
                forecast_quantiles_price = {}
                for qk, qarr in forecast_quantiles.items():
                    qa = np.asarray(qarr, dtype=float)
                    if use_returns:
                        qp = np.zeros_like(qa)
                        if qa.size > 0:
                            qp[0] = origin_price * math.exp(float(qa[0]))
                            for i in range(1, qa.size):
                                qp[i] = qp[i-1] * math.exp(float(qa[i]))
                        forecast_quantiles_price[str(qk)] = [float(v) for v in qp.tolist()]
                    else:
                        forecast_quantiles_price[str(qk)] = [float(v) for v in qa.tolist()]
        except Exception:
            forecast_quantiles_price = None

        # Rounding based on symbol digits
        digits = int(getattr(_info_before, "digits", 0) or 0)
        def _round(v: float) -> float:
            try:
                return round(float(v), digits) if digits >= 0 else float(v)
            except Exception:
                return float(v)

        _use_ctz = _use_client_tz_util(timezone)
        if _use_ctz:
            times_fmt = [_format_time_minimal_local_util(ts) for ts in future_times]
        else:
            times_fmt = [_format_time_minimal_util(ts) for ts in future_times]

        # Training window first/last candle timestamps (used for the forecast)
        try:
            train_first_epoch = float(df['time'].iloc[0])
            train_last_epoch = float(df['time'].iloc[-1])
        except Exception:
            train_first_epoch = float('nan')
            train_last_epoch = float('nan')
        if _use_ctz:
            train_first_time = _format_time_minimal_local_util(train_first_epoch) if math.isfinite(train_first_epoch) else None
            train_last_time = _format_time_minimal_local_util(train_last_epoch) if math.isfinite(train_last_epoch) else None
        else:
            train_first_time = _format_time_minimal_util(train_first_epoch) if math.isfinite(train_first_epoch) else None
            train_last_time = _format_time_minimal_util(train_last_epoch) if math.isfinite(train_last_epoch) else None

        # Overall forecast trend based on net change over horizon
        try:
            if out_forecast_price.size >= 2:
                delta = float(out_forecast_price[-1] - out_forecast_price[0])
                # Use half a rounding unit as flat threshold
                prec = max(0, int(getattr(_info_before, "digits", 0) or 0))
                unit = 10.0 ** (-prec) if prec <= 12 else 0.0
                thresh = 0.5 * unit if unit > 0 else 0.0
                if delta > thresh:
                    forecast_trend = "up"
                elif delta < -thresh:
                    forecast_trend = "down"
                else:
                    forecast_trend = "flat"
            else:
                forecast_trend = "flat"
        except Exception:
            forecast_trend = "flat"
        payload: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method_l,
            "quantity": quantity_l,
            "target": str(target),
            "params_used": params_used,
            "lookback_used": int(len(df)),
            "horizon": int(horizon),
            "seasonality_period": int(m or 0),
            "as_of": as_of or None,
            "train_start": train_first_time,
            "train_end": train_last_time,
            "times": times_fmt,
            "target_spec_used": target_info if custom_target_mode else None,
        }
        # Attach forecast outputs depending on target mode
        if custom_target_mode:
            payload["forecast_series"] = [float(v) for v in f_vals.tolist()]
            # Optional horizon aggregation
            try:
                ts = dict(target_spec or {})
                agg = str(ts.get('horizon_agg', 'last')).lower()
                norm = str(ts.get('normalize', 'none')).lower()
                val = None
                arr = np.asarray(f_vals, dtype=float)
                if agg == 'last':
                    val = float(arr[-1]) if arr.size else float('nan')
                elif agg == 'mean':
                    val = float(np.nanmean(arr))
                elif agg == 'sum':
                    val = float(np.nansum(arr))
                elif agg == 'slope':
                    h = arr.size
                    tt = np.arange(1, h + 1, dtype=float)
                    A = np.vstack([np.ones(h), tt]).T
                    coef, _, _, _ = np.linalg.lstsq(A, arr, rcond=None)
                    val = float(coef[1])
                elif agg == 'max':
                    val = float(np.nanmax(arr))
                elif agg == 'min':
                    val = float(np.nanmin(arr))
                elif agg == 'range':
                    val = float(np.nanmax(arr) - np.nanmin(arr))
                elif agg == 'vol':
                    # If transformed as returns/log_returns/diff, arr is already increments; else approximate via diff
                    inc = arr if target_info.get('transform','none') in ('return','log_return','diff','pct_change') else np.diff(arr)
                    val = float(math.sqrt(np.nansum(np.square(inc))))
                # normalization
                if val is not None and math.isfinite(val):
                    if norm == 'per_bar' and arr.size > 0:
                        val = float(val) / float(arr.size)
                    elif norm == 'pct':
                        val = float(val) * 100.0
                payload["forecast_agg"] = {"agg": agg, "normalize": norm, "value": float(val) if val is not None else None}
                # Optional classification
                cls = ts.get('classification')
                if cls:
                    cls_s = str(cls).lower()
                    thresh = float(ts.get('threshold', 0.0))
                    label = None
                    if cls_s == 'sign':
                        label = 1 if (val is not None and float(val) > 0.0) else (-1 if (val is not None and float(val) < 0.0) else 0)
                    elif cls_s == 'threshold':
                        label = int(1 if (val is not None and abs(float(val)) >= float(thresh)) else 0)
                    payload["forecast_label"] = label
            except Exception:
                pass
        else:
            payload["forecast_price"] = [_round(v) for v in out_forecast_price.tolist()]
        if not _use_ctz:
            payload["timezone"] = "UTC"
        payload["forecast_trend"] = forecast_trend
        if (not custom_target_mode) and use_returns:
            payload["forecast_return"] = [float(v) for v in f_vals.tolist()]
        if (not custom_target_mode) and do_ci and lower_price is not None and upper_price is not None:
            payload["lower_price"] = [_round(v) for v in lower_price.tolist()]
            payload["upper_price"] = [_round(v) for v in upper_price.tolist()]
            payload["ci_alpha"] = float(_alpha)
        if (not custom_target_mode) and forecast_quantiles_price:
            # Attach quantiles (rounded)
            qout: Dict[str, List[float]] = {}
            for k, arr in forecast_quantiles_price.items():
                qout[str(k)] = [_round(v) for v in arr]
            payload["forecast_quantiles"] = qout

        return payload
    except Exception as e:
        dbg = {}
        try:
            dbg = {
                "stage": __stage,
                "features_type": type(features).__name__,
                "params_type": type(params).__name__,
                "target_spec_type": type(target_spec).__name__,
                "features_preview": (str(features)[:200] if features is not None else None),
                "params_preview": (str(params)[:200] if params is not None else None),
                "target_spec_preview": (str(target_spec)[:200] if target_spec is not None else None),
            }
        except Exception:
            pass
        return {"error": f"Error computing forecast: {str(e)}", "debug": dbg}


# pattern_prepare_index has been removed; pattern_search builds/loads indexes on demand.


# pattern_search_recent removed; use pattern_search.
    try:
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        # Ensure cached index
        dn_key = _denoise_cache_key(denoise)
        eff_lb = int(lookback) if lookback is not None else None
        # Include dimension reduction in key when provided
        if dimred_method and str(dimred_method).lower() not in ("", "none"):
            dr_desc = f"dr={str(dimred_method).lower()}"
        else:
            dr_desc = f"pca={int(pca_components) if pca_components else 0}"
        cache_key = (str(timeframe), int(window_size), int(future_size),
                     f"dn={dn_key}|sc={str(scale).lower()}|mt={str(metric).lower()}|{dr_desc}|eng={(engine or 'ckdtree').lower()}|lb={eff_lb if eff_lb is not None else 'auto'}")
        idx = _PATTERN_INDEX_CACHE.get(cache_key)
        if idx is None or (symbol not in idx.symbols):
            # Try disk cache if requested
            if cache_id:
                try:
                    disk_idx = _load_pattern_index_from_disk(cache_dir, cache_id)
                except Exception:
                    disk_idx = None
                if disk_idx is not None and disk_idx.timeframe == timeframe and disk_idx.window_size == int(window_size) and disk_idx.future_size == int(future_size):
                    idx = disk_idx
                    _PATTERN_INDEX_CACHE[cache_key] = idx
            if idx is None:
                # Build minimal index for this symbol only
                eff_max = int(eff_lb) if eff_lb is not None else max(2000, window_size + future_size + 100)
                idx = _build_pattern_index(
                    symbols=[symbol],
                    timeframe=str(timeframe),
                window_size=int(window_size),
                future_size=int(future_size),
                max_bars_per_symbol=int(eff_max),
                denoise=denoise,
                scale=str(scale),
                metric=str(metric),
                pca_components=int(pca_components) if pca_components else None,
                dimred_method=dimred_method,
                dimred_params=dimred_params,
                engine=str(engine),
            )
                _PATTERN_INDEX_CACHE[cache_key] = idx
                _save_pattern_index_to_disk(idx, cache_dir, cache_id)

        # Fetch anchor last `window_size` closes for the queried symbol
        def _fetch_anchor(symbol: str, timeframe: str, window_size: int) -> Tuple[np.ndarray, float]:
            utc_now = datetime.utcnow()
            rates = _mt5_copy_rates_from(symbol, TIMEFRAME_MAP[timeframe], utc_now, int(window_size) + 2)
            if rates is None or len(rates) == 0:
                raise RuntimeError(f"Failed to fetch anchor bars for {symbol}")
            df = pd.DataFrame(rates)
            try:
                df['time'] = df['time'].astype(float).apply(_mt5_epoch_to_utc)
            except Exception:
                pass
            if 'volume' not in df.columns and 'tick_volume' in df.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df['volume'] = df['tick_volume']
            if denoise and isinstance(denoise, dict):
                try:
                    dn = dict(denoise)
                    dn.setdefault('when', 'pre_ti')
                    dn.setdefault('columns', ['close'])
                    dn.setdefault('keep_original', False)
                    _apply_denoise_util(df, dn, default_when='pre_ti')
                except Exception:
                    pass
            if len(df) < window_size:
                raise RuntimeError("Not enough bars for anchor window")
            from ..utils.utils import to_float_np
            closes = to_float_np(df['close'])
            try:
                end_epoch = float(df['time'].iloc[-1])
            except Exception:
                # Fallback: use server 'now'
                end_epoch = float(datetime.utcnow().timestamp())
            return closes[-int(window_size):], end_epoch

        anchor_vals, anchor_end_epoch = _fetch_anchor(symbol, str(timeframe), int(window_size))
        # Helper scalers and NCC for multi-scale refinement
        def _scale_vec(x: np.ndarray, how: str) -> np.ndarray:
            s = (how or 'minmax').lower()
            x = np.asarray(x, dtype=float)
            if s == 'zscore':
                mu = float(np.nanmean(x)); sd = float(np.nanstd(x))
                if not np.isfinite(sd) or sd <= 1e-12:
                    return np.zeros_like(x, dtype=float)
                return (x - mu) / sd
            if s == 'none':
                return x
            mn = float(np.nanmin(x)); mx = float(np.nanmax(x)); rng = mx - mn
            if not np.isfinite(rng) or rng <= 1e-12:
                return np.zeros_like(x, dtype=float)
            return (x - mn) / rng

        def _ncc_max(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
            a = np.asarray(a, dtype=float).ravel(); b = np.asarray(b, dtype=float).ravel()
            n = int(min(a.size, b.size))
            if n <= 2:
                return 0.0
            def zn(x):
                xm = float(np.nanmean(x)); xs = float(np.nanstd(x))
                if not np.isfinite(xs) or xs <= 1e-12: return np.zeros_like(x, dtype=float)
                return (x - xm) / xs
            a = zn(a); b = zn(b)
            L = int(max(0, allow_lag)); best = -1.0
            for lag in range(-L, L+1):
                if lag == 0:
                    aa, bb = a, b
                elif lag > 0:
                    aa = a[lag:]; bb = b[: n - lag]
                else:
                    aa = a[: n + lag]; bb = b[-lag:]
                m = int(min(aa.size, bb.size))
                if m <= 2: continue
                num = float(np.dot(aa, bb)); den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
                corr = (num / den) if (np.isfinite(den) and den > 1e-12) else 0.0
                if corr > best: best = corr
            return float(max(min(best, 1.0), -1.0))

        # Multi-scale list from a single span value: span=0.1 -> [0.9, 1.0, 1.1]
        use_scales: List[float] = []
        try:
            span = float(time_scale_span) if (time_scale_span is not None) else 0.0
        except Exception:
            span = 0.0
        if span and span > 0:
            lo = max(0.05, 1.0 - span)
            hi = 1.0 + span
            use_scales = [lo, 1.0, hi]
        else:
            use_scales = [1.0]

        # Build candidate pool across scales or default index
        initial_k = int(refine_k) if refine_k and int(refine_k) > int(top_k) else int(top_k)
        candidate_pool: List[Tuple[_PatternIndex, int, float]] = []
        for sc in use_scales:
            ws2 = int(round(float(window_size) * float(sc)))
            if ws2 < 5: continue
            # Acquire index for ws2
            if dimred_method and str(dimred_method).lower() not in ("", "none"):
                dr_desc2 = f"dr={str(dimred_method).lower()}"
            else:
                dr_desc2 = f"pca={int(pca_components) if pca_components else 0}"
            cache_key2 = (str(timeframe), int(ws2), int(future_size),
                          f"dn={dn_key}|sc={str(scale).lower()}|mt={str(metric).lower()}|{dr_desc2}|eng={(engine or 'ckdtree').lower()}|lb={eff_lb if eff_lb is not None else 'auto'}")
            idx2 = _PATTERN_INDEX_CACHE.get(cache_key2)
            if idx2 is None:
                try:
                    idx2 = _build_pattern_index(
                        symbols=idx.symbols if idx is not None else [symbol],
                        timeframe=str(timeframe),
                        window_size=int(ws2),
                        future_size=int(future_size),
                        max_bars_per_symbol=int(getattr(idx, 'max_bars_per_symbol', eff_lb if eff_lb is not None else max(2000, ws2 + future_size + 100))),
                        denoise=denoise,
                        scale=str(scale),
                        metric=str(metric),
                        pca_components=int(pca_components) if pca_components else None,
                        dimred_method=dimred_method,
                        dimred_params=dimred_params,
                        engine=str(engine),
                    )
                    _PATTERN_INDEX_CACHE[cache_key2] = idx2
                except Exception:
                    continue
            # Anchor for ws2
            try:
                a2, _ = _fetch_anchor(symbol, str(timeframe), int(ws2))
            except Exception:
                continue
            ii, dd = idx2.search(a2, top_k=int(initial_k))
            for j, dj in zip(ii.tolist(), dd.tolist()):
                # Store (index_obj, local_index, distance, time_scale) so later scoring can see scale
                candidate_pool.append((idx2, int(j), float(dj), float(sc)))

        # If only single scale and no shape metric, fallback to original flow
        if len(use_scales) == 1 and (not shape_metric or str(shape_metric).lower().strip() in ('', 'none')):
            # Convert candidate_pool to idxs,dists for legacy loop
            idxs = np.array([j for _, j, _ in candidate_pool], dtype=int)
            dists = np.array([d for _, _, d in candidate_pool], dtype=float)
        else:
            # Re-rank on resampled windows to main length (NCC/DTW/Soft-DTW/Affine)
            a_main = _scale_vec(np.asarray(anchor_vals, dtype=float), str(scale))
            scored: List[Tuple[float, _PatternIndex, int, float, Optional[float]]] = []
            for idx_obj, loc_idx, _orig_d, sc in candidate_pool:
                try:
                    w = idx_obj.get_match_values(int(loc_idx), include_future=False)
                    if w.size != int(window_size):
                        x = np.linspace(0.0, 1.0, num=w.size, dtype=float)
                        xi = np.linspace(0.0, 1.0, num=int(window_size), dtype=float)
                        w = np.interp(xi, x, w.astype(float))
                    w_scaled = _scale_vec(w, str(scale))
                    sm = str(shape_metric).lower().strip() if shape_metric else 'ncc'
                    if sm == 'ncc' or sm == '' or sm == 'none':
                        corr = _ncc_max(a_main, w_scaled, int(allow_lag) if allow_lag else 0)
                        score = 1.0 - float(corr)
                    elif sm == 'affine':
                        # Fit alpha, beta to align candidate amplitude/offset to anchor
                        aw = float(np.dot(a_main - np.mean(a_main), w_scaled - np.mean(w_scaled)))
                        ww = float(np.dot(w_scaled - np.mean(w_scaled), w_scaled - np.mean(w_scaled)))
                        alpha = (aw / ww) if (np.isfinite(ww) and ww > 1e-12) else 0.0
                        alpha = max(0.5, min(2.0, float(alpha)))
                        beta = float(np.mean(a_main) - alpha * np.mean(w_scaled))
                        resid = a_main - (alpha * w_scaled + beta)
                        score = float(np.sqrt(np.mean(resid * resid)))
                    else:
                        # DTW / Soft-DTW
                        n = a_main.size
                        band = None
                        if False:  # placeholder for future dtw_band param
                            pass
                        if sm == 'dtw':
                            # Simple DP fallback DTW
                            ca = np.full((n + 1, n + 1), np.inf, dtype=float)
                            ca[0, 0] = 0.0
                            for ii in range(1, n + 1):
                                for jj in range(1, n + 1):
                                    cost = abs(a_main[ii - 1] - w_scaled[jj - 1])
                                    ca[ii, jj] = cost + min(ca[ii - 1, jj], ca[ii, jj - 1], ca[ii - 1, jj - 1])
                            score = float(ca[n, n])
                        elif sm == 'softdtw':
                            # Fallback to DTW if no soft-dtw lib; using DP above
                            ca = np.full((n + 1, n + 1), np.inf, dtype=float)
                            ca[0, 0] = 0.0
                            for ii in range(1, n + 1):
                                for jj in range(1, n + 1):
                                    cost = abs(a_main[ii - 1] - w_scaled[jj - 1])
                                    ca[ii, jj] = cost + min(ca[ii - 1, jj], ca[ii, jj - 1], ca[ii - 1, jj - 1])
                            score = float(ca[n, n])
                        else:
                            # Default to euclidean
                            diff = a_main - w_scaled
                            score = float(np.sqrt(np.dot(diff, diff)))
                    # Record score with time scale and (if affine) amplitude alpha
                    try:
                        amp_alpha  # type: ignore[name-defined]
                    except Exception:
                        amp_alpha = None  # type: ignore[assignment]
                    scored.append((score, idx_obj, int(loc_idx), float(sc), amp_alpha))
                except Exception:
                    continue
            if not scored:
                return {"error": "No candidates found at any scale"}
            scored.sort(key=lambda t: t[0])
            # Replace idxs/dists with top_k selection proxy (scores used as distance analog)
            selected = scored[: int(top_k)]
            idxs = np.array([i for _, _, i, _, _ in selected], dtype=int)
            dists = np.array([s for s, _, _, _, _ in selected], dtype=float)
            # Also keep parallel lists for multi-scale metadata
            selected_idx_objs: List[_PatternIndex] = [obj for _, obj, _, _, _ in selected]
            selected_scales: List[float] = [ts for _, _, _, ts, _ in selected]
            selected_alphas: List[Optional[float]] = [al for _, _, _, _, al in selected]

        total_candidates = int(len(candidate_pool))
        matches = []
        changes: List[float] = []
        pct_changes: List[float] = []
        kept_dist: List[float] = []
        kept = 0
        # Compute overlap thresholds
        tf_secs = int(TIMEFRAME_SECONDS.get(str(timeframe), 0) or 0)
        anchor_start_epoch = float(anchor_end_epoch) - float(max(0, int(window_size) - 1) * tf_secs)
        # Track kept intervals per symbol to avoid any overlaps (not just with last kept)
        kept_intervals_by_sym: Dict[str, List[Tuple[float, float]]] = {}

        for idx_pos, (i, d) in enumerate(zip(idxs.tolist(), dists.tolist())):
            # When multi-scale used and shape re-rank applied, resolve per-candidate index
            idx_current: _PatternIndex = idx
            if not (len(use_scales) == 1 and (not shape_metric or str(shape_metric).lower().strip() in ('', 'none'))):
                try:
                    idx_current = selected_idx_objs[idx_pos]
                except Exception:
                    idx_current = idx
            tscale = 1.0
            amp_alpha = None
            if not (len(use_scales) == 1 and (not shape_metric or str(shape_metric).lower().strip() in ('', 'none'))):
                try:
                    tscale = float(selected_scales[idx_pos])
                    amp_alpha = selected_alphas[idx_pos]
                except Exception:
                    pass
            m_sym = idx_current.get_match_symbol(i)
            # Optional correlation filter for cross-instrument matches
            if min_symbol_correlation is not None and m_sym != symbol:
                try:
                    r_a = idx.get_symbol_returns(symbol, lookback=int(corr_lookback))
                    r_b = idx.get_symbol_returns(m_sym, lookback=int(corr_lookback))
                    if r_a is not None and r_b is not None:
                        n = min(r_a.size, r_b.size)
                        if n > 10:
                            c = float(np.corrcoef(r_a[-n:], r_b[-n:])[0, 1])
                            if not np.isfinite(c) or c < float(min_symbol_correlation):
                                continue
                except Exception:
                    pass
            vals = idx_current.get_match_values(i, include_future=True)
            times = idx_current.get_match_times(i, include_future=True)
            # Today's value is last of window; future is last of (window+future)
            if vals.size < (idx_current.window_size + max(0, idx_current.future_size)):
                # Skip malformed window
                continue
            today_v = float(vals[idx_current.window_size - 1])
            future_v = float(vals[min(vals.size - 1, idx_current.window_size + idx_current.future_size - 1)])
            change = float(future_v - today_v)
            pct = float((future_v - today_v) / today_v) if today_v != 0 else 0.0
            changes.append(change)
            pct_changes.append(pct)
            kept_dist.append(float(d))
            kept += 1
            start_epoch = float(times[0])
            end_epoch = float(times[idx_current.window_size - 1])
            # Skip overlapping with anchor window for same symbol
            if m_sym == symbol and end_epoch >= anchor_start_epoch - 1e-6:
                continue
            # Skip overlap with any previously kept intervals for this symbol
            kept_list = kept_intervals_by_sym.get(m_sym, [])
            cand_start, cand_end = float(start_epoch), float(end_epoch)
            overlap = False
            for ks, ke in kept_list:
                # Overlap if ranges intersect or touch within epsilon
                if not (cand_end <= ks - 1e-6 or cand_start >= ke + 1e-6):
                    overlap = True
                    break
            if overlap:
                continue
            start_time = _format_time_minimal_util(start_epoch)
            end_time = _format_time_minimal_util(end_epoch)
            _m = {
                "symbol": m_sym,
                "distance": float(d),
                "start_date": start_time,
                "end_date": end_time,
                "todays_value": today_v,
                "future_value": future_v,
                "change": change,
                "pct_change": pct,
                "time_scale": float(tscale),
            }
            if amp_alpha is not None:
                _m["amplitude_scale"] = float(amp_alpha)
            if bool(include_values):
                # Resample series to anchor window+future length for consistent plotting across scales
                try:
                    target_len = int(window_size) + int(future_size)
                    if target_len > 0 and vals.size != target_len:
                        x = np.linspace(0.0, 1.0, num=vals.size, dtype=float)
                        xi = np.linspace(0.0, 1.0, num=target_len, dtype=float)
                        vals_rs = np.interp(xi, x, vals.astype(float))
                        _m["values"] = [float(v) for v in vals_rs.tolist()]
                    else:
                        _m["values"] = [float(v) for v in vals.tolist()]
                except Exception:
                    _m["values"] = [float(v) for v in vals.tolist()]
            matches.append(_m)
            # Record interval as kept to prevent future overlaps
            kept_intervals_by_sym.setdefault(m_sym, []).append((cand_start, cand_end))

        if not matches:
            return {"error": "No matches found"}
        arr = np.array(changes, dtype=float)
        parr = np.array(pct_changes, dtype=float)
        d_arr = np.array(kept_dist, dtype=float)
        pos_ratio = float(np.mean(parr > 0.0)) if parr.size > 0 else 0.0
        mean_change = float(np.mean(arr)) if arr.size else 0.0
        median_change = float(np.median(arr)) if arr.size else 0.0
        std_change = float(np.std(arr, ddof=0)) if arr.size else 0.0
        mean_pct = float(np.mean(parr)) if parr.size else 0.0
        median_pct = float(np.median(parr)) if parr.size else 0.0
        std_pct = float(np.std(parr, ddof=0)) if parr.size else 0.0
        per_bar_mean_change = float(mean_change / max(1, int(future_size)))
        per_bar_mean_pct = float(mean_pct / max(1, int(future_size)))
        eps = 1e-9
        if d_arr.size:
            w = 1.0 / (d_arr + eps)
            w /= np.sum(w)
            w_mean_change = float(np.sum(w * arr))
            w_mean_pct = float(np.sum(w * parr))
        else:
            w_mean_change = mean_change
            w_mean_pct = mean_pct
        forecast_type = "gain" if pos_ratio > 0.5 else "loss"
        forecast_confidence = pos_ratio if pos_ratio > 0.5 else (1.0 - pos_ratio)

        payload: Dict[str, Any] = {
            "success": True,
            "anchor_symbol": symbol,
            "timeframe": timeframe,
            "window_size": int(window_size),
            "future_size": int(future_size),
            "top_k": int(top_k),
            "matches": matches,
            "forecast_type": forecast_type,
            "forecast_confidence": float(forecast_confidence),
            "n_matches": int(kept),
            "n_candidates": int(total_candidates),
            "prob_gain": float(pos_ratio),
            "avg_change": mean_change,
            "avg_pct_change": mean_pct,
            "per_bar_avg_change": per_bar_mean_change,
            "per_bar_avg_pct_change": per_bar_mean_pct,
            "anchor_end_epoch": float(anchor_end_epoch),
            "anchor_end_time": _format_time_minimal_util(float(anchor_end_epoch)),
            "refine_k": int(initial_k),
            "shape_metric": str(shape_metric) if shape_metric else "none",
            "allow_lag": int(allow_lag) if allow_lag else 0,
        }
        if not compact:
            payload.update({
                "median_change": median_change,
                "std_change": std_change,
                "median_pct_change": median_pct,
                "std_pct_change": std_pct,
                "distance_weighted_avg_change": w_mean_change,
                "distance_weighted_avg_pct_change": w_mean_pct,
                "scale": idx.scale,
                "metric": idx.metric,
                "pca_components": idx.pca_components or 0,
                "dimred_method": getattr(idx, 'dimred_method', 'none'),
                "dimred_params": getattr(idx, 'dimred_params', {}),
                "engine": getattr(idx, 'engine', 'ckdtree'),
                "max_bars_per_symbol": int(getattr(idx, 'max_bars_per_symbol', 0)),
                "bars_per_symbol": getattr(idx, 'bars_per_symbol', lambda: {})(),
                "windows_per_symbol": getattr(idx, 'windows_per_symbol', lambda: {})(),
                "lookback": int(getattr(idx, 'max_bars_per_symbol', eff_max if (lookback is not None) else 0)),
            })
        if include_anchor_values:
            payload["anchor_values"] = [float(v) for v in anchor_vals.tolist()]
        return payload
    except Exception as e:
        return {"error": f"Error in pattern search: {str(e)}"}

