from typing import Any, Dict, Optional, Literal
from datetime import datetime
import logging
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
from ..utils.mt5 import _mt5_epoch_to_utc, _mt5_copy_rates_from, _ensure_symbol_ready, get_symbol_info_cached
from ..utils.utils import (
    _parse_start_datetime,
    _format_time_minimal,
    _format_time_minimal_local,
    _use_client_tz,
    parse_kv_or_json as _parse_kv_or_json,
)
from ..utils.indicators import _parse_ti_specs as _parse_ti_specs_util, _apply_ta_indicators as _apply_ta_indicators_util
from ..utils.denoise import _apply_denoise, normalize_denoise_spec as _normalize_denoise_spec
from .common import fetch_history as _fetch_history
# Removed invalid import: from .registry import get_forecast_methods_data
from .helpers import (
    default_seasonality_period as _default_seasonality_period,
    next_times_from_last as _next_times_from_last,
    pd_freq_from_timeframe as _pd_freq_from_timeframe,
)
from .target_builder import build_target_series, aggregate_horizon_target
from .forecast_methods import get_forecast_methods_data

logger = logging.getLogger(__name__)
# Simple dimred factory used by the wrapper when building exogenous features.
def _create_dimred_reducer(method: Any, params: Optional[Dict[str, Any]]) -> Any:
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.feature_selection import SelectKBest, f_regression
    except Exception as ex:
        raise RuntimeError(f"dimred dependencies missing: {ex}")
    m = str(method).lower().strip()
    p = params or {}
    if m == 'pca':
        n_components = p.get('n_components', None)
        return PCA(n_components=n_components), {"n_components": n_components}
    if m == 'tsne':
        n_components = p.get('n_components', 2)
        return TSNE(n_components=n_components, random_state=42), {"n_components": n_components}
    if m == 'selectkbest':
        k = p.get('k', 5)
        selector = SelectKBest(score_func=f_regression, k=k)
        return selector, {"k": k}
    # Identity fallback to avoid crashes; caller already wraps in try/except.
    class _Identity:
        def fit_transform(self, X):
            return X
    return _Identity(), {"method": "identity"}

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
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}

        method_l = str(method).lower().strip()
        quantity_l = str(quantity).lower().strip()
        
        # Volatility models have a dedicated endpoint; route to that handler.
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

        # Parse method params via shared helper
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
        except (TypeError, ValueError):
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
        _info_before = get_symbol_info_cached(symbol)
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
            except Exception as ex:
                logger.debug("Denoise spec normalization failed: %s", ex)
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
                fcfg = _parse_kv_or_json(features)
                if not isinstance(fcfg, dict):
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
                    except Exception as exc:
                        logger.debug("Failed to apply indicators: %s", exc)
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
                    
                    # Lazy DT index loading
                    dt_train = None
                    dt_future = None
                    def _ensure_dt():
                        nonlocal dt_train, dt_future
                        if dt_train is None:
                            try:
                                dt_train = pd.to_datetime(df['time'].astype(float).to_numpy(), unit='s', utc=True)
                            except Exception:
                                dt_train = pd.Index([])
                        if dt_future is None:
                            try:
                                t_future = np.asarray(future_times, dtype=float)
                                dt_future = pd.to_datetime(t_future, unit='s', utc=True)
                            except Exception:
                                dt_future = pd.Index([])
                    
                    tr_list: list[np.ndarray] = []
                    tf_list: list[np.ndarray] = []
                    
                    for tok in tokens:
                        tl = tok.lower()
                        # Fourier terms
                        if tl.startswith('fourier:'):
                            try:
                                per = int(tl.split(':',1)[1])
                            except Exception:
                                per = 24
                            w = 2.0 * math.pi / float(max(1, per))
                            # No need for datetime index for fourier on index
                            idx_tr = np.arange(len(df), dtype=float)
                            idx_tf = np.arange(len(future_times), dtype=float)
                            tr_list.append(np.sin(w * idx_tr)); cal_cols.append(f'fx_sin_{per}')
                            tr_list.append(np.cos(w * idx_tr)); cal_cols.append(f'fx_cos_{per}')
                            tf_list.append(np.sin(w * idx_tf));
                            tf_list.append(np.cos(w * idx_tf));
                            continue

                        # Ensure DT indices are ready for date-based features
                        _ensure_dt()
                        if dt_train is None or dt_future is None: # Should not happen
                            continue
                            
                        if tl in ('hour','hr'):
                            vals_tr = dt_train.hour.to_numpy()
                            vals_tf = dt_future.hour.to_numpy()
                            w = 2.0 * math.pi / 24.0
                            tr_list.append(np.sin(w * vals_tr)); cal_cols.append('hr_sin')
                            tr_list.append(np.cos(w * vals_tr)); cal_cols.append('hr_cos')
                            tf_list.append(np.sin(w * vals_tf));
                            tf_list.append(np.cos(w * vals_tf));
                        elif tl in ('dow','wday','weekday'):
                            vals_tr = dt_train.weekday.to_numpy()
                            vals_tf = dt_future.weekday.to_numpy()
                            w = 2.0 * math.pi / 7.0
                            tr_list.append(np.sin(w * vals_tr)); cal_cols.append('dow_sin')
                            tr_list.append(np.cos(w * vals_tr)); cal_cols.append('dow_cos')
                            tf_list.append(np.sin(w * vals_tf));
                            tf_list.append(np.cos(w * vals_tf));
                        elif tl in ('month', 'mo'):
                            # Month 1-12 -> 0-11 for cyclic
                            vals_tr = dt_train.month.to_numpy() - 1
                            vals_tf = dt_future.month.to_numpy() - 1
                            w = 2.0 * math.pi / 12.0
                            tr_list.append(np.sin(w * vals_tr)); cal_cols.append('mo_sin')
                            tr_list.append(np.cos(w * vals_tr)); cal_cols.append('mo_cos')
                            tf_list.append(np.sin(w * vals_tf));
                            tf_list.append(np.cos(w * vals_tf));
                        elif tl in ('day', 'dom'):
                            # Day 1-31 -> 0-30
                            vals_tr = dt_train.day.to_numpy() - 1
                            vals_tf = dt_future.day.to_numpy() - 1
                            w = 2.0 * math.pi / 31.0
                            tr_list.append(np.sin(w * vals_tr)); cal_cols.append('dom_sin')
                            tr_list.append(np.cos(w * vals_tr)); cal_cols.append('dom_cos')
                            tf_list.append(np.sin(w * vals_tf));
                            tf_list.append(np.cos(w * vals_tf));
                        elif tl in ('doy', 'dayofyear'):
                            # Day of year 1-366 -> 0-365
                            vals_tr = dt_train.dayofyear.to_numpy() - 1
                            vals_tf = dt_future.dayofyear.to_numpy() - 1
                            w = 2.0 * math.pi / 365.25
                            tr_list.append(np.sin(w * vals_tr)); cal_cols.append('doy_sin')
                            tr_list.append(np.cos(w * vals_tr)); cal_cols.append('doy_cos')
                            tf_list.append(np.sin(w * vals_tf));
                            tf_list.append(np.cos(w * vals_tf));
                        elif tl in ('week', 'woy'):
                            # Week 1-53 -> 0-52. isocalendar().week returns UInt32, need cast
                            vals_tr = dt_train.isocalendar().week.to_numpy().astype(float) - 1
                            vals_tf = dt_future.isocalendar().week.to_numpy().astype(float) - 1
                            w = 2.0 * math.pi / 52.143 # 365/7
                            tr_list.append(np.sin(w * vals_tr)); cal_cols.append('woy_sin')
                            tr_list.append(np.cos(w * vals_tr)); cal_cols.append('woy_cos')
                            tf_list.append(np.sin(w * vals_tf));
                            tf_list.append(np.cos(w * vals_tf));
                        elif tl in ('minute', 'min'):
                            # Minute 0-59
                            vals_tr = dt_train.minute.to_numpy()
                            vals_tf = dt_future.minute.to_numpy()
                            w = 2.0 * math.pi / 60.0
                            tr_list.append(np.sin(w * vals_tr)); cal_cols.append('min_sin')
                            tr_list.append(np.cos(w * vals_tr)); cal_cols.append('min_cos')
                            tf_list.append(np.sin(w * vals_tf));
                            tf_list.append(np.cos(w * vals_tf));
                        elif tl in ('mod', 'minute_of_day'):
                            # Minute of day 0-1439
                            vals_tr = dt_train.hour.to_numpy() * 60 + dt_train.minute.to_numpy()
                            vals_tf = dt_future.hour.to_numpy() * 60 + dt_future.minute.to_numpy()
                            w = 2.0 * math.pi / 1440.0
                            tr_list.append(np.sin(w * vals_tr)); cal_cols.append('mod_sin')
                            tr_list.append(np.cos(w * vals_tr)); cal_cols.append('mod_cos')
                            tf_list.append(np.sin(w * vals_tf));
                            tf_list.append(np.cos(w * vals_tf));
                        elif tl in ('is_weekend', 'weekend'):
                            # 0 or 1
                            vals_tr = (dt_train.weekday >= 5).astype(float)
                            vals_tf = (dt_future.weekday >= 5).astype(float)
                            tr_list.append(vals_tr); cal_cols.append('is_weekend')
                            tf_list.append(vals_tf);
                        elif tl in ('is_holiday', 'holiday'):
                            try:
                                import holidays
                                country = fcfg.get('country', 'US')
                                # Gather years
                                years_tr = dt_train.year.unique()
                                years_tf = dt_future.year.unique()
                                all_years = np.unique(np.concatenate([years_tr, years_tf]))
                                # Init calendar
                                hol_cal = holidays.CountryHoliday(country, years=all_years)
                                # Map dates
                                # is_holiday check on datetime/date objects
                                # dt_train is DatetimeIndex
                                vals_tr = np.array([1.0 if d in hol_cal else 0.0 for d in dt_train], dtype=float)
                                vals_tf = np.array([1.0 if d in hol_cal else 0.0 for d in dt_future], dtype=float)
                                tr_list.append(vals_tr); cal_cols.append('is_holiday')
                                tf_list.append(vals_tf);
                            except Exception:
                                pass # Ignore if holidays lib missing or country invalid
                    
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
                    
                    # Save raw exog for future generation (before appending calendar features)
                    exog_raw = exog

                    # Append calendar features
                    if cal_train is not None:
                        exog = np.hstack([exog, cal_train]) if exog.size else cal_train
                    
                    # Align with return series if needed
                    if (quantity_l == 'return') or (str(target).lower() == 'return'):
                        exog = exog[1:]
                        if exog_raw.size > 0:
                            exog_raw = exog_raw[1:]
                    
                    # Build future exog by holding the last observed row of raw features + future calendar features
                    exog_f = None
                    if exog_raw.size > 0:
                        last_row = exog_raw[-1]
                        exog_f = np.tile(last_row.reshape(1, -1), (int(horizon), 1))
                    
                    if exog_f is not None and cal_future is not None:
                        exog_f = np.hstack([exog_f, cal_future])
                    elif exog_f is None and cal_future is not None:
                        exog_f = cal_future
                        
                    exog_used = exog
                    exog_future = exog_f
                    feat_info['selected_columns'] = sel_cols + cal_cols
                    feat_info['n_features'] = int(exog_used.shape[1]) if exog_used is not None else 0
                else:
                    feat_info['selected_columns'] = []
            except Exception as _ex:
                feat_info['error'] = f"feature_build_error: {str(_ex)}"
                __stage = 'features_error'

        # Use the unified forecast engine
        from .forecast_engine import forecast_engine
        
        # Map legacy arguments to engine arguments
        engine_params = params or {}
        # Inject context for methods that need it (like analog)
        engine_params['symbol'] = symbol
        engine_params['timeframe'] = timeframe
        
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
            target_spec=target_spec,
            exog_used=exog_used,
            exog_future=exog_future,
            prefetched_df=df,
            prefetched_base_col=base_col,
            prefetched_denoise_spec=dn_spec_used,
        )
        
        if "error" in result:
            return result
            
        return result

    except Exception as e:
        import traceback
        return {"error": f"Forecast failed: {str(e)}", "traceback": traceback.format_exc()}
