"""
Forecast data preprocessing and feature engineering.

Handles data preparation, feature creation, and input validation.
"""

from typing import Any, Dict, Optional, List, Tuple
import math
import numpy as np
import pandas as pd

from ..core.constants import TIMEFRAME_SECONDS
from ..utils.indicators import _parse_ti_specs as _parse_ti_specs_util, _apply_ta_indicators as _apply_ta_indicators_util
from ..utils.denoise import _apply_denoise, normalize_denoise_spec as _normalize_denoise_spec
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json


def _default_seasonality_period(timeframe: str) -> int:
    """Infer a default seasonality for forecasting based on timeframe."""
    try:
        sec = TIMEFRAME_SECONDS.get(timeframe)
        if not sec or sec <= 0:
            return 0
        if sec < 86400:
            return int(round(86400.0 / float(sec)))
        if timeframe == 'D1':
            return 5
        if timeframe == 'W1':
            return 52
        if timeframe == 'MN1':
            return 12
        return 0
    except Exception:
        return 0


def _next_times_from_last(last_epoch: float, tf_secs: int, horizon: int) -> List[float]:
    """Generate future timestamps for forecasting."""
    base = float(last_epoch)
    step = float(tf_secs)
    return [base + step * (i + 1) for i in range(int(horizon))]


def _pd_freq_from_timeframe(tf: str) -> str:
    """Convert MT5 timeframe to pandas frequency string."""
    tf = str(tf).upper().strip()
    if tf == 'M1':
        return '1min'
    if tf in ('M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30'):
        return f"{tf[1:]}min"
    if tf == 'H1':
        return '1H'
    if tf in ('H2', 'H3', 'H4', 'H6', 'H8', 'H12'):
        return f"{tf[1:]}H"
    if tf == 'D1':
        return '1D'
    if tf == 'W1':
        return '1W'
    if tf == 'MN1':
        return '1M'
    return '1H'  # fallback


def prepare_features(
    df: pd.DataFrame, 
    features_cfg: Optional[Dict[str, Any]], 
    future_times: List[float],
    n: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], List[str]]:
    """Prepare exogenous features for forecasting.
    
    Returns:
        - exog_used: Training features array
        - exog_future: Future features array  
        - include_cols: Column names to include as features
        - ti_cols: Technical indicator column names
    """
    if not features_cfg:
        return None, None, [], []
    
    try:
        # Parse features configuration
        if isinstance(features_cfg, str):
            # Try JSON first, then k=v format
            s = features_cfg.strip()
            if s.startswith('{') and s.endswith('}'):
                try:
                    import json
                    fcfg = json.loads(s)
                except Exception:
                    fcfg = _parse_kv_or_json(s)
            else:
                fcfg = _parse_kv_or_json(s)
        elif isinstance(features_cfg, dict):
            fcfg = dict(features_cfg)
        else:
            fcfg = {}
        
        # Process feature inclusion specification
        include_cols = _process_include_specification(df, fcfg)
        
        # Add technical indicators
        ti_cols = _add_technical_indicators(df, fcfg)
        
        # Add calendar/future covariates
        cal_train, cal_future, cal_cols = _add_calendar_features(df, fcfg, future_times)
        
        # Build final feature arrays
        exog_used, exog_future = _build_feature_arrays(
            df, include_cols, ti_cols, cal_train, cal_future, cal_cols, n
        )
        
        return exog_used, exog_future, include_cols, ti_cols
        
    except Exception:
        return None, None, [], []


def _process_include_specification(df: pd.DataFrame, fcfg: Dict[str, Any]) -> List[str]:
    """Process the 'include' specification for feature columns."""
    include = fcfg.get('include', 'ohlcv')
    include_cols: List[str] = []
    
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
    
    return include_cols


def _add_technical_indicators(df: pd.DataFrame, fcfg: Dict[str, Any]) -> List[str]:
    """Add technical indicators to the DataFrame."""
    ind_specs = fcfg.get('indicators')
    if not ind_specs:
        return []
    
    try:
        specs = _parse_ti_specs_util(str(ind_specs)) if isinstance(ind_specs, str) else ind_specs
        _apply_ta_indicators_util(df, specs, default_when='pre_ti')
    except Exception:
        pass
    
    # Collect newly created indicator columns
    ti_cols = []
    for c in df.columns:
        if c in ('time','open','high','low','close','volume','tick_volume','real_volume'):
            continue
        if df[c].dtype.kind in ('f','i'):
            ti_cols.append(c)
    
    return ti_cols


def _add_calendar_features(
    df: pd.DataFrame, 
    fcfg: Dict[str, Any], 
    future_times: List[float]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Add calendar and future-known covariates."""
    fut_cov = fcfg.get('future_covariates')
    if not fut_cov:
        return None, None, []
    
    tokens: List[str] = []
    if isinstance(fut_cov, str):
        tokens = [tok.strip() for tok in fut_cov.replace(',', ' ').split() if tok.strip()]
    elif isinstance(fut_cov, (list, tuple)):
        tokens = [str(tok).strip() for tok in fut_cov]
    
    if not tokens:
        return None, None, []
    
    t_train = df['time'].astype(float).to_numpy()
    t_future = np.asarray(future_times, dtype=float)
    tr_list: List[np.ndarray] = []
    tf_list: List[np.ndarray] = []
    cal_cols: List[str] = []
    
    for tok in tokens:
        tl = tok.lower()
        if tl.startswith('fourier:'):
            tr_f, tf_f, cols_f = _create_fourier_features(tl, t_train, t_future)
            tr_list.extend(tr_f)
            tf_list.extend(tf_f)
            cal_cols.extend(cols_f)
        elif tl in ('hour','hr'):
            tr_h, tf_h = _create_hour_features(t_train, t_future)
            if tr_h is not None:
                tr_list.append(tr_h)
                tf_list.append(tf_h)
                cal_cols.append('hour')
        elif tl in ('dow','dayofweek'):
            tr_d, tf_d = _create_dow_features(t_train, t_future)
            if tr_d is not None:
                tr_list.append(tr_d)
                tf_list.append(tf_d)
                cal_cols.append('dow')
    
    if not tr_list:
        return None, None, []
    
    cal_train = np.column_stack(tr_list) if len(tr_list) > 1 else tr_list[0].reshape(-1, 1)
    cal_future = np.column_stack(tf_list) if len(tf_list) > 1 else tf_list[0].reshape(-1, 1)
    
    return cal_train, cal_future, cal_cols


def _create_fourier_features(
    token: str, 
    t_train: np.ndarray, 
    t_future: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Create Fourier features."""
    try:
        per = int(token.split(':',1)[1])
    except Exception:
        per = 24
    
    w = 2.0 * math.pi / float(max(1, per))
    idx_tr = np.arange(t_train.size, dtype=float)
    idx_tf = np.arange(t_future.size, dtype=float)
    
    tr_features = [np.sin(w * idx_tr), np.cos(w * idx_tr)]
    tf_features = [np.sin(w * idx_tf), np.cos(w * idx_tf)]
    col_names = [f'fx_sin_{per}', f'fx_cos_{per}']
    
    return tr_features, tf_features, col_names


def _create_hour_features(
    t_train: np.ndarray, 
    t_future: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Create hour-of-day features."""
    try:
        hrs_tr = pd.to_datetime(t_train, unit='s', utc=True).hour.to_numpy()
        hrs_tf = pd.to_datetime(t_future, unit='s', utc=True).hour.to_numpy()
        return hrs_tr.astype(float), hrs_tf.astype(float)
    except Exception:
        return None, None


def _create_dow_features(
    t_train: np.ndarray, 
    t_future: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Create day-of-week features."""
    try:
        dow_tr = pd.to_datetime(t_train, unit='s', utc=True).dayofweek.to_numpy()
        dow_tf = pd.to_datetime(t_future, unit='s', utc=True).dayofweek.to_numpy()
        return dow_tr.astype(float), dow_tf.astype(float)
    except Exception:
        return None, None


def _build_feature_arrays(
    df: pd.DataFrame,
    include_cols: List[str],
    ti_cols: List[str], 
    cal_train: Optional[np.ndarray],
    cal_future: Optional[np.ndarray],
    cal_cols: List[str],
    n: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build final exogenous feature arrays."""
    # Combine all feature columns
    all_cols = include_cols + ti_cols
    
    if not all_cols and cal_train is None:
        return None, None
    
    arrays_tr: List[np.ndarray] = []
    arrays_tf: List[np.ndarray] = []
    
    # Add DataFrame columns
    if all_cols:
        for col in all_cols:
            if col in df.columns:
                arrays_tr.append(df[col].to_numpy().astype(float))
                # For future, repeat the last value
                last_val = float(df[col].iloc[-1]) if len(df) > 0 else 0.0
                arrays_tf.append(np.full(n, last_val, dtype=float))
    
    # Add calendar features
    if cal_train is not None and cal_future is not None:
        if cal_train.ndim == 1:
            arrays_tr.append(cal_train)
            arrays_tf.append(cal_future)
        else:
            for i in range(cal_train.shape[1]):
                arrays_tr.append(cal_train[:, i])
                arrays_tf.append(cal_future[:, i])
    
    if not arrays_tr:
        return None, None
    
    exog_used = np.column_stack(arrays_tr) if len(arrays_tr) > 1 else arrays_tr[0].reshape(-1, 1)
    exog_future = np.column_stack(arrays_tf) if len(arrays_tf) > 1 else arrays_tf[0].reshape(-1, 1)
    
    return exog_used, exog_future


def apply_preprocessing(
    df: pd.DataFrame,
    quantity: str,
    target: str,
    denoise: Optional[Dict[str, Any]]
) -> str:
    """Apply preprocessing transformations to prepare data for forecasting.
    
    Returns the name of the column to use for forecasting.
    """
    # Apply denoising if specified
    if denoise:
        try:
            denoise_spec = _normalize_denoise_spec(denoise)
            _apply_denoise(df, denoise_spec, default_when='pre_ti')
        except Exception:
            pass
    
    # Determine base column based on quantity and target
    if quantity == 'volatility':
        base_col = 'close'  # Will compute volatility from returns
    elif quantity == 'return':
        base_col = 'close'  # Will compute returns
    else:  # price
        if target == 'return':
            base_col = 'close'  # Will compute returns from price
        else:
            base_col = 'close'  # Direct price forecasting
    
    return base_col


__all__ = [
    '_default_seasonality_period',
    '_next_times_from_last', 
    '_pd_freq_from_timeframe',
    'prepare_features',
    'apply_preprocessing'
]
