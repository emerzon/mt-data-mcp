"""Target series construction and transformation logic."""
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.indicators import _apply_ta_indicators as _apply_ta_indicators_util
from ..utils.indicators import _parse_ti_specs as _parse_ti_specs_util

logger = logging.getLogger(__name__)


def resolve_alias_base(arrs: Dict[str, np.ndarray], name: str) -> Optional[np.ndarray]:
    """Resolve alias base columns like 'typical', 'hl2', 'ohlc4'."""
    nm = name.strip().lower()
    if nm in ('typical', 'tp'):
        if all(k in arrs for k in ('high', 'low', 'close')):
            return (arrs['high'] + arrs['low'] + arrs['close']) / 3.0
        return None
    if nm == 'hl2':
        if all(k in arrs for k in ('high', 'low')):
            return (arrs['high'] + arrs['low']) / 2.0
        return None
    if nm in ('ohlc4', 'ha_close', 'haclose'):
        if all(k in arrs for k in ('open', 'high', 'low', 'close')):
            return (arrs['open'] + arrs['high'] + arrs['low'] + arrs['close']) / 4.0
        return None
    return None


def build_target_series(
    df: pd.DataFrame,
    base_col: str,
    target_spec: Optional[Dict[str, Any]] = None,
    quantity: str = 'price',
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build target series from base column with optional transformations.
    
    Returns:
        (y_array, target_info_dict)
    """
    target_info: Dict[str, Any] = {}
    
    if not target_spec or not isinstance(target_spec, dict):
        if str(quantity).strip().lower() == 'return':
            log_prices = np.log(np.maximum(df[base_col].astype(float).to_numpy(), 1e-12))
            y = np.full(log_prices.shape, np.nan, dtype=float)
            if log_prices.size > 1:
                y[1:] = np.diff(log_prices)
            target_info = {'mode': 'return', 'base': base_col, 'transform': 'log_return'}
        else:
            y = df[base_col].astype(float).to_numpy()
            target_info = {'mode': 'price', 'base': base_col, 'transform': 'none'}
        return y, target_info
    
    # Custom target_spec mode
    ts = dict(target_spec)
    
    # Compute indicators if requested
    ts_inds = ts.get('indicators')
    if ts_inds:
        try:
            specs = _parse_ti_specs_util(str(ts_inds)) if isinstance(ts_inds, str) else ts_inds
            _apply_ta_indicators_util(df, specs, default_when='pre_ti')
        except Exception as exc:
            logger.warning("Failed to apply target_spec indicators %r: %s", ts_inds, exc)
            raise ValueError(f"Failed to apply target_spec indicators: {exc}") from exc
    
    base_name = str(ts.get('base', ts.get('column', base_col)))
    
    # Resolve base series
    if base_name in df.columns:
        y_base = df[base_name].astype(float).to_numpy()
    else:
        # Try alias resolution
        arrs = {c: df[c].to_numpy() for c in df.columns if c in ('open', 'high', 'low', 'close')}
        y_base = resolve_alias_base(arrs, base_name)
        if y_base is None:
            raise ValueError(f"Base column '{base_name}' not found and not a recognized alias")
    
    target_info['base'] = base_name
    
    # Apply transform
    transform = str(ts.get('transform', 'none')).lower()
    k = int(ts.get('k', 1))
    if k < 1:
        k = 1
    
    if transform == 'none':
        y = y_base
        target_info['transform'] = 'none'
    elif transform in ('return',):
        prev = np.roll(y_base, k)
        with np.errstate(divide='ignore', invalid='ignore'):
            y = (y_base - prev) / np.where(np.abs(prev) > 1e-12, prev, 1.0)
        y = y.astype(float, copy=False)
        y[:k] = np.nan
        target_info['transform'] = f'return(k={k})'
    elif transform == 'log_return':
        prev = np.roll(y_base, k)
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.log(np.maximum(y_base, 1e-12)) - np.log(np.maximum(prev, 1e-12))
        y = y.astype(float, copy=False)
        y[:k] = np.nan
        target_info['transform'] = f'log_return(k={k})'
    elif transform == 'log':
        y = np.log(np.maximum(y_base, 1e-12))
        target_info['transform'] = 'log'
    elif transform == 'diff':
        prev = np.roll(y_base, k)
        y = y_base - prev
        y = y.astype(float, copy=False)
        y[:k] = np.nan
        target_info['transform'] = f'diff(k={k})'
    elif transform in ('pct_change', 'pct'):
        prev = np.roll(y_base, k)
        with np.errstate(divide='ignore', invalid='ignore'):
            y = (y_base - prev) / np.where(np.abs(prev) > 1e-12, prev, 1.0)
        if transform == 'pct':
            y = 100.0 * y
        y = y.astype(float, copy=False)
        y[:k] = np.nan
        target_info['transform'] = f'{"pct" if transform == "pct" else "pct_change"}(k={k})'
    else:
        y = y_base
        target_info['transform'] = 'none'
    
    target_info['mode'] = 'custom'
    return y, target_info


def aggregate_horizon_target(
    y: np.ndarray,
    horizon: int,
    agg_spec: Optional[str] = None,
    normalize: str = 'none',
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Aggregate target over horizon windows.
    
    Args:
        y: Target series
        horizon: Window size
        agg_spec: 'last', 'mean', 'sum', 'slope', 'max', 'min', 'range', 'vol'
        normalize: 'none', 'per_bar', 'pct'
    
    Returns:
        (aggregated_array, agg_info_dict)
    """
    arr = np.asarray(y, dtype=float).ravel()
    horizon_i = max(1, int(horizon))
    agg_name = str(agg_spec or 'last').strip().lower()
    normalize_mode = str(normalize or 'none').strip().lower()

    if agg_name == 'last' or arr.size == 0 or horizon_i <= 1:
        return arr.astype(float, copy=False), {'agg': 'last' if agg_name == 'last' else agg_name, 'normalize': 'none' if agg_name == 'last' else normalize_mode}

    if agg_name not in {'mean', 'sum', 'slope', 'max', 'min', 'range', 'vol'}:
        return arr.astype(float, copy=False), {'agg': agg_name, 'normalize': normalize_mode}

    result = np.full(arr.shape, np.nan, dtype=float)
    if arr.size < horizon_i:
        return result, {'agg': agg_name, 'normalize': normalize_mode, 'horizon': horizon_i, 'aligned': 'forward'}

    windows = np.lib.stride_tricks.sliding_window_view(arr, horizon_i)
    valid_mask = np.all(np.isfinite(windows), axis=1)
    raw = np.full(windows.shape[0], np.nan, dtype=float)
    if np.any(valid_mask):
        valid_windows = windows[valid_mask]
        if agg_name == 'mean':
            raw[valid_mask] = np.mean(valid_windows, axis=1)
        elif agg_name == 'sum':
            raw[valid_mask] = np.sum(valid_windows, axis=1)
        elif agg_name == 'max':
            raw[valid_mask] = np.max(valid_windows, axis=1)
        elif agg_name == 'min':
            raw[valid_mask] = np.min(valid_windows, axis=1)
        elif agg_name == 'range':
            raw[valid_mask] = np.max(valid_windows, axis=1) - np.min(valid_windows, axis=1)
        elif agg_name == 'vol':
            ddof = 1 if horizon_i > 1 else 0
            raw[valid_mask] = np.std(valid_windows, axis=1, ddof=ddof)
        elif agg_name == 'slope':
            x = np.arange(horizon_i, dtype=float)
            x_centered = x - np.mean(x)
            denom = float(np.sum(x_centered * x_centered))
            centered = valid_windows - np.mean(valid_windows, axis=1, keepdims=True)
            raw[valid_mask] = (centered @ x_centered) / max(denom, 1e-12)

        if normalize_mode == 'per_bar':
            if agg_name in {'sum', 'range'}:
                raw[valid_mask] = raw[valid_mask] / float(horizon_i)
        elif normalize_mode == 'pct':
            base = np.abs(valid_windows[:, 0])
            pct_vals = np.full(valid_windows.shape[0], np.nan, dtype=float)
            base_ok = base > 1e-12
            if np.any(base_ok):
                pct_vals[base_ok] = 100.0 * raw[valid_mask][base_ok] / base[base_ok]
            raw[valid_mask] = pct_vals

    result[: raw.shape[0]] = raw
    return result, {'agg': agg_name, 'normalize': normalize_mode, 'horizon': horizon_i, 'aligned': 'forward'}
