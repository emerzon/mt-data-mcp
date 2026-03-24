"""Target series construction and transformation logic."""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from ..utils.indicators import _parse_ti_specs as _parse_ti_specs_util, _apply_ta_indicators as _apply_ta_indicators_util


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
        except Exception:
            pass
    
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
        target_info['transform'] = f'pct_change(k={k})'
    else:
        y = y_base
        target_info['transform'] = 'none'
    
    target_info['mode'] = 'custom'
    return y, target_info
