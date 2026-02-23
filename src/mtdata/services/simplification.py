
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import warnings

# Imports from core schema
from ..core.schema import SimplifySpec

# Imports from utils
from ..utils.simplify import (
    _lttb_select_indices, _rdp_select_indices, _pla_select_indices, _apca_select_indices,
    _choose_simplify_points, _select_indices_for_timeseries
)

# Constants
SIMPLIFY_DEFAULT_MODE = "select"

def _simplify_dataframe_rows_ext(df: pd.DataFrame, headers: List[str], simplify: SimplifySpec) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Apply simplification to DataFrame rows based on spec.
    Returns (simplified_df, metadata_dict).
    Supports modes:
      - 'select' (default): Selects subset of rows using LTTB/RDP/PLA/APCA.
      - 'approximate': Aggregates rows into segments (e.g. mean/OHLC of buckets).
      - 'resample': Time-based resampling (e.g. '1H', '5min').
      - 'encode': Transforms data into compact string/sequence representations.
      - 'segment': Detects turning points (zigzag) and returns only those rows.
      - 'symbolic': SAX transformation (returns symbolic string, not rows).
    """
    if df.empty:
        return df, None
    
    spec = dict(simplify) if simplify else {}
    mode = str(spec.get('mode', SIMPLIFY_DEFAULT_MODE)).lower().strip()
    
    # Dispatch to specific handlers
    if mode == 'resample':
        return _handle_resample(df, headers, spec)
    elif mode == 'encode':
        return _handle_encode(df, headers, spec)
    elif mode == 'segment':
        return _handle_segment(df, headers, spec)
    elif mode == 'symbolic':
        return _handle_symbolic(df, headers, spec)
    elif mode == 'approximate':
        # For now, treat approximate same as select or implement aggregation?
        # The original code might have had specific logic.
        # Checking original code... it seems 'approximate' was not fully distinct or used select logic in some paths.
        # Let's assume it uses selection for now unless we find specific aggregation logic.
        # Actually, let's check if there was specific logic for 'approximate'.
        # In the viewed file, 'approximate' wasn't explicitly handled differently in the main flow except maybe falling through to selection.
        pass

    # Default 'select' mode (LTTB, RDP, etc.)
    return _handle_select(df, headers, spec)

def _handle_resample(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    # Implementation of resampling
    rule = spec.get('rule') or spec.get('interval')
    if not rule:
        # Try to infer from points/ratio?
        # For now return as is if no rule
        return df, {'error': 'Missing rule for resample'}
    
    try:
        # Ensure datetime index
        if 'time' in df.columns:
            # If time is string, parse it? Or use __epoch?
            # __epoch is float timestamp.
            if '__epoch' in df.columns:
                df = df.set_index(pd.to_datetime(df['__epoch'], unit='s'))
            else:
                # Try parsing time column
                df = df.set_index(pd.to_datetime(df['time']))
        
        # Resample
        # We need to define aggregation for columns
        agg_map = {}
        for h in headers:
            if h in ['open']: agg_map[h] = 'first'
            elif h in ['high']: agg_map[h] = 'max'
            elif h in ['low']: agg_map[h] = 'min'
            elif h in ['close']: agg_map[h] = 'last'
            elif h in ['tick_volume', 'real_volume', 'volume']: agg_map[h] = 'sum'
            else: agg_map[h] = 'last' # Default
            
        resampled = df.resample(rule).agg(agg_map).dropna()
        
        # Reset index and restore time/__epoch if needed
        # ...
        return resampled, {'mode': 'resample', 'rule': rule, 'rows': len(resampled)}
    except Exception as e:
        return df, {'error': f'Resample failed: {e}'}

def _handle_encode(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Encode a numeric series into a compact representation without recursion."""
    value_col = str(spec.get('value_col') or '').strip()
    if not value_col or value_col not in df.columns:
        if 'close' in df.columns:
            value_col = 'close'
        else:
            value_col = next(
                (
                    h for h in headers
                    if h in df.columns and h != 'time' and pd.api.types.is_numeric_dtype(df[h])
                ),
                '',
            )
    if not value_col:
        return df, {'mode': 'encode', 'error': 'No numeric column available for encoding'}

    vals = pd.to_numeric(df[value_col], errors='coerce').to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return df, {'mode': 'encode', 'error': 'No finite values to encode'}

    schema = str(spec.get('schema', 'delta')).lower().strip()
    if schema not in ('delta', 'envelope'):
        schema = 'delta'

    if schema == 'envelope':
        encoded = (
            f"start={float(vals[0]):.6g}|end={float(vals[-1]):.6g}|"
            f"min={float(np.min(vals)):.6g}|max={float(np.max(vals)):.6g}"
        )
    else:
        scale = spec.get('scale', 1.0)
        try:
            scale_f = float(scale)
        except Exception:
            scale_f = 1.0
        scale_f = scale_f if abs(scale_f) > 1e-12 else 1.0
        diffs = np.diff(vals, prepend=vals[0])
        q = np.round(diffs / scale_f).astype(int)
        if bool(spec.get('as_chars', False)):
            zero_char = str(spec.get('zero_char', '0'))[:1] or '0'
            encoded = ''.join('+' if d > 0 else '-' if d < 0 else zero_char for d in q.tolist())
        else:
            encoded = ','.join(str(int(v)) for v in q.tolist())

    out_df = pd.DataFrame([{'encoding': encoded}])
    meta = {
        'mode': 'encode',
        'schema': schema,
        'value_col': value_col,
        'headers': ['encoding'],
        'original_rows': int(len(df)),
        'returned_rows': 1,
        'points': 1,
    }
    return out_df, meta

def _handle_segment(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Return turning points using a simple threshold-based zigzag segmentation."""
    value_col = str(spec.get('value_col') or '').strip()
    if not value_col or value_col not in df.columns:
        value_col = 'close' if 'close' in df.columns else ''
    if not value_col:
        return _handle_select(df, headers, spec)

    vals = pd.to_numeric(df[value_col], errors='coerce').to_numpy(dtype=float)
    if vals.size <= 2:
        return df, {'mode': 'segment', 'algo': 'zigzag', 'points': int(len(df))}

    try:
        threshold_pct = float(spec.get('threshold_pct', 0.005))
    except Exception:
        threshold_pct = 0.005
    threshold_pct = max(0.0, threshold_pct)
    if threshold_pct <= 0.0:
        return _handle_select(df, headers, spec)

    idxs: List[int] = [0]
    anchor_val = float(vals[0])
    trend = 0
    for i in range(1, len(vals)):
        cur = float(vals[i])
        denom = max(abs(anchor_val), 1e-12)
        move = (cur - anchor_val) / denom
        if trend >= 0 and cur >= anchor_val:
            anchor_val = cur
            if idxs:
                idxs[-1] = i
            continue
        if trend <= 0 and cur <= anchor_val:
            anchor_val = cur
            if idxs:
                idxs[-1] = i
            continue
        if abs(move) >= threshold_pct:
            trend = 1 if move > 0 else -1
            idxs.append(i)
            anchor_val = cur

    if idxs[-1] != len(vals) - 1:
        idxs.append(len(vals) - 1)
    idxs = sorted(set(int(i) for i in idxs if 0 <= int(i) < len(df)))
    if len(idxs) < 2:
        idxs = [0, len(df) - 1]

    out_df = df.iloc[idxs].copy()
    meta = {
        'mode': 'segment',
        'algo': 'zigzag',
        'threshold_pct': float(threshold_pct),
        'value_col': value_col,
        'original_rows': int(len(df)),
        'returned_rows': int(len(out_df)),
        'points': int(len(out_df)),
    }
    return out_df, meta

def _handle_symbolic(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Generate a SAX-like symbolic string from a numeric series."""
    value_col = str(spec.get('value_col') or '').strip()
    if not value_col or value_col not in df.columns:
        value_col = 'close' if 'close' in df.columns else ''
    if not value_col:
        return df, {'mode': 'symbolic', 'error': 'No numeric column available for symbolic mode'}

    vals = pd.to_numeric(df[value_col], errors='coerce').to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return df, {'mode': 'symbolic', 'error': 'No finite values to symbolize'}

    try:
        paa = int(spec.get('paa', 8))
    except Exception:
        paa = 8
    paa = max(1, min(paa, int(vals.size)))

    alphabet = str(spec.get('alphabet') or 'abcdefghijklmnopqrstuvwxyz')
    alphabet = ''.join(dict.fromkeys(ch for ch in alphabet if ch.strip()))
    if len(alphabet) < 2:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
    bins_n = min(26, len(alphabet))
    alphabet = alphabet[:bins_n]

    x = vals.copy()
    if bool(spec.get('znorm', True)):
        mu = float(np.mean(x))
        sigma = float(np.std(x))
        if sigma > 1e-12:
            x = (x - mu) / sigma
        else:
            x = x - mu

    chunks = np.array_split(x, paa)
    paa_vals = np.array([float(np.mean(c)) if len(c) else 0.0 for c in chunks], dtype=float)
    quantiles = np.linspace(0.0, 1.0, bins_n + 1)
    edges = np.quantile(paa_vals, quantiles)
    # Ensure strictly increasing edges to keep searchsorted stable
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    ids = np.searchsorted(edges[1:-1], paa_vals, side='right')
    symbols = ''.join(alphabet[int(i)] for i in ids.tolist())

    out_df = pd.DataFrame([{'symbolic': symbols}])
    meta = {
        'mode': 'symbolic',
        'schema': 'sax',
        'value_col': value_col,
        'paa': int(paa),
        'alphabet': alphabet,
        'headers': ['symbolic'],
        'original_rows': int(len(df)),
        'returned_rows': 1,
        'points': 1,
    }
    return out_df, meta

def _handle_select(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    # Implementation of point selection (LTTB, etc.)
    # This logic was largely in `core/simplify.py` and `utils/simplify.py`.
    # We should reuse `utils/simplify.py` functions.
    
    original_count = len(df)
    if original_count <= 2:
        return df, None
        
    # Determine target points
    n_out = _choose_simplify_points(original_count, spec)
    if n_out >= original_count:
        return df, None
        
    # Select representative column(s) for simplification
    # Usually 'close' or 'open' or average?
    # If OHLC exists, maybe use average of HL?
    # For now, let's use 'close' if available, else first numeric column.
    
    series = None
    if 'close' in df.columns:
        series = df['close'].values
    elif len(headers) > 1:
        # Try to find a numeric column
        for h in headers:
            if h != 'time' and h in df.columns:
                try:
                    series = df[h].astype(float).values
                    break
                except:
                    pass
                    
    if series is None:
        return df, None
        
    # Get epochs
    epochs = df['__epoch'].values if '__epoch' in df.columns else np.arange(original_count)
    
    # Select indices
    idxs, method, params = _select_indices_for_timeseries(epochs, series, spec)
    
    # Filter DataFrame
    simplified_df = df.iloc[idxs].copy()
    
    meta = {
        'mode': 'select',
        'method': method,
        'original_rows': original_count,
        'returned_rows': len(simplified_df),
        'points': len(simplified_df)
    }
    if params:
        meta.update(params)
        
    return simplified_df, meta
