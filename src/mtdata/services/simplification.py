from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.constants import SIMPLIFY_DEFAULT_MODE
from ..core.schema import SimplifySpec
from ..utils.simplify import (
    _choose_simplify_points,
    _handle_encode_mode as _handle_encode,
    _handle_resample_mode as _handle_resample,
    _handle_segment_mode as _handle_segment,
    _handle_symbolic_mode as _handle_symbolic,
    _select_indices_for_timeseries,
)


def _handle_select(df: pd.DataFrame, headers: List[str], spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Compatibility select implementation using shared utils primitives."""
    original_count = len(df)
    if original_count <= 2:
        return df, None

    n_out = _choose_simplify_points(original_count, spec)
    if n_out >= original_count:
        return df, None

    series = None
    if 'close' in df.columns:
        series = df['close'].values
    elif len(headers) > 1:
        for h in headers:
            if h != 'time' and h in df.columns:
                try:
                    series = df[h].astype(float).values
                    break
                except Exception:
                    pass

    if series is None:
        return df, None

    epochs = df['__epoch'].values if '__epoch' in df.columns else np.arange(original_count)
    idxs, method, params = _select_indices_for_timeseries(epochs, series, spec)
    simplified_df = df.iloc[idxs].copy()

    meta: Dict[str, Any] = {
        'mode': 'select',
        'method': method,
        'original_rows': int(original_count),
        'returned_rows': int(len(simplified_df)),
        'points': int(len(simplified_df)),
    }
    if params:
        meta.update(params)
    return simplified_df, meta


def _simplify_dataframe_rows_ext(
    df: pd.DataFrame,
    headers: List[str],
    simplify: SimplifySpec,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Compatibility dispatcher for service-level imports."""
    if df.empty:
        return df, None

    spec = dict(simplify) if simplify else {}
    mode = str(spec.get('mode', SIMPLIFY_DEFAULT_MODE)).lower().strip() or SIMPLIFY_DEFAULT_MODE

    if mode == 'resample':
        return _handle_resample(df, headers, spec)
    if mode == 'encode':
        return _handle_encode(df, headers, spec)
    if mode == 'segment':
        return _handle_segment(df, headers, spec)
    if mode == 'symbolic':
        return _handle_symbolic(df, headers, spec)
    return _handle_select(df, headers, spec)
