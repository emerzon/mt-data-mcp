from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..common import nf_setup_and_predict as _nf_setup_and_predict  # type: ignore


def forecast_neural(
    *,
    method: str,
    series: np.ndarray,
    fh: int,
    timeframe: str,
    n: int,
    m: int,
    params: Dict[str, Any],
    Y_df,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[list] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """NeuralForecast models (nhits, nbeatsx, tft, patchtst).

    Returns (forecast_values, params_used).
    """
    try:
        from neuralforecast.models import NHITS as _NF_NHITS, NBEATSx as _NF_NBEATSX, TFT as _NF_TFT, PatchTST as _NF_PATCHTST  # type: ignore
    except Exception as ex:
        raise RuntimeError(f"Failed to import neuralforecast models: {ex}")

    method_l = str(method).lower().strip()
    model_class = {
        'nhits': _NF_NHITS,
        'nbeatsx': _NF_NBEATSX,
        'tft': _NF_TFT,
        'patchtst': _NF_PATCHTST,
    }.get(method_l)
    if model_class is None:
        raise RuntimeError(f"Model '{method_l}' not available in installed neuralforecast version")

    # Hyperparameters with defaults and safety caps
    h = int(fh)
    input_size = None
    if params.get('input_size') is not None:
        requested = int(params['input_size'])
        input_size = int(min(requested, max(8, (n - h) if n > h else n)))
    else:
        base = max(64, (int(m) * 3) if m and int(m) > 0 else 96)
        input_size = int(min(n, base))
    steps = int(params.get('max_steps', params.get('max_epochs', 50)))
    batch_size = int(params.get('batch_size', 32))
    lr = params.get('learning_rate', None)

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
    pred_col = None
    for c in list(Yf.columns):
        if c not in ('unique_id', 'ds', 'y'):
            pred_col = c
            if c == 'y_hat':
                break
    if pred_col is None:
        raise RuntimeError(f"{method_l.upper()} prediction columns not found")
    vals = np.asarray(Yf[pred_col].to_numpy(), dtype=float)
    f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
    params_used = {'max_epochs': int(steps), 'input_size': int(input_size), 'batch_size': int(batch_size)}
    return f_vals.astype(float, copy=False), params_used

