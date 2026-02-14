from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common import nf_setup_and_predict as _nf_setup_and_predict  # type: ignore
from ..interface import ForecastMethod, ForecastResult
from ..registry import ForecastRegistry


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


class NeuralForecastMethod(ForecastMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "input_size", "type": "int|null", "description": "Lookback context for the model (auto if omitted)."},
        {"name": "max_steps", "type": "int|null", "description": "Training steps (fallback to max_epochs, default: 50)."},
        {"name": "max_epochs", "type": "int|null", "description": "Alias for max_steps."},
        {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)."},
        {"name": "learning_rate", "type": "float|null", "description": "Learning rate (model default if omitted)."},
    ]

    @property
    def category(self) -> str:
        return "neural"

    @property
    def required_packages(self) -> List[str]:
        return ["neuralforecast"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": False, "ci": False}

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> ForecastResult:
        from ..common import _create_training_dataframes

        p = dict(params or {})
        x = np.asarray(series.values, dtype=float)
        n = int(x.size)
        if n < 5:
            raise ValueError(f"{self.name} requires at least 5 observations")

        exog_used = kwargs.get("exog_used")
        if exog_used is None:
            exog_used = p.get("exog_used")
        exog_future_arr = kwargs.get("exog_future")
        if exog_future_arr is None:
            exog_future_arr = exog_future if exog_future is not None else p.get("exog_future")

        Y_df, _, _ = _create_training_dataframes(x, int(horizon), exog_used, exog_future_arr)
        f_vals, params_used = forecast_neural(
            method=self.name,
            series=x,
            fh=int(horizon),
            timeframe=str(p.get("timeframe") or kwargs.get("timeframe") or "H1"),
            n=n,
            m=int(seasonality or 0),
            params=p,
            Y_df=Y_df,
            exog_used=exog_used,
            exog_future=exog_future_arr,
            future_times=kwargs.get("future_times"),
        )
        return ForecastResult(forecast=f_vals, params_used=params_used)


@ForecastRegistry.register("nhits")
class NHITSMethod(NeuralForecastMethod):
    @property
    def name(self) -> str:
        return "nhits"


@ForecastRegistry.register("nbeatsx")
class NBEATSXMethod(NeuralForecastMethod):
    @property
    def name(self) -> str:
        return "nbeatsx"


@ForecastRegistry.register("tft")
class TFTMethod(NeuralForecastMethod):
    @property
    def name(self) -> str:
        return "tft"


@ForecastRegistry.register("patchtst")
class PatchTSTMethod(NeuralForecastMethod):
    @property
    def name(self) -> str:
        return "patchtst"
