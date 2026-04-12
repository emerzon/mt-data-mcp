from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common import _extract_forecast_values
from ..common import edge_pad_to_length as _edge_pad_to_length  # type: ignore
from ..common import nf_setup_and_predict as _nf_setup_and_predict
from ..common import (
    _NF_ENV_LOCK,
    _NfEnvGuard,
    _nf_resolve_accelerator,
    nf_build_model_kwargs,
    nf_create_and_fit,
    nf_predict_from_fitted,
)
from ..interface import (
    ForecastMethod,
    ForecastResult,
    ProgressCallback,
    TrainingProgress,
    TrainResult,
)
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
        from neuralforecast.models import NHITS as _NF_NHITS  # type: ignore
        from neuralforecast.models import TFT as _NF_TFT
        from neuralforecast.models import NBEATSx as _NF_NBEATSX
        from neuralforecast.models import PatchTST as _NF_PATCHTST
    except Exception as ex:
        raise RuntimeError(f"Failed to import neuralforecast models: {ex}")

    method_l = str(method).lower().strip()
    model_map = {
        'nhits': _NF_NHITS,
        'nbeatsx': _NF_NBEATSX,
        'tft': _NF_TFT,
        'patchtst': _NF_PATCHTST,
    }
    model_class = model_map.get(method_l)
    if model_class is None:
        available_models = ", ".join(sorted(model_map))
        raise RuntimeError(
            f"Unsupported NeuralForecast model '{method_l}'. "
            f"Supported mtdata neural models: {available_models}"
        )

    # Hyperparameters with defaults and safety caps
    h = int(fh)
    available_context = int(max(1, (n - h) if n > h else n))
    input_size = None
    if params.get('input_size') is not None:
        requested = int(params['input_size'])
        input_size = int(max(1, min(requested, available_context)))
    else:
        base = max(64, (int(m) * 3) if m and int(m) > 0 else 96)
        input_size = int(max(1, min(available_context, base)))
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
    f_vals = _extract_forecast_values(
        Yf,
        int(fh),
        f"{method_l.upper()} forecast",
        allow_actual_fallback=False,
    )
    params_used = {'max_epochs': int(steps), 'input_size': int(input_size), 'batch_size': int(batch_size)}
    return f_vals.astype(float, copy=False), params_used


def _resolve_nf_model_class(method_name: str):
    """Return the NeuralForecast model class for *method_name*."""
    try:
        from neuralforecast.models import NHITS as _NF_NHITS
        from neuralforecast.models import TFT as _NF_TFT
        from neuralforecast.models import NBEATSx as _NF_NBEATSX
        from neuralforecast.models import PatchTST as _NF_PATCHTST
    except Exception as ex:
        raise RuntimeError(f"Failed to import neuralforecast models: {ex}")
    model_map = {
        'nhits': _NF_NHITS, 'nbeatsx': _NF_NBEATSX,
        'tft': _NF_TFT, 'patchtst': _NF_PATCHTST,
    }
    cls = model_map.get(str(method_name).lower().strip())
    if cls is None:
        raise RuntimeError(f"Unknown neural method: {method_name}")
    return cls


def _neural_resolve_hyperparams(
    params: Dict[str, Any], n: int, fh: int, m: int,
) -> Tuple[int, int, int, Optional[float]]:
    """Return (input_size, steps, batch_size, learning_rate)."""
    h = int(fh)
    available_context = int(max(1, (n - h) if n > h else n))
    if params.get('input_size') is not None:
        requested = int(params['input_size'])
        input_size = int(max(1, min(requested, available_context)))
    else:
        base = max(64, (int(m) * 3) if m and int(m) > 0 else 96)
        input_size = int(max(1, min(available_context, base)))
    steps = int(params.get('max_steps', params.get('max_epochs', 50)))
    batch_size = int(params.get('batch_size', 32))
    lr = params.get('learning_rate', None)
    return input_size, steps, batch_size, float(lr) if lr is not None else None


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

    # ------------------------------------------------------------------
    # Train / predict lifecycle
    # ------------------------------------------------------------------

    @property
    def supports_training(self) -> bool:
        return True

    @property
    def training_category(self):
        return "heavy"

    def train(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        progress_callback: Optional[ProgressCallback] = None,
        exog: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> TrainResult:
        from ..common import _create_training_dataframes

        p = dict(params or {})
        x = np.asarray(series.values, dtype=float)
        n = int(x.size)
        if n < 5:
            raise ValueError(f"{self.name} requires at least 5 observations")

        exog_used = exog if exog is not None else p.get("exog_used")
        exog_future_arr = p.get("exog_future")

        input_size, steps, batch_size, lr = _neural_resolve_hyperparams(
            p, n, int(horizon), int(seasonality or 0),
        )
        timeframe = str(p.get("timeframe") or kwargs.get("timeframe") or "H1")
        model_class = _resolve_nf_model_class(self.name)

        Y_df, _, _ = _create_training_dataframes(x, int(horizon), exog_used, exog_future_arr)

        if progress_callback:
            progress_callback(TrainingProgress(
                step=0, total_steps=steps, message=f"Starting {self.name} training",
            ))

        accel = _nf_resolve_accelerator()
        model_kwargs = nf_build_model_kwargs(
            model_class=model_class,
            fh=int(horizon),
            input_size=input_size,
            batch_size=batch_size,
            steps=steps,
            learning_rate=lr,
            accel=accel,
        )

        with _NF_ENV_LOCK:
            with _NfEnvGuard(accel):
                nf = nf_create_and_fit(
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    timeframe=timeframe,
                    Y_df=Y_df,
                    exog_used=exog_used,
                )

        if progress_callback:
            progress_callback(TrainingProgress(
                step=steps, total_steps=steps, message="Training complete",
            ))

        artifact_bytes = self.serialize_artifact(nf)
        params_used = {
            'max_epochs': steps, 'input_size': input_size,
            'batch_size': batch_size,
        }
        return TrainResult(
            artifact_bytes=artifact_bytes,
            params_used=params_used,
            metadata={"accelerator": accel, "timeframe": timeframe},
        )

    def predict_with_model(
        self,
        model: Any,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> ForecastResult:
        nf = model  # deserialized NeuralForecast object
        p = dict(params or {})
        exog_future_arr = kwargs.get("exog_future")
        if exog_future_arr is None:
            exog_future_arr = exog_future if exog_future is not None else p.get("exog_future")

        accel = _nf_resolve_accelerator()
        with _NF_ENV_LOCK:
            with _NfEnvGuard(accel):
                Yf = nf_predict_from_fitted(
                    nf,
                    fh=int(horizon),
                    exog_future=exog_future_arr if isinstance(exog_future_arr, np.ndarray) else None,
                    future_times=kwargs.get("future_times"),
                )

        try:
            Yf = Yf[Yf['unique_id'] == 'ts']
        except Exception:
            pass
        f_vals = _extract_forecast_values(
            Yf, int(horizon),
            f"{self.name.upper()} forecast",
            allow_actual_fallback=False,
        )
        params_used = dict(p)
        return ForecastResult(forecast=f_vals.astype(float, copy=False), params_used=params_used)

    def training_fingerprint(
        self,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        timeframe: Optional[str] = None,
        has_exog: bool = False,
    ) -> Dict[str, Any]:
        fp = super().training_fingerprint(
            horizon, seasonality, params,
            timeframe=timeframe, has_exog=has_exog,
        )
        # Neural models also depend on input_size and batch_size
        p = params or {}
        fp["input_size"] = p.get("input_size")
        fp["batch_size"] = int(p.get("batch_size", 32))
        return fp

    # ------------------------------------------------------------------
    # Original forecast() — unchanged for backward compatibility
    # ------------------------------------------------------------------

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
