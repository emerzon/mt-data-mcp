from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..interface import ForecastMethod, ForecastResult
from ..registry import ForecastRegistry


def _build_list_dataset(series: np.ndarray, freq: str):
    try:
        from gluonts.dataset.common import ListDataset  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as ex:  # pragma: no cover
        raise RuntimeError(f"gluonts dataset deps missing: {ex}")

    idx = pd.date_range(start=pd.Timestamp('2000-01-01'), periods=len(series), freq=freq)
    return ListDataset([
        {
            'target': np.asarray(series, dtype=np.float32),
            'start': idx[0],
        }
    ], freq=freq)


def _extract_forecast_arrays(forecast_obj, fh: int, quantiles: Optional[List[float]]):
    import numpy as _np  # type: ignore
    vals = None
    try:
        vals = _np.asarray(forecast_obj.mean, dtype=float)
    except Exception:
        pass
    if vals is None or vals.size == 0:
        try:
            vals = _np.asarray(forecast_obj.quantile(0.5), dtype=float)
        except Exception:
            pass
    if (vals is None or vals.size == 0) and hasattr(forecast_obj, 'samples'):
        try:
            vals = _np.asarray(_np.mean(forecast_obj.samples, axis=0), dtype=float)
        except Exception:
            pass
    if vals is None:
        return None, None
    f_vals = vals[:fh] if vals.size >= fh else _np.pad(vals, (0, fh - vals.size), mode='edge')

    fq: Dict[str, List[float]] = {}
    if quantiles:
        for q in quantiles:
            try:
                qf = float(q)
            except Exception:
                continue
            try:
                q_arr = _np.asarray(forecast_obj.quantile(qf), dtype=float)
            except Exception:
                continue
            fq[str(qf)] = [float(v) for v in q_arr[:fh].tolist()]
    return f_vals, (fq or None)


def forecast_gt_deepar(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Forecast using GluonTS DeepAR (quick train on single series)."""
    p = params or {}
    ctx_len = int(p.get('context_length', min(64, int(n))))
    freq = str(p.get('freq', 'H'))
    epochs = int(p.get('train_epochs', 5))
    batch_size = int(p.get('batch_size', 32))
    lr = float(p.get('learning_rate', 1e-3))
    hidden_size = int(p.get('hidden_size', 40))
    num_layers = int(p.get('num_layers', 2))
    dropout = float(p.get('dropout', 0.1))
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None

    try:
        try:
            from gluonts.torch import DeepAREstimator  # type: ignore
        except Exception:
            from gluonts.torch.model.deepar import DeepAREstimator  # type: ignore
        from gluonts.evaluation import make_evaluation_predictions  # type: ignore
    except Exception as ex:
        return (None, None, {}, f"deepar requires gluonts[torch]. Install/upgrade: pip install 'gluonts[torch]' torch ({ex})")

    # Data
    ds = _build_list_dataset(series, freq=freq)

    try:
        estimator = DeepAREstimator(
            prediction_length=int(fh),
            context_length=int(ctx_len),
            freq=freq,
            batch_size=int(batch_size),
            lr=float(lr),
            trainer_kwargs={"max_epochs": int(epochs)},
            dropout_rate=dropout,
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
        predictor = estimator.train(training_data=ds)
        forecast_it = predictor.predict(ds)
        forecasts = list(forecast_it)
        if not forecasts:
            return (None, None, {}, "deepar produced no forecasts")
        f_vals, fq = _extract_forecast_arrays(forecasts[0], int(fh), quantiles)
        if f_vals is None:
            return (None, None, {}, "deepar could not extract forecast values")
    except Exception as ex:
        return (None, None, {}, f"deepar error: {ex}")

    params_used: Dict[str, Any] = {
        'context_length': int(ctx_len),
        'freq': freq,
        'train_epochs': int(epochs),
        'batch_size': int(batch_size),
        'learning_rate': float(lr),
        'hidden_size': int(hidden_size),
        'num_layers': int(num_layers),
        'dropout': float(dropout),
    }
    if fq:
        params_used['quantiles'] = sorted(fq.keys(), key=lambda x: float(x))
    return (f_vals, fq, params_used, None)


def forecast_gt_sfeedforward(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Forecast using GluonTS SimpleFeedForward (quick train)."""
    p = params or {}
    ctx_len = int(p.get('context_length', min(64, int(n))))
    freq = str(p.get('freq', 'H'))
    epochs = int(p.get('train_epochs', 5))
    batch_size = int(p.get('batch_size', 32))
    lr = float(p.get('learning_rate', 1e-3))
    hidden_dim = int(p.get('hidden_dim', 64))
    num_hidden_layers = int(p.get('num_hidden_layers', 2))
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None

    try:
        try:
            from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator  # type: ignore
        except Exception:
            from gluonts.torch import SimpleFeedForwardEstimator  # type: ignore
    except Exception as ex:
        return (None, None, {}, f"simple_feedforward requires gluonts[torch]. Install/upgrade: pip install 'gluonts[torch]' torch ({ex})")

    ds = _build_list_dataset(series, freq=freq)

    try:
        # Some versions expect hidden_dimensions=list[int] instead of hidden_dim/num_hidden_layers
        hidden_dimensions = [int(hidden_dim)] * int(max(1, num_hidden_layers))
        estimator = SimpleFeedForwardEstimator(
            prediction_length=int(fh),
            context_length=int(ctx_len),
            freq=freq,
            batch_size=int(batch_size),
            lr=float(lr),
            trainer_kwargs={"max_epochs": int(epochs)},
            hidden_dimensions=hidden_dimensions,
        )
        predictor = estimator.train(training_data=ds)
        forecasts = list(predictor.predict(ds))
        if not forecasts:
            return (None, None, {}, "simple_feedforward produced no forecasts")
        f_vals, fq = _extract_forecast_arrays(forecasts[0], int(fh), quantiles)
        if f_vals is None:
            return (None, None, {}, "simple_feedforward could not extract forecast values")
    except Exception as ex:
        return (None, None, {}, f"simple_feedforward error: {ex}")

    params_used: Dict[str, Any] = {
        'context_length': int(ctx_len),
        'freq': freq,
        'train_epochs': int(epochs),
        'batch_size': int(batch_size),
        'learning_rate': float(lr),
        'hidden_dim': int(hidden_dim),
        'num_hidden_layers': int(num_hidden_layers),
    }
    if fq:
        params_used['quantiles'] = sorted(fq.keys(), key=lambda x: float(x))
    return (f_vals, fq, params_used, None)


def forecast_gt_prophet(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Forecast using GluonTS Prophet wrapper (requires prophet).

    Params (optional unless noted):
    - freq: pandas frequency string (default inferred by caller)
    - prophet_params: dict passed through to ProphetPredictor (e.g., growth, seasonality_mode,
      yearly_seasonality, weekly_seasonality, daily_seasonality, changepoint_prior_scale).
    """
    p = params or {}
    freq = str(p.get('freq', 'H'))
    prophet_params = p.get('prophet_params') if isinstance(p.get('prophet_params'), dict) else {}

    try:
        from gluonts.model.prophet import ProphetPredictor  # type: ignore
    except Exception as ex:
        return (None, None, {}, f"prophet requires gluonts[prophet] and prophet. Install: pip install gluonts prophet ({ex})")

    ds = _build_list_dataset(series, freq=freq)

    try:
        predictor = ProphetPredictor(freq=freq, prediction_length=int(fh), **dict(prophet_params))
        forecasts = list(predictor.predict(ds))
        if not forecasts:
            return (None, None, {}, "prophet produced no forecasts")
        f_vals, fq = _extract_forecast_arrays(forecasts[0], int(fh), p.get('quantiles'))
        if f_vals is None:
            return (None, None, {}, "prophet could not extract forecast values")
    except Exception as ex:
        return (None, None, {}, f"prophet error: {ex}")

    params_used: Dict[str, Any] = {
        'freq': freq,
        'prophet_params': dict(prophet_params),
    }
    if fq:
        params_used['quantiles'] = sorted(fq.keys(), key=lambda x: float(x))
    return (f_vals, fq, params_used, None)


def forecast_gt_tft(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Forecast using GluonTS Temporal Fusion Transformer (PyTorch).

    Minimal setup for single-series quick training.
    """
    p = params or {}
    ctx_len = int(p.get('context_length', min(128, int(n))))
    freq = str(p.get('freq', 'H'))
    epochs = int(p.get('train_epochs', 5))
    batch_size = int(p.get('batch_size', 32))
    lr = float(p.get('learning_rate', 1e-3))
    dropout = float(p.get('dropout', 0.1))
    hidden_size = int(p.get('hidden_size', 64))
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None

    try:
        try:
            from gluonts.torch.model.tft import TemporalFusionTransformerEstimator  # type: ignore
        except Exception:
            from gluonts.torch import TemporalFusionTransformerEstimator  # type: ignore
    except Exception as ex:
        return (None, None, {}, f"tft requires gluonts[torch]. Install/upgrade: pip install 'gluonts[torch]' torch ({ex})")

    ds = _build_list_dataset(series, freq=freq)

    try:
        estimator = TemporalFusionTransformerEstimator(
            prediction_length=int(fh),
            context_length=int(ctx_len),
            freq=freq,
            batch_size=int(batch_size),
            lr=float(lr),
            trainer_kwargs={"max_epochs": int(epochs)},
            dropout_rate=dropout,
            hidden_size=hidden_size,
        )
        predictor = estimator.train(training_data=ds)
        forecasts = list(predictor.predict(ds))
        if not forecasts:
            return (None, None, {}, "tft produced no forecasts")
        f_vals, fq = _extract_forecast_arrays(forecasts[0], int(fh), quantiles)
        if f_vals is None:
            return (None, None, {}, "tft could not extract forecast values")
    except Exception as ex:
        return (None, None, {}, f"tft error: {ex}")

    params_used: Dict[str, Any] = {
        'context_length': int(ctx_len), 'freq': freq,
        'train_epochs': int(epochs), 'batch_size': int(batch_size), 'learning_rate': float(lr),
        'hidden_size': int(hidden_size), 'dropout': float(dropout),
    }
    if fq:
        params_used['quantiles'] = sorted(fq.keys(), key=lambda x: float(x))
    return (f_vals, fq, params_used, None)


def forecast_gt_wavenet(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Forecast using GluonTS WaveNet (PyTorch)."""
    p = params or {}
    ctx_len = int(p.get('context_length', min(128, int(n))))
    freq = str(p.get('freq', 'H'))
    epochs = int(p.get('train_epochs', 5))
    batch_size = int(p.get('batch_size', 32))
    lr = float(p.get('learning_rate', 1e-3))
    dilation_depth = int(p.get('dilation_depth', 5))
    num_stacks = int(p.get('num_stacks', int(p.get('num_blocks', 1))))
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None

    try:
        try:
            from gluonts.torch.model.wavenet import WaveNetEstimator  # type: ignore
        except Exception:
            from gluonts.torch import WaveNetEstimator  # type: ignore
    except Exception as ex:
        return (None, None, {}, f"wavenet requires gluonts[torch]. Install/upgrade: pip install 'gluonts[torch]' torch ({ex})")

    ds = _build_list_dataset(series, freq=freq)

    try:
        estimator = WaveNetEstimator(
            prediction_length=int(fh),
            context_length=int(ctx_len),
            freq=freq,
            batch_size=int(batch_size),
            lr=float(lr),
            trainer_kwargs={"max_epochs": int(epochs)},
            dilation_depth=int(dilation_depth),
            num_stacks=int(num_stacks),
        )
        predictor = estimator.train(training_data=ds)
        forecasts = list(predictor.predict(ds))
        if not forecasts:
            return (None, None, {}, "wavenet produced no forecasts")
        f_vals, fq = _extract_forecast_arrays(forecasts[0], int(fh), quantiles)
        if f_vals is None:
            return (None, None, {}, "wavenet could not extract forecast values")
    except Exception as ex:
        return (None, None, {}, f"wavenet error: {ex}")

    params_used: Dict[str, Any] = {
        'context_length': int(ctx_len), 'freq': freq,
        'train_epochs': int(epochs), 'batch_size': int(batch_size), 'learning_rate': float(lr),
        'dilation_depth': int(dilation_depth), 'num_stacks': int(num_stacks),
    }
    if fq:
        params_used['quantiles'] = sorted(fq.keys(), key=lambda x: float(x))
    return (f_vals, fq, params_used, None)


def forecast_gt_deepnpts(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Forecast using GluonTS DeepNPTS (PyTorch)."""
    p = params or {}
    ctx_len = int(p.get('context_length', min(128, int(n))))
    freq = str(p.get('freq', 'H'))
    epochs = int(p.get('train_epochs', 5))
    batch_size = int(p.get('batch_size', 32))
    lr = float(p.get('learning_rate', 1e-3))
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None

    try:
        try:
            from gluonts.torch.model.deep_npts import DeepNPTSEstimator  # type: ignore
        except Exception:
            from gluonts.torch import DeepNPTSEstimator  # type: ignore
    except Exception as ex:
        return (None, None, {}, f"deepnpts requires gluonts[torch]. Install/upgrade: pip install 'gluonts[torch]' torch ({ex})")

    ds = _build_list_dataset(series, freq=freq)

    try:
        estimator = DeepNPTSEstimator(
            prediction_length=int(fh),
            context_length=int(ctx_len),
            freq=freq,
            batch_size=int(batch_size),
            lr=float(lr),
            trainer_kwargs={"max_epochs": int(epochs)},
        )
        predictor = estimator.train(training_data=ds)
        forecasts = list(predictor.predict(ds))
        if not forecasts:
            return (None, None, {}, "deepnpts produced no forecasts")
        f_vals, fq = _extract_forecast_arrays(forecasts[0], int(fh), quantiles)
        if f_vals is None:
            return (None, None, {}, "deepnpts could not extract forecast values")
    except Exception as ex:
        return (None, None, {}, f"deepnpts error: {ex}")

    params_used: Dict[str, Any] = {
        'context_length': int(ctx_len), 'freq': freq,
        'train_epochs': int(epochs), 'batch_size': int(batch_size), 'learning_rate': float(lr),
    }
    if fq:
        params_used['quantiles'] = sorted(fq.keys(), key=lambda x: float(x))
    return (f_vals, fq, params_used, None)


def forecast_gt_mqf2(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Forecast using GluonTS MQF2 (PyTorch, quantile-focused)."""
    p = params or {}
    ctx_len = int(p.get('context_length', min(128, int(n))))
    freq = str(p.get('freq', 'H'))
    epochs = int(p.get('train_epochs', 5))
    batch_size = int(p.get('batch_size', 32))
    lr = float(p.get('learning_rate', 1e-3))
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None

    try:
        try:
            from gluonts.torch.model.mqf2.estimator import MQF2Estimator as _MQF2  # type: ignore
        except Exception:
            try:
                from gluonts.torch.model.mqf2 import MQF2Estimator as _MQF2  # type: ignore
            except Exception:
                # Some releases expose MQF2MultiHorizonEstimator instead
                from gluonts.torch.model.mqf2.estimator import MQF2MultiHorizonEstimator as _MQF2  # type: ignore
    except Exception as ex:
        # MQF2 depends on 'cpflows' and a GluonTS build that exposes the estimator.
        return (None, None, {}, "mqf2 requires gluonts[torch] (with MQF2 estimator) and cpflows. "
                                     f"Try: pip install 'gluonts[torch]' cpflows --upgrade. Details: {ex}")

    ds = _build_list_dataset(series, freq=freq)

    try:
        kwargs: Dict[str, Any] = {}
        if quantiles:
            try:
                kwargs['quantile_levels'] = [float(q) for q in quantiles]
            except Exception:
                pass
        estimator = _MQF2(
            prediction_length=int(fh),
            context_length=int(ctx_len),
            freq=freq,
            batch_size=int(batch_size),
            lr=float(lr),
            trainer_kwargs={"max_epochs": int(epochs)},
            **kwargs,
        )
        predictor = estimator.train(training_data=ds)
        forecasts = list(predictor.predict(ds))
        if not forecasts:
            return (None, None, {}, "mqf2 produced no forecasts")
        f_vals, fq = _extract_forecast_arrays(forecasts[0], int(fh), quantiles)
        if f_vals is None:
            return (None, None, {}, "mqf2 could not extract forecast values")
    except Exception as ex:
        return (None, None, {}, f"mqf2 error: {ex}")

    params_used: Dict[str, Any] = {
        'context_length': int(ctx_len), 'freq': freq,
        'train_epochs': int(epochs), 'batch_size': int(batch_size), 'learning_rate': float(lr),
    }
    if fq:
        params_used['quantiles'] = sorted(fq.keys(), key=lambda x: float(x))
    return (f_vals, fq, params_used, None)


def forecast_gt_npts(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Forecast using GluonTS NPTS (non-parametric time series).

    This is a non-neural nearest-neighbors method; fast and dependency-light.
    """
    p = params or {}
    freq = str(p.get('freq', 'H'))
    # Map generic 'kernel' to GluonTS kernel_type choices
    kernel_in = str(p.get('kernel', 'exponential')).lower()
    kernel_type = 'uniform' if kernel_in in ('uniform', 'climatological') else 'exponential'
    use_seasonal_model = bool(p.get('use_seasonal_model', True))
    num_default_time_features = int(p.get('num_default_time_features', 1))
    context_length = p.get('context_length')
    ctx_len = int(context_length) if context_length is not None else None
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None

    try:
        from gluonts.model.npts import NPTSPredictor  # type: ignore
    except Exception as ex:
        return (None, None, {}, f"npts requires gluonts. Install/upgrade: pip install gluonts ({ex})")

    ds = _build_list_dataset(series, freq=freq)

    try:
        predictor = NPTSPredictor(
            prediction_length=int(fh),
            context_length=ctx_len,
            kernel_type=kernel_type,
            use_seasonal_model=use_seasonal_model,
            use_default_time_features=True,
            num_default_time_features=int(num_default_time_features),
        )
        forecasts = list(predictor.predict(ds))
        if not forecasts:
            return (None, None, {}, "npts produced no forecasts")
        f_vals, fq = _extract_forecast_arrays(forecasts[0], int(fh), quantiles)
        if f_vals is None:
            return (None, None, {}, "npts could not extract forecast values")
    except Exception as ex:
        return (None, None, {}, f"npts error: {ex}")

    params_used: Dict[str, Any] = {
        'freq': freq,
        'context_length': int(ctx_len) if ctx_len is not None else None,
        'kernel_type': kernel_type,
        'use_seasonal_model': bool(use_seasonal_model),
        'num_default_time_features': int(num_default_time_features),
    }
    if fq:
        params_used['quantiles'] = sorted(fq.keys(), key=lambda x: float(x))
    return (f_vals, fq, params_used, None)


class GluonTSExtraMethod(ForecastMethod):
    REQUIRED_PACKAGES: List[str] = ["gluonts"]

    @property
    def category(self) -> str:
        return "gluonts_extra"

    @property
    def required_packages(self) -> List[str]:
        return list(self.REQUIRED_PACKAGES)

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
        impl_map: Dict[str, Callable[..., Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]]] = {
            "gt_deepar": forecast_gt_deepar,
            "gt_sfeedforward": forecast_gt_sfeedforward,
            "gt_prophet": forecast_gt_prophet,
            "gt_tft": forecast_gt_tft,
            "gt_wavenet": forecast_gt_wavenet,
            "gt_deepnpts": forecast_gt_deepnpts,
            "gt_mqf2": forecast_gt_mqf2,
            "gt_npts": forecast_gt_npts,
        }
        impl = impl_map.get(self.name)
        if impl is None:
            raise RuntimeError(f"Unsupported GluonTS method: {self.name}")

        s = np.asarray(series.values, dtype=float)
        f_vals, fq, params_used, error = impl(
            series=s,
            fh=int(horizon),
            params=dict(params or {}),
            n=int(s.size),
        )
        if error is not None:
            raise RuntimeError(error)
        if f_vals is None:
            raise RuntimeError(f"{self.name} produced no forecast values")
        metadata = {"quantiles": fq} if fq else None
        return ForecastResult(
            forecast=np.asarray(f_vals, dtype=float),
            params_used=params_used,
            metadata=metadata,
        )


@ForecastRegistry.register("gt_deepar")
class GTDeepARMethod(GluonTSExtraMethod):
    REQUIRED_PACKAGES = ["gluonts", "torch"]

    @property
    def name(self) -> str:
        return "gt_deepar"


@ForecastRegistry.register("gt_sfeedforward")
class GTSimpleFeedForwardMethod(GluonTSExtraMethod):
    REQUIRED_PACKAGES = ["gluonts", "torch"]

    @property
    def name(self) -> str:
        return "gt_sfeedforward"


@ForecastRegistry.register("gt_prophet")
class GTProphetMethod(GluonTSExtraMethod):
    REQUIRED_PACKAGES = ["gluonts", "prophet"]

    @property
    def name(self) -> str:
        return "gt_prophet"


@ForecastRegistry.register("gt_tft")
class GTTFTMethod(GluonTSExtraMethod):
    REQUIRED_PACKAGES = ["gluonts", "torch"]

    @property
    def name(self) -> str:
        return "gt_tft"


@ForecastRegistry.register("gt_wavenet")
class GTWaveNetMethod(GluonTSExtraMethod):
    REQUIRED_PACKAGES = ["gluonts", "torch"]

    @property
    def name(self) -> str:
        return "gt_wavenet"


@ForecastRegistry.register("gt_deepnpts")
class GTDeepNPTSMethod(GluonTSExtraMethod):
    REQUIRED_PACKAGES = ["gluonts", "torch"]

    @property
    def name(self) -> str:
        return "gt_deepnpts"


@ForecastRegistry.register("gt_mqf2")
class GTMQF2Method(GluonTSExtraMethod):
    REQUIRED_PACKAGES = ["gluonts", "torch", "cpflows"]

    @property
    def name(self) -> str:
        return "gt_mqf2"


@ForecastRegistry.register("gt_npts")
class GTNPTSMethod(GluonTSExtraMethod):
    REQUIRED_PACKAGES = ["gluonts"]

    @property
    def name(self) -> str:
        return "gt_npts"
