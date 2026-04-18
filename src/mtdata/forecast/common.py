from __future__ import annotations

import math
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..services.data_service import (
    _resolve_live_rate_auto_shift_seconds,
    _shift_rate_times,
)
from ..utils.mt5 import (
    _ensure_symbol_ready,
    _mt5_copy_rates_from,
    _mt5_copy_rates_from_pos,
    get_symbol_info_cached,
    mt5,
)
from ..utils.utils import _parse_start_datetime, _utc_epoch_seconds

_FORECAST_RESERVED_COLUMNS = {"unique_id", "ds", "y"}
_FORECAST_PREFERRED_COLUMNS = ("y_hat", "mean", "median", "pred", "forecast")
_FORECAST_AUXILIARY_COLUMN_RE = re.compile(
    r"(?:^|[-_])(lo|low|lower|hi|high|upper|interval|quantile|fitted|residual|cutoff)(?:[-_].*)?$",
    re.IGNORECASE,
)
_NF_ENV_LOCK = threading.RLock()


def edge_pad_to_length(values: np.ndarray, length: int) -> np.ndarray:
    """Trim or edge-pad a 1D array to exactly `length` elements."""
    target = max(0, int(length))
    vals = np.asarray(values, dtype=float).ravel()
    if target == 0:
        return np.array([], dtype=float)
    if vals.size >= target:
        return vals[:target].astype(float, copy=False)
    if vals.size == 0:
        return np.full(target, np.nan, dtype=float)
    return np.pad(vals, (0, target - vals.size), mode='edge').astype(float, copy=False)


def build_ci_diagnostics(
    *,
    provider: str,
    requested: bool,
    available: bool,
    status: str,
    alpha: Optional[float] = None,
    coverage: Optional[float] = None,
    level: Optional[int] = None,
    warning: Optional[str] = None,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    interval_columns: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Build standardized CI diagnostics metadata for forecast method results."""
    ci_diag: Dict[str, Any] = {
        "provider": str(provider),
        "requested": bool(requested),
        "available": bool(available),
        "status": str(status),
    }
    if alpha is not None:
        ci_diag["alpha"] = float(alpha)
    if coverage is not None:
        ci_diag["coverage"] = float(coverage)
    if level is not None:
        ci_diag["level"] = int(level)
    if warning:
        ci_diag["warning"] = str(warning)
    if error:
        ci_diag["error"] = str(error)
    if error_type:
        ci_diag["error_type"] = str(error_type)
    if interval_columns is not None:
        ci_diag["interval_columns"] = [str(col) for col in interval_columns]
    return {"diagnostics": {"ci": ci_diag}}


def log_returns_from_prices(prices: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute consecutive log-returns from a price array."""
    arr = np.asarray(prices, dtype=float).ravel()
    if arr.size < 2:
        return np.array([], dtype=float)
    try:
        eps_value = float(eps)
    except Exception as exc:
        raise ValueError("eps must be a positive finite number") from exc
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("eps must be a positive finite number")
    finite = arr[np.isfinite(arr)]
    if finite.size and np.any(finite < 0.0):
        raise ValueError("prices must not contain negative values")
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = np.diff(np.log(np.clip(arr, eps_value, None)))
    return np.asarray(rets, dtype=float)


def _normalize_weights(weights: Any, size: int) -> Optional[np.ndarray]:
    if weights is None:
        return None
    vals: List[float] = []
    if isinstance(weights, (list, tuple)):
        vals = [float(v) for v in list(weights)[:size]]
    elif isinstance(weights, str):
        parts = [p.strip() for p in weights.split(",") if p.strip()]
        vals = [float(p) for p in parts[:size]]
    else:
        return None
    if len(vals) != size:
        return None
    arr = np.asarray(vals, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    arr = np.clip(arr, a_min=0.0, a_max=None)
    total = float(np.sum(arr))
    if total <= 0:
        return None
    return arr / total


def _extract_forecast_values(
    Yf: Any,
    fh: int,
    method_name: str = "forecast",
    *,
    allow_actual_fallback: bool = False,
) -> "np.ndarray":
    """Extract forecast values from prediction DataFrame.
    
    Common logic for finding prediction columns and extracting values.
    """
    pred_col = None
    column_detection_error: Optional[Exception] = None
    try:
        columns = list(Yf.columns)
        pred_candidates = [c for c in columns if c not in _FORECAST_RESERVED_COLUMNS]
        numeric_candidates = []
        for candidate in pred_candidates:
            series = pd.to_numeric(Yf[candidate], errors="coerce")
            if bool(series.notna().any()):
                numeric_candidates.append(candidate)

        preferred_map = {str(candidate).lower(): candidate for candidate in numeric_candidates}
        for preferred_name in _FORECAST_PREFERRED_COLUMNS:
            pred_col = preferred_map.get(preferred_name)
            if pred_col is not None:
                break

        def _method_named_candidates(candidates: List[Any]) -> List[Any]:
            method_tokens = [
                token
                for token in re.split(r"[^a-z0-9]+", str(method_name).lower())
                if len(token) >= 3 and token not in {"forecast", "statsforecast", "neuralforecast"}
            ]
            if not method_tokens:
                return []
            matches = [
                candidate
                for candidate in candidates
                if any(token in str(candidate).lower() for token in method_tokens)
            ]
            return matches

        if pred_col is None:
            filtered_candidates = [
                candidate
                for candidate in numeric_candidates
                if _FORECAST_AUXILIARY_COLUMN_RE.search(str(candidate)) is None
            ]

            named_matches = _method_named_candidates(filtered_candidates)
            if len(named_matches) == 1:
                pred_col = named_matches[0]
            elif len(filtered_candidates) == 1:
                pred_col = filtered_candidates[0]
            elif len(numeric_candidates) == 1:
                pred_col = numeric_candidates[0]
            elif allow_actual_fallback and not numeric_candidates and "y" in columns:
                pred_col = "y"
    except Exception as exc:
        column_detection_error = exc
    
    if pred_col is None:
        columns = []
        try:
            columns = list(Yf.columns)
        except Exception:
            columns = []
        if not allow_actual_fallback and "y" in columns:
            error = RuntimeError(
                f"{method_name} prediction columns not found; refusing to use actuals column 'y'. "
                f"Available columns: {columns}"
            )
            if column_detection_error is not None:
                raise error from column_detection_error
            raise error
        error = RuntimeError(
            f"{method_name} prediction columns not found. Available columns: {columns}"
        )
        if column_detection_error is not None:
            raise error from column_detection_error
        raise error
    
    vals = np.asarray(Yf[pred_col].to_numpy(), dtype=float)
    return edge_pad_to_length(vals, int(fh))


def _create_training_dataframes(series: np.ndarray, fh: int, exog_used: Optional[np.ndarray] = None, exog_future: Optional[np.ndarray] = None) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """Create standardized training DataFrames for forecast methods.
    
    Returns (Y_df, X_df, Xf_df) where:
    - Y_df: training series DataFrame
    - X_df: training exogenous features DataFrame (if provided)
    - Xf_df: future exogenous features DataFrame (if provided)
    """
    import pandas as _pd
    
    train_len = int(len(series))
    train_index = _pd.RangeIndex(start=0, stop=train_len)
    base_train = _pd.DataFrame(
        {
            'unique_id': ['ts'] * train_len,
            'ds': train_index,
        }
    )
    Y_df = base_train.copy()
    Y_df['y'] = np.asarray(series, dtype=float)
    
    X_df = None
    Xf_df = None
    if exog_used is not None and isinstance(exog_used, np.ndarray) and exog_used.size:
        cols = [f'x{i}' for i in range(exog_used.shape[1])]
        X_df = base_train.copy()
        for j, cname in enumerate(cols):
            X_df[cname] = exog_used[:, j]
        if exog_future is not None and isinstance(exog_future, np.ndarray) and exog_future.size:
            future_len = int(fh)
            future_index = _pd.RangeIndex(start=train_len, stop=train_len + future_len)
            Xf_df = _pd.DataFrame(
                {
                    'unique_id': ['ts'] * future_len,
                    'ds': future_index,
                }
            )
            for j, cname in enumerate(cols):
                Xf_df[cname] = exog_future[:, j]
    
    return Y_df, X_df, Xf_df


def default_seasonality(timeframe: str) -> int:
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


def bars_per_year(timeframe: str) -> float:
    """Approximate number of bars per year for a timeframe."""
    try:
        tf = str(timeframe).upper().strip()
        secs = TIMEFRAME_SECONDS.get(tf)
        if not secs or secs <= 0:
            return float("nan")
        if tf == "MN1":
            return 12.0
        if tf == "W1":
            return 52.0
        if float(secs) >= 86400.0:
            return float((252.0 * 86400.0) / float(secs))
        return float((252.0 * 24.0 * 3600.0) / float(secs))
    except Exception:
        return float("nan")


def quantity_to_target(quantity: str) -> str:
    """Map a forecast quantity to the corresponding price/return target mode."""
    return "return" if str(quantity).strip().lower() == "return" else "price"


def next_times_from_last(last_epoch: float, tf_secs: int, horizon: int) -> List[float]:
    base = float(last_epoch)
    step = float(tf_secs)
    return [base + step * (i + 1) for i in range(int(horizon))]


def pd_freq_from_timeframe(tf: str) -> str:
    t = str(tf).upper()
    mapping = {
        'M1': '1min', 'M2': '2min', 'M3': '3min', 'M4': '4min', 'M5': '5min',
        'M10': '10min', 'M12': '12min', 'M15': '15min', 'M20': '20min', 'M30': '30min',
        'H1': '1h', 'H2': '2h', 'H3': '3h', 'H4': '4h', 'H6': '6h', 'H8': '8h', 'H12': '12h',
        'D1': '1d', 'W1': '1w', 'MN1': 'MS'
    }
    return mapping.get(t, 'D')


# ------------------------------------------------------------------
# Composable NeuralForecast building blocks (used by train/predict)
# ------------------------------------------------------------------

def _nf_resolve_accelerator() -> str:
    """Return 'cpu' or 'gpu' based on torch availability and env."""
    accel = 'cpu'
    try:
        import torch as _torch
        accel_env = os.environ.get('MTDATA_NF_ACCEL')
        if isinstance(accel_env, str):
            accel = 'gpu' if accel_env.strip().lower() == 'gpu' else 'cpu'
        else:
            accel = 'gpu' if hasattr(_torch, 'cuda') and _torch.cuda.is_available() else 'cpu'
        try:
            if accel == 'gpu' and hasattr(_torch, 'set_float32_matmul_precision'):
                _torch.set_float32_matmul_precision('high')
        except Exception:
            pass
    except Exception:
        accel = 'cpu'
    return accel


def nf_build_model_kwargs(
    *,
    model_class,
    fh: int,
    input_size: int,
    batch_size: int,
    steps: int,
    learning_rate: Optional[float] = None,
    accel: Optional[str] = None,
    enable_progress_bar: bool = False,
) -> Dict[str, Any]:
    """Build keyword arguments for a NeuralForecast model constructor.

    Does NOT instantiate the model — returns the kwargs dict so that
    callers can inspect or modify them before construction.
    """
    import inspect as _inspect

    if accel is None:
        accel = _nf_resolve_accelerator()

    try:
        ctor_params = _inspect.signature(model_class.__init__).parameters
    except Exception:
        ctor_params = {}

    model_kwargs: Dict[str, Any] = {
        'h': int(fh),
        'input_size': int(input_size),
        'batch_size': int(batch_size),
    }
    if 'max_steps' in ctor_params:
        model_kwargs['max_steps'] = int(steps)
    elif 'max_epochs' in ctor_params:
        model_kwargs['max_epochs'] = int(steps)
    else:
        model_kwargs['max_steps'] = int(steps)
    if learning_rate is not None:
        try:
            model_kwargs['learning_rate'] = float(learning_rate)
        except Exception:
            pass

    base_trainer: Dict[str, Any] = {
        'accelerator': accel,
        'devices': 1,
        'num_nodes': 1,
    }
    quiet_opts: Dict[str, Any] = {
        'logger': False,
        'enable_progress_bar': enable_progress_bar,
        'enable_checkpointing': False,
        'enable_model_summary': False,
        'log_every_n_steps': 0,
    }
    for _opt, _val in quiet_opts.items():
        base_trainer.setdefault(_opt, _val)
    for _opt, _val in base_trainer.items():
        model_kwargs.setdefault(_opt, _val)

    return model_kwargs


_NF_ENV_VARS_TO_CLEAR = (
    'KUBERNETES_SERVICE_HOST', 'KUBERNETES_SERVICE_PORT',
    'GROUP_RANK', 'NODE_RANK', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE',
    'GLOBAL_RANK', 'MASTER_ADDR', 'MASTER_PORT',
    'LT_CLOUD_PROVIDER', 'LT_CLUSTER', 'TORCHELASTIC_RUN_ID',
    'ETCD_HOST', 'ETCD_PORT',
)
_NF_MANAGED_ENV_VARS = _NF_ENV_VARS_TO_CLEAR + (
    'PL_TORCH_DISTRIBUTED_BACKEND',
    'LT_DISABLE_DISTRIBUTED',
    'CUDA_VISIBLE_DEVICES',
)


class _NfEnvGuard:
    """Context manager that sanitizes env vars for single-device NF training."""

    def __init__(self, accel: str = 'cpu') -> None:
        self._accel = accel
        self._snapshot: Dict[str, Optional[str]] = {}
        self._missing: set[str] = set()

    def __enter__(self) -> '_NfEnvGuard':
        for key in _NF_MANAGED_ENV_VARS:
            if key in os.environ:
                self._snapshot[key] = os.environ.get(key)
            else:
                self._missing.add(key)
        for _var in _NF_ENV_VARS_TO_CLEAR:
            os.environ.pop(_var, None)
        os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
        os.environ['LT_DISABLE_DISTRIBUTED'] = '1'
        if self._accel == 'gpu':
            try:
                cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                if ',' in cvd:
                    os.environ['CUDA_VISIBLE_DEVICES'] = cvd.split(',')[0].strip()
                elif cvd.strip() == '':
                    import torch as _torch
                    if _torch.cuda.device_count() > 1:
                        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            except Exception:
                pass
        return self

    def __exit__(self, *exc_info: Any) -> None:
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                _dist.destroy_process_group()
        except Exception:
            pass
        try:
            import torch as _torch
            if hasattr(_torch, 'cuda') and _torch.cuda.is_available():
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
        except Exception:
            pass
        for key in _NF_MANAGED_ENV_VARS:
            if key in self._missing:
                os.environ.pop(key, None)
                continue
            restored = self._snapshot.get(key)
            if restored is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = restored


def _nf_build_trainer_kwargs(accel: str) -> Dict[str, Any]:
    """Build trainer kwargs for the NeuralForecast constructor."""
    import inspect as _inspect

    base_trainer: Dict[str, Any] = {
        'accelerator': accel,
        'devices': 1,
        'num_nodes': 1,
    }
    cand_opts: Dict[str, Any] = {
        'logger': False,
        'enable_progress_bar': False,
        'enable_checkpointing': False,
        'log_every_n_steps': 0,
    }
    try:
        try:
            import lightning.pytorch as _L
            _Trainer = _L.Trainer
        except Exception:
            import pytorch_lightning as _pl
            _Trainer = _pl.Trainer
        _tparams = _inspect.signature(_Trainer.__init__).parameters
        nf_trainer = dict(base_trainer)
        for k, v in cand_opts.items():
            if k in _tparams and k not in nf_trainer:
                nf_trainer[k] = v
        return nf_trainer
    except Exception:
        return {**base_trainer, **cand_opts}


def nf_create_and_fit(
    *,
    model_class,
    model_kwargs: Dict[str, Any],
    timeframe: str,
    Y_df: pd.DataFrame,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[List[float]] = None,
) -> Any:
    """Instantiate a NeuralForecast wrapper, fit it, and return the fitted NF object.

    Must be called inside ``_NF_ENV_LOCK`` and ``_NfEnvGuard``.
    """
    import inspect as _inspect
    import warnings

    try:
        from neuralforecast import NeuralForecast as _NeuralForecast
    except Exception as ex:
        raise RuntimeError(f"Failed to import neuralforecast: {ex}")

    accel = str(model_kwargs.get('accelerator', 'cpu'))
    nf_kwargs: Dict[str, Any] = {
        'models': [model_class(**model_kwargs)],
        'freq': pd_freq_from_timeframe(timeframe),
    }
    try:
        _nf_init_params = _inspect.signature(_NeuralForecast.__init__).parameters
    except Exception:
        _nf_init_params = {}
    if 'trainer_kwargs' in _nf_init_params:
        nf_trainer = _nf_build_trainer_kwargs(accel)
        try:
            try:
                import lightning.pytorch as _L
                _Trainer = _L.Trainer
            except Exception:
                import pytorch_lightning as _pl
                _Trainer = _pl.Trainer
            trainer_obj = _Trainer(**nf_trainer)
            nf_kwargs['trainer'] = trainer_obj
        except Exception:
            nf_kwargs['trainer_kwargs'] = nf_trainer
    if 'num_workers_loader' in _nf_init_params:
        nf_kwargs['num_workers_loader'] = 0

    nf = _NeuralForecast(**nf_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _fit_params = _inspect.signature(nf.fit).parameters
        except Exception:
            _fit_params = {}
        supports_x = 'X_df' in _fit_params

        if exog_used is not None and isinstance(exog_used, np.ndarray) and exog_used.size and supports_x:
            X_df = pd.DataFrame({'unique_id': ['ts'] * len(Y_df), 'ds': Y_df['ds'].values})
            cols = [f'x{i}' for i in range(exog_used.shape[1])]
            for j, cname in enumerate(cols):
                X_df[cname] = exog_used[:, j]
            nf.fit(df=Y_df, X_df=X_df, verbose=False)
        else:
            nf.fit(df=Y_df, verbose=False)

    return nf


def nf_predict_from_fitted(
    nf: Any,
    *,
    fh: int,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Run predictions on an already-fitted NeuralForecast object.

    Must be called inside ``_NF_ENV_LOCK`` and ``_NfEnvGuard``.
    """
    import inspect as _inspect
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _pred_params = _inspect.signature(nf.predict).parameters
        except Exception:
            _pred_params = {}
        try:
            _fit_params = _inspect.signature(nf.fit).parameters
        except Exception:
            _fit_params = {}
        supports_x_predict = 'X_df' in _pred_params

        if exog_future is not None and isinstance(exog_future, np.ndarray) and exog_future.size and future_times is not None and supports_x_predict:
            ds_f = pd.to_datetime(pd.Series(future_times), unit='s', utc=True)
            cols = [f'x{i}' for i in range(exog_future.shape[1])]
            Xf_df = pd.DataFrame({'unique_id': ['ts'] * len(ds_f), 'ds': pd.Index(ds_f).to_pydatetime()})
            for j, cname in enumerate(cols):
                Xf_df[cname] = exog_future[:, j]
            if 'h' in _pred_params:
                return nf.predict(h=int(fh), X_df=Xf_df)
            else:
                return nf.predict(X_df=Xf_df)
        else:
            if 'h' in _pred_params:
                return nf.predict(h=int(fh))
            else:
                return nf.predict()


def nf_setup_and_predict(  # noqa: C901
    *,
    model_class,
    fh: int,
    timeframe: str,
    Y_df: pd.DataFrame,
    input_size: int,
    batch_size: int,
    steps: int,
    learning_rate: Optional[float] = None,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Create NeuralForecast model safely and return its predictions DataFrame.

    Handles max_steps/max_epochs differences, single-device training, optional X_df API,
    predict(h=...) signature differences, and quiet/performant trainer options.
    """
    import inspect as _inspect
    import warnings

    # Build model kwargs with compatibility
    try:
        ctor_params = _inspect.signature(model_class.__init__).parameters  # type: ignore[attr-defined]
    except Exception:
        ctor_params = {}
    model_kwargs: Dict[str, Any] = {
        'h': int(fh),
        'input_size': int(input_size),
        'batch_size': int(batch_size),
    }
    if 'max_steps' in ctor_params:
        model_kwargs['max_steps'] = int(steps)
    elif 'max_epochs' in ctor_params:
        model_kwargs['max_epochs'] = int(steps)
    else:
        model_kwargs['max_steps'] = int(steps)
    if learning_rate is not None:
        try:
            model_kwargs['learning_rate'] = float(learning_rate)
        except Exception:
            pass

    # Resolve accelerator, sanitize env, and build quiet single-device trainer defaults
    accel = 'cpu'
    try:
        import torch as _torch  # type: ignore
        accel_env = os.environ.get('MTDATA_NF_ACCEL')
        if isinstance(accel_env, str):
            accel = 'gpu' if accel_env.strip().lower() == 'gpu' else 'cpu'
        else:
            accel = 'gpu' if hasattr(_torch, 'cuda') and _torch.cuda.is_available() else 'cpu'
        try:
            if accel == 'gpu' and hasattr(_torch, 'set_float32_matmul_precision'):
                _torch.set_float32_matmul_precision('high')  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        accel = 'cpu'

    env_vars_to_clear = (
        'KUBERNETES_SERVICE_HOST', 'KUBERNETES_SERVICE_PORT',
        'GROUP_RANK', 'NODE_RANK', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE',
        'GLOBAL_RANK', 'MASTER_ADDR', 'MASTER_PORT',
        'LT_CLOUD_PROVIDER', 'LT_CLUSTER', 'TORCHELASTIC_RUN_ID',
        'ETCD_HOST', 'ETCD_PORT',
    )
    managed_env_vars = env_vars_to_clear + (
        'PL_TORCH_DISTRIBUTED_BACKEND',
        'LT_DISABLE_DISTRIBUTED',
        'CUDA_VISIBLE_DEVICES',
    )
    with _NF_ENV_LOCK:
        env_snapshot: Dict[str, Optional[str]] = {}
        env_missing: set[str] = set()
        for key in managed_env_vars:
            if key in os.environ:
                env_snapshot[key] = os.environ.get(key)
            else:
                env_missing.add(key)

        try:
            for _var in env_vars_to_clear:
                os.environ.pop(_var, None)
            os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
            os.environ['LT_DISABLE_DISTRIBUTED'] = '1'

            base_trainer: Dict[str, Any] = {
                'accelerator': accel,
                'devices': 1,
                'num_nodes': 1,
            }
            quiet_opts = {
                'logger': False,
                'enable_progress_bar': False,
                'enable_checkpointing': False,
                'enable_model_summary': False,
                'log_every_n_steps': 0,
            }
            for _opt, _val in quiet_opts.items():
                base_trainer.setdefault(_opt, _val)

            for _opt, _val in base_trainer.items():
                model_kwargs.setdefault(_opt, _val)

            # Instantiate model and NeuralForecast
            try:
                from neuralforecast import NeuralForecast as _NeuralForecast  # type: ignore
            except Exception as ex:
                raise RuntimeError(f"Failed to import neuralforecast: {ex}")

            nf_kwargs: Dict[str, Any] = {
                'models': [model_class(**model_kwargs)],  # type: ignore
                'freq': pd_freq_from_timeframe(timeframe),
            }

            try:
                _nf_init_params = _inspect.signature(_NeuralForecast.__init__).parameters  # type: ignore[attr-defined]
            except Exception:
                _nf_init_params = {}
            if 'trainer_kwargs' in _nf_init_params:
                nf_trainer = dict(base_trainer)
                cand_opts = {
                    'logger': False,
                    'enable_progress_bar': False,
                    'enable_checkpointing': False,
                    'log_every_n_steps': 0,
                }
                try:
                    try:
                        import lightning.pytorch as _L  # type: ignore
                        _Trainer = _L.Trainer  # type: ignore[attr-defined]
                    except Exception:
                        import pytorch_lightning as _pl  # type: ignore
                        _Trainer = _pl.Trainer  # type: ignore[attr-defined]
                    _tparams = _inspect.signature(_Trainer.__init__).parameters  # type: ignore[attr-defined]
                    for k, v in list(cand_opts.items()):
                        if k in _tparams and k not in nf_trainer:
                            nf_trainer[k] = v
                    try:
                        trainer_obj = _Trainer(**nf_trainer)  # type: ignore[call-arg]
                        nf_kwargs['trainer'] = trainer_obj
                    except Exception:
                        nf_kwargs['trainer_kwargs'] = nf_trainer
                except Exception:
                    nf_trainer.update(cand_opts)
                    nf_kwargs['trainer_kwargs'] = nf_trainer
            # Restrict visible GPUs to one when CUDA is available
            try:
                if accel == 'gpu':
                    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                    if ',' in cvd:
                        os.environ['CUDA_VISIBLE_DEVICES'] = cvd.split(',')[0].strip()
                    elif cvd.strip() == '':
                        import torch as _torch  # type: ignore
                        if _torch.cuda.device_count() > 1:  # type: ignore[attr-defined]
                            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            except Exception:
                pass
            if 'num_workers_loader' in _nf_init_params:
                nf_kwargs['num_workers_loader'] = 0

            nf = _NeuralForecast(**nf_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Detect X_df and predict(h) support
                try:
                    _fit_params = _inspect.signature(nf.fit).parameters  # type: ignore[attr-defined]
                except Exception:
                    _fit_params = {}
                try:
                    _pred_params = _inspect.signature(nf.predict).parameters  # type: ignore[attr-defined]
                except Exception:
                    _pred_params = {}
                supports_x = ('X_df' in _fit_params) and ('X_df' in _pred_params)

                if exog_used is not None and isinstance(exog_used, np.ndarray) and exog_used.size and supports_x:
                    # Build X_df and X_future for NF (newer API)
                    X_df = pd.DataFrame({'unique_id': ['ts'] * int(len(Y_df)), 'ds': Y_df['ds'].values})
                    cols = [f'x{i}' for i in range(exog_used.shape[1])]
                    for j, cname in enumerate(cols):
                        X_df[cname] = exog_used[:, j]
                    nf.fit(df=Y_df, X_df=X_df, verbose=False)  # type: ignore
                    if exog_future is not None and isinstance(exog_future, np.ndarray) and exog_future.size and future_times is not None:
                        ds_f = pd.to_datetime(pd.Series(future_times), unit='s', utc=True)
                        Xf_df = pd.DataFrame({'unique_id': ['ts'] * int(len(ds_f)), 'ds': pd.Index(ds_f).to_pydatetime()})
                        for j, cname in enumerate(cols):
                            Xf_df[cname] = exog_future[:, j]
                        if 'h' in _pred_params:
                            Yf = nf.predict(h=int(fh), X_df=Xf_df)  # type: ignore
                        else:
                            Yf = nf.predict(X_df=Xf_df)  # type: ignore
                    else:
                        if 'h' in _pred_params:
                            Yf = nf.predict(h=int(fh))  # type: ignore
                        else:
                            Yf = nf.predict()  # type: ignore
                else:
                    nf.fit(df=Y_df, verbose=False)  # type: ignore
                    if 'h' in _pred_params:
                        Yf = nf.predict(h=int(fh))  # type: ignore
                    else:
                        Yf = nf.predict()  # type: ignore
            return Yf
        finally:
            try:
                import torch.distributed as _dist  # type: ignore
                if _dist.is_available() and _dist.is_initialized():
                    _dist.destroy_process_group()
            except Exception:
                pass
            try:
                import torch as _torch  # type: ignore
                if hasattr(_torch, 'cuda') and _torch.cuda.is_available():
                    _torch.cuda.synchronize()
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            for key in managed_env_vars:
                if key in env_missing:
                    os.environ.pop(key, None)
                    continue
                restored = env_snapshot.get(key)
                if restored is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = restored


def fetch_history(
    symbol: str,
    timeframe: str,
    need: int,
    as_of: Optional[str] = None,
    *,
    drop_last_live: bool = True,
) -> pd.DataFrame:
    """Fetch last `need` bars for symbol/timeframe, normalize times to UTC seconds.

    - as_of: optional date/time string. If provided, fetch bars ending at that time. Else uses server time.
    - drop_last_live: when as_of is None, drop the forming last bar.
    Raises RuntimeError on MT5 errors.
    """
    if timeframe not in TIMEFRAME_MAP:
        raise RuntimeError(f"Invalid timeframe: {timeframe}")
    mt5_tf = TIMEFRAME_MAP[timeframe]
    # Ensure symbol visibility and restore later
    info_before = get_symbol_info_cached(symbol)
    was_visible = bool(info_before.visible) if info_before is not None else None
    err = _ensure_symbol_ready(symbol)
    if err:
        raise RuntimeError(err)
    try:
        if as_of:
            to_dt = _parse_start_datetime(as_of)
            if not to_dt:
                raise RuntimeError("Invalid as_of time.")

            # Anchor directly at as_of to avoid missing older historical windows.
            fetch_count = max(int(need), 1) + 2
            rates = _mt5_copy_rates_from(symbol, mt5_tf, to_dt, fetch_count)
        else:
            # Use position-based fetch for "latest" to avoid TZ issues and ensure open candle
            # start_pos=0 includes the current forming bar
            rates = _mt5_copy_rates_from_pos(symbol, mt5_tf, 0, int(need))
            auto_shift_seconds = _resolve_live_rate_auto_shift_seconds(
                symbol=symbol,
                timeframe=timeframe,
                start_datetime=None,
                end_datetime=None,
            )
            if auto_shift_seconds:
                rates = _shift_rate_times(rates, auto_shift_seconds)
    finally:
        if was_visible is False:
            try:
                mt5.symbol_select(symbol, False)
            except Exception:
                pass
    if rates is None or len(rates) < 1:
        raise RuntimeError(f"Failed to get rates for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    # Times are already normalized to UTC by _mt5_copy_rates_from_pos via _normalize_times_in_struct
    # DO NOT normalize again.
    
    # Manual truncation if as_of provided
    if as_of and not df.empty and 'time' in df.columns:
        to_dt = _parse_start_datetime(as_of)
        if to_dt:
            cutoff = _utc_epoch_seconds(to_dt)
            # Filter: include the bar exactly AT the cutoff if it exists
            df = df[df['time'] <= cutoff]
            # Take last 'need'
            if len(df) > need:
                df = df.iloc[-int(need):]
    
    if as_of is None and drop_last_live and len(df) >= 2:
        df = df.iloc[:-1]
    return df.reset_index(drop=True)

