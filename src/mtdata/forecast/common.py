from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import os
import math
import numpy as np
import pandas as pd

from ..core.constants import TIMEFRAME_SECONDS, TIMEFRAME_MAP
from ..utils.mt5 import (
    _mt5_epoch_to_utc,
    _ensure_symbol_ready,
    _mt5_copy_rates_from,
    _mt5_copy_rates_from_pos,
    get_symbol_info_cached,
)
import MetaTrader5 as mt5
from ..utils.utils import _parse_start_datetime, _utc_epoch_seconds



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


def log_returns_from_prices(prices: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute consecutive log-returns from a price array."""
    arr = np.asarray(prices, dtype=float).ravel()
    if arr.size < 2:
        return np.array([], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = np.diff(np.log(np.clip(arr, float(eps), None)))
    return np.asarray(rets, dtype=float)


def _extract_forecast_values(Yf: Any, fh: int, method_name: str = "forecast") -> "np.ndarray":
    """Extract forecast values from prediction DataFrame.
    
    Common logic for finding prediction columns and extracting values.
    """
    import numpy as np
    
    # Find the prediction column
    pred_col = None
    try:
        # First try standard prediction column
        if 'y' in Yf.columns:
            pred_col = 'y'
        else:
            # Look for other prediction columns
            for c in list(Yf.columns):
                if c not in ('unique_id', 'ds', 'y'):
                    pred_col = c
                    break
    except Exception:
        pass
    
    if pred_col is None:
        raise RuntimeError(f"{method_name} prediction columns not found")
    
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
    
    # Build single-series training dataframe
    Y_df = _pd.DataFrame({
        'unique_id': ['ts'] * int(len(series)),
        'ds': _pd.RangeIndex(start=0, stop=int(len(series))),
        'y': series.astype(float),
    })
    
    X_df = None
    Xf_df = None
    if exog_used is not None and isinstance(exog_used, np.ndarray) and exog_used.size:
        cols = [f'x{i}' for i in range(exog_used.shape[1])]
        X_df = _pd.DataFrame({'unique_id': ['ts'] * int(len(series)), 'ds': _pd.RangeIndex(start=0, stop=int(len(series)))})
        for j, cname in enumerate(cols):
            X_df[cname] = exog_used[:, j]
        if exog_future is not None and isinstance(exog_future, np.ndarray) and exog_future.size:
            Xf_df = _pd.DataFrame({'unique_id': ['ts'] * int(fh), 'ds': _pd.RangeIndex(start=int(len(series)), stop=int(len(series))+int(fh))})
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


def nf_setup_and_predict(
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
    import warnings
    import inspect as _inspect

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

    for _var in (
        'KUBERNETES_SERVICE_HOST', 'KUBERNETES_SERVICE_PORT',
        'GROUP_RANK', 'NODE_RANK', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE',
        'GLOBAL_RANK', 'MASTER_ADDR', 'MASTER_PORT',
        'LT_CLOUD_PROVIDER', 'LT_CLUSTER', 'TORCHELASTIC_RUN_ID',
        'ETCD_HOST', 'ETCD_PORT'
    ):
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
    trainer_kwargs = None
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
                pass
        except Exception:
            nf_trainer.update(cand_opts)
        trainer_kwargs = nf_trainer
        nf_kwargs['trainer_kwargs'] = trainer_kwargs
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
    # Ensure cleanup of any distributed process groups and GPU memory
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
    return Yf


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
            # We use a small epsilon for float comparison safety
            df = df[df['time'] <= cutoff + 1.0]
            # Take last 'need'
            if len(df) > need:
                df = df.iloc[-int(need):]
    
    if as_of is None and drop_last_live and len(df) >= 2:
        df = df.iloc[:-1]
    return df.reset_index(drop=True)

