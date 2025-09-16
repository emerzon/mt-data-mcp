from __future__ import annotations

from typing import Any, Dict, List, Optional

import os
import math
import numpy as np
import pandas as pd

from ..core.constants import TIMEFRAME_SECONDS, TIMEFRAME_MAP
from ..utils.mt5 import _mt5_epoch_to_utc, _ensure_symbol_ready, _mt5_copy_rates_from
import MetaTrader5 as mt5
from ..utils.utils import _parse_start_datetime as _parse_start_datetime_util


def parse_kv_or_json(obj: Any) -> Dict[str, Any]:
    """Parse params/features provided as dict, JSON string, or k=v pairs into a dict.

    - Dict: shallow-copied and returned
    - JSON-like string: parsed via json.loads (with simple fallback for colon/equals pairs)
    - Plain string: split on whitespace/commas into k=v assignments
    """
    import json

    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {}
        if (s.startswith('{') and s.endswith('}')):
            try:
                return json.loads(s)
            except Exception:
                # Fallback to simple token parser inside braces
                s = s.strip().strip('{}').strip()
        # Parse simple k=v tokens separated by whitespace/commas
        out: Dict[str, Any] = {}
        toks = [tok for tok in s.replace(',', ' ').split() if tok]
        i = 0
        while i < len(toks):
            tok = toks[i].strip().strip(',')
            if not tok:
                i += 1
                continue
            if '=' in tok:
                k, v = tok.split('=', 1)
                out[k.strip()] = v.strip().strip(',')
                i += 1
                continue
            if tok.endswith(':'):
                key = tok[:-1].strip()
                val = ''
                if i + 1 < len(toks):
                    val = toks[i + 1].strip().strip(',')
                    i += 2
                else:
                    i += 1
                out[key] = val
                continue
            i += 1
        return out
    return {}


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

    # Instantiate model and NeuralForecast
    try:
        from neuralforecast import NeuralForecast as _NeuralForecast  # type: ignore
    except Exception as ex:
        raise RuntimeError(f"Failed to import neuralforecast: {ex}")

    nf_kwargs: Dict[str, Any] = {
        'models': [model_class(**model_kwargs)],  # type: ignore
        'freq': pd_freq_from_timeframe(timeframe),
    }
    # Quiet single-device trainer
    try:
        _nf_init_params = _inspect.signature(_NeuralForecast.__init__).parameters  # type: ignore[attr-defined]
    except Exception:
        _nf_init_params = {}
    trainer_kwargs = None
    accel = 'cpu'
    try:
        import torch as _torch  # type: ignore
        accel = 'gpu' if hasattr(_torch, 'cuda') and _torch.cuda.is_available() else 'cpu'
        try:
            if accel == 'gpu' and hasattr(_torch, 'set_float32_matmul_precision'):
                _torch.set_float32_matmul_precision('medium')  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        accel = 'cpu'
    if 'trainer_kwargs' in _nf_init_params:
        base_trainer: Dict[str, Any] = {'accelerator': accel, 'devices': 1, 'strategy': 'single_device'}
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
                if k in _tparams:
                    base_trainer[k] = v
        except Exception:
            base_trainer.update(cand_opts)
        trainer_kwargs = base_trainer
        nf_kwargs['trainer_kwargs'] = trainer_kwargs
    # Restrict visible GPUs to one when CUDA is available
    try:
        if accel == 'gpu':
            cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if ',' in cvd:
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd.split(',')[0].strip()
            elif cvd.strip() == '':
                # Only set if multiple devices visible
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
    info_before = mt5.symbol_info(symbol)
    was_visible = bool(info_before.visible) if info_before is not None else None
    err = _ensure_symbol_ready(symbol)
    if err:
        raise RuntimeError(err)
    try:
        if as_of:
            to_dt = _parse_start_datetime_util(as_of)
            if not to_dt:
                raise RuntimeError("Invalid as_of time.")
            rates = _mt5_copy_rates_from(symbol, mt5_tf, to_dt, int(need))
        else:
            _tick = mt5.symbol_info_tick(symbol)
            if _tick is not None and getattr(_tick, 'time', None):
                t_utc = _mt5_epoch_to_utc(float(_tick.time))
                from datetime import datetime as _dt
                server_now_dt = _dt.utcfromtimestamp(t_utc)
            else:
                from datetime import datetime as _dt
                server_now_dt = _dt.utcnow()
            rates = _mt5_copy_rates_from(symbol, mt5_tf, server_now_dt, int(need))
    finally:
        if was_visible is False:
            try:
                mt5.symbol_select(symbol, False)
            except Exception:
                pass
    if rates is None or len(rates) < 1:
        raise RuntimeError(f"Failed to get rates for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    # Normalize times
    try:
        if 'time' in df.columns:
            df['time'] = df['time'].astype(float).apply(_mt5_epoch_to_utc)
    except Exception:
        pass
    if as_of is None and drop_last_live and len(df) >= 2:
        df = df.iloc[:-1]
    return df.reset_index(drop=True)
