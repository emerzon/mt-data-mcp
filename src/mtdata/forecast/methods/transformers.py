from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def forecast_chronos_bolt(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Chronos-Bolt forecasting via native Chronos or Transformers pipeline.

    Returns (f_vals, forecast_quantiles, params_used, error)
    """
    p = params or {}
    model_name = str(p.get('model_name', 'amazon/chronos-bolt-base'))
    ctx_len = int(p.get('context_length', 0) or 0)
    device = p.get('device')
    device_map = p.get('device_map', 'auto')
    quantization = str(p.get('quantization')) if p.get('quantization') is not None else None
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None
    revision = p.get('revision')
    trust_remote_code = bool(p.get('trust_remote_code', False))
    # Select context window
    if ctx_len and ctx_len > 0:
        context = series[-int(min(n, ctx_len)) :]
    else:
        context = series

    f_vals: Optional[np.ndarray] = None
    fq: Dict[str, List[float]] = {}
    last_err: Optional[Exception] = None

    # Try native Chronos first
    try:
        from chronos import BaseChronosPipeline as _BaseChronosPipeline  # type: ignore
        import torch as _torch  # type: ignore
        _kwargs: Dict[str, Any] = {}
        if quantization:
            if str(quantization).lower() in ('int8', '8bit', 'bnb.int8'):
                _kwargs['load_in_8bit'] = True
            elif str(quantization).lower() in ('int4', '4bit', 'bnb.int4'):
                _kwargs['load_in_4bit'] = True
        if revision:
            _kwargs['revision'] = revision
        # Optional dtype
        _torch_dtype = p.get('torch_dtype')
        if isinstance(_torch_dtype, str):
            _td = _torch_dtype.strip().lower()
            if _td in ('bf16', 'bfloat16'):
                _kwargs['torch_dtype'] = _torch.bfloat16
            elif _td in ('fp16', 'float16', 'half'):
                _kwargs['torch_dtype'] = _torch.float16
            elif _td in ('fp32', 'float32'):
                _kwargs['torch_dtype'] = _torch.float32
        pipe = _BaseChronosPipeline.from_pretrained(model_name, device_map=device_map, **_kwargs)  # type: ignore[arg-type]
        q_levels = list(quantiles) if quantiles else [0.5]
        q_levels = [float(q) for q in q_levels]
        _context_tensor = _torch.tensor(context, dtype=_torch.float32)
        q_tensor, mean_tensor = pipe.predict_quantiles(
            context=_context_tensor,
            prediction_length=int(fh),
            quantile_levels=q_levels,
        )
        arr_mean = mean_tensor.detach().cpu().numpy()[0]
        for i, ql in enumerate(q_levels):
            q_arr = q_tensor[:, :, i].detach().cpu().numpy()[0]
            fq[str(float(ql))] = [float(v) for v in np.asarray(q_arr, dtype=float)[:fh].tolist()]
        if quantiles and '0.5' in fq:
            f_vals = np.asarray(fq['0.5'], dtype=float)
        else:
            vals = np.asarray(arr_mean, dtype=float)
            f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
    except Exception as ex1:
        last_err = ex1
        return (None, None, {}, f"chronos error: {last_err}")

    params_used = {
        'model_name': model_name,
        'context_length': int(ctx_len) if ctx_len else int(n),
        'device': device,
        'device_map': device_map,
        'quantization': quantization,
        'revision': revision,
        'trust_remote_code': trust_remote_code,
    }
    if fq:
        params_used['quantiles'] = sorted(list(fq.keys()), key=lambda x: float(x))
    return (f_vals, fq or None, params_used, None)


def forecast_timesfm(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """TimesFM 2.5 forecasting via new torch API if available.

    Returns (f_vals, forecast_quantiles, params_used, error)
    """
    p = params or {}
    ctx_len = int(p.get('context_length', 0) or 0)
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None
    # Select context
    if ctx_len and ctx_len > 0:
        context = series[-int(min(n, ctx_len)) :]
    else:
        context = series

    f_vals: Optional[np.ndarray] = None
    fq: Dict[str, List[float]] = {}
    try:
        import timesfm as _timesfm  # type: ignore
        # Try to detect namespace shadowing
        if not (hasattr(_timesfm, 'TimesFM_2p5_200M_torch') or hasattr(_timesfm, 'ForecastConfig')):
            try:
                from timesfm import torch as _timesfm  # type: ignore
            except Exception:
                _p = getattr(_timesfm, '__path__', None)
                _p_str = str(list(_p)) if _p is not None else 'unknown'
                return (None, None, {}, f"timesfm import resolved to namespace at {_p_str}. Rename/remove local 'timesfm' folder or pip install -e the official repo.")
        # Prefer new API
        _cls_name = 'TimesFM_2p5_200M_torch'
        _has_new = hasattr(_timesfm, _cls_name) and hasattr(_timesfm, 'ForecastConfig')
        if _has_new:
            _Cls = getattr(_timesfm, _cls_name)
            _mdl = _Cls()
            _max_ctx = int(ctx_len) if ctx_len and int(ctx_len) > 0 else None
            _cfg_kwargs: Dict[str, Any] = {
                'max_context': _max_ctx or min(int(n), 1024),
                'max_horizon': int(fh),
                'normalize_inputs': True,
                'use_continuous_quantile_head': bool(quantiles) is True,
                'force_flip_invariance': True,
                'infer_is_positive': False,
                'fix_quantile_crossing': True,
            }
            _cfg = getattr(_timesfm, 'ForecastConfig')(**_cfg_kwargs)
            try:
                _mdl.load_checkpoint()
            except Exception:
                pass
            _mdl.compile(_cfg)
            _inp = [np.asarray(context, dtype=float)]
            pf, qf = _mdl.forecast(horizon=int(fh), inputs=_inp)
            if pf is not None:
                arr = np.asarray(pf, dtype=float)
                arr = arr[0] if arr.ndim == 2 else arr
                vals = np.asarray(arr, dtype=float)
                f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
            if quantiles and qf is not None:
                qarr = np.asarray(qf, dtype=float)
                if qarr.ndim == 3 and qarr.shape[0] >= 1:
                    Q = qarr.shape[-1]
                    level_map = {str(l/10.0): (l if Q >= 10 else None) for l in range(1, 10)}
                    for q in list(quantiles):
                        try:
                            key = f"{float(q):.1f}"
                        except Exception:
                            continue
                        idx = level_map.get(key)
                        if idx is None:
                            continue
                        col = qarr[0, :fh, idx] if idx < qarr.shape[-1] else None
                        if col is not None:
                            fq[key] = [float(v) for v in np.asarray(col, dtype=float).tolist()]
            params_used = {
                'timesfm_model': _cls_name,
                'context_length': int(_max_ctx or n),
                'quantiles': sorted(list(fq.keys()), key=lambda x: float(x)) if fq else None,
            }
            return (f_vals, fq or None, params_used, None)
        else:
            return (None, None, {}, "timesfm installed but API not recognized (missing TimesFM_2p5_200M_torch/ForecastConfig). Update the package.")
    except Exception as ex:
        return (None, None, {}, f"timesfm error: {ex}")
