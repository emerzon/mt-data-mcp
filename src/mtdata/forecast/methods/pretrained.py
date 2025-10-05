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
    """Chronos-Bolt forecasting via native Chronos pipeline.

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


def forecast_moirai(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Moirai one-shot forecasting via uni2ts.

    Params:
    - variant: model variant string (e.g., '1.0-R-small', '1.0-L-small')
    - context_length: int, context window to feed (<= len(series))
    - device: optional torch device string
    - quantiles: optional list of quantile levels to return
    - do_mean: bool, return mean estimate if available (default True)
    - do_median: bool, fallback to median if mean unavailable (default True)

    Returns (f_vals, forecast_quantiles, params_used, error)
    """
    p = params or {}
    variant = str(p.get('variant', '1.0-R-small'))
    ctx_len = int(p.get('context_length', 0) or 0)
    device = p.get('device')
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None
    do_mean = True if p.get('do_mean', None) is None else bool(p.get('do_mean'))
    do_median = True if p.get('do_median', None) is None else bool(p.get('do_median'))

    # Select context window
    if ctx_len and ctx_len > 0:
        context = np.asarray(series[-int(min(n, ctx_len)) :], dtype=float)
    else:
        context = np.asarray(series, dtype=float)

    try:
        import numpy as _np
        import torch  # type: ignore
        from uni2ts import get_timeseries_model  # type: ignore
    except Exception as ex:
        return (None, None, {}, f"moirai requires uni2ts and torch: {ex}")

    try:
        # Load model by variant; get_timeseries_model returns a callable forward/infer object
        model = get_timeseries_model(model=variant)
        # Ensure model to device if specified
        if device:
            try:
                model.to(device)
            except Exception:
                pass
        # Build input expected by uni2ts one-shot forecast: (batch, length)
        x = context.astype(float)
        x = _np.nan_to_num(x, nan=float(_np.nanmean(x) if _np.isfinite(_np.nanmean(x)) else 0.0))
        x = x.reshape(1, -1).astype(_np.float32)

        # Run forecast; most uni2ts models expose forecast(horizon, ...)
        # Fallback to __call__ returning dict with 'pred'
        f_vals: Optional[np.ndarray] = None
        fq: Dict[str, List[float]] = {}
        try:
            out = model.forecast(horizon=int(fh), context=x)
        except Exception:
            out = model(x, horizon=int(fh))

        # Parse outputs
        # Expected keys may be: 'mean', 'median', 'quantiles' or 'samples'/'pred'
        arr = None
        if isinstance(out, dict):
            if do_mean and 'mean' in out:
                arr = _np.asarray(out['mean'], dtype=float)
            elif do_median and ('median' in out or 'p50' in out):
                key = 'median' if 'median' in out else 'p50'
                arr = _np.asarray(out[key], dtype=float)
            elif 'pred' in out:
                arr = _np.asarray(out['pred'], dtype=float)
            elif 'samples' in out:
                try:
                    arr = _np.asarray(out['samples'], dtype=float)
                    if arr.ndim >= 2:
                        arr = _np.mean(arr, axis=0)
                except Exception:
                    arr = None
            # Quantiles
            if quantiles and 'quantiles' in out and isinstance(out['quantiles'], dict):
                for q in quantiles:
                    try:
                        qf = float(q)
                    except Exception:
                        continue
                    k = str(qf)
                    if k in out['quantiles']:
                        qarr = _np.asarray(out['quantiles'][k], dtype=float).ravel()
                        fq[k] = [float(v) for v in qarr[:fh].tolist()]
        else:
            # Tensor or ndarray
            try:
                arr = _np.asarray(out, dtype=float)
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr[0]
            except Exception:
                arr = None

        if arr is None:
            return (None, None, {}, "moirai output format not recognized")
        arr = arr.ravel()
        vals = arr[:fh] if arr.size >= fh else _np.pad(arr, (0, fh - arr.size), mode='edge')
        f_vals = vals

        params_used = {
            'variant': variant,
            'context_length': int(ctx_len) if ctx_len else int(n),
            'device': device,
        }
        if quantiles and fq:
            params_used['quantiles'] = sorted(list(fq.keys()), key=lambda x: float(x))
        return (f_vals, fq or None, params_used, None)
    except Exception as ex:
        return (None, None, {}, f"moirai error: {ex}")

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




def forecast_lag_llama(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Lag-Llama forecasting via native LagLlamaEstimator (no Transformers).

    Expected params:
    - ckpt_path: path to a Lag-Llama .ckpt checkpoint (required)
    - device: e.g., 'cuda:0' or 'cpu' (optional; auto if omitted)
    - context_length: input context length (default: 32)
    - num_samples: number of sample paths for quantile estimation (default: 100)
    - use_rope_scaling: bool to enable rope scaling when context_length+fh exceeds training context
    - freq: pandas frequency string for synthetic timestamps (default: 'H')
    - quantiles: list of quantiles to return (optional)
    """
    p = params or {}
    ckpt_path = p.get('ckpt_path') or p.get('checkpoint') or p.get('model_path')
    if not ckpt_path:
        # Try to fetch a default checkpoint from Hugging Face Hub
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            repo_id = str(p.get('hf_repo', 'time-series-foundation-models/Lag-Llama'))
            filename = str(p.get('hf_filename', 'lag-llama.ckpt'))
            revision = p.get('revision')
            token = p.get('hf_token')
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token)
            # Stash back into params for traceability
            p['ckpt_path'] = ckpt_path
            p['hf_repo'] = repo_id
            p['hf_filename'] = filename
        except Exception as ex:
            return (None, None, {}, "lag_llama requires params.ckpt_path or the ability to auto-download via huggingface_hub. "
                                     f"Tried default repo but failed: {ex}. Provide params: ckpt_path or hf_repo+hf_filename, and install huggingface_hub.")

    ctx_len = int(p.get('context_length', 32) or 32)
    num_samples = int(p.get('num_samples', 100) or 100)
    use_rope = bool(p.get('use_rope_scaling', False))
    freq = str(p.get('freq', 'H'))
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None

    # Select context window
    if ctx_len and ctx_len > 0:
        k = int(min(len(series), ctx_len))
        context = np.asarray(series[-k:], dtype=float)
    else:
        context = np.asarray(series, dtype=float)

    try:
        import torch  # type: ignore
        from lag_llama.gluon.estimator import LagLlamaEstimator  # type: ignore
        from gluonts.evaluation import make_evaluation_predictions  # type: ignore
        from gluonts.dataset.common import ListDataset  # type: ignore
        import pandas as pd  # type: ignore
        try:
            import huggingface_hub  # type: ignore
        except Exception:
            huggingface_hub = None  # optional, only needed when auto-downloading
    except Exception as ex:
        return (None, None, {}, f"lag_llama dependencies missing: {ex}. Install: pip install lag-llama gluonts torch (optional: huggingface_hub)")

    # PyTorch 2.6+ defaults torch.load(..., weights_only=True) which blocks unpickling
    # of custom classes used by GluonTS (e.g., StudentTOutput). When downstream code
    # inside GluonTS/Lag-Llama calls torch.load with the default, register the class
    # as a safe global so weights-only loading can succeed without disabling safety.
    try:
        _add_safe = getattr(torch.serialization, "add_safe_globals", None)
        if callable(_add_safe):
            _to_allow = []
            # Distributions commonly referenced by GluonTS Torch models
            for mod, name in (
                ("gluonts.torch.distributions.studentT", "StudentTOutput"),
                ("gluonts.torch.distributions.student_t", "StudentTOutput"),
                ("gluonts.torch.distributions.normal", "NormalOutput"),
                ("gluonts.torch.distributions.laplace", "LaplaceOutput"),
                ("gluonts.torch.modules.loss", "NegativeLogLikelihood"),
            ):
                try:
                    _m = __import__(mod, fromlist=[name])  # type: ignore
                    _cls = getattr(_m, name, None)
                    if _cls is not None:
                        _to_allow.append(_cls)
                except Exception:
                    continue
            if _to_allow:
                try:
                    _add_safe(_to_allow)
                except Exception:
                    pass
    except Exception:
        # Best-effort: proceed even if safe globals cannot be registered
        pass

    # Resolve device
    device_str = str(p.get('device')) if p.get('device') is not None else None
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint to get model hyperparameters
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        est_args = ckpt.get('hyper_parameters', {}).get('model_kwargs', {})
    except Exception as ex:
        return (None, None, {}, f"failed to load Lag-Llama checkpoint: {ex}")

    # Optional rope scaling when context exceeds training
    rope_scaling = None
    try:
        base_ctx = int(est_args.get('context_length', 32))
        if use_rope:
            factor = max(1.0, float((ctx_len + int(fh)) / max(1, base_ctx)))
            rope_scaling = {"type": "linear", "factor": float(factor)}
    except Exception:
        rope_scaling = None

    try:
        estimator = LagLlamaEstimator(
            ckpt_path=str(ckpt_path),
            prediction_length=int(fh),
            context_length=int(ctx_len),
            input_size=est_args.get('input_size', 1),
            n_layer=est_args.get('n_layer', 8),
            n_embd_per_head=est_args.get('n_embd_per_head', 64),
            n_head=est_args.get('n_head', 8),
            scaling=est_args.get('scaling', 'none'),
            time_feat=est_args.get('time_feat', 'none'),
            rope_scaling=rope_scaling,
            batch_size=1,
            num_parallel_samples=max(1, int(num_samples)),
            device=device,
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)

        # Build single-series GluonTS ListDataset with synthetic timestamps
        idx = pd.date_range(start=pd.Timestamp('2000-01-01'), periods=len(context), freq=freq)
        ds = ListDataset([
            {
                'target': np.asarray(context, dtype=np.float32),
                'start': idx[0],
            }
        ], freq=freq)

        forecast_it, ts_it = make_evaluation_predictions(dataset=ds, predictor=predictor, num_samples=max(1, int(num_samples)))
        forecasts = list(forecast_it)
        if not forecasts:
            return (None, None, {}, "lag_llama produced no forecasts")
        f = forecasts[0]

        # Point forecast: use mean if available, else median quantile, else samples average
        vals = None
        try:
            vals = np.asarray(f.mean, dtype=float)
        except Exception:
            pass
        if vals is None or vals.size == 0:
            try:
                vals = np.asarray(f.quantile(0.5), dtype=float)
            except Exception:
                pass
        if (vals is None or vals.size == 0) and hasattr(f, 'samples'):
            try:
                vals = np.asarray(np.mean(f.samples, axis=0), dtype=float)
            except Exception:
                pass
        if vals is None:
            return (None, None, {}, "lag_llama could not extract forecast values")
        f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')

        fq: Dict[str, List[float]] = {}
        if quantiles:
            for q in quantiles:
                try:
                    qf = float(q)
                except Exception:
                    continue
                try:
                    q_arr = np.asarray(f.quantile(qf), dtype=float)
                except Exception:
                    continue
                fq[str(qf)] = [float(v) for v in q_arr[:fh].tolist()]

    except Exception as ex:
        return (None, None, {}, f"lag_llama inference error: {ex}")

    params_used = {
        'ckpt_path': str(ckpt_path),
        'context_length': int(ctx_len),
        'device': str(device),
        'num_samples': int(num_samples),
        'use_rope_scaling': bool(use_rope),
        'freq': freq,
    }
    if quantiles:
        params_used['quantiles'] = sorted({str(float(q)) for q in quantiles}, key=lambda x: float(x))

    return (f_vals, fq or None, params_used, None)
