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
    """Chronos-2 forecasting via Chronos2Pipeline."""
    p = params or {}
    model_name = str(p.get('model_name', 'amazon/chronos-2'))
    ctx_len = int(p.get('context_length', 0) or 0)
    device_map = p.get('device_map', 'auto')
    series_id = str(p.get('series_id', 'series'))
    quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None

    # Select context window
    if ctx_len and ctx_len > 0:
        context = series[-int(min(n, ctx_len)) :]
    else:
        context = series

    # Build minimal context dataframe expected by Chronos2Pipeline
    try:
        import pandas as _pd  # type: ignore
        from chronos import Chronos2Pipeline as _Chronos2Pipeline  # type: ignore
    except Exception as ex:
        return (None, None, {}, f"chronos2 import error: {ex}")

    try:
        q_levels = [float(q) for q in (quantiles or [0.5])]
        context_df = _pd.DataFrame({
            "id": series_id,
            "timestamp": _pd.RangeIndex(len(context)),
            "target": _pd.Series(context, dtype=float),
        })

        pipe = _Chronos2Pipeline.from_pretrained(model_name, device_map=device_map)
        pred_df = pipe.predict_df(
            context_df,
            prediction_length=int(fh),
            quantile_levels=q_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        pred_df = pred_df[pred_df["id"] == series_id]
        if pred_df.empty:
            return (None, None, {}, "chronos2 error: empty prediction frame")

        fq: Dict[str, List[float]] = {}
        for q in q_levels:
            col_name = f"{float(q):g}"
            if col_name in pred_df.columns:
                fq[col_name] = [float(v) for v in pred_df[col_name].tolist()[:fh]]

        # Choose point forecast: prefer median quantile then mean/predictions column
        f_vals: Optional[np.ndarray] = None
        if "0.5" in pred_df.columns:
            f_vals = np.asarray(pred_df["0.5"].tolist(), dtype=float)[:fh]
        elif "predictions" in pred_df.columns:
            f_vals = np.asarray(pred_df["predictions"].tolist(), dtype=float)[:fh]
        elif fq:
            first_q = next(iter(fq.values()))
            f_vals = np.asarray(first_q, dtype=float)[:fh]

        params_used = {
            'model_name': model_name,
            'context_length': int(ctx_len) if ctx_len else int(n),
            'device_map': device_map,
        }
        if fq:
            params_used['quantiles'] = sorted(list(fq.keys()), key=lambda x: float(x))

        if f_vals is None:
            return (None, fq or None, params_used, "chronos2 error: no point forecast produced")
        return (f_vals, fq or None, params_used, None)
    except Exception as ex:
        return (None, None, {}, f"chronos2 error: {ex}")


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


def forecast_moirai(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Moirai one-shot forecasting via uni2ts.

    Params:
    - variant: model variant string (e.g., 'moirai-1.1-R-large', '1.0-R-small', '1.0-L-small')
    - context_length: int, context window to feed (<= len(series))
    - device: optional torch device string
    - quantiles: optional list of quantile levels to return
    - do_mean: bool, return mean estimate if available (default True)
    - do_median: bool, fallback to median if mean unavailable (default True)

    Returns (f_vals, forecast_quantiles, params_used, error)
    """
    p = params or {}
    variant = str(p.get('variant', 'moirai-1.1-R-large'))
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
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # type: ignore
        from gluonts.dataset.pandas import PandasDataset  # type: ignore
        from gluonts.dataset.split import split  # type: ignore
    except Exception as ex:
        return (None, None, {}, f"moirai requires uni2ts, gluonts and torch: {ex}")

    try:
        # Parse variant to determine model size
        # Expected formats: "moirai-1.1-R-large", "1.0-R-small", "1.0-L-base", etc.
        variant_parts = variant.split('-')
        if len(variant_parts) >= 3:
            # Handle both new format (moirai-1.1-R-large) and old format (1.0-R-small)
            if variant_parts[0] == 'moirai':
                # New format: moirai-1.1-R-large -> model_size = "large"
                model_size = variant_parts[-1]
            else:
                # Old format: 1.0-R-small -> model_size = "small"
                model_size = variant_parts[-1]
        else:
            model_size = "large"  # default fallback for new default variant
        
        # Debug information
        print(f"[DEBUG] Moirai variant: {variant}, model_size: {model_size}")
        print(f"[DEBUG] Original context length: {len(context)}, forecast horizon: {fh}")
        
        # Load pre-trained model module
        model_name = f"Salesforce/moirai-1.1-R-{model_size}"
        module = MoiraiModule.from_pretrained(model_name)
        
        # Create Moirai forecast model
        model = MoiraiForecast(
            module=module,
            prediction_length=int(fh),
            context_length=len(context),
            patch_size="auto",
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        
        # Move model to device if specified
        if device:
            try:
                model.to(device)
            except Exception:
                pass
        
        # Create predictor
        predictor = model.create_predictor(batch_size=32)
        
        # Prepare data in GluonTS format
        # Create a simple pandas DataFrame and convert to GluonTS dataset
        import pandas as pd
        from gluonts.dataset.pandas import PandasDataset
        from gluonts.dataset.common import ListDataset
        
        # Validate and clean the context data
        if len(context) == 0:
            return (None, None, {}, "moirai error: empty context data")
            
        # Check for all NaN values
        if not _np.any(_np.isfinite(context)):
            return (None, None, {}, "moirai error: context contains no finite values")
        
        # Clean the context data - replace NaN with mean of finite values
        finite_mean = _np.nanmean(context[_np.isfinite(context)])
        context_clean = _np.nan_to_num(context, nan=float(finite_mean if _np.isfinite(finite_mean) else 0.0))
        
        # Ensure we have valid data
        if not _np.any(_np.isfinite(context_clean)):
            return (None, None, {}, "moirai error: cleaned context contains no finite values")
            
        # Debug information after cleaning
        print(f"[DEBUG] Cleaned context length: {len(context_clean)}, valid finite values: {_np.sum(_np.isfinite(context_clean))}")
        
        # Create a simple dataset with one time series using proper GluonTS format
        # Method 1: Direct PandasDataset creation (preferred)
        try:
            # Create a proper time series with pandas DatetimeIndex
            timestamps = pd.date_range(start=pd.Timestamp('2000-01-01'), periods=len(context_clean), freq='H')
            df = pd.DataFrame({
                'timestamp': timestamps,
                'value': context_clean
            })
            
            # Create dataset using the direct constructor
            dataset = PandasDataset(
                df,
                target='value',
                timestamp='timestamp',
                freq='H'
            )
        except Exception:
            # Method 2: Fallback to ListDataset if PandasDataset fails
            try:
                from gluonts.dataset.common import ListDataset
                timestamps = pd.date_range(start=pd.Timestamp('2000-01-01'), periods=len(context_clean), freq='H')
                dataset = ListDataset([
                    {
                        'start': timestamps[0],
                        'target': context_clean.tolist()
                    }
                ], freq='H')
            except Exception as inner_ex:
                return (None, None, {}, f"moirai data preparation error: {inner_ex}")
        
        # Generate forecast with better error handling
        try:
            forecasts = list(predictor.predict(dataset))
        except Exception as pred_ex:
            return (None, None, {}, f"moirai prediction error: {pred_ex}")
        
        if not forecasts:
            return (None, None, {}, "moirai error: no forecasts generated")
        
        # Ensure we have at least one forecast
        if len(forecasts) == 0:
            return (None, None, {}, "moirai error: empty forecasts list")
        
        try:
            forecast = forecasts[0]
        except (IndexError, AttributeError) as ex:
            return (None, None, {}, f"moirai error: failed to extract forecast: {ex}")
        
        # Extract point forecast (mean or median)
        f_vals: Optional[np.ndarray] = None
        fq: Dict[str, List[float]] = {}
        
        # Get point forecast with better error handling
        try:
            if do_mean and hasattr(forecast, 'mean') and forecast.mean is not None:
                f_vals = _np.asarray(forecast.mean, dtype=float)
            elif do_median and hasattr(forecast, 'median') and forecast.median is not None:
                f_vals = _np.asarray(forecast.median, dtype=float)
            elif hasattr(forecast, 'samples') and forecast.samples is not None:
                # Use mean of samples if no direct mean/median available
                samples = _np.asarray(forecast.samples, dtype=float)
                if samples.ndim >= 2:
                    f_vals = _np.mean(samples, axis=0)
                else:
                    f_vals = samples
            else:
                return (None, None, {}, "moirai error: no forecast values available (no mean, median, or samples)")
        except Exception as extract_ex:
            return (None, None, {}, f"moirai error: failed to extract forecast values: {extract_ex}")
        
        # Validate extracted forecast
        if f_vals is None or len(f_vals) == 0:
            return (None, None, {}, "moirai error: extracted forecast is empty or invalid")
        
        # Extract quantiles if requested
        if quantiles and hasattr(forecast, 'quantile'):
            for q in quantiles:
                try:
                    qf = float(q)
                    q_value = forecast.quantile(qf)
                    if q_value is not None:
                        fq[str(qf)] = [float(v) for v in _np.asarray(q_value, dtype=float).tolist()]
                except Exception:
                    continue
        
        # Ensure forecast length matches requested horizon
        if f_vals is not None:
            f_vals = f_vals.ravel()
            if len(f_vals) < fh:
                # Pad with last value if forecast is shorter than requested
                f_vals = _np.pad(f_vals, (0, fh - len(f_vals)), mode='edge')
            elif len(f_vals) > fh:
                # Truncate if forecast is longer than requested
                f_vals = f_vals[:fh]
        
        # Process quantiles to match forecast horizon
        if fq:
            for q_key in fq:
                q_vals = _np.asarray(fq[q_key], dtype=float)
                if len(q_vals) < fh:
                    q_vals = _np.pad(q_vals, (0, fh - len(q_vals)), mode='edge')
                elif len(q_vals) > fh:
                    q_vals = q_vals[:fh]
                fq[q_key] = q_vals.tolist()

        params_used = {
            'variant': variant,
            'context_length': int(ctx_len) if ctx_len else int(n),
            'device': device,
            'model_name': model_name,
        }
        if quantiles and fq:
            params_used['quantiles'] = sorted(list(fq.keys()), key=lambda x: float(x))
            
        return (f_vals, fq or None, params_used, None)
    except Exception as ex:
        return (None, None, {}, f"moirai error: {ex}")


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
