from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .pretrained_helpers import (
    extract_context_window,
    validate_and_clean_data,
    extract_forecast_values,
    adjust_forecast_length,
    extract_quantiles_from_forecast,
    process_quantile_levels,
    build_params_used,
    safe_import_modules,
    validate_required_params,
    create_return_tuple
)


def forecast_chronos_bolt(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Chronos-2 forecasting via Chronos2Pipeline."""
    # Extract parameters with defaults
    p = params or {}
    model_name = str(p.get('model_name', 'amazon/chronos-2'))
    ctx_len = int(p.get('context_length', 0) or 0)
    device_map = p.get('device_map', 'auto')
    series_id = str(p.get('series_id', 'series'))
    quantiles = process_quantile_levels(p.get('quantiles'))
    
    # Extract context window
    context = extract_context_window(series, ctx_len, n, dtype=float)
    
    # Import required modules
    modules, import_error = safe_import_modules(
        ['pandas', 'chronos'],
        method_name='chronos2'
    )
    if import_error:
        return create_return_tuple(None, None, {}, import_error)
    
    _pd = modules['pandas']
    _Chronos2Pipeline = modules['chronos'].Chronos2Pipeline
    
    try:
        # Process quantiles and create context dataframe
        q_levels = quantiles if quantiles else [0.5]
        context_df = _pd.DataFrame({
            "id": series_id,
            "timestamp": _pd.RangeIndex(len(context)),
            "target": _pd.Series(context, dtype=float),
        })
        
        # Create pipeline and generate forecast
        pipe = _Chronos2Pipeline.from_pretrained(model_name, device_map=device_map)
        pred_df = pipe.predict_df(
            context_df,
            prediction_length=int(fh),
            quantile_levels=q_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        
        # Filter predictions for this series
        pred_df = pred_df[pred_df["id"] == series_id]
        if pred_df.empty:
            return create_return_tuple(None, None, {}, "chronos2 error: empty prediction frame")
        
        # Extract quantiles
        fq: Dict[str, List[float]] = {}
        for q in q_levels:
            col_name = f"{float(q):g}"
            if col_name in pred_df.columns:
                fq[col_name] = [float(v) for v in pred_df[col_name].tolist()[:fh]]
        
        # Extract point forecast with fallback logic
        f_vals: Optional[np.ndarray] = None
        if "0.5" in pred_df.columns:
            f_vals = np.asarray(pred_df["0.5"].tolist(), dtype=float)[:fh]
        elif "predictions" in pred_df.columns:
            f_vals = np.asarray(pred_df["predictions"].tolist(), dtype=float)[:fh]
        elif fq:
            first_q = next(iter(fq.values()))
            f_vals = np.asarray(first_q, dtype=float)[:fh]
        
        # Build params_used
        params_used = build_params_used(
            {
                'model_name': model_name,
                'context_length': int(ctx_len) if ctx_len else int(n),
                'device_map': device_map,
            },
            quantiles_dict=fq
        )
        
        if f_vals is None:
            return create_return_tuple(None, fq, params_used, "chronos2 error: no point forecast produced")
        
        return create_return_tuple(f_vals, fq or None, params_used, None)
        
    except Exception as ex:
        return create_return_tuple(None, None, {}, f"chronos2 error: {ex}")


def forecast_timesfm(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """TimesFM 2.5 forecasting via new torch API if available."""
    # Extract parameters
    p = params or {}
    ctx_len = int(p.get('context_length', 0) or 0)
    quantiles = process_quantile_levels(p.get('quantiles'))
    
    # Extract context window
    context = extract_context_window(series, ctx_len, n, dtype=float)
    
    # Import required modules with namespace handling
    try:
        import timesfm as _timesfm
        # Try to detect namespace shadowing
        if not (hasattr(_timesfm, 'TimesFM_2p5_200M_torch') or hasattr(_timesfm, 'ForecastConfig')):
            try:
                from timesfm import torch as _timesfm
            except Exception:
                _p = getattr(_timesfm, '__path__', None)
                _p_str = str(list(_p)) if _p is not None else 'unknown'
                return create_return_tuple(
                    None, None, {},
                    f"timesfm import resolved to namespace at {_p_str}. Rename/remove local 'timesfm' folder or pip install -e the official repo."
                )
    except Exception as ex:
        return create_return_tuple(None, None, {}, f"timesfm requires timesfm package: {ex}")
    
    try:
        # Prefer new API
        _cls_name = 'TimesFM_2p5_200M_torch'
        _has_new = hasattr(_timesfm, _cls_name) and hasattr(_timesfm, 'ForecastConfig')
        if not _has_new:
            return create_return_tuple(
                None, None, {},
                "timesfm installed but API not recognized (missing TimesFM_2p5_200M_torch/ForecastConfig). Update the package."
            )
        
        _Cls = getattr(_timesfm, _cls_name)
        _mdl = _Cls()
        _max_ctx = int(ctx_len) if ctx_len and int(ctx_len) > 0 else None
        
        # Configure forecast
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
        
        # Load model and generate forecast
        try:
            _mdl.load_checkpoint()
        except Exception:
            pass
        
        _mdl.compile(_cfg)
        _inp = [np.asarray(context, dtype=float)]
        pf, qf = _mdl.forecast(horizon=int(fh), inputs=_inp)
        
        # Extract point forecast
        f_vals: Optional[np.ndarray] = None
        if pf is not None:
            arr = np.asarray(pf, dtype=float)
            arr = arr[0] if arr.ndim == 2 else arr
            vals = np.asarray(arr, dtype=float)
            f_vals = adjust_forecast_length(vals, fh, method_name='timesfm')
        
        # Extract quantiles
        fq: Dict[str, List[float]] = {}
        if quantiles and qf is not None:
            qarr = np.asarray(qf, dtype=float)
            if qarr.ndim == 3 and qarr.shape[0] >= 1:
                Q = qarr.shape[-1]
                level_map = {str(l/10.0): (l if Q >= 10 else None) for l in range(1, 10)}
                for q in quantiles:
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
        
        # Build params_used
        params_used = build_params_used(
            {
                'timesfm_model': _cls_name,
                'context_length': int(_max_ctx or n),
            },
            quantiles_dict=fq
        )
        
        return create_return_tuple(f_vals, fq or None, params_used, None)
        
    except Exception as ex:
        return create_return_tuple(None, None, {}, f"timesfm error: {ex}")


def forecast_moirai(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Moirai one-shot forecasting via uni2ts."""
    # Extract parameters with defaults
    p = params or {}
    variant = str(p.get('variant', 'moirai-1.1-R-large'))
    ctx_len = int(p.get('context_length', 0) or 0)
    device = p.get('device')
    quantiles = process_quantile_levels(p.get('quantiles'))
    do_mean = True if p.get('do_mean', None) is None else bool(p.get('do_mean'))
    do_median = True if p.get('do_median', None) is None else bool(p.get('do_median'))
    
    # Extract and validate context window
    context = extract_context_window(series, ctx_len, n, dtype=float)
    cleaned_context, validation_error = validate_and_clean_data(context, method_name='moirai')
    if validation_error:
        return create_return_tuple(None, None, {}, validation_error)
    
    # Import required modules
    modules, import_error = safe_import_modules(
        ['numpy', 'torch', 'uni2ts.model.moirai', 'gluonts.dataset.pandas', 'gluonts.dataset.split'],
        method_name='moirai'
    )
    if import_error:
        return create_return_tuple(None, None, {}, import_error)
    
    _np = modules['numpy']
    torch = modules['torch']
    MoiraiForecast = modules['uni2ts.model.moirai'].MoiraiForecast
    MoiraiModule = modules['uni2ts.model.moirai'].MoiraiModule
    PandasDataset = modules['gluonts.dataset.pandas'].PandasDataset
    
    try:
        # Parse variant to determine model size
        model_size = "large"  # default fallback
        variant_parts = variant.split('-')
        if len(variant_parts) >= 3:
            model_size = variant_parts[-1]
        
        # Load pre-trained model module
        model_name = f"Salesforce/moirai-1.1-R-{model_size}"
        module = MoiraiModule.from_pretrained(model_name)
        
        # Create Moirai forecast model
        model = MoiraiForecast(
            module=module,
            prediction_length=int(fh),
            context_length=len(cleaned_context),
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
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        
        # Create dataset using the direct constructor
        try:
            timestamps = pd.date_range(start=pd.Timestamp('2000-01-01'), periods=len(cleaned_context), freq='H')
            df = pd.DataFrame({
                'timestamp': timestamps,
                'value': cleaned_context
            })
            
            dataset = PandasDataset(
                df,
                target='value',
                timestamp='timestamp',
                freq='H'
            )
        except Exception:
            # Fallback to ListDataset if PandasDataset fails
            timestamps = pd.date_range(start=pd.Timestamp('2000-01-01'), periods=len(cleaned_context), freq='H')
            dataset = ListDataset([
                {
                    'start': timestamps[0],
                    'target': cleaned_context.tolist()
                }
            ], freq='H')
        
        # Generate forecast
        forecasts = list(predictor.predict(dataset))
        if not forecasts:
            return create_return_tuple(None, None, {}, "moirai error: no forecasts generated")
        
        # Extract forecast
        forecast = forecasts[0]
        
        # Extract point forecast
        f_vals, extract_error = extract_forecast_values(
            forecast, fh, method_name='moirai',
            do_mean=do_mean, do_median=do_median
        )
        if extract_error:
            return create_return_tuple(None, None, {}, extract_error)
        
        # Adjust forecast length
        f_vals = adjust_forecast_length(f_vals, fh, method_name='moirai')
        
        # Extract quantiles
        fq = extract_quantiles_from_forecast(forecast, quantiles, fh, method_name='moirai')
        
        # Build params_used
        params_used = build_params_used(
            {
                'variant': variant,
                'device': device,
                'model_name': model_name,
                'do_mean': do_mean,
                'do_median': do_median,
            },
            quantiles_dict=fq,
            context_length=int(ctx_len) if ctx_len else int(n)
        )
        
        return create_return_tuple(f_vals, fq or None, params_used, None)
        
    except Exception as ex:
        return create_return_tuple(None, None, {}, f"moirai error: {ex}")


def forecast_lag_llama(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    """Lag-Llama forecasting via native LagLlamaEstimator."""
    # Extract parameters
    p = params or {}
    
    # Handle checkpoint path resolution
    ckpt_path = p.get('ckpt_path') or p.get('checkpoint') or p.get('model_path')
    if not ckpt_path:
        # Try to fetch a default checkpoint from Hugging Face Hub
        try:
            from huggingface_hub import hf_hub_download
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
            return create_return_tuple(
                None, None, {},
                "lag_llama requires params.ckpt_path or the ability to auto-download via huggingface_hub. "
                f"Tried default repo but failed: {ex}. Provide params: ckpt_path or hf_repo+hf_filename, and install huggingface_hub."
            )
    
    ctx_len = int(p.get('context_length', 32) or 32)
    num_samples = int(p.get('num_samples', 100) or 100)
    use_rope = bool(p.get('use_rope_scaling', False))
    freq = str(p.get('freq', 'H'))
    quantiles = process_quantile_levels(p.get('quantiles'))
    
    # Extract context window
    context = extract_context_window(series, ctx_len, n, dtype=float)
    
    # Import required modules
    modules, import_error = safe_import_modules(
        ['torch', 'lag_llama.gluon.estimator', 'gluonts.evaluation', 'gluonts.dataset.common', 'pandas'],
        method_name='lag_llama',
        fallback_imports={'lag_llama.gluon.estimator': 'lag_llama.gluon'}
    )
    if import_error:
        return create_return_tuple(None, None, {}, import_error)
    
    torch = modules['torch']
    LagLlamaEstimator = modules['lag_llama.gluon.estimator'].LagLlamaEstimator
    make_evaluation_predictions = modules['gluonts.evaluation'].make_evaluation_predictions
    ListDataset = modules['gluonts.dataset.common'].ListDataset
    pd = modules['pandas']
    
    # PyTorch 2.6+ compatibility for safe globals
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
                    _m = __import__(mod, fromlist=[name])
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
        return create_return_tuple(None, None, {}, f"failed to load Lag-Llama checkpoint: {ex}")
    
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
        # Create estimator
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
        
        # Create predictor
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)
        
        # Build dataset
        idx = pd.date_range(start=pd.Timestamp('2000-01-01'), periods=len(context), freq=freq)
        ds = ListDataset([
            {
                'target': np.asarray(context, dtype=np.float32),
                'start': idx[0],
            }
        ], freq=freq)
        
        # Generate forecast
        forecast_it, ts_it = make_evaluation_predictions(dataset=ds, predictor=predictor, num_samples=max(1, int(num_samples)))
        forecasts = list(forecast_it)
        if not forecasts:
            return create_return_tuple(None, None, {}, "lag_llama produced no forecasts")
        
        f = forecasts[0]
        
        # Extract point forecast with comprehensive fallback logic
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
            return create_return_tuple(None, None, {}, "lag_llama could not extract forecast values")
        
        f_vals = adjust_forecast_length(vals, fh, method_name='lag_llama')
        
        # Extract quantiles
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
                # Adjust length and convert to list
                q_arr = adjust_forecast_length(q_arr, fh, method_name='lag_llama')
                fq[str(qf)] = [float(v) for v in q_arr.tolist()]
        
        # Build params_used
        params_used = build_params_used(
            {
                'ckpt_path': str(ckpt_path),
                'context_length': int(ctx_len),
                'device': str(device),
                'num_samples': int(num_samples),
                'use_rope_scaling': bool(use_rope),
                'freq': freq,
            },
            quantiles_dict=fq
        )
        
        return create_return_tuple(f_vals, fq or None, params_used, None)
        
    except Exception as ex:
        return create_return_tuple(None, None, {}, f"lag_llama inference error: {ex}")