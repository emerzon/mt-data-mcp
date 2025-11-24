from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..interface import ForecastMethod, ForecastResult
from ..registry import ForecastRegistry
from .pretrained_helpers import (
    extract_context_window,
    validate_and_clean_data,
    extract_forecast_values,
    adjust_forecast_length,
    extract_quantiles_from_forecast,
    process_quantile_levels,
    build_params_used,
    safe_import_modules,
    create_return_tuple
)

class PretrainedMethod(ForecastMethod):
    """Base class for pretrained foundation models."""
    
    @property
    def category(self) -> str:
        return "pretrained"
        
    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": True}

@ForecastRegistry.register("chronos_bolt")
@ForecastRegistry.register("chronos2")
class ChronosBoltMethod(PretrainedMethod):
    @property
    def name(self) -> str:
        return "chronos_bolt"
        
    @property
    def required_packages(self) -> List[str]:
        return ["chronos", "torch"]

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        p = params or {}
        model_name = str(p.get('model_name', 'amazon/chronos-2'))
        ctx_len = int(p.get('context_length', 0) or 0)
        device_map = p.get('device_map', 'auto')
        series_id = str(p.get('series_id', 'series'))
        quantiles = process_quantile_levels(p.get('quantiles'))
        
        vals = series.values
        n = len(vals)

        # Extract context window using DRY helper
        context = extract_context_window(vals, ctx_len, n, dtype=float)
        
        # Import required modules using DRY helper
        modules, import_error = safe_import_modules(['pandas', 'chronos', 'torch'], 'chronos2')
        if import_error:
            raise RuntimeError(import_error)
        
        _pd = modules['pandas']
        _torch = modules['torch']
        _Chronos2Pipeline = modules['chronos'].Chronos2Pipeline
        
        try:
            q_levels = quantiles or [0.5]
            context_df = _pd.DataFrame({
                "id": series_id,
                "timestamp": _pd.RangeIndex(len(context)),
                "target": _pd.Series(context, dtype=float),
            })

            # Prepare covariates if available
            # Check params first as forecast_engine passes them there
            exog_hist = kwargs.get('exog_used') or p.get('exog_used')
            exog_fut = kwargs.get('exog_future') or exog_future or p.get('exog_future')
            
            known_covariates = None
            
            if exog_hist is not None and exog_fut is not None:
                try:
                    # Ensure arrays
                    hist_arr = np.asarray(exog_hist, dtype=float)
                    fut_arr = np.asarray(exog_fut, dtype=float)
                    
                    # Match context length (take last len(context) rows of history)
                    if len(hist_arr) >= len(context):
                        hist_slice = hist_arr[-len(context):]
                        
                        # Concatenate history + future
                        if hist_slice.shape[1] == fut_arr.shape[1]:
                            full_exog = np.vstack([hist_slice, fut_arr])
                            
                            # Convert to tensor (1, time, n_feat)
                            known_covariates = _torch.tensor(full_exog, dtype=_torch.float32).unsqueeze(0)
                            
                            # Handle device placement if explicit
                            if device_map and device_map not in ('auto', 'cpu'):
                                try:
                                    known_covariates = known_covariates.to(device_map)
                                except Exception:
                                    pass
                except Exception:
                    # Fallback: ignore covariates on error
                    pass

            pipe = _Chronos2Pipeline.from_pretrained(model_name, device_map=device_map)
            
            # Use predict() instead of predict_df to support known_covariates
            # Convert context to tensor
            # Chronos expects (batch, n_variates, time) -> (1, 1, L)
            ctx_tensor = _torch.tensor(context, dtype=_torch.float32).unsqueeze(0).unsqueeze(1)
            if device_map and device_map not in ('auto', 'cpu'):
                try:
                    ctx_tensor = ctx_tensor.to(device_map)
                except Exception:
                    pass

            # Attempt prediction with fallback for varying signatures
            # Strategy: 
            # 1. Try with known_covariates + num_samples (Bolt/Chronos-2-Bolt style)
            # 2. Try without known_covariates (T5 style or if covariates not supported)
            # 3. Try minimal args
            
            forecast_samples = None
            
            # Case 1: Full args
            if known_covariates is not None:
                try:
                    predict_kwargs = {
                        "prediction_length": int(horizon),
                        "num_samples": int(p.get('num_samples', 20)),
                        "known_covariates": known_covariates
                    }
                    forecast_samples = pipe.predict(ctx_tensor, **predict_kwargs)
                except (TypeError, ValueError) as e:
                    # "Unexpected keyword argument" or similar
                    pass
            
            # Case 2: No covariates (if Case 1 failed or no covariates)
            if forecast_samples is None:
                try:
                    predict_kwargs = {
                        "prediction_length": int(horizon),
                        "num_samples": int(p.get('num_samples', 20)),
                    }
                    forecast_samples = pipe.predict(ctx_tensor, **predict_kwargs)
                except (TypeError, ValueError):
                    pass
            
            # Case 3: Minimal (if num_samples also rejected)
            if forecast_samples is None:
                try:
                    forecast_samples = pipe.predict(ctx_tensor, prediction_length=int(horizon))
                except Exception as e:
                    raise RuntimeError(f"Chronos predict failed even with minimal args: {e}")

            # Extract samples for first batch
            if isinstance(forecast_samples, list):
                f_tensor = forecast_samples[0]
            else:
                f_tensor = forecast_samples[0]
                
            samples = f_tensor.detach().cpu().numpy() # (S, H) or (1, S, H)
            
            # Handle variate dimension if present (1, S, H) -> (S, H)
            if samples.ndim == 3 and samples.shape[0] == 1:
                samples = samples[0]
            
            fq: Dict[str, List[float]] = {}
            for q in q_levels:
                try:
                    q_val = np.quantile(samples, float(q), axis=0)
                    fq[f"{float(q):g}"] = [float(v) for v in q_val.tolist()]
                except Exception:
                    pass

            # Point forecast: median
            f_vals = np.quantile(samples, 0.5, axis=0)

            params_used = build_params_used(
                {'model_name': model_name, 'device_map': device_map},
                quantiles_dict=fq,
                context_length=ctx_len if ctx_len else n
            )
            
            if known_covariates is not None:
                params_used['covariates_used'] = True
                params_used['n_covariates'] = int(known_covariates.shape[-1])

            if f_vals is None:
                raise RuntimeError("chronos2 error: no point forecast produced")
                
            return ForecastResult(forecast=f_vals, params_used=params_used, metadata={"quantiles": fq})
            
        except Exception as ex:
            raise RuntimeError(f"chronos2 error: {ex}")

@ForecastRegistry.register("timesfm")
class TimesFMMethod(PretrainedMethod):
    @property
    def name(self) -> str:
        return "timesfm"
        
    @property
    def required_packages(self) -> List[str]:
        return ["timesfm", "torch"]

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        p = params or {}
        ctx_len = int(p.get('context_length', 0) or 0)
        quantiles = process_quantile_levels(p.get('quantiles'))
        
        vals = series.values
        n = len(vals)
        
        # Extract context window using DRY helper
        context = extract_context_window(vals, ctx_len, n, dtype=float)

        f_vals: Optional[np.ndarray] = None
        fq: Dict[str, List[float]] = {}
        
        # Import required modules using DRY helper
        modules, import_error = safe_import_modules(['timesfm'], 'timesfm')
        if import_error:
            raise RuntimeError(import_error)
        
        _timesfm = modules['timesfm']
        
        # Try to detect namespace shadowing
        if not (hasattr(_timesfm, 'TimesFM_2p5_200M_torch') or hasattr(_timesfm, 'ForecastConfig')):
            try:
                from timesfm import torch as _timesfm_torch  # type: ignore
                _timesfm = _timesfm_torch
            except Exception:
                _p = getattr(_timesfm, '__path__', None)
                _p_str = str(list(_p)) if _p is not None else 'unknown'
                raise RuntimeError(f"timesfm import resolved to namespace at {_p_str}. Rename/remove local 'timesfm' folder or pip install -e the official repo.")
        # Prefer new API
        _cls_name = 'TimesFM_2p5_200M_torch'
        _has_new = hasattr(_timesfm, _cls_name) and hasattr(_timesfm, 'ForecastConfig')
        if not _has_new:
            raise RuntimeError("timesfm installed but API not recognized (missing TimesFM_2p5_200M_torch/ForecastConfig). Update the package.")
        
        try:
            _Cls = getattr(_timesfm, _cls_name)
            _mdl = _Cls()
            _max_ctx = int(ctx_len) if ctx_len and int(ctx_len) > 0 else None
            _cfg_kwargs: Dict[str, Any] = {
                'max_context': _max_ctx or min(int(n), 1024),
                'max_horizon': int(horizon),
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
            pf, qf = _mdl.forecast(horizon=int(horizon), inputs=_inp)
            if pf is not None:
                arr = np.asarray(pf, dtype=float)
                arr = arr[0] if arr.ndim == 2 else arr
                vals_arr = np.asarray(arr, dtype=float)
                f_vals = adjust_forecast_length(vals_arr, horizon)
            if quantiles and qf is not None:
                qarr = np.asarray(qf, dtype=float)
                if qarr.ndim == 3 and qarr.shape[0] >= 1:
                    Q = qarr.shape[-1]
                    level_map = {str(l/10.0): (l if Q >= 10 else None) for l in range(1, 10)}
                    for q in list(quantiles or []):
                        try:
                            key = f"{float(q):.1f}"
                        except Exception:
                            continue
                        idx = level_map.get(key)
                        if idx is None:
                            continue
                        col = qarr[0, :horizon, idx] if idx < qarr.shape[-1] else None
                        if col is not None:
                            fq[key] = [float(v) for v in np.asarray(col, dtype=float).tolist()]
            params_used = build_params_used(
                {'timesfm_model': _cls_name},
                quantiles_dict=fq,
                context_length=int(_max_ctx or n)
            )
            
            if f_vals is None:
                 raise RuntimeError("timesfm error: no point forecast produced")
                 
            return ForecastResult(forecast=f_vals, params_used=params_used, metadata={"quantiles": fq})
        except Exception as ex:
            raise RuntimeError(f"timesfm error: {ex}")

@ForecastRegistry.register("lag_llama")
class LagLlamaMethod(PretrainedMethod):
    @property
    def name(self) -> str:
        return "lag_llama"
        
    @property
    def required_packages(self) -> List[str]:
        return ["lag-llama", "gluonts", "torch"]

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
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
                raise RuntimeError(f"lag_llama requires params.ckpt_path or the ability to auto-download via huggingface_hub. Tried default repo but failed: {ex}")

        ctx_len = int(p.get('context_length', 32) or 32)
        num_samples = int(p.get('num_samples', 100) or 100)
        use_rope = bool(p.get('use_rope_scaling', False))
        freq = str(p.get('freq', 'H'))
        quantiles = p.get('quantiles') if isinstance(p.get('quantiles'), (list, tuple)) else None

        vals = series.values
        
        # Select context window
        if ctx_len and ctx_len > 0:
            k = int(min(len(vals), ctx_len))
            context = np.asarray(vals[-k:], dtype=float)
        else:
            context = np.asarray(vals, dtype=float)

        try:
            import torch  # type: ignore
            from lag_llama.gluon.estimator import LagLlamaEstimator  # type: ignore
            from gluonts.evaluation import make_evaluation_predictions  # type: ignore
            from gluonts.dataset.common import ListDataset  # type: ignore
            import pandas as pd  # type: ignore
        except Exception as ex:
            raise RuntimeError(f"lag_llama dependencies missing: {ex}")

        # PyTorch 2.6+ defaults torch.load(..., weights_only=True) which blocks unpickling
        # of custom classes used by GluonTS (e.g., StudentTOutput).
        try:
            _add_safe = getattr(torch.serialization, "add_safe_globals", None)
            if callable(_add_safe):
                _to_allow = []
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
            raise RuntimeError(f"failed to load Lag-Llama checkpoint: {ex}")

        # Optional rope scaling when context exceeds training
        rope_scaling = None
        try:
            base_ctx = int(est_args.get('context_length', 32))
            if use_rope:
                factor = max(1.0, float((ctx_len + int(horizon)) / max(1, base_ctx)))
                rope_scaling = {"type": "linear", "factor": float(factor)}
        except Exception:
            rope_scaling = None

        try:
            estimator = LagLlamaEstimator(
                ckpt_path=str(ckpt_path),
                prediction_length=int(horizon),
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
                raise RuntimeError("lag_llama produced no forecasts")
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
                raise RuntimeError("lag_llama could not extract forecast values")
            f_vals = vals[:horizon] if vals.size >= horizon else np.pad(vals, (0, horizon - vals.size), mode='edge')

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
                    fq[str(qf)] = [float(v) for v in q_arr[:horizon].tolist()]

        except Exception as ex:
            raise RuntimeError(f"lag_llama inference error: {ex}")

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

        return ForecastResult(forecast=f_vals, params_used=params_used, metadata={"quantiles": fq})

# Backward compatibility wrappers
def forecast_chronos_bolt(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    try:
        res = ForecastRegistry.get("chronos_bolt").forecast(pd.Series(series), fh, 0, params)
        return res.forecast, res.metadata.get("quantiles"), res.params_used, None
    except Exception as e:
        return None, None, {}, str(e)

def forecast_timesfm(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    try:
        res = ForecastRegistry.get("timesfm").forecast(pd.Series(series), fh, 0, params)
        return res.forecast, res.metadata.get("quantiles"), res.params_used, None
    except Exception as e:
        return None, None, {}, str(e)

def forecast_lag_llama(
    *,
    series: np.ndarray,
    fh: int,
    params: Dict[str, Any],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]], Dict[str, Any], Optional[str]]:
    try:
        res = ForecastRegistry.get("lag_llama").forecast(pd.Series(series), fh, 0, params)
        return res.forecast, res.metadata.get("quantiles"), res.params_used, None
    except Exception as e:
        return None, None, {}, str(e)
