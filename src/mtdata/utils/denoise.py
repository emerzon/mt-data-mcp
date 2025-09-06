from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pywt as _pywt  # type: ignore
except Exception:
    _pywt = None  # optional
try:
    from PyEMD import EMD as _EMD, EEMD as _EEMD, CEEMDAN as _CEEMDAN  # type: ignore
except Exception:
    _EMD = _EEMD = _CEEMDAN = None  # optional


def _denoise_series(
    s: pd.Series,
    method: str = 'none',
    params: Optional[Dict[str, Any]] = None,
    causality: str = 'zero_phase',
) -> pd.Series:
    if params is None:
        params = {}
    method = (method or 'none').lower().strip()
    n = len(s)
    if n < 3:
        return s
    if method == 'none':
        return s
    # Forward/backward fill using modern accessors to avoid FutureWarning
    x = s.astype(float).ffill().bfill().values
    if method == 'ema':
        alpha = params.get('alpha')
        span = params.get('span', 10)
        if alpha is not None:
            y = pd.Series(x).ewm(alpha=float(alpha), adjust=False).mean().values
        else:
            y = pd.Series(x).ewm(span=int(span), adjust=False).mean().values
        if causality == 'zero_phase':
            y2 = pd.Series(y[::-1]).ewm(span=int(span), adjust=False).mean().values[::-1]
            y = 0.5 * (y + y2)
        return pd.Series(y, index=s.index)
    if method == 'sma':
        window = max(1, int(params.get('window', 10)))
        if causality == 'zero_phase':
            y = pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().values
        else:
            y = pd.Series(x).rolling(window=window, min_periods=1).mean().values
        return pd.Series(y, index=s.index)
    if method == 'median':
        window = max(1, int(params.get('window', 7)))
        if causality == 'zero_phase':
            y = pd.Series(x).rolling(window=window, center=True, min_periods=1).median().values
        else:
            y = pd.Series(x).rolling(window=window, min_periods=1).median().values
        return pd.Series(y, index=s.index)
    if method == 'lowpass_fft':
        cutoff_ratio = float(params.get('cutoff_ratio', 0.1))
        X = np.fft.rfft(x)
        kmax = int(len(X) * cutoff_ratio)
        Y = np.zeros_like(X)
        Y[:max(1, kmax)] = X[:max(1, kmax)]
        y = np.fft.irfft(Y, n=len(x))
        return pd.Series(y, index=s.index)
    if method == 'wavelet' and _pywt is not None:
        wavelet = str(params.get('wavelet', 'db4'))
        level = params.get('level')
        mode = str(params.get('mode', 'soft'))
        coeffs = _pywt.wavedec(x, wavelet, mode='periodization', level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs) > 1 else np.std(x)
        thr = params.get('threshold', 'auto')
        thr_val = float(sigma * np.sqrt(2 * np.log(len(x)))) if thr == 'auto' else float(thr)
        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(_pywt.threshold(c, thr_val, mode=mode))
        y = _pywt.waverec(new_coeffs, wavelet, mode='periodization')[: len(x)]
        return pd.Series(y, index=s.index)
    if method in ('emd', 'eemd', 'ceemdan') and any(x is not None for x in (_EMD, _EEMD, _CEEMDAN)):
        xnp = np.asarray(x, dtype=float)
        max_imfs = params.get('max_imfs', 'auto')
        if isinstance(max_imfs, str) and max_imfs == 'auto':
            import math
            k = int(max(2, min(10, round(math.log2(len(xnp))))))
        else:
            k = int(max_imfs)
        if method == 'emd' and _EMD is not None:
            emd = _EMD()
            imfs = emd.emd(xnp, max_imf=k)
        elif method == 'eemd' and _EEMD is not None:
            noise_strength = float(params.get('noise_strength', 0.2))
            trials = int(params.get('trials', 100))
            random_state = params.get('random_state')
            eemd = _EEMD(trials=trials, noise_strength=noise_strength)
            if random_state is not None:
                eemd.random_state = int(random_state)
            imfs = eemd.eemd(xnp, max_imf=k)
        else:
            if _CEEMDAN is not None:
                noise_strength = float(params.get('noise_strength', 0.2))
                trials = int(params.get('trials', 100))
                random_state = params.get('random_state')
                ce = _CEEMDAN(trials=trials, noise_strength=noise_strength)
                if random_state is not None:
                    ce.random_state = int(random_state)
                imfs = ce.ceemdan(xnp, max_imf=k)
            else:
                return s
        imfs = np.atleast_2d(imfs)
        resid = xnp - imfs.sum(axis=0)
        k_all = list(range(imfs.shape[0]))
        keep_imfs = params.get('keep_imfs')
        drop_imfs = params.get('drop_imfs', [0])
        if isinstance(keep_imfs, (list, tuple)) and len(keep_imfs) > 0:
            k_sel = [k for k in keep_imfs if 0 <= int(k) < imfs.shape[0]]
        elif isinstance(drop_imfs, (list, tuple)) and len(drop_imfs) > 0:
            drop = {int(k) for k in drop_imfs}
            k_sel = [k for k in k_all if k not in drop]
        else:
            k_sel = [k for k in k_all if k != 0]
        y = resid + imfs[k_sel].sum(axis=0) if len(k_sel) > 0 else resid
        return pd.Series(y, index=s.index)
    return s


def _apply_denoise(
    df: pd.DataFrame,
    spec: Optional[Dict[str, Any]],
    default_when: str = 'post_ti',
) -> List[str]:
    added_cols: List[str] = []
    if not spec or not isinstance(spec, dict):
        return added_cols
    method = str(spec.get('method', 'none')).lower()
    if method == 'none':
        return added_cols
    params = spec.get('params') or {}
    cols = spec.get('columns') or 'ohlcv'
    # Normalize columns selection
    if isinstance(cols, str):
        key = cols.strip().lower()
        if key in ('ohlcv', 'ohlc', 'price'):
            # Map to price + volume columns present
            selected = []
            for name in ('open', 'high', 'low', 'close'):
                if name in df.columns:
                    selected.append(name)
            # Prefer real volume, else tick_volume
            if 'volume' in df.columns:
                selected.append('volume')
            elif 'tick_volume' in df.columns:
                selected.append('tick_volume')
            cols = selected if selected else ['close']
        elif key in ('all', '*', 'numeric'):
            try:
                cols = [
                    c for c in df.columns
                    if c != 'time' and not str(c).startswith('_') and pd.api.types.is_numeric_dtype(df[c])
                ]
            except Exception:
                cols = ['close']
        else:
            # Support comma/space-separated list in CLI shorthand
            parts = [p.strip() for p in cols.replace(',', ' ').split() if p.strip()]
            cols = parts if parts else ['close']
    when = str(spec.get('when') or default_when)
    causality = str(spec.get('causality') or ('causal' if when == 'pre_ti' else 'zero_phase'))
    keep_original = bool(spec.get('keep_original')) if 'keep_original' in spec else (when != 'pre_ti')
    suffix = str(spec.get('suffix') or '_dn')

    for col in cols:
        if col not in df.columns:
            continue
        try:
            y = _denoise_series(df[col], method=method, params=params, causality=causality)
        except Exception:
            continue
        if keep_original:
            new_col = f"{col}{suffix}"
            df[new_col] = y
            added_cols.append(new_col)
        else:
            df[col] = y
    return added_cols


def get_denoise_methods_data() -> Dict[str, Any]:
    def avail_requires(name: str) -> Tuple[bool, str]:
        if name == 'wavelet':
            return (_pywt is not None, 'PyWavelets')
        if name in ('emd', 'eemd', 'ceemdan'):
            return (any(x is not None for x in (_EMD, _EEMD, _CEEMDAN)), 'EMD-signal')
        return (True, '')

    methods: List[Dict[str, Any]] = []
    base_defaults = {"when": "pre_ti", "columns": ["close"], "keep_original": False, "suffix": "_dn"}

    def add(method: str, description: str, params: List[Dict[str, Any]], supports: Dict[str, bool]):
        available, requires = avail_requires(method)
        methods.append({
            "method": method,
            "available": bool(available),
            "requires": requires,
            "description": description,
            "params": params,
            "supports": supports,
            "defaults": base_defaults,
        })

    add("none", "No denoising (identity).", [], {"causal": True, "zero_phase": True})
    add("ema", "Exponential moving average; causal by default; zero-phase via forward-backward pass.", [
        {"name": "span", "type": "int", "default": 10, "description": "Smoothing span; alternative to alpha."},
        {"name": "alpha", "type": "float", "default": None, "description": "Direct smoothing factor in (0,1]; overrides span if set."},
    ], {"causal": True, "zero_phase": True})
    add("sma", "Simple moving average; causal or zero-phase (centered convolution).", [
        {"name": "window", "type": "int", "default": 10, "description": "Window length in samples."},
    ], {"causal": True, "zero_phase": True})
    add("median", "Rolling median; robust to spikes; causal or zero-phase (centered).", [
        {"name": "window", "type": "int", "default": 7, "description": "Window length in samples (odd recommended)."},
    ], {"causal": True, "zero_phase": True})
    add("lowpass_fft", "Zero-phase low-pass filtering in frequency domain; parameterized by cutoff ratio.", [
        {"name": "cutoff_ratio", "type": "float", "default": 0.1, "description": "Cutoff as fraction of Nyquist (0, 0.5]."},
    ], {"causal": False, "zero_phase": True})
    add("wavelet", "Wavelet shrinkage denoising using PyWavelets; preserves sharp changes better than linear filters.", [
        {"name": "wavelet", "type": "str", "default": "db4", "description": "Wavelet family, e.g., 'db4', 'sym5'."},
        {"name": "level", "type": "int|null", "default": None, "description": "Decomposition level; auto if omitted."},
        {"name": "threshold", "type": "float|\"auto\"", "default": "auto", "description": "Shrinkage threshold; 'auto' uses universal threshold."},
        {"name": "mode", "type": "str", "default": "soft", "description": "Shrinkage mode: 'soft' or 'hard'."},
    ], {"causal": False, "zero_phase": True})
    add("emd", "Empirical Mode Decomposition; reconstruct after dropping high-frequency IMFs.", [
        {"name": "drop_imfs", "type": "int[]", "default": [0], "description": "IMF indices to drop (0 is highest frequency)."},
        {"name": "keep_imfs", "type": "int[]", "default": None, "description": "Explicit list of IMFs to keep; overrides drop_imfs."},
        {"name": "max_imfs", "type": "int|\"auto\"", "default": "auto", "description": "Max IMFs; 'auto' ≈ log2(n), capped to [2,10]."},
    ], {"causal": False, "zero_phase": True})
    add("eemd", "Ensemble EMD; averages decompositions with added noise for robustness.", [
        {"name": "drop_imfs", "type": "int[]", "default": [0], "description": "IMF indices to drop (0 is highest frequency)."},
        {"name": "keep_imfs", "type": "int[]", "default": None, "description": "Explicit list of IMFs to keep; overrides drop_imfs."},
        {"name": "max_imfs", "type": "int|\"auto\"", "default": "auto", "description": "Max IMFs; 'auto' ≈ log2(n), capped to [2,10]."},
        {"name": "noise_strength", "type": "float", "default": 0.2, "description": "Relative noise amplitude used in ensembles."},
        {"name": "trials", "type": "int", "default": 100, "description": "Number of ensemble trials."},
        {"name": "random_state", "type": "int", "default": None, "description": "Random seed for reproducibility."},
    ], {"causal": False, "zero_phase": True})
    add("ceemdan", "Complementary EEMD with adaptive noise; improved reconstruction quality.", [
        {"name": "drop_imfs", "type": "int[]", "default": [0], "description": "IMF indices to drop (0 is highest frequency)."},
        {"name": "keep_imfs", "type": "int[]", "default": None, "description": "Explicit list of IMFs to keep; overrides drop_imfs."},
        {"name": "max_imfs", "type": "int|\"auto\"", "default": "auto", "description": "Max IMFs; 'auto' ≈ log2(n), capped to [2,10]."},
        {"name": "noise_strength", "type": "float", "default": 0.2, "description": "Relative noise amplitude used in decomposition."},
        {"name": "trials", "type": "int", "default": 100, "description": "Used if falling back to EEMD implementation."},
        {"name": "random_state", "type": "int", "default": None, "description": "Random seed for reproducibility."},
    ], {"causal": False, "zero_phase": True})

    return {"success": True, "schema_version": 1, "methods": methods}
