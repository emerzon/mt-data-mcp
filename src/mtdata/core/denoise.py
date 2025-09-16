
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np

from .schema import DenoiseSpec
from .server import mcp

try:
    import pywt as _pywt  # type: ignore
except Exception:
    _pywt = None  # optional
try:
    from PyEMD import EMD as _EMD, EEMD as _EEMD, CEEMDAN as _CEEMDAN  # type: ignore
except Exception:
    _EMD = _EEMD = _CEEMDAN = None  # optional


_DENOISE_METHODS = (
    "none",        # no-op
    "ema",         # exponential moving average
    "sma",         # simple moving average
    "median",      # rolling median
    "lowpass_fft", # zero-phase FFT low-pass
    "wavelet",     # wavelet shrinkage (PyWavelets optional)
    "emd",         # empirical mode decomposition (PyEMD optional)
    "eemd",        # ensemble EMD (PyEMD optional)
    "ceemdan",     # complementary EEMD with adaptive noise (PyEMD optional)
)

def _denoise_series(
    s: pd.Series,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    causality: Optional[str] = None,
) -> pd.Series:
    """Apply denoising to a single numeric Series and return the result.

    Supported methods (no external deps): ema, sma, median, lowpass_fft.
    - ema: params {span:int|None, alpha:float|None}
    - sma: params {window:int}
    - median: params {window:int}
    - lowpass_fft: params {cutoff_ratio:float in (0, 0.5], taper:bool}
    - wavelet: params {wavelet:str, level:int|None, threshold:float|"auto", mode:"soft"|"hard"}
    - emd/eemd/ceemdan: params {drop_imfs:list[int], keep_imfs:list[int], max_imfs:int, noise_strength:float, trials:int, random_state:int}
    """
    if params is None:
        params = {}
    method = (method or 'none').lower()
    if method == 'none':
        return s

    # Ensure float for numeric stability
    x = s.astype('float64')

    if method == 'ema':
        span = params.get('span')
        alpha = params.get('alpha')
        if alpha is None and span is None:
            span = 10
        # Base forward EMA
        y_fwd = x.ewm(span=span, alpha=alpha, adjust=False).mean()
        if (causality or '').lower() == 'zero_phase':
            # Two-pass EMA (forward + backward) to approximate zero-phase
            y_bwd = y_fwd.iloc[::-1].ewm(span=span, alpha=alpha, adjust=False).mean().iloc[::-1]
            return y_bwd
        return y_fwd

    if method == 'sma':
        window = int(params.get('window', 10))
        if window <= 1:
            return x
        if (causality or 'causal').lower() == 'causal':
            return x.rolling(window=window, min_periods=1).mean()
        # zero-phase via symmetric convolution with reflection padding
        k = np.ones(window, dtype=float) / float(window)
        pad = window // 2
        xpad = np.pad(x.to_numpy(), (pad, pad), mode='reflect')
        y = np.convolve(xpad, k, mode='same')[pad:-pad]
        return pd.Series(y, index=x.index)

    if method == 'median':
        window = int(params.get('window', 7))
        if window <= 1:
            return x
        center = (causality or '').lower() == 'zero_phase'
        return x.rolling(window=window, min_periods=1, center=center).median()

    if method == 'lowpass_fft':
        # Zero-phase by construction; ignore 'causal' and document behavior
        cutoff_ratio = float(params.get('cutoff_ratio', 0.1))
        cutoff_ratio = max(1e-6, min(0.5, cutoff_ratio))
        # Fill NaNs before FFT to avoid propagation
        xnp = x.fillna(method='ffill').fillna(method='bfill').fillna(0.0).to_numpy()
        n = len(xnp)
        if n <= 2:
            return x
        X = np.fft.rfft(xnp)
        freqs = np.fft.rfftfreq(n, d=1.0)  # normalized per-sample
        cutoff = cutoff_ratio * 0.5 * 2.0  # interpret ratio vs. Nyquist; clamp in [0,0.5]
        mask = freqs <= cutoff
        X_filtered = X * mask
        y = np.fft.irfft(X_filtered, n=n)
        return pd.Series(y, index=x.index)

    if method == 'wavelet':
        if _pywt is None:
            return s
        wname = str(params.get('wavelet', 'db4'))
        mode = str(params.get('mode', 'soft')).lower()
        thr = params.get('threshold', 'auto')
        # choose decomposition level if not provided
        try:
            w = _pywt.Wavelet(wname)
            max_level = _pywt.dwt_max_level(len(x), w.dec_len)
        except Exception:
            w = _pywt.Wavelet('db4') if _pywt else None
            max_level = 4
        level = int(params.get('level', max(1, min(5, max_level))))
        # Fill NaNs
        xnp = x.fillna(method='ffill').fillna(method='bfill').fillna(0.0).to_numpy()
        coeffs = _pywt.wavedec(xnp, w, mode='symmetric', level=level)
        cA, cDs = coeffs[0], coeffs[1:]
        if thr == 'auto':
            # universal threshold using first detail level
            d1 = cDs[-1] if len(cDs) > 0 else cA
            sigma = np.median(np.abs(d1 - np.median(d1))) / 0.6745 if len(d1) else 0.0
            lam = sigma * np.sqrt(2.0 * np.log(len(xnp) + 1e-9))
        else:
            try:
                lam = float(thr)
            except Exception:
                lam = 0.0
        new_coeffs = [cA]
        for d in cDs:
            if mode == 'hard':
                d_new = d * (np.abs(d) >= lam)
            else:
                d_new = np.sign(d) * np.maximum(np.abs(d) - lam, 0.0)
            new_coeffs.append(d_new)
        y = _pywt.waverec(new_coeffs, w, mode='symmetric')
        if len(y) != len(xnp):
            y = y[:len(xnp)]
        return pd.Series(y, index=x.index)

    if method in ('emd', 'eemd', 'ceemdan'):
        if _EMD is None and _EEMD is None and _CEEMDAN is None:
            return s
        xnp = x.fillna(method='ffill').fillna(method='bfill').fillna(0.0).to_numpy()
        max_imfs = params.get('max_imfs')
        if isinstance(max_imfs, str) and str(max_imfs).lower() == 'auto':
            max_imfs = None
        drop_imfs = params.get('drop_imfs')
        keep_imfs = params.get('keep_imfs')
        ns = params.get('noise_strength', 0.2)
        trials = int(params.get('trials', 100))
        rng = params.get('random_state')
        # Sensible default for number of IMFs: ~log2(n), capped [2,10]
        n = len(xnp)
        if max_imfs is None:
            try:
                est = int(np.ceil(np.log2(max(8, n))))
            except Exception:
                est = 6
            max_imfs = max(2, min(10, est))

        try:
            if method == 'eemd' and _EEMD is not None:
                decomp = _EEMD()
                if rng is not None:
                    decomp.trials = trials
                    decomp.noise_seed(rng)
                else:
                    decomp.trials = trials
                decomp.noise_strength = ns
                imfs = decomp.eemd(xnp, max_imf=max_imfs)
            elif method == 'ceemdan' and _CEEMDAN is not None:
                decomp = _CEEMDAN()
                if rng is not None:
                    decomp.random_seed = rng
                decomp.noise_strength = ns
                imfs = decomp.ceemdan(xnp, max_imf=int(max_imfs) if max_imfs is not None else None)
            else:
                # fallback to plain EMD
                decomp = _EMD() if _EMD is not None else (_EEMD() if _EEMD is not None else _CEEMDAN())
                imfs = decomp.emd(xnp, max_imf=int(max_imfs) if max_imfs is not None else None) if hasattr(decomp, 'emd') else decomp.eemd(xnp, max_imf=int(max_imfs) if max_imfs is not None else None)
        except Exception:
            return s

        if imfs is None or len(imfs) == 0:
            return s
        imfs = np.atleast_2d(imfs)
        # Residual (trend) component not returned explicitly; reconstruct it
        resid = xnp - imfs.sum(axis=0)
        k_all = list(range(imfs.shape[0]))
        if isinstance(keep_imfs, (list, tuple)) and len(keep_imfs) > 0:
            k_sel = [k for k in keep_imfs if 0 <= int(k) < imfs.shape[0]]
        elif isinstance(drop_imfs, (list, tuple)) and len(drop_imfs) > 0:
            drop = {int(k) for k in drop_imfs}
            k_sel = [k for k in k_all if k not in drop]
        else:
            # Default: drop the first IMF (highest frequency)
            k_sel = [k for k in k_all if k != 0]
        y = resid + imfs[k_sel].sum(axis=0) if len(k_sel) > 0 else resid
        return pd.Series(y, index=x.index)

    # Unknown method: return original
    return s


def _apply_denoise(
    df: pd.DataFrame,
    spec: Optional[DenoiseSpec],
    default_when: str = 'post_ti',
) -> List[str]:
    """Apply denoising per spec to selected columns in-place.

    Returns list of columns added (if any). May also overwrite columns when keep_original=False.
    """
    added_cols: List[str] = []
    if not spec or not isinstance(spec, dict):
        return added_cols
    method = str(spec.get('method', 'none')).lower()
    if method == 'none':
        return added_cols
    params = spec.get('params') or {}
    cols = spec.get('columns') or ['close']
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

def _get_denoise_methods_data_safe() -> Dict[str, Any]:
    try:
        from ..utils.denoise import get_denoise_methods_data
        return get_denoise_methods_data()
    except Exception as e:
        return {"error": f"Error listing denoise methods: {e}"}


@mcp.tool()
def list_denoise_methods() -> Dict[str, Any]:
    """List available denoise methods and their parameters."""
    return _get_denoise_methods_data_safe()
