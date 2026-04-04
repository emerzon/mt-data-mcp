"""Spectral/frequency domain filters: FFT, Butterworth."""
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from skimage.restoration import denoise_tv_chambolle as _denoise_tv_chambolle

try:
    from scipy.signal import butter as _butter
    from scipy.signal import filtfilt as _filtfilt
    from scipy.signal import lfilter as _lfilter
except Exception:
    _butter = _filtfilt = _lfilter = None  # type: ignore

from ..base import register_filter, _series_like


@register_filter('lowpass_fft')
def _denoise_lowpass_fft_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    cutoff_ratio = float(params.get('cutoff_ratio', 0.1))
    X = np.fft.rfft(x)
    kmax = int(len(X) * cutoff_ratio)
    Y = np.zeros_like(X)
    Y[:max(1, kmax)] = X[:max(1, kmax)]
    y = np.fft.irfft(Y, n=len(x))
    return _series_like(s, y)


def _butterworth_filter(
    x: np.ndarray,
    cutoff: Any,
    order: int,
    btype: str,
    causality: str,
    padlen: Optional[int],
) -> np.ndarray:
    if _butter is None:
        return x
    try:
        order_val = max(1, int(order))
    except Exception:
        order_val = 4
    Wn: Any = None
    if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
        lo = float(cutoff[0])
        hi = float(cutoff[1])
        if not (0 < lo < hi < 0.5):
            return x
        Wn = [lo, hi]
        btype_val = btype or "bandpass"
    else:
        try:
            cval = float(cutoff)
        except Exception:
            cval = 0.1
        if not (0 < cval < 0.5):
            return x
        Wn = cval
        btype_val = btype or "low"
    b, a = _butter(order_val, Wn, btype=btype_val, analog=False)
    if causality == "zero_phase" and _filtfilt is not None:
        if padlen is None:
            return _filtfilt(b, a, x)
        return _filtfilt(b, a, x, padlen=int(padlen))
    if _lfilter is None:
        return x
    return _lfilter(b, a, x)


@register_filter('butterworth')
def _denoise_butterworth_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    cutoff = params.get('cutoff', 0.1)
    order = int(params.get('order', 4))
    btype = str(params.get('btype', 'low'))
    padlen = params.get('padlen')
    y = _butterworth_filter(x, cutoff=cutoff, order=order, btype=btype, causality=causality, padlen=padlen)
    return _series_like(s, y)
