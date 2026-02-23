from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
from skimage.restoration import denoise_tv_chambolle as _denoise_tv_chambolle

try:
    import pywt as _pywt  # type: ignore
except Exception:
    _pywt = None  # optional
try:
    from PyEMD import EMD as _EMD, EEMD as _EEMD, CEEMDAN as _CEEMDAN  # type: ignore
except Exception:
    _EMD = _EEMD = _CEEMDAN = None  # optional
try:
    from scipy import sparse as _sps  # type: ignore
    from scipy.sparse import linalg as _sps_linalg  # type: ignore
except Exception:
    _sps = _sps_linalg = None  # optional
try:
    from scipy.signal import savgol_filter as _savgol_filter  # type: ignore
except Exception:
    _savgol_filter = None  # optional
try:
    from scipy.signal import butter as _butter  # type: ignore
    from scipy.signal import filtfilt as _filtfilt  # type: ignore
    from scipy.signal import lfilter as _lfilter  # type: ignore
except Exception:
    _butter = _filtfilt = _lfilter = None  # optional
try:
    from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d  # type: ignore
except Exception:
    _gaussian_filter1d = None  # optional
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess  # type: ignore
except Exception:
    _lowess = None  # optional
try:
    from statsmodels.tsa.seasonal import STL as _STL  # type: ignore
except Exception:
    _STL = None  # optional
try:
    from vmdpy import VMD as _VMD  # type: ignore
except Exception:
    _VMD = None  # optional
def _hp_filter(x: np.ndarray, lamb: float) -> np.ndarray:
    if _sps is None or _sps_linalg is None:
        return x
    n = len(x)
    if n < 3:
        return x
    d = _sps.diags(
        [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )
    a = _sps.eye(n, format="csc") + float(lamb) * (d.T @ d)
    return np.asarray(_sps_linalg.spsolve(a, x))


def _whittaker_smooth(x: np.ndarray, lamb: float, order: int = 2) -> np.ndarray:
    if _sps is None or _sps_linalg is None:
        return x
    n = len(x)
    if n <= order or order < 1:
        return x
    coeffs = [(-1) ** k * math.comb(order, k) for k in range(order + 1)]
    diagonals = [np.full(n - order, coeffs[k], dtype=float) for k in range(order + 1)]
    offsets = list(range(order + 1))
    d = _sps.diags(diagonals, offsets, shape=(n - order, n), format="csc")
    a = _sps.eye(n, format="csc") + float(lamb) * (d.T @ d)
    return np.asarray(_sps_linalg.spsolve(a, x))


def _tv_denoise_1d(
    x: np.ndarray,
    weight: float,
    n_iter: int = 50,
    tol: float = 1e-4,
) -> np.ndarray:
    if weight <= 0:
        return x
    n = len(x)
    if n < 3:
        return x

    try:
        y = _denoise_tv_chambolle(
            x,
            weight=float(weight),
            eps=float(max(tol, 1e-12)),
            max_num_iter=max(1, int(n_iter)),
            channel_axis=None,
        )
    except TypeError:
        # Older scikit-image versions use n_iter_max naming.
        y = _denoise_tv_chambolle(
            x,
            weight=float(weight),
            eps=float(max(tol, 1e-12)),
            n_iter_max=max(1, int(n_iter)),
        )
    return np.asarray(y, dtype=float)


def _kalman_filter_1d(
    x: np.ndarray,
    process_var: float,
    measurement_var: float,
    initial_state: Optional[float] = None,
    initial_cov: Optional[float] = None,
) -> np.ndarray:
    n = len(x)
    xhat = np.zeros(n, dtype=float)
    p = np.zeros(n, dtype=float)
    meas = max(float(measurement_var), 1e-12)
    proc = max(float(process_var), 1e-12)
    xhat[0] = float(initial_state) if initial_state is not None else float(x[0])
    p[0] = float(initial_cov) if initial_cov is not None else meas
    for t in range(1, n):
        x_pred = xhat[t - 1]
        p_pred = p[t - 1] + proc
        k = p_pred / (p_pred + meas)
        xhat[t] = x_pred + k * (x[t] - x_pred)
        p[t] = (1 - k) * p_pred
    return xhat


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


def _hampel_filter(
    x: np.ndarray,
    window: int,
    n_sigmas: float,
    causality: str,
) -> np.ndarray:
    n = len(x)
    if n < 3:
        return x
    win = max(3, int(window))
    half = win // 2
    y = x.copy()
    for i in range(n):
        if causality == "causal":
            start = max(0, i - win + 1)
            end = i + 1
        else:
            start = max(0, i - half)
            end = min(n, i + half + 1)
        vals = x[start:end]
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        scale = 1.4826 * mad if mad > 0 else 0.0
        if scale > 0 and abs(x[i] - med) > float(n_sigmas) * scale:
            y[i] = med
    return y


def _bilateral_filter_1d(
    x: np.ndarray,
    sigma_s: float,
    sigma_r: float,
    truncate: float,
    causality: str,
) -> np.ndarray:
    n = len(x)
    if n < 3:
        return x
    if sigma_s <= 0 or sigma_r <= 0:
        return x
    radius = max(1, int(round(float(truncate) * float(sigma_s))))
    y = np.zeros_like(x)
    for i in range(n):
        if causality == "causal":
            start = max(0, i - radius)
            end = i + 1
        else:
            start = max(0, i - radius)
            end = min(n, i + radius + 1)
        idx = np.arange(start, end)
        if idx.size == 0:
            y[i] = x[i]
            continue
        dist = idx - i
        w_s = np.exp(-0.5 * (dist / float(sigma_s)) ** 2)
        w_r = np.exp(-0.5 * ((x[idx] - x[i]) / float(sigma_r)) ** 2)
        w = w_s * w_r
        denom = np.sum(w)
        y[i] = np.sum(w * x[idx]) / denom if denom > 0 else x[i]
    return y


def _wavelet_packet_denoise(
    x: np.ndarray,
    wavelet: str,
    level: Optional[int],
    threshold: Any,
    mode: str,
    threshold_scale: Any = None,
) -> np.ndarray:
    if _pywt is None:
        return x
    # PyWavelets' WaveletPacket/DWT path can fail with read-only views
    # (e.g. when upstream provides a Pandas-backed array). Always operate
    # on a writable, contiguous copy for robustness.
    try:
        x = np.array(x, dtype=float, copy=True, order="C")
    except Exception:
        pass
    try:
        w = _pywt.Wavelet(wavelet)
    except Exception:
        return x
    max_level = _pywt.dwt_max_level(len(x), w.dec_len)
    level_val = int(level) if level is not None else max(1, min(3, max_level))
    if level_val < 1:
        return x
    wp = _pywt.WaveletPacket(data=x, wavelet=wavelet, mode="periodization", maxlevel=level_val)
    nodes = wp.get_level(level_val, order="freq")
    if not nodes:
        return x
    coeffs = np.concatenate([node.data.ravel() for node in nodes])
    sigma = np.median(np.abs(coeffs)) / 0.6745 if coeffs.size else float(np.std(x))
    thr = threshold
    if threshold_scale == "auto":
        denom = float(np.std(x)) + 1e-12
        noise_ratio = min(1.0, sigma / denom) if denom > 0 else 0.0
        scale = 1.2 + 0.8 * noise_ratio
    elif threshold_scale is None:
        scale = 1.0
    else:
        scale = float(threshold_scale)
    if thr == "auto":
        thr_val = float(sigma * np.sqrt(2 * np.log(len(x)))) * scale
    else:
        thr_val = float(thr) * scale
    for node in nodes:
        node.data = _pywt.threshold(node.data, thr_val, mode=mode)
    y = wp.reconstruct(update=False)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)), mode="edge")
    return np.asarray(y[: len(x)])


def _ssa_denoise(
    x: np.ndarray,
    window: int,
    components: Optional[Any],
) -> np.ndarray:
    n = len(x)
    if n < 4:
        return x
    L = max(2, int(window))
    if L >= n:
        return x
    K = n - L + 1
    X = np.column_stack([x[i : i + L] for i in range(K)])
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if components is None:
        r = min(2, len(s))
    elif isinstance(components, float) and 0 < components <= 1:
        energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        r = int(np.searchsorted(energy, components) + 1)
    else:
        r = int(components)
    r = max(1, min(r, len(s)))
    Xr = (U[:, :r] * s[:r]) @ Vt[:r, :]
    y = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    for i in range(L):
        for j in range(K):
            y[i + j] += Xr[i, j]
            counts[i + j] += 1.0
    counts[counts == 0] = 1.0
    return y / counts


def _soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - float(thresh), 0.0)


def _l1_trend_filter(
    x: np.ndarray,
    lamb: float,
    n_iter: int,
    rho: float,
) -> np.ndarray:
    n = len(x)
    if n < 4 or lamb <= 0:
        return x
    if _sps is not None and _sps_linalg is not None:
        d = _sps.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n), format="csc")
        a = _sps.eye(n, format="csc") + float(rho) * (d.T @ d)
        solver = _sps_linalg.factorized(a) if hasattr(_sps_linalg, "factorized") else None
        z = np.zeros(n - 2, dtype=float)
        u = np.zeros(n - 2, dtype=float)
        y = x.copy()
        for _ in range(max(1, int(n_iter))):
            rhs = x + float(rho) * (d.T @ (z - u))
            y = solver(rhs) if solver is not None else _sps_linalg.spsolve(a, rhs)
            d_y = d @ y
            z = _soft_threshold(d_y + u, float(lamb) / float(rho))
            u = u + d_y - z
        return np.asarray(y)
    d = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        d[i, i : i + 3] = [1.0, -2.0, 1.0]
    a = np.eye(n, dtype=float) + float(rho) * (d.T @ d)
    z = np.zeros(n - 2, dtype=float)
    u = np.zeros(n - 2, dtype=float)
    y = x.copy()
    for _ in range(max(1, int(n_iter))):
        rhs = x + float(rho) * (d.T @ (z - u))
        y = np.linalg.solve(a, rhs)
        d_y = d @ y
        z = _soft_threshold(d_y + u, float(lamb) / float(rho))
        u = u + d_y - z
    return y


def _adaptive_lms_filter(
    x: np.ndarray,
    order: int,
    mu: float,
    eps: float = 1e-6,
    leak: float = 0.0,
    use_bias: bool = True,
) -> np.ndarray:
    n = len(x)
    k = max(1, int(order))
    mu_val = float(mu)
    if n < k + 1 or mu_val <= 0:
        return x
    leak_val = max(0.0, float(leak))
    if use_bias:
        w = np.zeros(k + 1, dtype=float)
        w[1:] = 1.0 / float(k)
    else:
        w = np.full(k, 1.0 / float(k), dtype=float)
    y = x.copy()
    for t in range(k, n):
        if use_bias:
            x_vec = np.concatenate(([1.0], x[t - k : t][::-1]))
        else:
            x_vec = x[t - k : t][::-1]
        y_hat = float(np.dot(w, x_vec))
        y[t] = y_hat
        err = x[t] - y_hat
        denom = float(np.dot(x_vec, x_vec)) + float(eps)
        step = mu_val / denom
        w = (1.0 - leak_val) * w + step * err * x_vec
    return y


def _adaptive_rls_filter(
    x: np.ndarray,
    order: int,
    lam: float,
    delta: float,
    use_bias: bool = True,
) -> np.ndarray:
    n = len(x)
    k = max(1, int(order))
    lam_val = float(lam)
    if n < k + 1 or lam_val <= 0 or lam_val > 1:
        return x
    delta_val = max(float(delta), 1e-6)
    if use_bias:
        w = np.zeros(k + 1, dtype=float)
        w[1:] = 1.0 / float(k)
        p = (1.0 / delta_val) * np.eye(k + 1, dtype=float)
    else:
        w = np.full(k, 1.0 / float(k), dtype=float)
        p = (1.0 / delta_val) * np.eye(k, dtype=float)
    y = x.copy()
    for t in range(k, n):
        if use_bias:
            x_vec = np.concatenate(([1.0], x[t - k : t][::-1]))
        else:
            x_vec = x[t - k : t][::-1]
        px = p @ x_vec
        denom = lam_val + float(np.dot(x_vec, px))
        if denom <= 0:
            y[t] = x[t]
            continue
        k_gain = px / denom
        y_hat = float(np.dot(w, x_vec))
        y[t] = y_hat
        err = x[t] - y_hat
        w = w + k_gain * err
        p = (p - np.outer(k_gain, x_vec) @ p) / lam_val
    return y


def _beta_irls_mean(
    values: np.ndarray,
    beta: float,
    n_iter: int,
    eps: float,
) -> float:
    if values.size == 0:
        return 0.0
    b = float(beta)
    if b >= 2.0:
        return float(np.mean(values))
    if b <= 0:
        return float(np.median(values))
    y = float(np.median(values)) if b <= 1.0 else float(np.mean(values))
    for _ in range(max(1, int(n_iter))):
        diff = np.abs(values - y)
        weights = (diff + eps) ** (b - 2.0)
        wsum = float(np.sum(weights))
        if wsum <= 0:
            break
        y_new = float(np.sum(weights * values) / wsum)
        if abs(y_new - y) <= eps:
            return y_new
        y = y_new
    return y


def _beta_smooth(
    x: np.ndarray,
    window: int,
    beta: float,
    n_iter: int,
    eps: float,
    causality: str,
) -> np.ndarray:
    n = len(x)
    if n < 3:
        return x
    win = max(3, int(window))
    half = win // 2
    y = x.copy()
    for i in range(n):
        if causality == "causal":
            start = max(0, i - win + 1)
            end = i + 1
        else:
            start = max(0, i - half)
            end = min(n, i + half + 1)
        vals = x[start:end]
        y[i] = _beta_irls_mean(vals, beta=beta, n_iter=n_iter, eps=eps)
    return y


def _vmd_denoise(
    x: np.ndarray,
    alpha: float,
    tau: float,
    k: int,
    dc: int,
    init: int,
    tol: float,
    keep_modes: Optional[Any],
    drop_modes: Optional[Any],
    keep_ratio: Optional[float],
) -> np.ndarray:
    if _VMD is None:
        return x
    k_val = max(1, int(k))
    u, _, omega = _VMD(x, float(alpha), float(tau), k_val, int(dc), int(init), float(tol))
    if u is None:
        return x
    modes = np.atleast_2d(u)
    if modes.shape[0] == len(x) and modes.shape[1] != len(x):
        modes = modes.T
        if omega is not None:
            omega_arr = np.atleast_2d(omega)
            if omega_arr.shape[0] == len(x) and omega_arr.shape[1] != len(x):
                omega = omega_arr.T
    idx_all = list(range(modes.shape[0]))
    if omega is not None:
        omega_arr = np.asarray(omega)
        if omega_arr.ndim > 1 and omega_arr.shape[0] != modes.shape[0] and omega_arr.shape[1] == modes.shape[0]:
            omega_arr = omega_arr.T
        if omega_arr.ndim > 1 and omega_arr.shape[0] == modes.shape[0]:
            freq = omega_arr[:, -1]
        else:
            freq = omega_arr
        try:
            order = list(np.argsort(freq))
        except Exception:
            order = idx_all
    else:
        order = idx_all
    if isinstance(keep_modes, (list, tuple)) and len(keep_modes) > 0:
        keep = []
        for idx in keep_modes:
            i = int(idx)
            if i < 0:
                i = modes.shape[0] + i
            if 0 <= i < modes.shape[0]:
                keep.append(i)
        idx_sel = keep
    else:
        if keep_ratio is not None and drop_modes is None:
            keep_ratio = max(0.0, min(float(keep_ratio), 1.0))
            total_energy = float(np.sum(modes ** 2))
            if total_energy > 0:
                cumulative = 0.0
                keep = []
                for i in order:
                    cumulative += float(np.sum(modes[i] ** 2))
                    keep.append(i)
                    if cumulative / total_energy >= keep_ratio:
                        break
                idx_sel = keep
            else:
                idx_sel = idx_all
        else:
            drop = drop_modes if drop_modes is not None else [-1]
            if isinstance(drop, (list, tuple)):
                drop_set = set()
                for idx in drop:
                    i = int(idx)
                    if i < 0:
                        i = modes.shape[0] + i
                    if 0 <= i < modes.shape[0]:
                        drop_set.add(i)
                idx_sel = [i for i in idx_all if i not in drop_set]
            else:
                idx_sel = idx_all
    if not idx_sel:
        idx_sel = idx_all
    y = modes[idx_sel].sum(axis=0)
    if len(y) != len(x) and modes.shape[0] == len(x):
        y = modes[idx_sel].sum(axis=1)
    return y

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
    if method == 'butterworth':
        cutoff = params.get('cutoff', 0.1)
        order = int(params.get('order', 4))
        btype = str(params.get('btype', 'low'))
        padlen = params.get('padlen')
        y = _butterworth_filter(x, cutoff=cutoff, order=order, btype=btype, causality=causality, padlen=padlen)
        return pd.Series(y, index=s.index)
    if method == 'hp':
        lamb = params.get('lamb', params.get('lambda', 1600.0))
        y = _hp_filter(x, float(lamb))
        return pd.Series(y, index=s.index)
    if method == 'savgol' and _savgol_filter is not None:
        window = int(params.get('window', 11))
        if window < 3:
            return s
        if window % 2 == 0:
            window += 1
        polyorder = int(params.get('polyorder', 2))
        polyorder = max(0, min(polyorder, window - 1))
        mode = str(params.get('mode', 'interp'))
        try:
            y = _savgol_filter(x, window_length=window, polyorder=polyorder, mode=mode)
        except Exception:
            return s
        return pd.Series(y, index=s.index)
    if method == 'tv':
        weight = params.get('weight', params.get('lambda', 'auto'))
        if weight == 'auto' or weight is None:
            scale = float(np.std(x))
            weight_val = 0.1 * scale if scale > 0 else 1.0
        else:
            weight_val = float(weight)
        n_iter = int(params.get('n_iter', 50))
        tol = float(params.get('tol', 1e-4))
        y = _tv_denoise_1d(x, weight=weight_val, n_iter=n_iter, tol=tol)
        return pd.Series(y, index=s.index)
    if method == 'kalman':
        measurement_var = params.get('measurement_var', params.get('r', 'auto'))
        process_var = params.get('process_var', params.get('q', 'auto'))
        series_var = float(np.var(x))
        if measurement_var == 'auto' or measurement_var is None:
            measurement_val = series_var if series_var > 0 else 1.0
        else:
            measurement_val = float(measurement_var)
        if process_var == 'auto' or process_var is None:
            process_val = measurement_val * 0.01
        else:
            process_val = float(process_var)
        init_state = params.get('initial_state')
        init_cov = params.get('initial_cov')
        y_fwd = _kalman_filter_1d(
            x,
            process_var=process_val,
            measurement_var=measurement_val,
            initial_state=init_state,
            initial_cov=init_cov,
        )
        if causality == 'zero_phase':
            y_bwd = _kalman_filter_1d(
                x[::-1],
                process_var=process_val,
                measurement_var=measurement_val,
                initial_state=init_state,
                initial_cov=init_cov,
            )[::-1]
            y = 0.5 * (y_fwd + y_bwd)
        else:
            y = y_fwd
        return pd.Series(y, index=s.index)
    if method == 'loess' and _lowess is not None:
        frac = float(params.get('frac', 0.2))
        it = int(params.get('it', 0))
        delta = float(params.get('delta', 0.0))
        exog = np.arange(len(x), dtype=float)
        y = _lowess(x, exog, frac=frac, it=it, delta=delta, return_sorted=False)
        return pd.Series(y, index=s.index)
    if method == 'stl' and _STL is not None:
        period = params.get('period')
        if period is None:
            return s
        period_val = int(period)
        if period_val < 2 or period_val >= len(x):
            return s
        seasonal = params.get('seasonal')
        trend = params.get('trend')
        low_pass = params.get('low_pass')
        robust = bool(params.get('robust', False))
        stl_kwargs: Dict[str, Any] = {"period": period_val, "robust": robust}
        if seasonal is not None:
            stl_kwargs["seasonal"] = int(seasonal)
        if trend is not None:
            stl_kwargs["trend"] = int(trend)
        if low_pass is not None:
            stl_kwargs["low_pass"] = int(low_pass)
        stl = _STL(x, **stl_kwargs)
        res = stl.fit()
        component = str(params.get('component', 'trend')).lower().strip()
        if component == 'seasonal':
            y = res.seasonal
        elif component == 'resid':
            y = res.resid
        elif component in ('trend+seasonal', 'trend_seasonal'):
            y = res.trend + res.seasonal
        elif component in ('trend+resid', 'trend_resid'):
            y = res.trend + res.resid
        else:
            y = res.trend
        return pd.Series(y, index=s.index)
    if method == 'whittaker':
        lamb = params.get('lamb', params.get('lambda', 1000.0))
        order = int(params.get('order', 2))
        y = _whittaker_smooth(x, float(lamb), order=order)
        return pd.Series(y, index=s.index)
    if method == 'gaussian' and _gaussian_filter1d is not None:
        sigma = float(params.get('sigma', 2.0))
        if sigma <= 0:
            return s
        truncate = float(params.get('truncate', 4.0))
        mode = str(params.get('mode', 'nearest'))
        y = _gaussian_filter1d(x, sigma=sigma, mode=mode, truncate=truncate)
        return pd.Series(y, index=s.index)
    if method == 'hampel':
        window = int(params.get('window', 7))
        n_sigmas = float(params.get('n_sigmas', 3.0))
        y = _hampel_filter(x, window=window, n_sigmas=n_sigmas, causality=causality)
        return pd.Series(y, index=s.index)
    if method == 'bilateral':
        sigma_s = float(params.get('sigma_s', 2.0))
        sigma_r = float(params.get('sigma_r', 0.5))
        truncate = float(params.get('truncate', 3.0))
        y = _bilateral_filter_1d(x, sigma_s=sigma_s, sigma_r=sigma_r, truncate=truncate, causality=causality)
        return pd.Series(y, index=s.index)
    if method == 'wavelet_packet' and _pywt is not None:
        wavelet = str(params.get('wavelet', 'db4'))
        level = params.get('level')
        mode = str(params.get('mode', 'soft'))
        thr = params.get('threshold', 'auto')
        thr_scale = params.get('threshold_scale', 'auto')
        y = _wavelet_packet_denoise(
            x,
            wavelet=wavelet,
            level=level,
            threshold=thr,
            mode=mode,
            threshold_scale=thr_scale,
        )
        return pd.Series(y, index=s.index)
    if method == 'ssa':
        window = int(params.get('window', max(10, len(x) // 3)))
        components = params.get('components')
        y = _ssa_denoise(x, window=window, components=components)
        return pd.Series(y, index=s.index)
    if method == 'l1_trend':
        lamb_param = params.get('lamb', params.get('lambda', 'auto'))
        n_iter = int(params.get('n_iter', 50))
        rho_param = params.get('rho', 'auto')
        rho = float(rho_param) if rho_param not in ('auto', None) else 1.0
        if lamb_param in ('auto', None):
            mean = float(np.mean(x))
            std = float(np.std(x))
            if std <= 0:
                return s
            x_scaled = (x - mean) / std
            scale = math.sqrt(max(len(x), 1) / 100.0)
            lamb = 5.0 * scale
            y_scaled = _l1_trend_filter(x_scaled, lamb=lamb, n_iter=n_iter, rho=rho)
            y = y_scaled * std + mean
        else:
            lamb = float(lamb_param)
            y = _l1_trend_filter(x, lamb=lamb, n_iter=n_iter, rho=rho)
        return pd.Series(y, index=s.index)
    if method == 'lms':
        order = int(params.get('order', 5))
        mu_param = params.get('mu', 'auto')
        mu = float(mu_param) if mu_param not in ('auto', None) else 0.5
        mu = max(1e-4, min(mu, 1.5))
        eps = float(params.get('eps', 1e-6))
        leak = float(params.get('leak', 0.0))
        bias = bool(params.get('bias', True))
        y_fwd = _adaptive_lms_filter(x, order=order, mu=mu, eps=eps, leak=leak, use_bias=bias)
        if causality == 'zero_phase':
            y_bwd = _adaptive_lms_filter(x[::-1], order=order, mu=mu, eps=eps, leak=leak, use_bias=bias)[::-1]
            y = 0.5 * (y_fwd + y_bwd)
        else:
            y = y_fwd
        return pd.Series(y, index=s.index)
    if method == 'rls':
        order = int(params.get('order', 5))
        lam_param = params.get('lambda_', params.get('lam', 'auto'))
        lam = float(lam_param) if lam_param not in ('auto', None) else 0.99
        lam = max(0.9, min(lam, 0.999))
        delta = float(params.get('delta', 1.0))
        bias = bool(params.get('bias', True))
        y_fwd = _adaptive_rls_filter(x, order=order, lam=lam, delta=delta, use_bias=bias)
        if causality == 'zero_phase':
            y_bwd = _adaptive_rls_filter(x[::-1], order=order, lam=lam, delta=delta, use_bias=bias)[::-1]
            y = 0.5 * (y_fwd + y_bwd)
        else:
            y = y_fwd
        return pd.Series(y, index=s.index)
    if method == 'beta':
        window = int(params.get('window', 9))
        beta = float(params.get('beta', 1.3))
        n_iter = int(params.get('n_iter', 20))
        eps = float(params.get('eps', 1e-6))
        y = _beta_smooth(x, window=window, beta=beta, n_iter=n_iter, eps=eps, causality=causality)
        return pd.Series(y, index=s.index)
    if method == 'vmd':
        alpha = float(params.get('alpha', 2000.0))
        tau = float(params.get('tau', 0.0))
        k = int(params.get('k', params.get('K', 5)))
        dc = int(params.get('dc', 0))
        init = int(params.get('init', 1))
        tol = float(params.get('tol', 1e-7))
        keep_modes = params.get('keep_modes')
        drop_modes = params.get('drop_modes')
        keep_ratio = params.get('keep_ratio', 'auto')
        if keep_ratio in ('auto', None):
            denom = float(np.std(x)) + 1e-12
            noise_level = float(np.std(np.diff(x))) / denom if denom > 0 else 0.0
            keep_ratio_val = 0.9 - 0.2 * min(1.0, noise_level)
            keep_ratio_val = max(0.7, min(0.95, keep_ratio_val))
        else:
            keep_ratio_val = float(keep_ratio)
        y = _vmd_denoise(
            x,
            alpha=alpha,
            tau=tau,
            k=k,
            dc=dc,
            init=init,
            tol=tol,
            keep_modes=keep_modes,
            drop_modes=drop_modes,
            keep_ratio=keep_ratio_val,
        )
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


def _resolve_denoise_base_col(
    df: pd.DataFrame,
    denoise: Optional[Dict[str, Any]],
    *,
    base_col: str = "close",
    default_when: str = "pre_ti",
) -> str:
    """Apply denoise when requested and return the effective base column name."""
    if not denoise:
        return base_col
    try:
        added = _apply_denoise(df, denoise, default_when=default_when)
        if f"{base_col}_dn" in added:
            return f"{base_col}_dn"
    except Exception:
        pass
    return base_col


def get_denoise_methods_data() -> Dict[str, Any]:
    def avail_requires(name: str) -> Tuple[bool, str]:
        if name == 'wavelet':
            return (_pywt is not None, 'PyWavelets')
        if name in ('emd', 'eemd', 'ceemdan'):
            return (any(x is not None for x in (_EMD, _EEMD, _CEEMDAN)), 'EMD-signal')
        if name in ('hp', 'whittaker'):
            return (_sps is not None and _sps_linalg is not None, 'scipy.sparse')
        if name == 'savgol':
            return (_savgol_filter is not None, 'scipy.signal')
        if name == 'butterworth':
            return (_butter is not None, 'scipy.signal')
        if name == 'gaussian':
            return (_gaussian_filter1d is not None, 'scipy.ndimage')
        if name == 'wavelet_packet':
            return (_pywt is not None, 'PyWavelets')
        if name == 'loess':
            return (_lowess is not None, 'statsmodels')
        if name == 'stl':
            return (_STL is not None, 'statsmodels')
        if name == 'vmd':
            return (_VMD is not None, 'vmdpy')
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
    add("butterworth", "Butterworth IIR low-pass/band-pass with optional zero-phase filtering.", [
        {"name": "cutoff", "type": "float|float[]", "default": 0.1, "description": "Normalized cutoff (0,0.5); list of two for band-pass."},
        {"name": "order", "type": "int", "default": 4, "description": "Filter order."},
        {"name": "btype", "type": "str", "default": "low", "description": "low|high|bandpass|bandstop."},
        {"name": "padlen", "type": "int|null", "default": None, "description": "Optional padding for filtfilt."},
    ], {"causal": True, "zero_phase": True})
    add("hp", "Hodrick-Prescott filter for trend-cycle decomposition; returns trend component.", [
        {"name": "lamb", "type": "float", "default": 1600.0, "description": "Smoothing strength (alias: lambda)."},
    ], {"causal": False, "zero_phase": True})
    add("savgol", "Savitzky-Golay smoothing that preserves local extrema better than SMA.", [
        {"name": "window", "type": "int", "default": 11, "description": "Odd window length."},
        {"name": "polyorder", "type": "int", "default": 2, "description": "Polynomial order (< window)."},
        {"name": "mode", "type": "str", "default": "interp", "description": "Edge handling mode passed to scipy.signal.savgol_filter."},
    ], {"causal": False, "zero_phase": True})
    add("tv", "Total variation denoising; preserves edges/levels while removing noise.", [
        {"name": "weight", "type": "float|\"auto\"", "default": "auto", "description": "Regularization weight (alias: lambda)."},
        {"name": "n_iter", "type": "int", "default": 50, "description": "Max iterations for TV solver."},
        {"name": "tol", "type": "float", "default": 1e-4, "description": "Convergence tolerance."},
    ], {"causal": False, "zero_phase": True})
    add("kalman", "1D Kalman filter for adaptive trend tracking on non-stationary series.", [
        {"name": "process_var", "type": "float|\"auto\"", "default": "auto", "description": "Process noise variance (alias: q)."},
        {"name": "measurement_var", "type": "float|\"auto\"", "default": "auto", "description": "Measurement noise variance (alias: r)."},
        {"name": "initial_state", "type": "float|null", "default": None, "description": "Initial state estimate."},
        {"name": "initial_cov", "type": "float|null", "default": None, "description": "Initial covariance estimate."},
    ], {"causal": True, "zero_phase": True})
    add("hampel", "Hampel filter for outlier suppression using rolling MAD.", [
        {"name": "window", "type": "int", "default": 7, "description": "Window length in samples."},
        {"name": "n_sigmas", "type": "float", "default": 3.0, "description": "Outlier threshold in MAD sigmas."},
    ], {"causal": True, "zero_phase": True})
    add("bilateral", "Bilateral filter preserving edges by combining distance + value kernels.", [
        {"name": "sigma_s", "type": "float", "default": 2.0, "description": "Spatial sigma (samples)."},
        {"name": "sigma_r", "type": "float", "default": 0.5, "description": "Range sigma (value units)."},
        {"name": "truncate", "type": "float", "default": 3.0, "description": "Kernel radius in sigmas."},
    ], {"causal": True, "zero_phase": True})
    add("wavelet_packet", "Wavelet packet denoising for finer band control than standard wavelets.", [
        {"name": "wavelet", "type": "str", "default": "db4", "description": "Wavelet family, e.g., 'db4', 'sym5'."},
        {"name": "level", "type": "int|null", "default": None, "description": "Packet level; auto if omitted."},
        {"name": "threshold", "type": "float|\"auto\"", "default": "auto", "description": "Shrinkage threshold; 'auto' uses universal threshold."},
        {"name": "mode", "type": "str", "default": "soft", "description": "Shrinkage mode: 'soft' or 'hard'."},
        {"name": "threshold_scale", "type": "float|\"auto\"", "default": "auto", "description": "Scale factor for threshold; 'auto' adapts to noise ratio."},
    ], {"causal": False, "zero_phase": True})
    add("ssa", "Singular Spectrum Analysis denoising with low-rank reconstruction.", [
        {"name": "window", "type": "int", "default": 10, "description": "SSA window length."},
        {"name": "components", "type": "int|float|null", "default": 2, "description": "Rank or energy ratio (0-1] to retain."},
    ], {"causal": False, "zero_phase": True})
    add("l1_trend", "L1 trend filtering for piecewise-linear trend extraction.", [
        {"name": "lamb", "type": "float|\"auto\"", "default": "auto", "description": "L1 penalty (alias: lambda); 'auto' scales by series std and length."},
        {"name": "n_iter", "type": "int", "default": 50, "description": "ADMM iterations."},
        {"name": "rho", "type": "float|\"auto\"", "default": "auto", "description": "ADMM rho parameter."},
    ], {"causal": False, "zero_phase": True})
    add("lms", "Adaptive LMS filter for regime-aware smoothing.", [
        {"name": "order", "type": "int", "default": 5, "description": "Filter length."},
        {"name": "mu", "type": "float|\"auto\"", "default": "auto", "description": "LMS step size; 'auto' uses normalized LMS."},
        {"name": "eps", "type": "float", "default": 1e-6, "description": "Stability epsilon for NLMS."},
        {"name": "leak", "type": "float", "default": 0.0, "description": "Leakage factor to prevent drift."},
        {"name": "bias", "type": "bool", "default": True, "description": "Include bias term for scale preservation."},
    ], {"causal": True, "zero_phase": True})
    add("rls", "Adaptive RLS filter for faster tracking under changing volatility.", [
        {"name": "order", "type": "int", "default": 5, "description": "Filter length."},
        {"name": "lambda_", "type": "float|\"auto\"", "default": "auto", "description": "Forgetting factor (alias: lam)."},
        {"name": "delta", "type": "float", "default": 1.0, "description": "Diagonal loading for initial covariance."},
        {"name": "bias", "type": "bool", "default": True, "description": "Include bias term for scale preservation."},
    ], {"causal": True, "zero_phase": True})
    add("beta", "Robust beta-IRLS smoothing using fractional-power residuals.", [
        {"name": "window", "type": "int", "default": 9, "description": "Window length in samples."},
        {"name": "beta", "type": "float", "default": 1.3, "description": "Robustness exponent (<2 downweights outliers)."},
        {"name": "n_iter", "type": "int", "default": 20, "description": "IRLS iterations per window."},
        {"name": "eps", "type": "float", "default": 1e-6, "description": "Stability epsilon for weights."},
    ], {"causal": True, "zero_phase": True})
    add("vmd", "Variational Mode Decomposition; reconstruct after dropping modes.", [
        {"name": "alpha", "type": "float", "default": 2000.0, "description": "Bandwidth constraint."},
        {"name": "tau", "type": "float", "default": 0.0, "description": "Noise tolerance (0 for pure VMD)."},
        {"name": "k", "type": "int", "default": 5, "description": "Number of modes."},
        {"name": "dc", "type": "int", "default": 0, "description": "Include DC component (0/1)."},
        {"name": "init", "type": "int", "default": 1, "description": "Initialization mode (0/1/2)."},
        {"name": "tol", "type": "float", "default": 1e-7, "description": "Convergence tolerance."},
        {"name": "keep_modes", "type": "int[]", "default": None, "description": "Explicit list of modes to keep."},
        {"name": "drop_modes", "type": "int[]", "default": None, "description": "Modes to drop (overrides keep_ratio)."},
        {"name": "keep_ratio", "type": "float|\"auto\"", "default": "auto", "description": "Energy ratio to keep using lowest-frequency modes."},
    ], {"causal": False, "zero_phase": True})
    add("loess", "LOESS/LOWESS local regression smoothing for local trend estimation.", [
        {"name": "frac", "type": "float", "default": 0.2, "description": "Fraction of data used in each local fit."},
        {"name": "it", "type": "int", "default": 0, "description": "Robustifying iterations (0 for none)."},
        {"name": "delta", "type": "float", "default": 0.0, "description": "Distance within which to use linear interpolation."},
    ], {"causal": False, "zero_phase": True})
    add("stl", "Seasonal-Trend decomposition (STL); returns selected component (default trend).", [
        {"name": "period", "type": "int", "default": None, "description": "Seasonal period (required unless index has an inferred frequency)."},
        {"name": "seasonal", "type": "int|null", "default": None, "description": "Seasonal smoothing length."},
        {"name": "trend", "type": "int|null", "default": None, "description": "Trend smoothing length."},
        {"name": "low_pass", "type": "int|null", "default": None, "description": "Low-pass filter length."},
        {"name": "robust", "type": "bool", "default": False, "description": "Enable robust fitting to outliers."},
        {"name": "component", "type": "str", "default": "trend", "description": "trend|seasonal|resid|trend+seasonal|trend+resid."},
    ], {"causal": False, "zero_phase": True})
    add("whittaker", "Whittaker smoother with penalized differences (B-spline-like).", [
        {"name": "lamb", "type": "float", "default": 1000.0, "description": "Smoothing strength (alias: lambda)."},
        {"name": "order", "type": "int", "default": 2, "description": "Difference order (1 or 2 typical)."},
    ], {"causal": False, "zero_phase": True})
    add("gaussian", "Gaussian kernel smoothing (Nadaraya-Watson style).", [
        {"name": "sigma", "type": "float", "default": 2.0, "description": "Gaussian stddev in samples."},
        {"name": "truncate", "type": "float", "default": 4.0, "description": "Kernel truncation radius in sigmas."},
        {"name": "mode", "type": "str", "default": "nearest", "description": "Edge handling mode for scipy.ndimage.gaussian_filter1d."},
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


def denoise_list_methods() -> Dict[str, Any]:
    """List available denoise methods and their parameters."""
    try:
        return get_denoise_methods_data()
    except Exception as e:
        return {"error": f"Error listing denoise methods: {e}"}


def normalize_denoise_spec(spec: Any, default_when: str = 'pre_ti') -> Optional[Dict[str, Any]]:
    """Normalize a denoise spec. Accepts dict-like or a method name string.

    Returns a dict with keys: method, params, columns, when, causality, keep_original, suffix.
    """
    base = {"when": default_when, "columns": ["close"], "keep_original": False, "suffix": "_dn"}
    if not spec:
        return None
    if isinstance(spec, dict):
        out = dict(base)
        out.update({k: v for k, v in spec.items() if v is not None})
        # Normalize columns field to list
        cols = out.get('columns')
        if isinstance(cols, str):
            parts = [p.strip() for p in cols.replace(',', ' ').split() if p.strip()]
            out['columns'] = parts if parts else ['close']
        if 'params' not in out or out['params'] is None:
            out['params'] = {}
        return out
    # String method name
    try:
        method = str(spec).strip().lower()
    except Exception:
        return None
    if method == '' or method == 'none':
        return None
    # Method-specific default params
    params: Dict[str, Any] = {}
    if method == 'ema':
        params = {"span": 10}
    elif method == 'sma':
        params = {"window": 10}
    elif method == 'median':
        params = {"window": 7}
    elif method == 'lowpass_fft':
        params = {"cutoff_ratio": 0.1}
    elif method == 'butterworth':
        params = {"cutoff": 0.1, "order": 4, "btype": "low", "padlen": None}
    elif method == 'hp':
        params = {"lamb": 1600.0}
    elif method == 'savgol':
        params = {"window": 11, "polyorder": 2, "mode": "interp"}
    elif method == 'tv':
        params = {"weight": "auto", "n_iter": 50, "tol": 1e-4}
    elif method == 'kalman':
        params = {"process_var": "auto", "measurement_var": "auto", "initial_state": None, "initial_cov": None}
    elif method == 'hampel':
        params = {"window": 7, "n_sigmas": 3.0}
    elif method == 'bilateral':
        params = {"sigma_s": 2.0, "sigma_r": 0.5, "truncate": 3.0}
    elif method == 'wavelet_packet':
        params = {"wavelet": "db4", "level": None, "threshold": "auto", "mode": "soft", "threshold_scale": "auto"}
    elif method == 'ssa':
        params = {"window": 10, "components": 2}
    elif method == 'l1_trend':
        params = {"lamb": "auto", "n_iter": 50, "rho": "auto"}
    elif method == 'lms':
        params = {"order": 5, "mu": "auto", "eps": 1e-6, "leak": 0.0, "bias": True}
    elif method == 'rls':
        params = {"order": 5, "lambda_": "auto", "delta": 1.0, "bias": True}
    elif method == 'beta':
        params = {"window": 9, "beta": 1.3, "n_iter": 20, "eps": 1e-6}
    elif method == 'vmd':
        params = {"alpha": 2000.0, "tau": 0.0, "k": 5, "dc": 0, "init": 1, "tol": 1e-7, "keep_modes": None, "drop_modes": None, "keep_ratio": "auto"}
    elif method == 'loess':
        params = {"frac": 0.2, "it": 0, "delta": 0.0}
    elif method == 'stl':
        params = {"period": None, "seasonal": None, "trend": None, "low_pass": None, "robust": False, "component": "trend"}
    elif method == 'whittaker':
        params = {"lamb": 1000.0, "order": 2}
    elif method == 'gaussian':
        params = {"sigma": 2.0, "truncate": 4.0, "mode": "nearest"}
    elif method == 'wavelet':
        params = {"wavelet": "db4", "level": None, "threshold": "auto", "mode": "soft"}
    elif method == 'emd':
        params = {"drop_imfs": [0], "keep_imfs": None, "max_imfs": "auto"}
    elif method == 'eemd':
        params = {"drop_imfs": [0], "keep_imfs": None, "max_imfs": "auto", "noise_strength": 0.2, "trials": 100}
    elif method == 'ceemdan':
        params = {"drop_imfs": [0], "keep_imfs": None, "max_imfs": "auto", "noise_strength": 0.2, "trials": 100}
    else:
        # Unknown method; ignore
        return None
    out = dict(base)
    out.update({"method": method, "params": params})
    return out
