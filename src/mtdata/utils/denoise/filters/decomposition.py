"""Signal decomposition filters: EMD, EEMD, CEEMDAN, VMD, SSA."""
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
import math
import logging

_logger = logging.getLogger(__name__)

try:
    from PyEMD import EMD as _EMD, EEMD as _EEMD, CEEMDAN as _CEEMDAN
except Exception:
    _EMD = _EEMD = _CEEMDAN = None  # type: ignore

try:
    from vmdpy import VMD as _VMD
except Exception:
    _VMD = None  # type: ignore

from ..base import register_filter, _series_like


def _ssa_denoise(
    x: np.ndarray,
    window: int,
    components: Optional[Any],
) -> np.ndarray:
    n = len(x)
    if n < 4:
        return x
    series_variance = float(np.var(x))
    if math.isfinite(series_variance) and series_variance <= 0.0:
        return x.copy()
    L = max(2, int(window))
    if L >= n:
        return x
    K = n - L + 1
    X = np.column_stack([x[i : i + L] for i in range(K)])
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if components is None:
        r = min(2, len(s))
    elif isinstance(components, float) and 0 < components <= 1:
        total_energy = float(np.sum(s ** 2))
        if not math.isfinite(total_energy) or total_energy <= 0.0:
            return np.full(n, float(np.mean(x)), dtype=float)
        energy = np.cumsum(s ** 2) / total_energy
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


@register_filter('ssa')
def _denoise_ssa_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    window = int(params.get('window', max(10, len(x) // 3)))
    components = params.get('components')
    y = _ssa_denoise(x, window=window, components=components)
    return _series_like(s, y)


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
    y = np.asarray(modes[idx_sel].sum(axis=0), dtype=float).reshape(-1)
    if y.size != len(x):
        alt = np.asarray(modes[idx_sel].sum(axis=1), dtype=float).reshape(-1)
        if alt.size == len(x):
            y = alt
    if y.size != len(x):
        return x
    return y


@register_filter('vmd')
def _denoise_vmd_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
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
    return _series_like(s, y)


def _denoise_emd_family_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
    *,
    method: str,
) -> pd.Series:
    del causality
    if not any(component is not None for component in (_EMD, _EEMD, _CEEMDAN)):
        return s
    xnp = np.asarray(x, dtype=float)
    max_imfs = params.get('max_imfs', 'auto')
    if isinstance(max_imfs, str) and max_imfs == 'auto':
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
        if _CEEMDAN is None:
            return s
        noise_strength = float(params.get('noise_strength', 0.2))
        trials = int(params.get('trials', 100))
        random_state = params.get('random_state')
        ce = _CEEMDAN(trials=trials, noise_strength=noise_strength)
        if random_state is not None:
            ce.random_state = int(random_state)
        imfs = ce.ceemdan(xnp, max_imf=k)
    imfs = np.atleast_2d(imfs)
    resid = xnp - imfs.sum(axis=0)
    k_all = list(range(imfs.shape[0]))
    keep_imfs = params.get('keep_imfs')
    drop_imfs = params.get('drop_imfs', [0])
    if isinstance(keep_imfs, (list, tuple)) and len(keep_imfs) > 0:
        k_sel = [idx for idx in keep_imfs if 0 <= int(idx) < imfs.shape[0]]
    elif isinstance(drop_imfs, (list, tuple)) and len(drop_imfs) > 0:
        drop = {int(idx) for idx in drop_imfs}
        k_sel = [idx for idx in k_all if idx not in drop]
    else:
        k_sel = [idx for idx in k_all if idx != 0]
    y = resid + imfs[k_sel].sum(axis=0) if len(k_sel) > 0 else resid
    return _series_like(s, y)


@register_filter('emd')
def _denoise_emd_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    return _denoise_emd_family_series(s, x, params, causality, method='emd')


@register_filter('eemd')
def _denoise_eemd_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    return _denoise_emd_family_series(s, x, params, causality, method='eemd')


@register_filter('ceemdan')
def _denoise_ceemdan_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    return _denoise_emd_family_series(s, x, params, causality, method='ceemdan')
