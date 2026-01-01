from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.spatial.ckdtree import cKDTree

try:
    import hnswlib as _HNSW  # type: ignore
except Exception:
    _HNSW = None  # optional ANN backend
# Optional matrix profile backend
try:
    import stumpy as _stumpy  # type: ignore
except Exception:
    _stumpy = None

# Optional DTW/Soft-DTW backends
try:
    from tslearn.metrics import dtw as _ts_dtw  # type: ignore
except Exception:
    _ts_dtw = None
try:
    from tslearn.metrics import soft_dtw as _ts_soft_dtw  # type: ignore
except Exception:
    _ts_soft_dtw = None
try:
    from dtaidistance import dtw as _dd_dtw  # type: ignore
except Exception:
    _dd_dtw = None

# Dimensionality reduction abstraction
from .dimred import create_reducer as _create_reducer, DimReducer as _DimReducer

# Reuse existing MT5 helpers and denoise utilities
from ..core.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from .mt5 import _mt5_copy_rates_from, _rates_to_df
from .denoise import _apply_denoise as _apply_denoise_util
from .utils import to_float_np, align_finite


def _minmax_scale_row(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    rng = float(mx - mn)
    if not np.isfinite(rng) or rng <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - mn) / rng
    return y.astype(np.float32, copy=False)


def _mass_distance_profile(query: np.ndarray, series: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Compute z-normalized sliding distances between query and all subsequences in series using FFT.

    Returns an array of length len(series) - len(query) + 1. Non-finite windows are mapped to inf.
    """
    q = np.asarray(query, dtype=float).ravel()
    s = np.asarray(series, dtype=float).ravel()
    m = q.size
    n = s.size
    if m == 0 or n < m:
        return np.array([], dtype=float)

    q_std = float(np.nanstd(q))
    if not np.isfinite(q_std) or q_std <= eps:
        return np.full(max(n - m + 1, 0), np.inf, dtype=float)
    q_norm = (q - float(np.nanmean(q))) / q_std

    # Fast convolution via FFT to get dot products
    k = int(1 << (n + m - 1).bit_length())  # next power of two for speed
    q_rev = np.flip(q_norm)
    q_rev = np.pad(q_rev, (0, k - m))
    s_pad = np.pad(s, (0, k - n))
    fft_q = np.fft.fft(q_rev)
    fft_s = np.fft.fft(s_pad)
    conv = np.fft.ifft(fft_q * fft_s).real
    cross = conv[m - 1 : m - 1 + (n - m + 1)]

    # Rolling mean/std for series windows
    cumsum = np.cumsum(s, dtype=float)
    cumsum2 = np.cumsum(s * s, dtype=float)
    window_sum = cumsum[m - 1 :] - np.concatenate(([0.0], cumsum[:-m]))
    window_sum2 = cumsum2[m - 1 :] - np.concatenate(([0.0], cumsum2[:-m]))
    means = window_sum / m
    stds = np.sqrt(np.maximum(window_sum2 / m - means * means, 0.0))
    stds[stds <= eps] = np.nan

    denom = m * stds
    with np.errstate(invalid="ignore", divide="ignore"):
        dist2 = 2.0 * m * (1.0 - (cross / denom))
    dist2[~np.isfinite(dist2)] = np.inf
    dist2 = np.maximum(dist2, 0.0)
    return np.sqrt(dist2)


@dataclass
class _SeriesStore:
    symbol: str
    time_epoch: np.ndarray  # float64 UTC epoch seconds, ascending
    close: np.ndarray       # float64, ascending


class PatternIndex:
    """In-memory sliding-window index for pattern similarity search.

    - Builds per-window min-max normalized vectors of length `window_size` from
      MT5 close prices.
    - Uses cKDTree for fast L2 nearest neighbor search.
    - Keeps mappings to reconstruct symbol, dates, and original values (+future).
    """

    def __init__(
        self,
        timeframe: str,
        window_size: int,
        future_size: int,
        symbols: List[str],
        tree: Any,
        X: np.ndarray,
        start_end_idx: np.ndarray,
        labels: np.ndarray,
        series: List[_SeriesStore],
        scale: str = "minmax",
        metric: str = "euclidean",
        pca_components: Optional[int] = None,
        pca_model: Optional[object] = None,
        dimred_method: Optional[str] = None,
        dimred_params: Optional[Dict[str, Any]] = None,
        reducer: Optional[_DimReducer] = None,
        engine: str = "ckdtree",
        max_bars_per_symbol: int = 5000,
    ):
        self.timeframe = timeframe
        self.window_size = int(window_size)
        self.future_size = int(future_size)
        self.symbols = list(symbols)
        self.tree = tree
        self.X = X
        self.start_end_idx = start_end_idx  # shape (N,2)
        self.labels = labels                 # shape (N,)
        self._series = series               # list aligned with label indices
        self.scale = (scale or "minmax").lower()
        self.metric = (metric or "euclidean").lower()
        # Back-compat: keep PCA fields, while new reducer API is used going forward
        self.pca_components = int(pca_components) if pca_components else None
        self._pca = pca_model
        self.dimred_method = (dimred_method or ("pca" if self.pca_components else "none")).lower()
        self.dimred_params = dict(dimred_params or ({} if not self.pca_components else {"n_components": int(self.pca_components)}))
        self._reducer = reducer  # type: ignore
        self.engine = (engine or "ckdtree").lower()
        self.max_bars_per_symbol = int(max_bars_per_symbol)

    def search(self, anchor_values: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Query by a raw (unscaled) anchor window. Returns (indices, distances)."""
        v = np.asarray(anchor_values, dtype=float).ravel()
        if v.size != self.window_size:
            raise ValueError(f"anchor_values must be length {self.window_size}")
        if self.engine in ("matrix_profile", "mass"):
            return self._profile_search(v, top_k=top_k)
        q = v.astype(float)
        # Scale
        q = _apply_scale_vector(q, self.scale)
        # Dimensionality reduction (new API), falling back to PCA model if present
        if self._reducer is not None:
            if not self._reducer.supports_transform():
                raise RuntimeError(f"Reducer '{self.dimred_method}' does not support transforming new samples")
            q = np.asarray(self._reducer.transform(q.reshape(1, -1)), dtype=np.float32).ravel()
        elif self._pca is not None:
            q = np.asarray(self._pca.transform(q.reshape(1, -1))[0], dtype=np.float32)
        # Metric post-process
        q = _apply_metric_vector(q, self.metric)
        k = min(int(top_k), len(self.X))
        if self.engine == "hnsw":
            # hnswlib with 'l2' space returns squared L2 distances; take sqrt to match cKDTree
            labels, distances = self.tree.knn_query(q.reshape(1, -1).astype(np.float32), k=k)
            idxs = labels[0].astype(int)
            dists = np.sqrt(distances[0].astype(float))
            return idxs, dists
        else:
            dists, idxs = self.tree.query(q, k=k)
            # Ensure 1D arrays
            if np.ndim(idxs) == 0:
                idxs = np.asarray([int(idxs)])
                dists = np.asarray([float(dists)])
            return idxs.astype(int), dists.astype(float)

    def _profile_search(self, anchor_values: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding search using matrix profile / MASS style distances."""
        if self.scale not in ("zscore",):
            raise ValueError("matrix_profile/mass engines require scale='zscore'")
        if self.metric not in ("euclidean", "l2"):
            raise ValueError("matrix_profile/mass engines require metric='euclidean'")

        q = np.asarray(anchor_values, dtype=float).ravel()
        m = q.size
        idxs_all: List[int] = []
        dists_all: List[float] = []
        offset = 0

        for ser in self._series:
            n = ser.close.size
            limit = n - (self.window_size + self.future_size) + 1
            if limit <= 0:
                continue
            series_slice = ser.close[: limit + self.window_size + max(self.future_size - 1, 0)]
            if series_slice.size < m:
                offset += max(limit, 0)
                continue

            if self.engine == "matrix_profile":
                if _stumpy is None:
                    raise RuntimeError("matrix_profile engine requested but 'stumpy' is not installed")
                # AB-join: distances from every subsequence in series_slice to the query subsequence
                mp = _stumpy.stump(series_slice.astype(float), m, T_B=q.astype(float), ignore_trivial=False)
                profile = np.asarray(mp[:, 0], dtype=float)
            else:
                profile = _mass_distance_profile(q, series_slice)

            if profile.size > limit:
                profile = profile[:limit]
            for i, d in enumerate(profile.tolist()):
                idxs_all.append(offset + i)
                dists_all.append(float(d))
            offset += max(limit, 0)

        if not idxs_all:
            return np.array([], dtype=int), np.array([], dtype=float)

        order = np.argsort(np.asarray(dists_all, dtype=float))
        k = min(int(top_k), order.size)
        sel = order[:k]
        return np.asarray([idxs_all[i] for i in sel], dtype=int), np.asarray([dists_all[i] for i in sel], dtype=float)

    def get_match_symbol(self, index: int) -> str:
        lbl = int(self.labels[int(index)])
        return self._series[lbl].symbol

    def get_match_times(self, index: int, include_future: bool = True) -> np.ndarray:
        s, e = self.start_end_idx[int(index)]
        ser = self._series[int(self.labels[int(index)])]
        if include_future and self.future_size > 0:
            end = min(int(e + self.future_size), len(ser.time_epoch) - 1)
            start = max(0, int(s))
        else:
            start, end = int(s), int(e)
        return ser.time_epoch[start : end + 1]

    def get_match_values(self, index: int, include_future: bool = True) -> np.ndarray:
        s, e = self.start_end_idx[int(index)]
        ser = self._series[int(self.labels[int(index)])]
        if include_future and self.future_size > 0:
            end = min(int(e + self.future_size), len(ser.close) - 1)
            start = max(0, int(s))
        else:
            start, end = int(s), int(e)
        return ser.close[start : end + 1]

    def _scaled_window(self, vals: np.ndarray) -> np.ndarray:
        # Apply the same scaling as index vectors for fair comparison
        return _apply_scale_vector(np.asarray(vals, dtype=float), self.scale)

    def _ncc_max(self, a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
        """Compute maximum normalized cross-correlation within +/- max_lag.
        a and b are same-length 1D arrays (window only).
        """
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n <= 2:
            return 0.0
        # Z-normalize for correlation
        def znorm(x: np.ndarray) -> np.ndarray:
            xm = float(np.nanmean(x))
            xs = float(np.nanstd(x))
            if not np.isfinite(xs) or xs <= 1e-12:
                return np.zeros_like(x, dtype=float)
            return (x - xm) / xs
        a = znorm(a)
        b = znorm(b)
        L = int(max(0, max_lag))
        best = -1.0
        for lag in range(-L, L + 1):
            if lag == 0:
                aa, bb = a, b
            elif lag > 0:
                aa = a[lag:]
                bb = b[: n - lag]
            else:  # lag < 0
                aa = a[: n + lag]
                bb = b[-lag:]
            m = int(min(aa.size, bb.size))
            if m <= 2:
                continue
            # Cosine similarity == correlation since both z-normalized
            num = float(np.dot(aa, bb))
            den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            if not np.isfinite(den) or den <= 1e-12:
                corr = 0.0
            else:
                corr = num / den
            if corr > best:
                best = corr
        if not np.isfinite(best):
            best = 0.0
        return float(max(min(best, 1.0), -1.0))

    def refine_matches(
        self,
        anchor_values: np.ndarray,
        idxs: np.ndarray,
        dists: np.ndarray,
        top_k: int,
        shape_metric: Optional[str] = None,
        allow_lag: int = 0,
        dtw_band_frac: Optional[float] = None,
        soft_dtw_gamma: Optional[float] = None,
        affine_alpha_min: float = 0.5,
        affine_alpha_max: float = 2.0,
        affine_penalty: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Re-rank candidates using a shape metric (e.g., NCC with lag) and return top_k.

        shape_metric:
          - 'ncc': normalized cross-correlation with +/- allow_lag bar shifts
          - None/'none': no re-ranking
        """
        if shape_metric is None:
            shape_metric = 'none'
        sm = str(shape_metric).lower().strip()
        if sm in ("", "none"):
            # No refinement; just truncate
            k = min(int(top_k), idxs.size)
            return idxs[:k], dists[:k]

        # Prepare scaled anchor window (window part only)
        a = self._scaled_window(np.asarray(anchor_values, dtype=float))
        scores: List[Tuple[float, int]] = []
        max_lag = int(allow_lag) if allow_lag and int(allow_lag) > 0 else 0
        for idx, _d in zip(idxs.tolist(), dists.tolist()):
            w = self.get_match_values(int(idx), include_future=False)
            w = self._scaled_window(w)
            if sm == 'ncc':
                corr = self._ncc_max(a, w, max_lag)
                # Convert to distance-like score: lower is better
                score = 1.0 - float(corr)
            elif sm == 'affine':
                # Fit alpha, beta minimizing ||a - (alpha*w + beta)||_2
                aw = float(np.dot(a - np.mean(a), w - np.mean(w)))
                ww = float(np.dot(w - np.mean(w), w - np.mean(w)))
                alpha = (aw / ww) if (np.isfinite(ww) and ww > 1e-12) else 0.0
                # Constrain alpha
                alpha = max(float(affine_alpha_min), min(float(affine_alpha_max), float(alpha)))
                beta = float(np.mean(a) - alpha * np.mean(w))
                resid = a - (alpha * w + beta)
                rmse = float(np.sqrt(np.mean(resid * resid)))
                # Optional penalty to discourage extreme scaling
                score = rmse + float(affine_penalty) * abs(float(alpha) - 1.0)
            elif sm in ('dtw', 'softdtw'):
                # Compute DTW/Soft-DTW distance; fallback to simple DP if libs unavailable
                n = a.size
                band = None
                if dtw_band_frac is not None and dtw_band_frac > 0:
                    band = max(1, int(round(float(dtw_band_frac) * n)))
                if sm == 'dtw':
                    dist = None
                    if _ts_dtw is not None:
                        try:
                            if band:
                                dist = float(_ts_dtw(a, w, global_constraint="sakoe_chiba", sakoe_chiba_radius=int(band)))
                            else:
                                dist = float(_ts_dtw(a, w))
                        except Exception:
                            dist = None
                    if dist is None and _dd_dtw is not None:
                        try:
                            if band:
                                dist = float(_dd_dtw.distance_fast(a, w, window=band))
                            else:
                                dist = float(_dd_dtw.distance_fast(a, w))
                        except Exception:
                            dist = None
                    if dist is None:
                        # Simple O(n^2) DTW DP fallback (no band)
                        ca = np.full((n + 1, n + 1), np.inf, dtype=float)
                        ca[0, 0] = 0.0
                        for i in range(1, n + 1):
                            for j in range(1, n + 1):
                                cost = abs(a[i - 1] - w[j - 1])
                                ca[i, j] = cost + min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1])
                        dist = float(ca[n, n])
                    score = dist
                else:  # softdtw
                    dist = None
                    if _ts_soft_dtw is not None:
                        try:
                            gamma = float(soft_dtw_gamma) if (soft_dtw_gamma is not None and soft_dtw_gamma > 0) else 1.0
                            dist = float(_ts_soft_dtw(a.reshape(1, -1), w.reshape(1, -1), gamma=gamma))
                        except Exception:
                            dist = None
                    if dist is None:
                        # Fallback to DTW if soft-DTW unavailable
                        if _ts_dtw is not None:
                            try:
                                if band:
                                    dist = float(_ts_dtw(a, w, global_constraint="sakoe_chiba", sakoe_chiba_radius=int(band)))
                                else:
                                    dist = float(_ts_dtw(a, w))
                            except Exception:
                                dist = None
                        if dist is None and _dd_dtw is not None:
                            try:
                                if band:
                                    dist = float(_dd_dtw.distance_fast(a, w, window=band))
                                else:
                                    dist = float(_dd_dtw.distance_fast(a, w))
                            except Exception:
                                dist = None
                        if dist is None:
                            # Final fallback to simple DTW
                            ca = np.full((n + 1, n + 1), np.inf, dtype=float)
                            ca[0, 0] = 0.0
                            for i in range(1, n + 1):
                                for j in range(1, n + 1):
                                    cost = abs(a[i - 1] - w[j - 1])
                                    ca[i, j] = cost + min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1])
                            dist = float(ca[n, n])
                    score = dist
            else:
                # Fallback to euclidean on scaled windows
                diff = a - w
                score = float(np.sqrt(np.dot(diff, diff)))
            scores.append((score, int(idx)))
        scores.sort(key=lambda x: x[0])
        take = min(int(top_k), len(scores))
        new_idxs = np.array([i for _, i in scores[:take]], dtype=int)
        new_scores = np.array([s for s, _ in scores[:take]], dtype=float)
        return new_idxs, new_scores

    def bars_per_symbol(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ser in self._series:
            out[ser.symbol] = int(len(ser.close))
        return out

    def windows_per_symbol(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ser in self._series:
            n = int(len(ser.close))
            w = max(0, n - (self.window_size + self.future_size) + 1)
            out[ser.symbol] = w
        return out

    def get_symbol_series(self, symbol: str) -> Optional[np.ndarray]:
        for ser in self._series:
            if ser.symbol == symbol:
                return ser.close
        return None

    def get_symbol_returns(self, symbol: str, lookback: int = 1000) -> Optional[np.ndarray]:
        arr = self.get_symbol_series(symbol)
        if arr is None or len(arr) < 3:
            return None
        # Simple log returns to stabilize scale
        x = np.asarray(arr, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = np.diff(np.log(x))
        ret = ret[np.isfinite(ret)]
        if ret.size <= 0:
            return None
        if lookback and lookback > 0 and ret.size > lookback:
            ret = ret[-int(lookback):]
        return ret.astype(np.float32, copy=False)


def _fetch_symbol_df(
    symbol: str,
    timeframe: str,
    bars: int,
    *,
    as_of: Optional[Any] = None,
    drop_last_live: bool = True,
) -> pd.DataFrame:
    """Fetch last `bars` candles for symbol/timeframe.

    Returns DataFrame with columns at least ['time','open','high','low','close','tick_volume','real_volume']
    where available. 'time' is UTC epoch seconds as float.
    """
    tf = TIMEFRAME_MAP.get(timeframe)
    if tf is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    # Use a small guard for extra bars; server fetch typically asks +2
    if as_of is not None:
        try:
            to_dt = pd.to_datetime(as_of)
            if to_dt.tzinfo is None:
                to_dt = to_dt.tz_localize("UTC")
            to_dt = to_dt.to_pydatetime()
        except Exception:
            to_dt = pd.Timestamp.utcnow().to_pydatetime()
    else:
        to_dt = pd.Timestamp.utcnow().to_pydatetime()
    rates = _mt5_copy_rates_from(symbol, tf, to_dt, int(bars) + 2)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch rates for {symbol}")
    df = _rates_to_df(rates)
    if drop_last_live and as_of is None and len(df) >= 2:
        df = df.iloc[:-1]
    # Keep last `bars` rows
    if len(df) > bars:
        df = df.iloc[-bars:].copy()
    return df


def _prepare_series(
    symbol: str,
    timeframe: str,
    max_bars: int,
    denoise: Optional[Dict[str, Any]] = None,
    *,
    as_of: Optional[Any] = None,
    drop_last_live: bool = True,
) -> Optional[_SeriesStore]:
    """Fetch and optionally denoise a symbol series; return _SeriesStore with ascending times."""
    df = _fetch_symbol_df(symbol, timeframe, max_bars, as_of=as_of, drop_last_live=drop_last_live)
    # Ensure 'volume' exists for denoise convenience
    if 'volume' not in df.columns and 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    # Apply optional denoise to 'close' in-place
    if denoise and isinstance(denoise, dict):
        try:
            dn = dict(denoise)
            # Default pre-window denoise, apply to 'close' only unless caller overrode
            dn.setdefault('when', 'pre_ti')
            dn.setdefault('columns', ['close'])
            dn.setdefault('keep_original', False)
            _apply_denoise_util(df, dn, default_when='pre_ti')
        except Exception:
            # Fallback to raw if denoise fails
            pass
    # Extract arrays (ascending time assumed)
    try:
        # Convert and align by finite mask across both arrays
        t, c = align_finite(df['time'], df['close'])
        if t.size < 10:
            return None
        return _SeriesStore(symbol=symbol, time_epoch=t, close=c)
    except Exception:
        return None


def build_index(
    symbols: List[str],
    timeframe: str,
    window_size: int,
    future_size: int,
    max_bars_per_symbol: int = 5000,
    denoise: Optional[Dict[str, Any]] = None,
    scale: str = "minmax",
    metric: str = "euclidean",
    pca_components: Optional[int] = None,
    # New flexible dimension reduction interface
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    engine: str = "ckdtree",
    *,
    as_of: Optional[Any] = None,
    drop_last_live: bool = True,
) -> PatternIndex:
    """Build a PatternIndex from MT5 data for the provided symbols.

    Notes:
    - Windows are created over ascending time arrays. Window i corresponds to
      close[i : i+window_size]. Matches can expose window+future values.
    - Per-window min-max normalization is applied for the index vectors.
    """
    assert window_size >= 5, "window_size too small"
    symbols_ok: List[str] = []
    series: List[_SeriesStore] = []
    for sym in symbols:
        ser = _prepare_series(
            sym,
            timeframe,
            max_bars=max_bars_per_symbol,
            denoise=denoise,
            as_of=as_of,
            drop_last_live=drop_last_live,
        )
        if ser is None:
            continue
        # Require enough bars for at least one window
        if ser.close.size >= (window_size + future_size):
            series.append(ser)
            symbols_ok.append(sym)
    if not series:
        raise RuntimeError("No symbols had sufficient data to build pattern index")

    # Build windows
    X_list: List[np.ndarray] = []
    start_end: List[Tuple[int, int]] = []
    labels: List[int] = []
    for lbl, ser in enumerate(series):
        n = ser.close.size
        limit = n - (window_size + future_size) + 1
        if limit <= 0:
            continue
        # Create sliding indices
        starts = np.arange(limit, dtype=int)
        ends = starts + (window_size - 1)
        # Gather windows using stride-based indexing
        idx = starts[:, None] + np.arange(window_size)[None, :]
        w = ser.close[idx]
        # Apply per-row scaling
        sc = (scale or "minmax").lower()
        if sc == "zscore":
            mu = np.nanmean(w, axis=1, keepdims=True)
            sd = np.nanstd(w, axis=1, keepdims=True)
            sd[sd <= 1e-12] = 1.0
            X_scaled = ((w - mu) / sd).astype(np.float32)
        elif sc == "none":
            X_scaled = w.astype(np.float32)
        else:  # minmax
            mn = np.nanmin(w, axis=1, keepdims=True)
            mx = np.nanmax(w, axis=1, keepdims=True)
            rng = (mx - mn)
            rng[rng <= 1e-12] = 1.0
            X_scaled = ((w - mn) / rng).astype(np.float32)
        X_list.append(X_scaled)
        start_end.extend(list(np.stack([starts, ends], axis=1)))
        labels.extend([lbl] * starts.size)

    if not X_list:
        raise RuntimeError("Failed to create any windows for the provided symbols")
    X = np.vstack(X_list)
    # Optional dimensionality reduction
    pca_model = None
    reducer: Optional[_DimReducer] = None
    # Back-compat: if pca_components provided, prefer PCA
    effective_dimred_method = (dimred_method or ("pca" if (pca_components and int(pca_components) > 0) else "none"))
    effective_dimred_params: Dict[str, Any] = dict(dimred_params or {})
    if (pca_components and int(pca_components) > 0) and (not dimred_method or str(dimred_method).lower() in ("", "none", "pca")):
        # Ensure components bound to window size
        effective_dimred_params.setdefault("n_components", max(1, min(int(pca_components), int(X.shape[1]))))
        effective_dimred_method = "pca"
    if effective_dimred_method and str(effective_dimred_method).lower() not in ("none", "false"):
        reducer, info = _create_reducer(effective_dimred_method, effective_dimred_params)
        # If reducer requires n_components, ensure it does not exceed window length
        try:
            if hasattr(reducer, "n_components"):
                nc = int(getattr(reducer, "n_components"))
                if nc > int(X.shape[1]):
                    # Recreate reducer with clipped components
                    effective_dimred_params["n_components"] = int(X.shape[1])
                    reducer, info = _create_reducer(effective_dimred_method, effective_dimred_params)
        except Exception:
            pass
        X = reducer.fit_transform(X)
        X = X.astype(np.float32, copy=False)

    # Metric transform (for cosine/correlation)
    met = (metric or "euclidean").lower()
    if met == "cosine":
        # L2-normalize rows
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        X = (X / norms).astype(np.float32)
    elif met == "correlation":
        # If not PCA-centered already, re-center rows; then L2 normalize
        if pca_model is None:
            X = X - np.nanmean(X, axis=1, keepdims=True)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        X = (X / norms).astype(np.float32)
    start_end_idx = np.asarray(start_end, dtype=int)
    labels_arr = np.asarray(labels, dtype=int)

    eng = (engine or "ckdtree").lower()
    if eng in ("matrix_profile", "mass"):
        tree_obj = None  # search will bypass tree
    elif eng == "hnsw":
        if _HNSW is None:
            raise RuntimeError("hnswlib not available; install hnswlib or use engine='ckdtree'")
        dim = int(X.shape[1])
        index = _HNSW.Index(space='l2', dim=dim)
        # Defaults tuned for good recall/speed tradeoff; can be parameterized later
        index.init_index(max_elements=int(X.shape[0]), ef_construction=200, M=16)
        index.add_items(X.astype(np.float32), np.arange(X.shape[0], dtype=np.int32))
        index.set_ef(64)  # ef_search
        tree_obj = index
    elif eng == "ckdtree":
        tree_obj = cKDTree(X)
    else:
        raise ValueError(f"Unknown engine '{eng}'")

    return PatternIndex(
        timeframe=timeframe,
        window_size=int(window_size),
        future_size=int(future_size),
        symbols=symbols_ok,
        tree=tree_obj,
        X=X,
        start_end_idx=start_end_idx,
        labels=labels_arr,
        series=series,
        scale=(scale or "minmax").lower(),
        metric=(metric or "euclidean").lower(),
        pca_components=int(pca_components) if pca_components else None,
        pca_model=pca_model,
        dimred_method=str(effective_dimred_method or 'none'),
        dimred_params=effective_dimred_params,
        reducer=reducer,
        engine=eng,
        max_bars_per_symbol=int(max_bars_per_symbol),
    )


def _apply_scale_vector(x: np.ndarray, scale: str) -> np.ndarray:
    s = (scale or "minmax").lower()
    x = np.asarray(x, dtype=float)
    if s == "zscore":
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd <= 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - mu) / sd).astype(np.float32)
    if s == "none":
        return x.astype(np.float32)
    # minmax
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    rng = mx - mn
    if not np.isfinite(rng) or rng <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / rng).astype(np.float32)


def _apply_metric_vector(x: np.ndarray, metric: str) -> np.ndarray:
    m = (metric or "euclidean").lower()
    v = np.asarray(x, dtype=np.float32)
    if m == "cosine":
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n <= 1e-12:
            return np.zeros_like(v, dtype=np.float32)
        return (v / n).astype(np.float32)
    if m == "correlation":
        v = v - float(np.nanmean(v))
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n <= 1e-12:
            return np.zeros_like(v, dtype=np.float32)
        return (v / n).astype(np.float32)
    return v
