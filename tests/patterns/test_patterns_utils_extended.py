"""Extended tests for mtdata.utils.patterns covering uncovered lines.

Targets: _minmax_scale_row, _mass_distance_profile, _apply_scale_vector,
         _apply_metric_vector, PatternIndex methods (search, _profile_search,
         get_match_*, _ncc_max, refine_matches, bars/windows_per_symbol,
         get_symbol_series, get_symbol_returns, _scaled_window).
         All pure logic – no MT5 calls.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from mtdata.utils.patterns import (
    _minmax_scale_row,
    _mass_distance_profile,
    _apply_scale_vector,
    _apply_metric_vector,
    _SeriesStore,
    PatternIndex,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_series(n: int = 200, seed: int = 0, start: float = 100.0) -> _SeriesStore:
    rng = np.random.RandomState(seed)
    close = start + np.cumsum(rng.randn(n) * 0.5)
    t = np.arange(n, dtype=float) * 3600.0
    return _SeriesStore(symbol="TEST", time_epoch=t, close=close)


def _make_index(
    window_size: int = 10,
    future_size: int = 5,
    n_bars: int = 100,
    scale: str = "minmax",
    metric: str = "euclidean",
    engine: str = "ckdtree",
) -> PatternIndex:
    """Build a minimal PatternIndex from synthetic data."""
    ser = _make_series(n_bars, seed=42)
    close = ser.close
    n = close.size
    limit = n - (window_size + future_size) + 1
    starts = np.arange(limit, dtype=int)
    ends = starts + (window_size - 1)
    idx_arr = starts[:, None] + np.arange(window_size)[None, :]
    w = close[idx_arr]
    # scale
    if scale == "minmax":
        mn = np.nanmin(w, axis=1, keepdims=True)
        mx = np.nanmax(w, axis=1, keepdims=True)
        rng = mx - mn
        rng[rng <= 1e-12] = 1.0
        X = ((w - mn) / rng).astype(np.float32)
    elif scale == "zscore":
        mu = np.nanmean(w, axis=1, keepdims=True)
        sd = np.nanstd(w, axis=1, keepdims=True)
        sd[sd <= 1e-12] = 1.0
        X = ((w - mu) / sd).astype(np.float32)
    else:
        X = w.astype(np.float32)

    start_end = np.stack([starts, ends], axis=1)
    labels = np.zeros(limit, dtype=int)

    tree = cKDTree(X) if engine == "ckdtree" else None

    return PatternIndex(
        timeframe="H1",
        window_size=window_size,
        future_size=future_size,
        symbols=["TEST"],
        tree=tree,
        X=X,
        start_end_idx=start_end,
        labels=labels,
        series=[ser],
        scale=scale,
        metric=metric,
        engine=engine,
    )


# ===================================================================
# _minmax_scale_row
# ===================================================================
class TestMinmaxScaleRow:
    def test_basic(self):
        r = _minmax_scale_row(np.array([1, 2, 3, 4, 5]))
        assert r.min() == pytest.approx(0.0)
        assert r.max() == pytest.approx(1.0)

    def test_empty(self):
        r = _minmax_scale_row(np.array([]))
        assert r.size == 0

    def test_constant(self):
        r = _minmax_scale_row(np.array([5, 5, 5]))
        assert np.all(r == 0)

    def test_single_element(self):
        r = _minmax_scale_row(np.array([42]))
        assert r[0] == 0.0

    def test_negative_values(self):
        r = _minmax_scale_row(np.array([-10, -5, 0, 5, 10]))
        assert r[0] == pytest.approx(0.0)
        assert r[-1] == pytest.approx(1.0)

    def test_nan_handling(self):
        r = _minmax_scale_row(np.array([1, np.nan, 3]))
        assert r.dtype == np.float32


# ===================================================================
# _mass_distance_profile
# ===================================================================
class TestMassDistanceProfile:
    def test_basic(self):
        q = np.sin(np.linspace(0, np.pi, 10))
        s = np.sin(np.linspace(0, 4 * np.pi, 100))
        p = _mass_distance_profile(q, s)
        assert p.size == 100 - 10 + 1
        assert np.all(np.isfinite(p) | (p == np.inf))

    def test_empty_query(self):
        p = _mass_distance_profile(np.array([]), np.arange(10))
        assert p.size == 0

    def test_query_longer_than_series(self):
        p = _mass_distance_profile(np.arange(20), np.arange(10))
        assert p.size == 0

    def test_non_finite_query(self):
        q = np.array([1, np.nan, 3, 4, 5])
        s = np.arange(20, dtype=float)
        p = _mass_distance_profile(q, s)
        assert np.all(p == np.inf)

    def test_constant_query(self):
        q = np.ones(5)
        s = np.arange(20, dtype=float)
        p = _mass_distance_profile(q, s)
        assert np.all(p == np.inf)  # zero std


# ===================================================================
# _apply_scale_vector
# ===================================================================
class TestApplyScaleVector:
    def test_minmax(self):
        v = _apply_scale_vector(np.array([1, 2, 3, 4, 5]), "minmax")
        assert v[0] == pytest.approx(0.0)
        assert v[-1] == pytest.approx(1.0)

    def test_zscore(self):
        v = _apply_scale_vector(np.array([1, 2, 3, 4, 5]), "zscore")
        assert abs(float(np.mean(v))) < 0.01

    def test_none_scale(self):
        x = np.array([1.0, 2.0, 3.0])
        v = _apply_scale_vector(x, "none")
        np.testing.assert_allclose(v, x, atol=1e-6)

    def test_constant_minmax(self):
        v = _apply_scale_vector(np.array([5, 5, 5]), "minmax")
        assert np.all(v == 0)

    def test_constant_zscore(self):
        v = _apply_scale_vector(np.array([5, 5, 5]), "zscore")
        assert np.all(v == 0)


# ===================================================================
# _apply_metric_vector
# ===================================================================
class TestApplyMetricVector:
    def test_euclidean(self):
        x = np.array([1, 2, 3], dtype=np.float32)
        v = _apply_metric_vector(x, "euclidean")
        np.testing.assert_allclose(v, x, atol=1e-6)

    def test_cosine(self):
        x = np.array([3, 4], dtype=np.float32)
        v = _apply_metric_vector(x, "cosine")
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)

    def test_correlation(self):
        x = np.array([1, 2, 3, 4], dtype=np.float32)
        v = _apply_metric_vector(x, "correlation")
        # Should be centered then L2-normalized
        assert abs(float(np.mean(v))) < 0.01
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_zero_vector(self):
        v = _apply_metric_vector(np.zeros(5, dtype=np.float32), "cosine")
        assert np.all(v == 0)

    def test_correlation_constant(self):
        v = _apply_metric_vector(np.full(5, 3.0, dtype=np.float32), "correlation")
        assert np.all(v == 0)


# ===================================================================
# PatternIndex.search (ckdtree)
# ===================================================================
class TestPatternIndexSearch:
    def test_basic_search(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=3)
        assert len(idxs) == 3
        assert dists[0] <= dists[1] <= dists[2]

    def test_top_k_exceeds_windows(self):
        pi = _make_index(n_bars=20)
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=1000)
        assert len(idxs) <= pi.X.shape[0]

    def test_wrong_size_raises(self):
        pi = _make_index(window_size=10)
        with pytest.raises(ValueError):
            pi.search(np.array([1, 2, 3]), top_k=1)


# ===================================================================
# PatternIndex accessor methods
# ===================================================================
class TestPatternIndexAccessors:
    def test_get_match_symbol(self):
        pi = _make_index()
        sym = pi.get_match_symbol(0)
        assert sym == "TEST"

    def test_get_match_times(self):
        pi = _make_index()
        times = pi.get_match_times(0, include_future=True)
        assert len(times) > 0

    def test_get_match_times_no_future(self):
        pi = _make_index()
        times = pi.get_match_times(0, include_future=False)
        assert len(times) == pi.window_size

    def test_get_match_values(self):
        pi = _make_index()
        vals = pi.get_match_values(0, include_future=True)
        assert len(vals) > pi.window_size

    def test_get_match_values_no_future(self):
        pi = _make_index()
        vals = pi.get_match_values(0, include_future=False)
        assert len(vals) == pi.window_size

    def test_bars_per_symbol(self):
        pi = _make_index()
        bps = pi.bars_per_symbol()
        assert bps["TEST"] == 100

    def test_windows_per_symbol(self):
        pi = _make_index()
        wps = pi.windows_per_symbol()
        assert wps["TEST"] > 0

    def test_get_symbol_series_found(self):
        pi = _make_index()
        arr = pi.get_symbol_series("TEST")
        assert arr is not None
        assert len(arr) == 100

    def test_get_symbol_series_missing(self):
        pi = _make_index()
        assert pi.get_symbol_series("NOEXIST") is None


# ===================================================================
# PatternIndex.get_symbol_returns
# ===================================================================
class TestGetSymbolReturns:
    def test_basic_returns(self):
        pi = _make_index()
        r = pi.get_symbol_returns("TEST")
        assert r is not None
        assert r.size > 0

    def test_returns_with_lookback(self):
        pi = _make_index(n_bars=200)
        r = pi.get_symbol_returns("TEST", lookback=50)
        assert r is not None
        assert r.size <= 50

    def test_returns_missing_symbol(self):
        pi = _make_index()
        assert pi.get_symbol_returns("MISSING") is None

    def test_returns_short_series(self):
        ser = _SeriesStore(symbol="X", time_epoch=np.array([0.0, 1.0]), close=np.array([100.0, 101.0]))
        pi = PatternIndex(
            timeframe="H1", window_size=2, future_size=0, symbols=["X"],
            tree=None, X=np.zeros((1, 2)), start_end_idx=np.array([[0, 1]]),
            labels=np.array([0]), series=[ser],
        )
        assert pi.get_symbol_returns("X") is None  # size < 3


# ===================================================================
# PatternIndex._ncc_max
# ===================================================================
class TestNccMax:
    def test_identical_signals(self):
        pi = _make_index()
        a = np.sin(np.linspace(0, 2 * np.pi, 20))
        corr = pi._ncc_max(a, a, max_lag=0)
        assert corr == pytest.approx(1.0, abs=0.01)

    def test_shifted_signals(self):
        pi = _make_index()
        a = np.zeros(20)
        a[5:15] = 1.0
        b = np.zeros(20)
        b[7:17] = 1.0
        corr = pi._ncc_max(a, b, max_lag=5)
        assert corr > 0.5

    def test_short_signal(self):
        pi = _make_index()
        corr = pi._ncc_max(np.array([1.0, 2.0]), np.array([1.0, 2.0]), max_lag=0)
        assert corr == 0.0  # n <= 2

    def test_constant_signal(self):
        pi = _make_index()
        a = np.ones(20)
        corr = pi._ncc_max(a, a, max_lag=0)
        assert corr == 0.0  # zero std


# ===================================================================
# PatternIndex.refine_matches
# ===================================================================
class TestRefineMatches:
    def test_ncc_refinement(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=10)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=5, shape_metric="ncc", allow_lag=2
        )
        assert len(new_idxs) == 5
        assert all(s >= 0 for s in new_scores)

    def test_none_metric(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=3, shape_metric=None
        )
        assert len(new_idxs) == 3

    def test_unknown_metric_fallback(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=3, shape_metric="unknown_metric"
        )
        assert len(new_idxs) == 3

    def test_affine_metric(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=3, shape_metric="affine"
        )
        assert len(new_idxs) == 3

    def test_dtw_metric(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=3, shape_metric="dtw"
        )
        assert len(new_idxs) == 3

    def test_softdtw_metric(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=3, shape_metric="softdtw", soft_dtw_gamma=1.0
        )
        assert len(new_idxs) == 3

    def test_dtw_with_band(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=3, shape_metric="dtw", dtw_band_frac=0.2
        )
        assert len(new_idxs) == 3


# ===================================================================
# PatternIndex._scaled_window
# ===================================================================
class TestScaledWindow:
    def test_minmax(self):
        pi = _make_index(scale="minmax")
        v = pi._scaled_window(np.array([1, 2, 3, 4, 5]))
        assert v[0] == pytest.approx(0.0)
        assert v[-1] == pytest.approx(1.0)

    def test_zscore(self):
        pi = _make_index(scale="zscore")
        v = pi._scaled_window(np.array([1, 2, 3, 4, 5]))
        assert abs(float(np.mean(v))) < 0.01
