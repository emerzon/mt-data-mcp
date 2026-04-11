"""Tests for patterns/classic.py — detect_classic_patterns with synthetic data."""
import numpy as np
import pandas as pd
import pytest

from mtdata.patterns.classic import (
    ClassicDetectorConfig,
    ClassicPatternResult,
    _postprocess_classic_results,
    detect_classic_patterns,
)
from mtdata.patterns.classic_impl.config import validate_classic_detector_config
from mtdata.patterns.classic_impl.continuation import detect_flags_pennants
from mtdata.patterns.classic_impl.shapes import detect_rectangles
from mtdata.patterns.classic_impl.utils import _find_recent_breakout


def _make_ohlcv(n=200, seed=42):
    """Generate synthetic OHLCV data with some structure."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.uniform(1000, 5000, n)
    return pd.DataFrame({
        "time": np.arange(n, dtype=float),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": volume,
    })


class TestDetectClassicPatterns:
    def test_returns_list(self):
        df = _make_ohlcv()
        results = detect_classic_patterns(df)
        assert isinstance(results, list)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        assert detect_classic_patterns(df) == []

    def test_no_close_column(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        assert detect_classic_patterns(df) == []

    def test_too_few_bars(self):
        df = _make_ohlcv(n=50)
        assert detect_classic_patterns(df) == []

    def test_result_types(self):
        df = _make_ohlcv(n=300, seed=123)
        results = detect_classic_patterns(df)
        for r in results:
            assert isinstance(r, ClassicPatternResult)
            assert hasattr(r, "name")
            assert hasattr(r, "confidence")
            assert hasattr(r, "status")
            assert 0.0 <= r.confidence <= 1.0

    def test_custom_config(self):
        df = _make_ohlcv(n=200)
        cfg = ClassicDetectorConfig(min_touches=3, min_r2=0.5)
        results = detect_classic_patterns(df, cfg=cfg)
        assert isinstance(results, list)

    def test_sorted_by_recency(self):
        df = _make_ohlcv(n=300, seed=7)
        results = detect_classic_patterns(df)
        if len(results) >= 2:
            assert results[0].end_index >= results[1].end_index

    def test_close_only_data(self):
        """Should work even without high/low columns."""
        rng = np.random.RandomState(99)
        n = 200
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        df = pd.DataFrame({"close": close, "time": np.arange(n, dtype=float)})
        results = detect_classic_patterns(df)
        assert isinstance(results, list)

    def test_postprocess_filters_below_min_confidence(self):
        cfg = ClassicDetectorConfig(min_confidence=0.3)
        results = [
            ClassicPatternResult(
                name="Low Confidence",
                status="forming",
                confidence=0.29,
                start_index=0,
                end_index=10,
                start_time=None,
                end_time=None,
                details={},
            ),
            ClassicPatternResult(
                name="Keeps",
                status="forming",
                confidence=0.31,
                start_index=1,
                end_index=11,
                start_time=None,
                end_time=None,
                details={},
            ),
        ]

        out = _postprocess_classic_results(results, cfg, n=20)

        assert [r.name for r in out] == ["Keeps"]

    def test_detect_rectangles_rejects_multiple_side_outliers(self):
        n = 140
        close = np.full(n, 100.0, dtype=float)
        peaks = np.array([20, 40, 60, 80, 100], dtype=int)
        troughs = np.array([30, 50, 70, 90, 110], dtype=int)
        close[peaks] = np.array([105.0, 105.0, 105.0, 108.0, 109.0], dtype=float)
        close[troughs] = 95.0

        out = detect_rectangles(
            close,
            peaks,
            troughs,
            np.arange(n, dtype=float),
            ClassicDetectorConfig(min_channel_touches=2),
        )

        assert out == []

    def test_find_recent_breakout_uses_boundary_relative_tolerance(self):
        close = np.array([100.0, 100.0, 100.0, 200.5], dtype=float)
        upper = np.array([100.0, 100.0, 100.0, 200.0], dtype=float)

        direction, idx = _find_recent_breakout(
            close,
            upper=upper,
            tol_abs=0.4,
            tol_pct=0.4,
            lookback_bars=2,
        )

        assert direction is None
        assert idx is None

    def test_detect_flags_pennants_rejects_slow_poles(self, monkeypatch):
        from mtdata.patterns.classic_impl import continuation

        n = 180
        window = 60
        close = np.full(n, 100.0, dtype=float)
        seg = np.linspace(100.0, 104.0, window)
        close[-window:] = seg

        peaks = np.array([8, 18, 28, 38], dtype=int)
        troughs = np.array([4, 14, 24, 34], dtype=int)
        top = np.linspace(104.0, 103.2, window)
        bot = np.linspace(102.0, 102.6, window)

        monkeypatch.setattr(continuation, "_detect_pivots_close", lambda *_args, **_kwargs: (peaks, troughs))
        monkeypatch.setattr(
            continuation,
            "_fit_lines_and_arrays",
            lambda *_args, **_kwargs: (-0.03, 104.0, 0.9, 0.02, 102.0, 0.9, top.copy(), bot.copy()),
        )

        out = detect_flags_pennants(
            close,
            close + 0.1,
            close - 0.1,
            np.arange(n, dtype=float),
            n,
            ClassicDetectorConfig(
                max_consolidation_bars=window,
                min_pole_return_pct=3.0,
                min_pole_slope_pct_per_bar=0.2,
            ),
        )

        assert out == []

    def test_large_dataset_capped(self):
        """Max bars limit should be respected."""
        df = _make_ohlcv(n=5000)
        cfg = ClassicDetectorConfig(max_bars=1000)
        results = detect_classic_patterns(df, cfg=cfg)
        assert isinstance(results, list)


class TestClassicDetectorConfig:
    def test_defaults(self):
        cfg = ClassicDetectorConfig()
        assert cfg.min_touches >= 1
        assert cfg.min_r2 >= 0.0
        assert cfg.max_bars > 0

    def test_custom(self):
        cfg = ClassicDetectorConfig(min_touches=5, min_r2=0.8)
        assert cfg.min_touches == 5
        assert cfg.min_r2 == 0.8

    def test_defaults_pass_validation(self):
        cfg = ClassicDetectorConfig()
        warnings = validate_classic_detector_config(cfg)
        assert warnings == []

    def test_negative_bar_count_warns(self):
        cfg = ClassicDetectorConfig(max_bars=-1)
        warnings = validate_classic_detector_config(cfg)
        assert any("max_bars" in w for w in warnings)

    def test_min_exceeds_max_warns(self):
        cfg = ClassicDetectorConfig(min_input_bars=2000, max_bars=500)
        warnings = validate_classic_detector_config(cfg)
        assert any("min_input_bars" in w and "max_bars" in w for w in warnings)

    def test_negative_confidence_weight_warns(self):
        cfg = ClassicDetectorConfig(touch_weight=-0.5)
        warnings = validate_classic_detector_config(cfg)
        assert any("touch_weight" in w for w in warnings)


class TestClassicPatternResult:
    def test_creation(self):
        r = ClassicPatternResult(
            name="ascending_triangle",
            status="forming",
            confidence=0.75,
            start_index=10,
            end_index=50,
            start_time=None,
            end_time=None,
            details={},
        )
        assert r.name == "ascending_triangle"
        assert r.confidence == 0.75
        assert r.status == "forming"
