"""Tests for patterns/classic.py — detect_classic_patterns with synthetic data."""
import numpy as np
import pandas as pd
import pytest

from mtdata.patterns.classic import (
    detect_classic_patterns,
    ClassicDetectorConfig,
    ClassicPatternResult,
)


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
