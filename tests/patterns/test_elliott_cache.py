"""Tests for ElliottWaveAnalyzer request-scoped pivot and feature caches."""
import numpy as np
import pytest

from mtdata.patterns.elliott import ElliottWaveAnalyzer, ElliottWaveConfig


def _zigzag_close(n: int = 100) -> np.ndarray:
    """Generate a zigzag close series with enough pivots."""
    base = np.linspace(100.0, 120.0, n)
    wave = 5.0 * np.sin(np.linspace(0, 6 * np.pi, n))
    return base + wave


class TestAnalyzerPivotCache:
    def test_pivot_cache_hit(self):
        close = _zigzag_close()
        times = np.arange(close.size, dtype=float)
        analyzer = ElliottWaveAnalyzer(close, times, ElliottWaveConfig())

        p1 = analyzer._get_pivots(0.5, 3)
        p2 = analyzer._get_pivots(0.5, 3)
        assert p1 is p2  # same object = cache hit

    def test_pivot_cache_different_keys(self):
        close = _zigzag_close()
        times = np.arange(close.size, dtype=float)
        analyzer = ElliottWaveAnalyzer(close, times, ElliottWaveConfig())

        p1 = analyzer._get_pivots(0.5, 3)
        p2 = analyzer._get_pivots(1.0, 3)
        assert p1 is not p2

    def test_pivot_signature_uses_cache(self):
        close = _zigzag_close()
        times = np.arange(close.size, dtype=float)
        analyzer = ElliottWaveAnalyzer(close, times, ElliottWaveConfig())

        sig = analyzer.pivot_signature(0.5, 3)
        assert isinstance(sig, tuple)
        assert all(isinstance(i, int) for i in sig)
        # Calling again should use cached pivots
        assert (0.5, 3) in analyzer._pivot_cache

    def test_analyze_once_populates_caches(self):
        close = _zigzag_close(200)
        times = np.arange(close.size, dtype=float)
        cfg = ElliottWaveConfig(min_distance=3, min_confidence=0.0, wave_min_len=1)
        analyzer = ElliottWaveAnalyzer(close, times, cfg)

        analyzer.analyze_once(0.5, 3)
        assert (0.5, 3) in analyzer._pivot_cache
        assert (0.5, 3) in analyzer._wave_feature_cache

    def test_analyze_once_reuses_cached_pivots(self):
        close = _zigzag_close(200)
        times = np.arange(close.size, dtype=float)
        cfg = ElliottWaveConfig(min_distance=3, min_confidence=0.0, wave_min_len=1)
        analyzer = ElliottWaveAnalyzer(close, times, cfg)

        # Pre-populate the cache via pivot_signature
        sig = analyzer.pivot_signature(0.5, 3)
        # analyze_once should reuse, not recompute
        analyzer.analyze_once(0.5, 3)
        cached_pivots = analyzer._pivot_cache[(0.5, 3)]
        assert tuple(int(i) for i in cached_pivots) == sig

    def test_wave_feature_cache_hit(self):
        close = _zigzag_close(200)
        times = np.arange(close.size, dtype=float)
        cfg = ElliottWaveConfig(min_distance=3, min_confidence=0.0, wave_min_len=1)
        analyzer = ElliottWaveAnalyzer(close, times, cfg)

        r1 = analyzer.analyze_once(0.5, 3)
        feat1 = analyzer._wave_feature_cache.get((0.5, 3))
        r2 = analyzer.analyze_once(0.5, 3)
        feat2 = analyzer._wave_feature_cache.get((0.5, 3))
        # Same cached object
        assert feat1 is feat2
        # Results should be identical
        assert len(r1) == len(r2)
