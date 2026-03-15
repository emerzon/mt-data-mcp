"""Extended coverage tests for patterns/elliott.py targeting uncovered lines."""
import numpy as np
import pandas as pd
import pytest
import mtdata.patterns.elliott as elliott_mod

from mtdata.patterns.elliott import (
    ElliottWaveConfig,
    ElliottWaveResult,
    ElliottRuleEvaluation,
    ElliottScenario,
    _normalize_pattern_types,
    _zigzag_pivots_indices,
    _enforce_min_distance_on_pivots,
    _segment_waves_from_pivots,
    _extract_wave_features,
    _classify_waves,
    _window_hit,
    _evaluate_impulse_rules,
    _impulse_rules_and_score,
    _evaluate_correction_rules,
    _classification_score_window,
    ElliottWaveAnalyzer,
    detect_elliott_waves,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _trending_close(n=200, seed=42):
    """Generate a zig-zag close array suitable for wave detection."""
    rng = np.random.RandomState(seed)
    base = np.cumsum(rng.randn(n) * 0.5) + 100.0
    return np.maximum(base, 10.0)


def _impulse_close():
    """Hand-crafted 5-wave bullish impulse: 0→1→2→3→4→5."""
    return np.array([100, 120, 108, 150, 135, 160], dtype=float)


def _correction_close():
    """Hand-crafted ABC bullish correction: S→A→B→C."""
    return np.array([100, 115, 107, 125], dtype=float)


def _make_df(close, n=None):
    if n is None:
        n = len(close)
    t = np.arange(n, dtype=float) * 3600 + 1_700_000_000
    return pd.DataFrame({"time": t[:len(close)], "close": close})


# ===== _zigzag_pivots_indices (lines 124-139) ==============================

class TestZigzagPivotsExtended:
    def test_empty_array(self):
        piv, dirs = _zigzag_pivots_indices(np.array([]), 1.0)
        assert piv == [] and dirs == []

    def test_single_element(self):
        piv, dirs = _zigzag_pivots_indices(np.array([100.0]), 1.0)
        assert piv == [] and dirs == []

    def test_both_up_and_down_start_low_first(self):
        """Lines 124-128: can_start_up and can_start_down, pre_low_i > pre_high_i → up."""
        close = np.array([100.0, 95.0, 110.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 3.0)
        assert len(piv) >= 1

    def test_both_start_high_first(self):
        """Lines 128-131: pre_high_i > pre_low_i → down."""
        close = np.array([100.0, 108.0, 90.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 3.0)
        assert len(piv) >= 1

    def test_both_start_equal_indices_up_wins(self):
        """Lines 132-135: same index, up_change >= down_change → up."""
        close = np.array([100.0, 100.0, 115.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 5.0)
        assert len(piv) >= 1

    def test_both_start_equal_indices_down_wins(self):
        """Lines 136-139: same index, down_change > up_change → down."""
        close = np.array([100.0, 100.0, 80.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 5.0)
        assert len(piv) >= 1

    def test_only_up_start(self):
        """Lines 140-142."""
        close = np.array([100.0, 99.0, 99.5, 115.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 10.0)
        assert len(piv) >= 1

    def test_only_down_start(self):
        """Lines 144-147."""
        close = np.array([100.0, 101.0, 100.5, 85.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 10.0)
        assert len(piv) >= 1

    def test_nan_skipped(self):
        """Line 108: non-finite values skipped."""
        close = np.array([100.0, np.nan, 95.0, 110.0, 90.0], dtype=float)
        piv, dirs = _zigzag_pivots_indices(close, 3.0)
        assert all(np.isfinite(close[i]) or i == 0 for i in piv)

    def test_long_zigzag(self):
        close = _trending_close(100, seed=0)
        piv, dirs = _zigzag_pivots_indices(close, 2.0)
        assert len(piv) >= 3


# ===== _enforce_min_distance_on_pivots (lines 218-225) =====================

class TestEnforceMinDistanceExtended:
    def test_empty(self):
        assert _enforce_min_distance_on_pivots([], np.array([1.0]), 3) == []

    def test_single_pivot(self):
        assert _enforce_min_distance_on_pivots([0], np.array([100.0]), 3) == [0]

    def test_close_pivots_merge_anchor(self):
        """Lines 218-225: pivots too close, anchor-based merge keeps better move."""
        close = np.array([100, 110, 108, 95, 130, 125], dtype=float)
        pivots = [0, 1, 2, 3, 4, 5]
        out = _enforce_min_distance_on_pivots(pivots, close, 3)
        assert len(out) < len(pivots)

    def test_close_pivots_first_pair(self):
        """Lines 218-220: len(out)==1, replace."""
        close = np.array([100, 110, 120], dtype=float)
        out = _enforce_min_distance_on_pivots([0, 1, 2], close, 5)
        assert len(out) >= 1

    def test_out_of_bounds_skipped(self):
        """Lines 199: idx out of bounds."""
        close = np.array([100.0, 110.0], dtype=float)
        out = _enforce_min_distance_on_pivots([-1, 0, 5, 1], close, 1)
        assert -1 not in out and 5 not in out

    def test_duplicate_pivot(self):
        """Line 203-204: duplicate index skipped."""
        close = np.array([100, 110, 120], dtype=float)
        out = _enforce_min_distance_on_pivots([0, 0, 1, 2], close, 1)
        assert out.count(0) <= 1


# ===== _classify_waves (lines 280-284) =====================================

class TestClassifyWaves:
    def test_too_few_features(self):
        features = np.array([[1.0, 0.1, 0.01]])
        labels, gmm, scaler, probs, imp = _classify_waves(features, ElliottWaveConfig())
        assert labels.size == 0

    def test_enough_features(self):
        """Lines 280-284: successful GMM classification."""
        rng = np.random.RandomState(42)
        features = np.vstack([
            rng.randn(10, 3) * [5, 0.02, 0.01],
            rng.randn(10, 3) * [3, 0.05, 0.02] + [0, 0.1, 0.05],
        ])
        labels, gmm, scaler, probs, imp_cluster = _classify_waves(features, ElliottWaveConfig())
        assert labels.shape[0] == 20
        assert gmm is not None
        assert probs is not None
        assert imp_cluster in (0, 1)

    def test_non_finite_scaled(self):
        """Lines 275-276: non-finite after scaling."""
        features = np.array([[np.inf, 0, 0], [1, 2, 3]], dtype=float)
        labels, _, _, _, _ = _classify_waves(features, ElliottWaveConfig())
        assert labels.size == 0


# ===== _classification_score_window (lines 422-440) ========================

class TestClassificationScoreWindow:
    def test_none_probs(self):
        """Line 420-421."""
        assert _classification_score_window(None, None, 0, True, 5, [0, 2], [1]) == 0.5

    def test_insufficient_probs(self):
        """Line 422-423."""
        probs = np.random.rand(3, 2)
        means = np.random.rand(2, 3)
        assert _classification_score_window(probs, means, 0, True, 5, [0], [1]) == 0.5

    def test_bad_cluster_means_shape(self):
        """Line 424-425."""
        probs = np.random.rand(20, 2)
        means = np.array([1.0])  # 1D
        assert _classification_score_window(probs, means, 0, True, 5, [0], [1]) == 0.5

    def test_valid_bullish(self):
        """Lines 427-440."""
        rng = np.random.RandomState(0)
        probs = rng.rand(20, 2)
        means = np.array([[0.1, 0.5, 0.1], [-0.1, -0.3, 0.0]])
        score = _classification_score_window(probs, means, 0, True, 5, [0, 2, 4], [1, 3])
        assert 0.0 <= score <= 1.0

    def test_valid_bearish(self):
        rng = np.random.RandomState(0)
        probs = rng.rand(20, 2)
        means = np.array([[0.1, 0.5, 0.1], [-0.1, -0.3, 0.0]])
        score = _classification_score_window(probs, means, 0, False, 5, [0, 2, 4], [1, 3])
        assert 0.0 <= score <= 1.0

    def test_empty_slots(self):
        """Lines 434-435: no trend_vals and no counter_vals."""
        probs = np.random.rand(20, 2)
        means = np.array([[0.1, 0.5], [-0.1, -0.3]])
        score = _classification_score_window(probs, means, 0, True, 5, [], [])
        assert score == 0.5

    def test_window_mismatch(self):
        """Line 429: window_probs.shape[0] != window_len."""
        probs = np.random.rand(7, 2)
        means = np.array([[0.1, 0.5], [-0.1, -0.3]])
        score = _classification_score_window(probs, means, 5, True, 5, [0], [1])
        assert score == 0.5


# ===== _evaluate_impulse_rules and _evaluate_correction_rules ==============

class TestEvaluateImpulseRules:
    def test_valid_bullish(self):
        c = _impulse_close()
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        assert ev.valid
        assert ev.fib_score >= 0

    def test_wrong_pivot_count(self):
        c = np.array([100, 120, 110], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2], bullish=True)
        assert not ev.valid
        assert "pivot_count_not_6" in ev.violations

    def test_wave2_over_retrace_bullish(self):
        """Lines 323-324."""
        c = np.array([100, 120, 90, 150, 135, 160], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        assert "wave2_over_retrace" in ev.violations

    def test_wave2_over_retrace_bearish(self):
        """Lines 325-326."""
        c = np.array([160, 135, 170, 100, 108, 90], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=False)
        assert "wave2_over_retrace" in ev.violations

    def test_wave3_shortest(self):
        """Lines 329-330."""
        c = np.array([100, 120, 115, 118, 110, 150], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        assert "wave3_shortest" in ev.violations

    def test_wave4_overlap_bullish(self):
        """Lines 333-334."""
        c = np.array([100, 120, 108, 150, 115, 160], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        assert "wave4_overlap" in ev.violations

    def test_wave4_overlap_bearish(self):
        """Lines 335-336."""
        c = np.array([160, 140, 148, 110, 145, 90], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=False)
        assert "wave4_overlap" in ev.violations

    def test_non_finite(self):
        c = np.array([100, np.nan, 110, 130, 120, 140], dtype=float)
        ev = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        assert "non_finite_prices" in ev.violations

    def test_backward_compat_wrapper(self):
        c = _impulse_close()
        valid, fib, metrics = _impulse_rules_and_score(c, [0, 1, 2, 3, 4, 5], True)
        assert isinstance(valid, bool)
        assert "fib_score" in metrics


class TestEvaluateCorrectionRules:
    def test_valid_bullish(self):
        c = _correction_close()
        ev = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=True)
        assert isinstance(ev.fib_score, float)

    def test_wrong_count(self):
        ev = _evaluate_correction_rules(np.array([1, 2, 3], dtype=float), [0, 1, 2], True)
        assert "pivot_count_not_4" in ev.violations

    def test_waveB_over_retrace_bullish(self):
        c = np.array([100, 115, 95, 125], dtype=float)
        ev = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=True)
        assert "waveB_over_retrace" in ev.violations

    def test_waveB_over_retrace_bearish(self):
        c = np.array([125, 110, 130, 100], dtype=float)
        ev = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=False)
        assert "waveB_over_retrace" in ev.violations

    def test_waveC_too_short(self):
        c = np.array([100, 120, 115, 116], dtype=float)
        ev = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=True)
        assert "waveC_too_short" in ev.violations


# ===== ElliottWaveAnalyzer.build_result (lines 576-577) ====================

class TestBuildResult:
    def test_correction_labels(self):
        """Lines 567-568: correction → S, A, B, C labels."""
        c = _correction_close()
        t = np.arange(4, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(c, t, ElliottWaveConfig())
        rule_eval = _evaluate_correction_rules(c, [0, 1, 2, 3], bullish=True)
        scenario = ElliottScenario(
            pivots=[0, 1, 2, 3], bullish=True, confidence=0.5, cls_score=0.5,
            rule_eval=rule_eval, threshold_used=1.0, min_distance_used=1,
            wave_type="Correction",
        )
        result = analyzer.build_result(scenario)
        labels = [wp["label"] for wp in result.details["wave_points_labeled"]]
        assert labels == ["S", "A", "B", "C"]
        assert result.details["pattern_family"] == "correction"
        assert "correction_metrics" in result.details

    def test_impulse_labels(self):
        c = _impulse_close()
        t = np.arange(6, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(c, t, ElliottWaveConfig())
        rule_eval = _evaluate_impulse_rules(c, [0, 1, 2, 3, 4, 5], bullish=True)
        scenario = ElliottScenario(
            pivots=[0, 1, 2, 3, 4, 5], bullish=True, confidence=0.7, cls_score=0.6,
            rule_eval=rule_eval, threshold_used=1.0, min_distance_used=1,
            wave_type="Impulse",
        )
        result = analyzer.build_result(scenario)
        labels = [wp["label"] for wp in result.details["wave_points_labeled"]]
        assert labels[0] == "W0"


# ===== ElliottWaveAnalyzer.build_fallback (lines 623-651) ==================

class TestBuildFallback:
    def test_disabled_fallback(self):
        """Line 618-619."""
        cfg = ElliottWaveConfig(include_fallback_candidate=False)
        c = _trending_close(50)
        t = np.arange(50, dtype=float) * 3600
        analyzer = ElliottWaveAnalyzer(c, t, cfg)
        assert analyzer.build_fallback(1.0, 3) is None

    def test_too_short(self):
        """Lines 621-623."""
        cfg = ElliottWaveConfig()
        c = np.array([100.0])
        t = np.array([0.0])
        analyzer = ElliottWaveAnalyzer(c, t, cfg)
        assert analyzer.build_fallback(1.0, 3) is None

    def test_fallback_produces_result(self):
        """Lines 625-667: full fallback path."""
        cfg = ElliottWaveConfig(include_fallback_candidate=True, min_distance=1)
        close = _trending_close(60, seed=10)
        t = np.arange(60, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(close, t, cfg)
        result = analyzer.build_fallback(0.5, 1)
        # May or may not produce depending on data, but shouldn't crash
        assert result is None or isinstance(result, ElliottWaveResult)

    def test_fallback_correction_only(self):
        """Line 637: correction-only pattern_types → preferred_len=4."""
        cfg = ElliottWaveConfig(
            include_fallback_candidate=True, min_distance=1,
            pattern_types=["correction"],
        )
        close = _trending_close(60, seed=20)
        t = np.arange(60, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(close, t, cfg)
        result = analyzer.build_fallback(0.5, 1)
        assert result is None or isinstance(result, ElliottWaveResult)

    def test_fallback_marks_synthetic_terminal_pivot(self, monkeypatch):
        cfg = ElliottWaveConfig(include_fallback_candidate=True, min_distance=1, pattern_types=["correction"])
        close = _trending_close(20, seed=21)
        t = np.arange(20, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(close, t, cfg)

        monkeypatch.setattr(elliott_mod, "_zigzag_pivots_indices", lambda *_args, **_kwargs: ([0, 4, 8], ["up", "down", "up"]))
        monkeypatch.setattr(elliott_mod, "_enforce_min_distance_on_pivots", lambda pivots, *_args, **_kwargs: list(pivots))

        result = analyzer.build_fallback(0.5, 1)

        assert result is not None
        assert result.wave_sequence[-1] == len(close) - 1
        assert result.details["synthetic_terminal_pivot"] is True


# ===== detect_elliott_waves (lines 672-728, 747-751) =======================

class TestDetectElliottWaves:
    def test_none_config(self):
        """Line 671-672."""
        df = _make_df(_trending_close(100))
        results = detect_elliott_waves(df, None)
        assert isinstance(results, list)

    def test_not_dataframe(self):
        """Line 674-675."""
        assert detect_elliott_waves("not_a_df") == []

    def test_no_close_column(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        assert detect_elliott_waves(df) == []

    def test_too_short(self):
        df = _make_df(np.array([100.0, 110.0]))
        assert detect_elliott_waves(df) == []

    def test_autotune_default_thresholds(self):
        """Lines 692-703: autotune with default grids."""
        close = _trending_close(200, seed=5)
        df = _make_df(close)
        cfg = ElliottWaveConfig(autotune=True, min_distance=3, min_confidence=0.0)
        results = detect_elliott_waves(df, cfg)
        assert isinstance(results, list)

    def test_autotune_custom_thresholds(self):
        """Lines 693-696: custom tune_thresholds."""
        close = _trending_close(200, seed=6)
        df = _make_df(close)
        cfg = ElliottWaveConfig(
            autotune=True, tune_thresholds=[0.5, 1.0],
            tune_min_distance=[2, 4], min_confidence=0.0,
        )
        results = detect_elliott_waves(df, cfg)
        assert isinstance(results, list)

    def test_no_autotune(self):
        """Lines 729-733: non-autotune path."""
        close = _trending_close(200, seed=7)
        df = _make_df(close)
        cfg = ElliottWaveConfig(autotune=False, min_prominence_pct=1.0, min_distance=3)
        results = detect_elliott_waves(df, cfg)
        assert isinstance(results, list)

    def test_no_autotune_with_swing_threshold(self):
        close = _trending_close(200, seed=8)
        df = _make_df(close)
        cfg = ElliottWaveConfig(autotune=False, swing_threshold_pct=0.5, min_distance=3)
        results = detect_elliott_waves(df, cfg)
        assert isinstance(results, list)

    def test_top_k_limits(self):
        """Line 754-756."""
        close = _trending_close(200, seed=9)
        df = _make_df(close)
        cfg = ElliottWaveConfig(autotune=True, min_distance=2, min_confidence=0.0, top_k=2)
        results = detect_elliott_waves(df, cfg)
        assert len(results) <= 2

    def test_correction_only(self):
        close = _trending_close(200, seed=11)
        df = _make_df(close)
        cfg = ElliottWaveConfig(autotune=True, min_distance=2, pattern_types=["correction"], min_confidence=0.0)
        results = detect_elliott_waves(df, cfg)
        assert all(r.wave_type in ("Correction", "Candidate") for r in results)

    def test_dedup_keeps_higher_confidence(self):
        """Lines 720-726: duplicate key keeps higher confidence."""
        close = _trending_close(200, seed=12)
        df = _make_df(close)
        cfg = ElliottWaveConfig(
            autotune=True, tune_thresholds=[0.3, 0.5, 0.7],
            tune_min_distance=[2, 3], min_confidence=0.0,
        )
        results = detect_elliott_waves(df, cfg)
        keys = [(r.wave_type, tuple(r.wave_sequence)) for r in results]
        assert len(keys) == len(set(keys)), "Duplicate wave sequences should be deduped"

    def test_fallback_appended_when_no_recent(self):
        """Lines 738-745: fallback candidate appended."""
        rng = np.random.RandomState(55)
        # Build data where waves end early, leaving tail without recent detection
        close = np.concatenate([
            _trending_close(50, seed=55),
            np.full(50, 100.0) + rng.randn(50) * 0.01,
        ])
        df = _make_df(close)
        cfg = ElliottWaveConfig(
            autotune=True, min_distance=2, min_confidence=0.0,
            include_fallback_candidate=True, recent_bars=3,
        )
        results = detect_elliott_waves(df, cfg)
        assert isinstance(results, list)

    def test_autotune_duplicate_sequence_keeps_highest_confidence(self, monkeypatch):
        df = _make_df(np.linspace(100.0, 120.0, 80))
        cfg = ElliottWaveConfig(
            autotune=True,
            tune_thresholds=[0.2, 0.5],
            tune_min_distance=[1],
            include_fallback_candidate=False,
            min_confidence=0.0,
            top_k=10,
        )

        def _fake_analyze(self, threshold_pct, min_distance):
            _ = self
            return [
                ElliottScenario(
                    pivots=[0, 10, 20, 30],
                    bullish=True,
                    confidence=float(threshold_pct),
                    cls_score=0.5,
                    rule_eval=ElliottRuleEvaluation(valid=True, fib_score=0.5, metrics={}),
                    threshold_used=float(threshold_pct),
                    min_distance_used=int(min_distance),
                    wave_type="Correction",
                )
            ]

        monkeypatch.setattr(elliott_mod.ElliottWaveAnalyzer, "analyze_once", _fake_analyze)

        results = detect_elliott_waves(df, cfg)

        assert len(results) == 1
        assert results[0].wave_type == "Correction"
        assert results[0].confidence == pytest.approx(0.5)

    def test_analyze_once_skips_correction_subsequence_of_impulse(self, monkeypatch):
        close = _impulse_close()
        t = np.arange(close.size, dtype=float) * 3600 + 1_700_000_000
        analyzer = ElliottWaveAnalyzer(
            close,
            t,
            ElliottWaveConfig(
                autotune=False,
                min_distance=1,
                wave_min_len=1,
                min_confidence=0.0,
                pattern_types=["impulse", "correction"],
                include_fallback_candidate=False,
            ),
        )

        monkeypatch.setattr(elliott_mod, "_zigzag_pivots_indices", lambda *_args, **_kwargs: ([0, 1, 2, 3, 4, 5], ["up"] * 6))

        fake_gmm = type("FakeGMM", (), {"means_": np.array([[0.0, -0.1, 0.0], [0.0, 0.1, 0.0]])})()
        monkeypatch.setattr(
            elliott_mod,
            "_classify_waves",
            lambda features, config: (
                np.array([1, 0, 1, 0, 1], dtype=int),
                fake_gmm,
                None,
                np.full((features.shape[0], 2), 0.5, dtype=float),
                1,
            ),
        )

        scenarios = analyzer.analyze_once(1.0, 1)

        assert any(s.wave_type == "Impulse" for s in scenarios)
        assert not any(s.wave_type == "Correction" for s in scenarios)


# ===== _window_hit =========================================================

class TestWindowHit:
    def test_inside(self):
        assert _window_hit(0.5, 0.3, 0.7) == 1.0

    def test_below_tapered(self):
        score = _window_hit(0.28, 0.3, 0.7)
        assert 0.0 < score < 1.0

    def test_above_tapered(self):
        score = _window_hit(0.72, 0.3, 0.7)
        assert 0.0 < score < 1.0

    def test_zero_range(self):
        assert _window_hit(0.5, 0.5, 0.5) == 0.0

    def test_far_below(self):
        assert _window_hit(-10.0, 0.3, 0.7) == 0.0

    def test_far_above(self):
        assert _window_hit(10.0, 0.3, 0.7) == 0.0


# ===== _normalize_pattern_types ============================================

class TestNormalizePatternTypes:
    def test_default(self):
        cfg = ElliottWaveConfig()
        assert _normalize_pattern_types(cfg) == {"impulse", "correction"}

    def test_impulses_alias(self):
        cfg = ElliottWaveConfig(pattern_types=["impulses"])
        assert "impulse" in _normalize_pattern_types(cfg)

    def test_abc_alias(self):
        cfg = ElliottWaveConfig(pattern_types=["abc"])
        assert "correction" in _normalize_pattern_types(cfg)

    def test_empty_falls_back(self):
        cfg = ElliottWaveConfig(pattern_types=[])
        assert _normalize_pattern_types(cfg) == {"impulse", "correction"}

    def test_unknown_ignored(self):
        cfg = ElliottWaveConfig(pattern_types=["unknown"])
        # Falls back to default since no valid entries
        assert _normalize_pattern_types(cfg) == {"impulse", "correction"}
