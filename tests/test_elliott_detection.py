import numpy as np
import pandas as pd

from src.mtdata.patterns.eliott import (
    ElliottWaveConfig,
    _impulse_rules_and_score,
    _zigzag_pivots_indices,
    detect_elliott_waves,
)


def test_impulse_rule_rejects_wave3_shortest():
    # W3 is intentionally shorter than both W1 and W5.
    close = np.array([100.0, 130.0, 120.0, 132.0, 131.0, 171.0], dtype=float)
    valid, score, metrics = _impulse_rules_and_score(close, [0, 1, 2, 3, 4, 5], bullish=True)
    assert valid is False
    assert float(score) >= 0.0
    assert isinstance(metrics, dict)


def test_detect_elliott_waves_returns_candidate_for_fallback():
    df = pd.DataFrame({"close": np.linspace(100.0, 150.0, 120)})
    cfg = ElliottWaveConfig(
        autotune=False,
        swing_threshold_pct=0.5,
        min_distance=5,
        include_fallback_candidate=True,
        top_k=5,
    )

    results = detect_elliott_waves(df, cfg)
    assert len(results) >= 1
    assert results[0].wave_type == "Candidate"
    assert bool(results[0].details.get("fallback_candidate")) is True
    assert len(results[0].wave_sequence) < 6


def test_zigzag_pretrend_uses_running_extrema_before_trend_is_set():
    close = np.array([100.0, 102.0, 98.0, 97.0], dtype=float)

    piv_idx, piv_dir = _zigzag_pivots_indices(close, 3.0)

    assert piv_idx == [1, 3]
    assert piv_dir == ["down", "down"]


def test_wave_min_len_filters_short_leg_sequences():
    x = np.arange(30, dtype=float)
    piv_idx = [0, 3, 6, 9, 12, 15, 29]
    piv_val = [100, 112, 106, 124, 116, 136, 140]
    close = np.interp(x, piv_idx, piv_val)
    df = pd.DataFrame({"close": close})

    loose_cfg = ElliottWaveConfig(
        autotune=False,
        swing_threshold_pct=2.0,
        min_distance=1,
        wave_min_len=1,
        pattern_types=["impulse"],
        include_fallback_candidate=False,
        top_k=10,
    )
    strict_cfg = ElliottWaveConfig(
        autotune=False,
        swing_threshold_pct=2.0,
        min_distance=1,
        wave_min_len=6,
        pattern_types=["impulse"],
        include_fallback_candidate=False,
        top_k=10,
    )

    loose_results = detect_elliott_waves(df, loose_cfg)
    strict_results = detect_elliott_waves(df, strict_cfg)

    assert len(loose_results) >= 1
    assert loose_results[0].wave_type == "Impulse"
    assert len(strict_results) == 0


def test_detect_abc_correction_when_enabled():
    x = np.arange(60, dtype=float)
    piv_idx = [0, 8, 16, 24, 32, 40, 48, 59]
    piv_val = [100, 90, 95, 85, 92, 82, 88, 80]
    close = np.interp(x, piv_idx, piv_val)
    df = pd.DataFrame({"close": close})

    cfg = ElliottWaveConfig(
        autotune=False,
        swing_threshold_pct=2.0,
        min_distance=2,
        wave_min_len=2,
        pattern_types=["correction"],
        include_fallback_candidate=False,
        top_k=10,
    )
    results = detect_elliott_waves(df, cfg)

    assert len(results) >= 1
    assert all(r.wave_type == "Correction" for r in results)
    assert len(results[0].wave_sequence) == 4
    assert "correction_metrics" in results[0].details


def test_pattern_type_filter_impulse_only_excludes_correction():
    x = np.arange(60, dtype=float)
    piv_idx = [0, 8, 16, 24, 32, 40, 48, 59]
    piv_val = [100, 90, 95, 85, 92, 82, 88, 80]
    close = np.interp(x, piv_idx, piv_val)
    df = pd.DataFrame({"close": close})

    cfg = ElliottWaveConfig(
        autotune=False,
        swing_threshold_pct=2.0,
        min_distance=2,
        wave_min_len=2,
        pattern_types=["impulse"],
        include_fallback_candidate=False,
        top_k=10,
    )
    results = detect_elliott_waves(df, cfg)
    assert len(results) == 0
