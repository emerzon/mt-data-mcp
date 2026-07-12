from __future__ import annotations

import numpy as np
import pandas as pd

from mtdata.patterns.harmonic import (
    HarmonicDetectorConfig,
    _ratio_abs_tolerance,
    detect_harmonic_patterns,
    validate_harmonic_detector_config,
)


def test_ratio_tolerance_is_proportional_for_small_and_large_ratios() -> None:
    cfg = HarmonicDetectorConfig(ratio_tolerance=0.06)

    assert _ratio_abs_tolerance(0.382, 0.382, cfg) == 0.06 * 0.382
    assert _ratio_abs_tolerance(2.618, 2.618, cfg) == 0.06 * 2.618


def _harmonic_sample_df() -> pd.DataFrame:
    anchors = [
        (0, 112.0),
        (20, 100.0),
        (40, 120.0),
        (60, 107.64),
        (80, 115.0),
        (100, 104.28),
        (120, 112.0),
    ]
    prices: list[float] = []
    for (idx0, price0), (idx1, price1) in zip(anchors, anchors[1:]):
        for idx in range(idx0, idx1):
            frac = (idx - idx0) / (idx1 - idx0)
            prices.append(price0 + (price1 - price0) * frac)
    prices.append(anchors[-1][1])

    close = np.asarray(prices, dtype=float)
    times = np.arange(1704067200, 1704067200 + len(close) * 3600, 3600)
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "time": times,
        }
    )


def _test_config(**overrides) -> HarmonicDetectorConfig:
    cfg = HarmonicDetectorConfig(
        pattern_types=["gartley"],
        pivot_use_hl=False,
        pivot_use_atr_adaptive_prominence=False,
        pivot_use_atr_adaptive_distance=False,
        min_prominence_pct=0.1,
        min_distance=5,
        recent_bars=30,
        min_input_bars=40,
        min_confidence=0.3,
        max_pivots=20,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_detects_bullish_gartley_from_fibonacci_pivots():
    patterns = detect_harmonic_patterns(_harmonic_sample_df(), _test_config())

    assert patterns
    pattern = patterns[0]
    assert pattern.name == "Bullish Gartley"
    assert pattern.status == "completed"
    assert pattern.details["completion_freshness"] == "recent"
    assert pattern.bias == "bullish"
    assert pattern.start_index == 20
    assert pattern.end_index == 100
    assert pattern.details["harmonic_pattern"] == "gartley"
    assert pattern.details["pivot_indexes"] == {
        "X": 20,
        "A": 40,
        "B": 60,
        "C": 80,
        "D": 100,
    }
    assert pattern.details["ratios"]["xab"] == 0.618
    assert pattern.details["prz_low"] <= pattern.entry_price <= pattern.details["prz_high"]
    assert pattern.target_prices[0] > pattern.entry_price
    assert pattern.invalidation_price < pattern.entry_price


def test_completed_harmonic_is_not_removed_when_forming_patterns_are_excluded():
    patterns = detect_harmonic_patterns(
        _harmonic_sample_df(),
        _test_config(include_forming=False),
    )

    assert patterns
    assert all(pattern.status == "completed" for pattern in patterns)


def test_pattern_type_filter_limits_candidates():
    patterns = detect_harmonic_patterns(
        _harmonic_sample_df(),
        _test_config(pattern_types=["crab"]),
    )

    assert patterns == []


def test_validate_harmonic_config_reports_unsupported_pattern_type():
    cfg = _test_config(pattern_types=["gartley", "not_real"])

    warnings = validate_harmonic_detector_config(cfg)

    assert any("not_real" in warning for warning in warnings)


def test_harmonic_detector_returns_empty_for_short_input():
    df = _harmonic_sample_df().iloc[:20].copy()

    assert detect_harmonic_patterns(df, _test_config()) == []
