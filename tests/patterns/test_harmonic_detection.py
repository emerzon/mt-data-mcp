from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.patterns.harmonic import (
    _OXABC_SPECS,
    _XABCD_SPECS,
    HarmonicDetectorConfig,
    _candidate_results,
    _ratio_abs_tolerance,
    _ratios_oxabc,
    _ratios_xabcd,
    _score_ratio,
    _score_spec,
    _SwingPoint,
    detect_harmonic_patterns,
    validate_harmonic_detector_config,
)


def test_ratio_tolerance_is_proportional_for_small_and_large_ratios() -> None:
    cfg = HarmonicDetectorConfig(ratio_tolerance=0.06)

    assert _ratio_abs_tolerance(0.382, 0.382, cfg) == 0.06 * 0.382
    assert _ratio_abs_tolerance(2.618, 2.618, cfg) == 0.06 * 2.618


@pytest.mark.parametrize("value", [0.382, 0.886])
def test_harmonic_ratio_band_canonical_endpoints_score_fully(value: float) -> None:
    score = _score_ratio(
        value,
        0.382,
        0.886,
        HarmonicDetectorConfig(ratio_tolerance=0.06),
    )

    assert score == 1.0


def _cypher_points(c_price: float, d_price: float) -> list[_SwingPoint]:
    return [
        _SwingPoint(0, "low", 100.0),
        _SwingPoint(1, "high", 200.0),
        _SwingPoint(2, "low", 150.0),
        _SwingPoint(3, "high", c_price),
        _SwingPoint(4, "low", d_price),
    ]


def test_cypher_scores_c_extension_against_xa() -> None:
    cfg = HarmonicDetectorConfig(ratio_tolerance=0.03)
    ratios = _ratios_xabcd(_cypher_points(240.0, 129.96))

    assert ratios is not None
    assert ratios["xca"] == pytest.approx(1.4)
    assert ratios["abc"] == pytest.approx(1.8)
    assert ratios["xcd"] == pytest.approx(0.786)
    assert _score_spec(ratios, _XABCD_SPECS["cypher"], cfg) is not None


def test_cypher_rejects_bc_ab_fit_without_xa_extension() -> None:
    cfg = HarmonicDetectorConfig(ratio_tolerance=0.03)
    ratios = _ratios_xabcd(_cypher_points(215.0, 124.61))

    assert ratios is not None
    assert ratios["abc"] == pytest.approx(1.3)
    assert ratios["xca"] == pytest.approx(1.15)
    assert _score_spec(ratios, _XABCD_SPECS["cypher"], cfg) is None


def test_shark_uses_its_oxabc_geometry() -> None:
    points = [
        _SwingPoint(0, "low", 100.0),
        _SwingPoint(1, "high", 200.0),
        _SwingPoint(2, "low", 150.0),
        _SwingPoint(3, "high", 210.0),
        _SwingPoint(4, "low", 102.0),
    ]
    cfg = HarmonicDetectorConfig(min_confidence=0.3)
    ratios = _ratios_oxabc(points)

    assert "shark" not in _XABCD_SPECS
    assert ratios is not None
    assert ratios["oxa"] == pytest.approx(0.5)
    assert ratios["xab"] == pytest.approx(1.2)
    assert ratios["abc"] == pytest.approx(1.8)
    assert ratios["xc_ox"] == pytest.approx(0.98)
    assert _score_spec(ratios, _OXABC_SPECS["shark"], cfg) is not None

    results = _candidate_results(
        points,
        [_OXABC_SPECS["shark"]],
        np.arange(10),
        10,
        cfg,
    )

    assert len(results) == 1
    assert results[0].details["pivot_labels"] == list("OXABC")
    assert results[0].details["expected_ratios"]["xc_ox"] == [0.886, 1.13]


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


def test_harmonic_config_does_not_advertise_unsupported_forming_output():
    cfg = _test_config()
    patterns = detect_harmonic_patterns(_harmonic_sample_df(), cfg)

    assert not hasattr(cfg, "include_forming")
    assert patterns
    assert all(pattern.status == "completed" for pattern in patterns)


def test_recent_terminal_harmonic_pivot_is_marked_forming():
    patterns = detect_harmonic_patterns(
        _harmonic_sample_df().iloc[:103].copy(),
        _test_config(min_distance=5),
    )

    assert patterns
    assert patterns[0].status == "forming"
    assert patterns[0].details["terminal_pivot_confirmed"] is False
    assert patterns[0].details["available_at_index"] == 102


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
