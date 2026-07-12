from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mtdata.patterns.elliott as elliott
from mtdata.patterns.elliott import (
    ElliottPivot,
    ElliottRuleEvaluation,
    ElliottScenario,
    ElliottWaveAnalyzer,
    ElliottWaveConfig,
    ElliottWaveResult,
    _build_pivot_records,
    _dedupe_similar_waves,
    _ohlc_zigzag_pivots_indices,
    detect_elliott_waves,
)


def test_v2_rejects_unknown_pattern_type() -> None:
    config = ElliottWaveConfig(pattern_types=["impuulse"])
    with pytest.raises(ValueError, match="pattern_types"):
        detect_elliott_waves(pd.DataFrame({"close": np.arange(100.0)}), config)


def test_v2_rejects_empty_pattern_types() -> None:
    with pytest.raises(ValueError, match="pattern_types"):
        ElliottWaveConfig(pattern_types=[]).validate()


@pytest.mark.parametrize(
    "config, message",
    [
        (ElliottWaveConfig(pivot_price_source="hybrid"), "pivot_price_source"),
        (ElliottWaveConfig(swing_threshold_pct=0.0), "threshold"),
        (ElliottWaveConfig(gmm_components=1), "gmm_components"),
        (ElliottWaveConfig(top_k=-1), "top_k"),
    ],
)
def test_v2_rejects_invalid_config(config: ElliottWaveConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_causal_pivot_records_capture_confirmation_bar() -> None:
    close = np.array([100.0, 110.0, 108.0, 104.0, 105.0])
    records = _build_pivot_records([0, 1, 3, 4], close, threshold_pct=5.0)

    assert records[1] == ElliottPivot(
        index=1, kind="peak", confirmed=True, confirmation_index=3
    )
    assert records[-1].confirmed is False
    assert records[-1].confirmation_index is None


def test_result_exposes_geometry_and_causal_availability_separately() -> None:
    close = np.array([100.0, 110.0, 104.0, 120.0, 112.0, 125.0, 118.0])
    times = np.arange(close.size, dtype=float)
    records = _build_pivot_records(list(range(6)), close, threshold_pct=4.0)
    scenario = ElliottScenario(
        pivots=list(range(6)),
        bullish=True,
        confidence=0.7,
        cls_score=0.5,
        rule_eval=ElliottRuleEvaluation(True, 0.8, {}, []),
        threshold_used=4.0,
        min_distance_used=1,
        pivot_confirmations=[record.confirmed for record in records],
        pivot_records=records,
    )
    result = ElliottWaveAnalyzer(
        close, times, ElliottWaveConfig(min_impulse_bars=1)
    ).build_result(scenario)

    assert result.end_index == 5
    assert result.available_at_index == close.size - 1
    assert result.structure_state == "developing"
    assert result.details["pivots"][0]["time"] == 0.0
    assert np.isfinite(result.details["pivots"][0]["price"])


def test_terminal_nan_never_becomes_fallback_geometry() -> None:
    close = np.array(([100.0, 110.0, 104.0, 115.0] * 10), dtype=float)
    close[-1] = np.nan
    analyzer = ElliottWaveAnalyzer(
        close,
        np.arange(close.size, dtype=float),
        ElliottWaveConfig(
            min_distance=1,
            wave_min_len=1,
            min_correction_bars=1,
            pattern_types=["correction"],
        ),
    )
    assert analyzer.build_fallback(0.5, 1) is None


def test_dedupe_retains_disjoint_patterns_with_nearby_end_bars() -> None:
    common = {
        "wave_type": "Impulse",
        "confidence": 0.8,
        "start_time": None,
        "end_time": None,
        "details": {"sequence_direction": "bull", "pattern_confirmed": True},
    }
    first = ElliottWaveResult(
        wave_sequence=[0, 2, 4, 6, 8, 10], start_index=0, end_index=10, **common
    )
    second = ElliottWaveResult(
        wave_sequence=[11, 13, 15, 17, 19, 20],
        start_index=11,
        end_index=20,
        **common,
    )
    assert len(_dedupe_similar_waves([first, second], proximity_bars=24)) == 2


def test_explicit_swing_threshold_overrides_legacy_multiscan(monkeypatch) -> None:
    calls: list[tuple[float, int]] = []

    def fake_analyze(self, threshold_pct, min_distance):
        calls.append((float(threshold_pct), int(min_distance)))
        return []

    monkeypatch.setattr(elliott.ElliottWaveAnalyzer, "analyze_once", fake_analyze)
    monkeypatch.setattr(elliott.ElliottWaveAnalyzer, "build_fallback", lambda *args: None)
    config = ElliottWaveConfig(
        autotune=True,
        swing_threshold_pct=0.8,
        include_fallback_candidate=False,
        pattern_types=["correction"],
    )
    detect_elliott_waves(pd.DataFrame({"close": np.linspace(100.0, 120.0, 100)}), config)
    assert calls == [(0.8, config.min_distance)]


def test_mixed_mode_does_not_apply_impulse_floor_to_corrections(monkeypatch) -> None:
    calls: list[float] = []

    def fake_analyze(self, threshold_pct, min_distance):
        calls.append(float(threshold_pct))
        return []

    monkeypatch.setattr(elliott.ElliottWaveAnalyzer, "analyze_once", fake_analyze)
    monkeypatch.setattr(elliott.ElliottWaveAnalyzer, "build_fallback", lambda *args: None)
    config = ElliottWaveConfig(
        min_distance=1,
        wave_min_len=1,
        min_impulse_bars=30,
        min_correction_bars=10,
        pattern_types=["impulse", "correction"],
        include_fallback_candidate=False,
    )
    detect_elliott_waves(pd.DataFrame({"close": np.linspace(100.0, 110.0, 20)}), config)
    assert calls


def test_ohlc_zigzag_uses_wick_extrema() -> None:
    high = np.array([101.0, 112.0, 109.0, 108.0, 116.0, 114.0])
    low = np.array([99.0, 106.0, 103.0, 100.0, 110.0, 105.0])
    pivots, _ = _ohlc_zigzag_pivots_indices(high, low, threshold_pct=5.0)
    assert pivots
    assert all(0 <= pivot < high.size for pivot in pivots)
