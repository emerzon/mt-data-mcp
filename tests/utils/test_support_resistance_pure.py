from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.utils.support_resistance import (
    _resolve_adaptive_settings,
    compute_support_resistance_levels,
    merge_support_resistance_results,
)


def _clustered_levels_frame() -> pd.DataFrame:
    closes = [
        104.0, 102.0, 100.6, 102.2, 104.0,
        106.0, 108.0, 109.8, 108.0, 106.0,
        104.0, 102.0, 100.8, 102.4, 104.2,
        106.2, 108.4, 109.6, 107.2, 105.0,
    ]
    highs = [value + 0.6 for value in closes]
    lows = [value - 0.6 for value in closes]
    lows[2] = 99.8
    lows[12] = 100.0
    highs[7] = 110.6
    highs[17] = 110.1
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_000_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _weighted_supports_frame() -> pd.DataFrame:
    closes = [
        104.0, 102.0, 100.4, 101.0, 101.4,
        102.0, 103.0, 104.0, 105.0, 106.0,
        107.0, 106.0, 105.0, 104.0, 103.0,
        102.0, 101.3, 104.0, 107.0, 109.5,
    ]
    highs = [value + 0.6 for value in closes]
    lows = [value - 0.6 for value in closes]
    lows[2] = 99.9
    lows[16] = 100.7
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_050_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _broken_support_frame() -> pd.DataFrame:
    closes = [104.5, 103.2, 102.0, 103.0, 104.2, 103.4, 102.3, 103.1, 104.0, 100.8, 99.4, 98.8, 99.5]
    highs = [value + 0.5 for value in closes]
    lows = [value - 0.5 for value in closes]
    lows[2] = 101.6
    lows[6] = 101.8
    highs[8] = 104.7
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_100_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _role_reversal_frame() -> pd.DataFrame:
    closes = [96.0, 98.0, 99.8, 98.7, 99.6, 98.9, 101.2, 102.0, 101.4, 100.3, 101.8, 103.0, 102.4]
    highs = [value + 0.5 for value in closes]
    lows = [value - 0.5 for value in closes]
    highs[2] = 100.4
    highs[4] = 100.2
    lows[9] = 99.9
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_150_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _noisy_trend_frame() -> pd.DataFrame:
    closes = [
        100.0, 100.8, 99.9, 100.9, 100.0,
        101.0, 100.1, 101.1, 100.2, 101.2,
        100.3, 101.3, 100.4, 101.4, 100.5,
        101.5, 100.6, 101.6, 100.7, 101.7,
    ]
    highs = [value + 0.25 for value in closes]
    lows = [value - 0.25 for value in closes]
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_200_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _episode_count_frame() -> pd.DataFrame:
    closes = [
        104.0, 102.0, 100.2, 102.3, 101.0, 100.1, 102.4, 104.0,
        106.0, 108.0, 106.2, 104.0, 102.0, 100.3, 102.2, 100.2,
        102.5, 104.2, 106.0, 108.0,
    ]
    highs = [value + 0.5 for value in closes]
    lows = [value - 0.5 for value in closes]
    lows[2] = 99.9
    lows[5] = 99.95
    lows[13] = 100.0
    lows[15] = 99.92
    highs[3] = 102.9
    highs[6] = 103.0
    highs[9] = 108.6
    highs[14] = 102.8
    highs[16] = 103.1
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_250_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _volatility_expansion_frame() -> pd.DataFrame:
    closes = [100.0 + 0.2 * idx for idx in range(20)]
    ranges = [0.4] * 12 + [1.8] * 8
    highs = [close + width / 2.0 for close, width in zip(closes, ranges)]
    lows = [close - width / 2.0 for close, width in zip(closes, ranges)]
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_300_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _volatility_compression_frame() -> pd.DataFrame:
    closes = [100.0 + 0.2 * idx for idx in range(20)]
    ranges = [1.8] * 12 + [0.4] * 8
    highs = [close + width / 2.0 for close, width in zip(closes, ranges)]
    lows = [close - width / 2.0 for close, width in zip(closes, ranges)]
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_350_000 + 3600 * i for i in range(len(closes))],
        }
    )


def test_compute_support_resistance_returns_ranked_levels_around_current_price():
    result = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_pct=0.005,
        min_touches=2,
        max_levels=3,
        reaction_bars=4,
    )

    assert result["current_price"] == 105.0
    assert len(result["supports"]) == 1
    assert len(result["resistances"]) == 1

    support = result["supports"][0]
    resistance = result["resistances"][0]
    assert support["type"] == "support"
    assert resistance["type"] == "resistance"
    assert support["value"] < result["current_price"] < resistance["value"]
    assert support["source_tests"]["support"] == 2
    assert resistance["source_tests"]["resistance"] == 2
    assert support["episodes"] == 2
    assert resistance["episodes"] == 2
    assert support["zone_low"] < support["value"] < support["zone_high"]
    assert resistance["zone_low"] < resistance["value"] < resistance["zone_high"]
    assert support["zone_width"] > 0.0
    assert resistance["zone_width_atr"] is not None and resistance["zone_width_atr"] > 0.0
    assert support["status"] == "active"
    assert support["breakout_analysis"]["decisive_break_count"] == 0
    assert support["score_breakdown"]["total"] == support["score"]
    assert resistance["score_breakdown"]["total"] == resistance["score"]
    assert support["strength_rank"] == 1
    assert resistance["strength_rank"] == 1


def test_compute_support_resistance_includes_fibonacci_levels_from_latest_relevant_swing():
    result = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_pct=0.005,
        min_touches=2,
        max_levels=3,
        reaction_bars=4,
    )

    fibonacci = result["fibonacci"]
    assert fibonacci["mode"] == "single"
    assert fibonacci["timeframe"] == "H1"
    assert fibonacci["swing"]["direction"] == "up"
    assert fibonacci["swing"]["contains_current_price"] is True
    assert fibonacci["swing"]["current_price_position"] == "within_swing"
    assert fibonacci["swing"]["anchor_low"]["value"] == pytest.approx(100.0)
    assert fibonacci["swing"]["anchor_high"]["value"] == pytest.approx(110.1)
    assert [level["label"] for level in fibonacci["levels"]] == [
        "78.6%",
        "61.8%",
        "50%",
        "38.2%",
        "23.6%",
        "127.2%",
        "161.8%",
    ]
    assert fibonacci["nearest"]["support"]["label"] == "61.8%"
    assert fibonacci["nearest"]["support"]["type"] == "support"
    assert fibonacci["nearest"]["support"]["value"] == pytest.approx(103.8582)
    assert fibonacci["nearest"]["resistance"]["label"] == "50%"
    assert fibonacci["nearest"]["resistance"]["type"] == "resistance"
    assert fibonacci["nearest"]["resistance"]["value"] == pytest.approx(105.05)


def test_recent_stronger_support_scores_above_older_weaker_support():
    result = compute_support_resistance_levels(
        _weighted_supports_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_pct=0.001,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
        adx_period=5,
    )

    supports = sorted(
        [level for level in result["supports"] if level.get("dominant_source") == "support"],
        key=lambda level: level["value"],
    )
    assert len(supports) >= 2

    older = supports[0]
    recent = supports[-1]
    assert older["value"] < recent["value"]
    assert older["score"] < recent["score"]
    assert older["strength_rank"] > recent["strength_rank"]
    assert older["avg_pretest_adx"] < recent["avg_pretest_adx"]


def test_no_extrema_returns_empty_levels():
    frame = pd.DataFrame(
        {
            "high": [1.0 + 0.01 * idx for idx in range(12)],
            "low": [0.9 + 0.01 * idx for idx in range(12)],
            "close": [0.95 + 0.01 * idx for idx in range(12)],
        }
    )

    result = compute_support_resistance_levels(frame, min_touches=1, max_levels=4)
    assert result["levels"] == []
    assert result["supports"] == []
    assert result["resistances"] == []


def test_falls_back_to_best_cluster_when_touch_requirement_is_strict():
    result = compute_support_resistance_levels(
        _weighted_supports_frame(),
        min_touches=5,
        max_levels=4,
        tolerance_pct=0.001,
        reaction_bars=4,
    )

    assert len(result["levels"]) == 1


def test_atr_filtered_swing_detection_reduces_whipsaw_noise_levels():
    result = compute_support_resistance_levels(
        _noisy_trend_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_pct=0.002,
        min_touches=1,
        max_levels=10,
        reaction_bars=3,
        adx_period=5,
    )

    assert len(result["levels"]) <= 2
    assert all(level["touches"] <= 1 for level in result["levels"])


def test_volatility_expansion_widens_tolerance_and_shortens_reaction_window():
    result = compute_support_resistance_levels(
        _volatility_expansion_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_pct=0.0015,
        min_touches=1,
        max_levels=4,
        reaction_bars=6,
        adx_period=5,
    )

    assert result["adaptive_mode"] == "atr_regime"
    assert result["effective_tolerance_pct"] > result["tolerance_pct"]
    assert result["effective_reaction_bars"] < result["reaction_bars"]
    assert result["volatility_ratio"] > 1.0


def test_volatility_compression_narrows_tolerance_and_extends_reaction_window():
    result = compute_support_resistance_levels(
        _volatility_compression_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_pct=0.0015,
        min_touches=1,
        max_levels=4,
        reaction_bars=6,
        adx_period=5,
    )

    assert result["adaptive_mode"] == "atr_regime"
    assert result["effective_tolerance_pct"] < result["tolerance_pct"]
    assert result["effective_reaction_bars"] > result["reaction_bars"]
    assert result["volatility_ratio"] < 1.0


def test_adaptive_settings_excludes_recent_window_from_baseline():
    closes = np.full(12, 100.0)
    atr = np.array([1.0] * 4 + [3.0] * 8, dtype=float)

    result = _resolve_adaptive_settings(
        closes,
        atr,
        base_tolerance_pct=0.0015,
        base_reaction_bars=6,
    )

    assert result["baseline_atr_pct"] == pytest.approx(0.01)
    assert result["current_atr_pct"] == pytest.approx(0.03)
    assert result["volatility_ratio"] == pytest.approx(3.0)


def test_episode_counting_keeps_raw_touches_secondary_to_distinct_tests():
    result = compute_support_resistance_levels(
        _episode_count_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_pct=0.004,
        min_touches=2,
        max_levels=4,
        reaction_bars=4,
        adx_period=5,
    )

    support = next(level for level in result["supports"] if level["dominant_source"] == "support")
    assert result["qualification_basis"] == "episodes"
    assert support["touches"] > support["episodes"]
    assert support["episodes"] == 2
    assert support["source_episodes"]["support"] == 2
    assert len(support["episode_details"]) == 2
    assert sum(detail["touches"] for detail in support["episode_details"]) == support["touches"]


def test_broken_support_levels_are_penalized_after_decisive_break():
    result = compute_support_resistance_levels(
        _broken_support_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_pct=0.004,
        min_touches=1,
        max_levels=4,
        reaction_bars=3,
        adx_period=5,
    )

    broken_support = next(level for level in result["levels"] if level["status"] == "broken_support")
    assert broken_support["dominant_source"] == "support"
    assert broken_support["type"] == "resistance"
    assert broken_support["breakout_analysis"]["decisive_break_count"] >= 1
    assert broken_support["breakout_analysis"]["avg_breach_atr"] is not None
    assert broken_support["score_breakdown"]["breakout_penalty"] > 0.0
    assert broken_support["score_breakdown"]["total"] < broken_support["score_breakdown"]["base"]


def test_role_reversal_levels_gain_bonus_after_break_and_retest():
    result = compute_support_resistance_levels(
        _role_reversal_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_pct=0.004,
        min_touches=1,
        max_levels=4,
        reaction_bars=3,
        adx_period=5,
    )

    role_reversed = next(level for level in result["levels"] if level["status"] == "role_reversed_support")
    assert role_reversed["dominant_source"] == "resistance"
    assert role_reversed["type"] == "support"
    assert role_reversed["breakout_analysis"]["decisive_break_count"] >= 1
    assert role_reversed["breakout_analysis"]["role_reversal_count"] >= 1
    assert role_reversed["score_breakdown"]["breakout_penalty"] > 0.0
    assert role_reversed["score_breakdown"]["role_reversal_bonus"] > 0.0
    assert role_reversed["score_breakdown"]["total"] > (
        role_reversed["score_breakdown"]["base"] - role_reversed["score_breakdown"]["breakout_penalty"]
    )


def test_merge_support_resistance_results_combines_multiple_timeframes():
    h1 = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_pct=0.005,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
    )
    h4 = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H4",
        limit=200,
        tolerance_pct=0.005,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
    )

    merged = merge_support_resistance_results(
        [h1, h4],
        symbol="EURUSD",
        timeframe="auto",
        limit=200,
        tolerance_pct=0.005,
        min_touches=2,
        max_levels=4,
        reaction_bars=4,
    )

    assert merged["timeframe"] == "auto"
    assert merged["mode"] == "auto"
    assert merged["timeframes_analyzed"] == ["H1", "H4"]
    assert len(merged["supports"]) == 1
    assert len(merged["resistances"]) == 1
    merged_support = merged["supports"][0]
    merged_resistance = merged["resistances"][0]
    assert merged_support["source_timeframes"] == ["H1", "H4"]
    assert merged_resistance["source_timeframes"] == ["H1", "H4"]
    assert merged_support["merge_details"]["cross_timeframe_dedupe_count"] == 1
    assert merged_resistance["merge_details"]["cross_timeframe_dedupe_count"] == 1
    assert merged_support["episodes"] == 2
    assert merged_support["score"] < (h1["supports"][0]["score"] + h4["supports"][0]["score"])
    assert merged_support["score"] > max(h1["supports"][0]["score"], h4["supports"][0]["score"])
    assert merged_support["score_breakdown"]["mtf_confirmation_bonus"] > 0.0
    assert merged_support["timeframe_contributions"][0]["merge_mode"] == "full"
    assert merged_support["timeframe_contributions"][1]["merge_mode"] == "deduped"
    assert merged["fibonacci"]["mode"] == "auto"
    assert merged["fibonacci"]["selected_timeframe"] == "H4"
    assert merged["fibonacci"]["available_timeframes"] == ["H1", "H4"]
    assert merged["fibonacci"]["swing"]["contains_current_price"] is True
    assert merged["fibonacci"]["nearest"]["support"]["type"] == "support"
