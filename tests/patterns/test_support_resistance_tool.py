from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from mtdata.utils.mt5 import MT5ConnectionError


def _get_support_resistance_fn():
    from mtdata.core.pivot import support_resistance_levels

    raw = support_resistance_levels
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    return raw


def _frame() -> pd.DataFrame:
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


def test_support_resistance_tool_returns_weighted_levels():
    fn = _get_support_resistance_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    with patch("mtdata.core.pivot.get_mt5_gateway", return_value=gateway) as mock_gateway, \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()) as mock_fetch:
        result = fn(
            "EURUSD",
            timeframe="H1",
            limit=200,
            tolerance_pct=0.005,
            min_touches=2,
            max_levels=3,
            reaction_bars=4,
        )

    mock_gateway.assert_called_once()
    mock_fetch.assert_called_once_with(symbol="EURUSD", timeframe="H1", need=200)
    assert result["success"] is True
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert result["method"] == "weighted_retests"
    assert len(result["levels"]) == 2
    assert result["level_counts"] == {"support": 1, "resistance": 1, "total": 2}
    assert result["nearest"]["support"]["type"] == "support"
    assert result["nearest"]["resistance"]["type"] == "resistance"
    assert "last_touch" in result["nearest"]["support"]
    assert "zone_low" in result["nearest"]["support"]
    assert "zone_high" in result["nearest"]["support"]
    assert "strength_percentile" in result["nearest"]["support"]
    assert "strength_score_normalized" in result["nearest"]["support"]
    assert result["fibonacci"]["swing"]["direction"] == "up"
    assert len(result["fibonacci"]["levels"]) == 7
    assert result["fibonacci"]["fib_grid_coverage"] == "both_sides"
    assert result["fibonacci"]["nearest"]["support"]["type"] == "support"
    assert "supports" not in result
    assert "resistances" not in result


def test_support_resistance_tool_compact_preserves_zone_overlap_and_fib_grid_metadata():
    fn = _get_support_resistance_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    payload = {
        "success": True,
        "symbol": "USDJPY",
        "timeframe": "H1",
        "mode": "single",
        "current_price": 159.4,
        "supports": [{"type": "support", "value": 159.22, "zone_low": 158.891, "zone_high": 159.543}],
        "resistances": [{"type": "resistance", "value": 159.69, "zone_low": 159.387, "zone_high": 159.933}],
        "levels": [
            {"type": "support", "value": 159.22, "zone_low": 158.891, "zone_high": 159.543},
            {"type": "resistance", "value": 159.69, "zone_low": 159.387, "zone_high": 159.933},
        ],
        "fibonacci": {
            "mode": "single",
            "timeframe": "H1",
            "fib_grid_coverage": "support_only",
            "fib_grid_counts": {"support": 7, "resistance": 0, "total": 7},
        },
        "zone_overlap": {
            "support_value": 159.22,
            "resistance_value": 159.69,
            "overlap_low": 159.387,
            "overlap_high": 159.543,
            "overlap_width": 0.156,
            "current_price_in_overlap": True,
        },
        "warnings": [
            {"code": "overlapping_nearest_zones"},
            {"code": "fibonacci_grid_support_only"},
        ],
    }

    with patch("mtdata.core.pivot.get_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot.compute_support_resistance_payload", return_value=payload):
        result = fn("USDJPY", timeframe="H1", detail="compact")

    assert result["detail"] == "compact"
    assert result["zone_overlap"]["current_price_in_overlap"] is True
    assert result["zone_overlap"]["overlap_width"] == 0.156
    assert result["fibonacci"]["fib_grid_coverage"] == "support_only"
    assert result["fibonacci"]["fib_grid_counts"] == {"support": 7, "resistance": 0, "total": 7}
    assert {warning["code"] for warning in result["warnings"]} == {
        "overlapping_nearest_zones",
        "fibonacci_grid_support_only",
    }


def test_support_resistance_tool_compact_exposes_coverage_gap_metadata_with_distance_filter():
    fn = _get_support_resistance_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    with patch("mtdata.core.pivot.get_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()):
        result = fn(
            "EURUSD",
            timeframe="H1",
            limit=200,
            tolerance_pct=0.005,
            min_touches=1,
            max_levels=4,
            max_distance_pct=0.04,
            reaction_bars=4,
    )

    assert result["max_distance_pct"] == 0.04
    assert "levels" not in result
    assert result["coverage_gaps"]["support"]["beyond_max_distance_filter"] is True
    assert result["coverage_gaps"]["resistance"]["beyond_max_distance_filter"] is True


def test_support_resistance_tool_compact_exposes_volume_metadata_when_enabled():
    fn = _get_support_resistance_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    frame = _frame().copy()
    frame["tick_volume"] = [100.0] * len(frame)
    frame.loc[2, "tick_volume"] = 40.0
    frame.loc[12, "tick_volume"] = 420.0
    frame["real_volume"] = 0.0
    with patch("mtdata.core.pivot.get_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=frame):
        result = fn(
            "EURUSD",
            timeframe="H1",
            limit=200,
            tolerance_pct=0.005,
            min_touches=1,
            max_levels=4,
            volume_weighting="auto",
            reaction_bars=4,
        )

    assert result["volume_weighting"] == "auto"
    assert result["volume_source"] == "tick_volume"
    assert result["nearest"]["support"]["avg_test_volume_ratio"] is not None
    assert result["nearest"]["support"]["volume_source"] == "tick_volume"


def test_support_resistance_tool_standard_detail_keeps_actionable_lists_without_full_diagnostics():
    fn = _get_support_resistance_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    with patch("mtdata.core.pivot.get_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()):
        result = fn(
            "EURUSD",
            timeframe="H1",
            limit=200,
            tolerance_pct=0.005,
            min_touches=2,
            max_levels=3,
            reaction_bars=4,
            detail="standard",
        )

    assert result["detail"] == "standard"
    assert len(result["supports"]) == 1
    assert len(result["resistances"]) == 1
    assert len(result["levels"]) == 2
    assert result["supports"][0]["type"] == "support"
    assert result["resistances"][0]["type"] == "resistance"
    assert result["nearest"]["support"]["type"] == "support"
    assert "score_breakdown" not in result["supports"][0]
    assert "source_tests" not in result["supports"][0]
    assert result["fibonacci"]["nearest"]["support"]["type"] == "support"


def test_support_resistance_tool_defaults_to_auto_mode():
    fn = _get_support_resistance_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()

    def _fetch(symbol: str, timeframe: str, need: int, as_of=None):
        assert symbol == "EURUSD"
        assert need == 200
        assert as_of is None
        frame = _frame().copy()
        frame["close"] = frame["close"] + (0.1 if timeframe == "D1" else 0.0)
        return frame

    with patch("mtdata.core.pivot.get_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", side_effect=_fetch) as mock_fetch:
        result = fn("EURUSD", limit=200, tolerance_pct=0.005, min_touches=2, max_levels=3, reaction_bars=4)

    assert mock_fetch.call_count == 4
    assert result["success"] is True
    assert result["timeframe"] == "auto"
    assert result["mode"] == "auto"
    assert result["timeframes_analyzed"] == ["M15", "H1", "H4", "D1"]
    assert result["level_counts"] == {"support": 1, "resistance": 1, "total": 2}
    assert result["nearest"]["support"]["source_timeframes"] == ["M15", "H1", "H4", "D1"]
    assert "last_touch" in result["nearest"]["support"]
    assert "strength_percentile" in result["nearest"]["support"]
    assert result["fibonacci"]["mode"] == "auto"
    assert result["fibonacci"]["selected_timeframe"] == "D1"
    assert result["fibonacci"]["available_timeframes"] == ["M15", "H1", "H4", "D1"]
    assert result["fibonacci"]["selection_summary"]["timeframe_candidate_count"] == 4
    assert result["fibonacci"]["timeframe_selection_candidates"][0]["selected"] is True
    assert "supports" not in result
    assert "resistances" not in result


def test_support_resistance_tool_full_detail_retains_support_and_resistance_lists():
    fn = _get_support_resistance_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    with patch("mtdata.core.pivot.get_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()):
        result = fn(
            "EURUSD",
            timeframe="H1",
            limit=200,
            tolerance_pct=0.005,
            min_touches=2,
            max_levels=3,
            reaction_bars=4,
            detail="full",
        )

    assert result["detail"] == "full"
    assert len(result["supports"]) == 1
    assert len(result["resistances"]) == 1
    assert result["supports"][0]["type"] == "support"
    assert result["resistances"][0]["type"] == "resistance"
    assert result["fibonacci"]["timeframe"] == "H1"
    assert len(result["fibonacci"]["retracements"]) == 5
    assert len(result["fibonacci"]["extensions"]) == 2


def test_support_resistance_tool_wraps_fetch_errors():
    fn = _get_support_resistance_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    with patch("mtdata.core.pivot.get_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", side_effect=RuntimeError("boom")):
        result = fn("EURUSD", timeframe="H1")

    assert "error" in result
    assert "boom" in result["error"]


def test_support_resistance_tool_wraps_connection_errors():
    fn = _get_support_resistance_fn()
    gateway = type(
        "Gateway",
        (),
        {"ensure_connection": lambda self: (_ for _ in ()).throw(MT5ConnectionError("No IPC connection"))},
    )()
    with patch("mtdata.core.pivot.get_mt5_gateway", return_value=gateway):
        result = fn("EURUSD", timeframe="H1")

    assert "error" in result
    assert "No IPC connection" in result["error"]
