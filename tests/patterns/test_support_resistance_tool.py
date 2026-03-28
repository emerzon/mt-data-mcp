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
    assert "supports" not in result
    assert "resistances" not in result


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

    assert len(result["supports"]) == 1
    assert len(result["resistances"]) == 1
    assert result["supports"][0]["type"] == "support"
    assert result["resistances"][0]["type"] == "resistance"


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
