from __future__ import annotations

import mtdata.core.market_snapshot as snapshot_mod


def _raw_market_snapshot(**kwargs):
    return snapshot_mod.market_snapshot.__wrapped__(**kwargs)


def test_market_snapshot_marks_invalid_symbol_failure(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {
                "success": False,
                "error": "Symbol 'NOTREAL' was not found or is not available in MT5.",
            }
        if name == "levels":
            return {
                "error": "Error computing support/resistance levels: Symbol 'NOTREAL' was not found in MT5.",
            }
        return {"success": True, "n_patterns": 0, "highlights": []}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="NOTREAL")

    assert result["success"] is False
    assert result["failure_reason"] == "invalid_symbol"
    assert result["failed_sections"] == ["quote", "levels"]
    assert "NOTREAL" in result["error"]
    assert result["summary"] == "NOTREAL snapshot; failed=quote,levels."


def test_market_snapshot_marks_partial_section_failure(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "levels":
            return {"error": "levels unavailable"}
        if name == "quote":
            return {"success": True, "symbol": symbol, "mid": 1.1}
        return {"success": True, "n_patterns": 0, "highlights": []}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD")

    assert result["success"] is True
    assert result["partial_failure"] is True
    assert result["failed_sections"] == ["levels"]
    assert "error" not in result
    assert result["summary"] == "EURUSD snapshot; mid=1.1; failed=levels."


def test_market_snapshot_summary_detail_returns_lean_snapshot(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {
                "success": True,
                "symbol": symbol,
                "bid": 1.1001,
                "ask": 1.1003,
                "mid": 1.1002,
                "spread_pips": 2.0,
            }
        if name == "levels":
            return {
                "success": True,
                "scan_window": {"start": "2026-01-01", "end": "2026-01-02"},
                "supports": [{"type": "support", "value": 1.098}],
                "resistances": [{"type": "resistance", "value": 1.105}],
                "warnings": [{"code": "overlapping_nearest_zones"}],
            }
        return {
            "success": True,
            "n_patterns": 2,
            "highlights": [{"pattern": "engulfing", "bias": "bullish"}],
            "note": "extra pattern guidance",
        }

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="summary")

    assert result["snapshot"] == {
        "bid": 1.1001,
        "ask": 1.1003,
        "mid": 1.1002,
        "spread_pips": 2.0,
        "nearest_support": 1.098,
        "nearest_resistance": 1.105,
        "pattern_bias": "bullish",
    }
    assert "quote" not in result
    assert "levels" not in result
    assert "patterns" not in result
    assert result["summary"] == "EURUSD snapshot; mid=1.1002; spread_pips=2.0."


def test_market_snapshot_compact_defaults_to_lean_snapshot(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {
                "success": True,
                "symbol": symbol,
                "bid": 1.1001,
                "ask": 1.1003,
                "mid": 1.1002,
                "spread_points": 2.0,
            }
        if name == "levels":
            return {
                "success": True,
                "scan_window": {"start": "2026-01-01", "end": "2026-01-02"},
                "supports": [{"type": "support", "value": 1.098}],
                "resistances": [{"type": "resistance", "value": 1.105}],
            }
        return {
            "success": True,
            "n_patterns": 2,
            "highlights": [{"pattern": "engulfing", "bias": "bullish"}],
            "calibration": {"minimum_confidence": 0.3},
        }

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="compact")

    assert result["snapshot"] == {
        "bid": 1.1001,
        "ask": 1.1003,
        "mid": 1.1002,
        "spread_points": 2.0,
        "nearest_support": 1.098,
        "nearest_resistance": 1.105,
        "pattern_bias": "bullish",
    }
    assert "quote" not in result
    assert "levels" not in result
    assert "patterns" not in result


def test_snapshot_patterns_section_requests_recent_candlestick_triggers(monkeypatch):
    captured = {}

    def fake_call_tool(func, **kwargs):
        captured["func_name"] = getattr(func, "__name__", "")
        captured["kwargs"] = kwargs
        return {"success": True, "highlights": []}

    monkeypatch.setattr(snapshot_mod, "call_tool_sync_structured", fake_call_tool)

    result = snapshot_mod._call_section(
        "patterns",
        symbol="EURUSD",
        timeframe="H1",
        horizon=8,
        detail="compact",
    )

    assert result == {"success": True, "highlights": []}
    assert captured["func_name"] == "patterns_detect"
    assert captured["kwargs"]["limit"] == 150
    assert captured["kwargs"]["top_k"] == 3
    assert captured["kwargs"]["last_n_bars"] == 3
    assert captured["kwargs"]["detail"] == "summary"
