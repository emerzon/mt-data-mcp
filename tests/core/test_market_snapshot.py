from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import mtdata.core.market_snapshot as snapshot_mod


def _raw_market_snapshot(**kwargs):
    return snapshot_mod.market_snapshot.__wrapped__(**kwargs)


def test_market_snapshot_help_discloses_builtin_section_methods():
    doc = snapshot_mod.market_snapshot.__doc__ or ""

    assert "regime (opt-in): HMM only" in doc
    assert "forecast (opt-in): Theta only" in doc
    assert "horizon`` applies here only" in doc
    assert "selectable analysis sections" in doc
    assert "detail`` mainly shapes" in doc
    assert "assembled_at" in doc
    assert "quote_as_of" in doc


def test_market_snapshot_quote_compaction_preserves_epoch_as_secondary_field():
    quote = snapshot_mod._compact_quote(
        {
            "success": True,
            "symbol": "EURUSD",
            "bid": 1.1,
            "ask": 1.1002,
            "time": 1700000000,
            "time_display": "2023-11-14 22:13 UTC",
            "meta": {"tool": "market_ticker"},
        }
    )

    assert quote["time"] == "2023-11-14 22:13 UTC"
    assert quote["time_epoch"] == 1700000000
    assert "time_display" not in quote
    assert "meta" not in quote


def test_market_snapshot_quote_compaction_formats_epoch_without_display():
    quote = snapshot_mod._compact_quote(
        {
            "success": True,
            "symbol": "EURUSD",
            "bid": 1.1,
            "ask": 1.1002,
            "time": 1700000000,
        }
    )

    assert quote["time"] == "2023-11-14T22:13:20Z"
    assert quote["time_epoch"] == 1700000000


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
        "pattern_count": 2,
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
                "freshness_state": "live",
                "data_age_seconds": 4.0,
                "usable_for_live_trading": True,
                "live_max_age_seconds": 30.0,
            }
        if name == "status":
            return {
                "success": True,
                "status": "open",
                "is_tradable": True,
                "is_tradable_confidence": "heuristic",
                "can_open_new_positions": True,
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

    assert "sections" not in result
    assert result["sections_requested"] == ["quote", "status", "levels", "patterns"]
    assert result["sections_summarized"] == ["quote", "status", "levels", "patterns"]
    assert result["snapshot"] == {
        "bid": 1.1001,
        "ask": 1.1003,
        "mid": 1.1002,
        "spread_points": 2.0,
        "freshness_state": "live",
        "data_age_seconds": 4.0,
        "execution": {
            "usable_for_live_trading": True,
            "live_max_age_seconds": 30.0,
            "status": "open",
            "is_tradable": True,
            "is_tradable_confidence": "heuristic",
            "can_open_new_positions": True,
        },
        "nearest_support": 1.098,
        "nearest_resistance": 1.105,
        "pattern_count": 2,
    }
    assert "quote" not in result
    assert "levels" not in result
    assert "patterns" not in result


def test_market_snapshot_compact_keeps_requested_regime_and_forecast(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "regime":
            return {"success": True, "current_regime": "trend_up", "confidence": 0.8}
        if name == "forecast":
            return {"success": True, "method": "theta", "forecast": [1.1, 1.2]}
        return {"success": True}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(
        symbol="EURUSD", sections="regime,forecast", detail="compact"
    )

    assert result["snapshot"]["regime"] == {
        "current_regime": "trend_up",
        "confidence": 0.8,
    }
    assert result["snapshot"]["forecast"] == {
        "method": "theta",
        "forecast": [1.1, 1.2],
    }


def test_market_snapshot_nearest_levels_respect_quote_side(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {
                "success": True,
                "symbol": symbol,
                "bid": 1.1000,
                "ask": 1.1002,
                "mid": 1.1001,
            }
        if name == "levels":
            return {
                "success": True,
                "supports": [
                    {"type": "support", "value": 1.10015},
                    {"type": "support", "value": 1.0999},
                ],
                "resistances": [
                    {"type": "resistance", "value": 1.1},
                    {"type": "resistance", "value": 1.1004},
                ],
            }
        return {"success": True, "n_patterns": 0, "highlights": []}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="compact")

    assert result["snapshot"]["nearest_support"] == 1.0999
    assert result["snapshot"]["nearest_resistance"] == 1.1004


def test_market_snapshot_exposes_quote_and_assembly_timestamps(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {
                "success": True,
                "symbol": symbol,
                "mid": 1.1002,
                "time": "2023-11-14 22:13 UTC",
                "time_epoch": 1_700_000_000,
            }
        if name == "levels":
            return {"success": True, "supports": [], "resistances": []}
        return {"success": True, "n_patterns": 0, "highlights": []}

    fixed_now = datetime(2026, 6, 15, 19, 34, 8, tzinfo=timezone.utc)
    fake_datetime = MagicMock()
    fake_datetime.now.return_value = fixed_now
    fake_datetime.fromtimestamp.side_effect = datetime.fromtimestamp

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    with patch.object(snapshot_mod, "datetime", fake_datetime):
        result = _raw_market_snapshot(symbol="EURUSD", detail="compact")

    assert result["as_of"] == "2023-11-14T22:13:20Z"
    assert result["quote_as_of"] == "2023-11-14T22:13:20Z"
    assert result["assembled_at"] == "2026-06-15T19:34:08Z"


def test_market_snapshot_standard_strips_nested_request_echoes(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "levels":
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "detail": "compact",
                "mode": "single",
                "levels": [{"value": 1.1}],
            }
        if name == "patterns":
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "mode": "candlestick",
                "is_signal": False,
                "usage": "information_only",
                "calibration": {"note": "static guidance"},
                "highlights": [],
            }
        return {"success": True, "symbol": symbol, "mid": 1.1}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="standard")

    assert result["symbol"] == "EURUSD"
    assert result["sections_embedded"] == ["quote", "status", "levels", "patterns"]
    assert "summary" not in result
    assert result["levels"] == {
        "success": True,
        "levels": [{"value": 1.1}],
    }
    assert result["patterns"] == {
        "success": True,
        "is_signal": False,
        "usage": "information_only",
        "highlights": [],
    }


def test_market_snapshot_qualifies_uncertain_pattern_bias(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {"success": True, "symbol": symbol, "mid": 1.1}
        if name == "levels":
            return {"success": True, "supports": [], "resistances": []}
        return {
            "success": True,
            "n_patterns": 4,
            "bias": "bearish",
            "pattern_status": "uncertain",
            "pattern_confidence": 0.225,
            "conflict": "both_bullish_and_bearish_patterns_present",
            "is_signal": False,
            "usage": "information_only",
            "applied_last_n_bars": 3,
        }

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="compact")

    snapshot = result["snapshot"]
    assert "pattern_bias" not in snapshot
    assert snapshot["pattern_status"] == "uncertain"
    assert snapshot["pattern_confidence"] == 0.225
    assert snapshot["pattern_conflict"] == "both_bullish_and_bearish_patterns_present"
    assert snapshot["pattern_count"] == 4
    assert snapshot["pattern_is_signal"] is False
    assert snapshot["pattern_usage"] == "information_only"
    assert snapshot["pattern_window_bars"] == 3


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
