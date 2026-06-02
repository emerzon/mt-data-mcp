from __future__ import annotations

import mtdata.core.market_snapshot as snapshot_mod


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
