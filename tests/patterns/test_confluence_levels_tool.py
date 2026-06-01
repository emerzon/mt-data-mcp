from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np


def _make_rate(open_=1.08, high=1.09, low=1.08, close=1.085, time_=1_700_000_000.0):
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "time": time_,
        "tick_volume": 100,
        "spread": 10,
        "real_volume": 0,
    }


def _make_symbol_info():
    info = MagicMock()
    info.digits = 5
    info.point = 0.00001
    info.trade_tick_size = 0.00001
    return info


def _make_tick():
    tick = MagicMock()
    tick.time = 0
    tick.bid = 1.0851
    tick.ask = 1.0853
    tick.last = 1.0852
    return tick


def _get_confluence_fn():
    from mtdata.core.pivot import confluence_levels

    raw = confluence_levels
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    return raw


@contextmanager
def _mock_symbol_guard():
    @contextmanager
    def _guard(symbol):
        yield None, _make_symbol_info()

    with patch("mtdata.core.pivot._symbol_ready_guard", _guard):
        yield


def test_confluence_levels_tool_combines_pivot_sr_and_fibonacci():
    fn = _get_confluence_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    sr_payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "auto",
        "mode": "auto",
        "timeframes_analyzed": ["M15", "H1", "H4", "D1"],
        "current_price": 1.0852,
        "levels": [
            {
                "type": "resistance",
                "value": 1.0853,
                "score": 5.0,
                "touches": 3,
                "source_timeframes": ["H1"],
            }
        ],
        "fibonacci": {
            "selection_rule": "test",
            "levels": [
                {
                    "label": "61.8%",
                    "ratio": 0.618,
                    "kind": "retracement",
                    "value": 1.0848,
                },
                {
                    "label": "127.2%",
                    "ratio": 1.272,
                    "kind": "extension",
                    "value": 1.0851,
                    "projection": "upside",
                },
            ],
        },
    }
    rates = np.array([_make_rate(time_=100.0), _make_rate(time_=200.0)])

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot.TIMEFRAME_MAP", {"D1": 1}), \
         patch("mtdata.core.pivot.TIMEFRAME_SECONDS", {"D1": 86400}), \
         _mock_symbol_guard(), \
         patch("mtdata.core.pivot.mt5.symbol_info_tick", return_value=_make_tick()), \
         patch("mtdata.core.pivot._mt5_copy_rates_from", return_value=rates), \
         patch("mtdata.core.pivot.compute_support_resistance_payload", return_value=sr_payload) as mock_sr:
        result = fn(
            "EURUSD",
            pivot_timeframe="D1",
            sr_timeframe="auto",
            tolerance_pct=0.001,
            pivot_method="classic",
            max_distance_pct=1.0,
            detail="standard",
        )

    assert result["success"] is True
    assert result["detail"] == "standard"
    assert result["pivot_timeframe"] == "D1"
    assert result["sr_timeframe"] == "auto"
    assert result["levels"]
    top = result["levels"][0]
    assert "pivot_formula" in top["source_families"]
    assert "touch_derived" in top["source_families"]
    assert "swing_fibonacci" in top["source_families"]
    assert mock_sr.call_args.kwargs["timeframe"] == "auto"
    assert mock_sr.call_args.kwargs["max_levels"] == 8


def test_confluence_levels_tool_rejects_invalid_pivot_method():
    fn = _get_confluence_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway):
        result = fn("EURUSD", pivot_method="quarterly")

    assert result["error"] == (
        "Invalid pivot method: quarterly. "
        "Valid methods: classic, fibonacci, camarilla, woodie, demark"
    )
