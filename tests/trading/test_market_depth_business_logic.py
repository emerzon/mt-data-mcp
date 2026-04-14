from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import mtdata.core.market_depth as market_depth_mod
from mtdata.core._mcp_tools import get_tool_functions
from mtdata.core.market_depth import market_depth_fetch, market_ticker
from mtdata.utils.mt5 import MT5ConnectionError


def _raw_market_depth_fetch(symbol: str, spread: bool = False, compact: bool = False):
    raw = getattr(market_depth_fetch, "__wrapped__", market_depth_fetch)
    return raw(symbol, spread=spread, compact=compact)


def _raw_market_ticker(symbol: str):
    return market_ticker.__wrapped__(symbol)


@pytest.fixture(autouse=True)
def _enable_market_depth(monkeypatch) -> None:
    monkeypatch.setenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", "1")


def test_market_depth_tick_fallback_includes_price_display() -> None:
    tick = SimpleNamespace(
        bid=65601.0,
        ask=65601.5,
        last=65601.0,
        volume=12,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    assert out["type"] == "tick_data"
    assert out["capabilities"]["dom_available"] is False
    assert out["capabilities"]["depth_source"] == "symbol_info_tick"
    assert out["data"]["recommended_alternative"] == "market_ticker"
    assert out["price_precision"] == 2
    assert out["data"]["bid"] == 65601.0
    assert out["data"]["ask"] == 65601.5
    assert out["data"]["last"] == 65601.0
    assert isinstance(out.get("query_latency_ms"), float)


def test_market_depth_tick_fallback_hides_zero_last_display() -> None:
    tick = SimpleNamespace(
        bid=65601.0,
        ask=65601.5,
        last=0.0,
        volume=12,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    assert out["data"]["last"] is None
    assert "last_display" not in out["data"]


def test_market_depth_full_depth_includes_price_display() -> None:
    depth = [
        {"price": 65601.0, "volume": 1.0, "volume_real": 1.0, "type": 0},
        {"price": 65602.5, "volume": 2.0, "volume_real": 2.0, "type": 1},
    ]
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_get.return_value = depth

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    assert out["type"] == "full_depth"
    assert out["capabilities"]["dom_available"] is True
    assert out["data"]["depth_levels"]["total"] == 2
    assert out["price_precision"] == 2
    assert out["data"]["buy_orders"][0]["price_display"] == "65601.00"
    assert out["data"]["sell_orders"][0]["price_display"] == "65602.50"


def test_market_depth_subscribes_and_releases_book_snapshot() -> None:
    depth = [
        {"price": 65601.0, "volume": 1.0, "volume_real": 1.0, "type": 0},
        {"price": 65602.5, "volume": 2.0, "volume_real": 2.0, "type": 1},
    ]
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_add.return_value = True
        mt5.market_book_get.return_value = depth

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    mt5.market_book_add.assert_called_once_with("BTCUSD")
    mt5.market_book_get.assert_called_once_with("BTCUSD")
    mt5.market_book_release.assert_called_once_with("BTCUSD")


def test_market_depth_releases_book_after_empty_snapshot() -> None:
    tick = SimpleNamespace(
        bid=65601.0,
        ask=65601.5,
        last=65601.0,
        volume=12,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2)
        mt5.market_book_add.return_value = True
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_depth_fetch("BTCUSD")

    assert out["success"] is True
    assert out["type"] == "tick_data"
    mt5.market_book_add.assert_called_once_with("BTCUSD")
    mt5.market_book_release.assert_called_once_with("BTCUSD")


def test_market_depth_tick_fallback_includes_spread_metrics_when_requested() -> None:
    tick = SimpleNamespace(
        bid=100.0,
        ask=101.0,
        last=100.5,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
        )
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_depth_fetch("BTCUSD", spread=True)

    assert out["success"] is True
    assert out["data"]["spread"] == 1.0
    assert out["data"]["spread_points"] == 100.0
    assert abs(out["data"]["spread_pct"] - (100.0 / 100.5)) < 1e-12
    assert out["data"]["spread_usd"] == 100.0
    assert out["capabilities"]["spread_overlay_applied"] is True


def test_market_depth_compact_mode_fails_fast_without_dom() -> None:
    tick = SimpleNamespace(
        bid=100.0,
        ask=101.0,
        last=100.5,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(digits=2, point=0.01)
        mt5.market_book_get.return_value = []
        mt5.symbol_info_tick.return_value = tick

        out = _raw_market_depth_fetch("BTCUSD", compact=True)

    assert out["error"] == "DOM not available for BTCUSD. Use market_ticker for bid/ask snapshot instead."
    assert out["recommended_alternative"] == "market_ticker"


def test_market_depth_full_depth_includes_spread_metrics_when_requested() -> None:
    depth = [
        {"price": 100.0, "volume": 1.0, "volume_real": 1.0, "type": 0},
        {"price": 101.0, "volume": 2.0, "volume_real": 2.0, "type": 1},
    ]
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
        )
        mt5.market_book_get.return_value = depth
        out = _raw_market_depth_fetch("BTCUSD", spread=True)

    assert out["success"] is True
    assert out["data"]["best_bid"] == 100.0
    assert out["data"]["best_ask"] == 101.0
    assert out["data"]["spread"] == 1.0
    assert out["capabilities"]["spread_overlay_applied"] is True


def test_market_ticker_returns_lightweight_spread_snapshot() -> None:
    tick = SimpleNamespace(
        bid=200.0,
        ask=201.0,
        last=200.5,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("BTCUSD")

    assert out["success"] is True
    assert out["type"] == "ticker"
    assert out["bid"] == 200.0
    assert out["ask"] == 201.0
    assert out["spread"] == 1.0
    assert out["spread_points"] == 100.0
    assert "spread_display" not in out
    assert out["diagnostics"]["cache_used"] is False
    assert out["diagnostics"]["source"] == "mt5.symbol_info_tick"
    assert isinstance(out["diagnostics"]["query_latency_ms"], float)


def test_market_ticker_rounds_to_symbol_precision() -> None:
    tick = SimpleNamespace(
        bid=1.17581,
        ask=1.1758999999999235,
        last=1.175856,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("EURUSD")

    assert out["bid"] == 1.17581
    assert out["ask"] == 1.1759
    assert out["last"] == 1.17586
    assert out["spread"] == 0.00009
    assert out["spread_points"] == 9.0


def test_market_depth_returns_connection_error_payload() -> None:
    with patch(
        "mtdata.core.market_depth.ensure_mt5_connection_or_raise",
        side_effect=MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running."),
    ):
        out = _raw_market_depth_fetch("BTCUSD")

    assert out == {"error": "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."}


def test_market_depth_returns_env_gate_error_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", raising=False)

    out = _raw_market_depth_fetch("BTCUSD")

    assert out["error"] == (
        "market_depth_fetch is disabled. "
        "Set MTDATA_ENABLE_MARKET_DEPTH_FETCH=1 to enable it."
    )
    assert out["recommended_alternative"] == "market_ticker"


def test_market_depth_tool_not_registered_when_env_disabled(monkeypatch) -> None:
    monkeypatch.delenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", raising=False)
    reloaded = importlib.reload(market_depth_mod)
    try:
        assert "market_depth_fetch" not in get_tool_functions()
    finally:
        monkeypatch.setenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", "1")
        importlib.reload(reloaded)


def test_market_ticker_logs_finish_event(caplog) -> None:
    tick = SimpleNamespace(
        bid=200.0,
        ask=201.0,
        last=200.5,
        volume=5,
        time=1700000000,
    )
    with patch("mtdata.core.market_depth.mt5") as mt5, patch(
        "mtdata.core.market_depth._use_client_tz", return_value=False
    ), caplog.at_level("INFO", logger="mtdata.core.market_depth"):
        mt5.symbol_select.return_value = True
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
        )
        mt5.symbol_info_tick.return_value = tick
        out = _raw_market_ticker("BTCUSD")

    assert out["success"] is True
    assert any(
        "event=finish operation=market_ticker success=True" in record.message
        for record in caplog.records
    )


def test_market_ticker_rewrites_invalid_symbol_selection_error() -> None:
    with patch("mtdata.core.market_depth.mt5") as mt5:
        mt5.symbol_select.return_value = False
        mt5.last_error.return_value = (-1, "Terminal: Call failed")

        out = _raw_market_ticker("FAKESYMBOL")

    assert out["error"] == "Symbol 'FAKESYMBOL' was not found or is not available in MT5."
