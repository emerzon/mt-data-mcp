from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from mtdata.core.market_depth import market_depth_fetch, market_ticker


def _raw_market_depth_fetch(symbol: str, spread: bool = False):
    # Bypass @mcp.tool and @_auto_connect_wrapper wrappers.
    return market_depth_fetch.__wrapped__.__wrapped__(symbol, spread=spread)


def _raw_market_ticker(symbol: str):
    return market_ticker.__wrapped__.__wrapped__(symbol)


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
    assert out["data"]["bid_display"] == "65601.00"
    assert out["data"]["ask_display"] == "65601.50"
    assert out["data"]["last_display"] == "65601.00"
    assert isinstance(out.get("query_latency_ms"), float)


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
    assert out["spread_display"] == "1.00"
    assert out["diagnostics"]["cache_used"] is False
    assert out["diagnostics"]["source"] == "mt5.symbol_info_tick"
    assert isinstance(out["diagnostics"]["query_latency_ms"], float)
