"""Tests for the symbols_top_markets MT5 market scanner tool."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _get_symbols_top_markets():
    from mtdata.core.symbols import symbols_top_markets

    raw = _unwrap(symbols_top_markets)

    def _call(*args, **kwargs):
        with patch("mtdata.core.symbols.ensure_mt5_connection_or_raise", return_value=None):
            return raw(*args, **kwargs)

    return _call


def _get_market_scan():
    from mtdata.core.symbols import market_scan

    raw = _unwrap(market_scan)

    def _call(*args, **kwargs):
        with patch("mtdata.core.symbols.ensure_mt5_connection_or_raise", return_value=None):
            return raw(*args, **kwargs)

    return _call


def _make_symbol(
    name: str,
    *,
    path: str = "Forex\\Majors",
    description: str = "Market",
    visible: bool = True,
    trade_mode: int = 1,
    point: float = 0.0001,
    trade_tick_size: float = 0.0001,
    trade_tick_value: float = 10.0,
):
    return SimpleNamespace(
        name=name,
        path=path,
        description=description,
        visible=visible,
        trade_mode=trade_mode,
        point=point,
        trade_tick_size=trade_tick_size,
        trade_tick_value=trade_tick_value,
    )


def _make_tick(*, bid: float, ask: float):
    return SimpleNamespace(bid=bid, ask=ask)


def _make_bars(closes, *, tick_volume: int = 100):
    closes = list(closes)
    bars = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index > 0 else close
        bars.append(
            {
                "time": 1700000000.0 + (index * 3600.0),
                "open": open_price,
                "close": close,
                "tick_volume": tick_volume + index,
                "real_volume": 0,
            }
        )
    return bars


@contextmanager
def _ready_guard_ok(symbol: str, info_before=None):
    yield None, info_before


@pytest.fixture(autouse=True)
def _set_disabled_trade_mode(monkeypatch):
    import mtdata.core.symbols as symbols_mod

    monkeypatch.setattr(symbols_mod.mt5, "SYMBOL_TRADE_MODE_DISABLED", 0, raising=False)


class TestSymbolsTopMarkets:
    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_spread_ranks_lowest_first_visible_default(self, mock_symbols_get, mock_tick, mock_group):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("XAUUSD", description="Gold", point=0.01, trade_tick_size=0.01, trade_tick_value=1.0),
            _make_symbol("HIDDEN", visible=False),
            _make_symbol("DISABLED", trade_mode=0),
        ]
        tick_map = {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "XAUUSD": _make_tick(bid=2000.0, ask=2002.0),
        }
        mock_tick.side_effect = lambda symbol: tick_map.get(symbol)

        fn = _get_symbols_top_markets()
        result = fn(rank_by="spread", limit=5)

        assert result["success"] is True
        assert result["ranking"] == "lowest_spread"
        assert result["universe"] == "visible"
        assert result["scanned_symbols"] == 2
        assert result["evaluated_symbols"] == 2
        assert result["timeframe_requested"] == "H1"
        assert result["timeframe_used"] is None
        assert [row["symbol"] for row in result["data"]] == ["EURUSD", "XAUUSD"]
        assert all(row["pricing_basis"] == "per_1_lot_estimate" for row in result["data"])

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_all_returns_all_leaderboards(self, mock_symbols_get, mock_tick, mock_rates, mock_group):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("GBPUSD", description="Pound"),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "GBPUSD": _make_tick(bid=1.3000, ask=1.3004),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": [{"time": 1700000000.0, "open": 1.1000, "close": 1.1010, "tick_volume": 100, "real_volume": 0}],
            "GBPUSD": [{"time": 1700000000.0, "open": 1.3000, "close": 1.3300, "tick_volume": 50, "real_volume": 0}],
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="all", limit=5, timeframe="H1")

        assert result["success"] is True
        assert result["results"]["lowest_spread"]["data"][0]["symbol"] == "EURUSD"
        assert result["results"]["highest_volume"]["data"][0]["symbol"] == "EURUSD"
        assert result["results"]["highest_price_change"]["data"][0]["symbol"] == "GBPUSD"
        assert result["timeframe_requested"] == "H1"
        assert result["timeframe_used"] == "H1"
        assert result["scan_stats"]["spread"]["evaluated_symbols"] == 2
        assert result["scan_stats"]["volume"]["evaluated_symbols"] == 2
        assert result["scan_stats"]["price_change"]["evaluated_symbols"] == 2

    def test_invalid_rank_by_returns_error(self):
        fn = _get_symbols_top_markets()

        result = fn(rank_by="unknown")

        assert result == {
            "error": "rank_by must be one of: all, spread, volume, price_change."
        }

    def test_invalid_timeframe_returns_error_for_bar_metrics(self):
        fn = _get_symbols_top_markets()

        result = fn(rank_by="volume", timeframe="BAD")

        assert "error" in result
        assert "Invalid timeframe" in result["error"]

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._symbol_ready_guard", side_effect=_ready_guard_ok)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_all_universe_activates_hidden_symbols(
        self,
        mock_symbols_get,
        mock_tick,
        mock_ready_guard,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", visible=True),
            _make_symbol("USDJPY", visible=False),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1002),
            "USDJPY": _make_tick(bid=150.00, ask=150.03),
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="spread", universe="all", limit=5)

        assert result["success"] is True
        assert result["scanned_symbols"] == 2
        assert [row["symbol"] for row in result["data"]] == ["EURUSD", "USDJPY"]
        mock_ready_guard.assert_called_once_with("USDJPY", info_before=mock_symbols_get.return_value[1])

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_volume_reports_skipped_symbols_when_bar_data_missing(self, mock_symbols_get, mock_rates, mock_group):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD"),
            _make_symbol("GBPUSD"),
        ]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": [{"time": 1700000000.0, "open": 1.1000, "close": 1.1010, "tick_volume": 100, "real_volume": 0}],
            "GBPUSD": None,
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="volume", timeframe="H1", limit=5)

        assert result["success"] is True
        assert result["evaluated_symbols"] == 1
        assert result["skipped_symbols"] == 1
        assert result["skipped_examples"][0]["symbol"] == "GBPUSD"


class TestMarketScan:
    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_filters_by_rsi_and_sma(self, mock_symbols_get, mock_tick, mock_rates, mock_group):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("GBPUSD", description="Pound"),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "GBPUSD": _make_tick(bid=1.3000, ask=1.3002),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": _make_bars([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], tick_volume=120),
            "GBPUSD": _make_bars([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], tick_volume=80),
        }[symbol]

        fn = _get_market_scan()
        result = fn(
            timeframe="H1",
            lookback=6,
            rsi_length=3,
            sma_period=3,
            rsi_above=60,
            price_vs_sma="above",
            limit=10,
        )

        assert result["success"] is True
        assert result["matched_symbols"] == 1
        assert result["filtered_out_symbols"] == 1
        assert result["data"][0]["symbol"] == "EURUSD"
        assert result["data"][0]["rsi"] == 100.0
        assert result["data"][0]["sma_value"] == 5.0

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._symbol_ready_guard", side_effect=_ready_guard_ok)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_group_universe_all_activates_hidden_symbols(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_ready_guard,
        mock_group,
    ):
        hidden_symbol = _make_symbol("USDJPY", visible=False)
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", visible=True),
            hidden_symbol,
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1002),
            "USDJPY": _make_tick(bid=150.00, ask=150.03),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": _make_bars([1.0, 1.1, 1.2, 1.3], tick_volume=100),
            "USDJPY": _make_bars([150.0, 150.2, 150.3, 150.4], tick_volume=90),
        }[symbol]

        fn = _get_market_scan()
        result = fn(group="Forex\\Majors", universe="all", lookback=4, min_tick_volume=50)

        assert result["success"] is True
        assert result["scope"] == "group"
        assert result["scanned_symbols"] == 2
        assert result["matched_symbols"] == 2
        mock_ready_guard.assert_called_once_with("USDJPY", info_before=hidden_symbol)

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_returns_no_action_when_no_symbols_match(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", description="Euro")]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1002)
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=20)

        fn = _get_market_scan()
        result = fn(lookback=4, min_price_change_pct=50.0)

        assert result["success"] is True
        assert result["no_action"] is True
        assert result["matched_symbols"] == 0
        assert result["message"] == "No symbols matched the requested market scan filters."

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_rejects_ambiguous_group(self, mock_symbols_get, mock_group):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", path="Forex\\Majors"),
            _make_symbol("AUDCAD", path="Forex\\Minors"),
        ]

        fn = _get_market_scan()
        result = fn(group="Forex")

        assert "error" in result
        assert "Ambiguous group" in result["error"]
