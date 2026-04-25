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


def _get_select_market_scan_symbols():
    from mtdata.core.symbols import _select_market_scan_symbols

    return _select_market_scan_symbols


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
        assert "universe" not in result
        assert "scanned_symbols" not in result
        assert "evaluated_symbols" not in result
        assert "detail" not in result
        assert "timeframe_requested" not in result
        assert "query_latency_ms" not in result
        assert [row["symbol"] for row in result["data"]] == ["EURUSD", "XAUUSD"]
        assert list(result["data"][0].keys()) == [
            "symbol",
            "group",
            "spread_pct",
            "spread_points",
        ]
        assert "pricing_basis" not in result["data"][0]

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
        result = fn(rank_by="all", limit=5, timeframe="H1", detail="full")

        assert result["success"] is True
        assert result["results"]["lowest_spread"]["data"][0]["symbol"] == "EURUSD"
        assert result["results"]["highest_volume"]["data"][0]["symbol"] == "EURUSD"
        assert result["results"]["highest_price_change"]["data"][0]["symbol"] == "GBPUSD"
        assert result["collection_kind"] == "groups"
        assert result["canonical_source"] == "results"
        assert "groups" not in result
        assert "rows" not in result["results"]["lowest_spread"]
        assert "success" not in result["results"]["lowest_spread"]
        assert "count" not in result["results"]["lowest_spread"]
        assert "success" not in result["results"]["highest_volume"]
        assert "count" not in result["results"]["highest_volume"]
        assert "success" not in result["results"]["highest_price_change"]
        assert "count" not in result["results"]["highest_price_change"]
        assert result["detail"] == "full"
        assert result["timeframe_requested"] == "H1"
        assert result["timeframe_used"] == "H1"
        assert result["scan_stats"]["spread"]["evaluated_symbols"] == 2
        assert result["scan_stats"]["volume"]["evaluated_symbols"] == 2
        assert result["scan_stats"]["price_change"]["evaluated_symbols"] == 2

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_spread_detail_compact_returns_ranking_focused_rows(
        self,
        mock_symbols_get,
        mock_tick,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol(
                "XAUUSD",
                description="Gold",
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
            ),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "XAUUSD": _make_tick(bid=2000.0, ask=2002.0),
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="spread", limit=5, detail="compact")

        assert result["success"] is True
        assert "detail" not in result
        assert list(result["data"][0].keys()) == [
            "symbol",
            "group",
            "spread_pct",
            "spread_points",
        ]
        assert "description" not in result["data"][0]
        assert "pricing_basis" not in result["data"][0]
        assert "collection_kind" not in result
        assert "collection_contract_version" not in result

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_all_detail_compact_applies_compact_rows_to_each_leaderboard(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("GBPUSD", description="Pound"),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "GBPUSD": _make_tick(bid=1.3000, ask=1.3004),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": [
                {
                    "time": 1700000000.0,
                    "open": 1.1000,
                    "close": 1.1010,
                    "tick_volume": 100,
                    "real_volume": 0,
                }
            ],
            "GBPUSD": [
                {
                    "time": 1700000000.0,
                    "open": 1.3000,
                    "close": 1.3300,
                    "tick_volume": 50,
                    "real_volume": 0,
                }
            ],
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="all", limit=5, timeframe="H1", detail="compact")

        assert result["success"] is True
        assert "detail" not in result
        assert "scan_stats" not in result
        assert "query_latency_ms" not in result
        assert list(result["results"]["lowest_spread"]["data"][0].keys()) == [
            "symbol",
            "group",
            "spread_pct",
            "spread_points",
        ]
        assert list(result["results"]["highest_volume"]["data"][0].keys()) == [
            "symbol",
            "group",
            "timeframe",
            "bar_age_hours",
            "data_stale",
            "tick_volume",
            "price_change_pct",
        ]
        assert result["results"]["highest_volume"]["data"][0]["data_stale"] is True
        assert result["results"]["highest_volume"]["data"][0]["bar_age_hours"] > 0
        assert list(result["results"]["highest_price_change"]["data"][0].keys()) == [
            "symbol",
            "group",
            "timeframe",
            "bar_age_hours",
            "data_stale",
            "price_change_pct",
            "tick_volume",
        ]
        assert "success" not in result["results"]["lowest_spread"]
        assert "count" not in result["results"]["highest_volume"]
        assert "collection_kind" not in result
        assert "collection_contract_version" not in result
        assert "collection_kind" not in result["results"]["lowest_spread"]
        assert "collection_contract_version" not in result["results"]["lowest_spread"]

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
        assert "scanned_symbols" not in result
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
        result = fn(rank_by="volume", timeframe="H1", limit=5, detail="full")

        assert result["success"] is True
        assert result["evaluated_symbols"] == 1
        assert result["skipped_symbols"] == 1
        assert result["skipped_examples"][0]["symbol"] == "GBPUSD"

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos", return_value=None)
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_volume_skipped_examples_are_deterministic(self, mock_symbols_get, mock_rates, mock_group):
        mock_symbols_get.return_value = [
            _make_symbol("usdjpy"),
            _make_symbol("EURUSD"),
        ]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="volume", timeframe="H1", limit=5, detail="full")

        assert result["success"] is True
        assert [row["symbol"] for row in result["skipped_examples"]] == ["EURUSD", "usdjpy"]


class TestMarketScan:
    def test_explicit_symbol_selection_preserves_requested_order(self):
        fn = _get_select_market_scan_symbols()

        selected, meta, error = fn(
            [
                _make_symbol("EURUSD"),
                _make_symbol("GBPUSD"),
                _make_symbol("USDJPY"),
            ],
            symbols="USDJPY, eurusd, MISSING",
            group=None,
            universe="visible",
        )

        assert error is None
        assert [symbol.name for symbol in selected] == ["USDJPY", "EURUSD"]
        assert meta["missing_symbols"] == ["MISSING"]

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
            detail="full",
        )

        assert result["success"] is True
        assert result["summary"]["counts"]["matched_symbols"] == 1
        assert result["summary"]["counts"]["filtered_out_symbols"] == 1
        assert result["data"]["table"]["columns"][0] == "symbol"
        assert result["data"]["table"]["row_count"] == 1
        assert result["data"]["table"]["rows"][0]["symbol"] == "EURUSD"
        assert result["data"]["table"]["rows"][0]["rsi"] == 100.0
        assert result["data"]["table"]["rows"][0]["sma_value"] == 5.0
        assert result["collection_kind"] == "table"
        assert result["canonical_source"] == "data.table.rows"
        assert "rows" not in result
        assert result["meta"]["request"]["timeframe"] == "H1"
        assert result["meta"]["request"]["rank_by"] == "abs_price_change_pct"
        assert result["meta"]["stats"]["matched_symbols"] == 1
        assert "matched_symbols" not in result

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_default_compact_detail_omits_redundant_columns(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", description="Euro")]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 2.0, 3.0, 4.0], tick_volume=120)

        fn = _get_market_scan()
        result = fn(timeframe="H1", lookback=4, limit=5)

        assert result["success"] is True
        assert "columns" not in result["data"]["table"]
        assert result["data"]["table"]["row_count"] == 1
        assert result["data"]["table"]["rows"][0]["symbol"] == "EURUSD"
        assert "rows" not in result
        assert result["meta"]["request"]["detail"] == "compact"
        assert "collection_kind" not in result
        assert "collection_contract_version" not in result

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_accepts_rank_by_aliases(
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
        result = fn(lookback=4, rank_by="spread")

        assert result["success"] is True
        assert result["meta"]["request"]["rank_by"] == "spread_pct"
        assert result["meta"]["request"]["rank_by_input"] == "spread"

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
        assert result["meta"]["request"]["scope"] == "group"
        assert result["summary"]["counts"]["scanned_symbols"] == 2
        assert result["summary"]["counts"]["matched_symbols"] == 2
        assert result["meta"]["stats"]["scanned_symbols"] == 2
        mock_ready_guard.assert_called_once_with("USDJPY", info_before=hidden_symbol)

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_symbols_updates_request_meta(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", description="Euro")]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=50)

        fn = _get_market_scan()
        result = fn(symbols="EURUSD", lookback=4)

        assert result["success"] is True
        assert result["meta"]["request"]["symbols_input"] == ["EURUSD"]

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_symbol_alias_filters_single_symbol(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("GBPUSD", description="Pound"),
        ]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=50)

        fn = _get_market_scan()
        result = fn(symbol="EURUSD", lookback=4)

        assert result["success"] is True
        assert result["meta"]["request"]["symbols_input"] == ["EURUSD"]
        assert result["meta"]["request"]["symbol_alias_used"] is True
        assert [row["symbol"] for row in result["data"]["table"]["rows"]] == ["EURUSD"]

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_rejects_symbol_and_symbols_together(self, mock_symbols_get, mock_group):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", description="Euro")]

        fn = _get_market_scan()
        result = fn(symbol="EURUSD", symbols="GBPUSD")

        assert result["success"] is False
        assert result["error"] == "Provide either symbol or symbols, not both."

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
        assert result["summary"]["empty"] is True
        assert result["summary"]["counts"]["matched_symbols"] == 0
        assert result["message"] == "No symbols matched the requested market scan filters."
        assert "no_action" not in result

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbols_get")
    @patch("mtdata.core.symbols.mt5.copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    def test_market_scan_expands_parent_group(
        self,
        mock_tick,
        mock_rates,
        mock_symbols_get,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", path="Forex\\Majors"),
            _make_symbol("AUDCAD", path="Forex\\Minors"),
        ]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=50)

        fn = _get_market_scan()
        result = fn(group="Forex", universe="all", lookback=4)

        assert result["success"] is True
        assert result["meta"]["request"]["group"] == "Forex"
        assert result["meta"]["request"]["groups"] == ["Forex\\Majors", "Forex\\Minors"]
        assert {row["symbol"] for row in result["data"]["table"]["rows"]} == {"EURUSD", "AUDCAD"}

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick", return_value=None)
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_group_skipped_examples_are_deterministic(
        self,
        mock_symbols_get,
        mock_tick,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("usdjpy", path="Forex\\Majors"),
            _make_symbol("EURUSD", path="Forex\\Majors"),
        ]

        fn = _get_market_scan()
        result = fn(group="Forex\\Majors", lookback=4)

        assert result["success"] is True
        assert result["meta"]["request"]["scope"] == "group"
        assert [row["symbol"] for row in result["meta"]["stats"]["skipped_examples"]] == ["EURUSD", "usdjpy"]
