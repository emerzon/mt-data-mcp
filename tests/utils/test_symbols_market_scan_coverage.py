"""Tests for the symbols_top_markets MT5 market scanner tool."""

from contextlib import contextmanager
from datetime import datetime, timezone
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


def test_market_scan_freshness_uses_broker_crypto_category_on_weekends() -> None:
    from mtdata.core.symbols import _market_scan_freshness_fields

    saturday = datetime(2026, 7, 11, 12, tzinfo=timezone.utc).timestamp()
    recent_bar = saturday - 3600
    symbol = SimpleNamespace(name="TRUMPUSD", path="Crypto\\Altcoins")

    with patch("mtdata.core.symbols.time.time", return_value=saturday):
        result = _market_scan_freshness_fields(
            recent_bar,
            timeframe="H1",
            symbol=symbol,
        )

    assert result["data_stale"] is False
    assert result["usable_for_live_trading"] is True
    assert "market_status" not in result


def test_market_scan_error_uses_standard_error_envelope():
    from mtdata.core.symbols import _market_scan_error

    result = _market_scan_error(
        "Scan failed.",
        code="market_scan_failed",
        request={"group": "Forex"},
        stats={"processed": 10},
    )

    assert result["success"] is False
    assert result["error"] == "Scan failed."
    assert result["error_code"] == "market_scan_failed"
    assert result["operation"] == "market_scan"
    assert isinstance(result.get("request_id"), str)
    assert result["meta"]["request"] == {"group": "Forex"}
    assert result["meta"]["stats"] == {"processed": 10}
    assert "remediation" not in result


def test_high_volume_preset_ranks_without_unrelated_price_filter() -> None:
    from mtdata.core.symbols import _MARKET_SCAN_PRESETS

    assert _MARKET_SCAN_PRESETS["high_volume"] == {"rank_by": "tick_volume"}


def test_market_scan_freshness_summary_counts_bool_like_stale_flags():
    from mtdata.core.symbols import _market_scan_freshness_summary

    class BoolLike:
        def __bool__(self) -> bool:
            return True

    result = _market_scan_freshness_summary(
        [
            {"symbol": "A", "data_stale": True},
            {"symbol": "B", "data_stale": BoolLike()},
            {"symbol": "C", "data_stale": False},
        ]
    )

    assert result["stale_rows"] == 2
    assert result["freshness"] == "mixed, 2/3 stale"


def test_market_scan_freshness_summary_labels_closed_weekend_snapshot():
    from mtdata.core.symbols import _market_scan_freshness_summary

    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()
    with patch("mtdata.core.symbols.time.time", return_value=saturday):
        result = _market_scan_freshness_summary(
            [
                {"symbol": "EURUSD", "data_stale": False},
                {"symbol": "GBPUSD", "data_stale": False},
            ]
        )

    assert result["stale_rows"] == 0
    assert result["session_status"] == "closed_weekend"
    assert result["freshness"] == "closed_weekend_snapshot"


def test_market_scan_freshness_summary_labels_mixed_closed_weekend():
    from mtdata.core import symbols as symbols_mod

    def _fake_closed(symbol, *, now_epoch=None):
        return str(symbol) == "EURUSD"

    with patch.object(symbols_mod, "closed_session_context", _fake_closed):
        result = symbols_mod._market_scan_freshness_summary(
            [
                {"symbol": "EURUSD", "data_stale": False},
                {"symbol": "BTCUSD", "data_stale": False},
            ]
        )

    # Some sessions closed, none stale: freshness must not bare-claim "fresh".
    assert result["stale_rows"] == 0
    assert result["session_status"] == "mixed, 1/2 closed_weekend"
    assert result["freshness"] == "mixed, 1/2 closed_weekend_snapshot"


def test_market_scan_bar_freshness_uses_timeframe_window():
    from mtdata.core.symbols import _market_scan_freshness_fields

    with patch("mtdata.core.symbols.time.time", return_value=1_700_000_000.0):
        result = _market_scan_freshness_fields(
            1_700_000_000.0 - (26 * 60 * 60),
            timeframe="H1",
        )

    assert result["stale_after_seconds"] == 2 * 60 * 60
    assert result["data_stale"] is True
    assert result["freshness"] == "stale, bar 1d 2h ago"


def test_market_scan_labels_recent_bars_as_completed_not_current():
    from mtdata.core.symbols import _market_scan_freshness_fields

    with patch("mtdata.core.symbols.time.time", return_value=1_700_000_000.0):
        result = _market_scan_freshness_fields(
            1_700_000_000.0 - (60 * 60),
            timeframe="H1",
        )

    assert result["data_stale"] is False
    assert result["freshness"] == "latest completed bar, 1h ago"


def test_market_scan_default_limit_is_concise():
    from inspect import signature

    from mtdata.core.symbols import market_scan

    assert signature(_unwrap(market_scan)).parameters["limit"].default == 10


@patch("mtdata.core.symbols.time.time", return_value=10_000.0)
@patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
def test_market_scan_completed_rates_keeps_latest_closed_bar(mock_rates, mock_time):
    from mtdata.core.symbols import _market_scan_completed_rates

    bars = _make_bars([1.0, 2.0, 3.0])
    bars[-1]["time"] = 6_000.0
    mock_rates.return_value = bars

    result = _market_scan_completed_rates(
        "EURUSD",
        timeframe="H1",
        mt5_timeframe=16385,
        count=2,
    )

    assert [bar["close"] for bar in result] == [2.0, 3.0]
    mock_rates.assert_called_once_with("EURUSD", 16385, 0, 3)


@patch("mtdata.core.symbols.time.time", return_value=10_000.0)
@patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
def test_market_scan_completed_rates_drops_forming_bar(mock_rates, mock_time):
    from mtdata.core.symbols import _market_scan_completed_rates

    bars = _make_bars([1.0, 2.0, 3.0])
    bars[-1]["time"] = 9_000.0
    mock_rates.return_value = bars

    result = _market_scan_completed_rates(
        "EURUSD",
        timeframe="H1",
        mt5_timeframe=16385,
        count=2,
    )

    assert [bar["close"] for bar in result] == [1.0, 2.0]


def test_market_scan_signal_price_change_uses_previous_close(monkeypatch):
    from mtdata.core import symbols as symbols_mod

    monkeypatch.setattr(
        symbols_mod,
        "_market_scan_completed_rates",
        lambda *args, **kwargs: [
            {
                "time": 1_700_000_000.0,
                "open": 99.0,
                "close": 100.0,
                "tick_volume": 10,
                "real_volume": 0,
            },
            {
                "time": 1_700_003_600.0,
                "open": 110.0,
                "close": 105.0,
                "tick_volume": 12,
                "real_volume": 0,
            },
        ],
    )

    row, error = symbols_mod._build_market_scan_signal_row(
        _make_symbol("TEST", digits=2),
        timeframe="H1",
        mt5_timeframe=16385,
        lookback=2,
        rsi_length=14,
        sma_period=20,
        include_rsi=False,
        include_sma=False,
    )

    assert error is None
    assert row["previous_close"] == 100.0
    assert row["open"] == 110.0
    assert row["close"] == 105.0
    assert row["price_change_pct"] == 5.0
    assert row["price_change_basis"] == (
        "previous_completed_close_to_latest_completed_close"
    )


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
    currency_profit=None,
    digits: int = 0,
):
    return SimpleNamespace(
        name=name,
        path=path,
        description=description,
        visible=visible,
        trade_mode=trade_mode,
        digits=digits,
        point=point,
        trade_tick_size=trade_tick_size,
        trade_tick_value=trade_tick_value,
        currency_profit=currency_profit,
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


def test_symbol_category_prefers_stock_group_over_crypto_substrings():
    from mtdata.core.symbols import _symbol_category

    stock = _make_symbol(
        "LINK.US",
        path="Stock CFD's\\Other US",
        description="Interlink Electronics shares",
    )
    crypto = _make_symbol(
        "LINKUSD",
        path="Crypto\\Majors",
        description="Chainlink vs US Dollar",
    )

    assert _symbol_category(stock) == "stocks"
    assert _symbol_category(crypto) == "crypto"


def test_symbol_category_recognizes_exotic_forex_pairs_and_groups():
    from types import SimpleNamespace

    from mtdata.core.symbols import _symbol_category

    gbpsgd = SimpleNamespace(
        name="GBPSGD",
        path="Forex\\Exotics",
        description="Great Britain Pound vs Singapore Dollar",
    )
    usddkk = SimpleNamespace(
        name="USDDKK",
        path="Forex\\Exotics",
        description="US Dollar vs Danish Krone",
    )

    assert _symbol_category(gbpsgd) == "forex"
    assert _symbol_category(usddkk) == "forex"


def test_symbol_category_recognizes_metal_group_and_codes():
    from mtdata.core.symbols import _symbol_category

    platinum = _make_symbol(
        "XPTUSD",
        path="Commodities\\Metals",
        description="Platinum spot",
    )
    copper = _make_symbol(
        "XCUUSD",
        path="Markets\\Spot",
        description="Copper spot",
    )

    assert _symbol_category(platinum) == "commodities"
    assert _symbol_category(copper) == "commodities"


@contextmanager
def _ready_guard_ok(symbol: str, info_before=None):
    yield None, info_before


@pytest.fixture(autouse=True)
def _set_disabled_trade_mode(monkeypatch):
    import mtdata.core.symbols as symbols_mod

    monkeypatch.setattr(symbols_mod.mt5, "SYMBOL_TRADE_MODE_DISABLED", 0, raising=False)


class TestSymbolsTopMarkets:
    def test_top_market_headers_are_ranking_focused(self):
        from mtdata.core.symbols import _top_markets_headers

        compact_spread_headers = _top_markets_headers("spread", detail_mode="compact")
        compact_bar_headers = _top_markets_headers("volume", detail_mode="compact")

        assert compact_bar_headers == _top_markets_headers("price_change", detail_mode="compact")
        assert "spread_points" in compact_spread_headers
        assert "close" not in compact_spread_headers
        assert "close" in compact_bar_headers
        assert "spread_points" not in compact_bar_headers

        full_spread_headers = _top_markets_headers("spread", detail_mode="full")
        full_bar_headers = _top_markets_headers("volume", detail_mode="full")

        assert full_bar_headers == _top_markets_headers("price_change", detail_mode="full")
        assert "pricing_basis" in full_spread_headers
        assert "open" not in full_spread_headers
        assert "open" in full_bar_headers
        assert "pricing_basis" not in full_bar_headers

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_default_returns_single_abs_price_change_leaderboard(
        self,
        mock_symbols_get,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro", digits=4),
            _make_symbol("GBPUSD", description="Pound", digits=4),
        ]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": [
                {
                    "time": 1700000000.0,
                    "open": 1.1000,
                    "close": 1.0450,
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
        result = fn(limit=1, timeframe="H1")

        assert result["success"] is True
        assert result["ranking"] == "largest_abs_price_change_pct"
        assert result["requested_limit"] == 1
        assert "returned_count" not in result
        assert len(result["data"]) == 1
        assert result["data"][0]["symbol"] == "EURUSD"
        assert result["data"][0]["close"] == 1.045
        assert "bid" not in result["data"][0]
        assert "spread_points" not in result["data"][0]
        assert result["units"]["tick_volume"] == "broker_tick_count"
        assert result["units"]["close"] == "price"
        assert result["volume_type"] == "tick_volume"
        assert result["volume_semantics"] == "tick_volume_is_broker_tick_count_not_lots"
        assert "lowest_spread" not in result
        assert "highest_volume" not in result
        assert "highest_price_change_pct" not in result

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_filters_group_and_category_for_comparable_universe(
        self,
        mock_symbols_get,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", path="Forex\\Majors", description="Euro", digits=4),
            _make_symbol("GBPUSD", path="Forex\\Majors", description="Pound", digits=4),
            _make_symbol("XAUUSD", path="Commodities\\Metals", description="Gold"),
            _make_symbol("BTCUSD", path="Crypto", description="Bitcoin"),
        ]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": [
                {
                    "time": 1700000000.0,
                    "open": 1.1000,
                    "close": 1.1100,
                    "tick_volume": 100,
                    "real_volume": 0,
                }
            ],
            "GBPUSD": [
                {
                    "time": 1700000000.0,
                    "open": 1.3000,
                    "close": 1.2870,
                    "tick_volume": 50,
                    "real_volume": 0,
                }
            ],
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(
            limit=5,
            timeframe="H1",
            group="Forex",
            category="forex",
        )

        assert result["success"] is True
        assert result["filters"] == {
            "group": "Forex\\Majors",
            "category": "forex",
        }
        assert result["universe_size"] == 2
        assert {row["symbol"] for row in result["data"]} == {"EURUSD", "GBPUSD"}
        assert {row["asset_class"] for row in result["data"]} == {"forex"}

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
        assert result["requested_limit"] == 5
        assert "returned_count" not in result
        assert result["universe_size"] == 2
        assert result["available_count"] == 2
        assert "only 2 symbols had usable spread data" in result["note"]
        assert [row["symbol"] for row in result["data"]] == ["EURUSD", "XAUUSD"]
        assert list(result["data"][0].keys()) == [
            "symbol",
            "group",
            "asset_class",
            "timeframe",
            "data_source",
            "time",
            "data_stale",
            "freshness",
            "bid",
            "ask",
            "spread_pct",
            "spread_points",
            "spread_pips",
        ]
        assert result["data"][0]["data_source"] == "live_tick"
        assert result["data"][0]["freshness"] is None
        assert "tick_volume" not in result["data"][0]
        assert "pricing_basis" not in result["data"][0]

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    def test_stock_cfd_group_takes_precedence_over_nasdaq_venue(self, mock_group):
        from mtdata.core.symbols import _symbol_category

        symbol = _make_symbol("TSLA.NAS", description="Tesla Inc")
        symbol.path = "Stock CFD's\\Nasdaq"

        assert _symbol_category(symbol) == "stocks"

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_rank_by_aliases_match_market_scan_names(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", description="Euro")]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 1.02], tick_volume=20)

        fn = _get_symbols_top_markets()
        result = fn(rank_by="tick_volume", limit=5, detail="full")

        assert result["success"] is True
        assert result["ranking"] == "highest_tick_volume"
        assert result["rank_by"] == "tick_volume"
        assert result["rank_by_input"] is None

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
        top_by_category = {
            row["rank_category"]: row
            for row in result["data"]
            if row["rank"] == 1
        }
        assert top_by_category["lowest_spread"]["symbol"] == "EURUSD"
        assert top_by_category["highest_tick_volume"]["symbol"] == "EURUSD"
        assert top_by_category["highest_price_change_pct"]["symbol"] == "GBPUSD"
        assert result["collection_kind"] == "table"
        assert result["canonical_source"] == "data"
        assert result["ranking"] == "all"
        assert result["rank_categories"] == [
            "lowest_spread",
            "highest_tick_volume",
            "highest_price_change_pct",
        ]
        assert "groups" not in result
        assert "results" not in result
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
            "asset_class",
            "timeframe",
            "data_source",
            "time",
            "data_stale",
            "freshness",
            "bid",
            "ask",
            "spread_pct",
            "spread_points",
            "spread_pips",
        ]
        assert result["data"][0]["data_source"] == "live_tick"
        assert result["data"][0]["freshness"] is None
        assert "tick_volume" not in result["data"][0]
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
        assert result["ranking"] == "all"
        assert result["requested_limit"] == 5
        assert result["universe_size"] == 2
        assert result["returned_counts"] == {
            "lowest_spread": 2,
            "highest_tick_volume": 2,
            "highest_price_change_pct": 2,
        }
        assert result["available_counts"] == result["returned_counts"]
        assert "notes" not in result
        assert "data" in result
        first_spread = next(
            row
            for row in result["data"]
            if row["rank_category"] == "lowest_spread" and row["rank"] == 1
        )
        first_volume = next(
            row
            for row in result["data"]
            if row["rank_category"] == "highest_tick_volume" and row["rank"] == 1
        )
        first_price_change = next(
            row
            for row in result["data"]
            if row["rank_category"] == "highest_price_change_pct" and row["rank"] == 1
        )
        assert first_spread["data_source"] == "live_tick"
        assert first_spread["freshness"] is None
        assert first_spread["asset_class"] == "forex"
        assert first_volume["data_source"] == "H1_bars"
        assert first_volume["asset_class"] == "forex"
        assert first_spread["symbol"] == "EURUSD"
        assert first_volume["symbol"] == "EURUSD"
        assert first_price_change["symbol"] == "GBPUSD"
        assert result["units"]["tick_volume"] == "broker_tick_count"
        assert result["units"]["close"] == "price"
        assert result["volume_type"] == "tick_volume"
        assert result["volume_semantics"] == "tick_volume_is_broker_tick_count_not_lots"
        assert "data_sources" not in result
        assert "collection_kind" not in result
        assert "collection_contract_version" not in result

    def test_invalid_rank_by_returns_error(self):
        fn = _get_symbols_top_markets()

        result = fn(rank_by="unknown")

        assert result == {
            "error": (
                "rank_by must be one of: all, spread/spread_pct, tick_volume, "
                "price_change/price_change_pct, abs_price_change/abs_price_change_pct."
            )
        }

    def test_invalid_timeframe_returns_error_for_bar_metrics(self):
        fn = _get_symbols_top_markets()

        result = fn(rank_by="tick_volume", timeframe="BAD")

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
        result = fn(rank_by="tick_volume", timeframe="H1", limit=5, detail="full")

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
        result = fn(rank_by="tick_volume", timeframe="H1", limit=5, detail="full")

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

    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_universe_all_requires_bounded_scope(self, mock_symbols_get):
        fn = _get_market_scan()
        result = fn(universe="all", limit=5)

        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert "requires symbols or group" in result["error"]
        mock_symbols_get.assert_not_called()

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
        assert result["columns"][0] == "symbol"
        assert set(result["data"][0]).issubset(set(result["columns"]))
        assert "bid" in result["columns"]
        assert "ask" in result["columns"]
        assert "open" in result["columns"]
        assert "real_volume" in result["columns"]
        assert result["count"] == 1
        assert result["rank_by"] == "abs_price_change_pct"
        assert result["ranking"] == "largest_abs_price_change_pct"
        assert result["price_change_basis"] == (
            "previous_completed_close_to_latest_completed_close"
        )
        assert result["data"][0]["symbol"] == "EURUSD"
        assert result["data"][0]["rsi"] == 100.0
        assert result["data"][0]["sma_value"] == 5.0
        assert result["collection_kind"] == "table"
        assert result["canonical_source"] == "data"
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
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro", currency_profit="USD")
        ]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 2.0, 3.0, 4.0], tick_volume=120)

        fn = _get_market_scan()
        result = fn(timeframe="H1", lookback=4, limit=5)

        assert result["success"] is True
        assert "columns" not in result
        assert result["count"] == 1
        assert result["rank_by"] == "abs_price_change_pct"
        assert result["ranking"] == "largest_abs_price_change_pct"
        assert result["requested_limit"] == 5
        assert "returned_count" not in result
        assert result["universe_size"] == 1
        assert result["freshness"] in {
            "fresh",
            "stale",
            "closed_weekend_snapshot",
        }
        assert result["stale_rows"] in {0, 1}
        assert result["data_as_of"]
        assert "only 1 symbols were available" in result["note"]
        assert result["units"]["price_change_pct"] == "percentage_points (1.0 = 1%)"
        assert result["units"]["tick_volume"] == "broker_tick_count"
        assert result["units"]["spread_points"] == "broker_points"
        assert result["units"]["spread_pips"] == "pips"
        assert result["volume_type"] == "tick_volume"
        assert result["volume_semantics"] == "tick_volume_is_broker_tick_count_not_lots"
        row = result["data"][0]
        assert row["symbol"] == "EURUSD"
        assert {
            "symbol",
            "group",
            "asset_class",
            "timeframe",
            "data_source",
            "time",
            "data_stale",
            "freshness",
            "close",
            "price_change_pct",
            "tick_volume",
            "spread_pct",
            "spread_points",
            "spread_pips",
        }.issubset(row)
        assert "market_status" not in row
        assert "market_status_reason" not in row
        assert "freshness_policy_relaxed" not in row
        assert row["time"].endswith("Z")
        assert row["spread_pips"] == 1.0
        assert mock_rates.call_args.args[2:] == (0, 3)
        assert "real_volume" not in row
        assert "rows" not in result
        assert result["meta"]["request"]["detail"] == "compact"
        assert "collection_kind" not in result
        assert "collection_contract_version" not in result

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_compact_omits_non_fx_null_spread_pips(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol(
                "BTCUSD",
                path="Crypto",
                description="Bitcoin",
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                currency_profit="USD",
                digits=2,
            )
        ]
        mock_tick.return_value = _make_tick(bid=40000.00, ask=40000.50)
        mock_rates.return_value = _make_bars([40000.0, 40010.0, 40020.0, 40030.0], tick_volume=120)

        fn = _get_market_scan()
        result = fn(timeframe="H1", lookback=4, limit=5)

        assert result["success"] is True
        row = result["data"][0]
        assert row["spread_points"] == 50
        assert isinstance(row["spread_points"], int)
        assert row["spread_pips"] is None
        assert "spread_pips" not in result["units"]

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_supports_offset_pagination(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("GBPUSD", description="Pound"),
            _make_symbol("USDJPY", description="Yen"),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "GBPUSD": _make_tick(bid=1.3000, ask=1.3002),
            "USDJPY": _make_tick(bid=150.00, ask=150.03),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=300),
            "GBPUSD": _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=200),
            "USDJPY": _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=100),
        }[symbol]

        fn = _get_market_scan()
        result = fn(lookback=4, rank_by="tick_volume", limit=1, offset=1)

        assert result["success"] is True
        assert result["count"] == 1
        assert result["data"][0]["symbol"] == "GBPUSD"
        assert result["offset"] == 1
        assert result["requested_limit"] == 1
        assert "returned_count" not in result
        assert result["total_count"] == 3
        assert result["has_more"] is True
        assert result["message"].startswith(
            "Showing 1 of 3 symbols matching the requested market scan filters."
        )
        assert result["meta"]["request"]["offset"] == 1

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
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_spread_ranking_puts_stale_rows_after_fresh(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        now = 1_700_000_000.0
        mock_symbols_get.return_value = [
            _make_symbol("STALETIGHT", description="Old tight spread"),
            _make_symbol("FRESHWIDE", description="Fresh wider spread"),
        ]
        mock_tick.side_effect = lambda symbol: {
            "STALETIGHT": _make_tick(bid=1.1000, ask=1.1001),
            "FRESHWIDE": _make_tick(bid=1.1000, ask=1.1005),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "STALETIGHT": [
                {
                    "time": now - (11 * 3600),
                    "open": 1.1000,
                    "close": 1.1000,
                    "tick_volume": 119,
                    "real_volume": 0,
                },
                {
                    "time": now - (10 * 3600),
                    "open": 1.1000,
                    "close": 1.1000,
                    "tick_volume": 120,
                    "real_volume": 0,
                }
            ],
            "FRESHWIDE": [
                {
                    "time": now - (2 * 3600),
                    "open": 1.1000,
                    "close": 1.1000,
                    "tick_volume": 119,
                    "real_volume": 0,
                },
                {
                    "time": now - 3600,
                    "open": 1.1000,
                    "close": 1.1000,
                    "tick_volume": 120,
                    "real_volume": 0,
                }
            ],
        }[symbol]

        fn = _get_market_scan()
        with patch("mtdata.core.symbols.time.time", return_value=now):
            result = fn(rank_by="spread_pct", limit=2, timeframe="H1", lookback=2)

        assert result["success"] is True
        assert [row["symbol"] for row in result["data"]] == ["FRESHWIDE", "STALETIGHT"]
        assert result["data"][0]["data_stale"] is False
        assert result["data"][1]["data_stale"] is True
        assert result["freshness"] == "mixed, 1/2 stale"
        assert result["stale_rows"] == 1
        assert "stale_symbols" not in result
        assert "Returned rows: 1/2 stale." in result["message"]

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
        assert "matched_symbols" not in result["summary"]["counts"]
        assert result["total_count"] == 2
        assert result["meta"]["stats"]["scanned_symbols"] == 2
        mock_ready_guard.assert_called_once_with("USDJPY", info_before=hidden_symbol)

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._symbol_ready_guard", side_effect=_ready_guard_ok)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_group_accepts_doubled_backslash_path(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        _mock_ready_guard,
        _mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", visible=True)]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1002)
        mock_rates.return_value = _make_bars([1.0, 1.1, 1.2, 1.3], tick_volume=100)

        fn = _get_market_scan()
        result = fn(group="Forex\\\\Majors", universe="all", lookback=4)

        assert result["success"] is True
        assert result["meta"]["request"]["group"] == "Forex\\Majors"

    @patch("mtdata.core.symbols._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols._symbol_ready_guard", side_effect=_ready_guard_ok)
    @patch("mtdata.core.symbols._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_group_accepts_common_singular_alias(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        _mock_ready_guard,
        _mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", path="Forex\\Majors", visible=True)]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1002)
        mock_rates.return_value = _make_bars([1.0, 1.1, 1.2, 1.3], tick_volume=100)

        fn = _get_market_scan()
        result = fn(group="forex_major", universe="all", lookback=4)

        assert result["success"] is True
        assert result["meta"]["request"]["group"] == "Forex\\Majors"
        assert result["meta"]["request"]["groups"] == ["Forex\\Majors"]

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
    def test_market_scan_symbols_filters_single_symbol(
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
        result = fn(symbols="EURUSD", lookback=4)

        assert result["success"] is True
        assert result["meta"]["request"]["symbols_input"] == ["EURUSD"]
        assert [row["symbol"] for row in result["data"]] == ["EURUSD"]

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
        assert "matched_symbols" not in result["summary"]["counts"]
        assert result["total_count"] == 0
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
        assert {row["symbol"] for row in result["data"]} == {"EURUSD", "AUDCAD"}

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
