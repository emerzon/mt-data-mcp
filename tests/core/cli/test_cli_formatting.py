"""Output formatting tests for mtdata.core.cli module.

Tests result formatting, CLI text output, and JSON serialization.
"""

import argparse
import copy
import json
import logging
import os
import sys
import types
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from unittest.mock import MagicMock, call, patch

import pytest

from mtdata.core.data.requests import DataFetchCandlesRequest
from mtdata.core.trading.requests import (
    TradeGetOpenRequest,
    TradeHistoryRequest,
    TradeRiskAnalyzeRequest,
)
from mtdata.forecast.requests import ForecastGenerateRequest

# ---------------------------------------------------------------------------
# Fixture: ensure the cli module is importable with heavy deps mocked
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Clear env vars that influence debug/colour behaviour between tests."""
    monkeypatch.delenv("MTDATA_CLI_DEBUG", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("MT5_TIME_OFFSET_MINUTES", raising=False)


# We import lazily inside tests where heavy server machinery is needed,
# but the pure-logic helpers can be imported directly.
from mtdata.core.cli import (
    _first_line,
    _format_cli_literal,
    _format_result_for_cli,
    _format_result_minimal,
    _json_default,
    _quote_cli_value,
    _render_cli_result,
    _write_cli_text,
)


# ========================================================================
# _json_default
# ========================================================================


class TestJsonDefault:
    def test_none(self):
        assert _json_default(None) is None

    def test_str(self):
        assert _json_default("hello") == "hello"

    def test_int(self):
        assert _json_default(42) == 42

    def test_float(self):
        assert _json_default(3.14) == 3.14

    def test_bool(self):
        assert _json_default(True) is True

    def test_bytes(self):
        assert _json_default(b"hello") == "hello"

    def test_bytearray(self):
        assert _json_default(bytearray(b"hello")) == "hello"

    def test_datetime(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _json_default(dt)
        assert "2025" in result

    def test_namedtuple(self):
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y"])
        p = Point(1, 2)
        result = _json_default(p)
        assert result == {"x": 1, "y": 2}

    def test_fallback_to_str(self):
        class Custom:
            def __str__(self):
                return "custom_obj"

        result = _json_default(Custom())
        assert result == "custom_obj"

    def test_isoformat_exception(self):
        class BadDate:
            def isoformat(self):
                raise RuntimeError("fail")

        result = _json_default(BadDate())
        assert isinstance(result, str)

    def test_namedtuple_exception(self):
        class BadNT:
            def _asdict(self):
                raise RuntimeError("fail")

        result = _json_default(BadNT())
        assert isinstance(result, str)


def test_format_result_for_cli_does_not_mutate_candle_payload() -> None:
    payload = {
        "success": True,
        "count": 1,
        "data": [{"time": "2025-01-01 00:00", "close": 1.1}],
        "meta": {"runtime": {"timezone": {"used": {"tz": "UTC"}}}},
    }
    original = copy.deepcopy(payload)

    rendered = _format_result_for_cli(
        payload,
        fmt="json",
        verbose=False,
        cmd_name="data_fetch_candles",
    )

    rendered_payload = json.loads(rendered)
    assert rendered_payload["bars"] == {
        "columns": ["time", "close"],
        "rows": [["2025-01-01 00:00", 1.1]],
    }
    assert "data" not in rendered_payload
    assert payload == original


def test_render_cli_result_does_not_mutate_input(capsys) -> None:
    args = argparse.Namespace(json=True, verbose=False, extras=None, precision=None)
    result = {
        "success": True,
        "symbol": {
            "name": "EURUSD",
            "time": "2025-01-01 00:00",
            "time_epoch": 1735689600,
        },
        "meta": {"tool": "symbols_describe"},
    }
    original = copy.deepcopy(result)

    _render_cli_result(result, args=args, cmd_name="symbols_describe")

    assert json.loads(capsys.readouterr().out)["symbol"]["name"] == "EURUSD"
    assert result == original


# ========================================================================
# _format_result_minimal
# ========================================================================


class TestFormatResultMinimal:
    def test_string_passthrough(self):
        result = _format_result_minimal("hello")
        assert isinstance(result, str)

    def test_none_returns_empty(self):
        result = _format_result_minimal(None)
        assert result == ""

    @patch("mtdata.core.cli.formatting._shared_minimal", side_effect=Exception("boom"))
    def test_fallback_on_exception(self, mock_shared):
        result = _format_result_minimal({"key": "value"})
        assert "key" in result


# ========================================================================
# _format_result_for_cli
# ========================================================================


class TestFormatResultForCli:
    def test_json_format(self):
        result = _format_result_for_cli(
            {"a": 1}, fmt="json", verbose=False, cmd_name="test"
        )
        parsed = json.loads(result)
        assert parsed["a"] == 1

    def test_json_format_with_datetime(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _format_result_for_cli(
            {"dt": dt}, fmt="json", verbose=False, cmd_name="test"
        )
        parsed = json.loads(result)
        assert "2025" in parsed["dt"]

    @patch("mtdata.core.cli.formatting._shared_minimal", return_value="minimal output")
    def test_toon_format(self, mock_shared):
        result = _format_result_for_cli(
            {"a": 1}, fmt="toon", verbose=False, cmd_name="test"
        )
        assert result == "minimal output"

    @patch("mtdata.core.cli.formatting._shared_minimal", side_effect=TypeError("bad"))
    def test_toon_format_fallback(self, mock_shared):
        result = _format_result_for_cli(
            {"a": 1}, fmt="toon", verbose=False, cmd_name="test_cmd"
        )
        assert isinstance(result, str)

    def test_trade_command_no_simplify_numbers(self):
        # trade_ commands should NOT simplify numbers
        result = _format_result_for_cli(
            {"price": 1.23456}, fmt="json", verbose=False, cmd_name="trade_place"
        )
        parsed = json.loads(result)
        assert parsed["price"] == 1.23456

    def test_json_format_replaces_non_finite_with_null(self):
        result = _format_result_for_cli(
            {"nan": float("nan"), "pos_inf": float("inf"), "neg_inf": float("-inf")},
            fmt="json",
            verbose=False,
            cmd_name="test",
        )
        parsed = json.loads(result)
        assert parsed["nan"] is None
        assert parsed["pos_inf"] is None
        assert parsed["neg_inf"] is None

    def test_none_fmt_defaults_to_toon(self):
        result = _format_result_for_cli(
            "hello", fmt=None, verbose=False, cmd_name="test"
        )
        assert isinstance(result, str)

    def test_news_toon_format_omits_null_tail_cells_and_uses_generic_time_header(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "USDJPY",
                "general_news": [
                    {"title": "Fed preview", "relative_time": "2 hours ago"}
                ],
                "related_news": [
                    {
                        "title": "USD/JPY market snapshot",
                        "relative_time": "just now",
                        "kind": "market_snapshot",
                        "summary": "Price: 159.80",
                    },
                    {
                        "title": "US CPI (USD)",
                        "time_utc": "2026-04-07 12:30 UTC",
                        "kind": "economic_event",
                        "summary": "Expected: 3.2% | Prior: 3.1%",
                    },
                    {
                        "title": "BoJ's Ueda warns on FX",
                        "relative_time": "8 days ago",
                    },
                ],
                "impact_news": [
                    {"title": "Oil jumps on war fears", "relative_time": "6 hours ago"}
                ],
                "upcoming_events": [
                    {
                        "title": "US CPI (USD)",
                        "time_utc": "2026-04-07 12:30 UTC",
                        "kind": "economic_event",
                        "summary": "Expected: 3.2% | Prior: 3.1%",
                    }
                ],
                "recent_events": [
                    {
                        "title": "US CPI (USD)",
                        "relative_time": "2 hours ago",
                        "kind": "economic_event",
                        "summary": "Actual: 3.2% | Expected: 3.1% | Prior: 3.0%",
                    }
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="news",
        )

        assert "general_news[1]{title,time}:" in result
        assert '"Fed preview",2 hours ago' in result
        assert "related_news[3]{title,time,kind,summary}:" in result
        assert (
            '"USD/JPY market snapshot",just now,market_snapshot,"Price: 159.80"'
            in result
        )
        assert '"US CPI (USD)","2026-04-07 12:30 UTC",economic_event,' in result
        assert '"BoJ\'s Ueda warns on FX",8 days ago' in result
        assert '"Oil jumps on war fears",6 hours ago' in result
        assert "upcoming_events[1]{title,time,kind,summary}:" in result
        assert "recent_events[1]{title,time,kind,summary}:" in result
        assert (
            '"US CPI (USD)",2 hours ago,economic_event,"Actual: 3.2% | Expected: 3.1% | Prior: 3.0%"'
            in result
        )
        assert "null" not in result

    def test_news_toon_format_uses_published_at_before_relative_time(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "general_news": [
                    {
                        "title": "Fed preview",
                        "published_at": "2026-04-25T17:25:00+00:00",
                        "relative_time": "4 hours ago",
                        "source": "Reuters",
                    }
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="news",
        )

        assert "general_news[1]{title,published_at,relative_time,source}:" in result
        assert '"Fed preview","2026-04-25T17:25:00+00:00",4 hours ago,Reuters' in result

    def test_toon_format_preserves_candle_diagnostics_in_shared_output(self):
        result = _format_result_for_cli(
            {
                "meta": {
                    "diagnostics": {
                        "query": {
                            "requested_bars": 100,
                            "warmup_bars": 50,
                            "warmup_retry": {"applied": False, "warmup_bars": 80},
                        }
                    }
                }
            },
            fmt="toon",
            verbose=False,
            cmd_name="data_fetch_candles",
        )
        assert "warmup_bars" in result
        assert "requested_bars" in result

    def test_toon_format_hides_barrier_probability_curves_in_default_view(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "prob_tp_first": 0.52,
                "prob_sl_first": 0.48,
                "tp_hit_prob_by_t": [0.1, 0.2],
                "sl_hit_prob_by_t": [0.3, 0.4],
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_barrier_prob",
        )
        assert "prob_tp_first" in result
        assert "prob_sl_first" in result
        assert "tp_hit_prob_by_t" not in result
        assert "sl_hit_prob_by_t" not in result

    def test_toon_format_hides_barrier_grid_and_param_help_in_shared_output(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "best": {
                    "tp": 0.5,
                    "sl": 0.25,
                    "ev": 0.1,
                    "ev_gross": None,
                    "ev_net": None,
                    "prob_win": 0.4,
                    "prob_loss": 0.5,
                },
                "results": [
                    {
                        "tp": 0.5,
                        "sl": 0.25,
                        "rr": 2.0,
                        "prob_win": 0.4,
                        "prob_loss": 0.5,
                        "prob_resolve": 0.9,
                        "ev": 0.1,
                        "ev_gross": None,
                        "ev_net": None,
                        "profit_factor": 1.4,
                    }
                ],
                "grid": [
                    {
                        "tp": 0.5,
                        "sl": 0.25,
                        "ev": 0.1,
                        "ev_gross": None,
                        "ev_net": None,
                    },
                    {
                        "tp": 0.75,
                        "sl": 0.25,
                        "ev": 0.08,
                        "ev_gross": None,
                        "ev_net": None,
                    },
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_barrier_optimize",
        )
        assert "best" in result
        assert "results" not in result
        assert "grid" not in result
        assert "ev_gross" not in result
        assert "ev_net" not in result

        vol_result = _format_result_for_cli(
            {"success": True, "params_explained": {"lambda_": "EWMA decay factor"}},
            fmt="toon",
            verbose=False,
            cmd_name="forecast_volatility_estimate",
        )
        assert "params_explained" in vol_result

    def test_toon_format_preserves_shared_time_and_tick_fields(self):
        ticker = _format_result_for_cli(
            {
                "success": True,
                "time": 1700000000,
                "time_display": "2023-11-14 22:13",
            },
            fmt="toon",
            verbose=False,
            cmd_name="market_ticker",
        )
        assert 'time: "2023-11-14 22:13"' in ticker
        assert "time_display" not in ticker

        depth = _format_result_for_cli(
            {
                "success": True,
                "data": {
                    "bid": 1.1,
                    "time": 1700000000,
                    "time_display": "2023-11-14 22:13",
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="market_depth_fetch",
        )
        assert "time_display" in depth
        assert "time:" in depth

        ticks = _format_result_for_cli(
            {
                "success": True,
                "start_epoch": 1.0,
                "end_epoch": 2.0,
                "stats": {
                    "bid": {"first": 1.17225123, "std": 0.000007},
                    "spread": {"mean": 0.00001234},
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="data_fetch_ticks",
        )
        assert "start_epoch" in ticks
        assert "end_epoch" in ticks
        assert "stats" in ticks
        assert "first: 1.17225123" in ticks
        assert "std: 0.000007" in ticks
        assert "spread.mean: 0.00001234" in ticks

    def test_market_ticker_json_uses_display_time_as_canonical_field(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": True,
                    "symbol": "BTCUSD",
                    "time": 1700000000,
                    "time_display": "2023-11-14 22:13",
                    "spread_points": 8.999999999992347,
                    "spread_pct": 0.007795818842487513,
                    "spread_pct_display": "0.007796%",
                },
                fmt="json",
                verbose=False,
                cmd_name="market_ticker",
            )
        )
        assert payload["time"] == "2023-11-14 22:13"
        assert payload["spread_points"] == 8.999999999992347
        assert "spread_pips" not in payload
        assert payload["spread_pct"] == 0.007795818842487513
        assert payload["spread_pct_display"] == "0.007796%"
        assert "time_display" not in payload
        assert "time_epoch" not in payload

    def test_market_ticker_verbose_toon_keeps_raw_epoch_separately(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "time": 1700000000,
                "time_display": "2023-11-14 22:13",
            },
            fmt="toon",
            verbose=True,
            cmd_name="market_ticker",
        )
        assert 'time: "2023-11-14 22:13"' in result
        assert "time_epoch: 1700000000" in result
        assert "time_display" not in result

    def test_market_scan_toon_prunes_echoed_query_metadata(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "count": 1,
                "headers": ["symbol"],
                "data": [["EURUSD"]],
                "scope": "group",
                "group": "Forex\\Majors",
                "timeframe": "H1",
                "lookback": 100,
                "limit": 3,
                "rank_by": "abs_price_change_pct",
                "scanned_symbols": 6,
                "evaluated_symbols": 6,
                "matched_symbols": 6,
                "filtered_out_symbols": 0,
                "skipped_symbols": 0,
                "query_latency_ms": 289.0,
            },
            fmt="toon",
            verbose=False,
            cmd_name="market_scan",
        )

        assert "success: true" in result
        assert "count: 1" in result
        assert "EURUSD" in result
        assert "scope:" not in result
        assert "query_latency_ms" not in result
        assert "rank_by" not in result

    def test_market_scan_json_keeps_metadata_for_scripts(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": True,
                    "count": 1,
                    "headers": ["symbol"],
                    "data": [["EURUSD"]],
                    "scope": "group",
                    "group": "Forex\\Majors",
                    "query_latency_ms": 289.0,
                },
                fmt="json",
                verbose=False,
                cmd_name="market_scan",
            )
        )

        assert payload["scope"] == "group"
        assert payload["group"] == "Forex\\Majors"
        assert payload["query_latency_ms"] == 289.0

    def test_trade_session_context_json_compacts_nested_sections(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": True,
                    "symbol": "EURUSD",
                    "state": "open_position",
                    "account": {
                        "success": True,
                        "balance": 10000.0,
                        "equity": 10010.0,
                        "margin_level": 250.0,
                        "terminal_connected": True,
                        "execution_ready": True,
                    },
                    "open_positions": {
                        "success": True,
                        "kind": "open_positions",
                        "count": 1,
                        "items": [
                            {
                                "Ticket": 123456,
                                "Time": "2023-11-14 22:13",
                                "Type": "BUY",
                                "Volume": 0.1,
                                "Open Price": 1.1,
                                "Current Price": 1.1004,
                                "SL": 1.095,
                                "TP": 1.11,
                                "Profit": 4.2,
                            }
                        ],
                    },
                    "pending_orders": {
                        "success": True,
                        "kind": "pending_orders",
                        "count": 0,
                        "message": "No pending orders for EURUSD",
                        "no_action": True,
                    },
                    "ticker": {
                        "success": True,
                        "symbol": "EURUSD",
                        "time": 1700000000,
                        "time_display": "2023-11-14 22:13",
                        "bid": 1.1,
                        "ask": 1.1002,
                        "spread": 0.0002,
                        "spread_points": 20.0,
                    },
                },
                fmt="json",
                verbose=False,
                cmd_name="trade_session_context",
            )
        )

        assert payload["account"] == {
            "balance": 10000.0,
            "equity": 10010.0,
            "margin_level": 250.0,
        }
        assert payload["ticker"]["time"] == "2023-11-14 22:13"
        assert payload["ticker"]["spread_points"] == 20.0
        assert "time_display" not in payload["ticker"]
        assert "time_epoch" not in payload["ticker"]
        assert payload["open_positions"] == [
            {
                "ticket": 123456,
                "time": "2023-11-14 22:13",
                "type": "BUY",
                "volume": 0.1,
                "open_price": 1.1,
                "current_price": 1.1004,
                "sl": 1.095,
                "tp": 1.11,
                "profit": 4.2,
            }
        ]
        assert "pending_orders" not in payload

    def test_trade_session_context_verbose_toon_keeps_full_sections_and_ticker_epoch(
        self,
    ):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "state": "flat",
                "open_positions": {
                    "success": True,
                    "kind": "open_positions",
                    "count": 0,
                    "message": "No open positions for EURUSD",
                    "no_action": True,
                },
                "ticker": {
                    "success": True,
                    "time": 1700000000,
                    "time_display": "2023-11-14 22:13",
                },
            },
            fmt="toon",
            verbose=True,
            cmd_name="trade_session_context",
        )

        assert 'time: "2023-11-14 22:13"' in result
        assert "time_epoch: 1700000000" in result
        assert "open_positions:" in result
        assert "message: No open positions for EURUSD" in result

    def test_trade_session_context_json_keeps_already_compact_nested_sections(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": True,
                    "symbol": "EURUSD",
                    "state": "pending_only",
                    "account": {
                        "balance": 10000.0,
                        "equity": 10010.0,
                        "margin_level": 250.0,
                    },
                    "open_positions": [
                        {
                            "ticket": 123456,
                            "time": "2023-11-14 22:13",
                            "type": "BUY",
                            "volume": 0.1,
                        }
                    ],
                    "ticker": {
                        "success": True,
                        "bid": 1.1,
                        "ask": 1.1002,
                        "time": 1700000000,
                        "time_display": "2023-11-14 22:13",
                    },
                },
                fmt="json",
                verbose=False,
                cmd_name="trade_session_context",
            )
        )

        assert payload["account"] == {
            "balance": 10000.0,
            "equity": 10010.0,
            "margin_level": 250.0,
        }
        assert payload["open_positions"] == [
            {
                "ticket": 123456,
                "time": "2023-11-14 22:13",
                "type": "BUY",
                "volume": 0.1,
            }
        ]
        assert payload["ticker"]["time"] == "2023-11-14 22:13"

    def test_symbols_describe_compact_view_hides_time_epoch(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": {
                    "name": "BTCUSD",
                    "time_epoch": 1700000000.0,
                    "time": "2023-11-14 22:13",
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="symbols_describe",
        )
        assert 'time: "2023-11-14 22:13"' in result
        assert "time_epoch" not in result
        assert "meta:" not in result

    def test_symbols_describe_json_compact_hides_meta_block(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": True,
                    "symbol": {
                        "name": "BTCUSD",
                        "time_epoch": 1700000000.0,
                        "time": "2023-11-14 22:13",
                    },
                    "meta": {"tool": "symbols_describe"},
                },
                fmt="json",
                verbose=False,
                cmd_name="symbols_describe",
            )
        )

        assert "time_epoch" not in payload["symbol"]
        assert "meta" not in payload

    def test_candle_json_uses_table_bars(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "data": [{"time": "2023-11-14 22:13", "close": 1.1}],
                    "success": True,
                    "count": 1,
                    "symbol": "EURUSD",
                },
                fmt="json",
                verbose=False,
                cmd_name="data_fetch_candles",
            )
        )
        assert payload["bars"] == {
            "columns": ["time", "close"],
            "rows": [["2023-11-14 22:13", 1.1]],
        }
        assert "data" not in payload
        assert "count" not in payload

    def test_compact_toon_hides_runtime_timezone_meta(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "meta": {
                    "tool": "support_resistance_levels",
                    "runtime": {
                        "timezone": {
                            "utc": {"tz": "UTC"},
                            "server": {"tz": "Europe/Nicosia"},
                            "client": {"tz": "America/Chicago"},
                        }
                    },
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="support_resistance_levels",
        )
        assert "runtime.timezone" not in result
        assert "meta:" not in result

    def test_compact_toon_keeps_used_timezone_as_meta_timezone(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "meta": {
                    "tool": "data_fetch_candles",
                    "runtime": {
                        "timezone": {
                            "used": {"tz": "America/Chicago"},
                            "utc": {"tz": "UTC"},
                            "server": {"tz": "Europe/Nicosia"},
                        }
                    },
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="data_fetch_candles",
        )
        assert "timezone: America/Chicago" in result
        assert "runtime.timezone" not in result

    def test_toon_format_preserves_shared_causal_and_regime_details(self):
        causal = _format_result_for_cli(
            {
                "success": True,
                "data": {
                    "links": [
                        {
                            "effect": "EURUSD",
                            "cause": "GBPUSD",
                            "lag": 1,
                            "p_value": 0.02,
                        }
                    ],
                    "summary_text": "Effect <- Cause | Lag | p-value",
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="causal_discover_signals",
        )
        assert "summary_text" in causal
        assert "links[1]" in causal

        regime = _format_result_for_cli(
            {
                "success": True,
                "params_used": {
                    "hazard_lambda": 168,
                    "auto_calibration": {
                        "calibrated": True,
                        "points": 219,
                        "sigma": 0.0031,
                        "kurtosis_excess": 1.2,
                        "jump_share_abs_z_ge_2_5": 0.08,
                        "trend_norm": 0.4,
                    },
                    "cp_threshold_calibration": {
                        "mode": "walkforward_quantile",
                        "calibrated": True,
                        "points": 219,
                        "window": 80,
                        "null_max_quantile": 0.42,
                    },
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="regime_detect",
        )
        assert "hazard_lambda" in regime
        assert "auto_calibration:" in regime
        assert "calibrated: true" in regime
        assert "sigma" in regime
        assert "kurtosis_excess" in regime
        assert "jump_share_abs_z_ge_2_5" in regime
        assert "trend_norm" in regime
        assert "window:" in regime
        assert "null_max_quantile" in regime

    def test_toon_format_hides_analysis_legends_by_default(self):
        payload = {
            "success": True,
            "data": {
                "links": [
                    {"effect": "EURUSD", "cause": "GBPUSD", "lag": 1, "p_value": 0.02}
                ],
            },
            "legends": {
                "transform": {"log_return": {"description": "Returns"}},
                "note_p_value": "Lower is better",
            },
        }

        compact = _format_result_for_cli(
            payload,
            fmt="toon",
            verbose=False,
            cmd_name="causal_discover_signals",
        )
        verbose = _format_result_for_cli(
            payload,
            fmt="toon",
            verbose=True,
            cmd_name="causal_discover_signals",
        )

        assert "legends" not in compact
        assert "note_p_value" not in compact
        assert "legends" in verbose

    def test_toon_format_compacts_forecast_list_methods_output(self):
        result = _format_result_for_cli(
            {
                "detail": "compact",
                "total": 3,
                "total_filtered": 3,
                "available": 2,
                "unavailable": 1,
                "categories": {
                    "native": ["theta"],
                    "statsforecast": ["sf_theta", "sf_ets"],
                },
                "category_summary": [
                    {"category": "native", "total": 1, "examples": ["theta"]},
                    {
                        "category": "statsforecast",
                        "total": 2,
                        "examples": ["sf_theta", "sf_ets"],
                    },
                ],
                "methods": [
                    {
                        "method": "theta",
                        "category": "native",
                        "available": True,
                        "supports_ci": True,
                        "params_count": 1,
                    },
                    {
                        "method": "sf_theta",
                        "category": "statsforecast",
                        "available": True,
                        "supports_ci": True,
                        "params_count": 0,
                    },
                ],
                "methods_shown": 2,
                "methods_hidden": 1,
                "note": "Use --extras metadata to see all methods.",
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_list_methods",
        )

        assert "methods[2]{method,category,available,supports_ci}" in result
        assert "category_summary" not in result
        assert "categories" not in result
        assert "params_count" not in result
        assert "show_all_hint" in result

    def test_toon_format_keeps_richer_columns_for_full_forecast_methods_output(self):
        result = _format_result_for_cli(
            {
                "detail": "full",
                "total": 2,
                "total_filtered": 2,
                "available": 2,
                "unavailable": 0,
                "methods_shown": 2,
                "methods_hidden": 0,
                "filters": {"search": "theta"},
                "methods": [
                    {
                        "method": "theta",
                        "category": "native",
                        "namespace": "native",
                        "available": True,
                        "description": "Classic theta forecast.",
                        "params": [{"name": "window_size"}],
                        "supports_ci": True,
                        "concept": "theta",
                        "method_id": "native:theta",
                    },
                    {
                        "method": "sf_theta",
                        "category": "statsforecast",
                        "namespace": "statsforecast",
                        "available": True,
                        "description": "StatsForecast theta.",
                        "params": [],
                        "supports_ci": True,
                        "concept": "theta",
                        "method_id": "statsforecast:theta",
                    },
                ],
                "note": "Methods include namespace/concept/method_id fields.",
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_list_methods",
        )

        assert (
            "methods[2]{method,library,category,available,description,params_count,supports_ci,concept,method_id}"
            in result
        )
        assert "Classic theta forecast." in result
        assert "native:theta" in result
        assert "statsforecast:theta" in result
        assert "show_all_hint" not in result

    def test_toon_format_compacts_forecast_list_library_models_output(self):
        result = _format_result_for_cli(
            {
                "library": "native",
                "models": ["analog", "theta"],
                "capabilities": [
                    {
                        "method": "analog",
                        "available": True,
                        "description": "Nearest-neighbor analog forecast using pattern matching.",
                        "params": "name=window_size; type=int; description=Long serialized parameter spec.",
                    },
                    {
                        "method": "theta",
                        "available": True,
                        "description": "Theta forecast.",
                        "params": "name=alpha; type=float; description=Another long parameter spec.",
                    },
                ],
                "usage": [
                    "mtdata-cli forecast_generate SYMBOL --library native --method analog",
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_list_library_models",
        )

        assert "models[2]{model,available,description}" in result
        assert (
            "analog,true,Nearest-neighbor analog forecast using pattern matching."
            in result
        )
        assert "capabilities" not in result
        assert "Long serialized parameter spec" not in result
        assert "show_all_hint" in result

    def test_toon_format_compacts_regime_all_output(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "timeframe": "H1",
                "method": "all",
                "target": "return",
                "detail": "compact",
                "comparison": {
                    "current_regimes": {
                        "bocpd": {
                            "bias": "bullish",
                            "volatility": "moderate_vol",
                            "status": "no_recent_change_detected",
                            "regime_confidence": 0.82,
                            "recent_transition_activity": "none",
                        },
                        "ensemble": {
                            "bias": "bullish",
                            "volatility": "moderate_vol",
                            "label": "positive_mod_vol",
                            "regime_confidence": 0.74,
                        },
                    },
                    "agreement": {
                        "direction": {
                            "majority": "bullish",
                            "agreement_pct": 75.0,
                            "methods_considered": ["hmm", "clustering", "ensemble"],
                        },
                        "volatility": {
                            "majority": "moderate_vol",
                            "agreement_pct": 66.67,
                            "methods_considered": ["garch", "ensemble"],
                        },
                    },
                    "methods_failed": ["wavelet"],
                },
                "results": {
                    "bocpd": {"regimes": [{"start": "2025-01-01"}]},
                },
                "params_used": {
                    "methods_attempted": ["bocpd", "ensemble", "wavelet"],
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="regime_detect",
        )

        assert "detail: compact" in result
        assert "current_regimes[2]" in result
        assert "agreement:" in result
        assert "methods_failed[1]: wavelet" in result
        assert "results" not in result
        assert "params_used" not in result
        assert "show_all_hint" in result

    def test_toon_format_shows_regime_all_results_in_full_detail(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "timeframe": "H1",
                "method": "all",
                "target": "return",
                "detail": "full",
                "comparison": {
                    "current_regimes": {
                        "bocpd": {
                            "bias": "bullish",
                            "status": "no_recent_change_detected",
                        }
                    }
                },
                "results": {
                    "bocpd": {
                        "current_regime": {"status": "no_recent_change_detected"},
                        "regime_context": {"bias": "bullish"},
                    },
                    "hmm": {
                        "current_regime": {
                            "label": "positive_mod_vol",
                            "regime_confidence": 0.84,
                        }
                    },
                },
                "params_used": {
                    "methods_attempted": ["bocpd", "hmm"],
                    "methods_succeeded": ["bocpd", "hmm"],
                    "methods_failed": [],
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="regime_detect",
        )

        assert "detail: full" in result
        assert "results:" in result
        assert "current_regime.status: no_recent_change_detected" in result
        assert "regime_confidence: 0.84" in result
        assert "show_all_hint" not in result

    def test_toon_format_keeps_regime_all_summary_compact(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "timeframe": "H1",
                "method": "all",
                "target": "return",
                "detail": "summary",
                "comparison": {
                    "current_regimes": {
                        "bocpd": {
                            "bias": "bullish",
                            "status": "no_recent_change_detected",
                        }
                    }
                },
                "params_used": {
                    "methods_attempted": ["bocpd", "hmm"],
                    "methods_succeeded": ["bocpd"],
                    "methods_failed": ["hmm"],
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="regime_detect",
        )

        assert "detail: summary" in result
        assert "comparison.current_regimes[1]{method,bias,summary}" in result
        assert "results" not in result
        assert "show_all_hint" in result

    def test_toon_format_compacts_support_resistance_output(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "detail": "compact",
                "symbol": "EURUSD",
                "timeframe": "H1",
                "mode": "auto",
                "method": "weighted_retests",
                "current_price": 1.176,
                "timeframes_analyzed": ["M15", "H1", "H4"],
                "window": {"start": "2025-01-01", "end": "2025-01-02"},
                "level_counts": {"support": 1, "resistance": 1, "total": 2},
                "nearest": {
                    "support": {
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    "resistance": {
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                },
                "levels": [
                    {
                        "type": "support",
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    {
                        "type": "resistance",
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                ],
                "fibonacci": {
                    "selected_timeframe": "D1",
                    "selection_rule": "verbose-internals",
                    "nearest": {
                        "support": {
                            "label": "127.2%",
                            "value": 1.169,
                            "distance_pct": -0.0059,
                        },
                        "resistance": {
                            "label": "23.6%",
                            "value": 1.179,
                            "distance_pct": 0.0022,
                        },
                    },
                    "levels": [
                        {
                            "label": "127.2%",
                            "type": "support",
                            "value": 1.169,
                            "distance_pct": -0.0059,
                        },
                        {
                            "label": "23.6%",
                            "type": "resistance",
                            "value": 1.179,
                            "distance_pct": 0.0022,
                        },
                    ],
                },
                "coverage_gaps": {"support": {"threshold_pct": 0.12}},
                "zone_overlap": {"has_overlap": False},
            },
            fmt="toon",
            verbose=False,
            cmd_name="support_resistance_levels",
        )

        assert "nearest:" in result
        assert "levels[2]{type,value,distance_pct,touches,status}" in result
        assert "fibonacci:" in result
        assert "zone_high" not in result
        assert "coverage_gaps" not in result
        assert "zone_overlap" not in result
        assert "show_all_hint" in result

    def test_toon_format_surfaces_full_support_resistance_fields(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "detail": "full",
                "symbol": "EURUSD",
                "timeframe": "H1",
                "mode": "auto",
                "method": "weighted_retests",
                "current_price": 1.176,
                "timeframes_analyzed": ["M15", "H1", "H4"],
                "window": {"start": "2025-01-01", "end": "2025-01-02"},
                "level_counts": {"support": 1, "resistance": 1, "total": 2},
                "nearest": {
                    "support": {
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    "resistance": {
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                },
                "levels": [
                    {
                        "type": "support",
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    {
                        "type": "resistance",
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                ],
                "fibonacci": {
                    "selected_timeframe": "D1",
                    "selection_rule": "verbose-internals",
                    "nearest": {
                        "support": {
                            "label": "127.2%",
                            "value": 1.169,
                            "distance_pct": -0.0059,
                        },
                        "resistance": {
                            "label": "23.6%",
                            "value": 1.179,
                            "distance_pct": 0.0022,
                        },
                    },
                    "levels": [
                        {
                            "label": "127.2%",
                            "type": "support",
                            "value": 1.169,
                            "distance_pct": -0.0059,
                        },
                        {
                            "label": "23.6%",
                            "type": "resistance",
                            "value": 1.179,
                            "distance_pct": 0.0022,
                        },
                    ],
                },
                "coverage_gaps": {"support": {"threshold_pct": 0.12}},
                "zone_overlap": {"has_overlap": False},
                "qualification_basis": {"mode": "weighted_retests"},
            },
            fmt="toon",
            verbose=False,
            cmd_name="support_resistance_levels",
        )

        assert "detail: full" in result
        assert "zone_high: 1.175" in result
        assert "selection_rule: verbose-internals" in result
        assert "coverage_gaps.support.threshold_pct: 0.12" in result
        assert "qualification_basis.mode: weighted_retests" in result
        assert "show_all_hint" not in result

    def test_toon_format_preserves_standard_support_resistance_fields(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "detail": "standard",
                "symbol": "EURUSD",
                "timeframe": "H1",
                "mode": "auto",
                "method": "weighted_retests",
                "current_price": 1.176,
                "level_counts": {"support": 1, "resistance": 1, "total": 2},
                "nearest": {
                    "support": {
                        "type": "support",
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    "resistance": {
                        "type": "resistance",
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                },
                "supports": [
                    {
                        "type": "support",
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    }
                ],
                "resistances": [
                    {
                        "type": "resistance",
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    }
                ],
                "levels": [
                    {
                        "type": "support",
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    {
                        "type": "resistance",
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                ],
                "fibonacci": {
                    "timeframe": "D1",
                    "nearest": {
                        "support": {
                            "label": "127.2%",
                            "value": 1.169,
                            "distance_pct": -0.0059,
                        },
                        "resistance": {
                            "label": "23.6%",
                            "value": 1.179,
                            "distance_pct": 0.0022,
                        },
                    },
                },
                "coverage_gaps": {"support": {"threshold_pct": 0.12}},
            },
            fmt="toon",
            verbose=False,
            cmd_name="support_resistance_levels",
        )

        assert "detail: standard" in result
        assert "supports[1]" in result
        assert "resistances[1]" in result
        assert "zone_high: 1.175" in result
        assert "coverage_gaps.support.threshold_pct: 0.12" in result
        assert "show_all_hint" not in result

    def test_toon_format_hides_trade_metadata_in_non_verbose_output(self):
        open_out = _format_result_for_cli(
            [
                {
                    "Symbol": "EURUSD",
                    "Comment Length": 11,
                    "Comment Limit": 31,
                    "Comment May Be Truncated": True,
                }
            ],
            fmt="toon",
            verbose=False,
            cmd_name="trade_get_open",
        )
        assert "Comment Length" not in open_out
        assert "Comment Limit" not in open_out
        assert "Comment May Be Truncated" not in open_out

        hist_out = _format_result_for_cli(
            [
                {
                    "symbol": "EURUSD",
                    "comment_visible_length": 11,
                    "comment_max_length": 31,
                    "comment_may_be_truncated": True,
                    "type_code": 0,
                    "entry_code": 1,
                    "reason_code": 2,
                    "timestamp_timezone": "UTC",
                    "time_msc": 1700000000000,
                }
            ],
            fmt="toon",
            verbose=False,
            cmd_name="trade_history",
        )
        assert "comment_visible_length" not in hist_out
        assert "comment_max_length" not in hist_out
        assert "comment_may_be_truncated" not in hist_out
        assert "type_code" not in hist_out
        assert "entry_code" not in hist_out
        assert "reason_code" not in hist_out
        assert "timestamp_timezone" not in hist_out
        assert "time_msc" in hist_out

    def test_toon_format_keeps_trade_metadata_in_verbose_output(self):
        hist_out = _format_result_for_cli(
            [
                {
                    "symbol": "EURUSD",
                    "comment_visible_length": 11,
                    "comment_max_length": 31,
                    "comment_may_be_truncated": True,
                    "type_code": 0,
                    "timestamp_timezone": "UTC",
                }
            ],
            fmt="toon",
            verbose=True,
            cmd_name="trade_history",
        )
        assert "comment_visible_length" in hist_out
        assert "comment_max_length" in hist_out
        assert "comment_may_be_truncated" in hist_out
        assert "type_code" in hist_out
        assert "timestamp_timezone" in hist_out

    def test_toon_format_hides_trade_history_metadata_inside_envelope(self):
        hist_out = _format_result_for_cli(
            {
                "success": True,
                "kind": "trade_history",
                "history_kind": "deals",
                "count": 1,
                "items": [
                    {
                        "symbol": "EURUSD",
                        "comment_visible_length": 11,
                        "comment_max_length": 31,
                        "comment_may_be_truncated": True,
                        "type_code": 0,
                        "entry_code": 1,
                        "reason_code": 2,
                        "timestamp_timezone": "UTC",
                        "time_msc": 1700000000000,
                    }
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="trade_history",
        )
        assert "comment_visible_length" not in hist_out
        assert "comment_max_length" not in hist_out
        assert "comment_may_be_truncated" not in hist_out
        assert "type_code" not in hist_out
        assert "entry_code" not in hist_out
        assert "reason_code" not in hist_out
        assert "timestamp_timezone" not in hist_out
        assert "time_msc" in hist_out

    def test_toon_format_preserves_pending_orders_message_shape(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "kind": "pending_orders",
                "count": 0,
                "items": [],
                "message": "No pending orders",
                "no_action": True,
            },
            fmt="toon",
            verbose=False,
            cmd_name="trade_get_pending",
        )
        assert "message: No pending orders" in result
        assert "count: 0" in result
        assert "No pending orders" in result


# ========================================================================
# _write_cli_text
# ========================================================================


class TestWriteCliText:
    def test_falls_back_for_unencodable_console_text(self):
        class _FakeStream:
            encoding = "cp1252"

            def __init__(self) -> None:
                self.parts: List[str] = []

            def write(self, text: str) -> int:
                if "→" in text:
                    raise UnicodeEncodeError(
                        "charmap", text, 0, len(text), "cannot encode"
                    )
                self.parts.append(text)
                return len(text)

            def flush(self) -> None:
                return None

        stream = _FakeStream()
        _write_cli_text("Price $267 → $268", stream=stream)
        assert "".join(stream.parts) == "Price $267 -> $268\n"

    def test_non_tty_stream_writes_utf8_bytes(self):
        class _FakeBuffer:
            def __init__(self) -> None:
                self.parts: List[bytes] = []

            def write(self, data: bytes) -> int:
                self.parts.append(data)
                return len(data)

        class _FakeStream:
            encoding = "cp1252"

            def __init__(self) -> None:
                self.buffer = _FakeBuffer()
                self.parts: List[str] = []

            def isatty(self) -> bool:
                return False

            def write(self, text: str) -> int:
                self.parts.append(text)
                return len(text)

            def flush(self) -> None:
                return None

        stream = _FakeStream()
        _write_cli_text("Lundi de Pâques 清明节", stream=stream)
        assert stream.parts == []
        assert (
            b"".join(stream.buffer.parts).decode("utf-8") == "Lundi de Pâques 清明节\n"
        )


# ========================================================================
# _render_cli_result
# ========================================================================


class TestRenderCliResult:
    def test_detail_full_uses_verbose_output_contract(self, capsys):
        args = argparse.Namespace(detail="full", json=False, verbose=False)

        _render_cli_result({"value": 1}, args=args, cmd_name="sample_tool")

        out = capsys.readouterr().out
        assert "meta:" in out
        assert "tool: sample_tool" in out

    def test_symbols_describe_default_full_detail_does_not_force_meta(self, capsys):
        args = argparse.Namespace(detail="full", json=True, verbose=False)

        _render_cli_result(
            {
                "success": True,
                "symbol": {"name": "EURUSD", "time_epoch": 1700000000.0},
            },
            args=args,
            cmd_name="symbols_describe",
        )

        payload = json.loads(capsys.readouterr().out)
        assert "meta" not in payload
        assert "time_epoch" not in payload["symbol"]


# ========================================================================
# _first_line
# ========================================================================


class TestFirstLine:
    def test_none(self):
        assert _first_line(None) == ""

    def test_empty(self):
        assert _first_line("") == ""

    def test_single_line(self):
        assert _first_line("hello") == "hello"

    def test_multi_line(self):
        assert _first_line("first\nsecond") == "first"

    def test_leading_blank_lines(self):
        assert _first_line("\n\nthird") == "third"

    def test_all_blank(self):
        assert _first_line("\n\n  \n") == ""


# ========================================================================
# _format_cli_literal
# ========================================================================


class TestFormatCliLiteral:
    def test_none(self):
        assert _format_cli_literal(None) is None

    def test_bool_true(self):
        assert _format_cli_literal(True) == "true"

    def test_bool_false(self):
        assert _format_cli_literal(False) == "false"

    def test_int(self):
        assert _format_cli_literal(42) == "42"

    def test_float(self):
        assert _format_cli_literal(3.14) == "3.14"

    def test_string(self):
        assert _format_cli_literal("hello") == "hello"

    def test_dict(self):
        result = _format_cli_literal({"a": 1})
        assert json.loads(result) == {"a": 1}

    def test_list(self):
        result = _format_cli_literal([1, 2])
        assert json.loads(result) == [1, 2]


# ========================================================================
# _quote_cli_value
# ========================================================================


class TestQuoteCliValue:
    def test_empty(self):
        assert _quote_cli_value("") == '""'

    def test_no_spaces(self):
        assert _quote_cli_value("abc") == "abc"

    def test_with_spaces(self):
        assert _quote_cli_value("a b") == '"a b"'

    def test_already_quoted(self):
        assert _quote_cli_value('"a b"') == '"a b"'
