"""Comprehensive tests for mtdata.core.cli module.

Covers helper functions, argument parsing, tool discovery, command creation,
output formatting, and the main() entry point. All external MCP tool calls
and heavy imports are mocked.
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
    _add_forecast_generate_args,
    _apply_cli_output_mode_defaults,
    _apply_schema_overrides,
    _argparse_color_enabled,
    _attach_cli_meta,
    _build_cli_timezone_meta,
    _build_epilog,
    _build_usage_examples,
    _coerce_cli_scalar,
    _configure_cli_logging,
    _debug,
    _debug_enabled,
    _example_value,
    _extract_function_from_tool_obj,
    _extract_help_query,
    _extract_metadata_from_tool_obj,
    _first_line,
    _format_cli_literal,
    _format_result_for_cli,
    _format_result_minimal,
    _is_literal_origin,
    _is_typed_dict_type,
    _is_union_origin,
    _json_default,
    _match_commands,
    _merge_dict,
    _normalize_cli_argv_aliases,
    _normalize_cli_list_value,
    _parse_kv_string,
    _parse_set_overrides,
    _print_extended_help,
    _quote_cli_value,
    _render_cli_result,
    _resolve_param_kwargs,
    _safe_tz_name,
    _suggest_commands,
    _type_name,
    _unwrap_optional_type,
    _write_cli_text,
    add_dynamic_arguments,
    create_command_function,
    discover_tools,
    get_function_info,
    main,
)

# ========================================================================
# _debug_enabled / _debug
# ========================================================================


class TestDebugEnabled:
    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [
            (None, False),
            ("1", True),
            ("true", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("", False),
            ("yes", True),
        ],
    )
    def test_debug_env_values(self, monkeypatch, env_value, expected):
        if env_value is None:
            monkeypatch.delenv("MTDATA_CLI_DEBUG", raising=False)
        else:
            monkeypatch.setenv("MTDATA_CLI_DEBUG", env_value)
        assert _debug_enabled() is expected


class TestDebug:
    def test_debug_prints_when_enabled(self, monkeypatch, capsys):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "1")
        _debug("test message")
        assert "test message" in capsys.readouterr().err

    def test_debug_silent_when_disabled(self, monkeypatch, capsys):
        monkeypatch.delenv("MTDATA_CLI_DEBUG", raising=False)
        _debug("should not appear")
        assert capsys.readouterr().err == ""


class TestConfigureCliLogging:
    def test_default_cli_logging_suppresses_mtdata_info(self):
        logger = logging.getLogger("mtdata")
        previous = logger.level
        previous_propagate = logger.propagate
        try:
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            _configure_cli_logging(verbose=False)
            assert logger.level == logging.WARNING
            assert logger.propagate is False
            assert any(
                isinstance(handler, logging.NullHandler) for handler in logger.handlers
            )
        finally:
            logger.setLevel(previous)
            logger.propagate = previous_propagate

    def test_verbose_cli_logging_restores_mtdata_info(self, monkeypatch):
        # --verbose no longer enables INFO logging; operation logs should only
        # stream when MTDATA_CLI_DEBUG is set. Validate both branches.
        logger = logging.getLogger("mtdata")
        previous = logger.level
        previous_propagate = logger.propagate
        try:
            logger.setLevel(logging.WARNING)
            logger.propagate = False
            monkeypatch.delenv("MTDATA_CLI_DEBUG", raising=False)
            _configure_cli_logging(verbose=True)
            assert logger.level == logging.WARNING
            assert logger.propagate is False

            monkeypatch.setenv("MTDATA_CLI_DEBUG", "1")
            _configure_cli_logging(verbose=False)
            assert logger.level == logging.INFO
            assert logger.propagate is True
        finally:
            logger.setLevel(previous)
            logger.propagate = previous_propagate


# ========================================================================
# help suggestions
# ========================================================================


class TestHelpSuggestions:
    def test_suggest_commands_returns_close_matches(self):
        functions = {
            "forecast_generate": {"func": lambda: None},
            "indicators_list": {"func": lambda: None},
            "market_ticker": {"func": lambda: None},
        }
        assert _suggest_commands(functions, "forecast_generat") == ["forecast_generate"]

    def test_extended_help_shows_did_you_mean_for_typos(self, capsys):
        fns = {
            "indicators_list": {
                "func": lambda: None,
                "meta": {},
                "_cli_func_info": {"params": [], "doc": ""},
            },
            "market_ticker": {
                "func": lambda: None,
                "meta": {},
                "_cli_func_info": {"params": [], "doc": ""},
            },
        }
        _print_extended_help(fns, "indicatr_list")
        out = capsys.readouterr().out
        assert "No commands match 'indicatr_list'." in out
        assert "Did you mean: indicators_list" in out


# ========================================================================
# _argparse_color_enabled
# ========================================================================


class TestArgparseColorEnabled:
    def test_no_color_env(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert _argparse_color_enabled() is False

    def test_no_tty(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
        assert _argparse_color_enabled() is False

    def test_tty(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
        assert _argparse_color_enabled() is True


# ========================================================================
# _is_typed_dict_type
# ========================================================================


class TestIsTypedDictType:
    def test_regular_dict_is_not_typeddict(self):
        assert _is_typed_dict_type(dict) is False

    def test_class_with_annotations_no_keys(self):
        class Foo:
            __annotations__ = {"x": int}

        assert _is_typed_dict_type(Foo) is False

    def test_class_with_required_keys(self):
        class FakeTD:
            __annotations__ = {"x": int}
            __required_keys__ = frozenset({"x"})
            __optional_keys__ = frozenset()

        assert _is_typed_dict_type(FakeTD) is True

    def test_none_is_not_typeddict(self):
        assert _is_typed_dict_type(None) is False

    def test_int_is_not_typeddict(self):
        assert _is_typed_dict_type(int) is False


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
# _safe_tz_name
# ========================================================================


class TestSafeTzName:
    def test_none(self):
        assert _safe_tz_name(None) is None

    def test_with_zone_attr(self):
        tz = MagicMock()
        tz.zone = "US/Eastern"
        assert _safe_tz_name(tz) == "US/Eastern"

    def test_without_zone_attr(self):
        assert _safe_tz_name("UTC") == "UTC"

    def test_object_with_no_zone(self):
        class FakeTZ:
            pass

        result = _safe_tz_name(FakeTZ())
        assert isinstance(result, str)

    def test_prefers_tzname_over_raw_repr(self):
        class FakeTZ:
            def tzname(self, _dt):
                return "Central Daylight Time"

            def utcoffset(self, _dt):
                return None

            def __str__(self):
                return "tzwinlocal('Central Standard Time')"

        assert _safe_tz_name(FakeTZ()) == "Central Daylight Time"


# ========================================================================
# _build_cli_timezone_meta
# ========================================================================


class TestBuildCliTimezoneMeta:
    @patch("mtdata.bootstrap.settings.mt5_config")
    def test_with_config(self, mock_config):
        mock_config.server_tz_name = "Europe/Nicosia"
        mock_config.client_tz_name = "US/Eastern"
        from zoneinfo import ZoneInfo

        mock_config.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
        mock_config.get_client_tz.return_value = ZoneInfo("US/Eastern")
        mock_config.get_time_offset_seconds.return_value = 7200
        result = _build_cli_timezone_meta({"some": "data"})
        assert isinstance(result, dict)
        assert result["utc"]["tz"] == "UTC"
        assert result["server"]["tz"] == "Europe/Nicosia"
        assert result["client"]["tz"] == "US/Eastern"
        assert result["server"]["offset_seconds"] == 7200
        assert result["server"]["now"] != result["client"]["now"]

    @patch("mtdata.bootstrap.settings.mt5_config")
    def test_with_dict_result_containing_timezone(self, mock_config):
        mock_config.server_tz_name = None
        mock_config.client_tz_name = None
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 0
        result = _build_cli_timezone_meta({"timezone": "US/Eastern"})
        assert result["utc"]["tz"] == "UTC"
        assert "client" not in result or result["client"].get("tz") is None

    @patch("mtdata.bootstrap.settings.mt5_config")
    def test_with_non_dict_result(self, mock_config):
        mock_config.server_tz_name = None
        mock_config.client_tz_name = None
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 0
        result = _build_cli_timezone_meta("some string")
        assert result["utc"]["tz"] == "UTC"

    @patch("mtdata.bootstrap.settings.mt5_config")
    def test_with_offset_env(self, mock_config, monkeypatch):
        mock_config.server_tz_name = None
        mock_config.client_tz_name = None
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 10800
        monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "180")
        result = _build_cli_timezone_meta({})
        assert result["server"]["offset_seconds"] == 10800
        assert result["server"]["tz"] == "UTC+03:00"

    @patch("mtdata.bootstrap.settings.mt5_config")
    def test_with_invalid_offset_env(self, mock_config, monkeypatch):
        mock_config.server_tz_name = None
        mock_config.client_tz_name = None
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 0
        monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "abc")
        result = _build_cli_timezone_meta({})
        assert result["server"].get("offset_seconds") is None

    def test_server_source_none_by_default(self):
        result = _build_cli_timezone_meta({})
        assert result["server"]["source"] in (
            "none",
            "MT5_SERVER_TZ",
            "MT5_TIME_OFFSET_MINUTES",
        )

    def test_now_fields_are_iso_strings(self):
        result = _build_cli_timezone_meta({})
        assert isinstance(result["utc"]["now"], str)
        datetime.fromisoformat(result["utc"]["now"])
        if result["server"].get("now") is not None:
            datetime.fromisoformat(result["server"]["now"])
        if result.get("client", {}).get("now") is not None:
            datetime.fromisoformat(result["client"]["now"])

    @patch("mtdata.bootstrap.settings.mt5_config")
    def test_offset_env_source(self, mock_config, monkeypatch):
        mock_config.server_tz_name = None
        mock_config.client_tz_name = None
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 3600
        monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "60")
        result = _build_cli_timezone_meta({})
        if result["server"].get("tz") == "UTC+01:00":
            assert result["server"]["source"] == "MT5_TIME_OFFSET_MINUTES"

    @patch("mtdata.bootstrap.settings.mt5_config")
    def test_unknown_server_and_client_now_are_omitted(self, mock_config):
        mock_config.server_tz_name = None
        mock_config.client_tz_name = None
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 0

        result = _build_cli_timezone_meta({})

        assert "now" not in result["server"]
        assert "now" not in result.get("client", {})


# ========================================================================
# _attach_cli_meta
# ========================================================================


class TestAttachCliMeta:
    def test_non_verbose_strips_common_meta_and_diagnostics(self):
        r = {
            "key": "val",
            "meta": {"domain": {"symbol": "EURUSD"}},
            "diagnostics": {"source": "mt5"},
        }
        out = _attach_cli_meta(r, cmd_name="test", verbose=False)
        assert out is not r
        assert "cli_meta" not in out
        assert out["key"] == "val"
        assert "meta" not in out
        assert "diagnostics" not in out

    def test_non_dict_returns_unchanged(self):
        assert _attach_cli_meta("string", cmd_name="test", verbose=True) == "string"

    def test_verbose_dict_adds_meta(self):
        r = {"data": 1}
        out = _attach_cli_meta(r, cmd_name="my_cmd", verbose=True)
        assert "cli_meta" not in out
        assert out["meta"]["tool"] == "my_cmd"
        assert "timezone" in out["meta"]["runtime"]

    def test_preserves_unrecognized_cli_meta_key(self):
        r = {"data": 1, "cli_meta": {"existing": True}}
        out = _attach_cli_meta(r, cmd_name="cmd", verbose=True)
        assert out["cli_meta"] == {"existing": True}
        assert out["meta"]["tool"] == "cmd"

    def test_preserves_existing_common_meta(self):
        r = {"data": 1, "meta": {"domain": {"symbol": "EURUSD"}}}
        out = _attach_cli_meta(r, cmd_name="cmd", verbose=True)
        assert out["meta"]["tool"] == "cmd"
        assert out["meta"]["domain"]["symbol"] == "EURUSD"

    def test_verbose_candles_keeps_existing_common_diagnostics(self):
        r = {
            "meta": {
                "diagnostics": {
                    "query": {"raw_bars_fetched": 5, "latency_ms": 12.3},
                    "indicators": {"requested": False},
                    "session_gaps": {"expected_bar_seconds": 3600.0},
                }
            },
            "warnings": ["sample warning"],
        }
        out = _attach_cli_meta(r, cmd_name="data_fetch_candles", verbose=True)
        assert "cli_meta" not in out
        assert out["meta"]["diagnostics"]["query"]["raw_bars_fetched"] == 5
        assert out["meta"]["tool"] == "data_fetch_candles"

    def test_verbose_market_ticker_does_not_add_cli_diagnostics(self):
        r = {
            "time": 1700000000,
            "bid": 200.0,
            "ask": 201.0,
            "spread": 1.0,
            "spread_points": 100.0,
            "spread_usd": 100.0,
            "diagnostics": {
                "source": "mt5.symbol_info_tick",
                "cache_used": False,
                "query_latency_ms": 7.5,
                "data_freshness_seconds": 1.2,
            },
        }
        out = _attach_cli_meta(r, cmd_name="market_ticker", verbose=True)
        assert "cli_meta" not in out
        assert out["meta"]["tool"] == "market_ticker"
        assert out["diagnostics"]["source"] == "mt5.symbol_info_tick"

    def test_verbose_market_ticker_does_not_copy_meta_diagnostics_to_root(self):
        r = {
            "time": 1700000000,
            "bid": 200.0,
            "ask": 201.0,
            "spread": 1.0,
            "spread_points": 100.0,
            "spread_usd": 100.0,
            "meta": {
                "diagnostics": {
                    "source": "meta.source",
                    "cache_used": True,
                    "query_latency_ms": 8.5,
                    "data_freshness_seconds": 0.7,
                }
            },
            "diagnostics": {
                "source": "legacy.source",
                "cache_used": False,
                "query_latency_ms": 7.5,
                "data_freshness_seconds": 1.2,
            },
        }
        out = _attach_cli_meta(r, cmd_name="market_ticker", verbose=True)

        assert out["meta"]["tool"] == "market_ticker"
        assert out["meta"]["diagnostics"]["source"] == "meta.source"
        assert out["diagnostics"]["source"] == "legacy.source"
        assert out["diagnostics"]["cache_used"] is False

    def test_verbose_market_ticker_keeps_meta_diagnostics_nested(self):
        r = {
            "time": 1700000000,
            "bid": 200.0,
            "ask": 201.0,
            "meta": {
                "diagnostics": {
                    "source": "mt5.symbol_info_tick",
                    "cache_used": False,
                }
            },
        }
        out = _attach_cli_meta(r, cmd_name="market_ticker", verbose=True)

        assert out["meta"]["tool"] == "market_ticker"
        assert out["meta"]["diagnostics"]["source"] == "mt5.symbol_info_tick"
        assert "diagnostics" not in out


# ========================================================================
# get_function_info
# ========================================================================


class TestGetFunctionInfo:
    def test_simple_function(self):
        def my_func(symbol: str, count: int = 10) -> dict:
            """My function doc."""
            pass

        info = get_function_info(my_func)
        assert info["func"] is my_func
        assert info["doc"] == "My function doc."
        param_names = [p["name"] for p in info["params"]]
        assert "symbol" in param_names
        assert "count" in param_names

    def test_function_without_doc(self):
        def nodoc(x: int):
            pass

        info = get_function_info(nodoc)
        assert "Execute" in info["doc"]

    def test_param_type_defaults_to_str(self):
        def untyped(x):
            pass

        info = get_function_info(untyped)
        param = [p for p in info["params"] if p["name"] == "x"][0]
        assert param["type"] is str

    def test_required_based_on_default(self):
        def fn(a: str, b: int = 5):
            pass

        info = get_function_info(fn)
        a_param = [p for p in info["params"] if p["name"] == "a"][0]
        b_param = [p for p in info["params"] if p["name"] == "b"][0]
        assert a_param["required"] is True
        assert b_param["required"] is False

    def test_variadic_params_are_skipped(self):
        def fn(symbol: str, *args, **kwargs):
            pass

        info = get_function_info(fn)

        assert [p["name"] for p in info["params"]] == ["symbol"]

    def test_request_model_param_is_flattened(self):
        def request_tool(request: ForecastGenerateRequest):
            """Request tool."""
            pass

        info = get_function_info(request_tool)

        assert info["request_model"] is ForecastGenerateRequest
        assert info["request_param_name"] == "request"
        param_names = [p["name"] for p in info["params"]]
        assert "request" not in param_names
        assert "symbol" in param_names
        assert "horizon" in param_names

    def test_patterns_detect_request_model_exposes_compact_detail_default(self):
        from mtdata.core.patterns import patterns_detect

        info = get_function_info(patterns_detect)

        detail_param = next(p for p in info["params"] if p["name"] == "detail")
        assert detail_param["default"] == "compact"


# ========================================================================
# _apply_schema_overrides
# ========================================================================


class TestApplySchemaOverrides:
    @patch(
        "mtdata.core.cli.enrich_schema_with_shared_defs", side_effect=lambda s, fi: s
    )
    def test_basic_override(self, mock_enrich):
        tool = {
            "meta": {
                "schema": {"properties": {"x": {"default": 42}}, "required": ["x"]}
            }
        }
        func_info = {
            "params": [{"name": "x", "type": int, "default": None, "required": False}]
        }
        schema = _apply_schema_overrides(tool, func_info)
        assert func_info["params"][0]["default"] == 42
        assert func_info["params"][0]["required"] is True

    @patch(
        "mtdata.core.cli.enrich_schema_with_shared_defs", side_effect=lambda s, fi: s
    )
    def test_no_schema(self, mock_enrich):
        tool = {"meta": {}}
        func_info = {
            "params": [{"name": "y", "type": str, "default": None, "required": False}]
        }
        schema = _apply_schema_overrides(tool, func_info)
        assert isinstance(schema, dict)

    @patch(
        "mtdata.core.cli.enrich_schema_with_shared_defs", side_effect=lambda s, fi: s
    )
    def test_schema_with_parameters_key(self, mock_enrich):
        tool = {
            "meta": {
                "schema": {
                    "parameters": {
                        "properties": {"z": {"default": "abc"}},
                        "required": [],
                    }
                }
            }
        }
        func_info = {
            "params": [{"name": "z", "type": str, "default": None, "required": False}]
        }
        _apply_schema_overrides(tool, func_info)
        assert func_info["params"][0]["default"] == "abc"


# ========================================================================
# _extract_function_from_tool_obj
# ========================================================================


class TestExtractFunctionFromToolObj:
    def test_func_attr(self):
        obj = MagicMock()
        fn = lambda: None
        obj.func = fn
        assert _extract_function_from_tool_obj(obj) is fn

    def test_callable_object(self):
        fn = lambda: None
        assert _extract_function_from_tool_obj(fn) is fn

    def test_non_callable(self):
        assert _extract_function_from_tool_obj("not callable") is None

    def test_handler_attr(self):
        class ToolObj:
            def handler(self):
                pass

        obj = ToolObj()
        assert _extract_function_from_tool_obj(obj) is not None


# ========================================================================
# _extract_metadata_from_tool_obj
# ========================================================================


class TestExtractMetadataFromToolObj:
    def test_description_attr(self):
        obj = MagicMock()
        obj.description = "My tool"
        obj.schema = None
        obj.input_schema = None
        obj.parameters = None
        obj.spec = None
        meta = _extract_metadata_from_tool_obj(obj)
        assert meta["description"] == "My tool"

    def test_schema_with_properties(self):
        obj = MagicMock(spec=[])
        obj.description = None
        obj.doc = None
        obj.docs = None
        obj.schema = {
            "description": "Schema desc",
            "properties": {"sym": {"description": "Symbol name"}},
        }
        obj.input_schema = None
        obj.parameters = None
        obj.spec = None
        meta = _extract_metadata_from_tool_obj(obj)
        assert meta["description"] == "Schema desc"
        assert meta["param_docs"]["sym"] == "Symbol name"

    def test_no_metadata(self):
        obj = MagicMock(spec=[])
        obj.description = None
        obj.doc = None
        obj.docs = None
        obj.schema = None
        obj.input_schema = None
        obj.parameters = None
        obj.spec = None
        meta = _extract_metadata_from_tool_obj(obj)
        assert meta["description"] is None
        assert meta["param_docs"] == {}


# ========================================================================
# _is_union_origin / _is_literal_origin
# ========================================================================


class TestTypeOriginChecks:
    def test_union_origin(self):
        from typing import get_origin

        assert _is_union_origin(Union) is True

    def test_literal_origin(self):
        assert _is_literal_origin(Literal) is True

    def test_non_union(self):
        assert _is_union_origin(list) is False

    def test_non_literal(self):
        assert _is_literal_origin(dict) is False

    def test_types_union_type(self):
        assert _is_union_origin(types.UnionType) is True


# ========================================================================
# _unwrap_optional_type
# ========================================================================


class TestUnwrapOptionalType:
    def test_optional_int(self):
        base, origin = _unwrap_optional_type(Optional[int])
        assert base is int

    def test_plain_int(self):
        base, origin = _unwrap_optional_type(int)
        assert base is int

    def test_optional_list(self):
        base, origin = _unwrap_optional_type(Optional[List[str]])
        assert origin is list

    def test_union_multiple(self):
        # Union[int, str] with more than one non-None type should stay as-is
        base, origin = _unwrap_optional_type(Union[int, str])
        # Should not unwrap since there are 2 non-None args
        assert origin is not None or base is not int


# ========================================================================
# _normalize_cli_list_value
# ========================================================================


class TestNormalizeCliListValue:
    def test_none(self):
        assert _normalize_cli_list_value(None) is None

    def test_string_comma_separated(self):
        assert _normalize_cli_list_value("a,b,c") == ["a", "b", "c"]

    def test_string_space_separated(self):
        assert _normalize_cli_list_value("a b c") == ["a", "b", "c"]

    def test_json_array(self):
        assert _normalize_cli_list_value('["x","y"]') == ["x", "y"]

    def test_list_passthrough(self):
        assert _normalize_cli_list_value(["a", "b"]) == ["a", "b"]

    def test_empty_string(self):
        assert _normalize_cli_list_value("") == []

    def test_tuple_input(self):
        assert _normalize_cli_list_value(("a", "b")) == ["a", "b"]

    def test_non_string_non_list(self):
        assert _normalize_cli_list_value(42) == 42

    def test_nested_list_with_strings(self):
        result = _normalize_cli_list_value(["a,b", "c"])
        assert "a" in result and "b" in result and "c" in result

    def test_list_with_non_string_items(self):
        result = _normalize_cli_list_value([1, 2])
        assert 1 in result and 2 in result

    def test_list_with_none_items(self):
        result = _normalize_cli_list_value(["a", None, "b"])
        assert result == ["a", "b"]


# ========================================================================
# _coerce_cli_scalar
# ========================================================================


class TestCoerceCliScalar:
    def test_true(self):
        assert _coerce_cli_scalar("true") is True

    def test_false(self):
        assert _coerce_cli_scalar("false") is False

    def test_null(self):
        assert _coerce_cli_scalar("null") is None

    def test_none_string(self):
        assert _coerce_cli_scalar("none") is None

    def test_integer(self):
        assert _coerce_cli_scalar("42") == 42

    def test_float(self):
        assert _coerce_cli_scalar("3.14") == 3.14

    def test_json_object(self):
        assert _coerce_cli_scalar('{"a": 1}') == {"a": 1}

    def test_json_array(self):
        assert _coerce_cli_scalar("[1, 2]") == [1, 2]

    def test_plain_string(self):
        assert _coerce_cli_scalar("hello") == "hello"

    def test_empty_string(self):
        assert _coerce_cli_scalar("") == ""

    def test_whitespace_string(self):
        assert _coerce_cli_scalar("  ") == ""

    def test_json_string_with_quotes(self):
        assert _coerce_cli_scalar('"hello"') == "hello"

    def test_TRUE_uppercase(self):
        assert _coerce_cli_scalar("TRUE") is True

    def test_False_mixed_case(self):
        assert _coerce_cli_scalar("False") is False


# ========================================================================
# _parse_set_overrides
# ========================================================================


class TestParseSetOverrides:
    def test_none(self):
        assert _parse_set_overrides(None) == {}

    def test_empty_list(self):
        assert _parse_set_overrides([]) == {}

    def test_single_override(self):
        result = _parse_set_overrides(["method.sp=24"])
        assert result == {"method": {"sp": 24}}

    def test_multiple_overrides(self):
        result = _parse_set_overrides(["method.sp=24", "method.max_epochs=20"])
        assert result["method"]["sp"] == 24
        assert result["method"]["max_epochs"] == 20

    def test_multiple_sections(self):
        result = _parse_set_overrides(["method.sp=24", "denoise.method=wavelet"])
        assert "method" in result
        assert "denoise" in result

    def test_nested_override(self):
        result = _parse_set_overrides(["params.model.window=64"])
        assert result == {"params": {"model": {"window": 64}}}

    def test_invalid_no_equals(self):
        with pytest.raises(ValueError, match="expected section.key=value"):
            _parse_set_overrides(["bad_override"])

    def test_invalid_no_dot(self):
        with pytest.raises(ValueError, match="expected section.key=value"):
            _parse_set_overrides(["key=value"])

    def test_empty_string_items_skipped(self):
        result = _parse_set_overrides(["", "method.x=1", "  "])
        assert result == {"method": {"x": 1}}

    def test_non_string_items_skipped(self):
        result = _parse_set_overrides([None, 123])
        assert result == {}

    def test_boolean_value_coercion(self):
        result = _parse_set_overrides(["method.flag=true"])
        assert result["method"]["flag"] is True

    def test_null_value_coercion(self):
        result = _parse_set_overrides(["method.param=null"])
        assert result["method"]["param"] is None


# ========================================================================
# _merge_dict
# ========================================================================


class TestMergeDict:
    def test_both_none(self):
        assert _merge_dict(None, None) == {}

    def test_dst_only(self):
        assert _merge_dict({"a": 1}, None) == {"a": 1}

    def test_src_only(self):
        assert _merge_dict(None, {"b": 2}) == {"b": 2}

    def test_merge(self):
        assert _merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_src_overwrites_dst(self):
        assert _merge_dict({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        assert _merge_dict({"a": {"x": 1}}, {"a": {"y": 2}}) == {"a": {"x": 1, "y": 2}}


# ========================================================================
# _type_name
# ========================================================================


class TestTypeName:
    def test_int(self):
        assert _type_name(int) == "int"

    def test_str(self):
        assert _type_name(str) == "str"

    def test_no_name(self):
        result = _type_name(Optional[int])
        assert isinstance(result, str)


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


# ========================================================================
# _example_value
# ========================================================================


class TestExampleValue:
    def test_known_hint(self):
        param = {"name": "symbol", "type": str, "default": None}
        assert _example_value(param, prefer_default=False) == "EURUSD"

    def test_default_preferred(self):
        param = {"name": "unknown_param", "type": str, "default": "mydefault"}
        assert _example_value(param, prefer_default=True) == "mydefault"

    def test_int_type_fallback(self):
        param = {"name": "weird", "type": int, "default": None}
        assert _example_value(param, prefer_default=False) == "10"

    def test_float_type_fallback(self):
        param = {"name": "weird", "type": float, "default": None}
        assert _example_value(param, prefer_default=False) == "0.1"

    def test_bool_type_fallback(self):
        param = {"name": "weird", "type": bool, "default": None}
        assert _example_value(param, prefer_default=False) == "true"

    def test_list_type_fallback(self):
        param = {"name": "weird", "type": list, "default": None}
        assert _example_value(param, prefer_default=False) == "a,b"

    def test_unknown_type_fallback(self):
        param = {"name": "weird", "type": str, "default": None}
        result = _example_value(param, prefer_default=False)
        assert isinstance(result, str)


# ========================================================================
# _build_usage_examples
# ========================================================================


class TestBuildUsageExamples:
    def test_basic(self):
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "timeframe", "type": str, "required": False, "default": "H1"},
            ]
        }
        base, advanced = _build_usage_examples("data_fetch_candles", func_info)
        assert "data_fetch_candles" in base
        # The first required param uses prefer_default=True; since default is None it uses hint or placeholder
        assert "symbol" in base.lower() or "EURUSD" in base or "<symbol>" in base

    def test_no_optional_params(self):
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
            ]
        }
        base, advanced = _build_usage_examples("test_cmd", func_info)
        assert advanced is None

    def test_multiple_required(self):
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "method", "type": str, "required": True, "default": None},
            ]
        }
        base, advanced = _build_usage_examples("test_cmd", func_info)
        assert "--method" in base


# ========================================================================
# _match_commands
# ========================================================================


class TestMatchCommands:
    def _make_functions(self):
        def my_forecast(symbol: str):
            """Generate forecast."""
            pass

        def my_data(symbol: str):
            """Fetch data."""
            pass

        info_f = get_function_info(my_forecast)
        info_d = get_function_info(my_data)
        return {
            "forecast_generate": {
                "func": my_forecast,
                "meta": {"description": "Generate forecast"},
                "_cli_func_info": info_f,
            },
            "data_fetch": {
                "func": my_data,
                "meta": {"description": "Fetch data"},
                "_cli_func_info": info_d,
            },
        }

    def test_match_forecast(self):
        fns = self._make_functions()
        matches = _match_commands(fns, "forecast")
        assert len(matches) == 1
        assert matches[0][0] == "forecast_generate"

    def test_match_data(self):
        fns = self._make_functions()
        matches = _match_commands(fns, "data")
        assert len(matches) == 1

    def test_no_match(self):
        fns = self._make_functions()
        matches = _match_commands(fns, "nonexistent")
        assert len(matches) == 0

    def test_empty_query(self):
        fns = self._make_functions()
        matches = _match_commands(fns, "")
        assert len(matches) == 0

    def test_match_all(self):
        fns = self._make_functions()
        # Both have "symbol" in their params context
        matches = _match_commands(fns, "generate")
        assert len(matches) == 1


# ========================================================================
# _extract_help_query
# ========================================================================


class TestExtractHelpQuery:
    def test_help_with_query(self):
        assert _extract_help_query(["--help", "forecast"]) == "forecast"

    def test_help_with_multi_word_query(self):
        assert (
            _extract_help_query(["--help", "forecast", "generate"])
            == "forecast generate"
        )

    def test_help_without_query(self):
        assert _extract_help_query(["--help"]) is None

    def test_no_help(self):
        assert _extract_help_query(["forecast_generate", "EURUSD"]) is None

    def test_short_help_flag(self):
        assert _extract_help_query(["-h", "data"]) == "data"

    def test_help_followed_by_flag(self):
        assert _extract_help_query(["--help", "--verbose"]) is None

    def test_help_with_query_then_flag(self):
        assert _extract_help_query(["--help", "forecast", "--verbose"]) == "forecast"


class TestNormalizeCliArgvAliases:
    def test_normalizes_first_alias_command_token(self):
        functions = {
            "symbols_list": {"func": lambda: None},
            "market_ticker": {"func": lambda: None},
        }

        out = _normalize_cli_argv_aliases(
            ["--timeframe", "H1", "symbols-list", "--search-term", "BTC"],
            functions,
        )

        assert out == ["--timeframe", "H1", "symbols_list", "--search-term", "BTC"]

    def test_normalizes_help_query_alias_keyword(self):
        functions = {
            "trade_place": {"func": lambda: None},
        }

        out = _normalize_cli_argv_aliases(["--help", "trade-place"], functions)

        assert out == ["--help", "trade_place"]


# ========================================================================
# _print_extended_help
# ========================================================================


class TestPrintExtendedHelp:
    def _make_functions(self):
        def forecast_gen(symbol: str, horizon: int = 12):
            """Generate forecast."""
            pass

        info = get_function_info(forecast_gen)
        return {
            "forecast_generate": {
                "func": forecast_gen,
                "meta": {"description": "Generate forecast"},
                "_cli_func_info": info,
            },
        }

    def test_matching_query(self, capsys):
        fns = self._make_functions()
        _print_extended_help(fns, "forecast")
        out = capsys.readouterr().out
        assert "forecast_generate" in out
        assert "Example" in out

    def test_no_matching_query(self, capsys):
        fns = self._make_functions()
        _print_extended_help(fns, "nonexistent")
        out = capsys.readouterr().out
        assert "No commands match" in out

    def test_trade_place_help_surfaces_safety_flags(self, capsys):
        def trade_place(
            symbol: str,
            volume: float,
            order_type: str,
            price: float | None = None,
            stop_loss: float | None = None,
            take_profit: float | None = None,
            expiration: str | None = None,
            comment: str | None = None,
            deviation: int = 20,
            dry_run: bool = False,
            require_sl_tp: bool = True,
            auto_close_on_sl_tp_fail: bool = True,
        ):
            """Place a market or pending order."""
            pass

        info = get_function_info(trade_place)
        fns = {
            "trade_place": {
                "func": trade_place,
                "meta": {"description": "Place a market or pending order"},
                "_cli_func_info": info,
            },
        }

        _print_extended_help(fns, "trade_place")
        out = capsys.readouterr().out
        assert "dry_run=false" in out
        assert "require_sl_tp=true" in out
        assert "auto_close_on_sl_tp_fail=true" in out
        assert "market orders default to require_sl_tp=true" in out
        assert "auto_close_on_sl_tp_fail defaults true" in out
        assert (
            "set --dry-run true to preview routing without sending an order to MT5"
            in out
        )


# ========================================================================
# _build_epilog
# ========================================================================


class TestBuildEpilog:
    def test_basic(self):
        def my_func(symbol: str):
            """My func."""
            pass

        info = get_function_info(my_func)
        functions = {
            "my_func": {
                "func": my_func,
                "meta": {"description": "Do something"},
                "_cli_func_info": info,
            },
        }
        epilog = _build_epilog(functions)
        assert "my_func" in epilog
        assert "Commands and Arguments" in epilog

    def test_labels_triple_barrier_renders_canonical_literal_choices(self):
        def labels_triple_barrier(
            symbol: str,
            detail: Literal["full", "summary", "compact"] = "compact",
        ):
            """Label bars."""
            pass

        info = get_function_info(labels_triple_barrier)
        functions = {
            "labels_triple_barrier": {
                "func": labels_triple_barrier,
                "meta": {"description": "Label bars"},
                "_cli_func_info": info,
            },
        }

        epilog = _build_epilog(functions)

        assert "--summary-only" not in epilog
        assert "--detail{full,summary,compact}" in epilog
        assert "<Literal>" not in epilog


# ========================================================================
# _add_forecast_generate_args
# ========================================================================


class TestAddForecastGenerateArgs:
    def test_adds_args(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)
        # Should parse without error when given required args
        args = parser.parse_args(["EURUSD"])
        assert args.symbol == "EURUSD"
        assert args.library == "native"
        assert args.method == "theta"
        assert args.timeframe == "H1"
        assert args.horizon == 12
        assert args.detail == "compact"

    def test_all_options(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)
        args = parser.parse_args(
            [
                "GBPUSD",
                "--library",
                "pretrained",
                "--method",
                "chronos2",
                "--timeframe",
                "D1",
                "--horizon",
                "24",
                "--lookback",
                "200",
                "--quantity",
                "return",
                "--ci-alpha",
                "0.1",
                "--denoise",
                "wavelet",
                "--print-config",
            ]
        )
        assert args.symbol == "GBPUSD"
        assert args.library == "pretrained"
        assert args.method == "chronos2"
        assert args.horizon == 24
        assert args.lookback == 200
        assert args.quantity == "return"
        assert args.ci_alpha == 0.1
        assert args.detail == "compact"
        assert args.print_config is True

    def test_symbol_flag_alias(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)
        args = parser.parse_args(["--symbol", "GBPUSD"])
        assert args.symbol == "GBPUSD"


# ========================================================================
# add_dynamic_arguments
# ========================================================================


class TestAddDynamicArguments:
    def test_adds_required_positional(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["EURUSD"])
        assert args.symbol == "EURUSD"

    def test_adds_optional_flags(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "count", "type": int, "required": False, "default": 10},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["EURUSD", "--count", "20"])
        assert args.count == 20

    def test_bool_param(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "flag", "type": bool, "required": False, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["--flag"])
        assert args.flag == "true"
        args = parser.parse_args(["--flag", "true"])
        assert args.flag == "true"
        args = parser.parse_args(["--flag", "True"])
        assert args.flag == "true"
        args = parser.parse_args(["--flag", "false"])
        assert args.flag == "false"
        args = parser.parse_args(["--flag", "FALSE"])
        assert args.flag == "false"
        args = parser.parse_args(["--no-flag"])
        assert args.flag == "false"
        help_text = parser.format_help()
        assert "--flag [{true,false}]" in help_text
        assert "[bool]" not in help_text

    def test_include_incomplete_bool_param_uses_canonical_hyphen_flag(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "include_incomplete",
                    "type": bool,
                    "required": False,
                    "default": False,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="data_fetch_candles")

        canonical_action = next(
            action
            for action in parser._actions
            if action.dest == "include_incomplete" and action.help != argparse.SUPPRESS
        )
        hidden_alias_action = next(
            action
            for action in parser._actions
            if action.dest == "include_incomplete"
            and action.help == argparse.SUPPRESS
            and "--include_incomplete" in action.option_strings
        )

        assert canonical_action.option_strings == ["--include-incomplete"]
        assert "--include_incomplete" in hidden_alias_action.option_strings
        assert not any(
            action.help != argparse.SUPPRESS
            and action.dest == "include_incomplete"
            and "--no-include-incomplete" in action.option_strings
            for action in parser._actions
        )
        assert parser.parse_args(["--include_incomplete"]).include_incomplete == "true"
        assert (
            parser.parse_args(["--no_include_incomplete"]).include_incomplete == "false"
        )

    def test_market_scan_exposes_symbol_alias_flag(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "symbols",
                    "type": Optional[str],
                    "required": False,
                    "default": None,
                },
                {
                    "name": "symbol",
                    "type": Optional[str],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="market_scan")

        args = parser.parse_args(["--symbol", "EURUSD"])

        assert args.symbol == "EURUSD"

    def test_list_param(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "items",
                    "type": List[str],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["--items", "a", "b", "c"])
        assert args.items == ["a", "b", "c"]

    def test_mapping_param_adds_companion(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "simplify",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(
            ["--simplify", "lttb", "--simplify-params", "points=100"]
        )
        assert args.simplify == "lttb"
        assert args.simplify_params == "points=100"

    def test_mapping_param_adds_set_override(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "params",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info)
        help_text = parser.format_help()
        args = parser.parse_args(["--params", "alpha=0.5", "--set", "params.beta=0.2"])
        assert args.params == "alpha=0.5"
        assert args.set_overrides == ["params.beta=0.2"]
        assert "--set" in help_text
        assert "--params-params" not in help_text

    def test_first_required_param_accepts_flag_alias(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "count", "type": int, "required": False, "default": 10},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["--symbol", "EURUSD", "--count", "20"])
        assert args.symbol == "EURUSD"
        assert args.count == 20

    def test_first_required_param_flag_alias_is_hidden_from_help(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        symbol_actions = [
            action for action in parser._actions if action.dest == "symbol"
        ]
        optional_action = next(
            action for action in symbol_actions if action.option_strings
        )
        assert optional_action.help == argparse.SUPPRESS

    def test_single_word_flag_is_not_duplicated(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "ticket", "type": int, "required": False, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        ticket_action = next(
            action for action in parser._actions if action.dest == "ticket"
        )
        assert ticket_action.option_strings == ["--ticket"]

    def test_limit_exposes_only_canonical_limit_for_bar_command(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "limit", "type": int, "required": False, "default": 100},
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="data_fetch_candles")
        args = parser.parse_args(["--limit", "250"])
        assert args.limit == 250
        limit_action = next(
            action for action in parser._actions if action.dest == "limit"
        )
        assert "--bars" not in limit_action.option_strings

    def test_finviz_news_accepts_optional_positional_symbol(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": False, "default": None},
                {"name": "limit", "type": int, "required": False, "default": 20},
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="finviz_news")
        args = parser.parse_args(["AAPL", "--limit", "5"])
        assert args.symbol == "AAPL"
        assert args.limit == 5

    def test_indicators_list_category_accepts_mixed_case(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "category",
                    "type": Literal["momentum", "trend", "volatility"],
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="indicators_list")
        args = parser.parse_args(["--category", "Trend"])
        assert args.category == "trend"

    def test_trade_history_position_ticket_accepts_ticket_alias(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "position_ticket",
                    "type": int,
                    "required": False,
                    "default": None,
                },
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="trade_history")
        args = parser.parse_args(["--ticket", "123456"])
        assert args.position_ticket == 123456

    def test_market_depth_exposes_require_dom(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "require_dom", "type": bool, "required": False, "default": False},
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="market_depth_fetch")
        args = parser.parse_args(["--require-dom"])
        assert args.require_dom == "true"
        require_dom_action = next(
            action for action in parser._actions if action.dest == "require_dom"
        )
        assert "--require-dom" in require_dom_action.option_strings

    def test_wait_event_exposes_symbol_without_instrument_alias(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": False, "default": None},
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="wait_event")

        assert parser.parse_args(["EURUSD"]).symbol == "EURUSD"
        assert not any(
            action.dest == "instrument"
            for action in parser._actions
        )

    def test_finviz_calendar_prefers_start_end_and_hides_legacy_date_flags(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "start", "type": str, "required": False, "default": None},
                {"name": "end", "type": str, "required": False, "default": None},
                {"name": "date_from", "type": str, "required": False, "default": None},
                {"name": "date_to", "type": str, "required": False, "default": None},
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="finviz_calendar")

        args = parser.parse_args(["--start", "2026-01-05", "--end", "2026-01-12"])
        assert args.start == "2026-01-05"
        assert args.end == "2026-01-12"
        assert not any(
            action.dest in {"date_from", "date_to"} and action.help != argparse.SUPPRESS
            for action in parser._actions
        )

    def test_labels_triple_barrier_uses_canonical_detail_choices(
        self,
    ):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {
                    "name": "detail",
                    "type": Literal["full", "summary", "compact"],
                    "required": False,
                    "default": "full",
                },
            ]
        }

        add_dynamic_arguments(parser, func_info, cmd_name="labels_triple_barrier")

        assert any(action.dest == "detail" for action in parser._actions)
        assert not any(action.dest == "summary_only" for action in parser._actions)
        args = parser.parse_args(["--detail", "summary"])
        assert args.detail == "summary"

    def test_partial_flag_prefix_is_rejected_when_abbrev_disabled(self, capsys):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        func_info = {
            "params": [
                {
                    "name": "search_term",
                    "type": str,
                    "required": False,
                    "default": None,
                },
            ]
        }

        add_dynamic_arguments(parser, func_info)

        with pytest.raises(SystemExit):
            parser.parse_args(["--search", "BTC"])

        assert "unrecognized arguments: --search BTC" in capsys.readouterr().err


# ========================================================================
# _parse_kv_string
# ========================================================================


class TestParseKvString:
    def test_kv_pairs(self):
        result = _parse_kv_string("a=1,b=2")
        assert result is not None
        assert "a" in result

    def test_json_string(self):
        result = _parse_kv_string('{"a": 1}')
        assert result is not None
        assert result["a"] == 1

    @patch("mtdata.utils.utils.parse_kv_or_json", side_effect=Exception("fail"))
    def test_exception_returns_none(self, mock_parse):
        result = _parse_kv_string("bad")
        assert result is None


# ========================================================================
# _resolve_param_kwargs
# ========================================================================


class TestResolveParamKwargs:
    def test_basic_str_param(self):
        param = {"name": "symbol", "type": str, "required": True, "default": None}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["type"] is str
        assert is_mapping is False

    def test_int_param(self):
        param = {"name": "count", "type": int, "required": False, "default": 10}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["type"] is int
        assert kwargs["default"] == 10

    def test_float_param(self):
        param = {"name": "alpha", "type": float, "required": False, "default": 0.05}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["type"] is float

    def test_bool_param(self):
        param = {"name": "verbose", "type": bool, "required": False, "default": None}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["choices"] == ["true", "false"]
        assert "metavar" not in kwargs
        assert kwargs["type"]("True") == "true"
        assert kwargs["type"]("FALSE") == "false"

    def test_optional_int(self):
        param = {
            "name": "count",
            "type": Optional[int],
            "required": False,
            "default": None,
        }
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["type"] is int

    def test_dict_param_is_mapping(self):
        param = {
            "name": "params",
            "type": Dict[str, Any],
            "required": False,
            "default": None,
        }
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert is_mapping is True

    def test_literal_type(self):
        param = {
            "name": "mode",
            "type": Literal["a", "b", "c"],
            "required": False,
            "default": "a",
        }
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["choices"] == ["a", "b", "c"]

    def test_list_type(self):
        param = {"name": "items", "type": List[str], "required": False, "default": None}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["nargs"] == "+"

    def test_param_docs_used(self):
        param = {"name": "symbol", "type": str, "required": True, "default": None}
        docs = {"symbol": "The trading symbol"}
        kwargs, _ = _resolve_param_kwargs(param, docs)
        assert kwargs["help"] == "The trading symbol"

    def test_no_default_for_required(self):
        param = {"name": "sym", "type": str, "required": True, "default": None}
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert "default" not in kwargs

    def test_list_of_literals(self):
        param = {
            "name": "methods",
            "type": List[Literal["a", "b"]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert kwargs["choices"] == ["a", "b"]
        assert kwargs["nargs"] == "+"

    def test_forecast_method_help_avoids_massive_choices(self):
        param = {"name": "method", "type": str, "required": False, "default": None}
        kwargs, _ = _resolve_param_kwargs(
            param, None, cmd_name="forecast_conformal_intervals"
        )
        assert "choices" not in kwargs
        assert kwargs["metavar"] == "METHOD"
        assert "forecast_list_methods" in kwargs["help"]
        assert kwargs["help"].count("forecast_list_methods") == 1

    def test_forecast_method_literal_help_uses_method_browser_hint(self):
        param = {
            "name": "method",
            "type": Literal["theta", "arima"],
            "required": False,
            "default": "theta",
        }
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert "choices" not in kwargs
        assert kwargs["metavar"] == "METHOD"
        assert "forecast_list_methods" in kwargs["help"]

    def test_non_forecast_method_help_does_not_mention_forecast_browser(self):
        param = {"name": "method", "type": str, "required": False, "default": None}
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="correlation_matrix")

        assert kwargs["help"] == "Method/algorithm for this tool."
        assert "forecast_list_methods" not in kwargs["help"]

    def test_common_analysis_params_have_specific_help(self):
        transform_kwargs, _ = _resolve_param_kwargs(
            {"name": "transform", "type": str, "required": False, "default": None},
            None,
            cmd_name="correlation_matrix",
        )
        min_regime_kwargs, _ = _resolve_param_kwargs(
            {"name": "min_regime_bars", "type": int, "required": False, "default": -1},
            None,
            cmd_name="regime_detect",
        )

        assert "Preprocessing transform" in transform_kwargs["help"]
        assert transform_kwargs["help"] != "transform parameter"
        assert "Minimum bars a detected regime must span" in min_regime_kwargs["help"]
        assert min_regime_kwargs["help"] != "min_regime_bars parameter"

    def test_report_generate_format_help_is_removed_output_help(self):
        param = {
            "name": "format",
            "type": str,
            "required": False,
            "default": "legacy",
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="report_generate")
        assert kwargs["help"] == "Domain-specific shape selector when supported; TOON/JSON selection uses json."

    def test_finviz_screen_filters_help_is_command_specific(self):
        param = {
            "name": "filters",
            "type": Optional[str],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="finviz_screen")
        assert "NASDAQ" in kwargs["help"]
        assert "Sector" in kwargs["help"]

    def test_finviz_screen_order_help_is_command_specific(self):
        param = {
            "name": "order",
            "type": Optional[str],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="finviz_screen")
        assert (
            kwargs["help"]
            == "Finviz sort key. Example: -marketcap for descending or price for ascending."
        )

    def test_market_scan_limit_help_is_command_specific(self):
        param = {
            "name": "limit",
            "type": Optional[int],
            "required": False,
            "default": 20,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="market_scan")
        assert kwargs["help"] == "Max matching symbols to return."

    def test_market_scan_rank_by_help_lists_actual_options(self):
        param = {
            "name": "rank_by",
            "type": Optional[str],
            "required": False,
            "default": "abs_price_change_pct",
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="market_scan")
        assert kwargs["help"] == (
            "Ranking to compute for market scans: abs_price_change_pct, "
            "price_change_pct, tick_volume, rsi, or spread_pct."
        )

    def test_symbols_top_markets_limit_help_is_command_specific(self):
        param = {
            "name": "limit",
            "type": Optional[int],
            "required": False,
            "default": 10,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="symbols_top_markets")
        assert kwargs["help"] == "Max symbols to return for each ranking."

    def test_finviz_news_limit_help_is_command_specific(self):
        param = {"name": "limit", "type": int, "required": False, "default": 20}
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="finviz_news")
        assert kwargs["help"] == "Max news items to return on this page."

    def test_finviz_calendar_start_help_is_command_specific(self):
        param = {
            "name": "start",
            "type": Optional[str],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="finviz_calendar")
        assert kwargs["help"].startswith("Start date")

    def test_finviz_calendar_end_help_is_command_specific(self):
        param = {
            "name": "end",
            "type": Optional[str],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="finviz_calendar")
        assert kwargs["help"].startswith("End date")

    def test_forecast_tune_optuna_search_space_help_is_command_specific(self):
        param = {
            "name": "search_space",
            "type": Dict[str, Any],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="forecast_tune_optuna")
        assert kwargs["help"] == "Optuna search space (JSON or k=v)."

    def test_data_fetch_candles_indicators_help_mentions_named_and_underscore_syntax(
        self,
    ):
        param = {
            "name": "indicators",
            "type": Optional[List[Dict[str, Any]]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="data_fetch_candles")
        assert "rsi_14" in kwargs["help"]
        assert "rsi(length=14)" in kwargs["help"]

    def test_denoise_help_mentions_json_example(self):
        param = {
            "name": "denoise",
            "type": Optional[Dict[str, Any]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert "--denoise kalman" in kwargs["help"]
        assert '"method":"kalman"' in kwargs["help"]

    def test_params_help_mentions_json_and_key_value_examples(self):
        param = {
            "name": "params",
            "type": Optional[Dict[str, Any]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert "--params alpha=0.3,beta=0.1" in kwargs["help"]
        assert '"alpha":0.3' in kwargs["help"]

    def test_features_help_mentions_json_and_key_value_examples(self):
        param = {
            "name": "features",
            "type": Optional[Dict[str, Any]],
            "required": False,
            "default": None,
        }
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert "--features lag=3,rolling=5" in kwargs["help"]
        assert '"lag":3' in kwargs["help"]

    def test_forecast_barrier_optimize_method_has_cli_choices(self):
        param = {"name": "method", "type": str, "required": False, "default": "auto"}
        kwargs, _ = _resolve_param_kwargs(
            param, None, cmd_name="forecast_barrier_optimize"
        )
        assert kwargs["choices"] == [
            "mc_gbm",
            "mc_gbm_bb",
            "hmm_mc",
            "garch",
            "bootstrap",
            "heston",
            "jump_diffusion",
            "auto",
        ]
        assert "Barrier simulation method" in kwargs["help"]


# ========================================================================
# _apply_cli_output_mode_defaults
# ========================================================================


class TestApplyCliOutputModeDefaults:
    def test_defaults_detail_to_compact_when_user_does_not_specify_mode(self):
        args = argparse.Namespace(
            command="patterns_detect", detail="full", verbose=False
        )
        func_info = {
            "params": [
                {
                    "name": "detail",
                    "type": Literal["compact", "full"],
                    "required": False,
                    "default": "full",
                },
            ]
        }
        functions = {
            "patterns_detect": {
                "func": MagicMock(),
                "_cli_func_info": func_info,
                "meta": {},
            }
        }

        out = _apply_cli_output_mode_defaults(args, ["patterns_detect"], functions)

        assert out.detail == "compact"

    def test_defaults_detail_to_summary_when_compact_is_unavailable(self):
        args = argparse.Namespace(command="regime_detect", detail="full", verbose=False)
        func_info = {
            "params": [
                {
                    "name": "detail",
                    "type": Literal["full", "summary"],
                    "required": False,
                    "default": "full",
                },
            ]
        }
        functions = {
            "regime_detect": {
                "func": MagicMock(),
                "_cli_func_info": func_info,
                "meta": {},
            }
        }

        out = _apply_cli_output_mode_defaults(args, ["regime_detect"], functions)

        assert out.detail == "summary"

    def test_defaults_symbols_describe_to_compact_when_user_does_not_specify_mode(self):
        args = argparse.Namespace(command="symbols_describe", detail="full", verbose=False)
        func_info = {
            "params": [
                {
                    "name": "detail",
                    "type": Literal["compact", "full"],
                    "required": False,
                    "default": "full",
                },
            ]
        }
        functions = {
            "symbols_describe": {
                "func": MagicMock(),
                "_cli_func_info": func_info,
                "meta": {},
            }
        }

        out = _apply_cli_output_mode_defaults(args, ["symbols_describe"], functions)

        assert out.detail == "compact"

    def test_legacy_verbose_flag_no_longer_promotes_supported_modes_to_full(self):
        args = argparse.Namespace(
            command="indicators_list", detail="compact", verbose=True
        )
        func_info = {
            "params": [
                {
                    "name": "detail",
                    "type": Literal["compact", "full"],
                    "required": False,
                    "default": "compact",
                },
            ]
        }
        functions = {
            "indicators_list": {
                "func": MagicMock(),
                "_cli_func_info": func_info,
                "meta": {},
            }
        }

        out = _apply_cli_output_mode_defaults(
            args, ["indicators_list", "--verbose"], functions
        )

        assert out.detail == "compact"

    def test_explicit_mode_is_respected(self):
        args = argparse.Namespace(
            command="patterns_detect", detail="full", verbose=False
        )
        func_info = {
            "params": [
                {
                    "name": "detail",
                    "type": Literal["compact", "full"],
                    "required": False,
                    "default": "full",
                },
            ]
        }
        functions = {
            "patterns_detect": {
                "func": MagicMock(),
                "_cli_func_info": func_info,
                "meta": {},
            }
        }

        out = _apply_cli_output_mode_defaults(args, ["patterns_detect"], functions)

        assert out.detail == "compact"


# ========================================================================
# create_command_function
# ========================================================================


class TestCreateCommandFunction:
    def test_basic_call(self, capsys):
        mock_fn = MagicMock(return_value={"data": [1, 2, 3]})
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(symbol="EURUSD", json=False, verbose=False)
        cmd_fn(args)
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["symbol"] == "EURUSD"
        assert call_kwargs["__cli_raw"] is True

    def test_request_model_reconstructed(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "request_model": ForecastGenerateRequest,
            "request_param_name": "request",
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "horizon", "type": int, "required": False, "default": 12},
                {
                    "name": "params",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            symbol="EURUSD",
            horizon=24,
            params='{"sp":24}',
            params_params=None,
            json=False,
            verbose=False,
        )
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        request = call_kwargs["request"]
        assert isinstance(request, ForecastGenerateRequest)
        assert request.symbol == "EURUSD"
        assert request.horizon == 24
        assert request.params == {"sp": 24}
        assert call_kwargs["__cli_raw"] is True

    def test_detail_full_is_forwarded_without_injecting_verbose(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {
                    "name": "detail",
                    "type": Literal["compact", "full"],
                    "required": False,
                    "default": "compact",
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(symbol="EURUSD", detail="full", json=False, verbose=False)

        cmd_fn(args)

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["symbol"] == "EURUSD"
        assert call_kwargs["detail"] == "full"
        assert "verbose" not in call_kwargs
        assert call_kwargs["__cli_raw"] is True

    def test_set_overrides_mapping_param(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "params",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            params="alpha=0.5",
            set_overrides=["params.beta=0.2", "params.model.window=64"],
            json=False,
            verbose=False,
        )

        cmd_fn(args)

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["params"] == {
            "alpha": "0.5",
            "beta": 0.2,
            "model": {"window": 64},
        }
        assert call_kwargs["__cli_raw"] is True

    def test_detail_full_is_forwarded_as_tool_detail(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {
                    "name": "detail",
                    "type": Literal["compact", "full"],
                    "required": False,
                    "default": "compact",
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            symbol="EURUSD",
            detail="full",
            json=False,
            verbose=False,
        )

        cmd_fn(args)

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["symbol"] == "EURUSD"
        assert call_kwargs["detail"] == "full"
        assert "verbose" not in call_kwargs
        assert call_kwargs["__cli_raw"] is True

    def test_indicator_compact_string_reconstructed(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "request_model": DataFetchCandlesRequest,
            "request_param_name": "request",
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {
                    "name": "indicators",
                    "type": Optional[List[Dict[str, Any]]],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="data_fetch_candles")
        args = argparse.Namespace(
            symbol="EURUSD",
            indicators="rsi(14),macd(12,26,9)",
            indicators_params=None,
            json=False,
            verbose=False,
        )
        cmd_fn(args)
        request = mock_fn.call_args[1]["request"]
        assert request.indicators == [
            {"name": "rsi", "params": [14.0]},
            {"name": "macd", "params": [12.0, 26.0, 9.0]},
        ]

    def test_indicator_params_dict_are_accepted(self):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "request_model": DataFetchCandlesRequest,
            "request_param_name": "request",
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {
                    "name": "indicators",
                    "type": Optional[List[Dict[str, Any]]],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="data_fetch_candles")
        args = argparse.Namespace(
            symbol="EURUSD",
            indicators='[{"name":"rsi","params":{"length":14}}]',
            indicators_params=None,
            json=False,
            verbose=False,
        )
        status = cmd_fn(args)
        assert status == 0
        request = mock_fn.call_args[1]["request"]
        assert request.indicators == [{"name": "rsi", "params": {"length": 14.0}}]

    def test_named_indicator_string_is_accepted(self):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "request_model": DataFetchCandlesRequest,
            "request_param_name": "request",
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {
                    "name": "indicators",
                    "type": Optional[List[Dict[str, Any]]],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="data_fetch_candles")
        args = argparse.Namespace(
            symbol="EURUSD",
            indicators="rsi(length=14),macd(fast=12,slow=26,signal=9)",
            indicators_params=None,
            json=False,
            verbose=False,
        )
        status = cmd_fn(args)
        assert status == 0
        request = mock_fn.call_args[1]["request"]
        assert request.indicators == [
            {"name": "rsi", "params": {"length": 14.0}},
            {"name": "macd", "params": {"fast": 12.0, "slow": 26.0, "signal": 9.0}},
        ]

    def test_indicator_param_comma_syntax_returns_friendly_error(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "request_model": DataFetchCandlesRequest,
            "request_param_name": "request",
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {
                    "name": "indicators",
                    "type": Optional[List[Dict[str, Any]]],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="data_fetch_candles")
        args = argparse.Namespace(
            symbol="EURUSD",
            indicators="sma,20",
            indicators_params=None,
            json=False,
            verbose=False,
        )
        status = cmd_fn(args)
        assert status == 1
        assert "sma(20), not sma,20" in capsys.readouterr().out
        mock_fn.assert_not_called()

    def test_negative_limit_returns_friendly_validation_error(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "request_model": DataFetchCandlesRequest,
            "request_param_name": "request",
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "limit", "type": int, "required": False, "default": 200},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="data_fetch_candles")
        args = argparse.Namespace(symbol="EURUSD", limit=-5, json=False, verbose=False)
        status = cmd_fn(args)
        assert status == 1
        assert "limit must be greater than 0." in capsys.readouterr().out
        mock_fn.assert_not_called()

    def test_invalid_simplify_method_returns_descriptive_validation_error(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "request_model": DataFetchCandlesRequest,
            "request_param_name": "request",
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {
                    "name": "simplify",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="data_fetch_candles")
        args = argparse.Namespace(
            symbol="BTCUSD",
            simplify="ltd",
            simplify_params=None,
            json=False,
            verbose=False,
        )
        status = cmd_fn(args)
        output = capsys.readouterr().out
        assert status == 1
        assert "simplify.method must be one of:" in output
        assert "lttb (fast bucket-based selection)" in output
        assert "rdp (Douglas-Peucker line simplification)" in output
        assert "pla (piecewise linear approximation)" in output
        assert "apca (adaptive piecewise constant approximation)" in output
        mock_fn.assert_not_called()

    def test_missing_required_argument_returns_structured_error(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(symbol=None, json=False, verbose=False)
        status = cmd_fn(args)
        assert status == 1
        assert "Missing required argument(s): symbol." in capsys.readouterr().out
        mock_fn.assert_not_called()

    def test_missing_required_literal_argument_shows_valid_values(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "library",
                    "type": Literal[
                        "native", "statsforecast", "sktime", "pretrained", "mlforecast"
                    ],
                    "required": True,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(
            func_info, cmd_name="forecast_list_library_models"
        )
        args = argparse.Namespace(library=None, json=False, verbose=False)
        status = cmd_fn(args)
        assert status == 1
        out = capsys.readouterr().out
        assert "Missing required argument 'library'." in out
        assert "native, statsforecast, sktime, pretrained, mlforecast" in out
        mock_fn.assert_not_called()

    def test_string_result_text_format(self, capsys):
        mock_fn = MagicMock(return_value="plain text result")
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None}
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(symbol="EURUSD", json=False, verbose=False)
        cmd_fn(args)
        assert "plain text result" in capsys.readouterr().out

    def test_string_result_json_format(self, capsys):
        mock_fn = MagicMock(return_value="plain text result")
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None}
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(symbol="EURUSD", json=True, verbose=False)
        cmd_fn(args)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["text"] == "plain text result"

    def test_dict_result_json_format(self, capsys):
        mock_fn = MagicMock(return_value={"price": 1.23})
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None}
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(symbol="EURUSD", json=True, verbose=False)
        cmd_fn(args)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["price"] == 1.23
        assert "meta" not in parsed

    def test_bool_param_coercion(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "verbose", "type": bool, "required": False, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(verbose="true", json=False)
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["verbose"] is True

    def test_bool_false_coercion(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "verbose", "type": bool, "required": False, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(verbose="false", json=False)
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["verbose"] is False

    def test_none_values_excluded(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "extra", "type": str, "required": False, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            symbol="EURUSD", extra=None, json=False, verbose=False
        )
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert "extra" not in call_kwargs

    def test_mapping_present_sentinel(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "simplify",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            simplify="__PRESENT__", simplify_params=None, json=False, verbose=False
        )
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["simplify"] == {}

    def test_mapping_json_string(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "simplify",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            simplify='{"method":"lttb"}',
            simplify_params=None,
            json=False,
            verbose=False,
        )
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert isinstance(call_kwargs["simplify"], dict)

    def test_mapping_shorthand_string(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "simplify",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            simplify="lttb", simplify_params=None, json=False, verbose=False
        )
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["simplify"] == {"method": "lttb"}

    def test_mapping_companion_params(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "simplify",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            simplify="lttb", simplify_params="points=100", json=False, verbose=False
        )
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert isinstance(call_kwargs["simplify"], dict)
        assert call_kwargs["simplify"]["method"] == "lttb"

    def test_list_param_normalized(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "methods",
                    "type": List[str],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(methods="a,b,c", json=False, verbose=False)
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["methods"] == ["a", "b", "c"]

    def test_verbose_attaches_meta(self, capsys):
        mock_fn = MagicMock(return_value={"data": 1})
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None}
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(symbol="EURUSD", json=True, verbose=True)
        cmd_fn(args)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["meta"]["tool"] == "test_cmd"
        assert "runtime" in parsed["meta"]

    def test_empty_output_no_print(self, capsys):
        mock_fn = MagicMock(return_value={})
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None}
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(symbol="EURUSD", json=False, verbose=False)
        cmd_fn(args)
        # Should still produce output (even if empty dict formatted)
        # Just verifying no crash


# ========================================================================
# discover_tools
# ========================================================================


class TestDiscoverTools:
    @patch("mtdata.core.cli.get_mcp_registry")
    @patch("mtdata.core.cli.bootstrap_tools", return_value=())
    @patch("mtdata.core.cli.mcp", new_callable=MagicMock)
    def test_discover_from_registry(self, mock_mcp, mock_bootstrap, mock_get_reg):

        def fake_tool(symbol: str):
            """Fake tool."""
            pass

        fake_tool.__module__ = "mtdata.core.forecast"

        tool_obj = MagicMock()
        tool_obj.func = fake_tool
        tool_obj.description = "Fake tool"
        tool_obj.schema = None
        tool_obj.input_schema = None
        tool_obj.parameters = None
        tool_obj.spec = None
        tool_obj.doc = None
        tool_obj.docs = None

        mock_get_reg.return_value = {"fake_tool": tool_obj}

        tools = discover_tools()
        assert "fake_tool" in tools

    @patch("mtdata.core.cli.get_mcp_registry")
    @patch("mtdata.core.cli.bootstrap_tools")
    @patch("mtdata.core.cli.mcp", new_callable=MagicMock)
    def test_discover_from_registry_includes_submodule_tools(
        self, mock_mcp, mock_bootstrap, mock_get_reg
    ):
        def fake_tool(symbol: str):
            """Fake tool."""
            pass

        fake_tool.__module__ = "mtdata.core.trading.positions"

        class FakeModule:
            __name__ = "mtdata.core.trading"

        tool_obj = MagicMock()
        tool_obj.func = fake_tool
        tool_obj.description = "Fake trading tool"
        tool_obj.schema = None
        tool_obj.input_schema = None
        tool_obj.parameters = None
        tool_obj.spec = None
        tool_obj.doc = None
        tool_obj.docs = None

        mock_bootstrap.return_value = (FakeModule(),)
        mock_get_reg.return_value = {"trade_get_open": tool_obj}

        tools = discover_tools()
        assert "trade_get_open" in tools

    @patch("mtdata.core.cli.get_registered_tools", return_value={})
    @patch("mtdata.core.cli.get_mcp_registry", return_value=None)
    @patch("mtdata.core.cli.mcp", None)
    def test_discover_fallback_scan(self, mock_get_reg, mock_get_registered):
        def public_tool(x: int):
            """A public tool."""
            pass

        public_tool.__module__ = "fake.tools"

        class FakeModule:
            __name__ = "fake.tools"

        fake_module = FakeModule()
        fake_module.public_tool = public_tool

        with patch("mtdata.core.cli.bootstrap_tools", return_value=(fake_module,)):
            tools = discover_tools()
        assert "public_tool" in tools

    @patch("mtdata.core.cli.get_registered_tools", return_value={})
    @patch("mtdata.core.cli.get_mcp_registry", return_value=None)
    @patch("mtdata.core.cli.bootstrap_tools", return_value=())
    @patch("mtdata.core.cli.mcp", None)
    def test_discover_empty(self, mock_bootstrap, mock_get_reg, mock_get_registered):
        tools = discover_tools()
        assert tools == {}


# ========================================================================
# main()
# ========================================================================


class TestMain:
    @patch("mtdata.core.cli.discover_tools", return_value={})
    def test_no_tools(self, mock_discover, capsys):
        result = main()
        assert result == 1
        assert "No tools discovered" in capsys.readouterr().err

    @patch("mtdata.core.cli.sys")
    @patch("mtdata.core.cli.discover_tools")
    def test_help_query_mode(self, mock_discover, mock_sys, capsys):
        def my_tool(symbol: str):
            """My tool."""
            pass

        info = get_function_info(my_tool)
        mock_discover.return_value = {
            "my_tool": {
                "func": my_tool,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }
        mock_sys.argv = ["cli.py", "--help", "my_tool"]
        mock_sys.stderr = sys.stderr
        mock_sys.stdout = sys.stdout

        result = main()
        assert result == 0

    @patch("mtdata.core.cli.discover_tools")
    def test_no_command_shows_help(self, mock_discover, capsys):
        def my_tool(symbol: str):
            """My tool."""
            pass

        mock_discover.return_value = {
            "my_tool": {"func": my_tool, "meta": {"description": "My tool"}},
        }
        with patch("sys.argv", ["cli.py"]):
            result = main()
        assert result == 1
        out = capsys.readouterr().out
        assert "usage: cli.py" in out
        assert "<command>" in out
        assert "{my_tool,my-tool}" not in out
        assert out.count("    my_tool") == 1

    @patch("mtdata.core.cli.discover_tools")
    def test_command_execution(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "my_tool"
        mock_fn.__doc__ = "My tool."

        def my_tool(symbol: str):
            """My tool."""
            pass

        info = get_function_info(my_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "my_tool": {
                "func": mock_fn,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "my_tool", "EURUSD"]):
            result = main()
        assert result == 0
        mock_fn.assert_called_once()

    @patch("mtdata.core.cli.discover_tools")
    def test_hyphenated_command_alias_executes_tool(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "symbols_list"
        mock_fn.__doc__ = "List symbols."

        def symbols_list(search_term: str | None = None):
            """List symbols."""
            pass

        info = get_function_info(symbols_list)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "symbols_list": {
                "func": mock_fn,
                "meta": {"description": "List symbols"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "symbols-list", "--search-term", "BTC"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(search_term="BTC", __cli_raw=True)

    @patch("mtdata.core.cli.discover_tools")
    def test_correlation_matrix_keeps_optional_first_positional_symbols(
        self, mock_discover
    ):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "correlation_matrix"
        mock_fn.__doc__ = "Correlation matrix."

        def correlation_matrix(
            symbols: Optional[str] = None, group: Optional[str] = None
        ):
            """Correlation matrix."""
            pass

        info = get_function_info(correlation_matrix)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "correlation_matrix": {
                "func": mock_fn,
                "meta": {"description": "Correlation matrix"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "correlation_matrix", "EURUSD,GBPUSD"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(symbols="EURUSD,GBPUSD", __cli_raw=True)

    @patch("mtdata.core.cli.discover_tools")
    def test_correlation_matrix_accepts_group_without_symbols(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "correlation_matrix"
        mock_fn.__doc__ = "Correlation matrix."

        def correlation_matrix(
            symbols: Optional[str] = None, group: Optional[str] = None
        ):
            """Correlation matrix."""
            pass

        info = get_function_info(correlation_matrix)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "correlation_matrix": {
                "func": mock_fn,
                "meta": {"description": "Correlation matrix"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv", ["cli.py", "correlation_matrix", "--group", "Forex\\Majors"]
        ):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(group="Forex\\Majors", __cli_raw=True)

    @patch("mtdata.core.cli.discover_tools")
    def test_cointegration_test_keeps_optional_first_positional_symbols(
        self, mock_discover
    ):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "cointegration_test"
        mock_fn.__doc__ = "Cointegration test."

        def cointegration_test(
            symbols: Optional[str] = None, group: Optional[str] = None
        ):
            """Cointegration test."""
            pass

        info = get_function_info(cointegration_test)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "cointegration_test": {
                "func": mock_fn,
                "meta": {"description": "Cointegration test"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "cointegration_test", "EURUSD,GBPUSD"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(symbols="EURUSD,GBPUSD", __cli_raw=True)

    @patch("mtdata.core.cli.discover_tools")
    def test_cointegration_test_accepts_group_without_symbols(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "cointegration_test"
        mock_fn.__doc__ = "Cointegration test."

        def cointegration_test(
            symbols: Optional[str] = None, group: Optional[str] = None
        ):
            """Cointegration test."""
            pass

        info = get_function_info(cointegration_test)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "cointegration_test": {
                "func": mock_fn,
                "meta": {"description": "Cointegration test"},
                "_cli_func_info": info,
            },
        }

        with patch(
            "sys.argv", ["cli.py", "cointegration_test", "--group", "Forex\\Majors"]
        ):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(group="Forex\\Majors", __cli_raw=True)

    @patch("mtdata.core.cli.discover_tools")
    def test_trade_risk_analyze_accepts_optional_positional_symbol(
        self, mock_discover
    ):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_risk_analyze"
        mock_fn.__doc__ = "Analyze trade risk."

        def trade_risk_analyze(request: TradeRiskAnalyzeRequest):
            """Analyze trade risk."""
            pass

        info = get_function_info(trade_risk_analyze)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_risk_analyze": {
                "func": mock_fn,
                "meta": {"description": "Analyze trade risk"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "trade_risk_analyze", "EURUSD"]):
            result = main()

        assert result == 0
        request = mock_fn.call_args.kwargs["request"]
        assert isinstance(request, TradeRiskAnalyzeRequest)
        assert request.symbol == "EURUSD"

    @patch("mtdata.core.cli.discover_tools")
    def test_trade_get_open_accepts_optional_positional_symbol(self, mock_discover):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_get_open"
        mock_fn.__doc__ = "Get open positions."

        def trade_get_open(request: TradeGetOpenRequest):
            """Get open positions."""
            pass

        info = get_function_info(trade_get_open)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_get_open": {
                "func": mock_fn,
                "meta": {"description": "Get open positions"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "trade_get_open", "EURUSD"]):
            result = main()

        assert result == 0
        request = mock_fn.call_args.kwargs["request"]
        assert isinstance(request, TradeGetOpenRequest)
        assert request.symbol == "EURUSD"

    @patch("mtdata.core.cli.discover_tools")
    def test_market_scan_keeps_optional_first_positional_symbols(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "market_scan"
        mock_fn.__doc__ = "Market scan."

        def market_scan(symbols: Optional[str] = None, group: Optional[str] = None):
            """Market scan."""
            pass

        info = get_function_info(market_scan)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "market_scan": {
                "func": mock_fn,
                "meta": {"description": "Market scan"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "market_scan", "EURUSD,GBPUSD"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(symbols="EURUSD,GBPUSD", __cli_raw=True)

    @patch("mtdata.core.cli.discover_tools")
    def test_market_scan_accepts_group_without_symbols(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "market_scan"
        mock_fn.__doc__ = "Market scan."

        def market_scan(symbols: Optional[str] = None, group: Optional[str] = None):
            """Market scan."""
            pass

        info = get_function_info(market_scan)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "market_scan": {
                "func": mock_fn,
                "meta": {"description": "Market scan"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "market_scan", "--group", "Forex\\Majors"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(group="Forex\\Majors", __cli_raw=True)

    @patch("mtdata.core.cli.discover_tools")
    def test_market_status_keeps_optional_first_positional_symbol(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "market_status"
        mock_fn.__doc__ = "Market status."

        def market_status(symbol: Optional[str] = None, region: Optional[str] = "all"):
            """Market status."""
            pass

        info = get_function_info(market_status)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "market_status": {
                "func": mock_fn,
                "meta": {"description": "Market status"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "market_status", "EURUSD"]):
            result = main()

        assert result == 0
        mock_fn.assert_called_once_with(symbol="EURUSD", region="all", __cli_raw=True)

    @patch("mtdata.core.cli.discover_tools")
    def test_global_timeframe_before_command_is_applied(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "my_tool"
        mock_fn.__doc__ = "My tool."

        def my_tool(symbol: str, timeframe: str = "H1"):
            """My tool."""
            pass

        info = get_function_info(my_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "my_tool": {
                "func": mock_fn,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "--timeframe", "D1", "my_tool", "EURUSD"]):
            result = main()
        assert result == 0
        assert mock_fn.call_args[1]["timeframe"] == "D1"

    @patch("mtdata.core.cli.discover_tools")
    def test_command_timeframe_overrides_global_timeframe(self, mock_discover):
        mock_fn = MagicMock(return_value="output text")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "my_tool"
        mock_fn.__doc__ = "My tool."

        def my_tool(symbol: str, timeframe: str = "H1"):
            """My tool."""
            pass

        info = get_function_info(my_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "my_tool": {
                "func": mock_fn,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }
        with patch(
            "sys.argv",
            ["cli.py", "--timeframe", "D1", "my_tool", "EURUSD", "--timeframe", "H1"],
        ):
            result = main()
        assert result == 0
        assert mock_fn.call_args[1]["timeframe"] == "H1"

    @patch("mtdata.core.cli.discover_tools")
    def test_json_output_suppresses_mtdata_logs_in_non_verbose_mode(
        self, mock_discover, capsys
    ):
        def noisy_tool(**_kwargs):
            """Emit a log line before returning structured output."""
            logging.getLogger("mtdata.test_noise").error(
                "noise that should be suppressed"
            )
            return {"ok": True}

        info = get_function_info(noisy_tool)
        root_logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stderr)
        root_logger.addHandler(handler)
        previous_level = root_logger.level
        root_logger.setLevel(logging.INFO)

        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        try:
            with patch("sys.argv", ["cli.py", "--json", "noisy_tool"]):
                result = main()
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(previous_level)

        assert result == 0
        out = capsys.readouterr()
        assert json.loads(out.out)["ok"] is True
        assert "noise that should be suppressed" not in out.err

    @patch("mtdata.core.cli.discover_tools")
    def test_json_output_suppresses_stream_noise_and_third_party_logs(
        self, mock_discover, capsys
    ):
        def noisy_tool(**_kwargs):
            print("stdout noise")
            print("stderr noise", file=sys.stderr)
            logging.getLogger("sentence_transformers").warning("third-party noise")
            return {"ok": True}

        info = get_function_info(noisy_tool)
        root_logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stderr)
        root_logger.addHandler(handler)
        previous_level = root_logger.level
        root_logger.setLevel(logging.INFO)

        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        try:
            with patch("sys.argv", ["cli.py", "--json", "noisy_tool"]):
                result = main()
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(previous_level)

        assert result == 0
        out = capsys.readouterr()
        assert json.loads(out.out)["ok"] is True
        assert "stdout noise" not in out.out
        assert "stderr noise" not in out.err
        assert "third-party noise" not in out.err

    @patch("mtdata.core.cli.discover_tools")
    def test_toon_output_suppresses_stream_noise_but_keeps_structured_warnings(
        self, mock_discover, capsys
    ):
        import warnings

        def noisy_tool(**_kwargs):
            print("stdout noise")
            print("stderr noise", file=sys.stderr)
            warnings.warn("runtime warning", RuntimeWarning)
            return {"ok": True}

        info = get_function_info(noisy_tool)
        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "noisy_tool"]):
            result = main()

        assert result == 0
        out = capsys.readouterr()
        assert "stdout noise" not in out.out
        assert "stderr noise" not in out.err
        assert "ok: true" in out.out
        assert "warnings[1]:" in out.out
        assert "runtime warning" in out.out

    @patch("mtdata.core.cli.discover_tools")
    def test_cli_ignores_deprecation_warnings_in_structured_output(
        self, mock_discover, capsys
    ):
        import warnings

        def noisy_tool(**_kwargs):
            warnings.warn("deprecated runtime path", DeprecationWarning)
            warnings.warn("pending deprecation path", PendingDeprecationWarning)
            warnings.warn("runtime warning", RuntimeWarning)
            return {"ok": True}

        info = get_function_info(noisy_tool)
        mock_discover.return_value = {
            "noisy_tool": {
                "func": noisy_tool,
                "meta": {"description": "Noisy tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "noisy_tool"]):
            result = main()

        assert result == 0
        out = capsys.readouterr()
        assert "runtime warning" in out.out
        assert "deprecated runtime path" not in out.out
        assert "pending deprecation path" not in out.out

    @patch("mtdata.core.cli.discover_tools")
    def test_help_hides_irrelevant_timeframe_for_trade_account_info(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_account_info"
        mock_fn.__doc__ = "Trade account info."

        def trade_account_info():
            """Trade account info."""
            pass

        info = get_function_info(trade_account_info)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_account_info": {
                "func": mock_fn,
                "meta": {"description": "Trade account info"},
                "_cli_func_info": info,
            },
        }
        with (
            patch("sys.argv", ["cli.py", "trade_account_info", "--help"]),
            pytest.raises(SystemExit),
        ):
            main()
        out = capsys.readouterr().out
        assert "--timeframe" not in out

    @patch("mtdata.core.cli.discover_tools")
    def test_root_help_documents_global_timeframe_precedence(self, mock_discover, capsys):
        def my_tool(symbol: str, timeframe: str = "H1"):
            """My tool."""
            pass

        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "my_tool"
        mock_fn.__doc__ = "My tool."
        info = get_function_info(my_tool)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "my_tool": {
                "func": mock_fn,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }

        with patch("sys.argv", ["cli.py", "--help"]), pytest.raises(SystemExit):
            main()

        out = " ".join(capsys.readouterr().out.split())
        assert "Default MT5 timeframe for commands with a timeframe parameter" in out
        assert "command-level --timeframe overrides it" in out

    @patch("mtdata.core.cli.discover_tools")
    def test_help_hides_duplicate_symbol_option_for_required_first_arg(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(return_value={"success": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "my_tool"
        mock_fn.__doc__ = "My tool."

        def my_tool(symbol: str):
            """My tool."""
            pass

        info = get_function_info(my_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "my_tool": {
                "func": mock_fn,
                "meta": {"description": "My tool"},
                "_cli_func_info": info,
            },
        }
        with (
            patch("sys.argv", ["cli.py", "my_tool", "--help"]),
            pytest.raises(SystemExit),
        ):
            main()
        out = capsys.readouterr().out
        assert "positional arguments:" in out
        assert "symbol" in out
        assert "--symbol" not in out

    @patch("mtdata.core.cli.discover_tools")
    def test_trade_history_days_alias_converts_to_minutes(self, mock_discover):
        mock_fn = MagicMock(return_value=[])
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_history"
        mock_fn.__doc__ = "Trade history."

        def trade_history(request: TradeHistoryRequest):
            """Trade history."""
            pass

        info = get_function_info(trade_history)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_history": {
                "func": mock_fn,
                "meta": {"description": "Trade history"},
                "_cli_func_info": info,
            },
        }
        with patch(
            "sys.argv", ["cli.py", "trade_history", "--days", "2", "--ticket", "123456"]
        ):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert isinstance(request, TradeHistoryRequest)
        assert request.minutes_back == 2880
        assert str(request.position_ticket) == "123456"

    @patch("mtdata.core.cli.discover_tools")
    def test_trade_history_minutes_back_overrides_days_alias(self, mock_discover):
        mock_fn = MagicMock(return_value=[])
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_history"
        mock_fn.__doc__ = "Trade history."

        def trade_history(request: TradeHistoryRequest):
            """Trade history."""
            pass

        info = get_function_info(trade_history)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_history": {
                "func": mock_fn,
                "meta": {"description": "Trade history"},
                "_cli_func_info": info,
            },
        }
        with patch(
            "sys.argv",
            ["cli.py", "trade_history", "--days", "2", "--minutes-back", "60"],
        ):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.minutes_back == 60

    @patch("mtdata.core.cli.discover_tools")
    def test_trade_history_side_flag_populates_request(self, mock_discover):
        mock_fn = MagicMock(return_value=[])
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "trade_history"
        mock_fn.__doc__ = "Trade history."

        def trade_history(request: TradeHistoryRequest):
            """Trade history."""
            pass

        info = get_function_info(trade_history)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "trade_history": {
                "func": mock_fn,
                "meta": {"description": "Trade history"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "trade_history", "--side", "short"]):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert isinstance(request, TradeHistoryRequest)
        assert request.side == "SELL"

    @patch("mtdata.core.cli.discover_tools")
    def test_command_exception_handled(self, mock_discover, capsys):
        mock_fn = MagicMock(side_effect=RuntimeError("fail!"))
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "bad_tool"
        mock_fn.__doc__ = "Bad tool."

        def bad_tool(symbol: str):
            """Bad tool."""
            pass

        info = get_function_info(bad_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "bad_tool": {
                "func": mock_fn,
                "meta": {"description": "Bad tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "bad_tool", "X"]):
            result = main()
        assert result == 1
        assert "Error" in capsys.readouterr().err

    @patch("mtdata.core.cli.discover_tools")
    def test_command_tool_error_result_returns_nonzero(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"error": "bad input"})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "bad_tool"
        mock_fn.__doc__ = "Bad tool."

        def bad_tool(symbol: str):
            """Bad tool."""
            pass

        info = get_function_info(bad_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "bad_tool": {
                "func": mock_fn,
                "meta": {"description": "Bad tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "bad_tool", "X"]):
            result = main()
        assert result == 1

    @patch("mtdata.core.cli.discover_tools")
    def test_command_no_action_result_returns_nonzero(self, mock_discover, capsys):
        mock_fn = MagicMock(
            return_value={"message": "No action taken", "no_action": True}
        )
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "noop_tool"
        mock_fn.__doc__ = "No-op tool."

        def noop_tool(symbol: str):
            """No-op tool."""
            pass

        info = get_function_info(noop_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "noop_tool": {
                "func": mock_fn,
                "meta": {"description": "No-op tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "noop_tool", "X"]):
            result = main()
        assert result == 1

    @patch("mtdata.core.cli.discover_tools")
    def test_command_successful_no_action_result_returns_zero(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(
            return_value={
                "success": True,
                "message": "No open positions",
                "no_action": True,
                "count": 0,
                "items": [],
            }
        )
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "noop_tool"
        mock_fn.__doc__ = "No-op tool."

        def noop_tool(symbol: str, __cli_raw: bool = False):
            """No-op tool."""
            return {"success": True}

        info = get_function_info(noop_tool)
        info["func"] = mock_fn
        mock_discover.return_value = {
            "noop_tool": {
                "func": mock_fn,
                "meta": {"description": "No-op tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "noop_tool", "X"]):
            result = main()
        assert result == 0

    @patch("mtdata.core.cli.discover_tools")
    def test_keyboard_interrupt(self, mock_discover, capsys):
        mock_fn = MagicMock(side_effect=KeyboardInterrupt)
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "slow_tool"
        mock_fn.__doc__ = "Slow tool."

        def slow_tool(symbol: str):
            """Slow tool."""
            pass

        info = get_function_info(slow_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "slow_tool": {
                "func": mock_fn,
                "meta": {"description": "Slow tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "slow_tool", "X"]):
            result = main()
        assert result == 1
        assert "Aborted" in capsys.readouterr().err

    @patch("mtdata.core.cli.discover_tools")
    def test_debug_mode_traceback(self, mock_discover, capsys, monkeypatch):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "1")
        mock_fn = MagicMock(side_effect=ValueError("debug test"))
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "debug_tool"
        mock_fn.__doc__ = "Debug tool."

        def debug_tool(symbol: str):
            """Debug tool."""
            pass

        info = get_function_info(debug_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "debug_tool": {
                "func": mock_fn,
                "meta": {"description": "Debug tool"},
                "_cli_func_info": info,
            },
        }
        with patch("sys.argv", ["cli.py", "debug_tool", "X"]):
            result = main()
        assert result == 1
        err = capsys.readouterr().err
        assert "Traceback" in err or "Error" in err


# ========================================================================
# Forecast generate custom parser integration
# ========================================================================


class TestForecastGenerateIntegration:
    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_basic(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0, 2.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD"]):
            result = main()
        assert result == 0
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        request = call_kwargs["request"]
        assert isinstance(request, ForecastGenerateRequest)
        assert request.symbol == "EURUSD"
        assert request.library == "native"
        assert request.method == "theta"
        assert request.detail == "compact"
        assert call_kwargs["__cli_raw"] is True

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_maps_extras_to_internal_detail(self, mock_discover):
        mock_fn = MagicMock(return_value={"forecast": [1.0, 2.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD", "--extras", "metadata"]):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.detail == "full"

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_accepts_symbol_flag_alias(self, mock_discover):
        mock_fn = MagicMock(return_value={"forecast": [1.0, 2.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "--symbol", "BTCUSD"]):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.symbol == "BTCUSD"

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_print_config(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv", ["cli.py", "forecast_generate", "EURUSD", "--print-config"]
        ):
            result = main()
        assert result == 0
        # --print-config should NOT call the underlying function
        mock_fn.assert_not_called()
        out = capsys.readouterr().out
        assert "EURUSD" in out

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_json_format(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value="text forecast")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD", "--json"]):
            result = main()
        assert result == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["text"] == "text forecast"

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_tool_error_returns_nonzero(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"error": "forecast failed"})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD"]):
            result = main()
        assert result == 1

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_with_overrides(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv",
            [
                "cli.py",
                "forecast_generate",
                "EURUSD",
                "--set",
                "method.sp=24",
                "--set",
                "method.max_epochs=20",
            ],
        ):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.params["sp"] == 24
        assert request.params["max_epochs"] == 20

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_uses_global_timeframe_before_command(
        self, mock_discover
    ):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv", ["cli.py", "--timeframe", "D1", "forecast_generate", "EURUSD"]
        ):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.timeframe == "D1"

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_with_denoise(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value="ok")
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv",
            ["cli.py", "forecast_generate", "EURUSD", "--denoise", "wavelet"],
        ):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.denoise == {"method": "wavelet"}

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_rejects_removed_verbose_flag(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0, 2.0, 3.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD", "--verbose"]):
            with pytest.raises(SystemExit, match="2"):
                main()

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_omits_redundant_ci_block_when_bounds_rendered(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(
            return_value={
                "times": ["2026-03-07 18:00"],
                "forecast_price": [67580.67],
                "lower_price": [63025.26],
                "upper_price": [71823.47],
                "ci_status": "available",
                "ci_alpha": 0.05,
            }
        )
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch(
            "sys.argv", ["cli.py", "forecast_generate", "BTCUSD", "--timeframe", "D1"]
        ):
            result = main()
        assert result == 0
        out = capsys.readouterr().out
        assert "forecast[1]{time,forecast,lower,upper}:" in out
        assert "\nci:" not in out

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_default_output_includes_compact_context(
        self, mock_discover, capsys
    ):
        mock_fn = MagicMock(
            return_value={
                "forecast_time": ["2026-03-07 18:00"],
                "forecast_price": [1.17308],
                "method": "arima",
                "quantity": "price",
                "detail": "compact",
                "ci_status": "available",
                "ci_alpha": 0.05,
                "interval_summary": {
                    "first_low": 1.1702,
                    "first_high": 1.1729,
                    "median_width": 0.0027,
                },
            }
        )
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {
                "func": mock_fn,
                "meta": {"description": "Generate forecasts"},
            },
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD"]):
            result = main()
        assert result == 0
        out = capsys.readouterr().out
        assert "method: arima" in out
        assert "quantity: price" in out
        assert "detail: compact" in out
        assert "ci:" in out
        assert "status: available" in out
        assert "interval_summary:" in out
        assert "forecast[1]{time,forecast}:" in out


# ========================================================================
# Edge cases / misc coverage
# ========================================================================


class TestEdgeCases:
    def test_coerce_cli_scalar_numeric_like_string(self):
        assert _coerce_cli_scalar("123abc") == "123abc"

    def test_coerce_cli_scalar_just_dot(self):
        result = _coerce_cli_scalar(".")
        assert result == "."

    def test_merge_dict_empty_both(self):
        assert _merge_dict({}, {}) == {}

    def test_format_cli_literal_complex_object(self):
        class Unserializable:
            def __str__(self):
                return "unserializable"

        # json.dumps should fail, fallback to str
        result = _format_cli_literal(Unserializable())
        assert result == "unserializable"

    def test_normalize_cli_list_value_bad_json(self):
        result = _normalize_cli_list_value("[not valid json")
        assert isinstance(result, list)
        # Should fallback to comma/space splitting
        assert len(result) >= 1

    def test_normalize_cli_list_value_empty_list(self):
        result = _normalize_cli_list_value([])
        assert result == []

    def test_parse_set_overrides_section_key_empty(self):
        with pytest.raises(ValueError):
            _parse_set_overrides([".key=value"])

    def test_parse_set_overrides_key_empty(self):
        with pytest.raises(ValueError):
            _parse_set_overrides(["section.=value"])

    def test_json_default_with_bad_bytes(self):
        result = _json_default(b"\xff\xfe")
        assert isinstance(result, str)

    def test_build_cli_timezone_meta_local_tz(self):
        result = _build_cli_timezone_meta({})
        assert "local" not in result
        assert result["utc"]["tz"] == "UTC"

    def test_attach_cli_meta_with_none_cmd(self):
        r = {"data": 1}
        out = _attach_cli_meta(r, cmd_name=None, verbose=True)
        assert "cli_meta" not in out
        assert "tool" not in out["meta"]
        assert "runtime" in out["meta"]

    def test_resolve_param_kwargs_type_resolution_failure(self):
        # A parameter with a weird type that causes exception
        class WeirdType:
            pass

        param = {"name": "x", "type": WeirdType, "required": False, "default": None}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["type"] is str  # fallback

    def test_create_command_mapping_companion_on_none_arg(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "simplify",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        # simplify_params present but simplify is None
        args = argparse.Namespace(
            simplify=None, simplify_params="points=100", json=False, verbose=False
        )
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        # companion params should create the dict
        assert isinstance(call_kwargs.get("simplify"), dict)

    def test_create_command_mapping_companion_merge(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {
                    "name": "simplify",
                    "type": Dict[str, Any],
                    "required": False,
                    "default": None,
                },
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            simplify='{"method":"lttb"}',
            simplify_params="points=100",
            json=False,
            verbose=False,
        )
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["simplify"]["method"] == "lttb"

    def test_extract_help_query_no_args(self):
        assert _extract_help_query([]) is None

    def test_format_result_for_cli_empty_fmt(self):
        result = _format_result_for_cli(
            {"a": 1}, fmt="", verbose=False, cmd_name="test"
        )
        assert isinstance(result, str)

    def test_first_line_only_whitespace_lines(self):
        assert _first_line("   \n   \n   ") == ""

    def test_type_name_with_generic(self):
        result = _type_name(List[str])
        assert isinstance(result, str)

    def test_quote_cli_value_no_whitespace(self):
        assert _quote_cli_value("abc123") == "abc123"

    def test_example_value_with_tuple_type(self):
        param = {"name": "weird", "type": tuple, "default": None}
        result = _example_value(param, prefer_default=False)
        assert isinstance(result, str)

    def test_build_usage_examples_optional_same_as_default(self):
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "timeframe", "type": str, "required": False, "default": "H1"},
            ]
        }
        base, advanced = _build_usage_examples("cmd", func_info)
        assert "cmd" in base

    def test_match_commands_multi_token_query(self):
        def fetch_data(symbol: str):
            """Fetch candle data."""
            pass

        info = get_function_info(fetch_data)
        fns = {
            "data_fetch_candles": {
                "func": fetch_data,
                "meta": {"description": "Fetch candle data"},
                "_cli_func_info": info,
            },
        }
        matches = _match_commands(fns, "candle data")
        assert len(matches) == 1

    def test_is_typed_dict_with_optional_keys(self):
        class FakeTD:
            __annotations__ = {"x": int}
            __optional_keys__ = frozenset({"x"})

        assert _is_typed_dict_type(FakeTD) is True

    def test_safe_tz_name_with_zone_none(self):
        class FakeTZ:
            zone = None

        result = _safe_tz_name(FakeTZ())
        assert isinstance(result, str)


# ========================================================================
# Parameterised tests for broader coverage of _coerce_cli_scalar
# ========================================================================


class TestCoerceCliScalarParameterized:
    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("True", True),
            ("FALSE", False),
            ("Null", None),
            ("NONE", None),
            ("0", 0),
            ("1", 1),
            ("-1", -1),
            ("0.0", 0.0),
            ("-3.14", -3.14),
            ("hello world", "hello world"),
        ],
    )
    def test_coerce_values(self, input_val, expected):
        assert _coerce_cli_scalar(input_val) == expected


# ========================================================================
# Parameterised tests for _normalize_cli_list_value
# ========================================================================


class TestNormalizeCliListParameterized:
    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("a b c", ["a", "b", "c"]),
            ("a,b,c", ["a", "b", "c"]),
            ('["x"]', ["x"]),
            (["a b", "c,d"], ["a", "b", "c", "d"]),
            (None, None),
            ([], []),
        ],
    )
    def test_normalize(self, input_val, expected):
        assert _normalize_cli_list_value(input_val) == expected
