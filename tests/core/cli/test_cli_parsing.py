"""Argument parsing tests for mtdata.core.cli module.

Tests argument parsing, parameter coercion, and CLI input normalization.
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
    _coerce_cli_scalar,
    _example_value,
    _merge_dict,
    _normalize_cli_argv_aliases,
    _normalize_cli_list_value,
    _parse_kv_string,
    _parse_set_overrides,
    _resolve_param_kwargs,
    add_dynamic_arguments,
)


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

    def test_news_accepts_optional_positional_symbol(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "symbol", "type": str, "required": False, "default": None},
                {"name": "detail", "type": str, "required": False, "default": "compact"},
                {"name": "limit", "type": int, "required": False, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info, cmd_name="news")
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
# _normalize_cli_argv_aliases
# ========================================================================


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
