"""Integration tests for mtdata.core.cli module.

Tests main entry point, command execution, and end-to-end CLI workflows.
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
    _argparse_color_enabled,
    _build_epilog,
    _build_usage_examples,
    _coerce_cli_scalar,
    _extract_help_query,
    _first_line,
    _format_cli_literal,
    _format_result_for_cli,
    _is_typed_dict_type,
    _json_default,
    _match_commands,
    _merge_dict,
    _normalize_cli_list_value,
    _parse_set_overrides,
    _print_extended_help,
    _quote_cli_value,
    _safe_tz_name,
    _suggest_commands,
    _type_name,
    create_command_function,
    get_function_info,
    main,
)


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
    def test_cli_ignores_third_party_future_warnings_in_structured_output(
        self, mock_discover, capsys
    ):
        import warnings

        def noisy_tool(**_kwargs):
            warnings.warn_explicit(
                "pandas concat behavior is deprecated",
                FutureWarning,
                filename=r"C:\env\Lib\site-packages\finvizfinance\screener\base.py",
                lineno=134,
            )
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
        assert "pandas concat behavior is deprecated" not in out.out
        assert "site-packages" not in out.out

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
        from mtdata.core.cli import _build_cli_timezone_meta
        result = _build_cli_timezone_meta({})
        assert "local" not in result
        assert result["utc"]["tz"] == "UTC"

    def test_attach_cli_meta_with_none_cmd(self):
        from mtdata.core.cli import _attach_cli_meta
        r = {"data": 1}
        out = _attach_cli_meta(r, cmd_name=None, verbose=True)
        assert "cli_meta" not in out
        assert "tool" not in out["meta"]
        assert "runtime" in out["meta"]

    def test_resolve_param_kwargs_type_resolution_failure(self):
        # A parameter with a weird type that causes exception
        from mtdata.core.cli import _resolve_param_kwargs
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
        from mtdata.core.cli import _example_value
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
