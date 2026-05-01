"""Tool discovery tests for mtdata.core.cli module.

Tests tool discovery, function introspection, and metadata extraction.
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
    _apply_schema_overrides,
    _extract_function_from_tool_obj,
    _extract_metadata_from_tool_obj,
    _is_typed_dict_type,
    _is_literal_origin,
    _is_union_origin,
    _type_name,
    _unwrap_optional_type,
    create_command_function,
    discover_tools,
    get_function_info,
)


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

    def test_is_typed_dict_with_optional_keys(self):
        class FakeTD:
            __annotations__ = {"x": int}
            __optional_keys__ = frozenset({"x"})

        assert _is_typed_dict_type(FakeTD) is True


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
