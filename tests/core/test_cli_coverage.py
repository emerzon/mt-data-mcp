"""Comprehensive tests for mtdata.core.cli module.

Covers helper functions, argument parsing, tool discovery, command creation,
output formatting, and the main() entry point. All external MCP tool calls
and heavy imports are mocked.
"""

import argparse
import json
import os
import sys
import types
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from unittest.mock import MagicMock, patch, call

import pytest

from mtdata.forecast.requests import ForecastGenerateRequest
from mtdata.core.data_requests import DataFetchCandlesRequest

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
    _debug_enabled,
    _debug,
    _argparse_color_enabled,
    _is_typed_dict_type,
    _format_result_minimal,
    _json_default,
    _format_result_for_cli,
    _safe_tz_name,
    _build_cli_timezone_meta,
    _attach_cli_meta,
    get_function_info,
    _apply_schema_overrides,
    _extract_function_from_tool_obj,
    _extract_metadata_from_tool_obj,
    _is_union_origin,
    _is_literal_origin,
    _resolve_param_kwargs,
    _parse_kv_string,
    _unwrap_optional_type,
    _normalize_cli_list_value,
    _coerce_cli_scalar,
    _parse_set_overrides,
    _merge_dict,
    create_command_function,
    _type_name,
    _first_line,
    _format_cli_literal,
    _quote_cli_value,
    _example_value,
    _build_usage_examples,
    _match_commands,
    _extract_help_query,
    _print_extended_help,
    _build_epilog,
    _add_forecast_generate_args,
    add_dynamic_arguments,
    discover_tools,
    main,
)


# ========================================================================
# _debug_enabled / _debug
# ========================================================================

class TestDebugEnabled:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("MTDATA_CLI_DEBUG", raising=False)
        assert _debug_enabled() is False

    def test_enabled_with_1(self, monkeypatch):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "1")
        assert _debug_enabled() is True

    def test_enabled_with_true(self, monkeypatch):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "true")
        assert _debug_enabled() is True

    def test_disabled_with_0(self, monkeypatch):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "0")
        assert _debug_enabled() is False

    def test_disabled_with_false(self, monkeypatch):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "false")
        assert _debug_enabled() is False

    def test_disabled_with_no(self, monkeypatch):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "no")
        assert _debug_enabled() is False

    def test_disabled_with_empty(self, monkeypatch):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "")
        assert _debug_enabled() is False

    def test_enabled_with_yes(self, monkeypatch):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "yes")
        assert _debug_enabled() is True


class TestDebug:
    def test_debug_prints_when_enabled(self, monkeypatch, capsys):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "1")
        _debug("test message")
        assert "test message" in capsys.readouterr().err

    def test_debug_silent_when_disabled(self, monkeypatch, capsys):
        monkeypatch.delenv("MTDATA_CLI_DEBUG", raising=False)
        _debug("should not appear")
        assert capsys.readouterr().err == ""


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

    @patch("mtdata.core.cli._shared_minimal", side_effect=Exception("boom"))
    def test_fallback_on_exception(self, mock_shared):
        result = _format_result_minimal({"key": "value"})
        assert "key" in result


# ========================================================================
# _format_result_for_cli
# ========================================================================

class TestFormatResultForCli:
    def test_json_format(self):
        result = _format_result_for_cli({"a": 1}, fmt="json", verbose=False, cmd_name="test")
        parsed = json.loads(result)
        assert parsed["a"] == 1

    def test_json_format_with_datetime(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _format_result_for_cli({"dt": dt}, fmt="json", verbose=False, cmd_name="test")
        parsed = json.loads(result)
        assert "2025" in parsed["dt"]

    @patch("mtdata.core.cli._shared_minimal", return_value="minimal output")
    def test_toon_format(self, mock_shared):
        result = _format_result_for_cli({"a": 1}, fmt="toon", verbose=False, cmd_name="test")
        assert result == "minimal output"

    @patch("mtdata.core.cli._shared_minimal", side_effect=TypeError("bad"))
    def test_toon_format_fallback(self, mock_shared):
        result = _format_result_for_cli({"a": 1}, fmt="toon", verbose=False, cmd_name="test_cmd")
        assert isinstance(result, str)

    def test_trade_command_no_simplify_numbers(self):
        # trade_ commands should NOT simplify numbers
        result = _format_result_for_cli({"price": 1.23456}, fmt="json", verbose=False, cmd_name="trade_place")
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
        result = _format_result_for_cli("hello", fmt=None, verbose=False, cmd_name="test")
        assert isinstance(result, str)


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


# ========================================================================
# _build_cli_timezone_meta
# ========================================================================

class TestBuildCliTimezoneMeta:
    @patch("mtdata.core.config.mt5_config")
    def test_with_config(self, mock_config):
        mock_config.server_tz_name = "Europe/Nicosia"
        mock_config.client_tz_name = "US/Eastern"
        from zoneinfo import ZoneInfo

        mock_config.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
        mock_config.get_client_tz.return_value = ZoneInfo("US/Eastern")
        mock_config.get_time_offset_seconds.return_value = 7200
        result = _build_cli_timezone_meta({"some": "data"})
        assert isinstance(result, dict)
        assert result["server"]["tz"]["configured"] == "Europe/Nicosia"
        assert result["client"]["tz"]["configured"] == "US/Eastern"
        assert result["server"]["tz"]["offset_seconds"] == 7200
        assert result["server"]["now"] != result["client"]["now"]

    @patch("mtdata.core.config.mt5_config")
    def test_with_dict_result_containing_timezone(self, mock_config):
        mock_config.server_tz_name = None
        mock_config.client_tz_name = None
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 0
        result = _build_cli_timezone_meta({"timezone": "US/Eastern"})
        assert result["output"]["tz"]["value"] == "US/Eastern"

    @patch("mtdata.core.config.mt5_config")
    def test_with_non_dict_result(self, mock_config):
        mock_config.server_tz_name = None
        mock_config.client_tz_name = None
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 0
        result = _build_cli_timezone_meta("some string")
        assert result["output"]["tz"].get("value") is None

    @patch("mtdata.core.config.mt5_config")
    def test_with_offset_env(self, mock_config, monkeypatch):
        mock_config.server_tz_name = None
        mock_config.client_tz_name = None
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 10800
        monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "180")
        result = _build_cli_timezone_meta({})
        assert result["server"]["tz"]["offset_seconds"] == 10800

    @patch("mtdata.core.config.mt5_config")
    def test_with_invalid_offset_env(self, mock_config, monkeypatch):
        mock_config.server_tz_name = None
        mock_config.client_tz_name = None
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 0
        monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "abc")
        result = _build_cli_timezone_meta({})
        assert result["server"].get("tz", {}).get("offset_seconds") is None

    def test_server_source_none_by_default(self):
        result = _build_cli_timezone_meta({})
        assert result["server"]["source"] in ("none", "MT5_SERVER_TZ", "MT5_TIME_OFFSET_MINUTES")

    def test_now_fields_are_iso_strings(self):
        result = _build_cli_timezone_meta({})
        assert isinstance(result["utc"]["now"], str)
        datetime.fromisoformat(result["utc"]["now"])
        if result["server"].get("now") is not None:
            datetime.fromisoformat(result["server"]["now"])
        if result.get("client", {}).get("now") is not None:
            datetime.fromisoformat(result["client"]["now"])

    @patch("mtdata.core.config.mt5_config")
    def test_offset_env_source(self, mock_config, monkeypatch):
        mock_config.server_tz_name = None
        mock_config.client_tz_name = None
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 3600
        monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "60")
        result = _build_cli_timezone_meta({})
        if result["server"].get("tz", {}).get("configured") is None:
            assert result["server"]["source"] == "MT5_TIME_OFFSET_MINUTES"

    @patch("mtdata.core.config.mt5_config")
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
    def test_non_verbose_adds_brief_meta(self):
        r = {"key": "val"}
        out = _attach_cli_meta(r, cmd_name="test", verbose=False)
        assert out is not r
        assert "cli_meta" in out
        assert out["cli_meta"]["command"] == "test"
        tz_meta = out["cli_meta"]["timezone"]
        assert "server" in tz_meta
        assert "client" in tz_meta
        assert "utc" in tz_meta
        assert "local" not in tz_meta
        assert "tz" in tz_meta["server"]
        assert "tz" in tz_meta["client"]
        assert "value" in tz_meta["server"]["tz"]
        assert "value" in tz_meta["client"]["tz"]
        assert "now" in tz_meta["utc"]

    def test_non_dict_returns_unchanged(self):
        assert _attach_cli_meta("string", cmd_name="test", verbose=True) == "string"

    def test_verbose_dict_adds_meta(self):
        r = {"data": 1}
        out = _attach_cli_meta(r, cmd_name="my_cmd", verbose=True)
        assert "cli_meta" in out
        assert out["cli_meta"]["command"] == "my_cmd"
        assert "timezone" in out["cli_meta"]

    def test_preserves_existing_cli_meta(self):
        r = {"data": 1, "cli_meta": {"existing": True}}
        out = _attach_cli_meta(r, cmd_name="cmd", verbose=True)
        assert out["cli_meta"]["existing"] is True
        assert out["cli_meta"]["command"] == "cmd"

    def test_verbose_candles_adds_command_diagnostics(self):
        r = {
            "meta": {
                "diagnostics": {
                    "query": {"rows_returned": 5, "latency_ms": 12.3},
                    "indicators": {"requested": False},
                    "session_gaps": {"count": 0},
                }
            },
            "warnings": ["sample warning"],
        }
        out = _attach_cli_meta(r, cmd_name="data_fetch_candles", verbose=True)
        cmd_diag = out["cli_meta"]["command_diagnostics"]["data_fetch_candles"]
        assert cmd_diag["query"]["rows_returned"] == 5
        assert cmd_diag["indicators"]["requested"] is False
        assert cmd_diag["warnings"] == ["sample warning"]

    def test_verbose_market_ticker_adds_command_diagnostics(self):
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
        cmd_diag = out["cli_meta"]["command_diagnostics"]["market_ticker"]
        assert cmd_diag["source"] == "mt5.symbol_info_tick"
        assert cmd_diag["cache_used"] is False
        assert cmd_diag["query_latency_ms"] == 7.5
        assert cmd_diag["spread_points"] == 100.0


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


# ========================================================================
# _apply_schema_overrides
# ========================================================================

class TestApplySchemaOverrides:
    @patch("mtdata.core.cli.enrich_schema_with_shared_defs", side_effect=lambda s, fi: s)
    def test_basic_override(self, mock_enrich):
        tool = {"meta": {"schema": {"properties": {"x": {"default": 42}}, "required": ["x"]}}}
        func_info = {"params": [{"name": "x", "type": int, "default": None, "required": False}]}
        schema = _apply_schema_overrides(tool, func_info)
        assert func_info["params"][0]["default"] == 42
        assert func_info["params"][0]["required"] is True

    @patch("mtdata.core.cli.enrich_schema_with_shared_defs", side_effect=lambda s, fi: s)
    def test_no_schema(self, mock_enrich):
        tool = {"meta": {}}
        func_info = {"params": [{"name": "y", "type": str, "default": None, "required": False}]}
        schema = _apply_schema_overrides(tool, func_info)
        assert isinstance(schema, dict)

    @patch("mtdata.core.cli.enrich_schema_with_shared_defs", side_effect=lambda s, fi: s)
    def test_schema_with_parameters_key(self, mock_enrich):
        tool = {"meta": {"schema": {"parameters": {"properties": {"z": {"default": "abc"}}, "required": []}}}}
        func_info = {"params": [{"name": "z", "type": str, "default": None, "required": False}]}
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
            "properties": {"sym": {"description": "Symbol name"}}
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
        assert _coerce_cli_scalar('[1, 2]') == [1, 2]

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
        result = _parse_set_overrides(["model.sp=24"])
        assert result == {"model": {"sp": 24}}

    def test_multiple_overrides(self):
        result = _parse_set_overrides(["model.sp=24", "model.max_epochs=20"])
        assert result["model"]["sp"] == 24
        assert result["model"]["max_epochs"] == 20

    def test_multiple_sections(self):
        result = _parse_set_overrides(["model.sp=24", "denoise.method=wavelet"])
        assert "model" in result
        assert "denoise" in result

    def test_invalid_no_equals(self):
        with pytest.raises(ValueError, match="expected section.key=value"):
            _parse_set_overrides(["bad_override"])

    def test_invalid_no_dot(self):
        with pytest.raises(ValueError, match="expected section.key=value"):
            _parse_set_overrides(["key=value"])

    def test_empty_string_items_skipped(self):
        result = _parse_set_overrides(["", "model.x=1", "  "])
        assert result == {"model": {"x": 1}}

    def test_non_string_items_skipped(self):
        result = _parse_set_overrides([None, 123])
        assert result == {}

    def test_boolean_value_coercion(self):
        result = _parse_set_overrides(["model.flag=true"])
        assert result["model"]["flag"] is True

    def test_null_value_coercion(self):
        result = _parse_set_overrides(["model.param=null"])
        assert result["model"]["param"] is None


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
            "forecast_generate": {"func": my_forecast, "meta": {"description": "Generate forecast"}, "_cli_func_info": info_f},
            "data_fetch": {"func": my_data, "meta": {"description": "Fetch data"}, "_cli_func_info": info_d},
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
        assert _extract_help_query(["--help", "forecast", "generate"]) == "forecast generate"

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
# _print_extended_help
# ========================================================================

class TestPrintExtendedHelp:
    def _make_functions(self):
        def forecast_gen(symbol: str, horizon: int = 12):
            """Generate forecast."""
            pass
        info = get_function_info(forecast_gen)
        return {
            "forecast_generate": {"func": forecast_gen, "meta": {"description": "Generate forecast"}, "_cli_func_info": info},
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
            require_sl_tp: bool = True,
            auto_close_on_sl_tp_fail: bool = False,
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
        assert "require_sl_tp=true" in out
        assert "auto_close_on_sl_tp_fail=false" in out
        assert "market orders default to require_sl_tp=true" in out


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
            "my_func": {"func": my_func, "meta": {"description": "Do something"}, "_cli_func_info": info},
        }
        epilog = _build_epilog(functions)
        assert "my_func" in epilog
        assert "Commands and Arguments" in epilog


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
        assert args.model == "theta"
        assert args.timeframe == "H1"
        assert args.horizon == 12

    def test_all_options(self):
        parser = argparse.ArgumentParser()
        _add_forecast_generate_args(parser)
        args = parser.parse_args([
            "GBPUSD",
            "--library", "pretrained",
            "--model", "chronos2",
            "--timeframe", "D1",
            "--horizon", "24",
            "--lookback", "200",
            "--quantity", "return",
            "--ci-alpha", "0.1",
            "--denoise", "wavelet",
            "--verbose",
            "--print-config",
        ])
        assert args.symbol == "GBPUSD"
        assert args.library == "pretrained"
        assert args.model == "chronos2"
        assert args.horizon == 24
        assert args.lookback == 200
        assert args.quantity == "return"
        assert args.ci_alpha == 0.1
        assert args.verbose is True
        assert args.print_config is True


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
        args = parser.parse_args(["--flag", "true"])
        assert args.flag == "true"

    def test_list_param(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "items", "type": List[str], "required": False, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["--items", "a", "b", "c"])
        assert args.items == ["a", "b", "c"]

    def test_mapping_param_adds_companion(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "simplify", "type": Dict[str, Any], "required": False, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        args = parser.parse_args(["--simplify", "lttb", "--simplify-params", "points=100"])
        assert args.simplify == "lttb"
        assert args.simplify_params == "points=100"

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

    def test_single_word_flag_is_not_duplicated(self):
        parser = argparse.ArgumentParser()
        func_info = {
            "params": [
                {"name": "ticket", "type": int, "required": False, "default": None},
            ]
        }
        add_dynamic_arguments(parser, func_info)
        ticket_action = next(action for action in parser._actions if action.dest == "ticket")
        assert ticket_action.option_strings == ["--ticket"]


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

    def test_optional_int(self):
        param = {"name": "count", "type": Optional[int], "required": False, "default": None}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert kwargs["type"] is int

    def test_dict_param_is_mapping(self):
        param = {"name": "params", "type": Dict[str, Any], "required": False, "default": None}
        kwargs, is_mapping = _resolve_param_kwargs(param, None)
        assert is_mapping is True

    def test_literal_type(self):
        param = {"name": "mode", "type": Literal["a", "b", "c"], "required": False, "default": "a"}
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
        param = {"name": "methods", "type": List[Literal["a", "b"]], "required": False, "default": None}
        kwargs, _ = _resolve_param_kwargs(param, None)
        assert kwargs["choices"] == ["a", "b"]
        assert kwargs["nargs"] == "+"

    def test_forecast_method_help_avoids_massive_choices(self):
        param = {"name": "method", "type": str, "required": False, "default": None}
        kwargs, _ = _resolve_param_kwargs(param, None, cmd_name="forecast_conformal_intervals")
        assert "choices" not in kwargs
        assert kwargs["metavar"] == "METHOD"
        assert "forecast_list_methods" in kwargs["help"]


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
            ]
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
                {"name": "model_params", "type": Dict[str, Any], "required": False, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(
            symbol="EURUSD",
            horizon=24,
            model_params='{"sp":24}',
            model_params_params=None,
            json=False,
            verbose=False,
        )
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        request = call_kwargs["request"]
        assert isinstance(request, ForecastGenerateRequest)
        assert request.symbol == "EURUSD"
        assert request.horizon == 24
        assert request.model_params == {"sp": 24}
        assert call_kwargs["__cli_raw"] is True

    def test_indicator_compact_string_reconstructed(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "request_model": DataFetchCandlesRequest,
            "request_param_name": "request",
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "indicators", "type": Optional[List[Dict[str, Any]]], "required": False, "default": None},
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

    def test_indicator_params_dict_returns_friendly_error(self, capsys):
        mock_fn = MagicMock(return_value={"ok": True})
        func_info = {
            "func": mock_fn,
            "request_model": DataFetchCandlesRequest,
            "request_param_name": "request",
            "params": [
                {"name": "symbol", "type": str, "required": True, "default": None},
                {"name": "indicators", "type": Optional[List[Dict[str, Any]]], "required": False, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="data_fetch_candles")
        args = argparse.Namespace(
            symbol="EURUSD",
            indicators='[{"name":"rsi","params":{"period":14}}]',
            indicators_params=None,
            json=False,
            verbose=False,
        )
        status = cmd_fn(args)
        assert status == 1
        assert "'params' must be a list of numbers" in capsys.readouterr().out
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

    def test_string_result_text_format(self, capsys):
        mock_fn = MagicMock(return_value="plain text result")
        func_info = {
            "func": mock_fn,
            "params": [{"name": "symbol", "type": str, "required": True, "default": None}],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(symbol="EURUSD", json=False, verbose=False)
        cmd_fn(args)
        assert "plain text result" in capsys.readouterr().out

    def test_string_result_json_format(self, capsys):
        mock_fn = MagicMock(return_value="plain text result")
        func_info = {
            "func": mock_fn,
            "params": [{"name": "symbol", "type": str, "required": True, "default": None}],
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
            "params": [{"name": "symbol", "type": str, "required": True, "default": None}],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(symbol="EURUSD", json=True, verbose=False)
        cmd_fn(args)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["price"] == 1.23
        assert "cli_meta" in parsed

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
        args = argparse.Namespace(symbol="EURUSD", extra=None, json=False, verbose=False)
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert "extra" not in call_kwargs

    def test_mapping_present_sentinel(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "simplify", "type": Dict[str, Any], "required": False, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(simplify="__PRESENT__", simplify_params=None, json=False, verbose=False)
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["simplify"] == {}

    def test_mapping_json_string(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "simplify", "type": Dict[str, Any], "required": False, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(simplify='{"method":"lttb"}', simplify_params=None, json=False, verbose=False)
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert isinstance(call_kwargs["simplify"], dict)

    def test_mapping_shorthand_string(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "simplify", "type": Dict[str, Any], "required": False, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(simplify="lttb", simplify_params=None, json=False, verbose=False)
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["simplify"] == {"method": "lttb"}

    def test_mapping_companion_params(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "simplify", "type": Dict[str, Any], "required": False, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(simplify="lttb", simplify_params="points=100", json=False, verbose=False)
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        assert isinstance(call_kwargs["simplify"], dict)
        assert call_kwargs["simplify"]["method"] == "lttb"

    def test_list_param_normalized(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "methods", "type": List[str], "required": False, "default": None},
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
            "params": [{"name": "symbol", "type": str, "required": True, "default": None}],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        args = argparse.Namespace(symbol="EURUSD", json=True, verbose=True)
        cmd_fn(args)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "cli_meta" in parsed

    def test_empty_output_no_print(self, capsys):
        mock_fn = MagicMock(return_value={})
        func_info = {
            "func": mock_fn,
            "params": [{"name": "symbol", "type": str, "required": True, "default": None}],
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
    def test_discover_from_registry_includes_submodule_tools(self, mock_mcp, mock_bootstrap, mock_get_reg):
        def fake_tool(symbol: str):
            """Fake tool."""
            pass

        fake_tool.__module__ = "mtdata.core.trading_positions"

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
            "my_tool": {"func": my_tool, "meta": {"description": "My tool"}, "_cli_func_info": info},
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
            "my_tool": {"func": mock_fn, "meta": {"description": "My tool"}, "_cli_func_info": info},
        }
        with patch("sys.argv", ["cli.py", "my_tool", "EURUSD"]):
            result = main()
        assert result == 0
        mock_fn.assert_called_once()

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
            "my_tool": {"func": mock_fn, "meta": {"description": "My tool"}, "_cli_func_info": info},
        }
        with patch("sys.argv", ["cli.py", "--timeframe", "D1", "my_tool", "EURUSD"]):
            result = main()
        assert result == 0
        assert mock_fn.call_args[1]["timeframe"] == "D1"

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
            "bad_tool": {"func": mock_fn, "meta": {"description": "Bad tool"}, "_cli_func_info": info},
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
            "bad_tool": {"func": mock_fn, "meta": {"description": "Bad tool"}, "_cli_func_info": info},
        }
        with patch("sys.argv", ["cli.py", "bad_tool", "X"]):
            result = main()
        assert result == 1

    @patch("mtdata.core.cli.discover_tools")
    def test_command_no_action_result_returns_nonzero(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"message": "No action taken", "no_action": True})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "noop_tool"
        mock_fn.__doc__ = "No-op tool."

        def noop_tool(symbol: str):
            """No-op tool."""
            pass

        info = get_function_info(noop_tool)
        info["func"] = mock_fn

        mock_discover.return_value = {
            "noop_tool": {"func": mock_fn, "meta": {"description": "No-op tool"}, "_cli_func_info": info},
        }
        with patch("sys.argv", ["cli.py", "noop_tool", "X"]):
            result = main()
        assert result == 1

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
            "slow_tool": {"func": mock_fn, "meta": {"description": "Slow tool"}, "_cli_func_info": info},
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
            "debug_tool": {"func": mock_fn, "meta": {"description": "Debug tool"}, "_cli_func_info": info},
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
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
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
        assert request.model == "theta"
        assert call_kwargs["__cli_raw"] is True

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_print_config(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD", "--print-config"]):
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
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
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
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
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
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
        }
        with patch("sys.argv", [
            "cli.py", "forecast_generate", "EURUSD",
            "--set", "model.sp=24",
            "--set", "model.max_epochs=20",
        ]):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.model_params["sp"] == 24
        assert request.model_params["max_epochs"] == 20

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_uses_global_timeframe_before_command(self, mock_discover):
        mock_fn = MagicMock(return_value={"forecast": [1.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
        }
        with patch("sys.argv", ["cli.py", "--timeframe", "D1", "forecast_generate", "EURUSD"]):
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
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD", "--denoise", "wavelet"]):
            result = main()
        assert result == 0
        request = mock_fn.call_args[1]["request"]
        assert request.denoise == {"method": "wavelet"}

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_dict_result_verbose(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={"forecast": [1.0, 2.0, 3.0]})
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "EURUSD", "--verbose"]):
            result = main()
        assert result == 0

    @patch("mtdata.core.cli.discover_tools")
    def test_forecast_generate_omits_redundant_ci_block_when_bounds_rendered(self, mock_discover, capsys):
        mock_fn = MagicMock(return_value={
            "times": ["2026-03-07 18:00"],
            "forecast_price": [67580.67],
            "lower_price": [63025.26],
            "upper_price": [71823.47],
            "ci_requested": True,
            "ci_alpha_requested": 0.05,
            "ci_available": True,
            "ci_status": "available",
            "ci_alpha": 0.05,
        })
        mock_fn.__module__ = "mtdata.core.server"
        mock_fn.__name__ = "forecast_generate"
        mock_fn.__doc__ = "Generate forecasts."

        mock_discover.return_value = {
            "forecast_generate": {"func": mock_fn, "meta": {"description": "Generate forecasts"}},
        }
        with patch("sys.argv", ["cli.py", "forecast_generate", "BTCUSD", "--timeframe", "D1"]):
            result = main()
        assert result == 0
        out = capsys.readouterr().out
        assert "forecast[1]{time,forecast,lower,upper}:" in out
        assert "\nci:" not in out


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
        assert "local" in result
        assert "name" in result["local"]["tz"]

    def test_attach_cli_meta_with_none_cmd(self):
        r = {"data": 1}
        out = _attach_cli_meta(r, cmd_name=None, verbose=True)
        assert out["cli_meta"]["command"] == ""

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
                {"name": "simplify", "type": Dict[str, Any], "required": False, "default": None},
            ],
        }
        cmd_fn = create_command_function(func_info, cmd_name="test_cmd")
        # simplify_params present but simplify is None
        args = argparse.Namespace(simplify=None, simplify_params="points=100", json=False, verbose=False)
        cmd_fn(args)
        call_kwargs = mock_fn.call_args[1]
        # companion params should create the dict
        assert isinstance(call_kwargs.get("simplify"), dict)

    def test_create_command_mapping_companion_merge(self, capsys):
        mock_fn = MagicMock(return_value="ok")
        func_info = {
            "func": mock_fn,
            "params": [
                {"name": "simplify", "type": Dict[str, Any], "required": False, "default": None},
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
        result = _format_result_for_cli({"a": 1}, fmt="", verbose=False, cmd_name="test")
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
            "data_fetch_candles": {"func": fetch_data, "meta": {"description": "Fetch candle data"}, "_cli_func_info": info},
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
    @pytest.mark.parametrize("input_val,expected", [
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
    ])
    def test_coerce_values(self, input_val, expected):
        assert _coerce_cli_scalar(input_val) == expected


# ========================================================================
# Parameterised tests for _normalize_cli_list_value
# ========================================================================

class TestNormalizeCliListParameterized:
    @pytest.mark.parametrize("input_val,expected", [
        ("a b c", ["a", "b", "c"]),
        ("a,b,c", ["a", "b", "c"]),
        ('["x"]', ["x"]),
        (["a b", "c,d"], ["a", "b", "c", "d"]),
        (None, None),
        ([], []),
    ])
    def test_normalize(self, input_val, expected):
        assert _normalize_cli_list_value(input_val) == expected
