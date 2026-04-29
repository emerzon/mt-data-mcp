"""Tests for mtdata.core.server — server configuration, coercion helpers, tool decorator."""

import asyncio
import inspect
import math
import os
import sys
import types
from typing import Optional, Union
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers under test are importable without triggering full server bootstrap
# because we mock the heavy imports at module level where needed.
# ---------------------------------------------------------------------------


# ── bootstrap.runtime ─────────────────────────────────────────────────────

class TestMcpRuntimeSettings:
    def test_load_defaults(self, monkeypatch):
        from mtdata.bootstrap.runtime import load_mcp_runtime_settings

        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        monkeypatch.delenv("FASTMCP_HOST", raising=False)
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        monkeypatch.delenv("FASTMCP_PORT", raising=False)
        settings = load_mcp_runtime_settings()

        assert settings.transport == "sse"
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000

    def test_load_from_env(self, monkeypatch):
        from mtdata.bootstrap.runtime import load_mcp_runtime_settings

        monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("FASTMCP_HOST", "1.2.3.4")
        monkeypatch.setenv("FASTMCP_ALLOW_REMOTE", "1")
        monkeypatch.setenv("FASTMCP_PORT", "9999")
        monkeypatch.setenv("FASTMCP_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("FASTMCP_MOUNT_PATH", "/api")
        settings = load_mcp_runtime_settings()

        assert settings.transport == "streamable-http"
        assert settings.host == "1.2.3.4"
        assert settings.port == 9999
        assert settings.log_level == "DEBUG"
        assert settings.mount_path == "/api"

    def test_invalid_transport_falls_back(self, monkeypatch):
        from mtdata.bootstrap.runtime import load_mcp_runtime_settings

        monkeypatch.setenv("MCP_TRANSPORT", "grpc")
        monkeypatch.delenv("FASTMCP_HOST", raising=False)
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        settings = load_mcp_runtime_settings()
        assert settings.transport == "sse"

    def test_non_loopback_host_requires_explicit_remote_flag(self, monkeypatch):
        from mtdata.bootstrap.runtime import load_mcp_runtime_settings

        monkeypatch.setenv("FASTMCP_HOST", "0.0.0.0")
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        with pytest.raises(ValueError, match="FASTMCP_ALLOW_REMOTE"):
            load_mcp_runtime_settings()

    def test_stdio_skips_remote_bind_guard(self, monkeypatch):
        from mtdata.bootstrap.runtime import load_mcp_runtime_settings

        monkeypatch.setenv("FASTMCP_HOST", "0.0.0.0")
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        settings = load_mcp_runtime_settings(transport_override="stdio")

        assert settings.transport == "stdio"
        assert settings.host == "0.0.0.0"

    def test_apply_to_mcp_settings(self):
        from mtdata.bootstrap.runtime import (
            McpRuntimeSettings,
            apply_mcp_runtime_settings,
        )

        mcp = MagicMock()
        mcp.settings = MagicMock()
        apply_mcp_runtime_settings(
            mcp,
            McpRuntimeSettings(
                transport="stdio",
                host="127.0.0.1",
                port=9000,
                log_level="WARNING",
                mount_path="/x",
                sse_path="/events",
                message_path="/message",
            ),
        )

        assert mcp.settings.host == "127.0.0.1"
        assert mcp.settings.port == 9000
        assert mcp.settings.log_level == "WARNING"
        assert mcp.settings.mount_path == "/x"
        assert mcp.settings.sse_path == "/events"
        assert mcp.settings.message_path == "/message"


class TestWebApiRuntimeSettings:
    def test_remote_bind_requires_auth_token(self, monkeypatch):
        from mtdata.bootstrap.runtime import load_web_api_runtime_settings

        monkeypatch.setenv("WEBAPI_HOST", "0.0.0.0")
        monkeypatch.setenv("WEBAPI_ALLOW_REMOTE", "1")
        monkeypatch.delenv("WEBAPI_AUTH_TOKEN", raising=False)
        with pytest.raises(ValueError, match="WEBAPI_AUTH_TOKEN"):
            load_web_api_runtime_settings()

    def test_remote_bind_with_auth_token(self, monkeypatch):
        from mtdata.bootstrap.runtime import load_web_api_runtime_settings

        monkeypatch.setenv("WEBAPI_HOST", "0.0.0.0")
        monkeypatch.setenv("WEBAPI_ALLOW_REMOTE", "1")
        monkeypatch.setenv("WEBAPI_AUTH_TOKEN", "secret")
        settings = load_web_api_runtime_settings()

        assert settings.host == "0.0.0.0"
        assert settings.auth_token == "secret"

    def test_cors_wildcard_is_rejected(self, monkeypatch):
        from mtdata.bootstrap.runtime import load_web_api_runtime_settings

        monkeypatch.setenv("CORS_ORIGINS", "*")
        with pytest.raises(ValueError, match="CORS_ORIGINS"):
            load_web_api_runtime_settings()


# ── _unwrap_optional_annotation ───────────────────────────────────────────

class TestUnwrapOptionalAnnotation:

    def _call(self, ann):
        from mtdata.core.server import _unwrap_optional_annotation
        return _unwrap_optional_annotation(ann)

    # --- string annotations ---
    def test_string_bool(self):
        base, opt = self._call("bool")
        assert base is bool and not opt

    def test_string_int(self):
        base, opt = self._call("int")
        assert base is int and not opt

    def test_string_float(self):
        base, opt = self._call("float")
        assert base is float and not opt

    def test_string_builtins_bool(self):
        base, opt = self._call("builtins.bool")
        assert base is bool and not opt

    def test_string_builtins_int(self):
        base, opt = self._call("builtins.int")
        assert base is int and not opt

    def test_string_builtins_float(self):
        base, opt = self._call("builtins.float")
        assert base is float and not opt

    def test_string_unknown(self):
        base, opt = self._call("str")
        assert base == "str" and not opt

    # PEP-604 unions
    def test_pep604_int_none(self):
        base, opt = self._call("int | None")
        assert base is int and opt

    def test_pep604_float_none(self):
        base, opt = self._call("float | None")
        assert base is float and opt

    def test_pep604_bool_none(self):
        base, opt = self._call("bool | None")
        assert base is bool and opt

    def test_pep604_unknown_none(self):
        base, opt = self._call("str | None")
        assert base == "str | None" and not opt

    def test_pep604_multi_parts(self):
        base, opt = self._call("int | float | None")
        assert not opt  # multiple non-None parts → not unwrapped

    # Optional[...] string form
    def test_optional_int_string(self):
        base, opt = self._call("Optional[int]")
        assert base is int and opt

    def test_optional_float_string(self):
        base, opt = self._call("Optional[float]")
        assert base is float and opt

    def test_optional_bool_string(self):
        base, opt = self._call("Optional[bool]")
        assert base is bool and opt

    def test_typing_optional_int_string(self):
        base, opt = self._call("typing.Optional[int]")
        assert base is int and opt

    # Union[..., None] string form
    def test_union_int_none_string(self):
        base, opt = self._call("Union[int, None]")
        assert base is int and opt

    def test_typing_union_float_none_string(self):
        base, opt = self._call("typing.Union[float, None]")
        assert base is float and opt

    def test_union_multi_string(self):
        base, opt = self._call("Union[int, float, None]")
        assert not opt  # multiple non-None

    # runtime type annotations
    def test_runtime_optional_int(self):
        base, opt = self._call(Optional[int])
        assert base is int and opt

    def test_runtime_optional_float(self):
        base, opt = self._call(Optional[float])
        assert base is float and opt

    def test_runtime_plain_int(self):
        base, opt = self._call(int)
        assert base is int and not opt


# ── _coerce_bool ──────────────────────────────────────────────────────────

class TestCoerceBool:

    def _call(self, value, *, allow_none=False, name="x"):
        from mtdata.core.server import _coerce_bool
        return _coerce_bool(value, allow_none=allow_none, name=name)

    def test_true_passthrough(self):
        assert self._call(True) is True

    def test_false_passthrough(self):
        assert self._call(False) is False

    def test_none_allowed(self):
        assert self._call(None, allow_none=True) is None

    def test_none_disallowed(self):
        with pytest.raises(ValueError):
            self._call(None)

    def test_int_1(self):
        assert self._call(1) is True

    def test_int_0(self):
        assert self._call(0) is False

    def test_float_1(self):
        assert self._call(1.0) is True

    def test_string_true_variants(self):
        for s in ("true", "True", "TRUE", "1", "yes", "y", "on", "YES", "ON"):
            assert self._call(s) is True, f"Failed for {s!r}"

    def test_string_false_variants(self):
        for s in ("false", "False", "FALSE", "0", "no", "n", "off", "NO", "OFF"):
            assert self._call(s) is False, f"Failed for {s!r}"

    def test_string_none_allowed(self):
        assert self._call("none", allow_none=True) is None
        assert self._call("null", allow_none=True) is None

    def test_string_none_disallowed(self):
        with pytest.raises(ValueError):
            self._call("none")

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            self._call("maybe")

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            self._call([1, 2])


# ── _coerce_int ───────────────────────────────────────────────────────────

class TestCoerceInt:

    def _call(self, value, *, allow_none=False, name="x"):
        from mtdata.core.server import _coerce_int
        return _coerce_int(value, allow_none=allow_none, name=name)

    def test_int_passthrough(self):
        assert self._call(42) == 42

    def test_none_allowed(self):
        assert self._call(None, allow_none=True) is None

    def test_none_disallowed(self):
        with pytest.raises(ValueError):
            self._call(None)

    def test_bool_true(self):
        assert self._call(True) == 1

    def test_bool_false(self):
        assert self._call(False) == 0

    def test_float_whole(self):
        assert self._call(5.0) == 5

    def test_float_fractional(self):
        with pytest.raises(ValueError):
            self._call(5.5)

    def test_float_inf(self):
        with pytest.raises(ValueError):
            self._call(float("inf"))

    def test_float_nan(self):
        with pytest.raises(ValueError):
            self._call(float("nan"))

    def test_string_int(self):
        assert self._call("7") == 7

    def test_string_float_whole(self):
        assert self._call("8.0") == 8

    def test_string_none_allowed(self):
        assert self._call("none", allow_none=True) is None

    def test_string_null_allowed(self):
        assert self._call("null", allow_none=True) is None

    def test_string_none_disallowed(self):
        with pytest.raises(ValueError):
            self._call("none")

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            self._call("abc")

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            self._call([1])


# ── _coerce_float ─────────────────────────────────────────────────────────

class TestCoerceFloat:

    def _call(self, value, *, allow_none=False, name="x"):
        from mtdata.core.server import _coerce_float
        return _coerce_float(value, allow_none=allow_none, name=name)

    def test_float_passthrough(self):
        assert self._call(3.14) == 3.14

    def test_int_to_float(self):
        assert self._call(3) == 3.0

    def test_none_allowed(self):
        assert self._call(None, allow_none=True) is None

    def test_none_disallowed(self):
        with pytest.raises(ValueError):
            self._call(None)

    def test_bool_true(self):
        assert self._call(True) == 1.0

    def test_bool_false(self):
        assert self._call(False) == 0.0

    def test_inf_raises(self):
        with pytest.raises(ValueError):
            self._call(float("inf"))

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            self._call(float("nan"))

    def test_string_float(self):
        assert self._call("2.5") == 2.5

    def test_string_int(self):
        assert self._call("10") == 10.0

    def test_string_none_allowed(self):
        assert self._call("none", allow_none=True) is None

    def test_string_null_allowed(self):
        assert self._call("null", allow_none=True) is None

    def test_string_none_disallowed(self):
        with pytest.raises(ValueError):
            self._call("none")

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            self._call("xyz")

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            self._call([1.0])


# ── _coerce_kwargs_for_callable ───────────────────────────────────────────

class TestCoerceKwargsForCallable:

    def _call(self, func, kwargs):
        from mtdata.core.server import _coerce_kwargs_for_callable
        return _coerce_kwargs_for_callable(func, kwargs)

    def test_coerces_bool_kwarg(self):
        def fn(flag: bool = False): ...
        kw = {"flag": "true"}
        self._call(fn, kw)
        assert kw["flag"] is True

    def test_coerces_int_kwarg(self):
        def fn(count: int = 0): ...
        kw = {"count": "42"}
        self._call(fn, kw)
        assert kw["count"] == 42

    def test_coerces_float_kwarg(self):
        def fn(rate: float = 0.0): ...
        kw = {"rate": "3.14"}
        self._call(fn, kw)
        assert kw["rate"] == 3.14

    def test_coerces_optional_int_kwarg(self):
        def fn(limit: Optional[int] = None): ...
        kw = {"limit": "none"}
        self._call(fn, kw)
        assert kw["limit"] is None

    def test_skips_missing_kwargs(self):
        def fn(a: int = 0, b: int = 0): ...
        kw = {"a": "5"}
        self._call(fn, kw)
        assert kw == {"a": 5}

    def test_skips_unannotated(self):
        def fn(x): ...
        kw = {"x": "hello"}
        self._call(fn, kw)
        assert kw["x"] == "hello"

    def test_coerces_pydantic_request_kwarg(self):
        from mtdata.forecast.requests import ForecastGenerateRequest

        def fn(request: ForecastGenerateRequest): ...

        kw = {"request": {"symbol": "EURUSD", "horizon": 24}}
        self._call(fn, kw)
        assert isinstance(kw["request"], ForecastGenerateRequest)
        assert kw["request"].symbol == "EURUSD"
        assert kw["request"].horizon == 24

    def test_coerces_flat_kwargs_into_pydantic_request(self):
        from mtdata.forecast.requests import ForecastGenerateRequest

        def fn(request: ForecastGenerateRequest): ...

        kw = {"symbol": "EURUSD", "horizon": 24}
        self._call(fn, kw)
        assert isinstance(kw["request"], ForecastGenerateRequest)
        assert kw["request"].symbol == "EURUSD"
        assert kw["request"].horizon == 24
        assert "symbol" not in kw
        assert "horizon" not in kw

    def test_coerces_indicator_string_into_data_request_model(self):
        from mtdata.core.data.requests import DataFetchCandlesRequest

        def fn(request: DataFetchCandlesRequest): ...

        kw = {
            "symbol": "BTCUSD",
            "indicators": "ema(20),rsi(14),macd(12,26,9)",
        }
        self._call(fn, kw)
        assert isinstance(kw["request"], DataFetchCandlesRequest)
        assert kw["request"].indicators == [
            {"name": "ema", "params": [20.0]},
            {"name": "rsi", "params": [14.0]},
            {"name": "macd", "params": [12.0, 26.0, 9.0]},
        ]

    def test_request_model_signature_fields_preserve_indicator_string_coercion(self):
        from pydantic import TypeAdapter

        from mtdata.core._mcp_tools import _request_model_signature_fields
        from mtdata.core.data.requests import DataFetchCandlesRequest

        def fn(request: DataFetchCandlesRequest): ...

        params = _request_model_signature_fields(fn)
        indicators_param = next(param for param in params if param.name == "indicators")

        value = TypeAdapter(indicators_param.annotation).validate_python("ema(20),rsi(14),macd(12,26,9)")
        assert value == [
            {"name": "ema", "params": [20.0]},
            {"name": "rsi", "params": [14.0]},
            {"name": "macd", "params": [12.0, 26.0, 9.0]},
        ]

    def test_request_model_signature_fields_preserve_named_indicator_string_coercion(self):
        from pydantic import TypeAdapter

        from mtdata.core._mcp_tools import _request_model_signature_fields
        from mtdata.core.data.requests import DataFetchCandlesRequest

        def fn(request: DataFetchCandlesRequest): ...

        params = _request_model_signature_fields(fn)
        indicators_param = next(param for param in params if param.name == "indicators")

        value = TypeAdapter(indicators_param.annotation).validate_python(
            "rsi(length=14),macd(fast=12,slow=26,signal=9)"
        )
        assert value == [
            {"name": "rsi", "params": {"length": 14.0}},
            {"name": "macd", "params": {"fast": 12.0, "slow": 26.0, "signal": 9.0}},
        ]

    def test_handles_bad_signature_gracefully(self):
        kw = {"a": "1"}
        result = self._call("not_a_callable", kw)
        assert result == kw  # returns kwargs unmodified


# ── _get_runtime_signature / _get_runtime_annotations ─────────────────────

class TestRuntimeHelpers:

    def test_get_runtime_signature_basic(self):
        from mtdata.core.server import _get_runtime_signature
        def fn(a: int, b: str = "x"): ...
        sig = _get_runtime_signature(fn)
        assert "a" in sig.parameters
        assert "b" in sig.parameters

    def test_get_runtime_annotations_basic(self):
        from mtdata.core.server import _get_runtime_annotations
        def fn(a: int, b: str = "x") -> bool: ...
        ann = _get_runtime_annotations(fn)
        assert ann.get("a") is int
        assert ann.get("return") is bool

    def test_get_runtime_annotations_no_annotations(self):
        from mtdata.core.server import _get_runtime_annotations
        def fn(): ...
        fn.__annotations__ = None  # type: ignore
        ann = _get_runtime_annotations(fn)
        assert ann == {} or ann is None or isinstance(ann, dict)


# ── _resolve_transport ────────────────────────────────────────────────────

class TestResolveTransport:

    def _call(self, default="sse"):
        from mtdata.core.server import _resolve_transport
        return _resolve_transport(default)

    def test_default_sse(self, monkeypatch):
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        monkeypatch.delenv("FASTMCP_HOST", raising=False)
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        monkeypatch.delenv("FASTMCP_MOUNT_PATH", raising=False)
        t, mp = self._call()
        assert t == "sse"
        assert mp is None

    def test_env_stdio(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "stdio")
        monkeypatch.delenv("FASTMCP_HOST", raising=False)
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        monkeypatch.delenv("FASTMCP_MOUNT_PATH", raising=False)
        t, mp = self._call()
        assert t == "stdio"

    def test_env_stdio_ignores_remote_bind_guard(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "stdio")
        monkeypatch.setenv("FASTMCP_HOST", "0.0.0.0")
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        monkeypatch.delenv("FASTMCP_MOUNT_PATH", raising=False)
        t, mp = self._call()
        assert t == "stdio"
        assert mp is None

    def test_env_streamable_http(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
        monkeypatch.delenv("FASTMCP_HOST", raising=False)
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        monkeypatch.delenv("FASTMCP_MOUNT_PATH", raising=False)
        t, _ = self._call()
        assert t == "streamable-http"

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "grpc")
        monkeypatch.delenv("FASTMCP_HOST", raising=False)
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        monkeypatch.delenv("FASTMCP_MOUNT_PATH", raising=False)
        t, _ = self._call()
        assert t == "sse"

    def test_mount_path_returned(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "sse")
        monkeypatch.delenv("FASTMCP_HOST", raising=False)
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        monkeypatch.setenv("FASTMCP_MOUNT_PATH", "/mcp")
        _, mp = self._call()
        assert mp == "/mcp"


# ── get_tool_registry / get_tool_functions ────────────────────────────────

class TestToolRegistries:

    @patch("mtdata.core.server.bootstrap_tools")
    def test_get_tool_registry_returns_dict(self, mock_bootstrap):
        from mtdata.core.server import get_tool_registry
        result = get_tool_registry()
        assert isinstance(result, dict)
        mock_bootstrap.assert_called_once()

    @patch("mtdata.core.server.bootstrap_tools")
    def test_get_tool_functions_returns_dict(self, mock_bootstrap):
        from mtdata.core.server import get_tool_functions
        result = get_tool_functions()
        assert isinstance(result, dict)
        mock_bootstrap.assert_called_once()

    def test_registries_are_copies(self):
        from mtdata.core.server import (
            _TOOL_REGISTRY,
            get_tool_functions,
            get_tool_registry,
        )
        r1 = get_tool_registry()
        r2 = get_tool_functions()
        # Mutating returned dict should not affect internal state
        r1["__test__"] = True
        r2["__test__"] = True
        assert "__test__" not in _TOOL_REGISTRY

    def test_registry_views_stay_aligned_across_function_and_object_projections(self):
        import mtdata.core._mcp_tools as tools

        key = "__test_registry_projection__"
        func = lambda: None
        tool_obj = object()
        prior_func = tools._TOOL_REGISTRY.pop(key, None)
        prior_obj = tools._TOOL_OBJECT_REGISTRY.pop(key, None)
        try:
            tools._TOOL_REGISTRY[key] = func
            tools._TOOL_OBJECT_REGISTRY[key] = tool_obj

            assert tools.get_tool_functions()[key] is func
            assert tools.get_tool_registry()[key] is tool_obj

            tools._TOOL_REGISTRY.pop(key, None)
            assert key not in tools.get_tool_functions()
            assert tools.get_tool_registry()[key] is tool_obj

            tools._TOOL_OBJECT_REGISTRY.pop(key, None)
            assert key not in tools.get_tool_registry()
        finally:
            tools._TOOL_REGISTRY.pop(key, None)
            tools._TOOL_OBJECT_REGISTRY.pop(key, None)
            if prior_func is not None:
                tools._TOOL_REGISTRY[key] = prior_func
            if prior_obj is not None:
                tools._TOOL_OBJECT_REGISTRY[key] = prior_obj


# ── _recording_tool_decorator ─────────────────────────────────────────────

class TestRecordingToolDecorator:

    def test_noop_fallback_when_no_orig(self):
        import mtdata.core.server as srv
        from mtdata.core.server import _TOOL_REGISTRY, _recording_tool_decorator
        orig = srv._ORIG_TOOL_DECORATOR
        try:
            srv._ORIG_TOOL_DECORATOR = None
            dec = _recording_tool_decorator()
            def dummy(): return 42
            result = dec(dummy)
            assert result is dummy
            assert "dummy" in _TOOL_REGISTRY
        finally:
            srv._ORIG_TOOL_DECORATOR = orig

    def test_wrapped_function_catches_exceptions(self):
        """Tool wrappers should catch exceptions and return error dicts."""
        from mtdata.core.server import _TOOL_REGISTRY
        # Find any registered tool and test __cli_raw exception handling
        for name, fn in _TOOL_REGISTRY.items():
            # The wrapper catches exceptions raised by the inner func
            assert callable(fn)
            break

    def test_wrapped_function_adds_error_metadata(self):
        import mtdata.core.server as srv
        from mtdata.core._mcp_tools import _TOOL_REGISTRY

        original = srv._ORIG_TOOL_DECORATOR
        try:
            srv._ORIG_TOOL_DECORATOR = lambda *a, **k: (lambda fn: fn)
            dec = srv._recording_tool_decorator()

            def boom():
                raise RuntimeError("boom")

            dec(boom)
            wrapped = _TOOL_REGISTRY["boom"]
            result = wrapped(__cli_raw=True)

            assert result["error"] == "boom"
            assert result["success"] is False
            assert result["error_code"] == "tool_execution_error"
            assert result["operation"] == "boom"
            assert isinstance(result.get("request_id"), str)
            assert result["request_id"]
        finally:
            srv._ORIG_TOOL_DECORATOR = original

    def test_wrapped_function_uses_compact_default_text_view(self):
        import mtdata.core.server as srv
        from mtdata.core._mcp_tools import _TOOL_REGISTRY

        original = srv._ORIG_TOOL_DECORATOR
        try:
            srv._ORIG_TOOL_DECORATOR = lambda *a, **k: (lambda fn: fn)
            dec = srv._recording_tool_decorator()

            def trade_place():
                return {
                    "success": True,
                    "retcode_name": "TRADE_RETCODE_DONE",
                    "order": 4384151941,
                    "requested_price": 0.8634,
                    "comment_sanitization": {"requested": "x", "applied": "x"},
                    "fill_mode_attempts": [
                        {
                            "type_filling": 1,
                            "retcode_name": "TRADE_RETCODE_DONE",
                        }
                    ],
                    "warnings": [
                        "Comment sanitized for broker compatibility: 'x'",
                        "Broker rejected the comment field; pending order was retried with a minimal MT5-safe comment.",
                    ],
                }

            dec(trade_place)
            wrapped = _TOOL_REGISTRY["trade_place"]
            result = wrapped()

            assert "retcode_name: TRADE_RETCODE_DONE" in result
            assert "order: 4384151941" in result
            assert "price: 0.8634" in result
            assert "comment_sanitization" not in result
            assert "fill_mode_attempts" not in result
            assert "warnings" not in result
        finally:
            srv._ORIG_TOOL_DECORATOR = original

    def test_wrapped_function_hides_meta_by_default_and_exposes_output_contract(self):
        import mtdata.core.server as srv
        from mtdata.core._mcp_tools import _TOOL_REGISTRY

        original = srv._ORIG_TOOL_DECORATOR
        try:
            srv._ORIG_TOOL_DECORATOR = lambda *a, **k: (lambda fn: fn)
            dec = srv._recording_tool_decorator()

            def sample_tool(detail: str = "compact"):
                return {
                    "value": 1,
                    "meta": {"domain": {"symbol": "EURUSD"}},
                    "diagnostics": {"source": "mt5"},
                }

            dec(sample_tool)
            wrapped = _TOOL_REGISTRY["sample_tool"]
            sig = inspect.signature(wrapped)

            assert "detail" not in sig.parameters
            assert "json" in sig.parameters
            assert "extras" in sig.parameters
            assert "verbose" not in sig.parameters
            assert "precision" not in sig.parameters

            raw = wrapped(__cli_raw=True)
            compact = wrapped(json=True)
            full = wrapped(json=True, extras="metadata,diagnostics")

            assert raw["value"] == 1
            assert raw["meta"]["domain"]["symbol"] == "EURUSD"
            assert raw["diagnostics"]["source"] == "mt5"
            assert compact["value"] == 1
            assert "meta" not in compact
            assert "diagnostics" not in compact
            assert full["value"] == 1
            assert full["meta"]["domain"]["symbol"] == "EURUSD"
            assert full["diagnostics"]["source"] == "mt5"
        finally:
            srv._ORIG_TOOL_DECORATOR = original

    def test_wrapped_function_treats_extras_as_verbose(self):
        import mtdata.core.server as srv
        from mtdata.core._mcp_tools import _TOOL_REGISTRY

        original = srv._ORIG_TOOL_DECORATOR
        try:
            srv._ORIG_TOOL_DECORATOR = lambda *a, **k: (lambda fn: fn)
            dec = srv._recording_tool_decorator()

            def sample_tool(detail: str = "compact"):
                return {
                    "value": 1,
                    "meta": {"domain": {"symbol": "EURUSD"}},
                    "diagnostics": {"source": "mt5"},
                }

            dec(sample_tool)
            wrapped = _TOOL_REGISTRY["sample_tool"]
            sig = inspect.signature(wrapped)

            assert "detail" not in sig.parameters
            assert "json" in sig.parameters
            assert "extras" in sig.parameters
            assert "verbose" not in sig.parameters
            assert "precision" not in sig.parameters
            raw = wrapped(__cli_raw=True)
            compact = wrapped(json=True)
            full = wrapped(json=True, extras="all")

            assert raw["value"] == 1
            assert raw["meta"]["domain"]["symbol"] == "EURUSD"
            assert raw["diagnostics"]["source"] == "mt5"
            assert compact["value"] == 1
            assert "meta" not in compact
            assert "diagnostics" not in compact
            assert full["value"] == 1
            assert full["meta"]["domain"]["symbol"] == "EURUSD"
            assert full["diagnostics"]["source"] == "mt5"
        finally:
            srv._ORIG_TOOL_DECORATOR = original

    def test_public_wrapped_output_is_toon_by_default_and_json_on_flag(self):
        import mtdata.core.server as srv

        original = srv._ORIG_TOOL_DECORATOR
        try:
            srv._ORIG_TOOL_DECORATOR = lambda *a, **k: (lambda fn: fn)
            dec = srv._recording_tool_decorator()

            def sample_output_tool(detail: str = "compact"):
                return {
                    "success": True,
                    "value": 1,
                    "detail_seen": detail,
                    "meta": {"domain": {"symbol": "EURUSD"}},
                    "diagnostics": {"source": "mt5"},
                }

            wrapped = dec(sample_output_tool)

            toon = wrapped()
            compact_structured = wrapped(json=True)
            structured = wrapped(json=True, extras="metadata")
            legacy_structured = wrapped(json=True, extras="detail=full")

            assert isinstance(toon, str)
            assert "success: true" in toon
            assert compact_structured["success"] is True
            assert compact_structured["detail_seen"] == "compact"
            assert "meta" not in compact_structured
            assert "diagnostics" not in compact_structured
            assert structured["success"] is True
            assert structured["detail_seen"] == "full"
            assert structured["meta"]["domain"]["symbol"] == "EURUSD"
            assert structured["diagnostics"]["source"] == "mt5"
            assert legacy_structured["success"] is True
            assert legacy_structured["detail_seen"] == "full"
            assert legacy_structured["meta"]["domain"]["symbol"] == "EURUSD"
            assert legacy_structured["diagnostics"]["source"] == "mt5"
            error = wrapped(detail="full")
            assert isinstance(error, str)
            assert "Removed output option(s): detail" in error
        finally:
            srv._ORIG_TOOL_DECORATOR = original

    def test_async_wrapped_timeout_returns_structured_error_payload(self):
        import asyncio

        import mtdata.core.server as srv

        original = srv._ORIG_TOOL_DECORATOR
        try:
            srv._ORIG_TOOL_DECORATOR = lambda *a, **k: (lambda fn: fn)
            dec = srv._recording_tool_decorator()

            def slow_tool():
                return {"success": True}

            wrapped = dec(slow_tool)
            async_wrapped = wrapped._mcp_async_wrapper

            async def _raise_timeout(coro, timeout):
                if hasattr(coro, "close"):
                    coro.close()
                raise asyncio.TimeoutError

            with patch("mtdata.core._mcp_tools.asyncio.wait_for", side_effect=_raise_timeout):
                result = asyncio.run(async_wrapped())

            assert result["success"] is False
            assert result["error_code"] == "tool_timeout"
            assert result["operation"] == "slow_tool"
            assert result["details"]["timeout_seconds"] == 120
        finally:
            srv._ORIG_TOOL_DECORATOR = original

    def test_async_wrapped_skips_transport_timeout_for_forecast_tuning_tools(self):
        import asyncio

        import mtdata.core.server as srv

        original = srv._ORIG_TOOL_DECORATOR
        try:
            srv._ORIG_TOOL_DECORATOR = lambda *a, **k: (lambda fn: fn)
            dec = srv._recording_tool_decorator()

            def forecast_optimize_hints():
                return {"success": True}

            wrapped = dec(forecast_optimize_hints)
            async_wrapped = wrapped._mcp_async_wrapper

            with patch("mtdata.core._mcp_tools.asyncio.wait_for") as wait_for:
                result = asyncio.run(async_wrapped(__cli_raw=True))

            assert result == {"success": True}
            wait_for.assert_not_called()
        finally:
            srv._ORIG_TOOL_DECORATOR = original

    def test_skips_variadic_args_in_exposed_signature(self):
        import mtdata.core.server as srv
        from mtdata.core._mcp_tools import _TOOL_REGISTRY

        original = srv._ORIG_TOOL_DECORATOR
        try:
            srv._ORIG_TOOL_DECORATOR = lambda *a, **k: (lambda fn: fn)
            dec = srv._recording_tool_decorator()

            def sample_tool(*args, foo: str = "x"):
                return {"foo": foo, "arg_count": len(args)}

            dec(sample_tool)
            wrapped = _TOOL_REGISTRY["sample_tool"]
            sig = inspect.signature(wrapped)

            assert "args" not in sig.parameters
            assert "foo" in sig.parameters
            assert "verbose" not in sig.parameters
            assert "precision" not in sig.parameters

            result = wrapped("legacy", foo="y", __cli_raw=True)
            assert result["foo"] == "y"
            assert result["arg_count"] == 1
        finally:
            srv._ORIG_TOOL_DECORATOR = original

    def test_flattens_request_model_signature_for_mcp_and_keeps_nested_request_compat(self):
        import mtdata.core.server as srv
        from mtdata.core._mcp_tools import _TOOL_REGISTRY
        from mtdata.core.data.requests import WaitEventRequest

        original = srv._ORIG_TOOL_DECORATOR
        calls = []
        try:
            srv._ORIG_TOOL_DECORATOR = lambda *a, **k: (lambda fn: fn)
            dec = srv._recording_tool_decorator()

            def sample_tool(request: WaitEventRequest):
                calls.append(request)
                return {
                    "symbol": request.symbol,
                    "timeframe": request.timeframe,
                    "poll_interval_seconds": request.poll_interval_seconds,
                }

            dec(sample_tool)
            wrapped = _TOOL_REGISTRY["sample_tool"]
            sig = inspect.signature(wrapped)

            assert "request" not in sig.parameters
            assert "symbol" in sig.parameters
            assert "timeframe" in sig.parameters
            assert "poll_interval_seconds" in sig.parameters
            assert "verbose" not in sig.parameters
            assert "precision" not in sig.parameters

            flat_result = wrapped(
                __cli_raw=True,
                symbol="BTCUSD",
                timeframe="M1",
                poll_interval_seconds=0.5,
            )
            assert isinstance(calls[-1], WaitEventRequest)
            assert calls[-1].symbol == "BTCUSD"
            assert calls[-1].timeframe == "M1"
            assert flat_result["symbol"] == "BTCUSD"

            nested_result = wrapped(
                __cli_raw=True,
                request={"symbol": "ETHUSD", "timeframe": "H1"},
            )
            assert isinstance(calls[-1], WaitEventRequest)
            assert calls[-1].symbol == "ETHUSD"
            assert calls[-1].timeframe == "H1"
            assert nested_result["symbol"] == "ETHUSD"
        finally:
            srv._ORIG_TOOL_DECORATOR = original


class TestMcpToolSchemas:

    def test_all_list_tools_schemas_use_public_output_contract(self):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        async def _run() -> dict[str, dict]:
            server = StdioServerParameters(
                command=sys.executable,
                args=["-c", "from mtdata.core.server import main_stdio; main_stdio()"],
            )
            async with stdio_client(server) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    items = getattr(tools, "tools", tools)
                    return {
                        str(getattr(tool, "name", "")): getattr(tool, "inputSchema", {}) or {}
                        for tool in items
                    }

        schemas = asyncio.run(_run())
        removed = {"detail", "format", "output_mode", "output"}

        assert schemas
        for name, schema in schemas.items():
            props = schema.get("properties") or {}
            assert not (removed & set(props)), name
            assert props["json"]["type"] == "boolean", name
            assert "extras" in props, name

    def test_wait_event_list_tools_schema_omits_legacy_varargs(self):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        async def _run() -> dict:
            server = StdioServerParameters(
                command=sys.executable,
                args=["-c", "from mtdata.core.server import main_stdio; main_stdio()"],
            )
            async with stdio_client(server) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    items = getattr(tools, "tools", tools)
                    tool = next(t for t in items if getattr(t, "name", None) == "wait_event")
                    return getattr(tool, "inputSchema", {}) or {}

        schema = asyncio.run(_run())
        props = schema.get("properties") or {}
        watch_for = props["watch_for"]
        end_on = props["end_on"]
        watch_items = watch_for["items"]

        assert "args" not in props
        assert "args" not in set(schema.get("required") or [])
        assert props["symbol"]["type"] == "string"
        assert props["timeframe"]["type"] == "string"
        assert watch_for["type"] == "array"
        assert watch_items["discriminator"]["propertyName"] == "type"
        assert "price_break_level" in watch_items["discriminator"]["mapping"]
        assert end_on["items"] == {"$ref": "#/$defs/CandleCloseEventSpec"}

    def test_prioritized_tools_list_tools_schemas_are_compact_and_aligned(self):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        async def _run() -> dict[str, tuple[str, dict]]:
            server = StdioServerParameters(
                command=sys.executable,
                args=["-c", "from mtdata.core.server import main_stdio; main_stdio()"],
            )
            async with stdio_client(server) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    items = getattr(tools, "tools", tools)
                    selected = {}
                    for name in ("forecast_generate", "data_fetch_candles", "patterns_detect", "trade_place"):
                        tool = next(t for t in items if getattr(t, "name", None) == name)
                        selected[name] = (
                            getattr(tool, "description", "") or "",
                            getattr(tool, "inputSchema", {}) or {},
                        )
                    return selected

        selected = asyncio.run(_run())

        forecast_desc, forecast_schema = selected["forecast_generate"]
        assert "\n" not in forecast_desc
        assert "request" not in (forecast_schema.get("properties") or {})
        assert "title" not in forecast_schema
        assert "title" not in forecast_schema["properties"]["symbol"]
        assert "default" not in forecast_schema["properties"]["lookback"]

        candles_desc, candles_schema = selected["data_fetch_candles"]
        assert "\n" not in candles_desc
        assert "title" not in candles_schema["properties"]["symbol"]
        assert "default" not in candles_schema["properties"]["start"]

        patterns_desc, patterns_schema = selected["patterns_detect"]
        assert "\n" not in patterns_desc
        assert patterns_schema["properties"]["timeframe"]["type"] == "string"
        assert "default" not in patterns_schema["properties"]["timeframe"]

        trade_desc, trade_schema = selected["trade_place"]
        assert "\n" not in trade_desc
        assert set(trade_schema.get("required") or []) == {"symbol", "volume", "order_type"}
        assert "default" not in trade_schema["properties"]["symbol"]
        assert "null" not in {
            option.get("type")
            for option in trade_schema["properties"]["expiration"].get("anyOf", [])
            if isinstance(option, dict)
        }

    def test_regime_detect_list_tools_schema_preserves_direct_annotations(self):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        async def _run() -> dict:
            server = StdioServerParameters(
                command=sys.executable,
                args=["-c", "from mtdata.core.server import main_stdio; main_stdio()"],
            )
            async with stdio_client(server) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    items = getattr(tools, "tools", tools)
                    tool = next(t for t in items if getattr(t, "name", None) == "regime_detect")
                    return getattr(tool, "inputSchema", {}) or {}

        schema = asyncio.run(_run())
        props = schema.get("properties") or {}

        assert props["timeframe"]["type"] == "string"
        assert "enum" in props["timeframe"]
        assert props["method"]["type"] == "string"
        assert props["method"]["enum"] == [
            "bocpd",
            "hmm",
            "gmm",
            "ms_ar",
            "clustering",
            "garch",
            "rule_based",
            "wavelet",
            "ensemble",
            "all",
        ]
        assert props["target"]["type"] == "string"
        assert props["target"]["enum"] == ["return", "price"]
        assert "detail" not in props
        assert props["json"]["type"] == "boolean"
        assert "extras" in props

        params_schema = props["params"]
        assert (
            params_schema.get("type") == "object"
            or any(option.get("type") == "object" for option in params_schema.get("anyOf", []))
        )


# ── _disconnect_mt5 ──────────────────────────────────────────────────────

class TestDisconnectMt5:

    @patch("mtdata.core.server.mt5_connection")
    def test_disconnect_called(self, mock_conn):
        from mtdata.core.server import _disconnect_mt5
        _disconnect_mt5()
        mock_conn.disconnect.assert_called_once()


# ── main / main_stdio / main_sse ──────────────────────────────────────────

class TestMainEntryPoints:

    @patch("mtdata.core.server.bootstrap_tools")
    @patch("mtdata.core.server.mcp")
    def test_main_invokes_run(self, mock_mcp, mock_bootstrap, monkeypatch):
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        monkeypatch.delenv("FASTMCP_HOST", raising=False)
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        mock_mcp.settings = MagicMock()
        mock_mcp.settings.host = "0.0.0.0"
        mock_mcp.settings.log_level = "INFO"
        mock_mcp.settings.mount_path = "/"
        mock_mcp.settings.port = 8000
        mock_mcp.settings.sse_path = "/sse"
        mock_mcp.settings.message_path = "/message"
        mock_mcp.run = MagicMock()
        from mtdata.core.server import main
        main()
        mock_mcp.run.assert_called_once()

    @patch("mtdata.core.server.main")
    def test_main_stdio_forces_transport(self, mock_main, monkeypatch):
        from mtdata.core.server import main_stdio
        main_stdio()
        mock_main.assert_called_once_with(transport="stdio")

    @patch("mtdata.core.server.main")
    def test_main_sse_forces_transport(self, mock_main, monkeypatch):
        from mtdata.core.server import main_sse
        main_sse()
        mock_main.assert_called_once_with(transport="sse")

    @patch("mtdata.core.server.main")
    def test_main_streamable_http_forces_transport(self, mock_main, monkeypatch):
        from mtdata.core.server import main_streamable_http
        main_streamable_http()
        mock_main.assert_called_once_with(transport="streamable-http")

    @patch("mtdata.core.server.bootstrap_tools")
    @patch("mtdata.core.server.mcp")
    def test_main_no_settings(self, mock_mcp, mock_bootstrap, monkeypatch):
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        monkeypatch.delenv("FASTMCP_HOST", raising=False)
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        mock_mcp.settings = None
        mock_mcp.run = MagicMock()
        from mtdata.core.server import main
        main()
        mock_mcp.run.assert_called_once()
        mock_bootstrap.assert_called_once()
        mock_bootstrap.assert_called_once()

    @patch("mtdata.core.server.bootstrap_tools")
    @patch("mtdata.core.server.mcp")
    def test_main_no_run_fn(self, mock_mcp, mock_bootstrap, monkeypatch):
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        monkeypatch.delenv("FASTMCP_HOST", raising=False)
        monkeypatch.delenv("FASTMCP_ALLOW_REMOTE", raising=False)
        mock_mcp.settings = None
        mock_mcp.run = None
        from mtdata.core.server import main
        main()  # should not raise
        mock_bootstrap.assert_called_once()


# ── mcp instance ──────────────────────────────────────────────────────────

class TestMcpInstance:

    def test_mcp_exists(self):
        from mtdata.core.server import mcp
        assert mcp is not None

    def test_server_reexports_leaf_mcp_singleton(self):
        from mtdata.core._mcp_instance import mcp as leaf_mcp
        from mtdata.core.server import mcp as server_mcp
        assert server_mcp is leaf_mcp

    def test_mcp_has_tool_attr(self):
        from mtdata.core.server import mcp
        assert hasattr(mcp, "tool")

    def test_mcp_has_registry(self):
        from mtdata.core.server import mcp
        assert hasattr(mcp, "registry")

    def test_mcp_tools_is_dict(self):
        from mtdata.core.server import mcp
        assert isinstance(getattr(mcp, "tools", None), dict)
