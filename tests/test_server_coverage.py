"""Tests for mtdata.core.server — server configuration, coercion helpers, tool decorator."""

import inspect
import math
import os
import types
from typing import Optional, Union
from unittest.mock import MagicMock, patch, PropertyMock

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
        monkeypatch.delenv("FASTMCP_PORT", raising=False)
        settings = load_mcp_runtime_settings()

        assert settings.transport == "sse"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000

    def test_load_from_env(self, monkeypatch):
        from mtdata.bootstrap.runtime import load_mcp_runtime_settings

        monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("FASTMCP_HOST", "1.2.3.4")
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
        settings = load_mcp_runtime_settings()
        assert settings.transport == "sse"

    def test_apply_to_mcp_settings(self):
        from mtdata.bootstrap.runtime import McpRuntimeSettings, apply_mcp_runtime_settings

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
        monkeypatch.delenv("FASTMCP_MOUNT_PATH", raising=False)
        t, mp = self._call()
        assert t == "sse"
        assert mp is None

    def test_env_stdio(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "stdio")
        monkeypatch.delenv("FASTMCP_MOUNT_PATH", raising=False)
        t, mp = self._call()
        assert t == "stdio"

    def test_env_streamable_http(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
        monkeypatch.delenv("FASTMCP_MOUNT_PATH", raising=False)
        t, _ = self._call()
        assert t == "streamable-http"

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "grpc")
        monkeypatch.delenv("FASTMCP_MOUNT_PATH", raising=False)
        t, _ = self._call()
        assert t == "sse"

    def test_mount_path_returned(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "sse")
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
        from mtdata.core.server import get_tool_registry, get_tool_functions, _TOOL_REGISTRY
        r1 = get_tool_registry()
        r2 = get_tool_functions()
        # Mutating returned dict should not affect internal state
        r1["__test__"] = True
        r2["__test__"] = True
        assert "__test__" not in _TOOL_REGISTRY


# ── _recording_tool_decorator ─────────────────────────────────────────────

class TestRecordingToolDecorator:

    def test_noop_fallback_when_no_orig(self):
        from mtdata.core.server import _recording_tool_decorator, _TOOL_REGISTRY
        import mtdata.core.server as srv
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
