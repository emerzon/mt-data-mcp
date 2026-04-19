"""Comprehensive tests for mtdata.core.web_api module."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mtdata.bootstrap.runtime import WebApiRuntimeSettings
from mtdata.core import web_api
from mtdata.core.web_api import (
    BacktestBody,
    ForecastPriceBody,
    ForecastVolBody,
    _call_tool_raw,
    _list_sktime_forecasters,
    app,
)
from mtdata.core.web_api_runtime import create_web_api_app, mount_webui
from mtdata.forecast.exceptions import ForecastError
from mtdata.utils.mt5 import MT5ConnectionError

_client = TestClient(app)

# ---------------------------------------------------------------------------
# Helper: convenience for building mock symbol objects
# ---------------------------------------------------------------------------

def _make_symbol(name: str, desc: str = "", visible: bool = True, path: str = ""):
    s = SimpleNamespace(name=name, description=desc, visible=visible, path=path)
    return s


# ===========================================================================
# _call_tool_raw
# ===========================================================================

class TestCallToolRaw:
    def test_returns_wrapped_when_present(self):
        inner = lambda: "raw"
        outer = lambda: "wrapped"
        outer.__wrapped__ = inner
        assert _call_tool_raw(outer) is inner

    def test_returns_func_when_no_wrapped(self):
        fn = lambda: 42
        assert _call_tool_raw(fn) is fn

    def test_returns_func_when_wrapped_not_callable(self):
        fn = lambda: 42
        fn.__wrapped__ = "not_callable"
        assert _call_tool_raw(fn) is fn


# ===========================================================================
# _list_sktime_forecasters
# ===========================================================================

class TestListSktimeForecasters:
    def test_sktime_not_installed(self):
        with patch("mtdata.core.web_api._find_spec", return_value=None):
            res = _list_sktime_forecasters()
        assert res["available"] is False
        assert res["estimators"] == []
        assert "not installed" in res["error"]

    def test_sktime_success(self):
        import pandas as pd
        rows = pd.DataFrame([
            {"name": "ThetaForecaster", "object": type("ThetaForecaster", (), {"__name__": "ThetaForecaster", "__module__": "sktime.forecasting"}), "module": "sktime.forecasting"},
            {"name": "NaiveForecaster", "object": type("NaiveForecaster", (), {"__name__": "NaiveForecaster", "__module__": "sktime.forecasting"}), "module": "sktime.forecasting"},
        ])
        mock_spec = MagicMock()
        mock_all_estimators = MagicMock(return_value=rows)
        with patch("mtdata.core.web_api._find_spec", return_value=mock_spec), \
             patch.dict(sys.modules, {"sktime": MagicMock(), "sktime.registry": MagicMock(all_estimators=mock_all_estimators)}):
            res = _list_sktime_forecasters()
        assert res["available"] is True
        names = [e["name"] for e in res["estimators"]]
        assert "NaiveForecaster" in names
        assert "ThetaForecaster" in names
        # Sorted alphabetically
        assert names == sorted(names, key=str.lower)

    def test_sktime_import_exception(self):
        mock_spec = MagicMock()
        with patch("mtdata.core.web_api._find_spec", return_value=mock_spec), \
             patch.dict(sys.modules, {"sktime": MagicMock(), "sktime.registry": MagicMock(all_estimators=MagicMock(side_effect=RuntimeError("boom")))}):
            res = _list_sktime_forecasters()
        assert res["available"] is False
        assert "boom" in res["error"]

    def test_sktime_row_missing_fields_skipped(self):
        import pandas as pd
        rows = pd.DataFrame([
            {"name": None, "object": None, "module": None},
            {"name": "Good", "object": type("Good", (), {"__name__": "Good", "__module__": "m"}), "module": "m"},
        ])
        mock_spec = MagicMock()
        mock_all_estimators = MagicMock(return_value=rows)
        with patch("mtdata.core.web_api._find_spec", return_value=mock_spec), \
             patch.dict(sys.modules, {"sktime": MagicMock(), "sktime.registry": MagicMock(all_estimators=mock_all_estimators)}):
            res = _list_sktime_forecasters()
        assert res["available"] is True
        assert len(res["estimators"]) == 1
        assert res["estimators"][0]["name"] == "Good"


# ===========================================================================
# Pydantic models
# ===========================================================================

class TestPydanticModels:
    def test_forecast_price_body_defaults(self):
        body = ForecastPriceBody(symbol="EURUSD")
        assert body.timeframe == "H1"
        assert body.method == "theta"
        assert body.horizon == 12
        assert body.ci_alpha == 0.05
        assert body.quantity == "price"
        assert body.to_domain_request().method == "theta"

    def test_forecast_vol_body_defaults(self):
        body = ForecastVolBody(symbol="EURUSD")
        assert body.timeframe == "H1"
        assert body.method == "ewma"
        assert body.horizon == 1

    def test_backtest_body_defaults(self):
        body = BacktestBody(symbol="EURUSD")
        assert body.steps == 5
        assert body.spacing == 20
        assert body.slippage_bps == 0.0
        assert body.trade_threshold == 0.0

    def test_forecast_price_body_custom(self):
        body = ForecastPriceBody(
            symbol="GBPUSD", timeframe="D1", method="arima",
            horizon=5, lookback=200, ci_alpha=0.1,
            quantity="return",
            denoise={"method": "wavelet"}, features={"rsi": {}},
            dimred_method="pca", dimred_params={"n_components": 3},
            target_spec={"col": "close"},
        )
        assert body.symbol == "GBPUSD"
        assert body.quantity == "return"
        assert body.dimred_method == "pca"

    def test_forecast_price_body_rejects_removed_target(self):
        with pytest.raises(ValidationError):
            ForecastPriceBody(symbol="GBPUSD", target="return")

    def test_backtest_body_custom(self):
        body = BacktestBody(
            symbol="USDJPY", methods=["theta", "arima"],
            params_per_method={"theta": {}}, slippage_bps=1.5,
            trade_threshold=0.01,
        )
        assert body.methods == ["theta", "arima"]
        assert body.to_domain_request().quantity == "price"


class TestWebApiSecurity:
    def test_remote_requests_require_token_or_loopback(self, monkeypatch):
        monkeypatch.delenv("WEBAPI_AUTH_TOKEN", raising=False)
        with patch("mtdata.core.web_api._is_local_api_client", return_value=False):
            resp = _client.get("/api/timeframes")
        assert resp.status_code == 403
        assert resp.json()["detail"]["error_code"] == "web_api_remote_forbidden"

    def test_configured_token_requires_auth_header(self, monkeypatch):
        monkeypatch.setenv("WEBAPI_AUTH_TOKEN", "secret")
        resp = _client.get("/api/timeframes")
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"
        assert resp.json()["detail"]["error_code"] == "web_api_auth_required"

    def test_bearer_token_allows_request(self, monkeypatch):
        monkeypatch.setenv("WEBAPI_AUTH_TOKEN", "secret")
        resp = _client.get("/api/timeframes", headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200

    def test_x_api_key_allows_request(self, monkeypatch):
        monkeypatch.setenv("WEBAPI_AUTH_TOKEN", "secret")
        resp = _client.get("/api/timeframes", headers={"X-API-Key": "secret"})
        assert resp.status_code == 200


class TestWebApiRuntimeHelpers:
    def test_create_web_api_app_rejects_wildcard_origins(self):
        with pytest.raises(ValueError, match="CORS_ORIGINS"):
            create_web_api_app(settings=WebApiRuntimeSettings(cors_origins=("*",)))

    def test_mount_webui_logs_warning(self, caplog):
        runtime = WebApiRuntimeSettings(webui_directory="missing-dist")
        runtime_app = create_web_api_app(settings=runtime)
        with caplog.at_level("WARNING"):
            mount_webui(runtime_app, settings=runtime)
        assert any("Skipping Web UI mount" in record.message for record in caplog.records)


class TestWebApiHandlers:
    def test_require_mt5_connection_uses_503_for_mt5_outage(self):
        from fastapi import HTTPException

        from mtdata.core import web_api_handlers

        gateway = MagicMock()
        gateway.ensure_connection.side_effect = MT5ConnectionError("MT5 unavailable")

        with patch("mtdata.core.web_api_handlers.get_default_mt5_gateway", return_value=gateway):
            with pytest.raises(HTTPException) as exc_info:
                web_api_handlers._require_mt5_connection()

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["error_code"] == "mt5_connection_error"


# ===========================================================================
# GET /api/timeframes
# ===========================================================================

class TestGetTimeframes:
    def test_returns_timeframe_list(self):
        resp = _client.get("/api/timeframes")
        assert resp.status_code == 200
        res = resp.json()
        assert "timeframes" in res
        assert "H1" in res["timeframes"]
        assert "D1" in res["timeframes"]
        assert "M1" in res["timeframes"]


# ===========================================================================
# GET /api/instruments
# ===========================================================================

class TestGetInstruments:
    def test_connection_failure(self):
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=False):
            resp = _client.get("/api/instruments")
        assert resp.status_code == 503

    def test_symbols_get_none(self):
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5:
            mock_mt5.symbols_get.return_value = None
            mock_mt5.last_error.return_value = (0, "err")
            resp = _client.get("/api/instruments")
        assert resp.status_code == 500

    def test_returns_visible_symbols_no_search(self):
        syms = [_make_symbol("EURUSD", "Euro vs USD", True), _make_symbol("HIDDEN", "X", False)]
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5, \
             patch("mtdata.core.web_api._extract_group_path_util", return_value="Forex"):
            mock_mt5.symbols_get.return_value = syms
            resp = _client.get("/api/instruments")
        res = resp.json()
        assert len(res["items"]) == 1
        assert res["items"][0]["name"] == "EURUSD"

    def test_search_filters(self):
        syms = [_make_symbol("EURUSD", "Euro"), _make_symbol("USDJPY", "Yen")]
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5, \
             patch("mtdata.core.web_api._extract_group_path_util", return_value="Forex"):
            mock_mt5.symbols_get.return_value = syms
            resp = _client.get("/api/instruments", params={"search": "yen"})
        res = resp.json()
        assert len(res["items"]) == 1
        assert res["items"][0]["name"] == "USDJPY"

    def test_limit(self):
        syms = [_make_symbol(f"SYM{i}", visible=True) for i in range(10)]
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5, \
             patch("mtdata.core.web_api._extract_group_path_util", return_value="Forex"):
            mock_mt5.symbols_get.return_value = syms
            resp = _client.get("/api/instruments", params={"limit": 3})
        res = resp.json()
        assert len(res["items"]) == 3

    def test_symbol_exception_skipped(self):
        bad = MagicMock()
        bad.visible = True
        type(bad).name = PropertyMock(side_effect=RuntimeError("boom"))
        good = _make_symbol("OK", "ok", True)
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5, \
             patch("mtdata.core.web_api._extract_group_path_util", return_value="G"):
            mock_mt5.symbols_get.return_value = [bad, good]
            resp = _client.get("/api/instruments")
        res = resp.json()
        assert any(i["name"] == "OK" for i in res["items"])


# ===========================================================================
# GET /api/methods
# ===========================================================================

class TestGetMethods:
    def test_returns_methods(self):
        data = {"methods": [{"method": "theta", "available": True, "requires": []}]}
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data):
            res = web_api.get_methods()
        assert res["methods"][0]["method"] == "theta"

    def test_returns_empty_on_none(self):
        with patch("mtdata.core.web_api._get_methods_impl", return_value=None):
            res = web_api.get_methods()
        assert res == {"methods": []}

    def test_returns_empty_on_no_methods_key(self):
        with patch("mtdata.core.web_api._get_methods_impl", return_value={"other": 1}):
            res = web_api.get_methods()
        assert res == {"methods": []}

    def test_uses_shared_snapshot_backed_methods_payload(self):
        data = {
            "methods": [
                {"method": "timesfm", "available": False, "requires": ["timesfm"]},
                {"method": "theta", "available": True, "requires": []},
            ]
        }
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data), patch(
            "mtdata.core.web_api_handlers.get_forecast_methods_payload",
            return_value={
                "methods": [
                    {
                        "method": "timesfm",
                        "available": True,
                        "requires": ["timesfm"],
                        "namespace": "pretrained",
                        "method_id": "pretrained:timesfm",
                    },
                    {
                        "method": "theta",
                        "available": False,
                        "requires": [],
                        "namespace": "native",
                        "method_id": "native:theta",
                    },
                ]
            },
        ):
            res = web_api.get_methods()
        assert res["methods"] == [
            {
                "method": "timesfm",
                "available": True,
                "requires": ["timesfm"],
                "namespace": "pretrained",
                "method_id": "pretrained:timesfm",
            },
            {
                "method": "theta",
                "available": False,
                "requires": [],
                "namespace": "native",
                "method_id": "native:theta",
            },
        ]

    def test_snapshot_exception_keeps_original_methods(self):
        data = {"methods": [{"method": "custom", "available": True, "requires": []}]}
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data), patch(
            "mtdata.core.web_api_handlers.get_forecast_methods_payload",
            side_effect=RuntimeError("boom"),
        ):
            res = web_api.get_methods()
        assert res == data


# ===========================================================================
# GET /api/models
# ===========================================================================

class TestGetModels:
    def test_returns_models_with_compact_detail_by_default(self):
        data = {
            "success": True,
            "detail": "compact",
            "count": 1,
            "models": [{"model_id": "nhits/EURUSD_H1/a", "method": "nhits"}],
        }
        with patch("mtdata.core.web_api._get_models_impl", return_value=data):
            resp = _client.get("/api/models")
        assert resp.status_code == 200
        assert resp.json() == data

    def test_passes_method_and_detail_to_models_impl(self):
        data = {
            "success": True,
            "detail": "full",
            "count": 1,
            "models": [
                {
                    "model_id": "nhits/EURUSD_H1/a",
                    "method": "nhits",
                    "metadata": {"epochs": 42},
                }
            ],
        }
        with patch("mtdata.core.web_api._get_models_impl", return_value=data) as mock_models:
            resp = _client.get("/api/models", params={"method": "nhits", "detail": "full"})
        assert resp.status_code == 200
        assert resp.json() == data
        mock_models.assert_called_once_with(method="nhits", detail="full")

    def test_returns_empty_payload_on_invalid_models_result(self):
        with patch("mtdata.core.web_api._get_models_impl", return_value=None):
            resp = _client.get("/api/models", params={"detail": "full"})
        assert resp.status_code == 200
        assert resp.json() == {"success": True, "detail": "full", "count": 0, "models": []}

    def test_rejects_invalid_detail_query(self):
        resp = _client.get("/api/models", params={"detail": "verbose"})
        assert resp.status_code == 422


# ===========================================================================
# GET /api/volatility/methods
# ===========================================================================

class TestGetVolMethods:
    def test_returns_dict(self):
        data = {"methods": [{"method": "ewma"}]}
        with patch("mtdata.core.web_api._get_vol_methods", return_value=data):
            res = web_api.get_vol_methods()
        assert res == data

    def test_non_dict_returns_empty(self):
        with patch("mtdata.core.web_api._get_vol_methods", return_value="bad"):
            res = web_api.get_vol_methods()
        assert res == {"methods": []}


# ===========================================================================
# GET /api/sktime/estimators
# ===========================================================================

class TestGetSktimeEstimators:
    def test_delegates_to_list_sktime(self):
        expected = {"available": False, "error": "sktime not installed", "estimators": []}
        with patch("mtdata.core.web_api._list_sktime_forecasters", return_value=expected):
            res = web_api.get_sktime_estimators()
        assert res == expected


# ===========================================================================
# GET /api/denoise/methods
# ===========================================================================

class TestGetDenoiseMethods:
    def test_returns_data(self):
        data = {"methods": [{"method": "wavelet"}]}
        with patch("mtdata.core.web_api._get_denoise_methods", return_value=data):
            res = web_api.get_denoise_methods()
        assert res == data

    def test_non_dict_returns_empty(self):
        with patch("mtdata.core.web_api._get_denoise_methods", return_value="bad"):
            res = web_api.get_denoise_methods()
        assert res == {"methods": []}

    def test_dict_no_methods_key(self):
        with patch("mtdata.core.web_api._get_denoise_methods", return_value={"other": 1}):
            res = web_api.get_denoise_methods()
        assert res == {"methods": []}


# ===========================================================================
# GET /api/dimred/methods
# ===========================================================================

class TestGetDimredMethods:
    def test_returns_items_with_params(self):
        base = {
            "pca": {"available": True, "description": "PCA"},
            "umap": {"available": False, "description": "UMAP"},
        }
        with patch("mtdata.core.web_api._list_dimred_methods", return_value=base):
            res = web_api.get_dimred_methods()
        methods = {m["method"]: m for m in res["methods"]}
        assert methods["pca"]["available"] is True
        assert len(methods["pca"]["params"]) > 0
        assert methods["umap"]["available"] is False

    def test_unknown_method_gets_empty_params(self):
        base = {"custom_thing": {"available": True, "description": "Custom"}}
        with patch("mtdata.core.web_api._list_dimred_methods", return_value=base):
            res = web_api.get_dimred_methods()
        assert res["methods"][0]["params"] == []


# ===========================================================================
# GET /api/denoise/wavelets
# ===========================================================================

class TestGetWavelets:
    def test_pywt_not_installed(self):
        with patch.dict(sys.modules, {"pywt": None}):
            resp = _client.get("/api/denoise/wavelets")
        res = resp.json()
        assert res["available"] is False
        assert res["wavelets"] == []

    def test_pywt_with_families(self):
        mock_pywt = MagicMock()
        mock_pywt.families.return_value = ["haar", "db"]
        mock_pywt.wavelist.side_effect = lambda f, **kw: [f"{f}1", f"{f}2"]
        with patch.dict(sys.modules, {"pywt": mock_pywt}):
            resp = _client.get("/api/denoise/wavelets")
        res = resp.json()
        assert res["available"] is True
        assert "haar" in res["families"]
        assert "haar1" in res["wavelets"]
        assert "haar1" in res["by_family"]["haar"]

    def test_pywt_families_empty_fallback(self):
        mock_pywt = MagicMock()
        mock_pywt.families.return_value = []
        mock_pywt.wavelist.return_value = ["db1", "db2"]
        with patch.dict(sys.modules, {"pywt": mock_pywt}):
            resp = _client.get("/api/denoise/wavelets")
        res = resp.json()
        assert res["available"] is True
        assert "db1" in res["wavelets"]

    def test_pywt_families_raises(self):
        mock_pywt = MagicMock()
        mock_pywt.families.side_effect = AttributeError("no families")
        mock_pywt.wavelist.return_value = ["fallback1"]
        with patch.dict(sys.modules, {"pywt": mock_pywt}):
            resp = _client.get("/api/denoise/wavelets")
        res = resp.json()
        assert res["available"] is True

    def test_pywt_wavelist_first_call_fails_falls_to_discrete(self):
        mock_pywt = MagicMock()
        mock_pywt.families.return_value = ["haar"]

        def wavelist_side(fam=None, kind=None, **kw):
            if fam and kind is None:
                raise TypeError("old api")
            return ["haar1"]

        mock_pywt.wavelist.side_effect = wavelist_side
        with patch.dict(sys.modules, {"pywt": mock_pywt}):
            resp = _client.get("/api/denoise/wavelets")
        res = resp.json()
        assert res["available"] is True


# ===========================================================================
# GET /api/history  (via TestClient to resolve Query() defaults)
# ===========================================================================

class TestGetHistory:
    def test_connection_failure(self):
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=False):
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        assert resp.status_code == 503

    def test_basic_success(self):
        payload = {"data": [{"time": 1.0, "close": 1.1}], "last_candle_open": False}
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.server_tz_name = "Europe/Nicosia"
            mock_cfg.client_tz_name = None
            mock_cfg.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
            mock_cfg.get_client_tz.return_value = None
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "timeframe": "H1", "limit": 500})
        res = resp.json()
        assert res["data"] == [{"time": 1.0, "close": 1.1}]
        assert res["candles"] == 1
        assert res["meta"]["tool"] == "data_fetch_candles"
        assert res["meta"]["runtime"]["timezone"] == {
            "utc": {"tz": "UTC"},
            "server": {
                "source": "MT5_SERVER_TZ",
                "tz": "Europe/Nicosia",
                "offset_seconds": 0,
            },
            "used": {"tz": "UTC"},
        }

    def test_strips_incomplete_candle(self):
        payload = {"data": [{"time": 1.0}, {"time": 2.0}], "last_candle_open": True}
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "include_incomplete": "false"})
        res = resp.json()
        assert len(res["data"]) == 1

    def test_keeps_incomplete_candle_when_flag(self):
        payload = {"data": [{"time": 1.0}, {"time": 2.0}], "last_candle_open": True}
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "include_incomplete": "true"})
        res = resp.json()
        assert len(res["data"]) == 2

    def test_fetch_exception(self):
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", side_effect=RuntimeError("fail")):
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert detail["error_code"] == "history_fetch_internal_error"
        assert detail["error"] == "History fetch failed."

    def test_fetch_mt5_exception(self):
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", side_effect=MT5ConnectionError("mt5 unavailable")):
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        assert resp.status_code == 503
        assert resp.json()["detail"]["error_code"] == "history_mt5_unavailable"

    def test_non_dict_result(self):
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value="bad"):
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        assert resp.status_code == 500

    def test_error_in_result(self):
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value={"error": "oops", "data": []}):
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        assert resp.status_code == 400

    def test_denoise_json_params(self):
        payload = {"data": [{"time": 1.0, "close": 1.1}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"params": {"wavelet": "db4"}, "columns": "close,high"})
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD", "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        assert resp.status_code == 200
        mock_norm.assert_called_once()

    def test_denoise_kv_params_fallback(self):
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD", "denoise_method": "wavelet",
                "denoise_params": "level=3,wavelet=db4",
            })
        assert resp.status_code == 200
        call_arg = mock_norm.call_args[0][0]
        assert call_arg["params"]["level"] == 3.0
        assert call_arg["params"]["wavelet"] == "db4"

    def test_denoise_kv_params_rejects_duplicate_keys(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods):
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": "level=3,level=4",
            })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "denoise_params_invalid"
        assert "duplicate key 'level'" in detail["error"]

    def test_denoise_unavailable_method(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": False, "requires": "pywt"}]}
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods):
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "denoise_method": "wavelet"})
        assert resp.status_code == 400

    def test_denoise_metadata_failure_is_not_swallowed(self):
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", side_effect=RuntimeError("bad metadata")):
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "denoise_method": "wavelet"})
        assert resp.status_code == 500
        assert resp.json()["detail"]["error_code"] == "denoise_validation_failed"

    def test_denoise_json_extra_params_no_params_key(self):
        """When JSON dict has no 'params' key, non-reserved keys become params."""
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"level": 3, "wavelet": "db4"})
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            _client.get("/api/history", params={
                "symbol": "EURUSD", "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        call_arg = mock_norm.call_args[0][0]
        assert call_arg["params"]["level"] == 3
        assert call_arg["params"]["wavelet"] == "db4"

    def test_denoise_json_with_columns_list_and_when(self):
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({
            "columns": ["close", "high"],
            "when": "pre_ti",
            "causality": "causal",
            "keep_original": False,
        })
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            _client.get("/api/history", params={
                "symbol": "EURUSD", "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        call_arg = mock_norm.call_args[0][0]
        assert call_arg["columns"] == ["close", "high"]
        assert call_arg["when"] == "pre_ti"
        assert call_arg["causality"] == "causal"
        assert call_arg["keep_original"] is False

    def test_denoise_json_rejects_invalid_columns_type(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"columns": {"nested": "object"}})
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "denoise_params_invalid"
        assert "columns" in detail["error"]

    def test_denoise_json_rejects_non_string_column_items(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"columns": ["close", {"bad": "item"}]})
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "denoise_params_invalid"
        assert "columns[1]" in detail["error"]

    def test_denoise_json_string_false_is_parsed_as_false(self):
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"keep_original": "false"})
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        call_arg = mock_norm.call_args[0][0]
        assert call_arg["keep_original"] is False

    def test_denoise_json_rejects_invalid_causality_type(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"causality": True})
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "denoise_params_invalid"
        assert "causality" in detail["error"]

    def test_denoise_json_rejects_non_object_params(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"params": [1, 2, 3]})
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "denoise_params_invalid"
        assert "params" in detail["error"]

    def test_data_not_list(self):
        payload = {"data": "not_a_list"}
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        res = resp.json()
        assert res["data"] == []


# ===========================================================================
# GET /api/pivots  (via TestClient)
# ===========================================================================

class TestGetPivots:
    def _pivot_result(self):
        return {
            "levels": [
                {"level": "P", "classic": 1.1000},
                {"level": "R1", "classic": 1.1100},
                {"level": "S1", "classic": 1.0900},
            ],
            "period": "2025-01-01",
            "symbol": "EURUSD",
            "timeframe": "D1",
        }

    def test_success(self):
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value=self._pivot_result())
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "timeframe": "D1", "method": "classic"})
        assert resp.status_code == 200
        res = resp.json()
        assert len(res["levels"]) == 3
        assert res["method"] == "classic"

    def test_string_result_parsed(self):
        raw_str = json.dumps(self._pivot_result())
        # Inject json module into web_api namespace for the json.loads call
        import json as _json_mod
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value=raw_str)
            with patch.object(web_api, "json", _json_mod, create=True):
                resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 200
        res = resp.json()
        assert len(res["levels"]) == 3

    def test_error_in_result(self):
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value={"error": "bad", "levels": []})
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 400
        assert resp.json()["detail"]["error_code"] == "pivot_tool_error"

    def test_no_levels_found(self):
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value={"levels": [{"level": "P", "fibonacci": 1.1}]})
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 404
        assert resp.json()["detail"]["error_code"] == "pivot_levels_missing"

    def test_non_dict_result(self):
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value=42)
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 500
        assert resp.json()["detail"]["error_code"] == "pivot_payload_invalid"

    def test_typeerror_fallback(self):
        """When the raw function raises TypeError, falls back to calling original."""
        raw_fn = MagicMock(side_effect=TypeError("wrong args"))
        with patch("mtdata.core.web_api._call_tool_raw", return_value=raw_fn), \
             patch("mtdata.core.web_api.pivot_compute_points", return_value=self._pivot_result()):
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 200
        assert len(resp.json()["levels"]) == 3

    def test_typeerror_fallback_resolves_async_original(self):
        raw_fn = MagicMock(side_effect=TypeError("wrong args"))

        async def async_pivot(**_kwargs):
            return self._pivot_result()

        with patch("mtdata.core.web_api._call_tool_raw", return_value=raw_fn), \
             patch("mtdata.core.web_api.pivot_compute_points", new=async_pivot):
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 200
        assert len(resp.json()["levels"]) == 3

    def test_generic_exception(self):
        raw_fn = MagicMock(side_effect=ValueError("boom"))
        with patch("mtdata.core.web_api._call_tool_raw", return_value=raw_fn):
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 500
        assert resp.json()["detail"]["error_code"] == "pivot_compute_failed"

    def test_string_not_json(self):
        import json as _json_mod
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value="not json")
            with patch.object(web_api, "json", _json_mod, create=True):
                resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 500
        assert resp.json()["detail"]["error_code"] == "pivot_output_invalid"


# ===========================================================================
# GET /api/tick
# ===========================================================================

class TestGetTick:
    def _mock_tick(self, time=100.0, bid=1.1, ask=1.2, last=1.15, volume=500.0):
        return SimpleNamespace(time=time, bid=bid, ask=ask, last=last, volume=volume)

    def test_connection_failure(self):
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=False):
            resp = _client.get("/api/tick", params={"symbol": "EURUSD"})
        assert resp.status_code == 503

    def test_success(self):
        tick = self._mock_tick()
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5:
            mock_mt5.symbol_info_tick.return_value = tick
            resp = _client.get("/api/tick", params={"symbol": "EURUSD"})
        res = resp.json()
        assert res["bid"] == 1.1
        assert res["ask"] == 1.2
        assert res["symbol"] == "EURUSD"

    def test_tick_none_retry_success(self):
        tick = self._mock_tick()
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5, \
             patch("mtdata.core.web_api._ensure_symbol_ready", return_value=None):
            mock_mt5.symbol_info_tick.side_effect = [None, tick]
            resp = _client.get("/api/tick", params={"symbol": "EURUSD"})
        assert resp.json()["bid"] == 1.1

    def test_tick_none_symbol_unknown(self):
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5, \
             patch("mtdata.core.web_api._ensure_symbol_ready", return_value="some error"):
            mock_mt5.symbol_info_tick.return_value = None
            mock_mt5.symbol_info.return_value = None
            resp = _client.get("/api/tick", params={"symbol": "FAKE"})
        assert resp.status_code == 404

    def test_tick_none_ensure_error_known_symbol(self):
        info = SimpleNamespace(name="EURUSD")
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5, \
             patch("mtdata.core.web_api._ensure_symbol_ready", return_value="error"):
            mock_mt5.symbol_info_tick.return_value = None
            mock_mt5.symbol_info.return_value = info
            resp = _client.get("/api/tick", params={"symbol": "EURUSD"})
        assert resp.status_code == 500

    def test_tick_none_after_retry(self):
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5, \
             patch("mtdata.core.web_api._ensure_symbol_ready", return_value=None):
            mock_mt5.symbol_info_tick.return_value = None
            resp = _client.get("/api/tick", params={"symbol": "EURUSD"})
        assert resp.status_code == 404


# ===========================================================================
# POST /api/forecast/price
# ===========================================================================

class TestPostForecastPrice:
    def test_success(self):
        result = {"forecast_price": [1.1, 1.2], "forecast_epoch": [1, 2]}
        with patch("mtdata.core.web_api._run_forecast_generate_impl", return_value=result):
            resp = _client.post("/api/forecast/price", json={"symbol": "EURUSD"})
        assert resp.status_code == 200
        assert resp.json() == result

    def test_error_in_result(self):
        with patch("mtdata.core.web_api._run_forecast_generate_impl", return_value={"error": "fail"}):
            resp = _client.post("/api/forecast/price", json={"symbol": "EURUSD"})
        assert resp.status_code == 400

    def test_passes_all_params(self):
        with patch("mtdata.core.web_api._run_forecast_generate_impl", return_value={}) as mock_fc:
            _client.post("/api/forecast/price", json={
                "symbol": "GBPUSD", "timeframe": "D1", "method": "arima",
                "horizon": 5, "lookback": 200, "as_of": "2025-01-01",
                "params": {"order": [1, 1, 1]}, "ci_alpha": 0.1,
                "quantity": "return",
                "denoise": {"method": "wavelet"}, "features": {"rsi": {}},
                "dimred_method": "pca", "dimred_params": {"n": 3},
                "target_spec": {"col": "close"},
            })
        request = mock_fc.call_args.args[0]
        assert request.symbol == "GBPUSD"
        assert request.method == "arima"
        assert request.horizon == 5
        assert request.quantity == "return"
        assert request.dimred_method == "pca"
        assert request.target_spec == {"col": "close"}
        assert request.params == {"order": [1, 1, 1]}

    def test_removed_target_is_rejected(self):
        with patch("mtdata.core.web_api._run_forecast_generate_impl", return_value={}) as mock_fc:
            resp = _client.post("/api/forecast/price", json={"symbol": "EURUSD", "target": "return"})
        assert resp.status_code == 422
        mock_fc.assert_not_called()

    def test_non_dict_result_returned(self):
        """Non-dict return passes through without error check."""
        body = ForecastPriceBody(symbol="EURUSD")
        with patch("mtdata.core.web_api._run_forecast_generate_impl", return_value="raw_string"):
            res = web_api.post_forecast_price(body)
        assert res == "raw_string"

    def test_typed_forecast_error_becomes_http_400(self):
        with patch("mtdata.core.web_api._run_forecast_generate_impl", side_effect=ForecastError("engine exploded")):
            resp = _client.post("/api/forecast/price", json={"symbol": "EURUSD"})
        assert resp.status_code == 400
        assert "engine exploded" in resp.text


# ===========================================================================
# POST /api/forecast/volatility
# ===========================================================================

class TestPostForecastVolatility:
    def test_success(self):
        result = {"forecast_vol": [0.01]}
        with patch("mtdata.core.web_api._forecast_vol_impl", return_value=result):
            resp = _client.post("/api/forecast/volatility", json={"symbol": "EURUSD"})
        assert resp.status_code == 200
        assert resp.json() == result

    def test_error(self):
        with patch("mtdata.core.web_api._forecast_vol_impl", return_value={"error": "fail"}):
            resp = _client.post("/api/forecast/volatility", json={"symbol": "EURUSD"})
        assert resp.status_code == 400

    def test_internal_exception_is_sanitized(self):
        with patch("mtdata.core.web_api._forecast_vol_impl", side_effect=RuntimeError("engine exploded")):
            resp = _client.post("/api/forecast/volatility", json={"symbol": "EURUSD"})
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert detail["error_code"] == "forecast_volatility_internal_error"
        assert "engine exploded" not in detail["error"]

    def test_passes_all_params(self):
        with patch("mtdata.core.web_api._forecast_vol_impl", return_value={}) as mock_fv:
            _client.post("/api/forecast/volatility", json={
                "symbol": "EURUSD", "timeframe": "D1", "horizon": 5,
                "method": "garch", "proxy": "close", "params": {"p": 1},
                "as_of": "2025-01-01", "denoise": {"method": "wavelet"},
            })
        kw = mock_fv.call_args.kwargs
        assert kw["method"] == "garch"
        assert kw["proxy"] == "close"
        assert kw["denoise"] == {"method": "wavelet"}


# ===========================================================================
# POST /api/backtest
# ===========================================================================

class TestPostBacktest:
    def test_success(self):
        result = {"results": []}
        with patch("mtdata.core.web_api._run_forecast_backtest_impl", return_value=result):
            resp = _client.post("/api/backtest", json={"symbol": "EURUSD"})
        assert resp.status_code == 200
        assert resp.json() == result

    def test_error(self):
        with patch("mtdata.core.web_api._run_forecast_backtest_impl", return_value={"error": "fail"}):
            resp = _client.post("/api/backtest", json={"symbol": "EURUSD"})
        assert resp.status_code == 400

    def test_internal_exception_is_sanitized(self):
        with patch("mtdata.core.web_api._run_forecast_backtest_impl", side_effect=RuntimeError("secret trace")):
            resp = _client.post("/api/backtest", json={"symbol": "EURUSD"})
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert detail["error_code"] == "backtest_internal_error"
        assert "secret trace" not in detail["error"]

    def test_passes_all_params(self):
        with patch("mtdata.core.web_api._run_forecast_backtest_impl", return_value={}) as mock_bt:
            _client.post("/api/backtest", json={
                "symbol": "EURUSD", "timeframe": "D1", "horizon": 10,
                "steps": 3, "spacing": 10, "methods": ["theta"],
                "params_per_method": {"theta": {}}, "quantity": "return",
                "denoise": {"method": "wavelet"},
                "params": {"extra": True}, "features": {"rsi": {}},
                "dimred_method": "pca", "dimred_params": {"n": 2},
                "slippage_bps": 1.5, "trade_threshold": 0.01, "detail": "full",
            })
        request = mock_bt.call_args.args[0]
        assert request.slippage_bps == 1.5
        assert request.trade_threshold == 0.01
        assert request.dimred_method == "pca"
        assert request.methods == ["theta"]
        assert request.quantity == "return"
        assert request.detail == "full"

    def test_backtest_removed_target_is_rejected(self):
        with patch("mtdata.core.web_api._run_forecast_backtest_impl", return_value={}) as mock_bt:
            resp = _client.post("/api/backtest", json={"symbol": "EURUSD", "target": "return"})
        assert resp.status_code == 422
        mock_bt.assert_not_called()


# ===========================================================================
# GET /
# ===========================================================================

class TestRoot:
    def test_root(self):
        resp = _client.get("/")
        assert resp.status_code == 200
        res = resp.json()
        assert res["service"] == "mtdata-webui"
        assert res["status"] == "ok"


# ===========================================================================
# main_webapi
# ===========================================================================

class TestMainWebapi:
    def test_main_calls_uvicorn(self):
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            web_api.main_webapi()
        mock_uvicorn.run.assert_called_once_with(app, host="127.0.0.1", port=8000)


# ===========================================================================
# GET /api/support-resistance  (via TestClient)
# ===========================================================================

class TestGetSupportResistance:
    def _sr_params(self, **kw):
        defaults = {"symbol": "EURUSD", "timeframe": "H1", "limit": 800,
                     "tolerance_pct": 0.0015, "min_touches": 2, "max_levels": 4}
        defaults.update(kw)
        return defaults

    def test_fetch_exception(self):
        with patch("mtdata.core.web_api._fetch_history_impl", side_effect=RuntimeError("fail")):
            resp = _client.get("/api/support-resistance", params=self._sr_params())
        assert resp.status_code == 400
        assert resp.json()["detail"]["error_code"] == "support_resistance_history_failed"

    def test_empty_df(self):
        import pandas as pd
        with patch("mtdata.core.web_api._fetch_history_impl", return_value=pd.DataFrame()):
            resp = _client.get("/api/support-resistance", params=self._sr_params())
        assert resp.status_code == 404
        assert resp.json()["detail"]["error_code"] == "support_resistance_history_missing"

    def test_none_df(self):
        with patch("mtdata.core.web_api._fetch_history_impl", return_value=None):
            resp = _client.get("/api/support-resistance", params=self._sr_params())
        assert resp.status_code == 404
        assert resp.json()["detail"]["error_code"] == "support_resistance_history_missing"

    def test_missing_columns(self):
        import pandas as pd
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with patch("mtdata.core.web_api._fetch_history_impl", return_value=df):
            resp = _client.get("/api/support-resistance", params=self._sr_params())
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "support_resistance_history_failed"
        assert "Missing columns" in detail["error"]

    def test_too_few_bars(self):
        import pandas as pd
        df = pd.DataFrame({"high": [1.1, 1.2], "low": [1.0, 1.05], "close": [1.05, 1.1], "time": [1, 2]})
        with patch("mtdata.core.web_api._fetch_history_impl", return_value=df):
            resp = _client.get("/api/support-resistance", params=self._sr_params())
        assert resp.status_code == 400

    def test_success_with_levels(self):
        import pandas as pd
        n = 20
        highs = [1.10, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09,
                 1.10, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09]
        lows =  [1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08,
                 1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08]
        df = pd.DataFrame({
            "high": highs, "low": lows,
            "close": [(h + l) / 2 for h, l in zip(highs, lows)],
            "time": [1700000000 + i * 3600 for i in range(n)],
        })
        with patch("mtdata.core.web_api._fetch_history_impl", return_value=df):
            resp = _client.get("/api/support-resistance", params=self._sr_params(
                tolerance_pct=0.01, min_touches=1, max_levels=4,
            ))
        res = resp.json()
        assert resp.status_code == 200
        assert "levels" in res
        assert len(res["levels"]) > 0
        assert res["symbol"] == "EURUSD"
        assert "window" in res
        assert "fibonacci" in res
        assert len(res["fibonacci"]["levels"]) == 7
        assert res["fibonacci"]["nearest"]["support"]["type"] == "support"

    def test_default_timeframe_uses_auto_mode(self):
        import pandas as pd

        n = 20
        frame = pd.DataFrame({
            "high": [1.10, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09,
                     1.10, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09],
            "low": [1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08,
                    1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08],
            "close": [1.09] * n,
            "time": [1700000000 + i * 3600 for i in range(n)],
        })

        with patch("mtdata.core.web_api._fetch_history_impl", return_value=frame) as mock_fetch:
            resp = _client.get("/api/support-resistance", params={"symbol": "EURUSD"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["timeframe"] == "auto"
        assert body["mode"] == "auto"
        assert body["timeframes_analyzed"] == ["M15", "H1", "H4", "D1"]
        assert body["fibonacci"]["mode"] == "auto"
        assert body["fibonacci"]["selected_timeframe"] == "D1"
        assert mock_fetch.call_count == 4

    def test_no_levels_detected(self):
        import pandas as pd
        # Strictly monotonic data: no local extrema (center never >= both neighbors for highs)
        n = 10
        df = pd.DataFrame({
            "high": [1.10 + 0.001 * i for i in range(n)],
            "low": [1.09 + 0.001 * i for i in range(n)],
            "close": [1.095 + 0.001 * i for i in range(n)],
            "time": [1700000000 + i * 3600 for i in range(n)],
        })
        with patch("mtdata.core.web_api._fetch_history_impl", return_value=df):
            resp = _client.get("/api/support-resistance", params=self._sr_params(
                min_touches=100, max_levels=4,
            ))
        assert resp.status_code == 404
        assert resp.json()["detail"]["error_code"] == "support_resistance_levels_missing"

    def test_no_time_column(self):
        import pandas as pd
        n = 20
        highs = [1.10, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09,
                 1.10, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09]
        lows =  [1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08,
                 1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08]
        df = pd.DataFrame({
            "high": highs, "low": lows,
            "close": [(h + l) / 2 for h, l in zip(highs, lows)],
        })
        with patch("mtdata.core.web_api._fetch_history_impl", return_value=df):
            resp = _client.get("/api/support-resistance", params=self._sr_params(
                tolerance_pct=0.01, min_touches=1,
            ))
        res = resp.json()
        assert resp.status_code == 200
        assert "levels" in res

    def test_datetime_timestamps_in_time(self):
        """Handles datetime objects in the time column."""
        import pandas as pd
        n = 20
        highs = [1.10, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09,
                 1.10, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09, 1.12, 1.10, 1.09]
        lows =  [1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08,
                 1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08, 1.09, 1.07, 1.08]
        times = [datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc) for i in range(n)]
        df = pd.DataFrame({
            "high": highs, "low": lows,
            "close": [(h + l) / 2 for h, l in zip(highs, lows)],
            "time": times,
        })
        with patch("mtdata.core.web_api._fetch_history_impl", return_value=df):
            resp = _client.get("/api/support-resistance", params=self._sr_params(
                tolerance_pct=0.01, min_touches=1,
            ))
        res = resp.json()
        assert resp.status_code == 200
        assert "levels" in res
        if "window" in res:
            assert res["window"]["start"] is not None

    def test_cluster_with_single_touch_fallback(self):
        """When min_touches is high, falls back to returning first cluster."""
        import pandas as pd
        n = 10
        highs = [1.10 + 0.01 * i for i in range(n)]
        lows = [1.08 + 0.01 * i for i in range(n)]
        highs[3] = max(highs) + 0.05
        lows[6] = min(lows) - 0.05
        df = pd.DataFrame({
            "high": highs, "low": lows,
            "close": [(h + l) / 2 for h, l in zip(highs, lows)],
            "time": [1700000000 + i * 3600 for i in range(n)],
        })
        with patch("mtdata.core.web_api._fetch_history_impl", return_value=df):
            resp = _client.get("/api/support-resistance", params=self._sr_params(
                tolerance_pct=0.0001, min_touches=1,
            ))
        assert resp.status_code == 200
        assert len(resp.json()["levels"]) >= 1


# ===========================================================================
# Additional edge-case tests
# ===========================================================================

class TestHistoryDenoiseEdgeCases:
    def test_denoise_method_whitespace_stripped(self):
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "denoise_method": "  wavelet  "})
        assert resp.status_code == 200
        assert resp.json()["data"] == [{"time": 1.0}]

    def test_denoise_empty_string_no_denoise(self):
        payload = {"data": [{"time": 1.0}]}
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload) as mock_fetch, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "denoise_method": "   "})
        assert resp.status_code == 200
        kw = mock_fetch.call_args.kwargs
        assert kw["denoise"] is None

    def test_denoise_json_non_dict_payload(self):
        """JSON payload that is a list should fall through to kv parsing."""
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps([1, 2, 3])
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            _client.get("/api/history", params={
                "symbol": "EURUSD", "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        mock_norm.assert_called_once()


class TestInstrumentSearchEdgeCases:
    def test_search_includes_hidden_symbols(self):
        """When search is provided, hidden symbols are also checked."""
        syms = [_make_symbol("EURUSD", "Euro", visible=False)]
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5, \
             patch("mtdata.core.web_api._extract_group_path_util", return_value="Forex"):
            mock_mt5.symbols_get.return_value = syms
            resp = _client.get("/api/instruments", params={"search": "eur"})
        assert len(resp.json()["items"]) == 1

    def test_empty_search_string(self):
        """Empty search string falls back to visible-only filter."""
        syms = [_make_symbol("EURUSD", "Euro", visible=True), _make_symbol("HIDDEN", "", visible=False)]
        with patch.object(web_api.mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api.mt5") as mock_mt5, \
             patch("mtdata.core.web_api._extract_group_path_util", return_value="G"):
            mock_mt5.symbols_get.return_value = syms
            resp = _client.get("/api/instruments", params={"search": ""})
        res = resp.json()
        assert len(res["items"]) == 1
        assert res["items"][0]["name"] == "EURUSD"


class TestMethodsAvailabilityEdgeCases:
    def test_chronos2_uses_snapshot_payload(self):
        data = {"methods": [{"method": "chronos2", "available": False, "requires": ["chronos"]}]}
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data), patch(
            "mtdata.core.web_api_handlers.get_forecast_methods_payload",
            return_value={
                "methods": [
                    {
                        "method": "chronos2",
                        "available": True,
                        "requires": ["chronos"],
                        "namespace": "pretrained",
                    }
                ]
            },
        ):
            res = web_api.get_methods()
        assert res["methods"][0]["available"] is True
        assert res["methods"][0]["requires"] == ["chronos"]
        assert res["methods"][0]["namespace"] == "pretrained"

    def test_snapshot_exception_passes(self):
        """Exceptions while reading the shared snapshot are swallowed."""
        data = {"methods": [{"method": "timesfm", "available": False}]}
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data), patch(
            "mtdata.core.web_api_handlers.get_forecast_methods_payload",
            side_effect=RuntimeError("boom"),
        ):
            res = web_api.get_methods()
        assert res == data

