from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
from inspect import signature
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from mtdata.core import forecast as cf
from mtdata.core import options as opt
from mtdata.forecast import barriers_shared
from mtdata.forecast import use_cases as forecast_use_cases
from mtdata.forecast.exceptions import ForecastError
from mtdata.forecast.requests import (
    ForecastBacktestRequest,
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastOptimizeHintsRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
    ForecastVolatilityEstimateRequest,
)
from mtdata.utils.mt5 import MT5ConnectionError


def _unwrap(fn):
    current = fn
    while hasattr(current, "__wrapped__"):
        current = current.__wrapped__
    return current


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", lambda: None)


def test_attach_timezone_removes_legacy_timestamp_timezone() -> None:
    result = cf._attach_timezone(
        {"success": True, "timestamp_timezone": "America/New_York"},
        operation="forecast_generate",
    )

    assert result["timezone"] == "UTC"
    assert "timestamp_timezone" not in result


def test_normalize_forecaster_name_and_resolve_variants(monkeypatch):
    monkeypatch.setattr(
        forecast_use_cases,
        "_discover_sktime_forecasters",
        lambda: {
            "thetaforecaster": ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster"),
            "naiveforecaster": ("NaiveForecaster", "sktime.forecasting.naive.NaiveForecaster"),
        },
    )

    assert forecast_use_cases._normalize_forecaster_name("Theta-Forecaster v2") == "thetaforecasterv2"
    assert cf._resolve_sktime_forecaster("theta") == (
        "ThetaForecaster",
        "sktime.forecasting.theta.ThetaForecaster",
    )
    assert cf._resolve_sktime_forecaster("naive_fore") == (
        "NaiveForecaster",
        "sktime.forecasting.naive.NaiveForecaster",
    )
    assert cf._resolve_sktime_forecaster("") is None


def test_discover_sktime_forecasters_filters_test_and_non_forecaster_modules(monkeypatch):
    cf._clear_discover_sktime_forecasters_cache()

    class BaseForecaster:
        pass

    theta_mod = ModuleType("sktime.forecasting.theta")
    theta_mod.ThetaForecaster = type(
        "ThetaForecaster",
        (BaseForecaster,),
        {"__module__": "sktime.forecasting.theta"},
    )
    theta_mod.NotAForecaster = type(
        "NotAForecaster",
        (),
        {"__module__": "sktime.forecasting.theta"},
    )
    theta_mod._PrivateForecaster = type(
        "_PrivateForecaster",
        (BaseForecaster,),
        {"__module__": "sktime.forecasting.theta"},
    )

    fake_sktime = ModuleType("sktime")
    fake_forecasting = ModuleType("sktime.forecasting")
    fake_forecasting.__path__ = ["fake"]
    fake_base = ModuleType("sktime.forecasting.base")
    fake_base.BaseForecaster = BaseForecaster

    modules = {
        "sktime.forecasting.theta": theta_mod,
        "sktime.forecasting.tests.something": ModuleType("sktime.forecasting.tests.something"),
    }

    def fake_import_module(name):
        if name in modules:
            return modules[name]
        return importlib.import_module(name)

    monkeypatch.setitem(sys.modules, "sktime", fake_sktime)
    monkeypatch.setitem(sys.modules, "sktime.forecasting", fake_forecasting)
    monkeypatch.setitem(sys.modules, "sktime.forecasting.base", fake_base)
    monkeypatch.setattr(
        pkgutil,
        "walk_packages",
        lambda _path, _prefix: [
            SimpleNamespace(name="sktime.forecasting.tests.something"),
            SimpleNamespace(name="sktime.forecasting.theta"),
        ],
    )
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    mapping = cf._discover_sktime_forecasters()

    assert "thetaforecaster" in mapping
    assert mapping["thetaforecaster"][0] == "ThetaForecaster"
    assert "privateforecaster" not in mapping
    cf._clear_discover_sktime_forecasters_cache()


def test_forecast_generate_routes_by_library_and_validates_inputs(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    captured = {}

    def fake_forecast_impl(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "method": kwargs["method"], "params": kwargs["params"]}

    monkeypatch.setattr(cf, "_forecast_impl", fake_forecast_impl)
    monkeypatch.setattr(
        cf,
        "_resolve_sktime_forecaster",
        lambda q: ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster") if q == "theta" else None,
    )

    with pytest.raises(Exception):
        ForecastGenerateRequest(symbol="EURUSD", horizon=0)

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="statsforecast", method=""))
    assert out["error"] == "method is required for library=statsforecast"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="sktime", method="unknown"))
    assert "Unknown sktime forecaster" in out["error"]

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="", params={"x": 1}))
    assert out["ok"] is True
    assert captured["method"] == "theta"
    assert captured["params"] == {"x": 1}

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="statsforecast", method="AutoARIMA", params={}))
    assert out["ok"] is True
    assert captured["method"] == "statsforecast"
    assert captured["params"]["model_name"] == "AutoARIMA"
    assert out["method"] == "AutoARIMA"
    assert out["library"] == "statsforecast"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="statsforecast", method="sf_autoarima", params={}))
    assert out["ok"] is True
    assert captured["method"] == "statsforecast"
    assert captured["params"]["model_name"] == "AutoARIMA"
    assert out["method"] == "sf_autoarima"
    assert out["library"] == "statsforecast"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="statsforecast:autoarima", params={}))
    assert out["ok"] is True
    assert captured["method"] == "statsforecast"
    assert captured["params"]["model_name"] == "AutoARIMA"
    assert out["method"] == "autoarima"
    assert out["library"] == "statsforecast"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="sktime", method="theta", params={}))
    assert out["ok"] is True
    assert captured["method"] == "sktime"
    assert captured["params"]["estimator"] == "sktime.forecasting.theta.ThetaForecaster"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="sktime", method="skt_theta", params={}))
    assert out["ok"] is True
    assert captured["method"] == "sktime"
    assert captured["params"]["estimator"] == "sktime.forecasting.theta.ThetaForecaster"
    assert out["method"] == "skt_theta"
    assert out["library"] == "sktime"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="sktime:theta", params={}))
    assert out["ok"] is True
    assert captured["method"] == "sktime"
    assert captured["params"]["estimator"] == "sktime.forecasting.theta.ThetaForecaster"

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            library="sktime",
            method="sktime.forecasting.naive.NaiveForecaster",
            params={},
        )
    )
    assert out["ok"] is True
    assert captured["params"]["estimator"] == "sktime.forecasting.naive.NaiveForecaster"

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            library="mlforecast",
            method="sklearn.linear_model.LinearRegression",
            params={},
        )
    )
    assert out["ok"] is True
    assert captured["method"] == "mlforecast"
    assert captured["params"]["model"] == "sklearn.linear_model.LinearRegression"
    assert out["method"] == "sklearn.linear_model.LinearRegression"
    assert out["library"] == "mlforecast"

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="mlforecast:rf", params={"lags": [1, 2, 3]}))
    assert out["ok"] is True
    assert captured["method"] == "mlf_rf"
    assert captured["params"]["lags"] == [1, 2, 3]

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="native:theta", params={"x": 2}))
    assert out["ok"] is True
    assert captured["method"] == "theta"
    assert captured["params"] == {"x": 2}

    with pytest.raises(Exception):
        ForecastGenerateRequest(symbol="EURUSD", library="unsupported", method="x")


def test_forecast_generate_native_theta_adds_disambiguation_warning(monkeypatch):
    raw = _unwrap(cf.forecast_generate)

    def fake_forecast_impl(**kwargs):
        return {"ok": True, "method": kwargs["method"]}

    monkeypatch.setattr(cf, "_forecast_impl", fake_forecast_impl)

    out = raw(request=ForecastGenerateRequest(symbol="BTCUSD", timeframe="H1", library="native", method="theta", horizon=12))

    assert out["ok"] is True
    assert out["success"] is True
    assert any("StatsForecast theta is available" in str(w) for w in out.get("warnings", []))


def test_forecast_generate_native_theta_suppresses_duplicate_interval_guidance(monkeypatch):
    raw = _unwrap(cf.forecast_generate)

    def fake_forecast_impl(**kwargs):
        return {
            "ok": True,
            "method": kwargs["method"],
            "warnings": [
                "Point forecast only for method 'theta'; confidence intervals are unavailable. "
                "Use forecast_conformal_intervals for uncertainty bands."
            ],
        }

    monkeypatch.setattr(cf, "_forecast_impl", fake_forecast_impl)

    out = raw(request=ForecastGenerateRequest(symbol="BTCUSD", timeframe="H1", library="native", method="theta", horizon=12))

    assert out["ok"] is True
    assert out["success"] is True
    assert len(out["warnings"]) == 1
    assert "forecast_conformal_intervals" in out["warnings"][0]
    assert all("StatsForecast theta is available" not in str(w) for w in out["warnings"])


def test_forecast_generate_defaults_to_compact_payload(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(forecast_use_cases, "_symbol_price_currency", lambda _symbol: "USD")
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "timezone": "UTC",
            "forecast_from": {"time": "t0", "anchor": "last_observation"},
            "forecast_anchor": "next_timeframe_bar_after_last_observation",
            "forecast_step_seconds": 3600,
            "forecast_time": ["t1", "t2", "t3"],
            "forecast_price": [1.0, 1.1, 1.2],
            "forecast_epoch": [1.0, 2.0, 3.0],
            "last_price": 1.05,
            "last_price_source": "candle_close",
            "last_price_age_seconds": 3600,
            "last_price_age": "1h 0m",
            "last_price_stale": False,
            "freshness_basis": "bar_policy",
            "stale_after_seconds": 10800,
            "digits": 5,
        },
    )

    out = raw(request=ForecastGenerateRequest(symbol="BTCUSD", timeframe="H1", method="theta", horizon=3))

    assert "detail" not in out
    assert out["symbol"] == "BTCUSD"
    assert out["timeframe"] == "H1"
    assert out["price_currency"] == "USD"
    assert out["timezone"] == "UTC"
    assert "forecast_from" not in out
    assert "forecast_anchor" not in out
    assert "forecast_step_seconds" not in out
    assert out["last_price"] == 1.05
    assert "last_price_source" not in out
    assert out["last_price_stale"] is False
    assert out["freshness"] == "fresh, anchor 1h 0m ago"
    assert "last_price_age_seconds" not in out
    assert "last_price_age" not in out
    assert "freshness_basis" not in out
    assert "stale_after_seconds" not in out
    assert out["forecast_vs_last_price"] == {
        "direction": "bullish",
        "direction_basis": "horizon_end",
        "direction_threshold_pct": 0.01,
        "first_step_delta": -0.05,
        "horizon_delta": 0.15,
        "first_step_delta_pct": -4.7619,
        "horizon_delta_pct": 14.2857,
    }
    assert "forecast_time" not in out
    assert "forecast_price" not in out
    assert out["forecast"] == [
        {"time": "t1", "value": 1.0},
        {"time": "t2", "value": 1.1},
        {"time": "t3", "value": 1.2},
    ]
    assert "series" not in out
    assert "collection_kind" not in out
    assert "collection_contract_version" not in out
    assert "forecast_epoch" not in out


def test_forecast_generate_compact_summarizes_stale_anchor(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1"],
            "forecast_price": [1.0],
            "last_price": 1.05,
            "last_price_age_seconds": 101884,
            "last_price_age": "1d 4h",
            "last_price_stale": True,
            "freshness_basis": "bar_policy",
            "stale_after_seconds": 10800,
            "stale_warning": (
                "Last forecast anchor is older than the bar freshness policy; "
                "market may be closed or broker data may be stale."
            ),
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            horizon=1,
        )
    )

    assert out["last_price_stale"] is True
    assert out["freshness"] == "stale, anchor 1d 4h ago (policy: 3h 0m)"
    assert "last_price_age_seconds" not in out
    assert "last_price_age" not in out
    assert "freshness_basis" not in out
    assert "stale_after_seconds" not in out
    assert "stale_warning" not in out


def test_forecast_generate_compact_normalizes_utc_times_and_neutral_delta(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "last_observation_time": "2026-06-02 19:00",
            "forecast_start_time": "2026-06-02 20:00",
            "forecast_start_gap_bars": 1.0,
            "forecast_start_gap_note": "Static gap explanation.",
            "last_price_age_seconds": 3600,
            "last_price_stale": False,
            "denoise_applied": False,
            "forecast_time": ["2026-06-02 20:00", "2026-06-02 21:00"],
            "forecast_price": [1.00004, 1.00005],
            "last_price": 1.0,
            "digits": 5,
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            horizon=2,
        )
    )

    assert out["last_observation_time"] == "2026-06-02T19:00Z"
    assert out["timezone"] == "UTC"
    assert out["data_window"] == {
        "last_observation": "2026-06-02T19:00Z",
        "last_bar_complete": True,
        "input_bar_policy": "closed_bars_only",
        "forecast_start": "2026-06-02T20:00Z",
        "forecast_start_gap_bars": 1.0,
        "last_observation_age_seconds": 3600,
        "last_observation_stale": False,
    }
    assert "forecast_start_gap_bars" not in out
    assert "forecast_start_gap_note" not in out
    assert "last_price_stale" not in out
    assert "denoise_applied" not in out
    assert out["forecast"] == [
        {"time": "2026-06-02T20:00Z", "value": 1.00004},
        {"time": "2026-06-02T21:00Z", "value": 1.00005},
    ]
    assert out["forecast_vs_last_price"]["direction"] == "neutral"
    assert out["forecast_vs_last_price"]["direction_threshold_pct"] == 0.01


def test_forecast_backtest_request_accepts_method_alias():
    request = ForecastBacktestRequest(symbol="EURUSD", method="theta")

    assert request.methods == ["theta"]


def test_forecast_generate_compact_omits_training_period(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1"],
            "forecast_price": [1.0],
            "diagnostics": {
                "history_start_time": "2026-01-01 00:00",
                "history_end_time": "2026-01-10 00:00",
                "history_bars_used": 200,
                "target_points_used": 199,
                "lookback_bars_requested": 250,
                "lookback_bars_fetched": 240,
            },
        },
    )

    expected_training_period = {
        "start": "2026-01-01 00:00",
        "end": "2026-01-10 00:00",
        "history_bars_used": 200,
        "target_points_used": 199,
        "lookback_bars_requested": 250,
        "lookback_bars_fetched": 240,
        "note": "Forecast was fit on the historical window summarized here.",
    }

    out = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
        )
    )
    assert "training_period" not in out
    assert "diagnostics" not in out

    standard = raw(
        request=ForecastGenerateRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            detail="standard",
        )
    )
    assert standard["training_period"] == expected_training_period


def test_forecast_generate_rounds_price_outputs_to_symbol_digits(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2"],
            "forecast_price": [1.1731445723463942, 1.1731467944693543],
            "last_price": 1.17266,
            "last_price_source": "candle_close",
            "digits": 5,
        },
    )

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", method="theta", horizon=2))

    assert out["forecast_vs_last_price"] == {
        "direction": "bullish",
        "direction_basis": "horizon_end",
        "direction_threshold_pct": 0.01,
        "first_step_delta": 0.00048,
        "horizon_delta": 0.00049,
        "first_step_delta_pct": 0.0409,
        "horizon_delta_pct": 0.0418,
    }
    assert "forecast_price" not in out
    assert out["forecast"] == [
        {"time": "t1", "value": 1.17314},
        {"time": "t2", "value": 1.17315},
    ]
    assert "series" not in out


def test_forecast_generate_compact_flags_flat_theta_display(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2", "t3"],
            "forecast_price": [1.168361, 1.168362, 1.168363],
            "last_price": 1.16317,
            "last_price_source": "candle_close",
            "digits": 5,
            "params_used": {"alpha": 0.2, "trend_slope": 0.000002},
        },
    )

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", method="theta", horizon=3))

    assert "forecast_price" not in out
    assert out["forecast"] == [
        {"time": "t1", "value": 1.16836},
        {"time": "t2", "value": 1.16836},
        {"time": "t3", "value": 1.16836},
    ]
    assert "theta_signal" not in out
    assert "params_used" not in out
    assert out["path_flat"] is True
    assert out["path_range"] == 0.0
    assert out["point_forecast_mode"] == "flat_anchor"
    assert out["forecast_vs_last_price"]["direction"] == "neutral"
    assert out["forecast_vs_last_price"]["direction_basis"] == "flat_path"
    assert out["forecast_vs_last_price"]["direction_suppressed_reason"] == "flat_path"
    assert any("near-flat at displayed price precision" in item for item in out["warnings"])


def test_forecast_generate_compact_flags_one_tick_flat_path(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2", "t3", "t4", "t5"],
            "forecast_price": [1.16426, 1.16426, 1.16427, 1.16427, 1.16427],
            "last_price": 1.16505,
            "digits": 5,
            "ci_status": "unavailable",
        },
    )

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", method="theta", horizon=5))

    assert out["path_flat"] is True
    assert out["path_range"] == 0.00001
    assert out["forecast_vs_last_price"]["direction"] == "neutral"
    assert out["forecast_vs_last_price"]["direction_basis"] == "flat_path"
    assert out["forecast_vs_last_price"]["direction_suppressed_reason"] == "flat_path"
    assert any("forecast_conformal_intervals" in item for item in out["warnings"])


def test_forecast_generate_compact_marks_unavailable_ci(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1"],
            "forecast_price": [1.0],
            "ci_status": "unavailable",
            "ci_available": False,
            "ci_alpha": 0.05,
            "ci": {
                "status": "unavailable",
                "hint": "Use forecast_conformal_intervals for uncertainty bands.",
            },
            "warnings": [
                "Point forecast only for method 'theta'; confidence intervals are unavailable. "
                "Use forecast_conformal_intervals for uncertainty bands.",
                "Native theta fallback used.",
            ],
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="BTCUSD",
            timeframe="H1",
            method="theta",
            horizon=1,
        )
    )

    assert "detail" not in out
    assert "ci" not in out
    assert "ci_status" not in out
    assert "ci_available" not in out
    assert "ci_alpha" not in out
    assert out["uncertainty"] == {
        "status": "unavailable",
        "mode": "point_only",
        "recommended_tool": "forecast_conformal_intervals",
        "requested_alpha": 0.05,
    }
    assert out["warnings"] == ["Native theta fallback used."]


def test_forecast_generate_compact_nests_available_ci(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2"],
            "forecast_price": [100.0, 101.0],
            "lower_price": [99.0, 99.5],
            "upper_price": [101.0, 102.5],
            "ci_status": "available",
            "ci_alpha": 0.05,
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="BTCUSD",
            timeframe="H1",
            method="arima",
            horizon=2,
        )
    )

    assert out["uncertainty"] == {
        "status": "available",
        "mode": "interval",
        "alpha": 0.05,
        "intervals": [
            {"time": "t1", "forecast": 100.0, "low": 99.0, "high": 101.0},
            {"time": "t2", "forecast": 101.0, "low": 99.5, "high": 102.5},
        ],
        "summary": {
            "first_low": 99.0,
            "first_high": 101.0,
            "last_low": 99.5,
            "last_high": 102.5,
            "median_width": 3.0,
        },
    }
    assert "ci_status" not in out
    assert "ci_alpha" not in out
    assert "interval_summary" not in out
    assert "lower_price" not in out
    assert "upper_price" not in out
    assert "forecast_time" not in out
    assert "forecast_price" not in out
    assert "forecast" not in out


def test_forecast_generate_standard_preserves_full_arrays(monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(
        cf,
        "_forecast_impl",
        lambda **kwargs: {
            "success": True,
            "method": kwargs["method"],
            "horizon": kwargs["horizon"],
            "quantity": kwargs["quantity"],
            "forecast_time": ["t1", "t2", "t3"],
            "forecast_price": [1.0, 1.1, 1.2],
            "forecast_epoch": [1.0, 2.0, 3.0],
        },
    )

    out = raw(
        request=ForecastGenerateRequest(
            symbol="BTCUSD",
            timeframe="H1",
            method="theta",
            horizon=3,
            detail="standard",
        )
    )

    assert out["detail"] == "standard"
    assert out["forecast_epoch"] == [1.0, 2.0, 3.0]
    assert out["series"][0] == {"time": "t1", "forecast_price": 1.0}
    assert "collection_kind" not in out
    assert "collection_contract_version" not in out


def test_run_forecast_generate_logs_finish_event(caplog):
    with caplog.at_level("DEBUG", logger="mtdata.forecast.use_cases"):
        result = forecast_use_cases.run_forecast_generate(
            ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", library="native", method="theta"),
            forecast_impl=lambda **kwargs: {"ok": True, "method": kwargs["method"]},
            resolve_sktime_forecaster=lambda query: None,
        )
    assert result["ok"] is True
    assert any(
        "event=finish operation=forecast_generate success=True" in record.message
        for record in caplog.records
    )


def test_run_forecast_backtest_derives_target_from_quantity():
    captured = {}

    def fake_backtest_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True}

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", quantity="return"),
        backtest_impl=fake_backtest_impl,
    )

    assert result["success"] is True
    assert captured["quantity"] == "return"
    assert "target" not in captured


def test_run_forecast_backtest_preserves_requested_noncompact_detail():
    def fake_backtest_impl(**kwargs):
        return {"success": True, "detail": "compact", "results": {}}

    standard = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="standard"),
        backtest_impl=fake_backtest_impl,
    )
    summary = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="summary"),
        backtest_impl=fake_backtest_impl,
    )

    assert standard["detail"] == "standard"
    assert summary["detail"] == "summary"


def test_run_forecast_backtest_strips_per_anchor_details_in_compact_mode():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "request": {"detail": "compact"},
            "resolved_request": {"detail": "compact", "methods": ["theta"]},
            "units": {"returns": "return_fraction", "forecast_error": "price"},
            "results": {
                "theta": {
                    "avg_mae": 1.0,
                    "metrics": {
                        "sample_warning": "Only 1 trades. Annualized risk metrics are suppressed.",
                        "sample_notice": {
                            "code": "annualization_suppressed_low_sample",
                            "trades_observed": 1,
                            "minimum_trades": 30,
                        },
                    },
                    "details": [{"anchor": "2026-01-01 00:00", "success": True}],
                }
            },
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="compact"),
        backtest_impl=fake_backtest_impl,
    )

    assert result["success"] is True
    assert "request" not in result
    assert "resolved_request" not in result
    assert "results" not in result
    assert result["units"]["returns"] == "return_fraction"
    assert result["units"]["forecast_error"] == "price"
    assert result["ranked_methods"][0]["method"] == "theta"
    assert result["ranked_methods"][0]["details_count"] == 1
    assert "metrics" not in result["ranked_methods"][0]


def test_run_forecast_backtest_compact_keeps_kelly_metrics():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "results": {
                "theta": {
                    "success": True,
                    "avg_mae": 1.0,
                    "metrics": {
                        "win_rate": 0.5,
                        "avg_win_return": 0.03000001,
                        "avg_loss_return": -0.01500001,
                        "avg_loss_magnitude": 0.01500001,
                        "avg_win_loss_ratio": 1.999998,
                        "kelly_fraction": 0.249999,
                        "half_kelly_fraction": 0.124999,
                        "trades_observed": 4,
                    },
                    "details": [{"anchor": "2026-01-01 00:00", "success": True}],
                }
            },
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="compact"),
        backtest_impl=fake_backtest_impl,
    )

    row = result["ranked_methods"][0]
    assert row["avg_win_return"] == 0.03
    assert row["avg_loss_return"] == -0.015
    assert row["avg_loss_magnitude"] == 0.015
    assert row["avg_win_loss_ratio"] == 2.0
    assert row["kelly_fraction"] == 0.25
    assert row["half_kelly_fraction"] == 0.125


def test_run_forecast_backtest_compact_suppresses_low_sample_kelly_metrics():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "results": {
                "theta": {
                    "success": True,
                    "avg_mae": 1.0,
                    "avg_rmse": 1.2,
                    "metrics": {
                        "win_rate_pct": 50.0,
                        "avg_win_loss_ratio": 20.06,
                        "kelly_fraction": 0.4751,
                        "half_kelly_fraction": 0.2375,
                        "trades_observed": 2,
                        "sample_notice": {
                            "code": "annualization_suppressed_low_sample",
                            "trades_observed": 2,
                            "minimum_trades": 30,
                        },
                    },
                    "details": [{"anchor": "2026-01-01 00:00", "success": True}],
                }
            },
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="compact"),
        backtest_impl=fake_backtest_impl,
    )

    row = result["ranked_methods"][0]
    assert row["metrics_reliability"] == "low"
    assert row["trades_observed"] == 2
    assert "win_rate_pct" not in row
    assert "avg_win_loss_ratio" not in row
    assert "kelly_fraction" not in row


def test_run_forecast_backtest_omits_trade_metrics_when_unavailable():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "request": {"detail": "compact"},
            "results": {
                "naive": {
                    "success": True,
                    "avg_mae": 1.0,
                    "avg_rmse": 1.2,
                    "avg_directional_accuracy": 0.0,
                    "successful_tests": 3,
                    "num_tests": 3,
                    "trade_status": "flat",
                    "metrics_available": False,
                    "metrics_reason": "no_non_flat_trades",
                    "metrics": {
                        "avg_return": None,
                        "avg_return_per_trade": None,
                        "win_rate": None,
                        "win_rate_display": None,
                        "max_drawdown": None,
                        "trades_observed": 0,
                    },
                    "details": [{"position": "flat"}],
                }
            },
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="compact"),
        backtest_impl=fake_backtest_impl,
    )

    row = result["ranked_methods"][0]
    assert row["trade_status"] == "flat"
    assert row["metrics_available"] is False
    assert row["metrics_reason"] == "no_non_flat_trades"
    assert row["metrics_note"] == (
        "No active long/short trades; win_rate and drawdown need at least one trade."
    )
    assert "trades_observed" not in row
    assert "details_count" not in row
    assert "win_rate" not in row
    assert "win_rate_display" not in row
    assert "max_drawdown" not in row
    assert "avg_return" not in row
    assert "avg_return_per_trade" not in row


def test_run_forecast_backtest_handles_numpy_metrics_available_false():
    def fake_backtest_impl(**kwargs):
        return {
            "success": True,
            "results": {
                "naive": {
                    "success": True,
                    "metrics_available": np.bool_(False),
                    "metrics_reason": "no_non_flat_trades",
                    "metrics": {"win_rate": 1.0, "trades_observed": 1},
                    "details": [{"position": "flat"}],
                }
            },
        }

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(symbol="EURUSD", detail="compact"),
        backtest_impl=fake_backtest_impl,
    )

    row = result["ranked_methods"][0]
    assert row["metrics_available"] == np.bool_(False)
    assert "metrics_note" in row
    assert "win_rate" not in row
    assert "details_count" not in row


def test_run_forecast_backtest_routes_date_range_to_impl():
    captured = {}

    def fake_backtest_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True, "results": {}}

    result = forecast_use_cases.run_forecast_backtest(
        ForecastBacktestRequest(
            symbol="EURUSD",
            start="2023-01-01",
            end="2023-12-31",
            detail="full",
        ),
        backtest_impl=fake_backtest_impl,
    )

    assert result["success"] is True
    assert captured["start"] == "2023-01-01"
    assert captured["end"] == "2023-12-31"


def test_run_forecast_generate_routes_date_range_to_impl(monkeypatch):
    captured = {}

    def fake_forecast_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True, "forecast_price": [1.0, 1.1]}

    result = forecast_use_cases.run_forecast_generate(
        ForecastGenerateRequest(
            symbol="EURUSD",
            start="2023-01-01",
            end="2023-03-31",
        ),
        forecast_impl=fake_forecast_impl,
        resolve_sktime_forecaster=lambda _query: None,
    )

    assert result["success"] is True
    assert captured["start"] == "2023-01-01"
    assert captured["end"] == "2023-03-31"


def test_run_forecast_generate_routes_volatility_proxy_to_impl():
    captured = {}

    def fake_forecast_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True, "method": kwargs["method"], "proxy": kwargs.get("proxy")}

    result = forecast_use_cases.run_forecast_generate(
        ForecastGenerateRequest(
            symbol="EURUSD",
            method="theta",
            quantity="volatility",
            proxy="abs_return",
        ),
        forecast_impl=fake_forecast_impl,
        resolve_sktime_forecaster=lambda _query: None,
    )

    assert result["success"] is True
    assert captured["quantity"] == "volatility"
    assert captured["proxy"] == "abs_return"


def test_run_forecast_generate_defaults_general_volatility_proxy():
    captured = {}

    def fake_forecast_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True, "method": kwargs["method"], "proxy": kwargs.get("proxy")}

    result = forecast_use_cases.run_forecast_generate(
        ForecastGenerateRequest(
            symbol="EURUSD",
            method="theta",
            quantity="volatility",
        ),
        forecast_impl=fake_forecast_impl,
        resolve_sktime_forecaster=lambda _query: None,
    )

    assert result["success"] is True
    assert captured["proxy"] == "squared_return"
    assert any("defaulted proxy=squared_return" in item for item in result["warnings"])


def test_run_forecast_volatility_routes_date_range_to_impl():
    captured = {}

    def fake_volatility_impl(**kwargs):
        captured.update(kwargs)
        return {"success": True, "sigma_bar_return": 0.01}

    result = forecast_use_cases.run_forecast_volatility_estimate(
        ForecastVolatilityEstimateRequest(
            symbol="EURUSD",
            start="2023-01-01",
            end="2023-03-31",
        ),
        forecast_volatility_impl=fake_volatility_impl,
    )

    assert result["success"] is True
    assert captured["start"] == "2023-01-01"
    assert captured["end"] == "2023-03-31"


def test_forecast_generate_converts_typed_forecast_errors(monkeypatch):
    raw = _unwrap(cf.forecast_generate)

    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: (_ for _ in ()).throw(ForecastError("engine exploded")))

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out["success"] is False
    assert out["error"] == "engine exploded"
    assert out["error_code"] == "forecast_generate_error"
    assert out["operation"] == "forecast_generate"
    assert isinstance(out.get("request_id"), str)


def test_forecast_generate_logs_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: {"success": True, "method": kwargs["method"]})

    with caplog.at_level(logging.DEBUG, logger=cf.logger.name):
        out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out["success"] is True
    assert any(
        "event=finish operation=forecast_generate success=True" in record.message
        for record in caplog.records
    )


def test_forecast_generate_wrapper_emits_single_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_generate)
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: {"success": True, "method": kwargs["method"]})

    with caplog.at_level(logging.DEBUG):
        out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out["success"] is True
    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=forecast_generate success=True" in record.message
    ]
    assert len(finish_records) == 1
    assert finish_records[0].name == cf.logger.name


def test_forecast_generate_returns_connection_error_payload(monkeypatch):
    raw = _unwrap(cf.forecast_generate)

    def fail_connection():
        raise MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running.")

    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", fail_connection)
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: pytest.fail("forecast implementation should not run"))

    out = raw(request=ForecastGenerateRequest(symbol="EURUSD", library="native", method="theta"))

    assert out["success"] is False
    assert out["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    assert out["error_code"] == "mt5_connection_error"
    assert out["operation"] == "mt5_ensure_connection"
    assert isinstance(out.get("request_id"), str)


def test_forecast_tune_genetic_logs_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_tune_genetic)
    monkeypatch.setattr(cf, "run_forecast_tune_genetic", lambda request, genetic_search_impl: {"success": True, "best": {}})

    with caplog.at_level(logging.DEBUG, logger=cf.logger.name):
        out = raw(request=ForecastTuneGeneticRequest(symbol="EURUSD", method="theta"))

    assert out["success"] is True
    assert any(
        "event=finish operation=forecast_tune_genetic success=True" in record.message
        for record in caplog.records
    )


def test_forecast_tune_detail_compacts_history_tail():
    def fake_genetic(**kwargs):
        return {
            "success": True,
            "history_count": 2,
            "history_tail": [{"score": 1.0}, {"score": 0.9}],
        }

    compact = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(symbol="EURUSD", method="theta"),
        genetic_search_impl=fake_genetic,
    )
    assert compact["detail"] == "compact"
    assert "history_tail" not in compact
    assert compact["history_tail_count"] == 2
    assert compact["history_count"] == 2

    full = forecast_use_cases.run_forecast_tune_genetic(
        ForecastTuneGeneticRequest(symbol="EURUSD", method="theta", detail="full"),
        genetic_search_impl=fake_genetic,
    )
    assert full["detail"] == "full"
    assert full["history_tail"] == [{"score": 1.0}, {"score": 0.9}]
    assert "history_tail_count" not in full


def test_forecast_tune_optuna_and_optimize_hints_accept_detail():
    def fake_optuna(**kwargs):
        return {"success": True, "history_count": 1, "history_tail": [{"score": 1.0}]}

    def fake_hints(**kwargs):
        return {"success": True, "history_count": 1, "history_tail": [{"best_score": 0.5}]}

    optuna = forecast_use_cases.run_forecast_tune_optuna(
        ForecastTuneOptunaRequest(symbol="EURUSD", method="theta", detail="standard"),
        optuna_search_impl=fake_optuna,
    )
    assert optuna["detail"] == "standard"
    assert "history_tail" not in optuna
    assert optuna["history_tail_count"] == 1

    hints = forecast_use_cases.run_forecast_optimize_hints(
        ForecastOptimizeHintsRequest(symbol="EURUSD", timeframe="H1", detail="summary"),
        optimize_hints_impl=fake_hints,
    )
    assert hints["detail"] == "summary"
    assert "history_tail" not in hints
    assert hints["history_tail_count"] == 1


def test_forecast_barrier_optimize_logs_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_barrier_optimize)
    monkeypatch.setattr(cf, "run_forecast_barrier_optimize", lambda request, parse_kv_or_json, barrier_optimize_impl: {"success": True, "best": {}})

    with caplog.at_level(logging.DEBUG, logger=cf.logger.name):
        out = raw(request=ForecastBarrierOptimizeRequest(symbol="EURUSD"))

    assert out["success"] is True
    assert any(
        "event=finish operation=forecast_barrier_optimize success=True" in record.message
        for record in caplog.records
    )


def test_forecast_barrier_optimize_request_defaults_to_summary_output():
    request = ForecastBarrierOptimizeRequest(symbol="EURUSD")
    assert request.search_profile == "medium"
    assert len(ForecastBarrierOptimizeRequest.model_fields) == 15
    assert "output_mode" not in ForecastBarrierOptimizeRequest.model_fields
    assert "tp_min" not in ForecastBarrierOptimizeRequest.model_fields
    assert "statistical_robustness" not in ForecastBarrierOptimizeRequest.model_fields
    assert "format" not in ForecastBarrierOptimizeRequest.model_fields


def test_forecast_barrier_requests_normalize_known_direction_aliases_only():
    assert ForecastBarrierProbRequest(symbol="EURUSD", direction="buy").direction == "long"
    assert ForecastBarrierOptimizeRequest(symbol="EURUSD", direction="DOWN").direction == "short"
    assert ForecastBarrierProbRequest(symbol="EURUSD", direction="weird").direction == "weird"


def test_forecast_barrier_optimize_request_rejects_removed_output_field():
    with pytest.raises(ValidationError, match="output was removed; use extras"):
        ForecastBarrierOptimizeRequest(symbol="EURUSD", output="summary")


def test_forecast_barrier_optimize_request_rejects_removed_format_field():
    with pytest.raises(ValidationError, match="format was removed; use json"):
        ForecastBarrierOptimizeRequest(symbol="EURUSD", format="full")


def test_forecast_barrier_optimize_request_rejects_top_level_advanced_params():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ForecastBarrierOptimizeRequest(symbol="EURUSD", tp_min=0.5)


def test_barrier_prob_price_rounding_uses_numeric_precision_without_padding():
    out = forecast_use_cases._round_barrier_prob_payload(
        {
            "success": True,
            "price_precision": 5,
            "last_price": "1.16976000",
            "tp_price": "1.18170000",
            "sl_price": "1.16415000",
            "prob_tp_first": 0.3333333333333,
        }
    )

    assert out["last_price"] == 1.16976
    assert out["tp_price"] == 1.1817
    assert out["sl_price"] == 1.16415
    assert isinstance(out["tp_price"], float)
    assert out["prob_tp_first"] == 0.333333


def test_barrier_symbol_price_precision_reads_cached_digits(monkeypatch):
    monkeypatch.setattr(
        "mtdata.utils.mt5.get_symbol_info_cached",
        lambda _symbol: SimpleNamespace(digits=5),
    )

    assert barriers_shared._symbol_price_precision("EURUSD") == 5


def test_forecast_list_library_models_and_list_methods(monkeypatch):
    stats_mod = ModuleType("statsforecast")
    models_mod = ModuleType("statsforecast.models")
    models_mod.__name__ = "statsforecast.models"
    models_mod.AutoARIMA = type(
        "AutoARIMA",
        (),
        {
            "__module__": "statsforecast.models",
            "fit": lambda self: None,
        },
    )
    models_mod.OtherHelper = 123
    stats_mod.models = models_mod

    monkeypatch.setitem(sys.modules, "statsforecast", stats_mod)
    monkeypatch.setattr(
        cf,
        "_discover_sktime_forecasters",
        lambda: {
            "thetaforecaster": ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster"),
            "naiveforecaster": ("NaiveForecaster", "sktime.forecasting.naive.NaiveForecaster"),
        },
    )

    raw_list_models = _unwrap(cf.forecast_list_library_models)
    out_native = raw_list_models("native", detail="full")
    assert out_native["library"] == "native"
    assert isinstance(out_native["models"], list)
    assert out_native["models"][0]["method"]
    assert out_native["models"][0]["available"] is True
    assert isinstance(out_native["capabilities"], list)
    assert out_native["capabilities"][0]["execution"]["library"] == "native"

    out_stats = raw_list_models("statsforecast", detail="full")
    assert out_stats["library"] == "statsforecast"
    assert any(
        row["method"] == "AutoARIMA" and row["model"] == "AutoARIMA"
        for row in out_stats["models"]
    )
    assert isinstance(out_stats["usage"], list)
    assert out_stats["capabilities"][0]["execution"]["library"] == "statsforecast"
    assert out_stats["capabilities"][0]["selector"]["key"] == "model_name"

    out_sktime = raw_list_models("sktime", detail="full")
    assert [row["model"] for row in out_sktime["models"]] == [
        "NaiveForecaster",
        "ThetaForecaster",
    ]
    assert out_sktime["capabilities"][0]["selector"]["key"] == "estimator"
    assert out_sktime["capabilities"][0]["execution"]["method"] == "sktime"

    out_ml = raw_list_models("mlforecast", detail="full")
    assert out_ml["library"] == "mlforecast"
    assert out_ml["models"][0]["method"] == "mlforecast"
    assert "note" in out_ml
    assert out_ml["capabilities"][0]["selector"]["key"] == "model"

    out_bad = raw_list_models("other")
    assert "Unsupported library" in out_bad["error"]

    class FalseLike:
        def __bool__(self):
            return False

    monkeypatch.setattr(
        cf,
        "_get_library_forecast_capabilities",
        lambda lib, **_kwargs: [
            {
                "method": "theta",
                "namespace": "native",
                "available": True,
                "execution": {"library": "native", "method": "theta"},
            },
            {
                "method": "unavailable_model",
                "namespace": "native",
                "available": FalseLike(),
                "execution": {"library": "native", "method": "unavailable_model"},
            },
        ]
        if lib == "native"
        else [],
    )
    out_native_available = raw_list_models("native")
    assert out_native_available["models"] == [
        {"method": "theta", "available": True}
    ]
    assert out_native_available["total"] == 2
    assert out_native_available["total_filtered"] == 1
    assert out_native_available["available"] == 1
    assert out_native_available["filters"]["show_unavailable"] is False
    out_native_all = raw_list_models("native", show_unavailable=True)
    assert out_native_all["models"] == [
        {"method": "theta", "available": True},
        {"method": "unavailable_model", "available": False},
    ]
    assert out_native_all["total"] == 2
    assert out_native_all["total_filtered"] == 2
    assert out_native_all["available"] == 1
    assert out_native_all["filters"]["show_unavailable"] is True

    monkeypatch.setattr(
        cf,
        "_get_registered_forecast_capabilities",
        lambda: [
            {
                "method": "theta",
                "supports": {"ci": True},
            },
            {
                "method": "mlf_rf",
                "supports": {"ci": False},
            },
            {
                "method": "sf_autoarima",
                "namespace": "statsforecast",
                "supports": {"ci": True},
            },
            {
                "method": "sf_theta",
                "namespace": "statsforecast",
                "supports": {"ci": True},
            },
            {
                "method": "sf_ets",
                "namespace": "statsforecast",
                "supports": {"ci": True},
            },
            {
                "method": "sf_naive",
                "namespace": "statsforecast",
                "supports": {"ci": True},
            },
        ],
    )

    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 2,
            "categories": {"classical": ["theta"], "ml": ["mlf_rf"]},
            "methods": [
                {
                    "method": "theta",
                    "available": True,
                    "description": "Theta model.",
                    "params": [{"name": "window"}],
                    "requires": [],
                    "supports_training": False,
                },
                {
                    "method": "mlf_rf",
                    "available": False,
                    "description": "RF model.",
                    "params": [{"name": "n_estimators"}, {"name": "max_depth"}],
                    "requires": ["mlforecast", "sklearn"],
                    "supports_training": False,
                },
            ],
        },
    )
    compact = _unwrap(cf.forecast_list_methods)()
    assert "detail" not in compact
    assert compact["total"] == 2
    assert compact["available"] == 1
    assert compact["unavailable"] == 0
    assert compact["methods"][0]["method"] == "theta"
    assert "category_summary" not in compact
    assert "categories" not in compact
    assert "params_count" not in compact["methods"][0]
    assert "description" not in compact["methods"][0]
    assert compact["methods"][0]["supports_ci"] is True
    assert compact["methods"][0]["supports_training"] is False
    assert "namespace" not in compact["methods"][0]
    assert "method_id" not in compact["methods"][0]
    assert "concept" not in compact["methods"][0]
    assert "capability_id" not in compact["methods"][0]
    assert "adapter_method" not in compact["methods"][0]
    assert "selector" not in compact["methods"][0]
    assert "params" not in compact["methods"][0]
    assert all("requires" not in row for row in compact["methods"])
    assert compact["methods_shown"] == 1
    assert compact["methods_hidden"] == 0
    assert "filters" not in compact
    assert "barrier_methods" not in compact
    assert "note" not in compact
    assert "volatility_methods" not in compact

    standard = _unwrap(cf.forecast_list_methods)(detail="standard")
    assert standard["detail"] == "standard"
    assert standard["methods"][0]["description"] == "Theta model."
    assert standard["methods"][0]["params_count"] == 1
    assert "volatility_methods" in standard

    compact_all = _unwrap(cf.forecast_list_methods)(show_unavailable=True)
    unavailable_method = next(row for row in compact_all["methods"] if row["available"] is False)
    assert unavailable_method["unavailable_reason"] == "Requires: mlforecast, sklearn"

    full = _unwrap(cf.forecast_list_methods)(detail="full", show_unavailable=True)
    assert full["detail"] == "full"
    assert full["total"] == 2
    assert full["total_filtered"] == 2
    assert full["methods_shown"] == 2
    assert full["methods_hidden"] == 0
    assert isinstance(full.get("methods"), list)
    assert "params" in full["methods"][0]
    assert full["methods"][0]["params"] == [{"name": "window"}]
    assert "method_id" not in full["methods"][0]
    assert "capability_id" not in full["methods"][0]
    assert "concept" not in full["methods"][0]
    assert "adapter_method" not in full["methods"][0]
    assert "execution" not in full["methods"][0]
    assert "selector" not in full["methods"][0]
    assert full["methods"][0]["supports_ci"] is True
    assert full["methods"][0]["supports_training"] is False
    assert full["methods"][1]["supports_ci"] is False
    assert full["methods"][1]["supports_training"] is False
    assert full["methods"][1]["library"] == "native"
    assert full["barrier_methods"]["optimizer_only_methods"] == ["ensemble"]

    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 1,
            "categories": {"classical": ["naive"]},
            "methods": [
                {
                    "method": "naive",
                    "available": True,
                    "description": "naive",
                    "params": [],
                    "requires": [],
                },
            ],
        },
    )
    compact_repeated_description = _unwrap(cf.forecast_list_methods)()
    assert "description" not in compact_repeated_description["methods"][0]

    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 5,
            "categories": {
                "classical": ["theta"],
                "statsforecast": ["sf_autoarima", "sf_theta", "sf_ets", "sf_naive"],
            },
            "methods": [
                {"method": "theta", "available": True, "description": "Theta.", "params": [], "requires": []},
                {"method": "sf_autoarima", "available": True, "description": "A", "params": [], "requires": []},
                {"method": "sf_theta", "available": True, "description": "B", "params": [], "requires": []},
                {"method": "sf_ets", "available": False, "description": "C", "params": [], "requires": []},
                {"method": "sf_naive", "available": True, "description": "D", "params": [], "requires": []},
            ],
        },
    )
    grouped = _unwrap(cf.forecast_list_methods)()
    params = signature(_unwrap(cf.forecast_list_methods)).parameters
    assert "search_term" in params
    assert "search" not in params
    assert "category" in params
    assert "library" in params
    assert "supports_ci" in params
    assert "show_unavailable" in params
    assert "all" not in params
    sf_rows = [r for r in grouped["methods"] if r.get("category") == "statsforecast"]
    assert len(sf_rows) == 3
    if sf_rows:
        assert all(str(r.get("namespace")) == "statsforecast" for r in sf_rows)
    assert grouped["methods_hidden"] == 0
    filtered = _unwrap(cf.forecast_list_methods)(search_term="theta", limit=1)
    assert "filters" not in filtered
    assert len(filtered["methods"]) == 1
    assert filtered["methods_hidden"] >= 1
    assert "theta" in str(filtered["methods"][0]["method"]).lower()
    sf_only = _unwrap(cf.forecast_list_methods)(library="statsforecast")
    assert "filters" not in sf_only
    assert all(
        row.get("category") == "statsforecast" for row in sf_only["methods"]
    )
    category_only = _unwrap(cf.forecast_list_methods)(category="statsforecast")
    assert "filters" not in category_only
    assert all(row.get("category") == "statsforecast" for row in category_only["methods"])
    ci_only = _unwrap(cf.forecast_list_methods)(supports_ci=True)
    assert "filters" not in ci_only
    assert ci_only["methods"]
    assert all(row.get("supports_ci") is True for row in ci_only["methods"])
    no_ci = _unwrap(cf.forecast_list_methods)(supports_ci=False, show_unavailable=True)
    assert "filters" not in no_ci
    assert all(row.get("supports_ci") is False for row in no_ci["methods"])
    with_unavailable = _unwrap(cf.forecast_list_methods)(show_unavailable=True)
    assert with_unavailable["unavailable"] == 1
    assert any(row["available"] is False for row in with_unavailable["methods"])
    unavailable_row = next(row for row in with_unavailable["methods"] if row["available"] is False)
    assert unavailable_row["unavailable_reason"] == "Unavailable in the current environment."

    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 25,
            "categories": {"classical": [f"m{i:02d}" for i in range(25)]},
            "methods": [
                {
                    "method": f"m{i:02d}",
                    "available": True,
                    "description": f"Method {i:02d}.",
                    "params": [],
                    "requires": [],
                }
                for i in range(25)
            ],
        },
    )
    default_out = _unwrap(cf.forecast_list_methods)()
    assert "filters" not in default_out
    assert default_out["methods_shown"] == 25
    assert default_out["methods_hidden"] == 0
    assert "has_more" not in default_out
    assert "truncation_reason" not in default_out

    page = _unwrap(cf.forecast_list_methods)(limit=5, offset=5)
    assert [row["method"] for row in page["methods"]] == [
        "m05",
        "m06",
        "m07",
        "m08",
        "m09",
    ]
    assert page["methods_before"] == 5
    assert page["methods_hidden"] == 15
    assert page["offset"] == 5
    assert page["has_more"] is True
    assert page["truncation_reason"] == (
        "Limit 5 at offset 5; set offset=10 for more filtered methods."
    )

    filtered_uncapped = _unwrap(cf.forecast_list_methods)(category="classical")
    assert "filters" not in filtered_uncapped
    assert filtered_uncapped["methods_shown"] == 25
    assert filtered_uncapped["methods_hidden"] == 0
    assert "truncation_reason" not in filtered_uncapped

    monkeypatch.setattr(cf, "_get_forecast_methods_data", lambda: {"methods": [1]})
    assert _unwrap(cf.forecast_list_methods)() == {"methods": [1]}
    monkeypatch.setattr(cf, "_get_forecast_methods_data", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert "Error listing forecast methods" in _unwrap(cf.forecast_list_methods)()["error"]


def test_forecast_list_methods_standard_exposes_ci_method(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {
            "total": 3,
            "categories": {"ets_arima": ["arima"], "monte_carlo": ["mc_gbm"]},
            "methods": [
                {
                    "method": "arima",
                    "available": True,
                    "category": "ets_arima",
                    "description": "ARIMA model.",
                    "params": [],
                    "requires": [],
                    "supports": {"ci": True},
                },
                {
                    "method": "mc_gbm",
                    "available": True,
                    "category": "monte_carlo",
                    "description": "Monte Carlo GBM.",
                    "params": [],
                    "requires": [],
                    "supports": {"ci": True},
                },
                {
                    "method": "naive",
                    "available": True,
                    "category": "classical",
                    "description": "Naive model.",
                    "params": [],
                    "requires": [],
                    "supports": {"ci": False},
                },
            ],
        },
    )

    compact = _unwrap(cf.forecast_list_methods)(supports_ci=True, profile="all")
    assert all("ci_method" not in row for row in compact["methods"])

    standard = _unwrap(cf.forecast_list_methods)(
        detail="standard",
        supports_ci=True,
        profile="all",
    )
    methods = {row["method"]: row for row in standard["methods"]}

    assert set(methods) == {"arima", "mc_gbm"}
    assert methods["arima"]["ci_method"] == "statsmodels_prediction_interval"
    assert methods["mc_gbm"]["ci_method"] == "simulation_quantile"


def test_forecast_list_library_models_defaults_to_compact_page(monkeypatch):
    rows = [
        {
            "method": f"model_{idx}",
            "namespace": "native",
            "available": True,
            "execution": {"library": "native", "method": f"model_{idx}"},
        }
        for idx in range(25)
    ]
    monkeypatch.setattr(
        cf,
        "_get_library_forecast_capabilities",
        lambda lib, **_kwargs: rows if lib == "native" else [],
    )

    compact_page = cf._forecast_list_library_models_impl("native")
    assert "models_shown" not in compact_page
    assert compact_page["total_filtered"] == 25
    assert compact_page["has_more"] is True
    assert compact_page["filters"]["limit"] == 20

    full_page = cf._forecast_list_library_models_impl("native", detail="full")
    assert full_page["models_shown"] == 25
    assert full_page["filters"]["limit"] is None


def test_forecast_list_library_models_compact_deduplicates_model_rows(monkeypatch):
    rows = [
        {
            "method": "ARIMA",
            "display_name": "ARIMA",
            "available": True,
            "execution": {"library": "statsforecast"},
        },
        {
            "method": "theta",
            "display_name": "ThetaForecaster",
            "available": True,
            "execution": {"library": "sktime"},
        },
    ]
    monkeypatch.setattr(
        cf,
        "_get_library_forecast_capabilities",
        lambda *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(cf, "import_module", lambda _name: ModuleType("statsforecast"))

    compact = cf._forecast_list_library_models_impl(
        "statsforecast",
        show_unavailable=True,
    )

    assert compact["models"] == [
        {"method": "ARIMA", "available": True},
        {
            "method": "theta",
            "available": True,
            "model": "ThetaForecaster",
            "library": "sktime",
        },
    ]
    assert "unavailable" not in compact
    assert "unavailable_hidden" not in compact
    assert "models_shown" not in compact


def test_forecast_list_library_models_reports_missing_statsforecast(monkeypatch):
    raw_list_models = _unwrap(cf.forecast_list_library_models)

    monkeypatch.setattr(
        cf,
        "_get_library_forecast_capabilities",
        lambda *_args, **_kwargs: [],
    )

    def fake_import_module(name):
        if name == "statsforecast":
            raise ModuleNotFoundError("No module named 'statsforecast'")
        return ModuleType(name)

    monkeypatch.setattr(cf, "import_module", fake_import_module)

    out = raw_list_models("statsforecast")

    assert out == {
        "library": "statsforecast",
        "error": "statsforecast import failed: No module named 'statsforecast'",
    }


def test_forecast_list_methods_standard_describes_builtin_methods():
    standard = _unwrap(cf.forecast_list_methods)(
        detail="standard",
        show_unavailable=True,
    )
    missing = [
        row["method"]
        for row in standard["methods"]
        if not row.get("description")
    ]

    assert missing == []


def test_forecast_list_methods_uses_shared_snapshot(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_snapshot",
        lambda: {
            "data": {
                "total": 1,
                "categories": {"statsforecast": ["sf_theta"]},
                "methods": [{"method": "sf_theta", "available": True}],
            },
            "method_to_category": {"sf_theta": "statsforecast"},
            "methods_valid": True,
            "methods": [
                {
                    "method": "sf_theta",
                    "available": True,
                    "category": "statsforecast",
                    "namespace": "statsforecast",
                    "description": "StatsForecast theta.",
                    "params": [{"name": "window"}],
                    "supports": {"ci": True},
                    "supports_training": True,
                    "training_category": "moderate",
                    "method_id": "statsforecast:theta",
                    "capability_id": "statsforecast:theta",
                    "adapter_method": "statsforecast",
                    "selector": {"mode": "class_name", "key": "model_name", "value": "Theta"},
                    "execution": {
                        "library": "statsforecast",
                        "method": "statsforecast",
                        "params": {"model_name": "Theta"},
                    },
                }
            ],
        },
    )

    compact = _unwrap(cf.forecast_list_methods)(profile="all")
    standard = _unwrap(cf.forecast_list_methods)(detail="standard", profile="all")
    full = _unwrap(cf.forecast_list_methods)(detail="full", profile="all")

    assert "namespace" not in compact["methods"][0]
    assert compact["methods"][0]["supports_ci"] is True
    assert compact["methods"][0]["supports_training"] is True
    assert standard["methods"][0]["namespace"] == "statsforecast"
    assert standard["methods"][0]["description"] == "StatsForecast theta."
    assert full["methods"][0]["method_id"] == "statsforecast:theta"
    assert full["methods"][0]["training_category"] == "moderate"
    assert full["methods"][0]["selector"]["key"] == "model_name"
    assert full["methods"][0]["execution"]["method"] == "statsforecast"


def test_forecast_list_library_models_derives_pretrained_models_from_capabilities(monkeypatch):
    raw_list_models = _unwrap(cf.forecast_list_library_models)
    pretrained_caps = [
        {
            "method": "custom_pretrained",
            "requires": ["pkg-a", "pkg-b"],
            "params": [{"name": "model_name", "type": "str"}],
            "notes": "registry-backed note",
        }
    ]

    def fake_get_library_capabilities(library, **kwargs):
        if library == "pretrained":
            return pretrained_caps
        return []

    monkeypatch.setattr(cf, "_get_library_forecast_capabilities", fake_get_library_capabilities)

    out = raw_list_models("pretrained", detail="full")

    assert out["library"] == "pretrained"
    assert out["capabilities"] == pretrained_caps
    assert out["models"] == [
        {
            "method": "custom_pretrained",
            "available": True,
        }
    ]


def test_forecast_list_methods_does_not_require_mt5_connection(monkeypatch):
    def fail_connection():
        raise MT5ConnectionError("should not be called")

    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", fail_connection)
    monkeypatch.setattr(
        cf,
        "_get_forecast_methods_data",
        lambda: {"total": 1, "categories": {}, "methods": [{"method": "theta", "available": True}]},
    )

    out = _unwrap(cf.forecast_list_methods)()

    assert out["methods"][0]["method"] == "theta"


def test_registered_forecast_capabilities_are_cached(monkeypatch):
    calls = {"count": 0}

    class _FakeCapabilitiesModule:
        @staticmethod
        def get_registered_capabilities():
            calls["count"] += 1
            return [{"method": "theta"}]

    cf._get_registered_forecast_capabilities.cache_clear()
    monkeypatch.setattr(cf, "_forecast_capabilities_module", lambda: _FakeCapabilitiesModule())

    assert cf._get_registered_forecast_capabilities() == [{"method": "theta"}]
    assert cf._get_registered_forecast_capabilities() == [{"method": "theta"}]
    assert calls["count"] == 1

    cf._get_registered_forecast_capabilities.cache_clear()


def test_forecast_list_library_models_logs_finish_event(caplog):
    raw_list_models = _unwrap(cf.forecast_list_library_models)

    with caplog.at_level(logging.DEBUG, logger=cf.logger.name):
        out = raw_list_models("mlforecast")

    assert out["library"] == "mlforecast"
    assert any(
        "event=finish operation=forecast_list_library_models success=True" in record.message
        for record in caplog.records
    )


def test_forecast_conformal_intervals_success_and_errors(monkeypatch):
    raw = _unwrap(cf.forecast_conformal_intervals)

    monkeypatch.setattr(
        cf,
        "_forecast_backtest_impl",
        lambda **kwargs: {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0, 11.0], "actual": [9.0, 12.0]},
                        {"forecast": [13.0, 14.0], "actual": [12.0, 15.0]},
                    ]
                }
            }
        },
    )
    def fake_forecast_impl(symbol, timeframe, method, horizon, params=None, denoise=None):
        return {"forecast_price": [100.0, 101.0]}

    monkeypatch.setattr(cf, "_forecast_impl", fake_forecast_impl)

    out = raw(
        request=ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=2,
            ci_alpha=0.1,
            steps=2,
        )
    )

    assert out["ci_alpha"] == 0.1
    assert out["confidence_level"] == 0.9
    assert out["ci_status"] == "available"
    assert out["ci_available"] is True
    assert out["detail"] == "compact"
    assert out["conformal"]["ci_alpha"] == 0.1
    assert out["conformal"]["empirical_coverage"] == 1.0
    assert out["conformal"]["min_calibration_points"] == 2
    assert len(out["lower_price"]) == 2
    assert len(out["upper_price"]) == 2
    assert out["lower_price"][0] <= 100.0 <= out["upper_price"][0]

    monkeypatch.setattr(cf, "_forecast_backtest_impl", lambda **kwargs: {"error": "backtest failed"})
    assert raw(request=ForecastConformalIntervalsRequest(symbol="EURUSD", method="theta", horizon=2))["error"] == "backtest failed"

    monkeypatch.setattr(cf, "_forecast_backtest_impl", lambda **kwargs: {"results": {"theta": {"details": []}}})
    assert "Conformal calibration failed" in raw(
        request=ForecastConformalIntervalsRequest(symbol="EURUSD", method="theta", horizon=2)
    )["error"]

    monkeypatch.setattr(cf, "_forecast_backtest_impl", lambda **kwargs: {"results": {"theta": {"details": [{}]}}})
    monkeypatch.setattr(cf, "_forecast_impl", lambda **kwargs: (_ for _ in ()).throw(ForecastError("engine exploded")))
    out = raw(request=ForecastConformalIntervalsRequest(symbol="EURUSD", method="theta", horizon=2))
    assert out["success"] is False
    assert out["error"] == "engine exploded"
    assert out["error_code"] == "forecast_conformal_intervals_error"
    assert out["operation"] == "forecast_conformal_intervals"
    assert isinstance(out.get("request_id"), str)


def test_forecast_conformal_intervals_request_defaults_and_spacing_validation():
    request = ForecastConformalIntervalsRequest(symbol="EURUSD")

    assert request.horizon == 12
    assert request.steps == 50
    assert request.spacing == 20
    assert request.detail == "compact"

    with pytest.raises(ValidationError, match="spacing must be greater than or equal to horizon when steps > 1"):
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            horizon=12,
            steps=2,
            spacing=10,
        )


def test_run_forecast_conformal_intervals_routes_method_params_consistently():
    captured = {}

    def fake_backtest(**kwargs):
        captured["backtest"] = kwargs
        return {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0], "actual": [9.0]},
                    ]
                }
            }
        }

    def fake_forecast(**kwargs):
        captured["forecast"] = kwargs
        return {"forecast_price": [100.0]}

    result = forecast_use_cases.run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=1,
            spacing=1,
            params={"seasonality": 24},
            detail="full",
        ),
        backtest_impl=fake_backtest,
        forecast_impl=fake_forecast,
    )

    assert captured["backtest"]["params_per_method"] == {"theta": {"seasonality": 24}}
    assert "params" not in captured["backtest"]
    assert captured["forecast"]["params"] == {"seasonality": 24}
    assert "detail" not in captured["forecast"]
    assert result["detail"] == "full"


def test_run_forecast_conformal_intervals_uses_finite_sample_quantile():
    result = forecast_use_cases.run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=3,
            spacing=1,
            ci_alpha=0.25,
            detail="full",
        ),
        backtest_impl=lambda **kwargs: {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0], "actual": [9.0]},
                        {"forecast": [10.0], "actual": [8.0]},
                        {"forecast": [10.0], "actual": [7.0]},
                    ]
                }
            }
        },
        forecast_impl=lambda **kwargs: {"forecast_price": [100.0]},
    )

    assert result["conformal"]["per_step_q"] == [3.0]
    assert result["conformal"]["empirical_coverage"] == pytest.approx(2.0 / 3.0)
    assert result["conformal"]["empirical_coverage_per_step"] == [
        pytest.approx(2.0 / 3.0)
    ]
    assert result["conformal"]["calibration_points_per_step"] == [3]
    assert result["lower_price"] == [97.0]
    assert result["upper_price"] == [103.0]
    assert result["ci_status"] == "available"


def test_run_forecast_conformal_intervals_compact_omits_technical_metadata():
    result = forecast_use_cases.run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=1,
            spacing=1,
            ci_alpha=0.1,
        ),
        backtest_impl=lambda **kwargs: {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0], "actual": [9.0]},
                    ]
                }
            }
        },
        forecast_impl=lambda **kwargs: {
            "forecast_time": ["2026-05-29 21:00"],
            "forecast_price": [100.123456789],
            "forecast_epoch": [1780088400.0],
            "forecast_anchor": "next_timeframe_bar_after_last_observation",
            "forecast_step_seconds": 3600,
            "forecast": [{"time": "2026-05-29 21:00", "value": 100.123456789}],
            "params_used": {"alpha": 0.2, "trend_slope": -0.000012493247702752267},
            "price_precision": 5,
        },
    )

    assert result["detail"] == "compact"
    assert result["forecast_time"] == ["2026-05-29 21:00"]
    assert result["forecast_price"] == [100.12346]
    assert result["lower_price"] == [99.12346]
    assert result["upper_price"] == [101.12346]
    assert result["conformal"] == {
        "ci_alpha": 0.1,
        "calibration_steps": 1,
        "calibration_spacing": 1,
        "coverage_target": 0.9,
        "coverage_evaluation": "leave_one_out_calibration_residuals",
        "min_calibration_points": 1,
    }
    assert "per_step_q" not in result["conformal"]
    for key in (
        "forecast_epoch",
        "forecast_anchor",
        "forecast_step_seconds",
        "forecast",
        "params_used",
    ):
        assert key not in result


def test_forecast_conformal_intervals_compact_marks_flat_point_forecast():
    out = forecast_use_cases._apply_conformal_intervals_detail(
        {
            "success": True,
            "method": "theta",
            "horizon": 2,
            "last_observation_time": "2026-06-02 19:00",
            "forecast_time": ["2026-06-02 20:00", "2026-06-02 21:00"],
            "forecast_price": [1.23456, 1.23456],
            "lower_price": [1.23, 1.23],
            "upper_price": [1.24, 1.24],
            "digits": 5,
            "ci_alpha": 0.1,
        },
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            horizon=2,
            detail="compact",
        ),
    )

    assert out["last_observation_time"] == "2026-06-02T19:00Z"
    assert out["forecast"] == [
        {"time": "2026-06-02T20:00Z", "value": 1.23456, "lower": 1.23, "upper": 1.24},
        {"time": "2026-06-02T21:00Z", "value": 1.23456, "lower": 1.23, "upper": 1.24},
    ]
    assert out["point_forecast_mode"] == "flat_anchor"


def test_run_forecast_conformal_intervals_rewrites_interval_unavailable_guidance():
    result = forecast_use_cases.run_forecast_conformal_intervals(
        ForecastConformalIntervalsRequest(
            symbol="EURUSD",
            method="theta",
            horizon=1,
            steps=1,
            spacing=1,
        ),
        backtest_impl=lambda **kwargs: {
            "results": {
                "theta": {
                    "details": [
                        {"forecast": [10.0], "actual": [9.0]},
                    ]
                }
            }
        },
        forecast_impl=lambda **kwargs: {
            "forecast_price": [100.0],
            "ci_status": "unavailable",
            "warnings": [
                "Point forecast only for method 'theta'; confidence intervals are unavailable. "
                "Use forecast_conformal_intervals for uncertainty bands.",
                "native theta fallback used",
            ],
        },
    )

    assert result["ci_status"] == "available"
    assert result["ci_available"] is True
    assert result["lower_price"] == [99.0]
    assert result["upper_price"] == [101.0]
    assert result["warnings"][0] == "native theta fallback used"
    assert "below 30" in result["warnings"][1]


def test_run_forecast_conformal_intervals_raises_typed_error_for_nested_error_payload():
    with pytest.raises(ForecastError, match="backtest failed"):
        forecast_use_cases.run_forecast_conformal_intervals(
            ForecastConformalIntervalsRequest(
                symbol="EURUSD",
                method="theta",
                horizon=1,
                steps=1,
                spacing=1,
            ),
            backtest_impl=lambda **kwargs: {"error": "backtest failed"},
            forecast_impl=lambda **kwargs: {"forecast_price": [100.0]},
        )


def test_forecast_tune_genetic_and_barrier_prob_routing(monkeypatch):
    raw_tune = _unwrap(cf.forecast_tune_genetic)
    raw_barrier = _unwrap(cf.forecast_barrier_prob)

    captured = {}
    ss_calls = {}

    def fake_genetic(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cf, "_genetic_search_impl", fake_genetic)

    import mtdata.forecast.tune as tune_mod

    def fake_default_search_space(method=None, methods=None):
        ss_calls["method"] = method
        ss_calls["methods"] = methods
        return {"theta": {"window": {"min": 1, "max": 3}}}

    monkeypatch.setattr(tune_mod, "default_search_space", fake_default_search_space)
    out = raw_tune(request=ForecastTuneGeneticRequest(symbol="EURUSD", method="theta", search_space=None))
    assert out == {"ok": True, "detail": "compact"}
    assert captured["method"] == "theta"
    assert ss_calls["method"] == "theta"
    assert ss_calls["methods"] is None
    assert "theta" in captured["search_space"]

    out = raw_tune(
        request=ForecastTuneGeneticRequest(
            symbol="EURUSD",
            method="theta",
            methods=["theta", "naive"],
            search_space={"x": {"type": "int"}},
        )
    )
    assert out == {"ok": True, "detail": "compact"}
    assert captured["method"] is None

    monkeypatch.setattr(cf, "_genetic_search_impl", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    assert "Error in genetic tuning" in raw_tune(request=ForecastTuneGeneticRequest(symbol="EURUSD"))["error"]

    monkeypatch.setattr(cf, "_build_barrier_kwargs_from", lambda _: {"tp_abs": 1.2, "sl_abs": 1.1})

    import mtdata.forecast.barriers_probabilities as barriers_mod

    called = {}

    def fake_mc(**kwargs):
        called.update(kwargs)
        return {"kind": "mc", "direction": kwargs["direction"], "method": kwargs["method"]}

    def fake_cf(**kwargs):
        called.update(kwargs)
        return {"kind": "cf", "direction": kwargs["direction"]}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_hit_probabilities", fake_mc)
    monkeypatch.setattr(barriers_mod, "forecast_barrier_closed_form", fake_cf)

    out = raw_barrier(
        request=ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="auto",
            direction="down",
        )
    )
    assert out["kind"] == "mc"
    assert out["method"] == "auto"
    assert out["direction"] == "short"
    assert out["detail"] == "compact"

    out = raw_barrier(
        request=ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="closed_form",
            direction="weird",
        )
    )
    assert "error" in out
    assert "Invalid direction" in out["error"]

    out = raw_barrier(request=ForecastBarrierProbRequest(symbol="EURUSD", method="mystery"))
    assert out["error_code"] == "unsupported_method"
    assert "Unsupported barrier method: mystery" in out["error"]
    assert "monte_carlo->mc_gbm" in out["error"]


def test_forecast_barrier_methods_accept_monte_carlo_aliases():
    called: dict[str, str] = {}

    def fake_barrier_hit(**kwargs):
        called["prob_method"] = kwargs["method"]
        return {"success": True}

    out = forecast_use_cases.run_forecast_barrier_prob(
        ForecastBarrierProbRequest(symbol="EURUSD", method="monte_carlo"),
        build_barrier_kwargs=lambda _values: {},
        normalize_trade_direction=lambda _direction: ("long", None),
        barrier_hit_probabilities_impl=fake_barrier_hit,
        barrier_closed_form_impl=lambda **_kwargs: {"unused": True},
    )

    assert out["success"] is True
    assert called["prob_method"] == "mc_gbm"

    def fake_optimize(**kwargs):
        called["optimize_method"] = kwargs["method"]
        called["optimize_output_mode"] = kwargs["output_mode"]
        assert "format" not in kwargs
        return {"success": True, "best": {}}

    out_opt = forecast_use_cases.run_forecast_barrier_optimize(
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="monte_carlo_bb"),
        parse_kv_or_json=lambda value: value or {},
        barrier_optimize_impl=fake_optimize,
    )

    assert out_opt["success"] is True
    assert called["optimize_method"] == "mc_gbm_bb"
    assert called["optimize_output_mode"] == "summary"


def test_forecast_barrier_prob_applies_default_pct_barriers_when_missing(monkeypatch):
    monkeypatch.setattr(forecast_use_cases, "_symbol_price_currency", lambda _symbol: "USD")
    called: dict[str, float] = {}

    def fake_barrier_hit(**kwargs):
        called["tp_pct"] = kwargs["tp_pct"]
        called["sl_pct"] = kwargs["sl_pct"]
        return {
            "success": True,
            "symbol": "EURUSD",
            "prob_tp_first": 0.5,
            "prob_sl_first": 0.4,
            "prob_no_hit": 0.1,
        }

    out = forecast_use_cases.run_forecast_barrier_prob(
        ForecastBarrierProbRequest(symbol="EURUSD"),
        build_barrier_kwargs=lambda _values: {},
        normalize_trade_direction=lambda _direction: ("long", None),
        barrier_hit_probabilities_impl=fake_barrier_hit,
        barrier_closed_form_impl=lambda **_kwargs: {"unused": True},
    )

    assert out["success"] is True
    assert out["price_currency"] == "USD"
    assert called == {"tp_pct": 1.0, "sl_pct": 1.0}
    assert out["warnings"] == [
        "Default 1% symmetrical barriers applied; pass tp_pct/sl_pct, "
        "tp_abs/sl_abs, or tp_ticks/sl_ticks to customize."
    ]
    assert out["tp_pct"] == 1.0
    assert out["sl_pct"] == 1.0
    assert out["barrier_unit"] == "percent"
    assert out["barrier_mode"] == "pct"
    assert out["probability_unit"] == "fraction"
    assert out["probability_edge"] == 0.1
    assert out["probability_edge_definition"] == "prob_tp_first - prob_sl_first"
    assert "edge" not in out
    assert out["units"]["horizon"] == "bars"
    assert out["units"]["tp_pct"] == "percentage_points"
    assert out["units"]["prob_tp_first"] == "probability_fraction"
    assert out["units"]["probability_edge"] == "probability_difference"
    assert out["verdict"] == "TP-first probability bias"


def test_forecast_barrier_prob_keeps_partial_barrier_inputs_strict():
    called: dict[str, object] = {}

    def fake_barrier_hit(**kwargs):
        called.update(kwargs)
        return {"error": "Missing barriers."}

    out = forecast_use_cases.run_forecast_barrier_prob(
        ForecastBarrierProbRequest(symbol="EURUSD", tp_pct=0.5),
        build_barrier_kwargs=lambda _values: {"tp_pct": 0.5},
        normalize_trade_direction=lambda _direction: ("long", None),
        barrier_hit_probabilities_impl=fake_barrier_hit,
        barrier_closed_form_impl=lambda **_kwargs: {"unused": True},
    )

    assert out == {"error": "Missing barriers."}
    assert called["tp_pct"] == 0.5
    assert "sl_pct" not in called


def test_forecast_barrier_optimize_rejects_unknown_method_without_traceback():
    called = False

    def fake_optimize(**_kwargs):
        nonlocal called
        called = True
        return {"success": True}

    out = forecast_use_cases.run_forecast_barrier_optimize(
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="mystery"),
        parse_kv_or_json=lambda value: value or {},
        barrier_optimize_impl=fake_optimize,
    )

    assert called is False
    assert out["error_code"] == "unsupported_method"
    assert "Unsupported barrier method: mystery" in out["error"]
    assert "traceback_summary" not in out


def test_forecast_barrier_optimize_rounds_public_float_artifacts():
    def fake_optimize(**_kwargs):
        return {
            "success": True,
            "price_precision": 5,
            "best": {
                "tp": 0.45833333333333337,
                "rr": 1.8333333333333335,
                "tp_price": 1.1764675416666668,
                "prob_resolve": 0.46950000000000003,
                "edge": -0.21050000000000002,
                "profit_factor": 0.6982843137254903,
            },
            "results": [
                {
                    "sl_price": 1.1681722500000001,
                    "edge_vs_breakeven": -0.2234411764705882,
                }
            ],
        }

    out = forecast_use_cases.run_forecast_barrier_optimize(
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="mc_gbm"),
        parse_kv_or_json=lambda value: value or {},
        barrier_optimize_impl=fake_optimize,
    )

    assert out["best"]["tp"] == 0.458333
    assert out["best"]["rr"] == 1.8333
    assert out["best"]["tp_price"] == 1.17647
    assert out["best"]["prob_resolve"] == 0.4695
    assert out["best"]["edge"] == -0.2105
    assert out["best"]["profit_factor"] == 0.698284
    assert out["results"][0]["sl_price"] == 1.16817
    assert out["results"][0]["edge_vs_breakeven"] == -0.223441
    assert out["barrier_unit"] == "percent"
    assert out["probability_unit"] == "fraction"
    assert "Expected reward/risk edge" in out["edge_definition"]


def test_forecast_barrier_optimize_compact_trims_blocked_status_noise():
    reason = "No viable TP/SL candidates satisfied the viability filter."

    def fake_optimize(**_kwargs):
        return {
            "success": True,
            "results_total": 0,
            "viable_results_total": 0,
            "best": None,
            "viable": False,
            "no_candidates": True,
            "status": "non_viable",
            "status_reason": reason,
            "no_action": True,
            "no_action_reason": reason,
            "actionability": "blocked",
            "actionability_reason": reason,
            "actionability_flags": ["status_non_viable", "warning"],
            "mathematically_viable": False,
            "tradable": False,
            "trade_gate_passed": False,
            "warning": reason,
            "output_mode": "concise",
            "viable_only": True,
            "concise": True,
            "reference_price": 1.16606,
            "reference_price_source": "live_tick_ask",
        }

    out = forecast_use_cases.run_forecast_barrier_optimize(
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="mc_gbm"),
        parse_kv_or_json=lambda value: value or {},
        barrier_optimize_impl=fake_optimize,
    )

    assert out["status"] == "non_viable"
    assert out["status_reason"] == reason
    assert out["tradable"] is False
    assert out["results_total"] == 0
    assert out["viable_results_total"] == 0
    assert out["best"] is None
    assert out["reference_price"] == 1.16606
    for key in (
        "no_action",
        "no_action_reason",
        "actionability",
        "actionability_reason",
        "actionability_flags",
        "mathematically_viable",
        "trade_gate_passed",
        "viable",
        "no_candidates",
        "warning",
        "output_mode",
        "viable_only",
        "concise",
    ):
        assert key not in out


def test_forecast_barrier_prob_closed_form_rejects_tp_sl_inputs_before_generic_error():
    called = False

    def fake_closed_form(**_kwargs):
        nonlocal called
        called = True
        return {"error": "Provide a positive barrier price"}

    out = forecast_use_cases.run_forecast_barrier_prob(
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="closed_form",
            tp_pct=0.5,
            sl_pct=0.5,
        ),
        build_barrier_kwargs=lambda _values: {},
        normalize_trade_direction=lambda _direction: ("long", None),
        barrier_hit_probabilities_impl=lambda **_kwargs: {"unused": True},
        barrier_closed_form_impl=fake_closed_form,
    )

    assert called is False
    assert out["error_code"] == "invalid_input"
    assert "closed_form method uses the absolute barrier parameter" in out["error"]
    assert "mc_gbm" in out["error"]


def test_forecast_barrier_prob_closed_form_rejects_barrier_with_tp_sl_inputs():
    called = False

    def fake_closed_form(**_kwargs):
        nonlocal called
        called = True
        return {"success": True}

    out = forecast_use_cases.run_forecast_barrier_prob(
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="closed_form",
            barrier=1.18,
            tp_pct=0.5,
            sl_pct=0.3,
        ),
        build_barrier_kwargs=lambda _values: {},
        normalize_trade_direction=lambda _direction: ("long", None),
        barrier_hit_probabilities_impl=lambda **_kwargs: {"unused": True},
        barrier_closed_form_impl=fake_closed_form,
    )

    assert called is False
    assert out["error_code"] == "invalid_input"
    assert "closed_form method uses the absolute barrier parameter only" in out["error"]
    assert "tp_pct, sl_pct" in out["error"]


def test_forecast_barrier_prob_wrapper_emits_single_finish_event(caplog, monkeypatch):
    raw = _unwrap(cf.forecast_barrier_prob)
    monkeypatch.setattr(cf, "_forecast_connection_error", lambda: None)
    monkeypatch.setattr(cf, "_build_barrier_kwargs_from", lambda _: {"tp_abs": 1.2, "sl_abs": 1.1})

    import mtdata.forecast.barriers_probabilities as barriers_mod

    monkeypatch.setattr(
        barriers_mod,
        "forecast_barrier_hit_probabilities",
        lambda **kwargs: {"success": True, "kind": "mc", "direction": kwargs["direction"]},
    )
    monkeypatch.setattr(
        barriers_mod,
        "forecast_barrier_closed_form",
        lambda **kwargs: {"success": True, "kind": "closed_form", "direction": kwargs["direction"]},
    )

    with caplog.at_level(logging.DEBUG):
        out = raw(request=ForecastBarrierProbRequest(symbol="EURUSD", timeframe="H1"))

    assert out["success"] is True
    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=forecast_barrier_prob success=True" in record.message
    ]
    assert len(finish_records) == 1


def test_forecast_barrier_prob_standard_hides_curves_only(monkeypatch):
    raw = _unwrap(cf.forecast_barrier_prob)
    monkeypatch.setattr(cf, "_forecast_connection_error", lambda: None)
    monkeypatch.setattr(cf, "_build_barrier_kwargs_from", lambda _: {"tp_abs": 1.2, "sl_abs": 1.1})

    import mtdata.forecast.barriers_probabilities as barriers_mod

    monkeypatch.setattr(
        barriers_mod,
        "forecast_barrier_hit_probabilities",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "timeframe": kwargs["timeframe"],
            "method": kwargs["method"],
            "direction": kwargs["direction"],
            "horizon": kwargs["horizon"],
            "last_price": 1.15,
            "tp_price": 1.2,
            "sl_price": 1.1,
            "prob_tp_first": 0.55,
            "prob_sl_first": 0.30,
            "prob_no_hit": 0.15,
            "prob_tp_first_ci95": {"low": 0.5, "high": 0.6},
            "tp_hit_prob_by_t": [0.1, 0.2],
            "sl_hit_prob_by_t": [0.05, 0.1],
            "sim_meta": {"foo": "bar"},
        },
    )
    monkeypatch.setattr(barriers_mod, "forecast_barrier_closed_form", lambda **kwargs: {"success": True})

    out = raw(request=ForecastBarrierProbRequest(symbol="EURUSD", detail="standard"))

    assert out["detail"] == "standard"
    assert "tp_hit_prob_by_t" not in out
    assert "sim_meta" not in out
    assert "prob_tp_first_ci95" in out


def test_forecast_barrier_prob_compact_nests_confidence_intervals_once():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "n_sims": 2000,
        "seed": 42,
        "prob_tp_first": 0.55,
        "prob_sl_first": 0.30,
        "prob_no_hit": 0.15,
        "prob_tp_first_se": 0.0111,
        "prob_sl_first_se": 0.0102,
        "prob_no_hit_se": 0.008,
        "prob_tp_first_ci95": {"low": 0.5, "high": 0.6},
        "prob_sl_first_ci95": {"low": 0.25, "high": 0.35},
        "prob_no_hit_ci95": {"low": 0.1, "high": 0.2},
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(symbol="EURUSD", detail="compact"),
    )

    assert out["confidence"] == {
        "prob_tp_first_ci95": {"low": 0.5, "high": 0.6},
        "prob_sl_first_ci95": {"low": 0.25, "high": 0.35},
        "prob_no_hit_ci95": {"low": 0.1, "high": 0.2},
        "prob_tp_first_se": 0.0111,
        "prob_sl_first_se": 0.0102,
        "prob_no_hit_se": 0.008,
    }
    assert out["n_sims"] == 2000
    assert out["seed"] == 42
    assert "prob_tp_first_ci95" not in out
    assert "prob_sl_first_ci95" not in out
    assert "prob_no_hit_ci95" not in out
    assert "prob_tp_first_se" not in out
    assert "prob_sl_first_se" not in out
    assert "prob_no_hit_se" not in out


def test_forecast_barrier_prob_compact_uses_reference_price_context():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "last_price": 1.16026,
        "last_price_close": 1.16016,
        "last_price_source": "live_tick_ask",
        "tp_price": 1.16606,
        "sl_price": 1.15678,
        "prob_tp_first": 0.55,
        "prob_sl_first": 0.35,
        "prob_no_hit": 0.10,
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            detail="compact",
            tp_pct=0.5,
            sl_pct=0.3,
        ),
    )

    assert out["reference_price"] == 1.16026
    assert out["reference_price_source"] == "live_tick_ask"
    assert out["tp_pct"] == 0.5
    assert out["sl_pct"] == 0.3
    assert "last_price" not in out
    assert "last_price_close" not in out
    assert "last_price_source" not in out


def test_forecast_barrier_prob_closed_form_compact_keeps_reference_source():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "last_price": 1.16594,
        "last_price_source": "candle_close",
        "barrier": 1.18,
        "prob_hit": 0.15,
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="closed_form",
            barrier=1.18,
            detail="compact",
        ),
    )

    assert out["reference_price"] == 1.16594
    assert out["reference_price_source"] == "candle_close"
    assert out["prob_hit"] == 0.15
    assert "last_price" not in out
    assert "last_price_source" not in out


def test_forecast_barrier_prob_detail_rounds_display_values():
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "last_price": 1.1720124100000001,
        "tp_price": 1.1780124100000001,
        "sl_price": 1.1690124100000001,
        "prob_tp_first": 0.5123456789,
        "prob_sl_first": 0.4876543211,
        "probability_edge": -0.17800000000000005,
        "prob_tp_first_ci95": {"low": 0.5000000001, "high": 0.6000000001},
    }

    out = forecast_use_cases._apply_barrier_prob_detail(
        payload,
        ForecastBarrierProbRequest(symbol="EURUSD", detail="compact"),
    )

    assert out["reference_price"] == 1.17201241
    assert "last_price" not in out
    assert out["tp_price"] == 1.17801241
    assert out["sl_price"] == 1.16901241
    assert out["prob_tp_first"] == 0.512346
    assert out["probability_edge"] == -0.178
    assert out["probability_unit"] == "fraction"
    assert out["probability_edge_definition"] == "prob_tp_first - prob_sl_first"
    assert "edge" not in out
    assert out["confidence"]["prob_tp_first_ci95"] == {"low": 0.5, "high": 0.6}


def test_forecast_barrier_optimize_uses_reference_price_context():
    def fake_optimize(**_kwargs):
        return {
            "success": True,
            "last_price": 1.16026,
            "last_price_close": 1.16016,
            "last_price_source": "live_tick_ask",
            "best": {"tp": 0.25, "sl": 0.25},
            "results": [],
        }

    out = forecast_use_cases.run_forecast_barrier_optimize(
        ForecastBarrierOptimizeRequest(symbol="EURUSD", method="mc_gbm"),
        parse_kv_or_json=lambda value: value or {},
        barrier_optimize_impl=fake_optimize,
    )

    assert out["reference_price"] == 1.16026
    assert out["reference_price_source"] == "live_tick_ask"
    assert "last_price" not in out
    assert "last_price_close" not in out
    assert "last_price_source" not in out


def test_forecast_tune_optuna_routing(monkeypatch):
    raw_tune = _unwrap(cf.forecast_tune_optuna)
    captured = {}
    ss_calls = {}

    def fake_optuna(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cf, "_optuna_search_impl", fake_optuna)

    import mtdata.forecast.tune as tune_mod

    def fake_default_search_space(method=None, methods=None):
        ss_calls["method"] = method
        ss_calls["methods"] = methods
        return {"theta": {"window": {"min": 1, "max": 3}}}

    monkeypatch.setattr(tune_mod, "default_search_space", fake_default_search_space)
    out = raw_tune(request=ForecastTuneOptunaRequest(symbol="EURUSD", method="theta", search_space=None))
    assert out == {"ok": True, "detail": "compact"}
    assert captured["method"] == "theta"
    assert ss_calls["method"] == "theta"
    assert ss_calls["methods"] is None
    assert "theta" in captured["search_space"]

    out = raw_tune(
        request=ForecastTuneOptunaRequest(
            symbol="EURUSD",
            method="theta",
            methods=["theta", "naive"],
            search_space={"x": {"type": "int"}},
        )
    )
    assert out == {"ok": True, "detail": "compact"}
    assert captured["method"] is None

    monkeypatch.setattr(cf, "_optuna_search_impl", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    assert "Error in optuna tuning" in raw_tune(request=ForecastTuneOptunaRequest(symbol="EURUSD"))["error"]


def _ready_options_provider():
    return {
        "configured_provider": "tradier",
        "effective_provider": "tradier",
        "api_key_configured": True,
        "chain_data_ready": True,
        "action_required": None,
        "remediation": None,
    }


def test_options_and_quantlib_tool_routing(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)
    raw_chain = _unwrap(opt.options_chain)
    raw_price = _unwrap(opt.options_barrier_price)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.forecast.quantlib_tools as quantlib_tools
    import mtdata.services.options_service as options_service

    monkeypatch.setattr(options_service, "get_options_expirations", lambda **kwargs: {"kind": "exp", **kwargs})
    monkeypatch.setattr(options_service, "get_options_chain", lambda **kwargs: {"kind": "chain", **kwargs})
    monkeypatch.setattr(quantlib_tools, "price_barrier_option_quantlib", lambda **kwargs: {"kind": "price", **kwargs})
    monkeypatch.setattr(quantlib_tools, "calibrate_heston_quantlib_from_options", lambda **kwargs: {"kind": "cal", **kwargs})
    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)

    out = raw_exp(symbol="AAPL")
    assert out["kind"] == "exp"
    assert out["symbol"] == "AAPL"

    out = raw_chain(symbol="AAPL", expiration="2026-06-19", option_type="call", min_open_interest=10, min_volume=5, limit=20)
    assert out["kind"] == "chain"
    assert out["symbol"] == "AAPL"
    assert out["option_type"] == "call"
    assert out["limit"] == 20

    out = raw_price(
        spot=100.0,
        strike=105.0,
        barrier=120.0,
        maturity_days=30,
        option_type="call",
        barrier_type="up_out",
        risk_free_rate=0.03,
        dividend_yield=0.01,
        volatility=0.25,
        rebate=0.0,
    )
    assert out["kind"] == "price"
    assert out["spot"] == 100.0

    out = raw_cal(
        symbol="AAPL",
        expiration="2026-06-19",
        option_type="put",
        risk_free_rate=0.03,
        dividend_yield=0.01,
        min_open_interest=10,
        min_volume=2,
        max_contracts=15,
    )
    assert out["kind"] == "cal"
    assert out["symbol"] == "AAPL"
    assert out["option_type"] == "put"


def test_options_tools_validate_and_normalize_symbols(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)
    raw_chain = _unwrap(opt.options_chain)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.services.options_service as options_service

    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)
    monkeypatch.setattr(
        options_service,
        "get_options_expirations",
        lambda **kwargs: {"success": True, **kwargs},
    )

    for tool in (raw_exp, raw_chain, raw_cal):
        out = tool(symbol="  ")
        assert out["success"] is False
        assert out["error_code"] == "invalid_symbol"
        assert out["error"] == "symbol is required"

        out = tool(symbol="AAPL?query=1")
        assert out["success"] is False
        assert out["error_code"] == "invalid_symbol"

    out = raw_exp(symbol=" brk.b ")
    assert out["success"] is True
    assert out["symbol"] == "BRK.B"


def test_options_chain_tools_short_circuit_when_provider_not_ready(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)
    raw_chain = _unwrap(opt.options_chain)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.forecast.quantlib_tools as quantlib_tools
    import mtdata.services.options_service as options_service

    def fail_call(**kwargs):
        raise AssertionError("options provider should not be queried")

    monkeypatch.setattr(options_service, "get_options_expirations", fail_call)
    monkeypatch.setattr(options_service, "get_options_chain", fail_call)
    monkeypatch.setattr(quantlib_tools, "calibrate_heston_quantlib_from_options", fail_call)
    monkeypatch.setattr(
        opt,
        "_options_provider_readiness",
        lambda: {
            "configured_provider": "tradier",
            "effective_provider": "tradier",
            "api_key_configured": False,
            "chain_data_ready": False,
            "action_required": "configure_options_provider",
            "remediation": "configure Tradier",
        },
    )

    for result in (
        raw_exp(symbol="AAPL"),
        raw_chain(symbol="AAPL"),
        raw_cal(symbol="AAPL"),
    ):
        assert result["success"] is False
        assert result["error_code"] == "options_provider_auth"
        assert result["provider"] == "tradier"
        assert result["next_tool"] == "options_provider_status"
        assert result["chain_data_ready"] is False
        assert result["action_required"] == "configure_options_provider"


def test_options_chain_tools_allow_yahoo_best_effort_provider(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)

    import mtdata.services.options_service as options_service

    monkeypatch.setattr(
        options_service,
        "get_options_expirations",
        lambda **kwargs: {"success": True, "provider": "yahoo", **kwargs},
    )
    monkeypatch.setattr(
        opt,
        "_options_provider_readiness",
        lambda: {
            "configured_provider": "yahoo",
            "effective_provider": "yahoo",
            "api_key_configured": False,
            "chain_data_ready": True,
            "provider_mode": "best_effort",
            "action_required": None,
        },
    )

    result = raw_exp(symbol="AAPL")

    assert result["success"] is True
    assert result["provider"] == "yahoo"


def test_options_chain_logs_finish_event(caplog, monkeypatch):
    raw_chain = _unwrap(opt.options_chain)

    import mtdata.services.options_service as options_service

    monkeypatch.setattr(options_service, "get_options_chain", lambda **kwargs: {"success": True, **kwargs})
    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)

    with caplog.at_level(logging.DEBUG, logger=opt.logger.name):
        out = raw_chain(symbol="AAPL", expiration="2026-06-19", option_type="call", limit=25)

    assert out["success"] is True
    assert any(
        "event=finish operation=options_chain success=True" in record.message
        for record in caplog.records
    )


def test_options_barrier_compact_keeps_numeric_pricing_inputs():
    payload = {
        "success": True,
        "price": 1.23,
        "delta": 0.4,
        "params_used": {
            "risk_free_rate": 0.05,
            "dividend_yield": 0.01,
            "volatility": 0.2,
            "rebate": 0.0,
        },
    }

    result = opt._apply_options_detail(
        payload,
        detail="compact",
        kind="barrier_price",
    )

    assert result["pricing_inputs"] == {
        "risk_free_rate": 0.05,
        "dividend_yield": 0.01,
        "volatility": 0.2,
        "rebate": 0.0,
        "rate_unit": "decimal_fraction",
    }


def test_options_tools_support_compact_and_full_detail(monkeypatch):
    raw_exp = _unwrap(opt.options_expirations)
    raw_chain = _unwrap(opt.options_chain)
    raw_price = _unwrap(opt.options_barrier_price)
    raw_cal = _unwrap(opt.options_heston_calibrate)

    import mtdata.forecast.quantlib_tools as quantlib_tools
    import mtdata.services.options_service as options_service

    monkeypatch.setattr(opt, "_options_provider_readiness", _ready_options_provider)
    monkeypatch.setattr(
        options_service,
        "get_options_expirations",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "underlying_price": 100.0,
            "currency": "USD",
            "expirations": ["2026-06-19"],
            "expiration_count": 1,
        },
    )
    monkeypatch.setattr(
        options_service,
        "get_options_chain",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "expiration": "2026-06-19",
            "underlying_price": 100.0,
            "currency": "USD",
            "option_type": kwargs["option_type"],
            "count": 1,
            "calls_count": 1,
            "puts_count": 0,
            "contract_size": "REGULAR",
            "expirations": ["2026-06-19"],
            "options": [
                {
                    "side": "call",
                    "contract": "AAPL260619C00100000",
                    "strike": 100.0,
                    "last": 2.0,
                    "bid": 1.9,
                    "ask": 2.1,
                    "implied_volatility": 0.2,
                    "in_the_money": True,
                    "last_trade_epoch": 1700000000,
                    "volume": 10,
                    "open_interest": 20,
                }
            ],
        },
    )
    monkeypatch.setattr(
        quantlib_tools,
        "price_barrier_option_quantlib",
        lambda **kwargs: {
            "success": True,
            "price": 1.23,
            "delta": 0.4,
            "gamma": 0.01,
            "vega": 0.2,
            "params_used": {
                "spot": kwargs["spot"],
                "strike": kwargs["strike"],
                "barrier": kwargs["barrier"],
                "option_type": kwargs["option_type"],
                "barrier_type": kwargs["barrier_type"],
                "maturity_days": kwargs["maturity_days"],
                "risk_free_rate": kwargs["risk_free_rate"],
                "dividend_yield": kwargs["dividend_yield"],
                "volatility": kwargs["volatility"],
                "rebate": kwargs["rebate"],
            },
        },
    )
    monkeypatch.setattr(
        quantlib_tools,
        "calibrate_heston_quantlib_from_options",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "expiration": "2026-06-19",
            "days_to_expiry": 30,
            "contracts_used": 5,
            "spot": 100.0,
            "calibration_error_rmse": 0.01,
            "params": {"kappa": 1.0},
            "sample_contracts": [{"strike": 100.0, "iv": 0.2}],
        },
    )

    assert "underlying_price" not in raw_exp("AAPL", detail="compact")
    assert raw_exp("AAPL", detail="full")["underlying_price"] == 100.0

    compact_chain = raw_chain("AAPL", detail="compact")
    assert compact_chain["detail"] == "compact"
    assert "contract_size" not in compact_chain
    assert "implied_volatility" not in compact_chain["options"][0]
    assert raw_chain("AAPL", detail="full")["options"][0]["implied_volatility"] == 0.2

    compact_price = raw_price(100, 105, 120, 30, detail="compact")
    assert compact_price["price"] == 1.23
    assert compact_price["delta"] == 0.4
    assert compact_price["detail"] == "compact"
    assert compact_price["units"] == {
        "price": "premium_per_underlying_unit",
        "delta": "premium_change_per_underlying_price_unit",
    }
    assert compact_price["pricing_inputs"] == {
        "risk_free_rate": 0.02,
        "dividend_yield": 0.0,
        "volatility": 0.2,
        "rebate": 0.0,
        "rate_unit": "decimal_fraction",
    }
    full_price = raw_price(100, 105, 120, 30, detail="full")
    assert full_price["delta"] == 0.4
    assert full_price["units"]["price"] == "premium_per_underlying_unit"
    assert full_price["params_used"]["spot"] == 100

    compact_cal = raw_cal("AAPL", detail="compact")
    assert compact_cal["params"] == {"kappa": 1.0}
    assert "sample_contracts" not in compact_cal
    assert raw_cal("AAPL", detail="full")["sample_contracts"] == [{"strike": 100.0, "iv": 0.2}]


def test_forecast_barrier_optimize_routes_profile_args(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {
            "ok": True,
            "search_profile": kwargs.get("search_profile"),
            "fast_defaults_param": kwargs.get("params", {}).get("fast_defaults"),
        }

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(
        request=ForecastBarrierOptimizeRequest(
            symbol="EURUSD",
            search_profile="long",
            params={"fast_defaults": True},
        )
    )
    assert out["ok"] is True
    assert out["fast_defaults_param"] is True
    assert out["search_profile"] == "long"
    assert called["params"]["fast_defaults"] is True
    assert called["fast_defaults"] is False
    assert called["search_profile"] == "long"


def test_forecast_barrier_optimize_routes_statistical_robustness_args(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(
        request=ForecastBarrierOptimizeRequest(
            symbol="EURUSD",
            params={
                "statistical_robustness": True,
                "target_ci_width": 0.02,
                "n_seeds_stability": 4,
                "enable_bootstrap": True,
                "n_bootstrap": 250,
                "enable_convergence_check": False,
                "convergence_window": 80,
                "convergence_threshold": 0.005,
                "enable_power_analysis": True,
                "power_effect_size": 0.02,
                "enable_sensitivity_analysis": True,
                "sensitivity_params": ["tp", "sl"],
            },
        )
    )
    assert out["ok"] is True
    assert called["statistical_robustness"] is False
    assert called["params"]["statistical_robustness"] is True
    assert called["params"]["target_ci_width"] == 0.02
    assert called["params"]["n_seeds_stability"] == 4
    assert called["params"]["enable_bootstrap"] is True
    assert called["params"]["n_bootstrap"] == 250
    assert called["params"]["enable_convergence_check"] is False
    assert called["params"]["convergence_window"] == 80
    assert called["params"]["convergence_threshold"] == 0.005
    assert called["params"]["enable_power_analysis"] is True
    assert called["params"]["power_effect_size"] == 0.02
    assert called["params"]["enable_sensitivity_analysis"] is True
    assert called["params"]["sensitivity_params"] == ["tp", "sl"]


def test_forecast_barrier_optimize_routes_advanced_grid_params(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(
        request=ForecastBarrierOptimizeRequest(
            symbol="EURUSD",
            params={"vol_sl_multiplier": 2.1},
        )
    )

    assert out["ok"] is True
    assert called["params"]["vol_sl_multiplier"] == 2.1
    assert called["vol_sl_multiplier"] == 1.8


def test_forecast_barrier_optimize_keeps_grid_default_path(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="BTCUSD"))
    assert out["ok"] is True
    assert out["detail"] == "compact"
    assert called["method"] == "auto"
    assert called["search_profile"] == "medium"
    assert called["output_mode"] == "summary"
    assert "format" not in called
    assert called["concise"] is True
    assert called["return_grid"] is False
    assert "seed" not in called["params"]
    assert "optimizer" not in called["params"]
    assert "sampler" not in called["params"]
    assert "pruner" not in called["params"]
    assert "n_jobs" not in called["params"]


def test_forecast_barrier_optimize_standard_disables_concise_only(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="BTCUSD", detail="standard"))

    assert out["detail"] == "standard"
    assert called["output_mode"] == "summary"
    assert "format" not in called
    assert called["concise"] is False


def test_forecast_barrier_optimize_applies_optuna_defaults_when_requested(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="BTCUSD", params={"optimizer": "optuna"}))
    assert out["ok"] is True
    assert called["params"]["optimizer"] == "optuna"
    assert called["params"]["sampler"] == "tpe"
    assert called["params"]["pruner"] == "median"
    assert int(called["params"]["n_jobs"]) >= 1


def test_forecast_barrier_optimize_preserves_explicit_seed(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)
    called = {}

    import mtdata.forecast.barriers_optimization as barriers_mod

    def fake_optimize(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(barriers_mod, "forecast_barrier_optimize", fake_optimize)
    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="BTCUSD", params={"seed": 17}))
    assert out["ok"] is True
    assert called["params"]["seed"] == 17


def test_forecast_barrier_optimize_returns_connection_error_payload(monkeypatch):
    raw_opt = _unwrap(cf.forecast_barrier_optimize)

    def fail_connection():
        raise MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running.")

    monkeypatch.setattr(cf, "ensure_mt5_connection_or_raise", fail_connection)

    out = raw_opt(request=ForecastBarrierOptimizeRequest(symbol="EURUSD"))

    assert out["success"] is False
    assert out["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    assert out["error_code"] == "mt5_connection_error"
    assert out["operation"] == "mt5_ensure_connection"
    assert isinstance(out.get("request_id"), str)
