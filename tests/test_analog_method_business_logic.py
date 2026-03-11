from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.methods.analog import AnalogMethod


def test_analog_method_metadata_properties():
    method = AnalogMethod()
    assert method.name == "analog"
    assert method.category == "analog"
    assert "scipy" in method.required_packages
    assert method.supports_features["price"] is True
    assert method.supports_features["return"] is False


def test_analog_method_rejects_derived_or_missing_series():
    method = AnalogMethod()
    with pytest.raises(ValueError, match="price series only"):
        method.forecast(pd.Series([], dtype=float), horizon=3, seasonality=1, params={"symbol": "EURUSD", "timeframe": "H1"})

    with pytest.raises(ValueError, match="price series only"):
        method.forecast(pd.Series([0.1, 0.2], name="__log_return"), horizon=3, seasonality=1, params={"symbol": "EURUSD", "timeframe": "H1"})


def test_analog_method_requires_symbol_and_timeframe():
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 105.0, 10), name="close")

    with pytest.raises(ValueError, match="requires 'symbol' and 'timeframe'"):
        method.forecast(series, horizon=3, seasonality=1, params={})


def test_analog_method_raises_when_primary_search_fails(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 105.0, 80), name="close")

    def fake_run(*args, **kwargs):
        method._timeframe_diagnostics["H1"] = {"reason": "search_failed", "message": "search backend unavailable"}
        return ([], [])

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    with pytest.raises(RuntimeError, match="Primary analog search failed.*search failed.*search backend unavailable"):
        method.forecast(series, horizon=3, seasonality=1, params={"symbol": "EURUSD", "timeframe": "H1"})


def test_analog_method_aggregates_primary_and_secondary_paths(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        if timeframe == "H1":
            return (
                [
                    np.array([101.0, 102.0, 103.0], dtype=float),
                    np.array([100.0, 101.0, 102.0], dtype=float),
                ],
                [{"score": 0.1}, {"score": 0.2}],
            )
        # Secondary horizon for M30 becomes >= 6 with primary horizon=3
        return (
            [np.array([99.0, 99.5, 100.0, 100.5, 101.0, 101.5], dtype=float)],
            [{"score": 0.3}],
        )

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    out = method.forecast(
        series,
        horizon=3,
        seasonality=1,
        params={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "secondary_timeframes": "M30",
            "ci_alpha": 2.0,  # invalid should normalize to 0.05
        },
    )

    assert out.forecast.shape[0] == 3
    assert out.ci_values is not None
    assert out.params_used is not None
    assert out.params_used["ci_alpha"] == 0.05
    assert out.params_used["n_paths"] == 3
    assert out.metadata is not None
    assert out.metadata["components"] == ["H1", "M30"]
    assert out.metadata["requested_components"] == ["H1", "M30"]
    assert len(out.metadata["analogs"]) == 2
    assert out.metadata["component_status"][0]["status"] == "contributed"
    assert out.metadata["component_status"][1]["status"] == "contributed"


def test_analog_method_rejects_non_close_base_column():
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 105.0, 10), name="open")

    with pytest.raises(ValueError, match="close-based price series"):
        method.forecast(
            series,
            horizon=3,
            seasonality=1,
            params={"symbol": "EURUSD", "timeframe": "H1", "window_size": 5},
        )


def test_analog_method_requires_denoise_spec_for_close_dn_series():
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 105.0, 10), name="close_dn")

    with pytest.raises(ValueError, match="requires a denoise spec"):
        method.forecast(
            series,
            horizon=3,
            seasonality=1,
            params={"symbol": "EURUSD", "timeframe": "H1", "window_size": 5, "base_col": "close_dn"},
        )


def test_analog_method_reports_only_contributing_components(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        if timeframe == "H1":
            method._timeframe_diagnostics["H1"] = {"status": "ok", "returned_paths": 1}
            return ([np.array([101.0, 102.0, 103.0], dtype=float)], [{"score": 0.1}])
        method._timeframe_diagnostics[timeframe] = {"status": "ok", "returned_paths": 1}
        return ([np.array([99.0, 100.0, 101.0], dtype=float)], [{"score": 0.3}])

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    out = method.forecast(
        series,
        horizon=3,
        seasonality=1,
        params={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "secondary_timeframes": "H4",
        },
    )

    assert out.metadata is not None
    assert out.metadata["components"] == ["H1"]
    assert out.metadata["requested_components"] == ["H1", "H4"]
    secondary_status = next(item for item in out.metadata["component_status"] if item["timeframe"] == "H4")
    assert secondary_status["status"] == "skipped_insufficient_coverage"


def test_analog_method_reports_search_symbol_universe(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 80), name="close")

    def fake_run(symbol, timeframe, horizon, params, query_vector=None, **kwargs):
        method._timeframe_diagnostics[timeframe] = {
            "status": "ok",
            "returned_paths": 1,
            "search_symbols_used": ["EURUSD", "GBPUSD"],
        }
        return ([np.array([101.0, 102.0, 103.0], dtype=float)], [{"score": 0.1, "symbol": "GBPUSD"}])

    monkeypatch.setattr(method, "_run_single_timeframe", fake_run)

    out = method.forecast(
        series,
        horizon=3,
        seasonality=1,
        params={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "search_symbols": ["GBPUSD", "USDJPY"],
        },
    )

    assert out.params_used["search_symbols"] == ["EURUSD", "GBPUSD", "USDJPY"]
    assert out.metadata["search_symbols"] == ["EURUSD", "GBPUSD", "USDJPY"]
    assert out.metadata["analogs"][0]["meta"]["symbol"] == "GBPUSD"
