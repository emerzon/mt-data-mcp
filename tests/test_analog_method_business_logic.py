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
    series = pd.Series(np.linspace(100.0, 105.0, 10), name="close")

    monkeypatch.setattr(method, "_run_single_timeframe", lambda *args, **kwargs: ([], []))

    with pytest.raises(RuntimeError, match="Primary analog search failed"):
        method.forecast(series, horizon=3, seasonality=1, params={"symbol": "EURUSD", "timeframe": "H1"})


def test_analog_method_aggregates_primary_and_secondary_paths(monkeypatch):
    method = AnalogMethod()
    series = pd.Series(np.linspace(100.0, 108.0, 20), name="close")

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
    assert len(out.metadata["analogs"]) == 2
