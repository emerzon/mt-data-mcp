from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.interface import ForecastResult
from mtdata.forecast.methods import classical as cl


def test_classical_base_metadata_via_naive_method():
    method = cl.NaiveMethod()
    assert method.name == "naive"
    assert method.category == "classical"
    assert method.supports_features == {
        "price": True,
        "return": True,
        "volatility": True,
        "ci": False,
    }


def test_naive_and_drift_forecasts_return_expected_values():
    series = pd.Series([10.0, 12.0, 15.0])

    naive = cl.NaiveMethod().forecast(series, horizon=3, seasonality=0, params={})
    assert np.allclose(naive.forecast, [15.0, 15.0, 15.0])
    assert naive.params_used == {}

    drift = cl.DriftMethod().forecast(series, horizon=3, seasonality=0, params={})
    assert np.allclose(drift.forecast, [17.5, 20.0, 22.5])
    assert drift.params_used == {"slope": 2.5}

    single = cl.DriftMethod().forecast(pd.Series([5.0]), horizon=2, seasonality=0, params={})
    assert np.allclose(single.forecast, [5.0, 5.0])
    assert single.params_used == {"slope": 0.0}


def test_seasonal_naive_validation_and_repeating_pattern():
    method = cl.SeasonalNaiveMethod()
    with pytest.raises(ValueError, match="Insufficient data"):
        method.forecast(pd.Series([1.0, 2.0]), horizon=1, seasonality=3, params={})
    with pytest.raises(ValueError, match="Insufficient data"):
        method.forecast(pd.Series([1.0, 2.0]), horizon=1, seasonality=0, params={})

    out = method.forecast(pd.Series([1.0, 2.0, 3.0, 4.0]), horizon=5, seasonality=2, params={})
    assert np.allclose(out.forecast, [3.0, 4.0, 3.0, 4.0, 3.0])
    assert out.params_used == {"m": 2}


def test_theta_forecast_tracks_trend_and_reports_alpha_and_slope():
    method = cl.ThetaMethod()
    series = pd.Series([2.0, 4.0, 6.0, 8.0])
    out = method.forecast(series, horizon=2, seasonality=0, params={"alpha": 0.5})

    # On linear data y=2t, OLS slope should be near 2.
    assert out.params_used is not None
    assert out.params_used["alpha"] == 0.5
    assert out.params_used["trend_slope"] == pytest.approx(2.0, abs=1e-10)
    assert out.forecast.shape == (2,)
    assert np.all(out.forecast > 0.0)


def test_fourier_ols_default_and_custom_params():
    method = cl.FourierOLSMethod()
    series = pd.Series(np.linspace(10.0, 20.0, 24))

    default = method.forecast(series, horizon=3, seasonality=12, params={})
    assert default.params_used == {"m": 12, "K": 3, "trend": True}
    assert default.forecast.shape == (3,)
    assert np.issubdtype(default.forecast.dtype, np.floating)

    no_seasonality = method.forecast(series, horizon=2, seasonality=0, params={"terms": None, "trend": False})
    assert no_seasonality.params_used == {"m": 0, "K": 2, "trend": False}
    assert no_seasonality.forecast.shape == (2,)

    custom = method.forecast(series, horizon=2, seasonality=24, params={"terms": 1, "trend": True})
    assert custom.params_used == {"m": 24, "K": 1, "trend": True}
    assert custom.forecast.shape == (2,)


def test_classical_legacy_wrappers_route_to_registry(monkeypatch):
    calls = []

    class FakeMethod:
        def __init__(self, name):
            self.name = name

        def forecast(self, series, horizon, seasonality, params, **kwargs):
            calls.append(
                {
                    "name": self.name,
                    "series_type": type(series).__name__,
                    "horizon": horizon,
                    "seasonality": seasonality,
                    "params": params,
                }
            )
            return ForecastResult(forecast=np.array([99.0], dtype=float), params_used={"method": self.name})

    class FakeRegistry:
        @staticmethod
        def get(name):
            return FakeMethod(name)

    monkeypatch.setattr(cl, "ForecastRegistry", FakeRegistry)

    naive_f, naive_p = cl.forecast_naive(np.array([1.0, 2.0]), fh=1)
    drift_f, drift_p = cl.forecast_drift(np.array([1.0, 2.0]), fh=1, n=5)
    seas_f, seas_p = cl.forecast_seasonal_naive(np.array([1.0, 2.0]), fh=1, m=12)
    theta_f, theta_p = cl.forecast_theta(np.array([1.0, 2.0]), fh=1, alpha=0.4)
    fou_f, fou_p = cl.forecast_fourier_ols(np.array([1.0, 2.0]), fh=1, m=12, K=2, trend=False)

    assert np.allclose(naive_f, [99.0])
    assert naive_p == {"method": "naive"}
    assert np.allclose(drift_f, [99.0])
    assert drift_p == {"method": "drift"}
    assert np.allclose(seas_f, [99.0])
    assert seas_p == {"method": "seasonal_naive"}
    assert np.allclose(theta_f, [99.0])
    assert theta_p == {"method": "theta"}
    assert np.allclose(fou_f, [99.0])
    assert fou_p == {"method": "fourier_ols"}

    assert [c["name"] for c in calls] == [
        "naive",
        "drift",
        "seasonal_naive",
        "theta",
        "fourier_ols",
    ]
    assert all(c["series_type"] == "Series" for c in calls)
    assert calls[0]["params"] == {}
    assert calls[1]["params"] == {}
    assert calls[2]["seasonality"] == 12
    assert calls[3]["params"] == {"alpha": 0.4}
    assert calls[4]["params"] == {"terms": 2, "trend": False}
