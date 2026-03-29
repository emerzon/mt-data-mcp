from __future__ import annotations

import numpy as np
import pandas as pd

from mtdata.forecast.interface import ForecastCallContext
from mtdata.forecast import forecast_engine as fe
from mtdata.forecast.methods import ensemble as em


def test_ensemble_bma_weights_remain_non_degenerate():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    def dispatch(method_name, series_in, horizon, seasonality, params):
        return np.array([1.0, 2.0], dtype=float) if method_name == "naive" else np.array([1.1, 2.1], dtype=float)

    def prepare_cv(*args, **kwargs):
        X_cv = np.array(
            [
                [1.0, 1.000001],
                [2.0, 2.000002],
                [3.0, 3.000003],
                [4.0, 4.000004],
            ],
            dtype=float,
        )
        y_cv = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        return X_cv, y_cv

    out = method.forecast(
        series,
        horizon=2,
        seasonality=1,
        params={"methods": ["naive", "theta"], "mode": "bma"},
        ensemble_dispatch_method=dispatch,
        prepare_ensemble_cv=prepare_cv,
        get_available_methods=lambda: ("naive", "theta"),
    )

    weights = out.metadata["weights"]
    assert weights[0] > 0.5
    assert weights[1] > 0.05


def test_ensemble_reports_component_failures():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    class Dispatch:
        def __call__(self, method_name, series_in, horizon, seasonality, params):
            if method_name == "bad":
                self._last_error = {"method": "bad", "error": "boom", "error_type": "RuntimeError"}
                return None
            self._last_error = None
            return np.array([5.0, 6.0], dtype=float)

    dispatch = Dispatch()

    out = method.forecast(
        series,
        horizon=2,
        seasonality=1,
        params={"methods": ["naive", "bad"], "mode": "average"},
        ensemble_dispatch_method=dispatch,
        get_available_methods=lambda: ("naive", "bad"),
    )

    assert np.allclose(out.forecast, [5.0, 6.0])
    assert out.metadata["component_failures"][0]["method"] == "bad"
    assert out.metadata["component_failures"][0]["error"] == "boom"


def test_ensemble_prepare_forecast_call_injects_engine_helpers():
    method = em.EnsembleMethod()
    context = ForecastCallContext(
        method="ensemble",
        symbol="EURUSD",
        timeframe="H1",
        quantity="price",
        horizon=2,
        seasonality=24,
        base_col="close",
        ci_alpha=0.05,
        as_of=None,
        denoise_spec_used=None,
        history_df=pd.DataFrame({"time": [1.0], "close": [100.0]}),
        target_series=pd.Series([100.0], name="close"),
        exog_used=None,
        future_exog=None,
    )

    params, kwargs = method.prepare_forecast_call({"methods": ["naive"]}, {}, context)

    assert params == {"methods": ["naive"]}
    assert kwargs["ensemble_dispatch_method"] is fe._ensemble_dispatch_method
    assert kwargs["prepare_ensemble_cv"] is fe._prepare_ensemble_cv
    assert kwargs["normalize_weights"] is fe._normalize_weights
    assert kwargs["get_available_methods"] is fe._get_available_methods
