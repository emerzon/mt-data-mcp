from __future__ import annotations

import numpy as np
import pandas as pd

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
