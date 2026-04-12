from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import forecast_engine as fe
from mtdata.forecast.interface import ForecastCallContext
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


def test_ensemble_bma_keeps_valid_members_when_some_cv_rmse_are_invalid():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    def dispatch(method_name, series_in, horizon, seasonality, params):
        if method_name == "naive":
            return np.array([10.0, 20.0], dtype=float)
        return np.array([1.0, 2.0], dtype=float)

    def prepare_cv(*args, **kwargs):
        X_cv = np.array(
            [
                [1.0, np.nan],
                [2.0, np.nan],
                [3.0, np.nan],
            ],
            dtype=float,
        )
        y_cv = np.array([1.0, 2.0, 3.0], dtype=float)
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

    assert out.metadata["mode_used"] == "bma"
    assert out.metadata["weights"] == pytest.approx([1.0, 0.0])
    assert np.allclose(out.forecast, [10.0, 20.0])


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


def test_ensemble_stacking_reports_normalized_weights_and_raw_coefficients():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    def dispatch(method_name, series_in, horizon, seasonality, params):
        if method_name == "naive":
            return np.array([10.0, 20.0], dtype=float)
        return np.array([1.0, 2.0], dtype=float)

    def prepare_cv(*args, **kwargs):
        return np.ones((3, 2), dtype=float), np.ones(3, dtype=float)

    with patch.object(
        em.np.linalg,
        "lstsq",
        return_value=(np.array([0.5, 2.0, 1.0], dtype=float), np.array([]), 2, np.array([])),
    ):
        out = method.forecast(
            series,
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "theta"], "mode": "stacking"},
            ensemble_dispatch_method=dispatch,
            prepare_ensemble_cv=prepare_cv,
            get_available_methods=lambda: ("naive", "theta"),
        )

    assert np.allclose(out.forecast, [21.5, 42.5])
    assert out.metadata["weights"] == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert out.metadata["coefficients"] == [2.0, 1.0]
    assert out.metadata["weight_semantics"] == "normalized_coefficients"
    assert out.metadata["intercept"] == 0.5


def test_ensemble_stacking_falls_back_to_raw_coefficients_for_non_positive_sum():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    def dispatch(method_name, series_in, horizon, seasonality, params):
        if method_name == "naive":
            return np.array([10.0, 20.0], dtype=float)
        return np.array([1.0, 2.0], dtype=float)

    def prepare_cv(*args, **kwargs):
        return np.ones((3, 2), dtype=float), np.ones(3, dtype=float)

    with patch.object(
        em.np.linalg,
        "lstsq",
        return_value=(np.array([0.5, 1.0, -1.0], dtype=float), np.array([]), 2, np.array([])),
    ):
        out = method.forecast(
            series,
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "theta"], "mode": "stacking"},
            ensemble_dispatch_method=dispatch,
            prepare_ensemble_cv=prepare_cv,
            get_available_methods=lambda: ("naive", "theta"),
        )

    assert np.allclose(out.forecast, [9.5, 18.5])
    assert out.metadata["weights"] == [1.0, -1.0]
    assert out.metadata["coefficients"] == [1.0, -1.0]
    assert out.metadata["weight_semantics"] == "raw_coefficients"
    assert out.metadata["intercept"] == 0.5


def test_ensemble_reports_component_failures_from_dispatch_exceptions():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    def dispatch(method_name, series_in, horizon, seasonality, params):
        if method_name == "bad":
            raise RuntimeError("boom")
        return np.array([5.0, 6.0], dtype=float)

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
    assert out.metadata["component_failures"][0]["error_type"] == "RuntimeError"


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
    assert kwargs["ensemble_dispatch_with_error"] is fe._ensemble_dispatch_with_error
    assert kwargs["prepare_ensemble_cv"] is fe._prepare_ensemble_cv
    assert kwargs["normalize_weights"] is fe._normalize_weights
    assert kwargs["get_available_methods"] is fe._get_available_methods


def test_ensemble_default_normalize_weights_reuses_shared_helper():
    assert em._normalize_weights_default is fe._normalize_weights


def test_ensemble_params_applied_metadata():
    """Verify component-specific params are tracked in metadata."""
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    captured_params = {}

    def dispatch(method_name, series_in, horizon, seasonality, params):
        captured_params[method_name] = params
        return np.array([5.0, 6.0], dtype=float)

    def dispatch_with_error(method_name, series_in, horizon, seasonality, params):
        captured_params[method_name] = params
        return np.array([5.0, 6.0], dtype=float), None

    out = method.forecast(
        series,
        horizon=2,
        seasonality=1,
        params={
            "methods": ["naive", "theta"],
            "mode": "average",
            "method_params": {"naive": {"alpha": 0.5}, "theta": {}},
        },
        ensemble_dispatch_method=dispatch,
        ensemble_dispatch_with_error=dispatch_with_error,
        get_available_methods=lambda: ("naive", "theta"),
    )

    # naive had non-empty params → should appear in params_applied
    assert "params_applied" in out.metadata
    assert out.metadata["params_applied"]["naive"] == {"alpha": 0.5}
    # theta had empty dict → should NOT appear
    assert "theta" not in out.metadata["params_applied"]
    # verify dispatch actually received the params
    assert captured_params["naive"] == {"alpha": 0.5}
    assert captured_params["theta"] == {}


def test_component_dispatch_fn_type_alias_exists():
    """Verify the canonical dispatch type alias is importable."""
    assert hasattr(em, "ComponentDispatchFn")
    # Should be a typing alias (callable type)
    assert em.ComponentDispatchFn is not None
