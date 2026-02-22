from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import common as common_mod
from mtdata.forecast.interface import ForecastResult
from mtdata.forecast.methods import mlforecast as mlm


class _DummyMLMethod(mlm.MLForecastMethod):
    @property
    def name(self) -> str:
        return "dummy_ml"

    def _get_model(self, params):
        return {"model": "dummy"}


def test_mlforecast_base_metadata_on_concrete_method():
    method = _DummyMLMethod()
    assert method.name == "dummy_ml"
    assert method.category == "machine_learning"
    assert method.required_packages == ["mlforecast"]
    assert method.supports_features == {
        "price": True,
        "return": True,
        "volatility": True,
        "ci": False,
    }


def test_mlforecast_forecast_raises_runtime_on_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlforecast", ModuleType("mlforecast"))
    with pytest.raises(RuntimeError, match="Failed to import mlforecast"):
        _DummyMLMethod().forecast(pd.Series([1.0, 2.0, 3.0]), horizon=1, seasonality=0, params={})


def test_mlforecast_forecast_with_default_lags_exog_and_internal_param_filter(monkeypatch):
    calls = {}

    class FakeMLForecast:
        def __init__(self, models, freq, lags):
            calls["init"] = {"models": models, "freq": freq, "lags": lags}

        def fit(self, Y_df, X_df=None):
            calls["fit"] = {"Y_df": Y_df, "X_df": X_df}

        def predict(self, h, X_df=None):
            calls["predict"] = {"h": h, "X_df": X_df}
            return pd.DataFrame(
                {
                    "unique_id": ["ts", "other", "ts"],
                    "dummy_ml": [1.0, 100.0, 2.0],
                }
            )

    fake_ml_mod = ModuleType("mlforecast")
    fake_ml_mod.MLForecast = FakeMLForecast
    monkeypatch.setitem(sys.modules, "mlforecast", fake_ml_mod)

    Y_df = pd.DataFrame({"unique_id": ["ts"], "ds": [1], "y": [1.0]})
    X_df = pd.DataFrame({"unique_id": ["ts"], "ds": [1], "x1": [9.0]})
    Xf_df = pd.DataFrame({"unique_id": ["ts"], "ds": [2], "x1": [10.0]})
    monkeypatch.setattr(common_mod, "_create_training_dataframes", lambda *args, **kwargs: (Y_df, X_df, Xf_df))
    monkeypatch.setattr(
        common_mod,
        "_extract_forecast_values",
        lambda Yf, horizon, method_name: np.array([11.0, 12.0, 13.0], dtype=float),
    )

    exog_used = pd.DataFrame({"x1": [0.1, 0.2, 0.3]})
    exog_future = pd.DataFrame({"x1": [0.4, 0.5, 0.6]})
    out = _DummyMLMethod().forecast(
        pd.Series([1.0, 2.0, 3.0]),
        horizon=3,
        seasonality=5,
        params={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "as_of": "2024-01-01",
            "custom": 7,
        },
        exog_future=exog_future,
        exog_used=exog_used,
    )

    assert np.allclose(out.forecast, [11.0, 12.0, 13.0])
    assert out.ci_values is None
    assert out.params_used == {"custom": 7, "lags": [1, 2, 3, 4, 5]}
    assert calls["init"]["freq"] == 1
    assert calls["init"]["lags"] == [1, 2, 3, 4, 5]
    assert calls["fit"]["X_df"] is X_df
    assert calls["predict"] == {"h": 3, "X_df": Xf_df}


def test_mlforecast_forecast_without_exog_uses_simple_fit_and_predict(monkeypatch):
    calls = {}

    class FakeMLForecast:
        def __init__(self, models, freq, lags):
            calls["lags"] = lags

        def fit(self, Y_df, X_df=None):
            calls["fit_x"] = X_df

        def predict(self, h, X_df=None):
            calls["predict_x"] = X_df
            return pd.DataFrame({"dummy_ml": [9.0]})

    fake_ml_mod = ModuleType("mlforecast")
    fake_ml_mod.MLForecast = FakeMLForecast
    monkeypatch.setitem(sys.modules, "mlforecast", fake_ml_mod)

    monkeypatch.setattr(
        common_mod,
        "_create_training_dataframes",
        lambda *args, **kwargs: (pd.DataFrame({"y": [1.0]}), None, None),
    )
    monkeypatch.setattr(common_mod, "_extract_forecast_values", lambda Yf, horizon, method_name: np.array([9.0], dtype=float))

    out = _DummyMLMethod().forecast(
        pd.Series([1.0, 2.0]),
        horizon=1,
        seasonality=0,
        params={"lags": [2, 4], "exog_future": "internal"},
    )

    assert np.allclose(out.forecast, [9.0])
    assert out.params_used == {"lags": [2, 4]}
    assert calls["lags"] == [2, 4]
    assert calls["fit_x"] is None
    assert calls["predict_x"] is None


def test_mlforecast_forecast_wraps_runtime_errors(monkeypatch):
    class FakeMLForecast:
        def __init__(self, models, freq, lags):
            pass

        def fit(self, Y_df, X_df=None):
            raise ValueError("fit exploded")

    fake_ml_mod = ModuleType("mlforecast")
    fake_ml_mod.MLForecast = FakeMLForecast
    monkeypatch.setitem(sys.modules, "mlforecast", fake_ml_mod)
    monkeypatch.setattr(
        common_mod,
        "_create_training_dataframes",
        lambda *args, **kwargs: (pd.DataFrame({"y": [1.0]}), None, None),
    )

    with pytest.raises(RuntimeError, match="dummy_ml error: fit exploded"):
        _DummyMLMethod().forecast(pd.Series([1.0, 2.0]), horizon=1, seasonality=0, params={})


def test_mlforecast_random_forest_get_model(monkeypatch):
    captured = {}
    sklearn_mod = ModuleType("sklearn")
    ensemble_mod = ModuleType("sklearn.ensemble")

    class FakeRF:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    ensemble_mod.RandomForestRegressor = FakeRF
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_mod)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", ensemble_mod)

    model = mlm.MLFRandomForest()._get_model({"n_estimators": "55", "max_depth": 4})
    assert isinstance(model, FakeRF)
    assert captured["kwargs"] == {"n_estimators": 55, "max_depth": 4, "random_state": 42}
    assert mlm.MLFRandomForest().required_packages == ["mlforecast", "scikit-learn"]


def test_mlforecast_lightgbm_get_model(monkeypatch):
    captured = {}
    lgbm_mod = ModuleType("lightgbm")

    class FakeLGBM:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    lgbm_mod.LGBMRegressor = FakeLGBM
    monkeypatch.setitem(sys.modules, "lightgbm", lgbm_mod)

    model = mlm.MLFLightGBM()._get_model(
        {"n_estimators": "10", "learning_rate": "0.1", "num_leaves": "16", "max_depth": "6"}
    )
    assert isinstance(model, FakeLGBM)
    assert captured["kwargs"] == {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "num_leaves": 16,
        "max_depth": 6,
        "random_state": 42,
    }
    assert mlm.MLFLightGBM().required_packages == ["mlforecast", "lightgbm"]


def test_generic_mlforecast_model_import_validation_and_param_filtering(monkeypatch):
    method = mlm.GenericMLForecastMethod()
    with pytest.raises(ValueError, match="requires 'model'"):
        method._get_model({})

    with pytest.raises(ValueError, match="Could not import ML model"):
        method._get_model({"model": "missing.module.Class"})

    fake_mod = ModuleType("fake_models")

    class FakeModel:
        def __init__(self, a, b=2):
            self.a = a
            self.b = b

    fake_mod.FakeModel = FakeModel
    monkeypatch.setitem(sys.modules, "fake_models", fake_mod)

    model = method._get_model({"model": "fake_models.FakeModel", "a": 5, "b": 6, "c": 7})
    assert isinstance(model, FakeModel)
    assert model.a == 5
    assert model.b == 6


def test_mlforecast_legacy_wrappers_route_to_registry(monkeypatch):
    calls = []

    class FakeMethod:
        def __init__(self, name):
            self._name = name

        def forecast(self, series, horizon, seasonality, params, **kwargs):
            calls.append(
                {
                    "name": self._name,
                    "series_type": type(series).__name__,
                    "horizon": horizon,
                    "seasonality": seasonality,
                    "params": params,
                    "kwargs": kwargs,
                }
            )
            return ForecastResult(forecast=np.array([5.0, 6.0], dtype=float), params_used={"method": self._name})

    class FakeRegistry:
        @staticmethod
        def get(name):
            return FakeMethod(name)

    monkeypatch.setattr(mlm, "ForecastRegistry", FakeRegistry)
    exog_used = np.array([[1.0], [2.0]], dtype=float)
    exog_future = np.array([[3.0], [4.0]], dtype=float)

    rf_f, rf_params = mlm.forecast_mlf_rf(
        series=np.array([1.0, 2.0, 3.0], dtype=float),
        fh=2,
        timeframe="H1",
        lags=[1, 2],
        rolling_agg="mean",
        exog_used=exog_used,
        exog_future=exog_future,
    )
    lgbm_f, lgbm_params = mlm.forecast_mlf_lightgbm(
        series=np.array([1.0, 2.0, 3.0], dtype=float),
        fh=2,
        timeframe="H1",
        lags=[3],
        rolling_agg="std",
        n_estimators=10,
        learning_rate=0.1,
        num_leaves=11,
        max_depth=4,
        exog_used=exog_used,
        exog_future=exog_future,
    )

    assert np.allclose(rf_f, [5.0, 6.0])
    assert rf_params == {"method": "mlf_rf"}
    assert np.allclose(lgbm_f, [5.0, 6.0])
    assert lgbm_params == {"method": "mlf_lightgbm"}

    assert [c["name"] for c in calls] == ["mlf_rf", "mlf_lightgbm"]
    assert all(c["series_type"] == "Series" for c in calls)
    assert calls[0]["params"] == {"lags": [1, 2], "rolling_agg": "mean"}
    assert calls[1]["params"] == {
        "lags": [3],
        "rolling_agg": "std",
        "n_estimators": 10,
        "learning_rate": 0.1,
        "num_leaves": 11,
        "max_depth": 4,
    }
    assert np.array_equal(calls[0]["kwargs"]["exog_used"], exog_used)
    assert np.array_equal(calls[1]["kwargs"]["exog_future"], exog_future)
