from __future__ import annotations

import sys
from types import ModuleType

import pandas as pd
import pytest

from mtdata.forecast import common as common_mod
from mtdata.forecast.methods import statsforecast as sfm


class _DummyStatsMethod(sfm.StatsForecastMethod):
    @property
    def name(self) -> str:
        return "dummy_stats"

    def _get_model(self, seasonality: int, params):
        return {"model": "dummy", "seasonality": seasonality}


def test_statsforecast_forecast_rejects_invalid_ci_alpha(monkeypatch):
    fake_stats_mod = ModuleType("statsforecast")
    fake_stats_mod.StatsForecast = object
    monkeypatch.setitem(sys.modules, "statsforecast", fake_stats_mod)
    monkeypatch.setattr(
        common_mod,
        "_create_training_dataframes",
        lambda *args, **kwargs: (pd.DataFrame({"y": [1.0]}), None, None),
    )

    with pytest.raises(ValueError, match="ci_alpha must be between 0 and 1"):
        _DummyStatsMethod().forecast(
            pd.Series([1.0, 2.0, 3.0]),
            horizon=1,
            seasonality=1,
            params={},
            ci_alpha=1.0,
        )


def test_statsforecast_forecast_requires_unique_id_rows(monkeypatch):
    class FakeStatsForecast:
        def __init__(self, models, freq):
            pass

        def fit(self, Y_df, X_df=None):
            return None

        def predict(self, h, X_df=None, level=None):
            return pd.DataFrame({"dummy_stats": [1.0]})

    fake_stats_mod = ModuleType("statsforecast")
    fake_stats_mod.StatsForecast = FakeStatsForecast
    monkeypatch.setitem(sys.modules, "statsforecast", fake_stats_mod)
    monkeypatch.setattr(
        common_mod,
        "_create_training_dataframes",
        lambda *args, **kwargs: (pd.DataFrame({"y": [1.0]}), None, None),
    )

    with pytest.raises(
        RuntimeError,
        match="StatsForecast dummy_stats error: StatsForecast output missing unique_id column",
    ):
        _DummyStatsMethod().forecast(
            pd.Series([1.0, 2.0, 3.0]), horizon=1, seasonality=1, params={}
        )
