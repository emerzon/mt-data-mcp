"""Regression tests for trainable artifacts refreshed with live history."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from mtdata.forecast.interface import ForecastResult
from mtdata.forecast.methods.mlforecast import MLForecastMethod
from mtdata.forecast.methods.sktime import SktimeMethod
from mtdata.forecast.methods.statsforecast import StatsForecastMethod


class _MlMethod(MLForecastMethod):
    @property
    def name(self) -> str:
        return "ml_test"

    def _get_model(self, params: Dict[str, Any]) -> Any:
        raise NotImplementedError


class _StatsMethod(StatsForecastMethod):
    @property
    def name(self) -> str:
        return "stats_test"

    def _get_model(self, seasonality: int, params: Dict[str, Any]) -> Any:
        raise NotImplementedError


class _SktimeMethod(SktimeMethod):
    @property
    def name(self) -> str:
        return "sktime_test"

    def _get_estimator(self, seasonality: int, params: Dict[str, Any]) -> Any:
        raise NotImplementedError


class _MlArtifact:
    def __init__(self) -> None:
        self.call: Dict[str, Any] | None = None

    def predict(self, **kwargs: Any) -> pd.DataFrame:
        self.call = kwargs
        return pd.DataFrame({"unique_id": ["ts", "ts"], "ml_test": [4.0, 5.0]})


class _StatsArtifact:
    def __init__(self) -> None:
        self.call: Dict[str, Any] | None = None

    def forecast(self, **kwargs: Any) -> pd.DataFrame:
        self.call = kwargs
        return pd.DataFrame({"unique_id": ["ts", "ts"], "stats_test": [4.0, 5.0]})


class _SktimeArtifact:
    cutoff = 1

    def __init__(self) -> None:
        self.updated: pd.Series | None = None

    def update(self, *, y: pd.Series, X: Any, update_params: bool) -> None:
        assert X is None
        assert update_params is False
        self.updated = y.copy()

    def predict(self, *, fh: np.ndarray) -> np.ndarray:
        return np.asarray([4.0, 5.0])


def test_mlforecast_predict_refreshes_fitted_history() -> None:
    artifact = _MlArtifact()
    series = pd.Series([1.0, 2.0, 3.0])

    result = _MlMethod().predict_with_model(artifact, series, 2, 1, {})

    assert isinstance(result, ForecastResult)
    assert artifact.call is not None
    assert artifact.call["new_df"]["y"].tolist() == [1.0, 2.0, 3.0]


def test_statsforecast_predict_refreshes_fitted_history() -> None:
    artifact = _StatsArtifact()
    series = pd.Series([1.0, 2.0, 3.0])

    result = _StatsMethod().predict_with_model(artifact, series, 2, 1, {})

    assert isinstance(result, ForecastResult)
    assert artifact.call is not None
    assert artifact.call["df"]["y"].tolist() == [1.0, 2.0, 3.0]


def test_sktime_predict_updates_only_new_history(monkeypatch) -> None:
    from mtdata.forecast.methods import sktime

    monkeypatch.setattr(sktime, "_HAS_SKTIME", True)
    artifact = _SktimeArtifact()
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = _SktimeMethod().predict_with_model(artifact, series, 2, 1, {})

    assert isinstance(result, ForecastResult)
    assert artifact.updated is not None
    assert artifact.updated.tolist() == [3.0, 4.0]
