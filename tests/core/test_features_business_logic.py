from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from mtdata.core.features import extract_rolling_features


@pytest.mark.parametrize("window_size", [0, -1, 2.5, True])
def test_extract_rolling_features_rejects_invalid_window_size(window_size):
    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        extract_rolling_features(np.array([1.0, 2.0, 3.0]), window_size=window_size)


def test_extract_rolling_features_returns_empty_for_short_series(monkeypatch):
    tsfresh = types.ModuleType("tsfresh")
    tsfresh.__path__ = []
    tsfresh.extract_features = lambda *args, **kwargs: pd.DataFrame()

    feature_extraction = types.ModuleType("tsfresh.feature_extraction")
    feature_extraction.EfficientFCParameters = lambda: {"eff": None}
    feature_extraction.MinimalFCParameters = lambda: {"min": None}

    utilities = types.ModuleType("tsfresh.utilities")
    utilities.__path__ = []

    dataframe_functions = types.ModuleType("tsfresh.utilities.dataframe_functions")
    dataframe_functions.roll_time_series = lambda *args, **kwargs: pd.DataFrame()

    monkeypatch.setitem(sys.modules, "tsfresh", tsfresh)
    monkeypatch.setitem(sys.modules, "tsfresh.feature_extraction", feature_extraction)
    monkeypatch.setitem(sys.modules, "tsfresh.utilities", utilities)
    monkeypatch.setitem(sys.modules, "tsfresh.utilities.dataframe_functions", dataframe_functions)

    out = extract_rolling_features(np.array([1.0, 2.0, 3.0]), window_size=5)

    assert isinstance(out, pd.DataFrame)
    assert out.empty
