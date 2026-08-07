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


def test_minimal_rolling_features_match_tsfresh_reference_values():
    series = np.arange(30, dtype=float) + np.sin(np.arange(30, dtype=float))

    out = extract_rolling_features(series, window_size=10, minimal=True)

    assert out.shape == (30, 9)
    assert out.iloc[:9].isna().all(axis=None)
    expected = {
        "value__variance": 8.884218201371812,
        "value__autocorrelation__lag_1": 0.7481041656494541,
        "value__autocorrelation__lag_3": 0.1405876012176145,
        "value__approximate_entropy__m_2__r_0.5": 0.2404008900485699,
        'value__linear_trend__attr_"slope"': 1.0122363318861707,
        'value__linear_trend__attr_"stderr"': 0.08081811516396234,
        "value__mean_abs_change": 1.0457909428046397,
        "value__skewness": 0.32050226884165134,
        "value__kurtosis": -1.0110468216353956,
    }
    for column, value in expected.items():
        assert out.loc[9, column] == pytest.approx(value, rel=1e-10, abs=1e-12)


def test_minimal_rolling_features_do_not_require_tsfresh(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tsfresh" or name.startswith("tsfresh."):
            raise AssertionError("minimal feature extraction imported tsfresh")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    out = extract_rolling_features(np.arange(25, dtype=float), window_size=5)

    assert out.shape == (25, 9)
