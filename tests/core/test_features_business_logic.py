from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
import types

from mtdata.core.features import extract_rolling_features


@pytest.mark.parametrize("window_size", [0, -1, 2.5, True])
def test_extract_rolling_features_rejects_invalid_window_size(window_size):
    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        extract_rolling_features(np.array([1.0, 2.0, 3.0]), window_size=window_size)


def test_extract_rolling_features_returns_empty_for_short_series(monkeypatch):
    fake = types.SimpleNamespace(extract_features=lambda *args, **kwargs: pd.DataFrame())
    feature_extraction = types.SimpleNamespace(
        EfficientFCParameters=lambda: {"eff": None},
        MinimalFCParameters=lambda: {"min": None},
    )
    dataframe_functions = types.SimpleNamespace(
        roll_time_series=lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setitem(sys.modules, "tsfresh", fake)
    monkeypatch.setitem(sys.modules, "tsfresh.feature_extraction", feature_extraction)
    monkeypatch.setitem(sys.modules, "tsfresh.utilities.dataframe_functions", dataframe_functions)

    out = extract_rolling_features(np.array([1.0, 2.0, 3.0]), window_size=5)

    assert isinstance(out, pd.DataFrame)
    assert out.empty
