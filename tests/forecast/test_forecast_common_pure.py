"""Tests for src/mtdata/forecast/common.py — pure forecast utility functions."""
import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.common import (
    edge_pad_to_length,
    log_returns_from_prices,
    _extract_forecast_values,
    _create_training_dataframes,
    default_seasonality,
    next_times_from_last,
    pd_freq_from_timeframe,
)


class TestEdgePadToLength:
    def test_exact_length(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = edge_pad_to_length(arr, 3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_truncate(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = edge_pad_to_length(arr, 3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_pad(self):
        arr = np.array([1.0, 2.0])
        result = edge_pad_to_length(arr, 5)
        assert len(result) == 5
        assert result[-1] == 2.0  # edge-padded with last value

    def test_empty_input(self):
        arr = np.array([])
        result = edge_pad_to_length(arr, 3)
        assert len(result) == 3
        assert all(np.isnan(result))

    def test_zero_length(self):
        arr = np.array([1.0, 2.0])
        result = edge_pad_to_length(arr, 0)
        assert len(result) == 0


class TestLogReturnsFromPrices:
    def test_basic(self):
        prices = np.array([100.0, 110.0, 121.0])
        result = log_returns_from_prices(prices)
        assert len(result) == 2
        assert abs(result[0] - np.log(1.1)) < 1e-10

    def test_single_price(self):
        result = log_returns_from_prices(np.array([100.0]))
        assert len(result) == 0

    def test_empty(self):
        result = log_returns_from_prices(np.array([]))
        assert len(result) == 0

    def test_near_zero_clipped(self):
        prices = np.array([0.0, 1.0])
        result = log_returns_from_prices(prices)
        assert len(result) == 1
        assert np.isfinite(result[0])


class TestExtractForecastValues:
    def test_y_column(self):
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        result = _extract_forecast_values(df, 3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_non_y_column(self):
        df = pd.DataFrame({"unique_id": ["ts", "ts"], "ds": [0, 1], "pred": [10.0, 20.0]})
        result = _extract_forecast_values(df, 2)
        np.testing.assert_array_equal(result, [10.0, 20.0])

    def test_pads_if_short(self):
        df = pd.DataFrame({"y": [1.0, 2.0]})
        result = _extract_forecast_values(df, 5)
        assert len(result) == 5
        assert result[-1] == 2.0  # edge-padded

    def test_no_pred_col_raises(self):
        df = pd.DataFrame({"unique_id": ["ts"], "ds": [0]})
        with pytest.raises(RuntimeError):
            _extract_forecast_values(df, 1)


class TestCreateTrainingDataframes:
    def test_basic(self):
        series = np.array([1.0, 2.0, 3.0, 4.0])
        Y_df, X_df, Xf_df = _create_training_dataframes(series, 2)
        assert len(Y_df) == 4
        assert list(Y_df.columns) == ["unique_id", "ds", "y"]
        assert X_df is None
        assert Xf_df is None

    def test_with_exog(self):
        series = np.array([1.0, 2.0, 3.0])
        exog = np.array([[0.1], [0.2], [0.3]])
        exog_future = np.array([[0.4], [0.5]])
        Y_df, X_df, Xf_df = _create_training_dataframes(series, 2, exog, exog_future)
        assert X_df is not None
        assert "x0" in X_df.columns
        assert Xf_df is not None
        assert len(Xf_df) == 2


class TestDefaultSeasonality:
    def test_h1(self):
        result = default_seasonality("H1")
        assert result == 24  # 86400 / 3600

    def test_m15(self):
        result = default_seasonality("M15")
        assert result == 96  # 86400 / 900

    def test_d1(self):
        assert default_seasonality("D1") == 5

    def test_w1(self):
        assert default_seasonality("W1") == 52

    def test_mn1(self):
        assert default_seasonality("MN1") == 12

    def test_unknown(self):
        assert default_seasonality("UNKNOWN") == 0


class TestNextTimesFromLast:
    def test_basic(self):
        result = next_times_from_last(1000.0, 60, 3)
        assert result == [1060.0, 1120.0, 1180.0]

    def test_single_step(self):
        result = next_times_from_last(0.0, 3600, 1)
        assert result == [3600.0]


class TestPdFreqFromTimeframe:
    def test_m1(self):
        assert pd_freq_from_timeframe("M1") == "1min"

    def test_h1(self):
        assert pd_freq_from_timeframe("H1") == "1h"

    def test_d1(self):
        assert pd_freq_from_timeframe("D1") == "1d"

    def test_w1(self):
        assert pd_freq_from_timeframe("W1") == "1w"

    def test_mn1(self):
        assert pd_freq_from_timeframe("MN1") == "MS"

    def test_unknown(self):
        assert pd_freq_from_timeframe("UNKNOWN") == "D"

    def test_case_insensitive(self):
        assert pd_freq_from_timeframe("h1") == "1h"
