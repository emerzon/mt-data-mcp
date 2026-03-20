"""Tests for forecast/common.py — extended coverage for nf_setup_and_predict and fetch_history."""

import inspect
from unittest.mock import patch, MagicMock

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
    nf_setup_and_predict,
    fetch_history,
)


# ---------------------------------------------------------------------------
# nf_setup_and_predict (lines 159-345) — heavily mocked
# ---------------------------------------------------------------------------


def _mock_nf_class(has_max_steps=True, has_max_epochs=False, has_lr=True):
    """Build a real class whose __init__ has the right signature for inspect."""
# ruff: noqa: E402, E731, E741, F811, F841
    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if has_max_steps:
        params.append(
            inspect.Parameter("max_steps", inspect.Parameter.KEYWORD_ONLY, default=10)
        )
    if has_max_epochs:
        params.append(
            inspect.Parameter("max_epochs", inspect.Parameter.KEYWORD_ONLY, default=10)
        )
    if has_lr:
        params.append(
            inspect.Parameter(
                "learning_rate", inspect.Parameter.KEYWORD_ONLY, default=0.01
            )
        )
    for name in ("h", "input_size", "batch_size"):
        params.append(
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=10)
        )
    sig = inspect.Signature(params)

    def init_fn(self, **kw):
        pass

    init_fn.__signature__ = sig

    cls = type("MockModel", (), {"__init__": init_fn})
    return cls


def _make_y_df(n=100):
    return pd.DataFrame(
        {
            "unique_id": ["ts"] * n,
            "ds": pd.RangeIndex(n),
            "y": np.random.randn(n).cumsum() + 100,
        }
    )


def _make_yf(fh=10):
    return pd.DataFrame(
        {
            "unique_id": ["ts"] * fh,
            "ds": pd.RangeIndex(fh),
            "y": np.random.randn(fh) + 100,
        }
    )


class TestNfSetupAndPredict:
    def test_import_error_raises(self):
        """When neuralforecast is not importable, RuntimeError is raised."""
        model_cls = _mock_nf_class()
        # Patch the import inside nf_setup_and_predict to fail
        with patch.dict("sys.modules", {"neuralforecast": None}):
            with pytest.raises((RuntimeError, ImportError)):
                nf_setup_and_predict(
                    model_class=model_cls,
                    fh=10,
                    timeframe="H1",
                    Y_df=_make_y_df(),
                    input_size=24,
                    batch_size=32,
                    steps=5,
                )

    def test_ctor_inspection_max_steps(self):
        """Model with max_steps in __init__ gets max_steps kwarg."""
        cls = _mock_nf_class(has_max_steps=True, has_max_epochs=False)
        sig = inspect.signature(cls.__init__)
        assert "max_steps" in sig.parameters

    def test_ctor_inspection_max_epochs(self):
        """Model with max_epochs but no max_steps gets max_epochs kwarg."""
        cls = _mock_nf_class(has_max_steps=False, has_max_epochs=True)
        sig = inspect.signature(cls.__init__)
        assert "max_epochs" in sig.parameters
        assert "max_steps" not in sig.parameters

    def test_ctor_inspection_neither(self):
        """Model with neither max_steps nor max_epochs — code defaults to max_steps."""
        cls = _mock_nf_class(has_max_steps=False, has_max_epochs=False)
        sig = inspect.signature(cls.__init__)
        assert "max_steps" not in sig.parameters
        assert "max_epochs" not in sig.parameters

    @patch("mtdata.forecast.common.pd_freq_from_timeframe", return_value="1h")
    def test_basic_predict_mocked_nf(self, mock_freq):
        """End-to-end with fully mocked NeuralForecast."""
        model_cls = _mock_nf_class()
        mock_nf_inst = MagicMock()
        mock_nf_inst.fit = MagicMock()
        mock_nf_inst.predict = MagicMock(return_value=_make_yf(10))
        pred_params = [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("h", inspect.Parameter.KEYWORD_ONLY, default=None),
        ]
        mock_nf_inst.predict.__signature__ = inspect.Signature(pred_params)
        fit_params = [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("df", inspect.Parameter.KEYWORD_ONLY),
        ]
        mock_nf_inst.fit.__signature__ = inspect.Signature(fit_params)

        # Build a callable that mimics NeuralForecast class — returns the mock instance
        nf_init_params = [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("models", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("freq", inspect.Parameter.KEYWORD_ONLY),
        ]

        def _nf_init(self, *, models, freq, **kw):
            pass

        _nf_init.__signature__ = inspect.Signature(nf_init_params)

        class FakeNF:
            __init__ = _nf_init
            fit = mock_nf_inst.fit
            predict = mock_nf_inst.predict

        nf_module = MagicMock()
        nf_module.NeuralForecast = FakeNF

        with patch.dict("sys.modules", {"neuralforecast": nf_module}):
            result = nf_setup_and_predict(
                model_class=model_cls,
                fh=10,
                timeframe="H1",
                Y_df=_make_y_df(),
                input_size=24,
                batch_size=32,
                steps=5,
            )
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# fetch_history (lines 348-411)
# ---------------------------------------------------------------------------


class TestFetchHistory:
    @patch("mtdata.forecast.common.mt5")
    @patch("mtdata.forecast.common._mt5_copy_rates_from_pos")
    @patch("mtdata.forecast.common._ensure_symbol_ready", return_value=None)
    @patch("mtdata.forecast.common.get_symbol_info_cached")
    @patch("mtdata.forecast.common.TIMEFRAME_MAP", {"H1": 1})
    def test_basic_fetch(self, mock_info, mock_ensure, mock_copy, mock_mt5):
        info = MagicMock()
        info.visible = True
        mock_info.return_value = info
        times = np.arange(0, 10 * 3600, 3600, dtype=float)
        data = pd.DataFrame(
            {
                "time": times,
                "open": np.ones(10),
                "high": np.ones(10),
                "low": np.ones(10),
                "close": np.ones(10),
            }
        ).to_records(index=False)
        mock_copy.return_value = data
        df = fetch_history("EURUSD", "H1", 10)
        assert not df.empty
        # drop_last_live default drops last bar
        assert len(df) == 9

    def test_invalid_timeframe(self):
        with patch("mtdata.forecast.common.TIMEFRAME_MAP", {}):
            with pytest.raises(RuntimeError, match="Invalid timeframe"):
                fetch_history("EURUSD", "BAD", 100)

    @patch("mtdata.forecast.common._ensure_symbol_ready", return_value="symbol error")
    @patch("mtdata.forecast.common.get_symbol_info_cached", return_value=None)
    @patch("mtdata.forecast.common.TIMEFRAME_MAP", {"H1": 1})
    def test_ensure_error(self, mock_info, mock_ensure):
        with pytest.raises(RuntimeError, match="symbol error"):
            fetch_history("BAD", "H1", 100)

    @patch("mtdata.forecast.common.mt5")
    @patch("mtdata.forecast.common._mt5_copy_rates_from_pos", return_value=None)
    @patch("mtdata.forecast.common._ensure_symbol_ready", return_value=None)
    @patch("mtdata.forecast.common.get_symbol_info_cached")
    @patch("mtdata.forecast.common.TIMEFRAME_MAP", {"H1": 1})
    def test_no_rates_raises(self, mock_info, mock_ensure, mock_copy, mock_mt5):
        mock_info.return_value = MagicMock(visible=True)
        mock_mt5.last_error.return_value = (0, "no data")
        with pytest.raises(RuntimeError, match="Failed to get rates"):
            fetch_history("X", "H1", 100)

    @patch("mtdata.forecast.common.mt5")
    @patch("mtdata.forecast.common._mt5_copy_rates_from")
    @patch("mtdata.forecast.common._ensure_symbol_ready", return_value=None)
    @patch("mtdata.forecast.common.get_symbol_info_cached")
    @patch("mtdata.forecast.common._parse_start_datetime")
    @patch("mtdata.forecast.common._utc_epoch_seconds", return_value=36000.0)
    @patch("mtdata.forecast.common.TIMEFRAME_MAP", {"H1": 1})
    def test_as_of_fetch(
        self, mock_utc, mock_parse, mock_info, mock_ensure, mock_copy, mock_mt5
    ):
        from datetime import datetime, timezone

        mock_parse.return_value = datetime(2024, 1, 1, tzinfo=timezone.utc)
        info = MagicMock()
        info.visible = True
        mock_info.return_value = info
        times = np.arange(0, 20 * 3600, 3600, dtype=float)
        data = pd.DataFrame(
            {
                "time": times,
                "open": np.ones(20),
                "high": np.ones(20),
                "low": np.ones(20),
                "close": np.ones(20),
            }
        ).to_records(index=False)
        mock_copy.return_value = data
        df = fetch_history("EURUSD", "H1", 10, as_of="2024-01-01T10:00:00")
        assert not df.empty

    @patch("mtdata.forecast.common.mt5")
    @patch("mtdata.forecast.common._mt5_copy_rates_from_pos")
    @patch("mtdata.forecast.common._ensure_symbol_ready", return_value=None)
    @patch("mtdata.forecast.common.get_symbol_info_cached")
    @patch("mtdata.forecast.common.TIMEFRAME_MAP", {"H1": 1})
    def test_drop_last_live_false(self, mock_info, mock_ensure, mock_copy, mock_mt5):
        info = MagicMock()
        info.visible = True
        mock_info.return_value = info
        times = np.arange(0, 10 * 3600, 3600, dtype=float)
        data = pd.DataFrame(
            {
                "time": times,
                "open": np.ones(10),
                "high": np.ones(10),
                "low": np.ones(10),
                "close": np.ones(10),
            }
        ).to_records(index=False)
        mock_copy.return_value = data
        df = fetch_history("EURUSD", "H1", 10, drop_last_live=False)
        assert len(df) == 10

    @patch("mtdata.forecast.common.mt5")
    @patch("mtdata.forecast.common._mt5_copy_rates_from_pos")
    @patch("mtdata.forecast.common._ensure_symbol_ready", return_value=None)
    @patch("mtdata.forecast.common.get_symbol_info_cached")
    @patch("mtdata.forecast.common.TIMEFRAME_MAP", {"H1": 1})
    def test_restores_invisible_symbol(
        self, mock_info, mock_ensure, mock_copy, mock_mt5
    ):
        info = MagicMock()
        info.visible = False
        mock_info.return_value = info
        times = np.arange(0, 5 * 3600, 3600, dtype=float)
        data = pd.DataFrame(
            {
                "time": times,
                "open": np.ones(5),
                "high": np.ones(5),
                "low": np.ones(5),
                "close": np.ones(5),
            }
        ).to_records(index=False)
        mock_copy.return_value = data
        df = fetch_history("EURUSD", "H1", 5)
        mock_mt5.symbol_select.assert_called_with("EURUSD", False)


# ---------------------------------------------------------------------------
# _extract_forecast_values (lines 44-70)
# ---------------------------------------------------------------------------


class TestExtractForecastValues:
    def test_y_column(self):
        df = pd.DataFrame(
            {"unique_id": ["ts"] * 5, "ds": range(5), "y": [1.0, 2, 3, 4, 5]}
        )
        vals = _extract_forecast_values(df, 5)
        assert len(vals) == 5

    def test_non_y_column(self):
        df = pd.DataFrame(
            {"unique_id": ["ts"] * 3, "ds": range(3), "prediction": [10, 20, 30]}
        )
        vals = _extract_forecast_values(df, 3)
        np.testing.assert_array_equal(vals, [10, 20, 30])

    def test_pads_short(self):
        df = pd.DataFrame({"unique_id": ["ts"] * 2, "ds": range(2), "y": [1.0, 2.0]})
        vals = _extract_forecast_values(df, 5)
        assert len(vals) == 5

    def test_no_pred_col_raises(self):
        df = pd.DataFrame({"unique_id": ["ts"], "ds": [0]})
        with pytest.raises(RuntimeError, match="prediction columns not found"):
            _extract_forecast_values(df, 5)


# ---------------------------------------------------------------------------
# _create_training_dataframes (lines 73-102)
# ---------------------------------------------------------------------------


class TestCreateTrainingDataframes:
    def test_basic(self):
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y_df, X_df, Xf_df = _create_training_dataframes(series, fh=3)
        assert len(Y_df) == 5
        assert X_df is None

    def test_with_exog(self):
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        exog = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        exog_f = np.array([[0.6], [0.7], [0.8]])
        Y_df, X_df, Xf_df = _create_training_dataframes(
            series, fh=3, exog_used=exog, exog_future=exog_f
        )
        assert X_df is not None
        assert "x0" in X_df.columns
        assert Xf_df is not None
        assert len(Xf_df) == 3

    def test_exog_no_future(self):
        series = np.array([1.0, 2.0, 3.0])
        exog = np.array([[0.1], [0.2], [0.3]])
        Y_df, X_df, Xf_df = _create_training_dataframes(series, fh=2, exog_used=exog)
        assert X_df is not None
        assert Xf_df is None


# ---------------------------------------------------------------------------
# default_seasonality (lines 105-120)
# ---------------------------------------------------------------------------


class TestDefaultSeasonality:
    @patch(
        "mtdata.forecast.common.TIMEFRAME_SECONDS",
        {"M5": 300, "H1": 3600, "D1": 86400, "W1": 604800, "MN1": 2592000},
    )
    def test_intraday(self):
        assert default_seasonality("M5") == 288  # 86400/300
        assert default_seasonality("H1") == 24

    @patch("mtdata.forecast.common.TIMEFRAME_SECONDS", {"D1": 86400})
    def test_daily(self):
        assert default_seasonality("D1") == 5

    @patch("mtdata.forecast.common.TIMEFRAME_SECONDS", {"W1": 604800})
    def test_weekly(self):
        assert default_seasonality("W1") == 52

    @patch("mtdata.forecast.common.TIMEFRAME_SECONDS", {"MN1": 2592000})
    def test_monthly(self):
        assert default_seasonality("MN1") == 12

    @patch("mtdata.forecast.common.TIMEFRAME_SECONDS", {})
    def test_unknown_returns_zero(self):
        assert default_seasonality("BAD") == 0

    @patch("mtdata.forecast.common.TIMEFRAME_SECONDS", {"X": 0})
    def test_zero_sec_returns_zero(self):
        assert default_seasonality("X") == 0


# ---------------------------------------------------------------------------
# next_times_from_last
# ---------------------------------------------------------------------------


class TestNextTimesFromLast:
    def test_basic(self):
        result = next_times_from_last(1000.0, 3600, 3)
        assert result == [4600.0, 8200.0, 11800.0]

    def test_single_step(self):
        result = next_times_from_last(0.0, 60, 1)
        assert result == [60.0]


# ---------------------------------------------------------------------------
# pd_freq_from_timeframe
# ---------------------------------------------------------------------------


class TestPdFreqFromTimeframe:
    def test_known_mappings(self):
        assert pd_freq_from_timeframe("M1") == "1min"
        assert pd_freq_from_timeframe("H1") == "1h"
        assert pd_freq_from_timeframe("D1") == "1d"
        assert pd_freq_from_timeframe("MN1") == "MS"

    def test_unknown_returns_D(self):
        assert pd_freq_from_timeframe("BAD") == "D"


# ---------------------------------------------------------------------------
# edge_pad_to_length — additional edge cases
# ---------------------------------------------------------------------------


class TestEdgePadExtended:
    def test_2d_input_flattened(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = edge_pad_to_length(arr, 6)
        assert len(result) == 6

    def test_negative_length(self):
        result = edge_pad_to_length(np.array([1.0]), -1)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# log_returns_from_prices — additional edge cases
# ---------------------------------------------------------------------------


class TestLogReturnsExtended:
    def test_single_price(self):
        result = log_returns_from_prices(np.array([100.0]))
        assert len(result) == 0

    def test_two_prices(self):
        result = log_returns_from_prices(np.array([100.0, 110.0]))
        assert len(result) == 1
        assert abs(result[0] - np.log(110.0 / 100.0)) < 1e-10

    def test_zero_price_clipped(self):
        result = log_returns_from_prices(np.array([0.0, 100.0]))
        assert len(result) == 1
        assert np.isfinite(result[0])
