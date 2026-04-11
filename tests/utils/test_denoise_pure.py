"""Comprehensive tests for mtdata.utils.denoise module."""

import numpy as np
import pandas as pd
import pytest

from mtdata.utils.denoise.api import _run_denoise_handler
from mtdata.utils.denoise import (
    _apply_denoise,
    _denoise_series,
    _resolve_denoise_base_col,
    denoise_list_methods,
    get_denoise_methods_data,
    normalize_denoise_spec,
)
from mtdata.utils.denoise.filters.adaptive import (
    _adaptive_lms_filter,
    _adaptive_rls_filter,
)
from mtdata.utils.denoise.filters.decomposition import _ssa_denoise, _vmd_denoise
from mtdata.utils.denoise.filters.specialized import (
    _bilateral_filter_1d,
    _hampel_filter,
    _kalman_filter_1d,
    _tv_denoise_1d,
)
from mtdata.utils.denoise.filters.spectral import _butterworth_filter
from mtdata.utils.denoise.filters.trend import (
    _beta_irls_mean,
    _beta_smooth,
    _hp_filter,
    _l1_trend_filter,
    _soft_threshold,
    _whittaker_smooth,
)
from mtdata.utils.denoise.filters.wavelet import _wavelet_packet_denoise

# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------

N = 200
_t = np.linspace(0, 4 * np.pi, N)
_rng = np.random.RandomState(42)
NOISY_SIGNAL = np.sin(_t) + _rng.normal(0, 0.3, N)

# A random-walk for filters that work better with trending data
_rng2 = np.random.RandomState(42)
RANDOM_WALK = np.cumsum(_rng2.normal(0, 1, N))


def _smoothness(x: np.ndarray) -> float:
    """Variance of first differences – lower means smoother."""
    return float(np.var(np.diff(x)))


def _check_basic(result, length: int) -> None:
    """Assert common output properties: correct length, all finite."""
    assert len(result) == length
    assert np.all(np.isfinite(result))


# ======================================================================
# 1. normalize_denoise_spec
# ======================================================================

class TestNormalizeDenoiseSec:
    def test_string_ema(self):
        out = normalize_denoise_spec("ema")
        assert out is not None
        assert out["method"] == "ema"
        assert "span" in out["params"]

    def test_string_wavelet(self):
        out = normalize_denoise_spec("wavelet")
        assert out is not None
        assert out["method"] == "wavelet"
        assert out["params"]["wavelet"] == "db4"

    def test_string_sma(self):
        out = normalize_denoise_spec("sma")
        assert out["method"] == "sma"

    def test_string_median(self):
        out = normalize_denoise_spec("median")
        assert out["method"] == "median"

    def test_string_hp(self):
        out = normalize_denoise_spec("hp")
        assert out["method"] == "hp"
        assert out["params"]["lamb"] == 1600.0

    def test_string_butterworth(self):
        out = normalize_denoise_spec("butterworth")
        assert out["method"] == "butterworth"

    def test_string_savgol(self):
        out = normalize_denoise_spec("savgol")
        assert out["method"] == "savgol"

    def test_string_tv(self):
        out = normalize_denoise_spec("tv")
        assert out["method"] == "tv"

    def test_string_kalman(self):
        out = normalize_denoise_spec("kalman")
        assert out["method"] == "kalman"

    def test_string_hampel(self):
        out = normalize_denoise_spec("hampel")
        assert out["method"] == "hampel"

    def test_string_bilateral(self):
        out = normalize_denoise_spec("bilateral")
        assert out["method"] == "bilateral"

    def test_string_ssa(self):
        out = normalize_denoise_spec("ssa")
        assert out["method"] == "ssa"

    def test_string_l1_trend(self):
        out = normalize_denoise_spec("l1_trend")
        assert out["method"] == "l1_trend"

    def test_string_lms(self):
        out = normalize_denoise_spec("lms")
        assert out["method"] == "lms"

    def test_string_rls(self):
        out = normalize_denoise_spec("rls")
        assert out["method"] == "rls"

    def test_string_beta(self):
        out = normalize_denoise_spec("beta")
        assert out["method"] == "beta"

    def test_string_loess(self):
        out = normalize_denoise_spec("loess")
        assert out["method"] == "loess"

    def test_string_stl(self):
        out = normalize_denoise_spec("stl")
        assert out["method"] == "stl"

    def test_string_lowpass_fft(self):
        out = normalize_denoise_spec("lowpass_fft")
        assert out["method"] == "lowpass_fft"

    def test_string_gaussian(self):
        out = normalize_denoise_spec("gaussian")
        assert out["method"] == "gaussian"

    def test_string_whittaker(self):
        out = normalize_denoise_spec("whittaker")
        assert out["method"] == "whittaker"

    def test_string_wavelet_packet(self):
        out = normalize_denoise_spec("wavelet_packet")
        assert out["method"] == "wavelet_packet"

    def test_string_vmd(self):
        out = normalize_denoise_spec("vmd")
        assert out["method"] == "vmd"

    def test_string_emd(self):
        out = normalize_denoise_spec("emd")
        assert out["method"] == "emd"

    def test_string_eemd(self):
        out = normalize_denoise_spec("eemd")
        assert out["method"] == "eemd"

    def test_string_ceemdan(self):
        out = normalize_denoise_spec("ceemdan")
        assert out["method"] == "ceemdan"

    def test_dict_spec(self):
        out = normalize_denoise_spec({"method": "ema", "params": {"span": 20}})
        assert out is not None
        assert out["params"]["span"] == 20

    def test_dict_columns_string(self):
        out = normalize_denoise_spec({"method": "sma", "columns": "open, high"})
        assert out["columns"] == ["open", "high"]

    def test_none_returns_none(self):
        assert normalize_denoise_spec(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_denoise_spec("") is None

    def test_none_string_returns_none(self):
        assert normalize_denoise_spec("none") is None

    def test_unknown_method_returns_none(self):
        out = normalize_denoise_spec("nonexistent_filter")
        assert out is not None
        assert out["method"] == "nonexistent_filter"
        assert out["params"] == {}

    def test_default_when(self):
        out = normalize_denoise_spec("ema", default_when="post_ti")
        assert out["when"] == "post_ti"

    def test_dict_preserves_suffix(self):
        out = normalize_denoise_spec({"method": "sma", "suffix": "_smooth"})
        assert out["suffix"] == "_smooth"

    def test_dict_missing_params_defaults_to_empty(self):
        out = normalize_denoise_spec({"method": "ema"})
        assert out["params"] == {}


# ======================================================================
# 2. _denoise_series – dispatch through pd.Series
# ======================================================================

def _make_series(arr: np.ndarray) -> pd.Series:
    return pd.Series(arr, dtype=float)


class TestDenoiseSeriesDispatch:
    """Test _denoise_series for every supported method."""

    def test_none_returns_identity(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="none")
        pd.testing.assert_series_equal(result, s)

    def test_short_series_returns_identity(self):
        s = _make_series(np.array([1.0, 2.0]))
        result = _denoise_series(s, method="ema")
        pd.testing.assert_series_equal(result, s)

    def test_short_series_unknown_method_still_raises(self):
        s = _make_series(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="Unknown denoise method"):
            _denoise_series(s, method="nonexistent_method")

    def test_short_series_missing_optional_dependency_still_raises(self, monkeypatch):
        s = _make_series(np.array([1.0, 2.0]))
        monkeypatch.setattr("mtdata.utils.denoise._pywt", None)

        with pytest.raises(RuntimeError, match="requires PyWavelets"):
            _denoise_series(s, method="wavelet", params={"wavelet": "db4"})

    def test_ema(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="ema", params={"span": 10})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_ema_with_alpha(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="ema", params={"alpha": 0.2})
        _check_basic(result.values, N)

    def test_ema_zero_phase_reuses_alpha_for_backward_pass(self):
        s = _make_series(NOISY_SIGNAL)
        alpha = 0.2

        result = _denoise_series(s, method="ema", params={"alpha": alpha}, causality="zero_phase")

        forward = pd.Series(NOISY_SIGNAL).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        backward = pd.Series(forward[::-1]).ewm(alpha=alpha, adjust=False).mean().to_numpy()[::-1]
        expected = 0.5 * (forward + backward)

        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_sma(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="sma", params={"window": 10})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_median(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="median", params={"window": 7})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_lowpass_fft(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="lowpass_fft", params={"cutoff_ratio": 0.1})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_hp(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="hp", params={"lamb": 1600.0})
        _check_basic(result.values, N)

    def test_whittaker(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="whittaker", params={"lamb": 1000.0, "order": 2})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_savgol(self):
        pytest.importorskip("scipy.signal")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="savgol", params={"window": 11, "polyorder": 2})
        _check_basic(result.values, N)
        assert _smoothness(result.values) <= _smoothness(NOISY_SIGNAL)

    def test_savgol_rejects_window_longer_than_series(self):
        pytest.importorskip("scipy.signal")
        s = _make_series(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        with pytest.raises(ValueError, match="window_length"):
            _denoise_series(s, method="savgol", params={"window": 9, "polyorder": 2})

    def test_gaussian(self):
        pytest.importorskip("scipy.ndimage")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="gaussian", params={"sigma": 2.0})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_butterworth(self):
        pytest.importorskip("scipy.signal")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="butterworth", params={"cutoff": 0.1, "order": 4})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_hampel(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="hampel", params={"window": 7, "n_sigmas": 3.0})
        _check_basic(result.values, N)

    def test_bilateral(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="bilateral", params={"sigma_s": 2.0, "sigma_r": 0.5})
        _check_basic(result.values, N)

    def test_kalman(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="kalman")
        _check_basic(result.values, N)

    def test_tv(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="tv")
        _check_basic(result.values, N)

    def test_wavelet(self):
        pytest.importorskip("pywt")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="wavelet", params={"wavelet": "db4"})
        _check_basic(result.values, N)

    def test_wavelet_packet(self):
        pytest.importorskip("pywt")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="wavelet_packet", params={"wavelet": "db4"})
        _check_basic(result.values, N)

    def test_ssa(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="ssa", params={"window": 30, "components": 2})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_l1_trend(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="l1_trend", params={"lamb": 5.0})
        _check_basic(result.values, N)

    def test_lms(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="lms", params={"order": 5})
        _check_basic(result.values, N)

    def test_rls(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="rls", params={"order": 5})
        _check_basic(result.values, N)

    def test_beta(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="beta", params={"window": 9, "beta": 1.3})
        _check_basic(result.values, N)

    def test_vmd(self):
        pytest.importorskip("vmdpy")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="vmd")
        _check_basic(result.values, N)

    def test_emd(self):
        pytest.importorskip("PyEMD")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="emd")
        _check_basic(result.values, N)

    def test_eemd(self):
        pytest.importorskip("PyEMD")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="eemd", params={"trials": 10, "random_state": 42})
        _check_basic(result.values, N)

    def test_ceemdan(self):
        pytest.importorskip("PyEMD")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="ceemdan", params={"trials": 10, "random_state": 42})
        _check_basic(result.values, N)

    def test_loess(self):
        pytest.importorskip("statsmodels")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="loess", params={"frac": 0.2})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_stl(self):
        pytest.importorskip("statsmodels")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="stl", params={"period": 50})
        _check_basic(result.values, N)

    def test_stl_no_period_returns_identity(self):
        pytest.importorskip("statsmodels")
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="stl", params={})
        pd.testing.assert_series_equal(result, s)

    def test_unknown_method_returns_identity(self):
        s = _make_series(NOISY_SIGNAL)
        with pytest.raises(ValueError, match="Unknown denoise method"):
            _denoise_series(s, method="nonexistent_method")

    def test_missing_optional_dependency_raises_clear_error(self, monkeypatch):
        s = _make_series(NOISY_SIGNAL)
        monkeypatch.setattr("mtdata.utils.denoise._pywt", None)

        with pytest.raises(RuntimeError, match="requires PyWavelets"):
            _denoise_series(s, method="wavelet", params={"wavelet": "db4"})

    def test_ema_causal(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="ema", params={"span": 10}, causality="causal")
        _check_basic(result.values, N)

    def test_sma_causal(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="sma", params={"window": 10}, causality="causal")
        _check_basic(result.values, N)

    def test_kalman_causal(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="kalman", causality="causal")
        _check_basic(result.values, N)

    def test_lms_causal(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="lms", causality="causal")
        _check_basic(result.values, N)

    def test_rls_causal(self):
        s = _make_series(NOISY_SIGNAL)
        result = _denoise_series(s, method="rls", causality="causal")
        _check_basic(result.values, N)


# ======================================================================
# 3. _apply_denoise – DataFrame level
# ======================================================================

class TestApplyDenoise:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "close": NOISY_SIGNAL,
            "open": NOISY_SIGNAL + 0.1,
            "high": NOISY_SIGNAL + 0.5,
            "low": NOISY_SIGNAL - 0.5,
            "volume": np.abs(NOISY_SIGNAL) * 1000,
        })

    def test_none_spec_no_change(self):
        df = self._make_df()
        added = _apply_denoise(df, None)
        assert added == []

    def test_empty_spec_no_change(self):
        df = self._make_df()
        added = _apply_denoise(df, {})
        assert added == []

    def test_method_none_no_change(self):
        df = self._make_df()
        added = _apply_denoise(df, {"method": "none"})
        assert added == []

    def test_ema_keep_original(self):
        df = self._make_df()
        spec = {"method": "ema", "params": {"span": 10}, "columns": ["close"], "keep_original": True}
        added = _apply_denoise(df, spec)
        assert "close_dn" in added
        assert "close_dn" in df.columns
        assert len(df["close_dn"]) == N

    def test_ema_overwrite(self):
        df = self._make_df()
        original_close = df["close"].values.copy()
        spec = {"method": "ema", "params": {"span": 10}, "columns": ["close"], "keep_original": False}
        added = _apply_denoise(df, spec)
        assert added == []
        assert not np.array_equal(df["close"].values, original_close)

    def test_ohlcv_columns(self):
        df = self._make_df()
        spec = {"method": "sma", "params": {"window": 5}, "columns": "ohlcv", "keep_original": True}
        added = _apply_denoise(df, spec)
        for col in ("open_dn", "high_dn", "low_dn", "close_dn", "volume_dn"):
            assert col in added

    def test_custom_suffix(self):
        df = self._make_df()
        spec = {"method": "sma", "params": {"window": 5}, "columns": ["close"], "keep_original": True, "suffix": "_smooth"}
        added = _apply_denoise(df, spec)
        assert "close_smooth" in added

    def test_missing_column_skipped(self):
        df = self._make_df()
        spec = {"method": "sma", "params": {"window": 5}, "columns": ["nonexistent"], "keep_original": True}
        added = _apply_denoise(df, spec)
        assert added == []
        assert "denoise_warnings" in df.attrs
        assert "skipped missing column 'nonexistent'" in df.attrs["denoise_warnings"][0]

    def test_missing_column_warning_does_not_block_valid_columns(self):
        df = self._make_df()
        spec = {"method": "sma", "params": {"window": 5}, "columns": ["close", "nonexistent"], "keep_original": True}
        added = _apply_denoise(df, spec)
        assert "close_dn" in added
        assert "denoise_warnings" in df.attrs
        assert any("skipped missing column 'nonexistent'" in msg for msg in df.attrs["denoise_warnings"])

    def test_all_columns(self):
        df = self._make_df()
        spec = {"method": "sma", "params": {"window": 5}, "columns": "all", "keep_original": True}
        added = _apply_denoise(df, spec)
        assert len(added) >= 4

    def test_unknown_method_records_warning_and_returns_raw_data(self):
        df = self._make_df()
        original = df["close"].copy()

        added = _apply_denoise(df, {"method": "nonexistent_method", "columns": ["close"], "keep_original": False})

        assert added == []
        pd.testing.assert_series_equal(df["close"], original)
        assert "denoise_warnings" in df.attrs
        assert "Unknown denoise method 'nonexistent_method'" in df.attrs["denoise_warnings"][0]

    def test_all_nan_series_appends_warning_and_skips_column(self):
        df = pd.DataFrame({"close": [np.nan, np.nan, np.nan]})

        added = _apply_denoise(df, {"method": "ema", "columns": ["close"]})

        assert added == []
        assert "denoise_warnings" in df.attrs
        assert "contains no finite values for denoise" in df.attrs["denoise_warnings"][0]

    def test_unsupported_causality_appends_warning_and_skips_column(self):
        df = self._make_df()

        added = _apply_denoise(
            df,
            {"method": "wavelet", "columns": ["close"], "causality": "causal"},
        )

        assert added == []
        assert "denoise_warnings" in df.attrs
        assert "does not support causality='causal'" in df.attrs["denoise_warnings"][0]


# ======================================================================
# 4. _resolve_denoise_base_col
# ======================================================================

class TestResolveDenoisBaseCol:
    def test_no_denoise_returns_base(self):
        df = pd.DataFrame({"close": NOISY_SIGNAL})
        result = _resolve_denoise_base_col(df, None)
        assert result == "close"

    def test_empty_denoise_returns_base(self):
        df = pd.DataFrame({"close": NOISY_SIGNAL})
        result = _resolve_denoise_base_col(df, {})
        assert result == "close"

    def test_with_denoise_returns_dn_col(self):
        df = pd.DataFrame({"close": NOISY_SIGNAL})
        spec = {"method": "sma", "params": {"window": 5}, "columns": ["close"], "keep_original": True, "suffix": "_dn"}
        result = _resolve_denoise_base_col(df, spec, default_when="post_ti")
        assert result == "close_dn"
        assert "close_dn" in df.columns

    def test_denoise_overwrites_returns_base(self):
        df = pd.DataFrame({"close": NOISY_SIGNAL})
        spec = {"method": "sma", "params": {"window": 5}, "columns": ["close"], "keep_original": False}
        result = _resolve_denoise_base_col(df, spec)
        assert result == "close"


def test_run_denoise_handler_rejects_all_nan_series():
    s = pd.Series([np.nan, np.nan, np.nan], name="close")

    with pytest.raises(ValueError, match="contains no finite values for denoise"):
        _run_denoise_handler(
            s,
            lambda series, values, params, causality: series,
            {},
            "causal",
        )


def test_denoise_series_rejects_unsupported_causal_mode():
    s = pd.Series(np.arange(10.0), name="close")

    with pytest.raises(ValueError, match="does not support causality='causal'"):
        _denoise_series(s, method="wavelet", causality="causal")


# ======================================================================
# 5. Individual algorithm functions
# ======================================================================

class TestHpFilter:
    def test_basic(self):
        y = _hp_filter(NOISY_SIGNAL, lamb=1600.0)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_short_array(self):
        y = _hp_filter(np.array([1.0, 2.0]), lamb=1600.0)
        assert len(y) == 2

    def test_high_lambda_very_smooth(self):
        y = _hp_filter(NOISY_SIGNAL, lamb=1e8)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL) * 0.1


class TestWhittakerSmooth:
    def test_basic(self):
        y = _whittaker_smooth(NOISY_SIGNAL, lamb=1000.0, order=2)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_order_1(self):
        y = _whittaker_smooth(NOISY_SIGNAL, lamb=1000.0, order=1)
        _check_basic(y, N)

    def test_short_array(self):
        y = _whittaker_smooth(np.array([1.0, 2.0]), lamb=100.0, order=2)
        assert len(y) == 2


class TestTvDenoise1d:
    def test_basic(self):
        y = _tv_denoise_1d(NOISY_SIGNAL, weight=0.1, n_iter=50)
        _check_basic(y, N)

    def test_zero_weight_returns_input(self):
        y = _tv_denoise_1d(NOISY_SIGNAL, weight=0.0)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_short_array(self):
        y = _tv_denoise_1d(np.array([1.0, 2.0]), weight=0.1)
        assert len(y) == 2


class TestKalmanFilter1d:
    def test_basic(self):
        y = _kalman_filter_1d(NOISY_SIGNAL, process_var=0.01, measurement_var=1.0)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_with_initial_state(self):
        y = _kalman_filter_1d(NOISY_SIGNAL, process_var=0.01, measurement_var=1.0,
                              initial_state=0.0, initial_cov=1.0)
        _check_basic(y, N)

    def test_low_process_var_smooth(self):
        y = _kalman_filter_1d(NOISY_SIGNAL, process_var=1e-6, measurement_var=1.0)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL) * 0.1


class TestButterworthFilter:
    def test_basic_lowpass(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=0.1, order=4, btype="low",
                                causality="zero_phase", padlen=None)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_causal(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=0.1, order=4, btype="low",
                                causality="causal", padlen=None)
        _check_basic(y, N)

    def test_bandpass(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=[0.05, 0.2], order=2, btype="bandpass",
                                causality="zero_phase", padlen=None)
        _check_basic(y, N)

    def test_invalid_cutoff_returns_input(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=0.6, order=4, btype="low",
                                causality="zero_phase", padlen=None)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_invalid_bandpass_returns_input(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=[0.3, 0.1], order=2, btype="bandpass",
                                causality="zero_phase", padlen=None)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)


class TestHampelFilter:
    def test_basic(self):
        y = _hampel_filter(NOISY_SIGNAL, window=7, n_sigmas=3.0, causality="zero_phase")
        _check_basic(y, N)

    def test_replaces_outliers(self):
        rng = np.random.RandomState(123)
        x = rng.normal(0, 1, 50)
        x[25] = 50.0  # extreme outlier relative to normal data
        y = _hampel_filter(x, window=7, n_sigmas=3.0, causality="zero_phase")
        assert abs(y[25]) < 40.0  # outlier should be pulled toward median

    def test_causal(self):
        y = _hampel_filter(NOISY_SIGNAL, window=7, n_sigmas=3.0, causality="causal")
        _check_basic(y, N)

    def test_short_array(self):
        y = _hampel_filter(np.array([1.0, 2.0]), window=7, n_sigmas=3.0, causality="zero_phase")
        assert len(y) == 2


class TestBilateralFilter1d:
    def test_basic(self):
        y = _bilateral_filter_1d(NOISY_SIGNAL, sigma_s=2.0, sigma_r=0.5,
                                 truncate=3.0, causality="zero_phase")
        _check_basic(y, N)

    def test_causal(self):
        y = _bilateral_filter_1d(NOISY_SIGNAL, sigma_s=2.0, sigma_r=0.5,
                                 truncate=3.0, causality="causal")
        _check_basic(y, N)

    def test_zero_sigma_returns_input(self):
        y = _bilateral_filter_1d(NOISY_SIGNAL, sigma_s=0.0, sigma_r=0.5,
                                 truncate=3.0, causality="zero_phase")
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_short_array(self):
        y = _bilateral_filter_1d(np.array([1.0, 2.0]), sigma_s=2.0, sigma_r=0.5,
                                 truncate=3.0, causality="zero_phase")
        assert len(y) == 2


class TestSoftThreshold:
    def test_basic(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        y = _soft_threshold(x, thresh=2.0)
        expected = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(y, expected)

    def test_zero_threshold(self):
        x = np.array([1.0, -2.0, 3.0])
        y = _soft_threshold(x, thresh=0.0)
        np.testing.assert_array_almost_equal(y, x)

    def test_large_threshold_zeros(self):
        x = np.array([1.0, -2.0, 0.5])
        y = _soft_threshold(x, thresh=10.0)
        np.testing.assert_array_almost_equal(y, np.zeros(3))


class TestBetaIrlsMean:
    def test_beta_2_is_mean(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _beta_irls_mean(vals, beta=2.0, n_iter=20, eps=1e-6)
        assert abs(result - 3.0) < 1e-6

    def test_beta_0_is_median(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = _beta_irls_mean(vals, beta=0.0, n_iter=20, eps=1e-6)
        assert abs(result - 3.0) < 1e-6

    def test_empty_array(self):
        result = _beta_irls_mean(np.array([]), beta=1.3, n_iter=20, eps=1e-6)
        assert result == 0.0

    def test_intermediate_beta(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = _beta_irls_mean(vals, beta=1.3, n_iter=20, eps=1e-6)
        # Should be between median (3) and mean (22)
        assert 2.0 < result < 22.0


class TestBetaSmooth:
    def test_basic(self):
        y = _beta_smooth(NOISY_SIGNAL, window=9, beta=1.3, n_iter=20, eps=1e-6,
                         causality="zero_phase")
        _check_basic(y, N)

    def test_causal(self):
        y = _beta_smooth(NOISY_SIGNAL, window=9, beta=1.3, n_iter=20, eps=1e-6,
                         causality="causal")
        _check_basic(y, N)

    def test_short_array(self):
        y = _beta_smooth(np.array([1.0, 2.0]), window=9, beta=1.3, n_iter=20, eps=1e-6,
                         causality="zero_phase")
        assert len(y) == 2


class TestAdaptiveLmsFilter:
    def test_basic(self):
        y = _adaptive_lms_filter(NOISY_SIGNAL, order=5, mu=0.5)
        _check_basic(y, N)

    def test_zero_mu_returns_input(self):
        y = _adaptive_lms_filter(NOISY_SIGNAL, order=5, mu=0.0)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_no_bias(self):
        y = _adaptive_lms_filter(NOISY_SIGNAL, order=5, mu=0.5, use_bias=False)
        _check_basic(y, N)

    def test_with_leak(self):
        y = _adaptive_lms_filter(NOISY_SIGNAL, order=5, mu=0.5, leak=0.01)
        _check_basic(y, N)


class TestAdaptiveRlsFilter:
    def test_basic(self):
        y = _adaptive_rls_filter(NOISY_SIGNAL, order=5, lam=0.99, delta=1.0)
        _check_basic(y, N)

    def test_no_bias(self):
        y = _adaptive_rls_filter(NOISY_SIGNAL, order=5, lam=0.99, delta=1.0, use_bias=False)
        _check_basic(y, N)

    def test_invalid_lambda_returns_input(self):
        y = _adaptive_rls_filter(NOISY_SIGNAL, order=5, lam=1.5, delta=1.0)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)


class TestSsaDenoise:
    def test_basic(self):
        y = _ssa_denoise(NOISY_SIGNAL, window=30, components=2)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_energy_ratio_components(self):
        y = _ssa_denoise(NOISY_SIGNAL, window=30, components=0.9)
        _check_basic(y, N)

    def test_short_array(self):
        y = _ssa_denoise(np.array([1.0, 2.0, 3.0]), window=10, components=2)
        assert len(y) == 3

    def test_window_too_large(self):
        y = _ssa_denoise(NOISY_SIGNAL, window=N + 10, components=2)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_zero_energy_ratio_returns_finite_output(self):
        y = _ssa_denoise(np.zeros(20, dtype=float), window=5, components=0.9)
        np.testing.assert_array_equal(y, np.zeros(20, dtype=float))

    def test_constant_series_skips_svd(self, monkeypatch):
        monkeypatch.setattr(
            "mtdata.utils.denoise.filters.decomposition.np.linalg.svd",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("svd should be skipped")),
        )

        y = _ssa_denoise(np.full(20, 5.0, dtype=float), window=5, components=0.9)

        np.testing.assert_array_equal(y, np.full(20, 5.0, dtype=float))


class TestL1TrendFilter:
    def test_basic(self):
        y = _l1_trend_filter(NOISY_SIGNAL, lamb=5.0, n_iter=50, rho=1.0)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_zero_lambda_returns_input(self):
        y = _l1_trend_filter(NOISY_SIGNAL, lamb=0.0, n_iter=50, rho=1.0)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_short_array(self):
        y = _l1_trend_filter(np.array([1.0, 2.0, 3.0]), lamb=5.0, n_iter=50, rho=1.0)
        assert len(y) == 3


class TestWaveletPacketDenoise:
    def test_basic(self):
        pytest.importorskip("pywt")
        y = _wavelet_packet_denoise(NOISY_SIGNAL, wavelet="db4", level=None,
                                    threshold="auto", mode="soft", threshold_scale="auto")
        _check_basic(y, N)

    def test_hard_mode(self):
        pytest.importorskip("pywt")
        y = _wavelet_packet_denoise(NOISY_SIGNAL, wavelet="db4", level=2,
                                    threshold="auto", mode="hard")
        _check_basic(y, N)

    def test_numeric_threshold(self):
        pytest.importorskip("pywt")
        y = _wavelet_packet_denoise(NOISY_SIGNAL, wavelet="db4", level=2,
                                    threshold=0.5, mode="soft")
        _check_basic(y, N)

    def test_invalid_wavelet_returns_input(self):
        pytest.importorskip("pywt")
        y = _wavelet_packet_denoise(NOISY_SIGNAL, wavelet="INVALID_WAVELET", level=2,
                                    threshold="auto", mode="soft")
        np.testing.assert_array_equal(y, NOISY_SIGNAL)


class TestVmdDenoise:
    def test_basic(self):
        pytest.importorskip("vmdpy")
        y = _vmd_denoise(NOISY_SIGNAL, alpha=2000.0, tau=0.0, k=3, dc=0,
                         init=1, tol=1e-7, keep_modes=None, drop_modes=[-1],
                         keep_ratio=None)
        _check_basic(y, N)

    def test_keep_modes(self):
        pytest.importorskip("vmdpy")
        y = _vmd_denoise(NOISY_SIGNAL, alpha=2000.0, tau=0.0, k=3, dc=0,
                         init=1, tol=1e-7, keep_modes=[0, 1], drop_modes=None,
                         keep_ratio=None)
        _check_basic(y, N)

    def test_keep_ratio(self):
        pytest.importorskip("vmdpy")
        y = _vmd_denoise(NOISY_SIGNAL, alpha=2000.0, tau=0.0, k=3, dc=0,
                         init=1, tol=1e-7, keep_modes=None, drop_modes=None,
                         keep_ratio=0.9)
        _check_basic(y, N)

    def test_transposed_mode_fallback_uses_matching_axis(self, monkeypatch):
        import mtdata.utils.denoise as denoise_mod

        monkeypatch.setattr(
            denoise_mod,
            "_VMD",
            lambda *args, **kwargs: (
                np.vstack([np.linspace(0.0, 1.0, 8), np.linspace(1.0, 2.0, 8)]).T,
                None,
                None,
            ),
        )
        x = np.linspace(1.0, 2.0, 8)
        y = _vmd_denoise(x, alpha=2000.0, tau=0.0, k=2, dc=0, init=1, tol=1e-7, keep_modes=[0], drop_modes=None, keep_ratio=None)
        assert len(y) == len(x)
        assert np.all(np.isfinite(y))


# ======================================================================
# 6. get_denoise_methods_data
# ======================================================================

class TestGetDenoiseMethodsData:
    def test_returns_dict(self):
        data = get_denoise_methods_data()
        assert isinstance(data, dict)
        assert data["success"] is True
        assert "methods" in data

    def test_methods_is_list(self):
        data = get_denoise_methods_data()
        assert isinstance(data["methods"], list)
        assert len(data["methods"]) > 20

    def test_each_method_has_required_keys(self):
        data = get_denoise_methods_data()
        for m in data["methods"]:
            assert "method" in m
            assert "available" in m
            assert "description" in m
            assert "params" in m
            assert "supports" in m

    def test_none_method_present(self):
        data = get_denoise_methods_data()
        methods = [m["method"] for m in data["methods"]]
        assert "none" in methods

    def test_ema_method_present(self):
        data = get_denoise_methods_data()
        methods = [m["method"] for m in data["methods"]]
        assert "ema" in methods

    def test_schema_version(self):
        data = get_denoise_methods_data()
        assert data["schema_version"] == 1

    def test_reports_method_specific_causality_support(self):
        data = get_denoise_methods_data()
        methods = {entry["method"]: entry for entry in data["methods"]}

        assert methods["ema"]["supports"]["causality"] == ["causal", "zero_phase"]
        assert methods["wavelet"]["supports"]["causality"] == ["zero_phase"]


# ======================================================================
# 7. denoise_list_methods
# ======================================================================

class TestDenoiseListMethods:
    def test_returns_dict(self):
        result = denoise_list_methods()
        assert isinstance(result, dict)
        assert "methods" in result

    def test_matches_get_data(self):
        result = denoise_list_methods()
        data = get_denoise_methods_data()
        assert len(result["methods"]) == len(data["methods"])
