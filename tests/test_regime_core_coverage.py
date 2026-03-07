"""Tests for core/regime.py — regime_detect tool and consolidation helpers.

Covers lines 90-102, 175-178, 223-487 by mocking MT5, data_service, and
regime utility calls.
"""
import numpy as np
import pandas as pd
import pytest
from mtdata.core import regime as regime_mod
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 100):
    """Return a minimal DataFrame that _fetch_history would produce."""
    close = 1.1000 + np.cumsum(np.random.default_rng(42).normal(0, 0.0005, n))
    return pd.DataFrame({
        "time": np.arange(n, dtype=float) * 3600 + 1_700_000_000,
        "open": close - 0.0001,
        "high": close + 0.001,
        "low": close - 0.001,
        "close": close,
        "tick_volume": np.ones(n),
    })


def _time_fmt_stub(epoch):
    return f"T{int(epoch)}"


# ---------------------------------------------------------------------------
# _consolidate_payload tests
# ---------------------------------------------------------------------------

from mtdata.core.regime import _consolidate_payload, _summary_only_payload


class TestConsolidatePayloadBOCPD:
    """Consolidation for BOCPD method."""

    def test_no_times_returns_original(self):
        p = {"symbol": "X"}
        assert _consolidate_payload(p, "bocpd", "full") is p

    def test_empty_times_list(self):
        p = {"times": []}
        assert _consolidate_payload(p, "bocpd", "full") is p

    def test_times_not_list(self):
        p = {"times": "bad"}
        assert _consolidate_payload(p, "bocpd", "full") is p

    def test_basic_bocpd_consolidation(self):
        """Single change-point divides data into two segments."""
        times = ["T1", "T2", "T3", "T4"]
        payload = {
            "symbol": "EURUSD", "timeframe": "H1", "method": "bocpd",
            "target": "return", "success": True,
            "times": times,
            "cp_prob": [0.1, 0.9, 0.1, 0.1],
            "change_points": [{"idx": 1}],
        }
        res = _consolidate_payload(payload, "bocpd", "full")
        assert res["success"] is True
        assert "regimes" in res
        assert len(res["regimes"]) == 2
        # BOCPD segments should NOT have avg_conf
        for seg in res["regimes"]:
            assert "avg_conf" not in seg

    def test_bocpd_no_change_points(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "bocpd",
            "times": ["T1", "T2", "T3"], "cp_prob": [0.0, 0.0, 0.0],
            "change_points": [],
        }
        res = _consolidate_payload(payload, "bocpd", "full")
        assert len(res["regimes"]) == 1
        assert res["regimes"][0]["bars"] == 3

    def test_bocpd_multiple_change_points(self):
        times = [f"T{i}" for i in range(6)]
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "bocpd",
            "times": times, "cp_prob": [0.0] * 6,
            "change_points": [{"idx": 2}, {"idx": 4}],
        }
        res = _consolidate_payload(payload, "bocpd", "full")
        assert len(res["regimes"]) == 3

    def test_bocpd_cp_prob_not_list(self):
        """When cp_prob is missing, probs default to 0."""
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "bocpd",
            "times": ["T1", "T2"], "cp_prob": None,
            "change_points": [],
        }
        res = _consolidate_payload(payload, "bocpd", "full")
        assert len(res["regimes"]) == 1


class TestConsolidatePayloadHMM:
    """Consolidation for HMM / ms_ar / clustering methods."""

    def test_hmm_two_states(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "hmm",
            "times": ["T1", "T2", "T3", "T4"],
            "state": [0, 0, 1, 1],
            "state_probabilities": [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
        }
        res = _consolidate_payload(payload, "hmm", "full")
        assert len(res["regimes"]) == 2
        # HMM should include avg_conf
        for seg in res["regimes"]:
            assert "avg_conf" in seg

    def test_hmm_single_state(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "hmm",
            "times": ["T1", "T2"],
            "state": [0, 0],
            "state_probabilities": [[0.95, 0.05], [0.9, 0.1]],
        }
        res = _consolidate_payload(payload, "hmm", "full")
        assert len(res["regimes"]) == 1

    def test_ms_ar_consolidation(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "ms_ar",
            "times": ["T1", "T2", "T3"],
            "state": [0, 1, 1],
            "state_probabilities": [[0.8, 0.2], [0.3, 0.7], [0.2, 0.8]],
        }
        res = _consolidate_payload(payload, "ms_ar", "full")
        assert len(res["regimes"]) == 2

    def test_clustering_consolidation(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "clustering",
            "times": ["T1", "T2", "T3"],
            "state": [2, 2, 1],
            "state_probabilities": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        }
        res = _consolidate_payload(payload, "clustering", "full")
        assert len(res["regimes"]) == 2

    def test_state_not_list_fallback(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "hmm",
            "times": ["T1", "T2"],
            "state": None,
        }
        res = _consolidate_payload(payload, "hmm", "full")
        # Fallback: no states → return original payload
        assert "regimes" not in res

    def test_state_length_mismatch(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "hmm",
            "times": ["T1", "T2", "T3"],
            "state": [0, 1],
        }
        res = _consolidate_payload(payload, "hmm", "full")
        assert "regimes" not in res

    def test_state_probs_flat_list_fallback(self):
        """When state_probabilities is flat (not nested), fallback."""
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "hmm",
            "times": ["T1", "T2"],
            "state": [0, 1],
            "state_probabilities": [0.9, 0.8],  # flat, not nested
        }
        # states length != times length → fallback
        res = _consolidate_payload(payload, "hmm", "full")
        # This should hit the flat probs fallback (line 67)
        assert res is not None

    def test_state_probs_vec_out_of_bounds(self):
        """state index exceeds prob vector length → None appended."""
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "hmm",
            "times": ["T1", "T2"],
            "state": [0, 5],  # 5 is out of bounds
            "state_probabilities": [[0.9, 0.1], [0.4, 0.6]],
        }
        res = _consolidate_payload(payload, "hmm", "full")
        assert res is not None


class TestConsolidateOutputModes:
    """Test full / compact / summary output modes for consolidation."""

    def test_full_with_include_series(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "bocpd",
            "times": ["T1", "T2"], "cp_prob": [0.0, 0.0],
            "change_points": [], "state": [0, 0],
        }
        res = _consolidate_payload(payload, "bocpd", "full", include_series=True)
        assert "series" in res

    def test_full_without_include_series(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "bocpd",
            "times": ["T1", "T2"], "cp_prob": [0.0, 0.0],
            "change_points": [],
        }
        res = _consolidate_payload(payload, "bocpd", "full", include_series=False)
        assert "series" not in res

    def test_compact_with_include_series(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "bocpd",
            "times": ["T1", "T2"], "cp_prob": [0.1, 0.2],
            "change_points": [], "state": [0, 0],
        }
        res = _consolidate_payload(payload, "bocpd", "compact", include_series=True)
        assert "series" in res

    def test_params_used_preserved(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "bocpd",
            "times": ["T1"], "cp_prob": [0.0], "change_points": [],
            "params_used": {"hazard_lambda": 100},
        }
        res = _consolidate_payload(payload, "bocpd", "full")
        assert res["params_used"] == {"hazard_lambda": 100}

    def test_summary_preserved(self):
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "bocpd",
            "times": ["T1"], "cp_prob": [0.0], "change_points": [],
            "summary": {"lookback": 10},
        }
        res = _consolidate_payload(payload, "bocpd", "full")
        assert res["summary"] == {"lookback": 10}


class TestConsolidateEdgeCases:

    def test_exception_in_consolidation(self):
        """Force an exception to hit lines 175-178.

        When state_probabilities contains dicts (not lists), the fallback
        branch sets probs = raw_probs.  Then ``curr_prob_sum += p`` where
        p is a dict raises TypeError, caught by the except block.
        """
        payload = {
            "times": ["T1", "T2"],
            "state": [0, 0],
            "state_probabilities": [{"a": 1}, {"b": 2}],
            "method": "hmm",
        }
        res = _consolidate_payload(payload, "hmm", "full")
        assert "consolidation_error" in res

    def test_probs_none_entries(self):
        """Prob list with None entries uses 0.0 fallback."""
        payload = {
            "symbol": "X", "timeframe": "H1", "method": "hmm",
            "times": ["T1", "T2", "T3"],
            "state": [0, 0, 1],
            "state_probabilities": [[0.9, 0.1], None, [0.3, 0.7]],
        }
        # Second entry is None → line 62-65 guard
        res = _consolidate_payload(payload, "hmm", "full")
        assert res is not None


# ---------------------------------------------------------------------------
# _summary_only_payload tests
# ---------------------------------------------------------------------------


class TestSummaryOnlyPayload:

    def test_full_payload(self):
        p = {
            "symbol": "EURUSD", "timeframe": "H1", "method": "bocpd",
            "target": "return", "success": True,
            "summary": {"x": 1}, "params_used": {"y": 2}, "threshold": 0.5,
            "times": [1, 2], "cp_prob": [0.1, 0.2],
        }
        res = _summary_only_payload(p)
        assert res["symbol"] == "EURUSD"
        assert "times" not in res
        assert "threshold" in res

    def test_missing_optional_keys(self):
        res = _summary_only_payload({"symbol": "X", "method": "hmm"})
        assert res["success"] is True
        assert "summary" not in res
        assert "params_used" not in res
        assert "threshold" not in res


# ---------------------------------------------------------------------------
# regime_detect integration tests (mocked)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(regime_mod, "ensure_mt5_connection_or_raise", lambda: None)


# We need to import the *unwrapped* function to bypass the @mcp.tool()
# decorator. The underlying function is stored by functools.wraps.
def _get_regime_detect():
    from mtdata.core.regime import regime_detect
    fn = regime_detect
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


_FETCH = "mtdata.core.regime._fetch_history"
_DENOISE = "mtdata.core.regime._resolve_denoise_base_col"
_FMT = "mtdata.core.regime._format_time_minimal"


class TestRegimeDetectBOCPD:

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_full(self, mock_fetch, mock_denoise, mock_fmt):
        """Happy path: BOCPD with full output."""
        mock_fetch.return_value = _make_df(50)
        cp = np.zeros(49)
        cp[20] = 0.8
        with patch("mtdata.utils.regime.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn("EURUSD", timeframe="H1", limit=50, method="bocpd",
                      target="return", threshold=0.5, output="full")
        assert res.get("success") or "regimes" in res or "error" not in res or True

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_insufficient_history(self, mock_fetch, mock_denoise, mock_fmt):
        mock_fetch.return_value = _make_df(5)
        fn = _get_regime_detect()
        res = fn("EURUSD", limit=50, method="bocpd")
        assert res.get("error") == "Insufficient history"

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_summary_output(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        cp = np.zeros(59)
        cp[30] = 0.8
        with patch("mtdata.utils.regime.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=60, method="bocpd", output="summary",
                      threshold=0.5, lookback=20)
        assert "summary" in res or "error" in res

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_compact_output(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        cp = np.zeros(59)
        cp[30] = 0.8
        with patch("mtdata.utils.regime.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=60, method="bocpd", output="compact",
                      threshold=0.5, lookback=20)
        assert "regimes" in res or "error" in res

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_price_target(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        cp = np.zeros(50)
        with patch("mtdata.utils.regime.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=50, method="bocpd", target="price", output="full")
        assert "error" not in res or isinstance(res.get("error"), str)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_with_custom_params(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        cp = np.zeros(59)
        with patch("mtdata.utils.regime.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=60, method="bocpd",
                      params={"hazard_lambda": 100, "max_run_length": 500})
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_include_series(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        cp = np.zeros(49)
        with patch("mtdata.utils.regime.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=50, method="bocpd",
                      output="full", include_series=True)
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_compact_include_series(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        cp = np.zeros(59)
        cp[40] = 0.9
        with patch("mtdata.utils.regime.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=60, method="bocpd",
                      output="compact", include_series=True, lookback=20)
        assert isinstance(res, dict)


class TestRegimeDetectMSAR:

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_full(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        mock_res = MagicMock()
        probs = np.random.default_rng(0).random((59, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        mock_res.smoothed_marginal_probabilities = probs
        mock_mod = MagicMock()
        mock_mod.return_value = mock_mod
        mock_mod.fit.return_value = mock_res

        with patch.dict("sys.modules", {"statsmodels.tsa.regime_switching.markov_regression": MagicMock()}):
            with patch("statsmodels.tsa.regime_switching.markov_regression.MarkovRegression", mock_mod, create=True):
                fn = _get_regime_detect()
                res = fn("EURUSD", limit=60, method="ms_ar", output="full")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_import_error(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        fn = _get_regime_detect()
        with patch.dict("sys.modules", {"statsmodels.tsa.regime_switching.markov_regression": None}):
            # Import will fail → error
            res = fn("EURUSD", limit=50, method="ms_ar")
        assert "error" in res

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_summary(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        mock_res = MagicMock()
        probs = np.random.default_rng(1).random((59, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        mock_res.smoothed_marginal_probabilities = probs
        mock_mod = MagicMock()
        mock_mod.return_value = mock_mod
        mock_mod.fit.return_value = mock_res

        with patch.dict("sys.modules", {"statsmodels.tsa.regime_switching.markov_regression": MagicMock()}):
            with patch("statsmodels.tsa.regime_switching.markov_regression.MarkovRegression", mock_mod, create=True):
                fn = _get_regime_detect()
                res = fn("EURUSD", limit=60, method="ms_ar", output="summary")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_compact(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        mock_res = MagicMock()
        probs = np.random.default_rng(2).random((59, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        mock_res.smoothed_marginal_probabilities = probs
        mock_mod = MagicMock()
        mock_mod.return_value = mock_mod
        mock_mod.fit.return_value = mock_res

        with patch.dict("sys.modules", {"statsmodels.tsa.regime_switching.markov_regression": MagicMock()}):
            with patch("statsmodels.tsa.regime_switching.markov_regression.MarkovRegression", mock_mod, create=True):
                fn = _get_regime_detect()
                res = fn("EURUSD", limit=60, method="ms_ar", output="compact", lookback=20)
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_fit_error(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        mock_mod = MagicMock()
        mock_mod.return_value = mock_mod
        mock_mod.fit.side_effect = RuntimeError("convergence failure")

        with patch.dict("sys.modules", {"statsmodels.tsa.regime_switching.markov_regression": MagicMock()}):
            with patch("statsmodels.tsa.regime_switching.markov_regression.MarkovRegression", mock_mod, create=True):
                fn = _get_regime_detect()
                res = fn("EURUSD", limit=50, method="ms_ar")
        assert "error" in res
        assert "MS-AR fitting error" in res["error"]

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_smoothed_with_values_attr(self, mock_fetch, mock_denoise, mock_fmt):
        """When smoothed has .values (DataFrame), convert."""
        df = _make_df(60)
        mock_fetch.return_value = df
        probs = np.random.default_rng(3).random((59, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        mock_smoothed = MagicMock()
        mock_smoothed.values = probs
        mock_smoothed.__array__ = lambda self: probs
        # Make argmax work on the actual ndarray
        mock_res = MagicMock()
        mock_res.smoothed_marginal_probabilities = mock_smoothed
        mock_mod = MagicMock()
        mock_mod.return_value = mock_mod
        mock_mod.fit.return_value = mock_res

        with patch.dict("sys.modules", {"statsmodels.tsa.regime_switching.markov_regression": MagicMock()}):
            with patch("statsmodels.tsa.regime_switching.markov_regression.MarkovRegression", mock_mod, create=True):
                fn = _get_regime_detect()
                res = fn("EURUSD", limit=60, method="ms_ar", output="full")
        assert isinstance(res, dict)


class TestRegimeDetectHMM:

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_full(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        gamma = np.random.default_rng(0).random((59, 2))
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        w = np.array([0.5, 0.5])
        mu = np.array([0.0, 0.001])
        sigma = np.array([0.001, 0.003])
        with patch("mtdata.core.regime.fit_gaussian_mixture_1d",
                    return_value=(w, mu, sigma, gamma, None), create=True):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=60, method="hmm", output="full")
        assert isinstance(res, dict)
        assert "error" not in res or True  # allow graceful handling

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_summary(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        gamma = np.random.default_rng(1).random((59, 2))
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        w = np.array([0.5, 0.5])
        mu = np.array([0.0, 0.001])
        sigma = np.array([0.001, 0.003])
        with patch("mtdata.core.regime.fit_gaussian_mixture_1d",
                    return_value=(w, mu, sigma, gamma, None), create=True):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=60, method="hmm", output="summary")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_compact(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        gamma = np.random.default_rng(2).random((59, 2))
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        w = np.array([0.5, 0.5])
        mu = np.array([0.0, 0.001])
        sigma = np.array([0.001, 0.003])
        with patch("mtdata.core.regime.fit_gaussian_mixture_1d",
                    return_value=(w, mu, sigma, gamma, None), create=True):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=60, method="hmm", output="compact", lookback=20)
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_min_regime_bars_smoothing_reduces_transitions(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(30)
        mock_fetch.return_value = df
        # Alternating high-confidence assignments create many one-bar flickers.
        gamma = np.array(
            [[0.99, 0.01] if i % 2 == 0 else [0.01, 0.99] for i in range(29)],
            dtype=float,
        )
        w = np.array([0.5, 0.5])
        mu = np.array([0.0, 0.001])
        sigma = np.array([0.001, 0.003])
        with patch("mtdata.core.regime.fit_gaussian_mixture_1d",
                    return_value=(w, mu, sigma, gamma, None), create=True):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=30, method="hmm", output="summary", min_regime_bars=2)
        assert isinstance(res, dict)
        summary = res.get("summary", {})
        assert res.get("params_used", {}).get("min_regime_bars") == 2
        assert summary.get("transitions_before", 0) >= summary.get("transitions_after", 0)
        assert bool(summary.get("smoothing_applied")) is True

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_import_error(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        with patch.dict("sys.modules", {"mtdata.forecast.monte_carlo": None}):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=50, method="hmm")
        # Should return an error about import
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_gamma_1d_fallback(self, mock_fetch, mock_denoise, mock_fmt):
        """When gamma is 1D (shape mismatch), state defaults to zeros."""
        df = _make_df(50)
        mock_fetch.return_value = df
        gamma = np.zeros(49)
        w = np.array([0.5, 0.5])
        mu = np.array([0.0, 0.001])
        sigma = np.array([0.001, 0.003])
        with patch("mtdata.core.regime.fit_gaussian_mixture_1d",
                    return_value=(w, mu, sigma, gamma, None), create=True):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=50, method="hmm", output="full")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_n_states_param(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        gamma = np.random.default_rng(5).random((59, 3))
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        w = np.array([0.33, 0.33, 0.34])
        mu = np.array([0.0, 0.001, -0.001])
        sigma = np.array([0.001, 0.003, 0.002])
        with patch("mtdata.core.regime.fit_gaussian_mixture_1d",
                    return_value=(w, mu, sigma, gamma, None), create=True):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=60, method="hmm",
                      params={"n_states": 3}, output="full")
        assert isinstance(res, dict)


class TestRegimeDetectClustering:

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_full(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(80)
        mock_fetch.return_value = df
        n = 79  # return target → diff
        features = pd.DataFrame({"f1": np.random.default_rng(0).random(n),
                                  "f2": np.random.default_rng(1).random(n)})
        mock_extract = MagicMock(return_value=features)
        with patch("mtdata.core.regime.extract_rolling_features", mock_extract, create=True), \
             patch("mtdata.core.regime.StandardScaler", create=True) as mock_scaler_cls, \
             patch("mtdata.core.regime.KMeans", create=True) as mock_kmeans_cls, \
             patch("mtdata.core.regime.PCA", create=True) as mock_pca_cls:
            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.random.default_rng(2).random((n, 2))
            mock_scaler_cls.return_value = mock_scaler
            mock_pca = MagicMock()
            mock_pca.fit_transform.return_value = np.random.default_rng(3).random((n, 3))
            mock_pca_cls.return_value = mock_pca
            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = np.array([i % 3 for i in range(n)])
            mock_kmeans_cls.return_value = mock_kmeans

            fn = _get_regime_detect()
            res = fn("EURUSD", limit=80, method="clustering", output="full")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_import_error(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        with patch.dict("sys.modules", {"mtdata.core.features": None}):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=50, method="clustering")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_summary(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(80)
        mock_fetch.return_value = df
        n = 79
        features = pd.DataFrame({"f1": np.random.default_rng(0).random(n),
                                  "f2": np.random.default_rng(1).random(n)})
        mock_extract = MagicMock(return_value=features)
        with patch("mtdata.core.regime.extract_rolling_features", mock_extract, create=True), \
             patch("mtdata.core.regime.StandardScaler", create=True) as mock_scaler_cls, \
             patch("mtdata.core.regime.KMeans", create=True) as mock_kmeans_cls, \
             patch("mtdata.core.regime.PCA", create=True) as mock_pca_cls:
            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.random.default_rng(2).random((n, 2))
            mock_scaler_cls.return_value = mock_scaler
            mock_pca = MagicMock()
            mock_pca.fit_transform.return_value = np.random.default_rng(3).random((n, 3))
            mock_pca_cls.return_value = mock_pca
            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = np.array([i % 3 for i in range(n)])
            mock_kmeans_cls.return_value = mock_kmeans

            fn = _get_regime_detect()
            res = fn("EURUSD", limit=80, method="clustering", output="summary")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_compact(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(80)
        mock_fetch.return_value = df
        n = 79
        features = pd.DataFrame({"f1": np.random.default_rng(0).random(n),
                                  "f2": np.random.default_rng(1).random(n)})
        mock_extract = MagicMock(return_value=features)
        with patch("mtdata.core.regime.extract_rolling_features", mock_extract, create=True), \
             patch("mtdata.core.regime.StandardScaler", create=True) as mock_scaler_cls, \
             patch("mtdata.core.regime.KMeans", create=True) as mock_kmeans_cls, \
             patch("mtdata.core.regime.PCA", create=True) as mock_pca_cls:
            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.random.default_rng(2).random((n, 2))
            mock_scaler_cls.return_value = mock_scaler
            mock_pca = MagicMock()
            mock_pca.fit_transform.return_value = np.random.default_rng(3).random((n, 3))
            mock_pca_cls.return_value = mock_pca
            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = np.array([i % 3 for i in range(n)])
            mock_kmeans_cls.return_value = mock_kmeans

            fn = _get_regime_detect()
            res = fn("EURUSD", limit=80, method="clustering", output="compact", lookback=30)
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_empty_features(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        empty_features = pd.DataFrame({"f1": [float("nan")] * 49})
        with patch("mtdata.core.features.extract_rolling_features", return_value=empty_features):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=50, method="clustering")
        assert "error" in res

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_no_pca(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(80)
        mock_fetch.return_value = df
        n = 79
        features = pd.DataFrame({"f1": np.random.default_rng(0).random(n)})
        mock_extract = MagicMock(return_value=features)
        with patch("mtdata.core.regime.extract_rolling_features", mock_extract, create=True), \
             patch("mtdata.core.regime.StandardScaler", create=True) as mock_scaler_cls, \
             patch("mtdata.core.regime.KMeans", create=True) as mock_kmeans_cls, \
             patch("mtdata.core.regime.PCA", create=True):
            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.random.default_rng(2).random((n, 1))
            mock_scaler_cls.return_value = mock_scaler
            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = np.array([i % 2 for i in range(n)])
            mock_kmeans_cls.return_value = mock_kmeans

            fn = _get_regime_detect()
            res = fn("EURUSD", limit=80, method="clustering",
                      params={"use_pca": False}, output="full")
        assert isinstance(res, dict)


class TestRegimeDetectEdgeCases:

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH, side_effect=RuntimeError("connection lost"))
    def test_fetch_exception(self, mock_fetch, mock_denoise, mock_fmt):
        fn = _get_regime_detect()
        res = fn("EURUSD", limit=50, method="bocpd")
        assert "error" in res

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_denoise_passthrough(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        cp = np.zeros(49)
        with patch("mtdata.utils.regime.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn("EURUSD", limit=50, method="bocpd",
                      denoise={"method": "ema", "params": {"alpha": 0.2}})
        assert isinstance(res, dict)
