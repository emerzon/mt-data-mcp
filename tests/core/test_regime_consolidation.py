"""Tests for core/regime.py — pure consolidation helpers (no MT5)."""
import pytest

from mtdata.core.regime import (
    _consolidate_payload,
    _summary_only_payload,
)


class TestSummaryOnlyPayload:
    def test_basic(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "target": "return",
            "success": True,
            "summary": {"regimes_found": 3},
            "params_used": {"hazard_lambda": 250},
            "threshold": 0.5,
            "times": [1, 2, 3],
            "cp_prob": [0.1, 0.8, 0.1],
        }
        result = _summary_only_payload(payload)
        assert result["symbol"] == "EURUSD"
        assert result["method"] == "bocpd"
        assert result["success"] is True
        assert "summary" in result
        assert "params_used" in result
        assert "threshold" in result
        # Should NOT include raw data
        assert "times" not in result
        assert "cp_prob" not in result

    def test_minimal(self):
        payload = {"symbol": "X", "method": "hmm"}
        result = _summary_only_payload(payload)
        assert result["symbol"] == "X"
        assert result["success"] is True
        assert "summary" not in result


class TestConsolidatePayload:
    def test_with_regimes(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "success": True,
            "summary": {"total": 3},
            "times": [1.0, 2.0, 3.0, 4.0, 5.0],
            "state": [0, 0, 1, 1, 0],
            "cp_prob": [0.1, 0.1, 0.9, 0.1, 0.8],
        }
        result = _consolidate_payload(payload, "bocpd", "full", include_series=True)
        assert result["symbol"] == "EURUSD"
        assert result["success"] is True
        assert "regimes" in result
        assert "current_regime" in result
        assert "transition_summary" in result

    def test_compact_mode(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "success": True,
            "times": [1.0, 2.0, 3.0],
            "state": [0, 0, 1],
            "cp_prob": [0.1, 0.1, 0.9],
        }
        result = _consolidate_payload(payload, "bocpd", "compact", include_series=True)
        assert "regimes" in result
        assert "current_regime" in result

    def test_no_series_when_not_requested(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "success": True,
            "times": [1.0, 2.0],
            "state": [0, 1],
            "cp_prob": [0.1, 0.9],
        }
        result = _consolidate_payload(payload, "bocpd", "full", include_series=False)
        assert "series" not in result

    def test_params_preserved(self):
        payload = {
            "symbol": "X",
            "method": "hmm",
            "success": True,
            "params_used": {"n_states": 2},
            "times": [1.0],
            "state": [0],
        }
        result = _consolidate_payload(payload, "hmm", "full")
        assert result.get("params_used") == {"n_states": 2}

    def test_bocpd_regimes_are_canonicalized_by_series_mean(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "success": True,
            "times": [1.0, 2.0, 3.0, 4.0],
            "cp_prob": [0.1, 0.1, 0.9, 0.1],
            "change_points": [{"idx": 2, "time": 3.0, "prob": 0.9}],
            "_series_values": [0.4, 0.5, -0.2, -0.1],
            "params_used": {},
        }

        result = _consolidate_payload(payload, "bocpd", "full")

        assert [regime["bias"] for regime in result["regimes"]] == [
            "bullish",
            "bearish",
        ]
        assert result["params_used"]["relabeled"] is True
        assert result["params_used"]["label_mapping"] == {"0": 1, "1": 0}

    def test_all_invalid_states_are_dropped_from_consolidated_regimes(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "clustering",
            "success": True,
            "times": [1.0, 2.0, 3.0],
            "state": [-1, -1, -1],
            "state_probabilities": [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        }

        result = _consolidate_payload(payload, "clustering", "full")

        assert result["success"] is True
        assert result["regimes"] == []

    def test_hmm_regime_info_uses_canonical_label_mapping(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "hmm",
            "target": "return",
            "success": True,
            "times": ["T1", "T2"],
            "state": [0, 1],
            "state_probabilities": [[0.95, 0.05], [0.05, 0.95]],
            "regime_params": {
                "weights": [0.6, 0.4],
                "mu": [0.002, -0.001],
                "sigma": [0.0035, 0.0004],
            },
            "params_used": {
                "relabeled": True,
                "label_mapping": {"0": 1, "1": 0},
            },
        }

        result = _consolidate_payload(payload, "hmm", "full")

        assert result["regimes"][0]["label"] == "negative_very_low_vol"
        assert result["regimes"][1]["label"] == "positive_high_vol"
        assert result["regime_info"][0]["mean_return"] == pytest.approx(-0.001)
        assert result["regime_info"][0]["volatility"] == pytest.approx(0.0004)
        assert result["regime_info"][1]["mean_return"] == pytest.approx(0.002)
        assert result["regime_info"][1]["volatility"] == pytest.approx(0.0035)
        assert result["regime_info"][0]["observed_in_window"] is True
        assert result["regime_info"][1]["observed_in_window"] is True

    def test_hmm_regime_info_marks_unobserved_model_states(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "hmm",
            "target": "return",
            "success": True,
            "times": ["T1", "T2"],
            "state": [0, 0],
            "state_probabilities": [[0.95, 0.05], [0.90, 0.10]],
            "regime_params": {
                "weights": [0.7, 0.3],
                "mu": [-0.001, 0.002],
                "sigma": [0.0004, 0.0035],
            },
        }

        result = _consolidate_payload(payload, "hmm", "compact")

        assert result["regime_info"][0]["observed_in_window"] is True
        assert result["regime_info"][1]["observed_in_window"] is False
        assert "not observed" in result["regime_info"][1]["note"]
        assert "not necessarily the fraction" in result["regime_info"][1]["weight_note"]

    def test_hmm_price_target_uses_price_level_metadata(self):
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "hmm",
            "target": "price",
            "success": True,
            "times": ["T1", "T2"],
            "state": [0, 1],
            "state_probabilities": [[0.95, 0.05], [0.05, 0.95]],
            "regime_params": {
                "weights": [0.55, 0.45],
                "mu": [1.105, 1.255],
                "sigma": [0.01, 0.015],
            },
        }

        result = _consolidate_payload(payload, "hmm", "full")

        assert result["regimes"][0]["label"] == "low_price_regime"
        assert result["regimes"][1]["label"] == "high_price_regime"
        assert result["regime_info"][0]["mean_price"] == pytest.approx(1.105)
        assert result["regime_info"][0]["std_dev"] == pytest.approx(0.01)
        assert "mean_return" not in result["regime_info"][0]
        assert "volatility_pct" not in result["regime_info"][0]
