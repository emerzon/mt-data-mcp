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

        assert [segment["regime"] for segment in result["regimes"]] == [1, 0]
        assert result["params_used"]["relabeled"] is True
