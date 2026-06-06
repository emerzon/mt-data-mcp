"""Tests for denoise_describe causality guidance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mtdata.core.denoise import denoise_describe


def _describe(method):
    fn = getattr(denoise_describe, "__wrapped__", denoise_describe)
    return fn(method)


def test_denoise_describe_warns_on_zero_phase_only():
    out = _describe("wavelet")
    method = out.get("method", {})
    causality = (method.get("supports") or {}).get("causality")
    if causality == ["zero_phase"]:
        assert "look-ahead" in method.get("causality_note", "")


def test_denoise_describe_notes_dual_causality():
    out = _describe("ema")
    method = out.get("method", {})
    causality = (method.get("supports") or {}).get("causality")
    if isinstance(causality, list) and "causal" in causality and "zero_phase" in causality:
        assert "causal" in method.get("causality_note", "").lower()
