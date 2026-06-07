"""Tests for options_provider_status compact remediation gating."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mtdata.core.options import options_provider_status


def _call(detail):
    fn = getattr(options_provider_status, "__wrapped__", options_provider_status)
    return fn(detail=detail)


def test_compact_provider_status_keeps_actionable_setup_steps():
    out = _call("compact")
    if out.get("action_required"):
        assert "remediation" not in out
        assert out["remediation_hint"] == (
            "Reliable options-chain access requires Tradier credentials."
        )
        assert out["next_steps"] == [
            "Set MTDATA_OPTIONS_PROVIDER=tradier.",
            "Set MTDATA_OPTIONS_API_KEY to a Tradier API token, then restart mtdata.",
            "Use Yahoo only as an unauthenticated fallback that may return 401/429.",
        ]


def test_full_provider_status_keeps_remediation_when_unconfigured():
    out = _call("full")
    if out.get("action_required"):
        assert out.get("remediation")
