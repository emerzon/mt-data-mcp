"""Tests for options_provider_status compact remediation gating."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mtdata.core.options import options_provider_status


def _call(detail):
    fn = getattr(options_provider_status, "__wrapped__", options_provider_status)
    return fn(detail=detail)


def test_compact_provider_status_drops_verbose_remediation():
    out = _call("compact")
    # When unconfigured, the actionable flag stays but the long paragraph is gated.
    if out.get("action_required"):
        assert "remediation" not in out
        assert out.get("remediation_hint")


def test_full_provider_status_keeps_remediation_when_unconfigured():
    out = _call("full")
    if out.get("action_required"):
        assert out.get("remediation")
