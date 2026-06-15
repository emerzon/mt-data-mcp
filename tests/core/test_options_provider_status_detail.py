"""Tests for options_provider_status compact remediation gating."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mtdata.bootstrap.settings import options_data_config
from mtdata.core.options import options_expirations, options_provider_status


def _call(detail):
    fn = getattr(options_provider_status, "__wrapped__", options_provider_status)
    return fn(detail=detail)


def _call_expirations(*, symbol: str, detail: str = "compact"):
    fn = getattr(options_expirations, "__wrapped__", options_expirations)
    return fn(symbol=symbol, detail=detail)


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
            "Yahoo fallback is best-effort only and may still return 401/429.",
        ]


def test_full_provider_status_keeps_remediation_when_unconfigured():
    out = _call("full")
    if out.get("action_required"):
        assert out.get("remediation")


def test_provider_status_marks_tradier_without_key_as_yahoo_fallback(monkeypatch):
    monkeypatch.setattr(options_data_config, "provider", "tradier")
    monkeypatch.setattr(options_data_config, "api_key", None)

    out = _call("full")

    assert out["configured_provider"] == "tradier"
    assert out["effective_provider"] == "yahoo"
    assert out["configured_provider_ready"] is False
    assert out["chain_data_ready"] is True
    assert out["action_required"] == "configure_options_provider"
    assert "retry unauthenticated Yahoo as a best-effort fallback" in out["remediation"]


def test_options_expirations_compact_keeps_fallback_warning(monkeypatch):
    import mtdata.services.options_service as options_service

    monkeypatch.setattr(options_data_config, "provider", "tradier")
    monkeypatch.setattr(options_data_config, "api_key", None)
    monkeypatch.setattr(
        options_service,
        "get_options_expirations",
        lambda **_kwargs: {
            "success": True,
            "provider": "yahoo",
            "configured_provider": "tradier",
            "provider_effective": "yahoo",
            "cached": False,
            "data_age_seconds": 0,
            "symbol": "AAPL",
            "expirations": ["2026-04-17"],
            "expiration_count": 1,
            "warnings": [
                "Yahoo fallback returned data after Tradier options provider failed: boom"
            ],
        },
    )

    out = _call_expirations(symbol="AAPL", detail="compact")

    assert out["success"] is True
    assert out["provider"] == "yahoo"
    assert out["configured_provider"] == "tradier"
    assert out["provider_effective"] == "yahoo"
    assert out["warnings"] == [
        "Yahoo fallback returned data after Tradier options provider failed: boom"
    ]
