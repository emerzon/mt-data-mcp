from __future__ import annotations

import pytest

from mtdata.core.asset_performance import asset_performance
from mtdata.services.research.registry import reset_research_registry


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_asset_performance_forex_stamps_research_quote_role(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_forex",
        lambda **kwargs: {"success": True, "pairs": [{"symbol": "EURUSD"}]},
    )
    reset_research_registry()

    result = _unwrap(asset_performance)(universe="forex")

    assert result["success"] is True
    assert result["providers_used"] == ["finviz"]
    assert result["universe"] == "forex"
    assert result["quote_role"] == "research_context_not_live_broker_quote"


def test_asset_performance_rejects_inapplicable_universe_selectors(monkeypatch) -> None:
    reset_research_registry()
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_crypto",
        lambda **_kwargs: pytest.fail("crypto must not fetch with a forex symbol"),
    )

    result = _unwrap(asset_performance)(universe="crypto", symbol="EURUSD")

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["invalid"] == ["symbol"]


def test_asset_performance_rejects_insider_offset(monkeypatch) -> None:
    reset_research_registry()
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_insider_activity",
        lambda **_kwargs: pytest.fail("insider must not fetch with offset"),
    )

    result = _unwrap(asset_performance)(universe="insider", offset=999)

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["invalid"] == ["offset"]
