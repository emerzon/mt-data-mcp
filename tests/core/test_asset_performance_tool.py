from __future__ import annotations

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
