from __future__ import annotations

from mtdata.core.equity_profile import equity_profile
from mtdata.services.research.registry import reset_research_registry


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_equity_profile_default_summary_uses_fundamentals(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_fundamentals",
        lambda symbol, detail="compact", category="summary", fields=None: {
            "success": True,
            "symbol": symbol,
            "category": category,
        },
    )
    reset_research_registry()

    result = _unwrap(equity_profile)("AAPL")

    assert result["success"] is True
    assert result["symbol"] == "AAPL"
    assert result["providers_used"] == ["finviz"]
    assert result["sections"] == ["summary"]


def test_equity_profile_mt5_pin_is_unsupported() -> None:
    from mtdata.services.research.capabilities import NEWS
    from mtdata.services.research.registry import get_research_registry

    class Mt5News:
        name = "mt5"

        def is_available(self) -> bool:
            return True

    reset_research_registry()
    get_research_registry().register(Mt5News(), capabilities={NEWS})

    result = _unwrap(equity_profile)("AAPL", source="mt5")

    assert result["success"] is False
    assert result["error_code"] == "research_capability_unsupported"
