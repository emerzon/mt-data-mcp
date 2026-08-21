from __future__ import annotations

from mtdata.core.screener import screener
from mtdata.services.research.registry import reset_research_registry


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_screener_list_filters_uses_finviz_catalog(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_filters_list",
        lambda **kwargs: {"success": True, "items": [{"filter": "Exchange"}]},
    )
    reset_research_registry()

    result = _unwrap(screener)(list_filters=True)

    assert result["success"] is True
    assert result["providers_used"] == ["finviz"]
    assert result["items"][0]["filter"] == "Exchange"
