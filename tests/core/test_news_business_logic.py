from __future__ import annotations

from inspect import signature

from mtdata.core.news import news


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_news_tool_has_only_optional_symbol_parameter() -> None:
    raw = _unwrap(news)
    params = list(signature(raw).parameters.values())

    assert [param.name for param in params] == ["symbol"]
    assert params[0].default is None


def test_news_tool_forwards_symbol(monkeypatch) -> None:
    raw = _unwrap(news)

    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda symbol=None: {
            "success": True,
            "symbol": symbol,
            "general_news": [],
            "related_news": [],
        },
    )

    result = raw(symbol="EURUSD")

    assert result["success"] is True
    assert result["symbol"] == "EURUSD"
