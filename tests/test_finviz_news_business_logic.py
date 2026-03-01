from __future__ import annotations

from unittest.mock import patch

from mtdata.core.finviz import finviz_news
from mtdata.services.finviz_service import get_stock_news


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_finviz_news_rejects_pair_style_symbol_before_service_call() -> None:
    raw = _unwrap(finviz_news)
    with patch("mtdata.core.finviz.get_stock_news") as mock_news:
        out = raw(symbol="BTCUSD", limit=5, page=1)

    assert "error" in out
    assert "not a Finviz-supported equity ticker" in out["error"]
    mock_news.assert_not_called()


def test_get_stock_news_returns_clean_message_for_404_like_errors() -> None:
    class Boom:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("404 Client Error: Not Found for url: https://finviz.com/quote.ashx?t=BTCUSD")

    with patch("mtdata.services.finviz_service._apply_finvizfinance_timeout_patch", lambda: None), patch.dict(
        "sys.modules",
        {"finvizfinance.quote": type("Q", (), {"finvizfinance": Boom})},
    ):
        out = get_stock_news("BTCUSD", limit=5, page=1)

    assert out["error"] == "BTCUSD is not a Finviz-supported symbol. finviz_news only covers US equities."
