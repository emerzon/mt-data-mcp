from __future__ import annotations

import logging
from unittest.mock import patch

from mtdata.core import finviz as core_finviz
from mtdata.core.finviz import finviz_market_news, finviz_news
from mtdata.services.finviz import get_stock_news


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

    with patch("mtdata.services.finviz._apply_finvizfinance_timeout_patch", lambda: None), patch.dict(
        "sys.modules",
        {"finvizfinance.quote": type("Q", (), {"finvizfinance": Boom})},
    ):
        out = get_stock_news("BTCUSD", limit=5, page=1)

    assert out["error"] == "BTCUSD is not a Finviz-supported symbol. finviz_news only covers US equities."


def test_finviz_news_logs_finish_event_for_success(caplog) -> None:
    raw = _unwrap(finviz_news)

    with patch("mtdata.core.finviz.get_stock_news", return_value={"success": True, "items": []}), caplog.at_level(
        logging.INFO,
        logger=core_finviz.logger.name,
    ):
        out = raw(symbol="AAPL", limit=5, page=1)

    assert out["success"] is True
    assert any(
        "event=finish" in record.message and "operation=finviz_news" in record.message
        for record in caplog.records
    )


def test_finviz_news_adds_normalized_items_alias_for_stock_results() -> None:
    raw = _unwrap(finviz_news)

    service_result = {
        "success": True,
        "symbol": "AAPL",
        "count": 1,
        "total": 1,
        "page": 1,
        "pages": 1,
        "news": [
            {
                "Title": "  Apple launches new chips  ",
                "Source": " Reuters ",
                "Date": " 2026-04-18 ",
                "Link": " https://example.test/apple ",
            }
        ],
    }

    with patch("mtdata.core.finviz.get_stock_news", return_value=service_result):
        out = raw(symbol="AAPL", limit=5, page=1)

    assert out["news"] == service_result["news"]
    assert out["items"] == [
        {
            "title": "Apple launches new chips",
            "source": "Reuters",
            "published_at": "2026-04-18",
            "url": "https://example.test/apple",
        }
    ]


def test_finviz_news_without_symbol_keeps_general_news_shape() -> None:
    raw = _unwrap(finviz_news)
    service_result = {
        "success": True,
        "type": "news",
        "count": 1,
        "items": [{"Title": "Market wrap"}],
    }

    with patch("mtdata.core.finviz.get_general_news", return_value=service_result):
        out = raw(symbol=None, limit=5, page=1)

    assert out == service_result


def test_finviz_news_helpers_are_registered_tools() -> None:
    assert hasattr(finviz_news, "__wrapped__")
    assert hasattr(finviz_market_news, "__wrapped__")
