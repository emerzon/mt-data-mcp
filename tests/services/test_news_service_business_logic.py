from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from mtdata.services import news_service as svc


def _record(timestamp: datetime, subject: str = "Fed preview"):
    return SimpleNamespace(
        timestamp=timestamp,
        category="FXStreet",
        source="FXStreet",
        subject=subject,
        to_dict=lambda now=None: {
            "subject": subject,
            "category": "FXStreet",
            "source": "FXStreet",
            "published_at": timestamp.isoformat(),
        },
    )


def test_get_mt5_news_surfaces_warning_for_inverted_date_range(monkeypatch) -> None:
    class FakeParser:
        header_info = {"version": 1}

        def __init__(self, path: str) -> None:
            assert path == "C:/tmp/news.dat"

        def parse(self):
            return [_record(datetime(2026, 3, 15, tzinfo=timezone.utc))]

    monkeypatch.setattr(svc, "MT5NewsParser", FakeParser)
    monkeypatch.setattr(svc, "_use_client_tz", lambda: False)

    result = svc.get_mt5_news(
        news_db_path="C:/tmp/news.dat",
        from_date="2026-03-20",
        to_date="2026-03-10",
    )

    assert result["success"] is True
    assert result["count"] == 0
    assert result["news"] == []
    assert result["warning"] == "from_date is after to_date; returning no results"


def test_get_mt5_news_reports_invalid_date_filter_as_input_error(monkeypatch) -> None:
    class FakeParser:
        header_info = {"version": 1}

        def __init__(self, path: str) -> None:
            assert path == "C:/tmp/news.dat"

        def parse(self):
            return [_record(datetime(2026, 3, 15, tzinfo=timezone.utc))]

    monkeypatch.setattr(svc, "MT5NewsParser", FakeParser)
    monkeypatch.setattr(svc, "_use_client_tz", lambda: False)

    result = svc.get_mt5_news(
        news_db_path="C:/tmp/news.dat",
        from_date="not-a-date",
    )

    assert "Invalid news date filter" in result["error"]
    assert "ISO dates" in result["hint"]
