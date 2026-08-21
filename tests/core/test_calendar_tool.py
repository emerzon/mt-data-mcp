from __future__ import annotations

import pytest

from mtdata.core.calendar import calendar
from mtdata.services.research.capabilities import CALENDAR
from mtdata.services.research.registry import reset_research_registry


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_calendar_range_uses_finviz_adapter(monkeypatch) -> None:
    captured = {}

    def _fake_run_finviz_calendar(**kwargs):
        captured.update(kwargs)
        return {"success": True, "items": [{"title": "CPI"}], "count": 1}

    monkeypatch.setattr(
        "mtdata.core.finviz.run_finviz_calendar",
        _fake_run_finviz_calendar,
    )
    reset_research_registry()

    result = _unwrap(calendar)(kind="economic", impact="high", currency="USD")

    assert result["success"] is True
    assert result["providers_used"] == ["finviz"]
    assert result["provider"] == "finviz"
    assert captured["calendar"] == "economic"
    assert captured["impact"] == "high"
    assert captured["currency"] == "USD"


def test_calendar_period_view_requires_earnings() -> None:
    reset_research_registry()

    result = _unwrap(calendar)(kind="economic", view="period")

    assert result["success"] is False
    assert result["error_code"] == "calendar_invalid_view"


def test_calendar_mt5_pin_is_capability_unsupported() -> None:
    from mtdata.services.research.capabilities import NEWS
    from mtdata.services.research.registry import get_research_registry

    class Mt5News:
        name = "mt5"

        def is_available(self) -> bool:
            return True

    reset_research_registry()
    get_research_registry().register(Mt5News(), capabilities={NEWS})

    result = _unwrap(calendar)(source="mt5")

    assert result["success"] is False
    assert result["error_code"] == "research_capability_unsupported"
    assert result["capability"] == CALENDAR
    assert "finviz" in result["valid_values"]["source"]


def test_calendar_period_view_rejects_range_controls(monkeypatch) -> None:
    reset_research_registry()
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_earnings",
        lambda **_kwargs: pytest.fail("period view must not fetch with range controls"),
    )

    result = _unwrap(calendar)(
        kind="earnings",
        view="period",
        period="next-week",
        start="2000-01-01",
        end="2000-01-02",
    )

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert "start" in result["details"]["invalid"]
    assert "end" in result["details"]["invalid"]


def test_calendar_range_view_rejects_period_controls(monkeypatch) -> None:
    reset_research_registry()
    monkeypatch.setattr(
        "mtdata.core.finviz.run_finviz_calendar",
        lambda **_kwargs: pytest.fail("range view must not fetch with period controls"),
    )

    result = _unwrap(calendar)(
        kind="earnings",
        view="range",
        period="next-week",
    )

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["invalid"] == ["period"]


def test_calendar_period_view_uses_earnings_alias(monkeypatch) -> None:
    def _fake_earnings(**kwargs):
        return {
            "success": True,
            "period": kwargs.get("period"),
            "items": [{"ticker": "AAPL"}],
            "count": 1,
        }

    monkeypatch.setattr("mtdata.core.finviz.finviz_earnings", _fake_earnings)
    reset_research_registry()

    result = _unwrap(calendar)(
        kind="earnings",
        view="period",
        period="this-week",
    )

    assert result["success"] is True
    assert result["period"] == "this-week"
    assert result["providers_used"] == ["finviz"]
