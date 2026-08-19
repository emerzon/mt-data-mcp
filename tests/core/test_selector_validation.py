from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from mtdata.core.analytics_requests import MarketRelativeStrengthRequest
from mtdata.core.news import news
from mtdata.core.pivot import confluence_levels, support_resistance_levels
from mtdata.core.symbols import market_scan
from mtdata.shared.annotations import get_runtime_annotations


def _unwrap(function):
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function


def test_relative_strength_rejects_empty_and_mixed_selectors() -> None:
    with pytest.raises(ValidationError, match="contains no symbols"):
        MarketRelativeStrengthRequest(symbols=" , ")
    with pytest.raises(ValidationError, match="cannot combine symbols with group"):
        MarketRelativeStrengthRequest(symbols="EURUSD,GBPUSD", group="Forex")

    assert MarketRelativeStrengthRequest().symbols is None


def test_market_scan_rejects_explicit_empty_symbols_before_io() -> None:
    result = _unwrap(market_scan)(symbols=" , ; ")

    assert result["success"] is False
    assert result["error_code"] == "empty_symbol_selector"


def test_news_rejects_explicit_empty_symbol() -> None:
    result = _unwrap(news)(symbol="  ")

    assert result["success"] is False
    assert result["error_code"] == "empty_symbol_selector"


@pytest.mark.parametrize("tool", [confluence_levels, support_resistance_levels])
def test_level_distance_cap_accepts_explicit_none(tool) -> None:
    annotation = get_runtime_annotations(_unwrap(tool))["max_distance_pct"]
    adapter = TypeAdapter(annotation)

    assert adapter.validate_python(None) is None
    assert adapter.validate_python(0.0) == 0.0
    with pytest.raises(ValidationError):
        adapter.validate_python(-0.1)
