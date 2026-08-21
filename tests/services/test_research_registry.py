from __future__ import annotations

from mtdata.services.research.capabilities import CALENDAR, NEWS
from mtdata.services.research.registry import ResearchRegistry, reset_research_registry


class _FakeSource:
    def __init__(self, name: str, *, available: bool = True) -> None:
        self.name = name
        self._available = available

    def is_available(self) -> bool:
        return self._available


def test_registry_prefers_finviz_then_mt5() -> None:
    registry = ResearchRegistry()
    registry.register(_FakeSource("ycnbc"), capabilities={NEWS})
    registry.register(_FakeSource("mt5"), capabilities={NEWS})
    registry.register(_FakeSource("finviz"), capabilities={NEWS})

    assert [source.name for source in registry.resolve(NEWS)] == [
        "finviz",
        "mt5",
        "ycnbc",
    ]


def test_registry_keeps_same_provider_name_on_separate_capabilities() -> None:
    registry = ResearchRegistry()
    news = _FakeSource("finviz")
    calendar = _FakeSource("finviz")
    registry.register(news, capabilities={NEWS})
    registry.register(calendar, capabilities={CALENDAR})

    assert registry.resolve(NEWS, "finviz")[0] is news
    assert registry.resolve(CALENDAR, "finviz")[0] is calendar


def test_registry_pin_returns_capability_error_for_wrong_family() -> None:
    registry = ResearchRegistry()
    registry.register(_FakeSource("mt5"), capabilities={NEWS})
    registry.register(_FakeSource("finviz"), capabilities={CALENDAR})

    adapters, error = registry.resolve_or_error(CALENDAR, "mt5", operation="calendar")

    assert adapters == []
    assert error is not None
    assert error["error_code"] == "research_capability_unsupported"
    assert error["valid_values"]["source"] == ["auto", "finviz"]


def test_registry_unknown_pin_returns_unavailable() -> None:
    registry = ResearchRegistry()
    registry.register(_FakeSource("finviz"), capabilities={NEWS})

    adapters, error = registry.resolve_or_error(NEWS, "yahoo")

    assert adapters == []
    assert error is not None
    assert error["error_code"] == "research_source_unavailable"
    assert "finviz" in error["valid_values"]["source"]


def test_registry_skips_unavailable_adapters() -> None:
    registry = ResearchRegistry()
    registry.register(_FakeSource("finviz", available=False), capabilities={NEWS})
    registry.register(_FakeSource("mt5"), capabilities={NEWS})

    assert [source.name for source in registry.resolve(NEWS)] == ["mt5"]


def test_reset_research_registry_replaces_process_singleton() -> None:
    first = reset_research_registry()
    first.register(_FakeSource("finviz"), capabilities={NEWS})
    second = reset_research_registry()

    assert first is not second
    assert second.available_names(NEWS) == []
