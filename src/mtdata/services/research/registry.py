"""In-process registry of research source adapters."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .capabilities import PREFERRED_SOURCE_ORDER, RESEARCH_CAPABILITIES
from .errors import capability_unsupported_error, source_unavailable_error

_SOURCE_PIN_AUTO = frozenset({"", "auto", "all"})


class ResearchRegistry:
    """Map capability names to registered adapters.

    The same provider name may register a different adapter object per
    capability (Finviz news vs Finviz calendar).
    """

    def __init__(self) -> None:
        self._by_capability: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        source: Any,
        *,
        capabilities: Iterable[str],
    ) -> None:
        name = str(getattr(source, "name", "") or "").strip()
        if not name:
            raise ValueError("Research source must expose a non-empty name.")
        caps = frozenset(
            str(item).strip()
            for item in capabilities
            if str(item).strip()
        )
        unknown = caps - RESEARCH_CAPABILITIES
        if unknown:
            raise ValueError(
                "Unknown research capability: " + ", ".join(sorted(unknown))
            )
        if not caps:
            raise ValueError(f"Research source '{name}' must declare capabilities.")
        for capability in caps:
            bucket = self._by_capability.setdefault(capability, {})
            bucket[name] = source

    def known_names(self, capability: Optional[str] = None) -> list[str]:
        if capability:
            names = list(self._by_capability.get(str(capability).strip(), {}))
        else:
            names = {
                name
                for bucket in self._by_capability.values()
                for name in bucket
            }
            names = list(names)
        return _sorted_source_names(names)

    def available_names(self, capability: str) -> list[str]:
        capability_name = str(capability or "").strip()
        bucket = self._by_capability.get(capability_name, {})
        names = [
            name
            for name, source in bucket.items()
            if _source_available(source)
        ]
        return _sorted_source_names(names)

    def resolve(
        self,
        capability: str,
        source: Optional[str] = "auto",
    ) -> List[Any]:
        """Return available adapters for a capability, optionally pinned."""
        capability_name = str(capability or "").strip()
        bucket = self._by_capability.get(capability_name, {})
        available = [bucket[name] for name in self.available_names(capability_name)]
        pin = str(source or "auto").strip().lower()
        if pin in _SOURCE_PIN_AUTO:
            return available
        matched = [item for item in available if str(item.name) == pin]
        return matched

    def resolve_or_error(
        self,
        capability: str,
        source: Optional[str] = "auto",
        *,
        operation: Optional[str] = None,
    ) -> tuple[List[Any], Optional[Dict[str, Any]]]:
        """Resolve adapters or return a structured pin/capability error."""
        capability_name = str(capability or "").strip()
        pin = str(source or "auto").strip().lower() or "auto"
        available_names = self.available_names(capability_name)
        resolved = self.resolve(capability_name, pin)
        if resolved:
            return resolved, None
        known = self.known_names()
        if pin not in _SOURCE_PIN_AUTO and pin in known:
            return [], capability_unsupported_error(
                capability=capability_name,
                source=pin,
                available=available_names,
                operation=operation,
            )
        return [], source_unavailable_error(
            capability=capability_name,
            source=pin,
            available=available_names,
            operation=operation,
        )


def _source_available(source: Any) -> bool:
    checker = getattr(source, "is_available", None)
    if not callable(checker):
        return True
    try:
        return bool(checker())
    except Exception:
        return False


def _sorted_source_names(names: Iterable[str]) -> list[str]:
    preferred = {name: index for index, name in enumerate(PREFERRED_SOURCE_ORDER)}
    return sorted(
        names,
        key=lambda name: (preferred.get(name, len(preferred)), name),
    )


_REGISTRY: Optional[ResearchRegistry] = None


def get_research_registry() -> ResearchRegistry:
    """Return the process-wide research registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ResearchRegistry()
    return _REGISTRY


def reset_research_registry() -> ResearchRegistry:
    """Replace the process-wide registry. Tests use this to isolate adapters."""
    global _REGISTRY
    _REGISTRY = ResearchRegistry()
    return _REGISTRY
