"""Structured errors for research source resolution."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def _available_source_values(available: Iterable[str]) -> list[str]:
    names = ["auto"]
    seen = {"auto"}
    for name in available:
        text = str(name or "").strip()
        if not text or text in seen:
            continue
        names.append(text)
        seen.add(text)
    return names


def source_unavailable_error(
    *,
    capability: str,
    source: str,
    available: Iterable[str],
    operation: Optional[str] = None,
) -> Dict[str, Any]:
    """Return an error when ``source=`` does not match an available adapter."""
    valid = _available_source_values(available)
    pin = str(source or "").strip() or "auto"
    return {
        "success": False,
        "error": (
            f"Research source '{pin}' is not available for {capability}."
            if pin != "auto"
            else f"No research sources are available for {capability}."
        ),
        "error_code": "research_source_unavailable",
        "capability": capability,
        "source": pin,
        "valid_values": {"source": valid},
        "operation": operation or capability,
        "remediation": (
            "Pass source=auto to use every available adapter, or pick a name "
            "from valid_values.source."
        ),
    }


def finviz_only_source_error(
    source: Optional[str],
    *,
    capability: str,
    operation: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return an error when a Finviz-only tool is pinned to another source."""
    pin = str(source or "auto").strip().lower() or "auto"
    if pin in {"", "auto", "finviz"}:
        return None
    available = ["finviz"]
    if pin == "mt5":
        return capability_unsupported_error(
            capability=capability,
            source=pin,
            available=available,
            operation=operation,
        )
    return source_unavailable_error(
        capability=capability,
        source=pin,
        available=available,
        operation=operation,
    )


def capability_unsupported_error(
    *,
    capability: str,
    source: str,
    available: Iterable[str],
    operation: Optional[str] = None,
) -> Dict[str, Any]:
    """Return an error when a pin exists but cannot serve this capability."""
    valid = _available_source_values(available)
    pin = str(source or "").strip()
    return {
        "success": False,
        "error": (
            f"Source '{pin}' does not support {capability}."
            if pin
            else f"No registered source supports {capability}."
        ),
        "error_code": "research_capability_unsupported",
        "capability": capability,
        "source": pin or None,
        "valid_values": {"source": valid},
        "operation": operation or capability,
        "remediation": (
            "Use source=auto, or pin a source listed in valid_values.source."
        ),
    }
