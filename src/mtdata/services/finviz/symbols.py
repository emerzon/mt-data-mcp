"""Finviz symbol classification helpers."""

_PAIR_SUFFIXES = frozenset(
    {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
)


def looks_like_non_equity_symbol(symbol: str) -> bool:
    """Return whether a symbol resembles a forex or namespaced instrument."""
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return False
    if "/" in normalized or ":" in normalized:
        return True
    return (
        len(normalized) == 6
        and normalized[:3].isalpha()
        and normalized[3:].isalpha()
        and normalized[3:] in _PAIR_SUFFIXES
    )
