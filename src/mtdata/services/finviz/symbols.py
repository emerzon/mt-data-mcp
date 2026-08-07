"""Finviz symbol classification helpers."""

import re

_PAIR_SUFFIXES = frozenset(
    {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
)
_EQUITY_BROKER_SUFFIXES = frozenset(
    {
        "AMEX",
        "ARCA",
        "ASE",
        "BATS",
        "L",
        "NAS",
        "NASDAQ",
        "NQ",
        "NY",
        "NYSE",
        "NYS",
        "NYQ",
        "O",
        "OTC",
        "TQ",
        "US",
    }
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


def normalize_finviz_equity_symbol(symbol: str) -> str:
    """Strip a recognized MT5 broker suffix from a Finviz equity ticker.

    Broker symbol names commonly append an exchange or routing suffix with a
    dot, underscore, or hyphen. Unknown suffixes are retained so exchange
    share-class tickers such as ``BRK.B`` are not rewritten.
    """
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return normalized

    match = re.fullmatch(r"([A-Z0-9]{1,6})[._-]([A-Z0-9]+)(?:[._-].*)?", normalized)
    if match is None:
        return normalized
    base, suffix = match.groups()
    if suffix not in _EQUITY_BROKER_SUFFIXES or looks_like_non_equity_symbol(base):
        return normalized
    return base
