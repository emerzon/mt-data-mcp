"""Crypto detection utilities for regime detection."""
from typing import Any

_CRYPTO_SYMBOL_HINTS = (
    "BTC",
    "ETH",
    "XRP",
    "LTC",
    "BCH",
    "DOGE",
    "SOL",
    "ADA",
    "DOT",
    "AVAX",
    "BNB",
    "TRX",
    "LINK",
    "MATIC",
)


def _is_probably_crypto_symbol(symbol: Any) -> bool:
    """Detect if symbol is likely a cryptocurrency based on common tickers."""
    s = str(symbol or "").upper().strip()
    if not s:
        return False
    normalized = "".join(ch for ch in s if ch.isalnum())
    if not normalized:
        return False
    return any(token in normalized for token in _CRYPTO_SYMBOL_HINTS)


__all__ = ["_is_probably_crypto_symbol", "_CRYPTO_SYMBOL_HINTS"]
