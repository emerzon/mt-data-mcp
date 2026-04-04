from __future__ import annotations

from typing import Any

CRYPTO_SYMBOL_HINTS = (
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
    "NEAR",
    "ATOM",
    "FIL",
    "UNI",
)


def is_probably_crypto_symbol(symbol: Any) -> bool:
    text = str(symbol or "").upper().strip()
    if not text:
        return False
    normalized = "".join(ch for ch in text if ch.isalnum())
    if not normalized:
        return False
    return any(token in normalized for token in CRYPTO_SYMBOL_HINTS)
