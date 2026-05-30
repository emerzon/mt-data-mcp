from __future__ import annotations

from typing import Any

FIAT_CURRENCY_CODES = frozenset(
    {
        "AUD",
        "CAD",
        "CHF",
        "EUR",
        "GBP",
        "JPY",
        "NZD",
        "USD",
    }
)

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


def finviz_forex_symbol_to_mt5(symbol: Any) -> str | None:
    text = str(symbol or "").strip().upper()
    if not text:
        return None
    if "/" in text:
        left, right = text.split("/", 1)
    elif len(text) == 6:
        left, right = text[:3], text[3:]
    else:
        return None
    if left in FIAT_CURRENCY_CODES and right in FIAT_CURRENCY_CODES:
        return f"{left}{right}"
    return None


def is_probably_crypto_symbol(symbol: Any) -> bool:
    text = str(symbol or "").upper().strip()
    if not text:
        return False
    normalized = "".join(ch for ch in text if ch.isalnum())
    if not normalized:
        return False
    return any(token in normalized for token in CRYPTO_SYMBOL_HINTS)
