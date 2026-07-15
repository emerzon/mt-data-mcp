from __future__ import annotations

from typing import Any, AbstractSet

# Major G10-style fiat codes used for conservative pair detection and news.
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

# Extended FX codes for pip heuristics, weekend projection, and broader pair ID.
FOREX_CURRENCY_CODES = frozenset(
    {
        *FIAT_CURRENCY_CODES,
        "CNH",
        "CNY",
        "HKD",
        "MXN",
        "NOK",
        "SEK",
        "SGD",
        "ZAR",
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


def _alnum_upper(symbol: Any) -> str:
    return "".join(ch for ch in str(symbol or "").upper().strip() if ch.isalnum())


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


def is_probably_forex_symbol(
    symbol: Any,
    *,
    currency_codes: AbstractSet[str] | None = None,
) -> bool:
    """Return True when the symbol looks like a 6-letter FX pair.

    Defaults to major fiat codes. Pass ``FOREX_CURRENCY_CODES`` for the
    extended set used by pip/weekend heuristics.
    """
    codes = FIAT_CURRENCY_CODES if currency_codes is None else currency_codes
    normalized = _alnum_upper(symbol)
    if len(normalized) < 6:
        return False
    base = normalized[:3]
    quote = normalized[3:6]
    return base in codes and quote in codes
