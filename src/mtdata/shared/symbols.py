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

_CRYPTO_QUOTE_CODES = frozenset(
    {
        *FOREX_CURRENCY_CODES,
        "USDT",
        "USDC",
        "BUSD",
        "DAI",
        "BTC",
        "ETH",
    }
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
    normalized = _alnum_upper(symbol)
    if not normalized:
        return False
    # Classify only a recognizable base/quote pair. Substring containment made
    # equities such as SOLV, ATOM and UNIT look like 24/7 crypto instruments.
    for base in sorted(CRYPTO_SYMBOL_HINTS, key=len, reverse=True):
        if not normalized.startswith(base):
            continue
        remainder = normalized[len(base) :]
        for quote in sorted(_CRYPTO_QUOTE_CODES, key=len, reverse=True):
            if remainder.startswith(quote):
                return True
    return False


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
