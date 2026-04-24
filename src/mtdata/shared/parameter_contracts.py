from __future__ import annotations

from typing import Any, Callable, Final, Optional

PARAMETER_HELP: Final[dict[str, str]] = {
    "symbol": (
        "Trading symbol (e.g. EURUSD). Some multi-symbol selectors also accept "
        "`symbol` as a compatibility alias for `symbols`."
    ),
    "symbols": (
        "Comma-separated trading symbols (e.g. EURUSD,GBPUSD). Multi-symbol "
        "tools also accept `symbol` as a compatibility alias."
    ),
    "group": (
        "MT5 symbol group path (for example Forex\\Majors). Mutually exclusive "
        "with explicit symbol selectors when supported."
    ),
    "timeframe": "MT5 timeframe (e.g. H1/M30/D1).",
}


def _canonicalize_selector_tokens(tokens: list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = str(token or "").strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def normalize_symbol_selector_aliases(
    *,
    symbol: Optional[str],
    symbols: Optional[str],
    parse_selector: Callable[[Optional[str]], list[str]],
) -> tuple[list[str], dict[str, Any], Optional[str]]:
    """Normalize compatibility-safe `symbol` / `symbols` selectors.

    Returns canonical selector tokens, request metadata, and an optional
    validation error when both aliases are present but disagree.
    """

    meta: dict[str, Any] = {}
    symbol_tokens = list(parse_selector(symbol))
    symbols_tokens = list(parse_selector(symbols))

    if symbol_tokens:
        meta["symbol_input"] = list(symbol_tokens)
    if symbols_tokens:
        meta["symbols_input"] = list(symbols_tokens)

    if symbol_tokens and symbols_tokens:
        if _canonicalize_selector_tokens(symbol_tokens) != _canonicalize_selector_tokens(
            symbols_tokens
        ):
            return [], meta, (
                "Provide either symbol or symbols, not both, unless they resolve "
                "to the same selector set."
            )
        return list(symbols_tokens), meta, None

    if symbol_tokens:
        meta["symbols_input"] = list(symbol_tokens)
        return list(symbol_tokens), meta, None

    if symbols_tokens:
        return list(symbols_tokens), meta, None

    return [], meta, None
