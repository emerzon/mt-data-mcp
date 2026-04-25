from __future__ import annotations

from typing import Final

PARAMETER_HELP: Final[dict[str, str]] = {
    "symbol": "Trading symbol (e.g. EURUSD).",
    "symbols": "Comma-separated trading symbols (e.g. EURUSD,GBPUSD).",
    "group": (
        "MT5 symbol group path (for example Forex\\Majors). Mutually exclusive "
        "with explicit symbol selectors when supported."
    ),
    "timeframe": "MT5 timeframe (e.g. H1/M30/D1).",
}
