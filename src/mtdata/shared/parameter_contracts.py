from __future__ import annotations

from typing import Final

OUTPUT_EXTRAS: Final[frozenset[str]] = frozenset(
    {
        "metadata",
        "diagnostics",
        "request",
        "raw",
        "raw_rows",
        "method_docs",
    }
)
OUTPUT_EXTRA_FULL_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "all",
        "full",
        "verbose",
    }
)
PUBLIC_OUTPUT_PARAMS: Final[frozenset[str]] = frozenset({"json", "extras"})
REMOVED_PUBLIC_OUTPUT_PARAMS: Final[frozenset[str]] = frozenset(
    {"detail", "format", "output_mode", "output"}
)
OUTPUT_EXTRAS_HELP: Final[str] = (
    "Comma-separated richer output sections such as "
    f"{', '.join(sorted(OUTPUT_EXTRAS))}. Use "
    f"{'/'.join(sorted(OUTPUT_EXTRA_FULL_ALIASES))} for every supported section."
)

PARAMETER_HELP: Final[dict[str, str]] = {
    "symbol": "Trading symbol (e.g. EURUSD).",
    "symbols": "Comma-separated trading symbols (e.g. EURUSD,GBPUSD).",
    "group": (
        "MT5 symbol group path (for example Forex\\Majors). Mutually exclusive "
        "with explicit symbol selectors when supported."
    ),
    "timeframe": "MT5 timeframe (e.g. H1/M30/D1).",
    "max_lag": "Maximum lag in bars for lagged statistical tests.",
    "significance": "p-value threshold for statistical significance.",
    "normalize": "Normalize input price series before analysis.",
    "trend": "Cointegration trend term to include in the test model.",
    "tp_ticks": "Take-profit barrier distance in ticks.",
    "sl_ticks": "Stop-loss barrier distance in ticks.",
    "rsi_length": "RSI lookback period in bars.",
    "sma_period": "Simple moving-average lookback period in bars.",
    "min_price_change_pct": "Minimum price-change filter in percent.",
    "max_price_change_pct": "Maximum price-change filter in percent.",
    "max_spread_pct": "Maximum bid/ask spread filter in percent of price.",
    "min_tick_volume": "Minimum tick-volume filter.",
    "rsi_below": "Filter to symbols with RSI below this threshold.",
    "rsi_above": "Filter to symbols with RSI above this threshold.",
    "price_vs_sma": "Filter by price position versus SMA: above, below, or any.",
    "strategy": "Backtest strategy to run.",
    "position_mode": "Trade direction mode for generated strategy positions.",
    "fast_period": "Fast moving-average period in bars.",
    "slow_period": "Slow moving-average period in bars.",
    "oversold": "RSI oversold threshold.",
    "overbought": "RSI overbought threshold.",
    "max_hold_bars": "Maximum bars to hold each simulated trade.",
    "proxy": "Volatility proxy source or model.",
    "vol_sl_multiplier": "Stop-loss distance as a multiple of estimated volatility.",
    "vol_floor_ticks": "Minimum volatility-derived barrier distance in ticks.",
    "min_prob_win": "Minimum acceptable probability of hitting the profit barrier first.",
    "max_prob_no_hit": "Maximum acceptable probability that neither barrier is hit.",
    "max_median_time": "Maximum median time-to-barrier estimate in bars.",
    "output_mode": "Barrier optimization output shape.",
    "json": "Return structured JSON instead of default TOON text.",
    "extras": OUTPUT_EXTRAS_HELP,
    "detail": "Removed output option; compact output is implicit and richer sections use extras.",
    "format": "Domain-specific shape selector when supported; TOON/JSON selection uses json.",
}
