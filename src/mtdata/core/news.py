"""Unified news MCP tool."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..services.unified_news import fetch_unified_news
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation

logger = logging.getLogger(__name__)


@mcp.tool()
def news(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch important general news and, optionally, symbol-relevant news.

    With no symbol, returns the most important recent general news from all
    available sources.

    With a symbol, returns two buckets:
    - `general_news`: important recent market-wide items.
    - `related_news`: items relevant to the instrument, including direct symbol
      news when available, asset-specific market snapshots, and macro events
      whose text and metadata suggest likely impact on the instrument.
    - `impact_news`: high-importance systemic headlines, such as war or energy
      shocks, that may matter even when they are not direct lexical matches.
    - `upcoming_events`: future economic-calendar items relevant to the
      instrument, surfaced separately so scheduled releases are easy to spot.
    - `recent_events`: the latest relevant economic releases, surfaced
      separately so actual values are easy to scan.

    Matching uses symbol aliases, asset-class terms, MT5 symbol metadata, and a
    lightweight cosine-similarity score over headline/event text.

    Parameters
    ----------
    symbol : str, optional
        Instrument to contextualize the news for, such as `AAPL`, `EURUSD`, or
        `BTCUSD`.

    Returns
    -------
    dict
        Unified response containing:
        - `instrument`: inferred symbol context when `symbol` is provided
        - `general_news`: important recent general news
        - `related_news`: symbol-relevant news and events
        - `impact_news`: high-importance systemic market headlines
        - `upcoming_events`: future scheduled events relevant to the instrument
        - `recent_events`: latest relevant scheduled releases for the instrument
        - `source_details`: per-source candidate and selected counts
        - `matching`: summary of the relevance model
    """

    def _run() -> Dict[str, Any]:
        return fetch_unified_news(symbol=symbol)

    return run_logged_operation(
        logger,
        operation="news",
        symbol=symbol,
        func=_run,
    )
