"""
Internal Finviz research adapters.

Public tools are domain-named (`news`, `calendar`, `equity_profile`,
`screener`, `asset_performance`). This module keeps Finviz fetch/normalize
helpers and is not part of the MCP catalog.
Note: Data is delayed 15-20 minutes; US stocks only.
"""

from mtdata.core.finviz.calendar import (
    finviz_calendar,
    finviz_earnings,
    run_finviz_calendar,
)
from mtdata.core.finviz.fundamentals import (
    finviz_description,
    finviz_fundamentals,
)
from mtdata.core.finviz.insider import (
    finviz_insider,
    finviz_insider_activity,
    finviz_peers,
    finviz_ratings,
)
from mtdata.core.finviz.markets import (
    finviz_crypto,
    finviz_forex,
    finviz_futures,
)
from mtdata.core.finviz.news import (
    finviz_market_news,
    finviz_news,
)
from mtdata.core.finviz.screen import (
    finviz_filters_list,
    finviz_screen,
)

__all__ = [
    "finviz_calendar",
    "finviz_crypto",
    "finviz_description",
    "finviz_earnings",
    "finviz_filters_list",
    "finviz_forex",
    "finviz_fundamentals",
    "finviz_futures",
    "finviz_insider",
    "finviz_insider_activity",
    "finviz_market_news",
    "finviz_news",
    "finviz_peers",
    "finviz_ratings",
    "finviz_screen",
    "run_finviz_calendar",
]
