"""Indicator discovery and doc-parsing helpers extracted from core.server.

This module delegates to utils.indicators to avoid duplicate discovery logic.
"""
from typing import Any, Dict, List

from ..utils.indicators import (
    list_ta_indicators as _list_ta_indicators,
)


def list_ta_indicators() -> List[Dict[str, Any]]:
    """Return [{'name','params','description','category'}, ...] discovered from pandas_ta."""
    return _list_ta_indicators(detailed=True)
