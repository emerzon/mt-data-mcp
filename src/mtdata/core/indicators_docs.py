"""Indicator discovery and doc-parsing helpers extracted from core.server.

This module delegates to utils.indicators to avoid duplicate discovery logic.
"""
from typing import Any, Dict, List, Optional

from ..utils.indicators import (
    clean_help_text,
    infer_defaults_from_doc,
    list_ta_indicators as _list_ta_indicators,
    _try_number,
)


def list_ta_indicators() -> List[Dict[str, Any]]:
    """Return [{'name','params','description','category'}, ...] discovered from pandas_ta."""
    return _list_ta_indicators(detailed=True)

