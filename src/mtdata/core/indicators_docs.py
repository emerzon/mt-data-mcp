"""Indicator discovery and doc-parsing helpers extracted from core.server.

This module delegates to utils.indicators to avoid duplicate discovery logic.
"""

from typing import Any, Dict, List

from ..utils.indicators import (
    list_ta_indicators as _list_ta_indicators,
    clean_help_text as _clean_help_text_impl,
    infer_defaults_from_doc as _infer_defaults_impl,
    _try_number as _try_number_impl,
)


def list_ta_indicators() -> List[Dict[str, Any]]:
    """Return [{'name','params','description','category'}, ...] discovered from pandas_ta."""
    return _list_ta_indicators(detailed=True)


def clean_help_text(text: str, func_name: str | None = None) -> str:
    """Clean raw pydoc/inspect output for end-user indicator docs."""
    return _clean_help_text_impl(text, func_name=func_name)


def infer_defaults_from_doc(
    func_name: str, doc_text: str, params: List[Dict[str, Any]]
):
    """Infer parameter defaults from docstring text."""
    return _infer_defaults_impl(func_name, doc_text, params)


def _try_number(s: str):
    """Parse int/float from string, returning None on failure."""
    return _try_number_impl(s)
