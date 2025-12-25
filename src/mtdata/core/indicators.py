
from typing import Any, Dict, Optional, List

from .schema import CategoryLiteral, IndicatorNameLiteral
from ..utils.utils import _csv_from_rows
from .server import mcp
# Import the actual implementation from utils
from ..utils.indicators import _list_ta_indicators

def _infer_defaults_from_doc(func_name: str, doc_text: str, params: List[Dict[str, Any]]):
    """Delegate to utils implementation."""
    from ..core.indicators_docs import infer_defaults_from_doc as _impl
    return _impl(func_name, doc_text, params)

def _try_number(s: str):
    """Delegate to utils implementation.""" 
    from ..core.indicators_docs import _try_number as _impl
    return _impl(s)

def _clean_help_text(text: str, func_name: Optional[str] = None, func: Optional[Any] = None) -> str:
    """Delegate to utils implementation."""
    from ..core.indicators_docs import clean_help_text as _impl
    return _impl(text, func_name=func_name)

@mcp.tool()
def indicators_list(search_term: Optional[str] = None, category: Optional[CategoryLiteral] = None) -> Dict[str, Any]:  # type: ignore
    """List indicators as CSV with columns: name,category. Optional filters: search_term, category.

    Parameters: search_term?, category?
    """
    try:
        items = _list_ta_indicators()
        if search_term:
            q = search_term.strip().lower()
            filtered = []
            for it in items:
                name = it.get('name', '').lower()
                desc = (it.get('description') or '').lower()
                cat = (it.get('category') or '').lower()
                if q in name or q in desc or q in cat:
                    filtered.append(it)
            items = filtered
        if category:
            cat_q = category.strip().lower()
            items = [it for it in items if (it.get('category') or '').lower() == cat_q]
        items.sort(key=lambda x: (x.get('category') or '', x.get('name') or ''))
        rows = [[it.get('name',''), it.get('category','')] for it in items]
        return _csv_from_rows(["name", "category"], rows)
    except Exception as e:
        return {"error": f"Error listing indicators: {e}"}


# Note: category annotation is set at definition time above to be captured in the MCP schema

@mcp.tool()
def indicators_describe(name: IndicatorNameLiteral) -> Dict[str, Any]:  # type: ignore
    """Return detailed indicator information (name, category, params, description).

    Parameters: name
    """
    try:
        items = _list_ta_indicators()
        target = next((it for it in items if it.get('name','').lower() == str(name).lower()), None)
        if not target:
            return {"error": f"Indicator '{name}' not found"}
        return {"success": True, "indicator": target}
    except Exception as e:
        return {"error": f"Error getting indicator details: {e}"}


