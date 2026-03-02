
from typing import Any, Dict, Optional, List
import re

from .schema import CategoryLiteral, IndicatorNameLiteral
from .constants import DEFAULT_ROW_LIMIT
from ..utils.utils import _table_from_rows
from .server import mcp
# Import the actual implementation from utils
from ..utils.indicators import list_ta_indicators as _list_ta_indicators

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


_DOC_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _/\-]{1,48})\s*:\s*$")
_DOC_PARAM_RE = re.compile(r"^[\-\*\u2022]?\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.+)$")
_DOC_SIG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(.*\)\s*$")


def _canonical_doc_section(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")
    aliases = {
        "arg": "parameters",
        "args": "parameters",
        "argument": "parameters",
        "arguments": "parameters",
        "param": "parameters",
        "params": "parameters",
        "kwargs": "parameters",
        "keyword_arguments": "parameters",
        "sources": "sources",
        "source": "sources",
        "references": "sources",
        "reference": "sources",
        "calculation": "calculation",
        "calculations": "calculation",
        "formula": "calculation",
        "formulas": "calculation",
        "interpretation": "interpretation",
        "interpretations": "interpretation",
        "notes": "interpretation",
        "signals": "interpretation",
    }
    return aliases.get(key, key)


def _parse_doc_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"overview": []}
    current = "overview"
    for raw_line in str(text or "").splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        match = _DOC_SECTION_RE.match(line)
        if match:
            current = _canonical_doc_section(match.group(1))
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _parse_parameter_docs(lines: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in lines or []:
        text = str(line or "").strip()
        if not text:
            continue
        match = _DOC_PARAM_RE.match(text)
        if not match:
            continue
        pname = str(match.group(1)).strip()
        pdesc = str(match.group(2)).strip()
        if pname and pdesc and pname not in out:
            out[pname] = pdesc
    return out


def _join_doc_lines(lines: List[str]) -> str:
    clean = [str(x).strip() for x in (lines or []) if str(x).strip()]
    return "\n".join(clean).strip()


def _extract_interpretation(sections: Dict[str, List[str]]) -> Optional[str]:
    explicit = _join_doc_lines(sections.get("interpretation", []))
    if explicit:
        return explicit
    overview = [ln for ln in sections.get("overview", []) if not _DOC_SIG_RE.match(str(ln or "").strip())]
    # Keep the first concise paragraph from overview as interpretation fallback.
    return _join_doc_lines(overview[:3]) or None


def _build_indicator_documentation(target: Dict[str, Any]) -> Dict[str, Any]:
    name = str(target.get("name") or "")
    raw_desc = str(target.get("description") or "")
    cleaned_desc = _clean_help_text(raw_desc, func_name=name) if raw_desc else ""
    sections = _parse_doc_sections(cleaned_desc)
    param_docs = _parse_parameter_docs(sections.get("parameters", []))

    params_out: List[Dict[str, Any]] = []
    for raw in (target.get("params") or []):
        if not isinstance(raw, dict):
            continue
        p = dict(raw)
        pname = str(p.get("name") or "").strip()
        if pname and pname in param_docs:
            p["description"] = param_docs[pname]
        params_out.append(p)

    calc_text = _join_doc_lines(sections.get("calculation", [])) or None
    interp_text = _extract_interpretation(sections)
    sources = []
    for item in sections.get("sources", []):
        src = re.sub(r"^[\-\*\u2022]\s*", "", str(item or "").strip())
        if src:
            sources.append(src)

    return {
        "description": cleaned_desc,
        "calculation": calc_text,
        "parameters": params_out,
        "interpretation": interp_text,
        "sources": sources,
    }

@mcp.tool()
def indicators_list(
    search_term: Optional[str] = None,
    category: Optional[CategoryLiteral] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
) -> Dict[str, Any]:  # type: ignore
    """List indicators as a tabular result with columns: name, category. Optional filters: search_term, category.

    Parameters: search_term?, category?, limit?
    """
    try:
        items = _list_ta_indicators(detailed=False)
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
        limit_value = None
        try:
            if limit is not None:
                limit_value = int(float(limit))
        except Exception:
            limit_value = None
        if limit_value and limit_value > 0:
            items = items[:limit_value]
        rows = [[it.get('name',''), it.get('category','')] for it in items]
        return _table_from_rows(["name", "category"], rows)
    except Exception as e:
        return {"error": f"Error listing indicators: {e}"}


# Note: category annotation is set at definition time above to be captured in the MCP schema

@mcp.tool()
def indicators_describe(name: IndicatorNameLiteral) -> Dict[str, Any]:  # type: ignore
    """Return detailed indicator information (name, category, params, description).

    Parameters: name
    """
    try:
        items = _list_ta_indicators(detailed=True)
        target = next((it for it in items if it.get('name','').lower() == str(name).lower()), None)
        if not target:
            return {"error": f"Indicator '{name}' not found"}
        indicator = dict(target)
        docs = _build_indicator_documentation(indicator)
        indicator["description"] = docs.get("description") or indicator.get("description") or ""
        indicator["documentation"] = {
            "calculation": docs.get("calculation"),
            "parameters": docs.get("parameters") or [],
            "interpretation": docs.get("interpretation"),
            "sources": docs.get("sources") or [],
        }
        return {"success": True, "indicator": indicator}
    except Exception as e:
        return {"error": f"Error getting indicator details: {e}"}


