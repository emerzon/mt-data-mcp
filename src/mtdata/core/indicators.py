import logging
import re
from typing import Any, Dict, List, Literal, Optional

from ..utils.indicators import _try_number
from ..utils.indicators import clean_help_text as _clean_help_text
from ..utils.indicators import infer_defaults_from_doc as _infer_defaults_from_doc
from ..utils.indicators import list_ta_indicators as _list_ta_indicators
from ..utils.utils import _table_from_rows
from ._mcp_instance import mcp
from ..shared.constants import DEFAULT_ROW_LIMIT
from .execution_logging import run_logged_operation
from ..shared.schema import CategoryLiteral, CompactFullDetailLiteral, IndicatorNameLiteral

logger = logging.getLogger(__name__)

_DOC_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _/\-]{1,48})\s*:\s*$")
_DOC_PARAM_RE = re.compile(r"^[\-\*\u2022]?\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.+)$")
_DOC_SIG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(.*\)\s*(?:->\s*.+)?$")


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
    current_param: Optional[str] = None
    for line in lines or []:
        text = str(line or "").strip()
        if not text:
            continue
        match = _DOC_PARAM_RE.match(text)
        if match:
            pname = str(match.group(1)).strip()
            pdesc = str(match.group(2)).strip()
            if not pname or not pdesc:
                current_param = None
                continue
            if pname in out:
                out[pname] = f"{out[pname]} {pdesc}".strip()
            else:
                out[pname] = pdesc
            current_param = pname
            continue
        if current_param and not _DOC_SECTION_RE.match(text):
            out[current_param] = f"{out[current_param]} {text}".strip()
            continue
        current_param = None
    return out


def _join_doc_lines(lines: List[str]) -> str:
    clean = [str(x).strip() for x in (lines or []) if str(x).strip()]
    return "\n".join(clean).strip()


def _clean_overview_lines(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    in_signature = False
    paren_depth = 0
    for raw in lines or []:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.lower().startswith("python library documentation:"):
            continue
        if _DOC_SIG_RE.match(text):
            continue
        if not cleaned and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\($", text):
            in_signature = True
            paren_depth = text.count("(") - text.count(")")
            continue
        if in_signature:
            paren_depth += text.count("(") - text.count(")")
            if paren_depth <= 0:
                in_signature = False
            continue
        cleaned.append(text)
    return cleaned


def _extract_interpretation(sections: Dict[str, List[str]]) -> Optional[str]:
    explicit = _join_doc_lines(sections.get("interpretation", []))
    if explicit:
        return explicit
    overview = _clean_overview_lines(sections.get("overview", []))
    return _join_doc_lines(overview) or None


def _extract_short_description(description: Optional[str]) -> Optional[str]:
    """Extract first line or first sentence from description for compact display."""
    if not description:
        return None
    text = str(description or "").strip()
    if not text:
        return None
    # Get first line
    lines = text.split('\n')
    first_line = lines[0].strip()
    if not first_line:
        return None
    # Truncate to reasonable length for compact display (around 80 chars)
    if len(first_line) > 80:
        # Try to break at a word boundary
        truncated = first_line[:77]
        last_space = truncated.rfind(' ')
        if last_space > 40:  # Only break if we have at least 40 chars
            return truncated[:last_space].strip() + "..."
        return truncated.strip() + "..."
    return first_line


def _build_indicator_documentation(target: Dict[str, Any]) -> Dict[str, Any]:
    name = str(target.get("name") or "")
    raw_desc = str(target.get("description") or "")
    cleaned_desc = _clean_help_text(raw_desc, func_name=name) if raw_desc else ""
    sections = _parse_doc_sections(cleaned_desc)
    overview = _clean_overview_lines(sections.get("overview", []))
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
        "description": _join_doc_lines(overview) or cleaned_desc,
        "calculation": calc_text,
        "parameters": params_out,
        "interpretation": interp_text,
        "sources": sources,
    }


def _indicator_search_rank(item: Dict[str, Any], query: str) -> tuple[int, str] | None:
    q = str(query or "").strip().lower()
    if not q:
        return None

    name = str(item.get("name") or "").strip().lower()
    aliases = [
        str(alias).strip().lower()
        for alias in (item.get("aliases") or [])
        if str(alias).strip()
    ]
    category = str(item.get("category") or "").strip().lower()

    if name == q:
        return (0, name)
    if q in aliases:
        return (1, name)
    if name.startswith(q):
        return (2, name)
    if any(alias.startswith(q) for alias in aliases):
        return (3, name)
    if q in name:
        return (4, name)
    if any(q in alias for alias in aliases):
        return (5, name)
    if category == q:
        return (6, name)
    if category.startswith(q):
        return (7, name)
    if q in category:
        return (8, name)
    return None


def _format_indicator_param_summary(params: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for raw in params or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        if "default" in raw:
            parts.append(f"{name}={raw.get('default')}")
        else:
            parts.append(name)
        if len(parts) >= 4:
            break
    return ",".join(parts)

@mcp.tool()
def indicators_list(
    search_term: Optional[str] = None,
    category: Optional[CategoryLiteral] = None,
    limit: Optional[int] = DEFAULT_ROW_LIMIT,
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:  # type: ignore
    """List indicators as a tabular result with optional search, category, and detail filters.

    Parameters: search_term?, category?, limit?, detail?
    """
    def _run() -> Dict[str, Any]:
        try:
            detail_mode = str(detail or "compact").strip().lower()
            if detail_mode not in {"compact", "full"}:
                detail_mode = "compact"
            detailed = detail_mode == "full"
            items = _list_ta_indicators(detailed=detailed)
            search_active = False
            if search_term:
                q = search_term.strip().lower()
                ranked = []
                for it in items:
                    rank = _indicator_search_rank(it, q)
                    if rank is not None:
                        ranked.append((rank, it))
                items = [it for _rank, it in sorted(ranked, key=lambda pair: pair[0])]
                search_active = True
            if category:
                cat_q = category.strip().lower()
                items = [it for it in items if (it.get('category') or '').lower() == cat_q]
            if not search_active:
                items.sort(key=lambda x: (x.get('category') or '', x.get('name') or ''))
            total_matches = len(items)
            limit_value = None
            try:
                if limit is not None:
                    limit_value = int(float(limit))
            except Exception:
                limit_value = None
            if limit_value and limit_value > 0:
                items = items[:limit_value]
            if detailed:
                rows = []
                for it in items:
                    docs = _build_indicator_documentation(it)
                    params = docs.get("parameters") or it.get("params") or []
                    rows.append(
                        [
                            it.get("name", ""),
                            it.get("category", ""),
                            _extract_short_description(docs.get("description") or it.get("description", "")),
                            len(params),
                            params,
                            ", ".join(str(alias) for alias in (it.get("aliases") or []) if str(alias).strip()),
                            docs.get("description") or it.get("description", ""),
                        ]
                    )
                result = _table_from_rows(
                    ["name", "category", "summary", "params_count", "params", "aliases", "description"],
                    rows,
                )
            else:
                rows = [
                    [
                        it.get('name',''),
                        it.get('category',''),
                        _extract_short_description(it.get('description', '')),
                        len(it.get("params") or []),
                        _format_indicator_param_summary(it.get("params") or []),
                    ]
                    for it in items
                ]
                result = _table_from_rows(["name", "category", "description", "params_count", "params"], rows)
            result["detail"] = detail_mode
            if total_matches > len(items):
                result["total_count"] = total_matches
                result["more_available"] = total_matches - len(items)
                result["truncated"] = True
                result["show_all_hint"] = "Set limit to a higher value to view more matching indicators."
            return result
        except Exception as exc:
            return {"error": f"Error listing indicators: {exc}"}

    return run_logged_operation(
        logger,
        operation="indicators_list",
        search_term=search_term,
        category=category,
        limit=limit,
        detail=detail,
        func=_run,
    )


# Note: category annotation is set at definition time above to be captured in the MCP schema

@mcp.tool()
def indicators_describe(name: IndicatorNameLiteral) -> Dict[str, Any]:  # type: ignore
    """Return detailed indicator information (name, category, params, description).

    Parameters: name
    """
    def _run() -> Dict[str, Any]:
        try:
            items = _list_ta_indicators(detailed=True)
            target = next(
                (
                    it
                    for it in items
                    if it.get('name','').lower() == str(name).lower()
                    or str(name).lower() in {
                        str(alias).strip().lower()
                        for alias in (it.get("aliases") or [])
                        if str(alias).strip()
                    }
                ),
                None,
            )
            if not target:
                return {"error": f"Indicator '{name}' not found"}
            indicator = dict(target)
            docs = _build_indicator_documentation(indicator)
            indicator["description"] = docs.get("description") or indicator.get("description") or ""
            indicator["params"] = docs.get("parameters") or indicator.get("params") or []
            indicator["documentation"] = {
                "calculation": docs.get("calculation"),
                "interpretation": docs.get("interpretation"),
                "sources": docs.get("sources") or [],
            }
            return {"success": True, "indicator": indicator}
        except Exception as exc:
            return {"error": f"Error getting indicator details: {exc}"}

    return run_logged_operation(
        logger,
        operation="indicators_describe",
        name=name,
        func=_run,
    )


