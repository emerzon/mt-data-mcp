"""
Indicator discovery and doc-parsing helpers extracted from core.server.

This module inspects pandas_ta to list indicators with params, categories,
and cleaned descriptions. No MCP tools are defined here.
"""
from typing import Any, Dict, List, Optional
import inspect
import pydoc
import re

try:
    import pandas_ta as pta
except ModuleNotFoundError:
    try:
        import pandas_ta_classic as pta
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "pandas_ta not found. Install 'pandas-ta-classic' (or 'pandas-ta')."
        ) from e


def clean_help_text(text: str, func_name: Optional[str] = None) -> str:
    if not isinstance(text, str):
        return ''
    cleaned = re.sub(r'.\x08', '', text)
    lines = [ln.rstrip() for ln in cleaned.splitlines()]
    sig_re = re.compile(rf"^\s*{re.escape(func_name)}\s*\(.*\)") if func_name else re.compile(r"^\s*\w+\s*\(.*\)")
    start = 0
    for i, ln in enumerate(lines):
        if sig_re.match(ln):
            start = i
            break
    kept = lines[start:]
    if kept:
        kept[0] = re.sub(r"\s+method of.*", "", kept[0], flags=re.IGNORECASE)
        if len(kept) > 1 and re.search(r"method of", kept[1], re.IGNORECASE):
            kept.pop(1)
    return "\n".join(kept).strip()


def _try_number(s: str):
    try:
        if '.' in s:
            return float(s)
        return int(s)
    except Exception:
        return None


def infer_defaults_from_doc(func_name: str, doc_text: str, params: List[Dict[str, Any]]):
    if not doc_text:
        return
    text = re.sub(r'.\x08', '', doc_text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sig_line = None
    for ln in lines:
        if ln.startswith(func_name + '(') or re.match(rf"^\s*{re.escape(func_name)}\s*\(.*\)", ln):
            sig_line = ln
            break
    if sig_line:
        inside = sig_line[sig_line.find('(') + 1 : sig_line.rfind(')')] if '(' in sig_line and ')' in sig_line else ''
        for part in re.split(r'[\s,]+', inside):
            if '=' in part:
                k, v = part.split('=', 1)
                k = k.strip()
                v = v.strip().strip(',)')
                num = _try_number(v)
                if num is not None:
                    for p in params:
                        if p.get('name') == k and 'default' not in p:
                            p['default'] = num
    for p in params:
        if 'default' in p:
            continue
        k = p.get('name')
        if not k:
            continue
        m = re.search(rf"{re.escape(k)}[^\n]*?(?:Default|default)\s*:?[\s]*([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            p['default'] = _try_number(m.group(1))


def list_ta_indicators() -> List[Dict[str, Any]]:
    """Return [{'name','params','description','category'}, ...] discovered from pandas_ta."""
    ind_list: List[Dict[str, Any]] = []
    seen = set()
    categories = ['candles', 'momentum', 'overlap', 'performance', 'statistics', 'trend', 'volatility', 'volume', 'cycles']
    for category in categories:
        try:
            cat_module = getattr(pta, category, None)
            if not cat_module or not hasattr(cat_module, '__file__'):
                continue
            for func_name in dir(cat_module):
                if func_name.startswith('_'):
                    continue
                func = getattr(cat_module, func_name, None)
                if not callable(func):
                    continue
                name = func_name.lower()
                if name in seen:
                    continue
                seen.add(name)
                try:
                    sig = inspect.signature(func)
                except (TypeError, ValueError):
                    continue
                params: List[Dict[str, Any]] = []
                for p in sig.parameters.values():
                    if p.name in {"open", "high", "low", "close", "volume"}:
                        continue
                    if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                        continue
                    entry: Dict[str, Any] = {"name": p.name}
                    if p.default is not inspect._empty and p.default is not None:
                        entry["default"] = p.default
                    params.append(entry)
                try:
                    raw = pydoc.render_doc(func)
                    desc = clean_help_text(raw, func_name=name)
                except Exception:
                    desc = inspect.getdoc(func) or ''
                try:
                    doc_text = inspect.getdoc(func) or raw if 'raw' in locals() else ''
                    infer_defaults_from_doc(name, doc_text, params)
                except Exception:
                    pass
                ind_list.append({
                    "name": name,
                    "params": params,
                    "description": desc,
                    "category": category,
                })
        except Exception:
            continue
    ind_list.sort(key=lambda x: x["name"])
    return ind_list
