import re
from typing import Any, Callable, Optional, Sequence


def _normalize_symbol_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _normalize_group_path_query(value: str) -> str:
    text = str(value or "").strip().replace("/", "\\")
    text = re.sub(r"\\+", r"\\", text)
    return text.strip("\\")


def _extract_group_path(sym) -> str:
    """Extract pure group path from a symbol, stripping the symbol name if present.

    MT5 sometimes reports `symbol.path` including the symbol at the tail. This trims the
    last component when it equals the symbol name (case-insensitive).
    """
    raw = getattr(sym, 'path', '') or ''
    name = getattr(sym, 'name', '') or ''
    if not raw:
        return 'Unknown'
    parts = raw.split('\\')
    tail = parts[-1] if parts else ""
    tail_norm = _normalize_symbol_token(tail)
    name_norm = _normalize_symbol_token(name)
    tail_matches_symbol = bool(
        tail
        and name
        and (
            tail.lower() == name.lower()
            or tail_norm == name_norm
            or (
                len(name_norm) >= 3
                and tail_norm.startswith(name_norm)
                and any(sep in tail for sep in (".", "-", "_"))
            )
        )
    )
    if parts and tail_matches_symbol:
        parts = parts[:-1]
    group = '\\'.join(parts).strip('\\')
    return group or 'Unknown'


def match_symbol_infos(
    symbols: Sequence[Any],
    query: str,
    *,
    limit: int = 5,
    group_of: Optional[Callable[[Any], str]] = None,
    sort_key: Optional[Callable[[Any], Any]] = None,
) -> list[Any]:
    """Return symbol infos whose name/description/group contain ``query``."""
    text = str(query or "").strip()
    if not text:
        return []
    query_upper = text.upper()
    matches: list[Any] = []
    for info in symbols:
        name = str(getattr(info, "name", "") or "")
        description = str(getattr(info, "description", "") or "")
        if group_of is not None:
            group = str(group_of(info) or "")
        else:
            group = str(getattr(info, "path", "") or "")
        if query_upper in f"{name} {description} {group}".upper():
            matches.append(info)
    if sort_key is None:
        matches.sort(
            key=lambda info: (
                not str(getattr(info, "name", "") or "").upper().startswith(query_upper),
                str(getattr(info, "name", "") or "").casefold(),
            )
        )
    else:
        matches.sort(key=sort_key)
    return list(matches[: max(1, int(limit))])
