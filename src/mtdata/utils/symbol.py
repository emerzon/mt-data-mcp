import re


def _normalize_symbol_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _extract_group_path(sym) -> str:
    """Extract pure group path from a symbol, stripping the symbol name if present.

    MT5 sometimes reports `symbol.path` including the symbol at the tail. This trims the
    last component when it equals the symbol name (case-insensitive).
    """
    raw = getattr(sym, "path", "") or ""
    name = getattr(sym, "name", "") or ""
    if not raw:
        return "Unknown"
    parts = raw.split("\\")
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
    group = "\\".join(parts).strip("\\")
    return group or "Unknown"
