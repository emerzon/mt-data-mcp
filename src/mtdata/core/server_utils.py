"""Server utilities and helper functions."""

from typing import Optional, Set


def coerce_scalar(s: str) -> str:
    """Try to coerce a scalar string to int or float; otherwise return original string.

    Args:
        s: String to coerce

    Returns:
        Coerced value (int, float, or original string)
    """
    try:
        if s is None:
            return s
        st = str(s).strip()
        if st == "":
            return st
        if st.isdigit() or (st.startswith('-') and st[1:].isdigit()):
            return int(st)
        v = float(st)
        return v
    except Exception:
        return s


def normalize_ohlcv_arg(ohlcv: Optional[str]) -> Optional[Set[str]]:
    """Normalize user-provided OHLCV selection into a set of letters.

    Accepts forms like: 'close', 'price', 'ohlc', 'ohlcv', 'all', 'cl', 'OHLCV',
    or names 'open,high,low,close,volume'. Returns None when not specified.

    Args:
        ohlcv: OHLCV specification string

    Returns:
        Set of OHLCV letters or None if not specified
    """
    if ohlcv is None:
        return None
    text = str(ohlcv).strip()
    if text == "":
        return None
    t = text.lower()
    if t in ("all", "ohlcv"):
        return {"O", "H", "L", "C", "V"}
    if t in ("ohlc",):
        return {"O", "H", "L", "C"}
    if t in ("price", "close"):
        return {"C"}
    # Compact letters like 'cl', 'oh', etc.
    if all(ch in "ohlcv" for ch in t):
        return {ch.upper() for ch in t}
    # Comma separated names
    parts = [p.strip().lower() for p in t.replace(";", ",").split(",") if p.strip() != ""]
    if not parts:
        return None
    mapping = {
        "o": "O", "open": "O",
        "h": "H", "high": "H",
        "l": "L", "low": "L",
        "c": "C", "close": "C", "price": "C",
        "v": "V", "vol": "V", "volume": "V", "tick_volume": "V",
    }
    out: Set[str] = set()
    for p in parts:
        key = mapping.get(p)
        if key:
            out.add(key)
    return out or None