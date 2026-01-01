from typing import Any, Optional, Tuple, Literal, Dict

from .mt5 import get_symbol_info_cached


def get_pip_size(symbol: str, symbol_info: Optional[Any] = None) -> Optional[float]:
    """Return the tick size for a symbol based on MT5 symbol info."""
    try:
        info = symbol_info or get_symbol_info_cached(symbol)
        if info is None:
            return None
        tick_size = getattr(info, "trade_tick_size", None)
        try:
            tick_size = float(tick_size) if tick_size is not None else None
        except Exception:
            tick_size = None
        if tick_size is None or tick_size <= 0:
            point = float(getattr(info, "point", 0.0) or 0.0)
            if point <= 0:
                return None
            tick_size = point
        return float(tick_size)
    except Exception:
        return None


def resolve_barrier_prices(
    *,
    price: float,
    direction: Literal["long", "short"] = "long",
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_pips: Optional[float] = None,
    sl_pips: Optional[float] = None,
    pip_size: Optional[float] = None,
    adjust_inverted: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve TP/SL barrier prices from absolute, percentage, or tick offsets."""
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(str(value))
        except Exception:
            return None

    tp_price = _coerce_float(tp_abs)
    sl_price = _coerce_float(sl_abs)
    r_tp = _coerce_float(tp_pct)
    r_sl = _coerce_float(sl_pct)
    p_tp = _coerce_float(tp_pips)
    p_sl = _coerce_float(sl_pips)

    dir_long = str(direction).lower() == "long"

    if tp_price is None:
        if r_tp is not None:
            tp_price = price * (1.0 + (r_tp / 100.0)) if dir_long else price * (1.0 - (r_tp / 100.0))
        elif p_tp is not None and pip_size is not None and pip_size > 0:
            tp_price = price + p_tp * pip_size if dir_long else price - p_tp * pip_size

    if sl_price is None:
        if r_sl is not None:
            sl_price = price * (1.0 - (r_sl / 100.0)) if dir_long else price * (1.0 + (r_sl / 100.0))
        elif p_sl is not None and pip_size is not None and pip_size > 0:
            sl_price = price - p_sl * pip_size if dir_long else price + p_sl * pip_size

    if tp_price is None or sl_price is None:
        return None, None

    if adjust_inverted:
        if dir_long:
            if tp_price <= price:
                tp_price = price * 1.000001
            if sl_price >= price:
                sl_price = price * 0.999999
        else:
            if tp_price >= price:
                tp_price = price * 0.999999
            if sl_price <= price:
                sl_price = price * 1.000001

    return float(tp_price), float(sl_price)


def build_barrier_kwargs(
    *,
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_pips: Optional[float] = None,
    sl_pips: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Collect barrier arguments into a single kwargs dict."""
    return {
        "tp_abs": tp_abs,
        "sl_abs": sl_abs,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "tp_pips": tp_pips,
        "sl_pips": sl_pips,
    }


def build_barrier_kwargs_from(values: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Build barrier kwargs from a dict of values (e.g., locals())."""
    return build_barrier_kwargs(
        tp_abs=values.get("tp_abs"),
        sl_abs=values.get("sl_abs"),
        tp_pct=values.get("tp_pct"),
        sl_pct=values.get("sl_pct"),
        tp_pips=values.get("tp_pips"),
        sl_pips=values.get("sl_pips"),
    )
