from typing import Any, Dict, List, Optional

from ..schema import DenoiseSpec
from .basic import template_basic
from ..report_utils import attach_report_timeframes, attach_market_and_timeframes


def build_report_with_timeframes(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Dict[str, Any],
    *,
    default_extra: List[str],
    default_pivots: Optional[List[str]] = None,
) -> Dict[str, Any]:
    base = template_basic(symbol, horizon, denoise, params)
    attach_report_timeframes(base, symbol, denoise, params, default_extra=default_extra, default_pivots=default_pivots)
    return base


def build_report_with_market(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Dict[str, Any],
    *,
    default_extra: List[str],
    default_pivots: Optional[List[str]] = None,
    snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = template_basic(symbol, horizon, denoise, params)
    attach_market_and_timeframes(
        base, symbol, denoise, params,
        default_extra=default_extra, default_pivots=default_pivots, snapshot=snapshot,
    )
    return base
