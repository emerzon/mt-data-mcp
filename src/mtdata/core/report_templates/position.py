from typing import Any, Dict, Optional
from ..schema import DenoiseSpec
from .basic import template_basic
from ..report_utils import merge_params, attach_multi_timeframes


def template_position(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    p = merge_params(params, {
        'context_limit': 500,
        'context_tail': 120,
        'backtest_steps': 50,
        'backtest_spacing': 30,
        'backtest_rmse_tolerance': 0.05,
        'patterns_limit': 300,
        'mode': 'pct',
        'tp_min': 1.0, 'tp_max': 8.0, 'tp_steps': 8,
        'sl_min': 1.0, 'sl_max': 8.0, 'sl_steps': 8,
        'top_k': 5,
    })
    if 'timeframe' not in p:
        p['timeframe'] = 'D1'
    base = template_basic(symbol, horizon, denoise, p)
    attach_multi_timeframes(
        base,
        symbol,
        denoise,
        extra_timeframes=p.get('extra_timeframes') or ['H4','D1','W1','MN1'],
        pivot_timeframes=p.get('pivot_timeframes') or ['W1','MN1']
    )
    return base
