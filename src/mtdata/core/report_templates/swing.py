from typing import Any, Dict, Optional
from ..schema import DenoiseSpec
from .basic import template_basic
from ..report_utils import merge_params, attach_multi_timeframes


def template_swing(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    p = merge_params(params, {
        'context_limit': 400,
        'context_tail': 80,
        'backtest_steps': 40,
        'backtest_spacing': 20,
        'backtest_rmse_tolerance': 0.05,
        'patterns_limit': 250,
        'mode': 'pct',
        'tp_min': 0.5, 'tp_max': 3.0, 'tp_steps': 6,
        'sl_min': 0.5, 'sl_max': 3.0, 'sl_steps': 6,
        'top_k': 5,
    })
    if 'timeframe' not in p:
        p['timeframe'] = 'H4'
    base = template_basic(symbol, horizon, denoise, p)
    attach_multi_timeframes(
        base,
        symbol,
        denoise,
        extra_timeframes=p.get('extra_timeframes') or ['H1','H4','D1','W1'],
        pivot_timeframes=p.get('pivot_timeframes') or ['D1','W1']
    )
    return base
