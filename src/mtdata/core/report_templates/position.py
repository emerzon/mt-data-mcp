from typing import Any, Dict, Optional
from ..schema import DenoiseSpec
from .common import build_report_with_timeframes
from ..report_utils import merge_params


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
        # Barrier optimization defaults
        'objective': 'ev',
        'params': {'spread_bps': 0.3, 'slippage_bps': 0.1, 'rr_min': 1.2, 'rr_max': 3.0},
    })
    if 'timeframe' not in p:
        p['timeframe'] = 'D1'
    return build_report_with_timeframes(symbol, horizon, denoise, p, default_extra=['H4','D1','W1','MN1'], default_pivots=['W1','MN1'])
