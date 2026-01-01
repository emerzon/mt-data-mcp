from typing import Any, Dict, Optional
from ..schema import DenoiseSpec
from .common import build_report_with_timeframes
from ..report_utils import merge_params


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
        # Barrier optimization defaults
        'objective': 'ev',
        'params': {'spread_bps': 0.5, 'slippage_bps': 0.2, 'rr_min': 1.0, 'rr_max': 2.5},
    })
    if 'timeframe' not in p:
        p['timeframe'] = 'H4'
    return build_report_with_timeframes(symbol, horizon, denoise, p, default_extra=['H1','H4','D1','W1'], default_pivots=['D1','W1'])
