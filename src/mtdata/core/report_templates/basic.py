from typing import Any, Dict, Optional, List
from ..schema import DenoiseSpec
from ..report_utils import now_utc_iso, parse_csv_tail, pick_best_forecast_method, summarize_barrier_grid


def template_basic(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    p = dict(params or {})
    tf = str(p.get('timeframe', 'H1'))
    report: Dict[str, Any] = {
        'meta': {
            'symbol': symbol,
            'timeframe': tf,
            'horizon': int(horizon),
            'template': 'basic',
            'generated_at': now_utc_iso(),
        },
        'sections': {},
    }

    # Context
    indicators = "ema(20),ema(50),rsi(14),macd(12,26,9)"
    from ..data import data_fetch_candles as _fetch_candles
    ctx = _fetch_candles(
        symbol=symbol,
        timeframe=tf,
        limit=int(p.get('context_limit', 300)),
        indicators=indicators,  # type: ignore[arg-type]
        denoise=denoise,
        simplify={'mode': 'select', 'method': 'lttb', 'ratio': 0.2},  # type: ignore[arg-type]
    )
    if 'error' in ctx:
        report['sections']['context'] = {'error': ctx['error']}
    else:
        tail_rows = parse_csv_tail(ctx.get('csv_header'), ctx.get('csv_data'), tail=int(p.get('context_tail', 40)))
        last = tail_rows[-1] if tail_rows else {}
        clos: List[float] = []
        for r in tail_rows[-30:]:
            v = r.get('close')
            try:
                clos.append(float(v))
            except Exception:
                continue
        report['sections']['context'] = {
            'last_snapshot': last,
            'sparkline_close': clos,
            'notes': 'Indicators included: EMA(20), EMA(50), RSI(14), MACD(12,26,9).',
        }

    # Pivots (D1)
    from ..pivot import pivot_compute_points as _compute_pivot_points
    piv = _compute_pivot_points(symbol=symbol, timeframe='D1')
    if 'error' in piv:
        report['sections']['pivot'] = {'error': piv['error']}
    else:
        report['sections']['pivot'] = {
            'levels': piv.get('levels'),
            'methods': piv.get('methods'),
            'source': piv.get('source'),
            'period': piv.get('period'),
        }

    # Volatility (EWMA)
    from ..forecast import forecast_volatility_estimate as _forecast_volatility
    vol = _forecast_volatility(symbol=symbol, timeframe=tf, horizon=int(horizon), method='ewma', params={'lambda_': 0.94})
    if 'error' in vol:
        report['sections']['volatility'] = {'error': vol['error']}
    else:
        report['sections']['volatility'] = {
            'sigma_bar_return': vol.get('sigma_bar_return'),
            'horizon_sigma_return': vol.get('horizon_sigma_return'),
            'sigma_bar_price': vol.get('sigma_bar_price'),
            'horizon_sigma_price': vol.get('horizon_sigma_price'),
            'method': vol.get('method'),
        }

    # Backtest select best
    steps = int(p.get('backtest_steps', 25))
    spacing = int(p.get('backtest_spacing', 10))
    try:
        rmse_tol = float(p.get('backtest_rmse_tolerance', 0.05))
    except Exception:
        rmse_tol = 0.05
    from ..forecast import forecast_backtest_run as _forecast_backtest
    methods = p.get('methods')
    bt = _forecast_backtest(symbol=symbol, timeframe=tf, horizon=int(horizon), steps=steps, spacing=spacing, methods=methods)
    sec_bt: Dict[str, Any]
    if 'error' in bt:
        sec_bt = {'error': bt['error']}
        best = None
    else:
        ranking: List[Dict[str, Any]] = []
        try:
            res = bt.get('results', {})
            for m, r in res.items():
                if not isinstance(r, dict):
                    continue
                ranking.append({
                    'method': m,
                    'avg_rmse': r.get('avg_rmse'),
                    'avg_mae': r.get('avg_mae'),
                    'avg_directional_accuracy': r.get('avg_directional_accuracy'),
                    'successful_tests': r.get('successful_tests'),
                })
            ranking.sort(key=lambda x: (float(x.get('avg_rmse') or 1e9), -float(x.get('avg_directional_accuracy') or 0.0)))
        except Exception:
            pass
        topk = int(p.get('backtest_top_k', 3))
        sec_bt = {'ranking': ranking[:max(1, topk)], 'horizon': int(horizon), 'steps': steps, 'spacing': spacing}
        best = pick_best_forecast_method(bt, rmse_tolerance=rmse_tol)
    report['sections']['backtest'] = sec_bt

    if best is not None:
        best_name, best_stats = best
        from ..forecast import forecast_generate as _forecast
        fc = _forecast(symbol=symbol, timeframe=tf, method=best_name, horizon=int(horizon))
        if 'error' in fc:
            report['sections']['forecast'] = {'error': fc['error'], 'method': best_name}
        else:
            report['sections']['forecast'] = {
                'method': best_name,
                'forecast_price': fc.get('forecast_price'),
                'lower_price': fc.get('lower_price'),
                'upper_price': fc.get('upper_price'),
                'trend': fc.get('trend'),
                'ci_alpha': fc.get('ci_alpha'),
            }
        report['sections']['backtest']['best_method'] = {'method': best_name, 'stats': {
            'avg_rmse': best_stats.get('avg_rmse'),
            'avg_mae': best_stats.get('avg_mae'),
            'avg_directional_accuracy': best_stats.get('avg_directional_accuracy'),
            'successful_tests': best_stats.get('successful_tests'),
        }}

    # Barriers (grid)
    from ..forecast import forecast_barrier_optimize as _barrier_optimize
    mode_val = str(p.get('mode', 'pct'))
    grid = _barrier_optimize(
        symbol=symbol,
        timeframe=tf,
        horizon=int(horizon),
        method='hmm_mc',
        mode=mode_val,
        tp_min=float(p.get('tp_min', 0.2)),
        tp_max=float(p.get('tp_max', 1.0)),
        tp_steps=int(p.get('tp_steps', 5)),
        sl_min=float(p.get('sl_min', 0.2)),
        sl_max=float(p.get('sl_max', 1.0)),
        sl_steps=int(p.get('sl_steps', 5)),
        return_grid=False,
        top_k=int(p.get('top_k', 5)),
        output='summary',
    )
    if 'error' in grid:
        report['sections']['barriers'] = {'error': grid['error']}
    else:
        report['sections']['barriers'] = summarize_barrier_grid(grid, top_k=int(p.get('top_k', 5)))

    # Patterns
    from ..patterns import patterns_detect_candlesticks as _detect_candles_patterns
    pats = _detect_candles_patterns(symbol=symbol, timeframe=tf, limit=int(p.get('patterns_limit', 120)))
    if 'error' in pats:
        report['sections']['patterns'] = {'error': pats['error']}
    else:
        rows = parse_csv_tail(pats.get('csv_header'), pats.get('csv_data'), tail=20)
        detections = rows[-5:] if rows else []
        report['sections']['patterns'] = {'recent': detections}

    return report
