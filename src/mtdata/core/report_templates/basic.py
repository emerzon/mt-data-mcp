from typing import Any, Dict, Optional, List
from ..schema import DenoiseSpec
from ..report_utils import now_utc_iso, parse_csv_tail, pick_best_forecast_method, summarize_barrier_grid


def _get_raw_result(func, *args, **kwargs):
    """Call function and return raw dict, handling both wrapped and unwrapped cases."""
    try:
        # First try calling the function normally
        result = func(*args, **kwargs)
        
        # If it returns a dict, use it directly
        if isinstance(result, dict):
            return result
            
        # If it returns a formatted string, parse what we can
        if isinstance(result, str):
            # Try to parse key-value format or structured output
            return _parse_formatted_output(result)
        
        return {'error': f'Unexpected result type: {type(result)}'}
        
    except Exception as e:
        return {'error': f'Function call failed: {str(e)}'}


def _parse_formatted_output(output: str) -> Dict[str, Any]:
    """Parse formatted string output back into structured data."""
    try:
        lines = [line.rstrip() for line in output.split('\n')]  # Keep leading spaces
        result = {}
        current_section = None
        csv_data = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if not line.strip():
                i += 1
                continue
                
            # Key-value pairs like "symbol: BTCUSD"
            if ':' in line and not line.startswith(' ') and ',' not in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                
                # Handle sections that have subsections
                if key in ['period', 'best', 'summary', 'levels'] and not val:
                    current_section = key
                    result[key] = {}
                    # Look ahead for CSV data or nested structure
                    i += 1
                    
                    if key == 'levels':
                        # Special handling for levels - expect CSV data
                        csv_data = []
                        while i < len(lines):
                            next_line = lines[i]
                            stripped_line = next_line.strip()
                            if not stripped_line:
                                i += 1
                                continue
                            # Look for CSV data (may be indented)
                            if ',' in stripped_line and (stripped_line.startswith('level,') or 
                                all(c.isdigit() or c in '.,- ' or c.isalpha() for c in stripped_line[:20])):
                                csv_data.append(stripped_line)
                                i += 1
                            elif stripped_line and not next_line.startswith(' ') and ':' in stripped_line and ',' not in stripped_line:
                                # New section starting
                                break
                            else:
                                i += 1
                        
                        if csv_data:
                            result[key] = _parse_csv_data(csv_data)
                        i -= 1  # Back up one since we'll increment at end of loop
                    else:
                        # Handle other nested sections
                        while i < len(lines) and (lines[i].startswith(' ') or not lines[i].strip()):
                            next_line = lines[i]
                            if next_line.startswith(' ') and ':' in next_line:
                                sub_key, sub_val = next_line.strip().split(':', 1)
                                result[current_section][sub_key.strip()] = _parse_value(sub_val.strip())
                            i += 1
                        i -= 1  # Back up one
                        
                elif current_section:
                    # Add to current section
                    result[current_section][key] = _parse_value(val)
                else:
                    result[key] = _parse_value(val)
                    
            # Standalone CSV data
            elif ',' in line and not line.startswith(' '):
                csv_data = [line]
                i += 1
                # Collect following CSV rows
                while i < len(lines) and ',' in lines[i] and not (lines[i].count(':') > lines[i].count(',')):
                    csv_data.append(lines[i])
                    i += 1
                
                if csv_data:
                    parsed_csv = _parse_csv_data(csv_data)
                    if current_section:
                        result[current_section] = parsed_csv
                    else:
                        result['data'] = parsed_csv
                i -= 1  # Back up one
                        
            i += 1
        
        # If we couldn't parse anything meaningful, return error
        if not result:
            result = {'error': 'Could not parse formatted output', 'raw_output': output[:200]}
            
        return result
    except Exception as e:
        return {'error': f'Failed to parse output: {str(e)}', 'raw_output': output[:200]}


def _parse_value(val: str) -> Any:
    """Parse a string value into appropriate type."""
    val = val.strip()
    if not val or val.lower() in ['null', 'none', '']:
        return None
    if val.lower() in ['true', 'yes']:
        return True
    if val.lower() in ['false', 'no']:
        return False
    try:
        if '.' in val or 'e' in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val


def _parse_csv_data(csv_lines: List[str]) -> Any:
    """Parse CSV lines into structured data."""
    if not csv_lines:
        return None
        
    try:
        # First line should be headers
        headers = [h.strip() for h in csv_lines[0].split(',')]
        
        if len(csv_lines) == 1:
            return {'headers': headers}
            
        # Parse data rows
        rows = []
        for line in csv_lines[1:]:
            if not line.strip():
                continue
            values = [v.strip() for v in line.split(',')]
            row = {}
            for i, header in enumerate(headers):
                val = values[i] if i < len(values) else ''
                row[header] = _parse_value(val) if val else None
            rows.append(row)
            
        return rows if rows else {'headers': headers}
    except Exception:
        return csv_lines


def _unwrap_mcp_function(func):
    """Legacy function - now just returns the original function."""
    return func


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
    from ..data import data_fetch_candles
    
    ctx = _get_raw_result(data_fetch_candles,
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
        # Try to parse the CSV data if available
        csv_header = ctx.get('csv_header') if isinstance(ctx, dict) else None
        csv_data = ctx.get('csv_data') if isinstance(ctx, dict) else None
        if csv_header and csv_data:
            tail_rows = parse_csv_tail(csv_header, csv_data, tail=int(p.get('context_tail', 40)))
        else:
            # Fallbacks when calling through minimal formatter
            if isinstance(ctx, dict) and isinstance(ctx.get('data'), list):
                tail_rows = ctx.get('data')[-int(p.get('context_tail', 40)):]  # type: ignore[index]
            elif isinstance(ctx, list):
                tail_rows = ctx
            else:
                tail_rows = []
        
        last = tail_rows[-1] if tail_rows else {}
        clos: List[float] = []
        for r in (tail_rows[-30:] if tail_rows else []):
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
    from ..pivot import pivot_compute_points

    piv = _get_raw_result(pivot_compute_points, symbol=symbol, timeframe='D1')

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
    from ..forecast import forecast_volatility_estimate
    vol = _get_raw_result(forecast_volatility_estimate, symbol=symbol, timeframe=tf, horizon=int(horizon), method='ewma', params={'lambda_': 0.94})
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
    from ..forecast import forecast_backtest_run
    methods = p.get('methods')
    bt = _get_raw_result(forecast_backtest_run, symbol=symbol, timeframe=tf, horizon=int(horizon), steps=steps, spacing=spacing, methods=methods)
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
        from ..forecast import forecast_generate
        fc = _get_raw_result(forecast_generate, symbol=symbol, timeframe=tf, method=best_name, horizon=int(horizon))
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
    from ..forecast import forecast_barrier_optimize
    mode_val = str(p.get('mode', 'pct'))
    grid = _get_raw_result(forecast_barrier_optimize,
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
        params=p.get('params'),
        objective=str(p.get('objective','ev_uncond')),
        return_grid=False,
        top_k=int(p.get('top_k', 5)),
        output='summary',
    )
    if 'error' in grid:
        report['sections']['barriers'] = {'error': grid['error']}
    else:
        report['sections']['barriers'] = summarize_barrier_grid(grid, top_k=int(p.get('top_k', 5)))

    # Patterns
    from ..patterns import patterns_detect_candlesticks
    pats = _get_raw_result(patterns_detect_candlesticks, symbol=symbol, timeframe=tf, limit=int(p.get('patterns_limit', 120)))
    if 'error' in pats:
        report['sections']['patterns'] = {'error': pats['error']}
    else:
        rows = parse_csv_tail(pats.get('csv_header'), pats.get('csv_data'), tail=20)
        detections = rows[-5:] if rows else []
        report['sections']['patterns'] = {'recent': detections}

    return report
