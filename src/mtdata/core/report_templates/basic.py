from typing import Any, Dict, Optional, List

from math import isfinite
from ..schema import DenoiseSpec
from ..report_utils import now_utc_iso, parse_csv_tail, pick_best_forecast_method, summarize_barrier_grid, attach_multi_timeframes


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


# --- Compact trend metrics -------------------------------------------------

def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(x)
        return v if isfinite(v) else default
    except Exception:
        return default


def _ema(values: List[float], length: int) -> List[float]:
    if length <= 1 or not values:
        return list(values)
    k = 2.0 / (length + 1.0)
    out: List[float] = []
    ema_val = values[0]
    out.append(ema_val)
    for v in values[1:]:
        ema_val = ema_val + k * (v - ema_val)
        out.append(ema_val)
    return out


def _compute_tr(high: List[float], low: List[float], close: List[float]) -> List[float]:
    n = len(close)
    if n == 0:
        return []
    tr: List[float] = []
    prev_close = close[0]
    for i in range(n):
        h = high[i] if i < len(high) and high[i] is not None else prev_close
        l = low[i] if i < len(low) and low[i] is not None else prev_close
        c = close[i] if close[i] is not None else prev_close
        a = abs(h - l)
        b = abs(h - prev_close)
        d = abs(l - prev_close)
        tr_val = max(a, b, d)
        tr.append(tr_val)
        prev_close = c
    return tr


def _linreg_slope_r2(series: List[float]) -> Optional[tuple]:
    try:
        n = len(series)
        if n < 2:
            return None
        x = list(range(n))
        sx = sum(x)
        sy = sum(series)
        sxx = sum(i * i for i in x)
        sxy = sum(i * y for i, y in zip(x, series))
        denom = n * sxx - sx * sx
        if denom == 0:
            return None
        slope = (n * sxy - sx * sy) / denom
        # R^2 via correlation
        mean_x = sx / n
        mean_y = sy / n
        num = sum((i - mean_x) * (y - mean_y) for i, y in zip(x, series))
        denx = sum((i - mean_x) ** 2 for i in x)
        deny = sum((y - mean_y) ** 2 for y in series)
        r2 = 0.0
        if denx > 0 and deny > 0:
            r = num / ((denx ** 0.5) * (deny ** 0.5))
            r2 = float(r * r)
        return slope, r2
    except Exception:
        return None


def _percentile_rank(values: List[float], current: float) -> int:
    try:
        if not values:
            return 0
        sorted_vals = sorted(v for v in values if isfinite(v))
        if not sorted_vals:
            return 0
        # rank = percentage of values <= current
        cnt = 0
        for v in sorted_vals:
            if v <= current:
                cnt += 1
        pct = int(round(100.0 * cnt / len(sorted_vals)))
        return max(0, min(100, pct))
    except Exception:
        return 0


def _compute_compact_trend(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows or len(rows) < 5:
        return None
    closes: List[Optional[float]] = [_safe_float(r.get('close')) for r in rows]
    highs: List[Optional[float]] = [_safe_float(r.get('high')) for r in rows]
    lows: List[Optional[float]] = [_safe_float(r.get('low')) for r in rows]
    # Replace None with previous close where possible
    clean_close: List[float] = []
    lastc = None
    for c in closes:
        if c is None:
            c = lastc if lastc is not None else 0.0
        clean_close.append(c)
        lastc = c
    clean_high = [h if h is not None else c for h, c in zip(highs, clean_close)]
    clean_low = [l if l is not None else c for l, c in zip(lows, clean_close)]

    # ATR(14) in price units
    tr = _compute_tr(clean_high, clean_low, clean_close)
    atr_series = _ema(tr, 14)
    atr = atr_series[-1] if atr_series else 0.0
    last_price = clean_close[-1] if clean_close else 0.0

    # Slope windows
    wins = [5, 20, 60]
    s_vals: List[int] = []
    r_vals: List[int] = []
    for w in wins:
        seg = clean_close[-w:] if len(clean_close) >= w else clean_close
        if len(seg) < 2:
            s_vals.append(0)
            r_vals.append(0)
            continue
        # Use log price for scale invariance
        import math
        logs = [math.log(max(1e-12, v)) for v in seg]
        fit = _linreg_slope_r2(logs)
        if not fit:
            s_vals.append(0)
            r_vals.append(0)
            continue
        slope, r2 = fit
        # Normalize by ATR% (ATR/price) to get slope in ATR-per-bar units
        atr_pct = (atr / last_price) if (last_price and atr) else 0.0
        norm = (slope / atr_pct) if atr_pct > 0 else 0.0
        s_vals.append(int(round(norm * 100)))  # scale to int
        r_vals.append(int(round(max(0.0, min(1.0, r2)) * 100)))

    # Squeeze via Bollinger Bandwidth percentile
    import statistics
    L = 20
    M = 60
    widths: List[float] = []
    if len(clean_close) >= L:
        for i in range(max(0, len(clean_close) - M), len(clean_close) - L + 1):
            window = clean_close[i:i + L]
            try:
                mid = sum(window) / L
                std = statistics.pstdev(window) if len(window) > 1 else 0.0
                width = (2.0 * 2.0 * std) / mid if mid > 0 else 0.0  # 2*sigma bands
            except Exception:
                width = 0.0
            widths.append(width)
    q = 0
    if widths:
        curr_width = widths[-1]
        q = _percentile_rank(widths, curr_width)

    # Regime code
    s5, s20 = s_vals[0], s_vals[1] if len(s_vals) > 1 else (s_vals[0] if s_vals else 0)
    r20 = r_vals[1] if len(r_vals) > 1 else 0
    # Donchian breakout check
    g = 0
    if len(clean_high) >= 21 and len(clean_low) >= 21:
        prev_high = max(clean_high[-21:-1])
        prev_low = min(clean_low[-21:-1])
        eps = 1e-9
        if last_price >= prev_high - eps and s5 > 0:
            g = 3
        elif last_price <= prev_low + eps and s5 < 0:
            g = 4
    # Trend if no breakout
    if g == 0:
        if s20 > 8 and r20 >= 40:
            g = 1
        elif s20 < -8 and r20 >= 40:
            g = 2
        else:
            g = 0

    # Distances to recent swing high/low (bars since)
    lookback = min(60, len(clean_close))
    h_idx = 0
    l_idx = 0
    if lookback >= 2:
        segment_h = clean_high[-lookback:]
        segment_l = clean_low[-lookback:]
        try:
            last_peak = max(range(len(segment_h)), key=lambda i: segment_h[i])
            last_trough = min(range(len(segment_l)), key=lambda i: segment_l[i])
            h_idx = (lookback - 1) - last_peak
            l_idx = (lookback - 1) - last_trough
        except Exception:
            h_idx = 0
            l_idx = 0

    # ATR% of price as basis points (bps)
    v = int(round(((atr / last_price) * 10000.0) if last_price > 0 and atr > 0 else 0.0))

    return {
        's': s_vals,   # slopes ATRu*100
        'r': r_vals,   # R2%
        'v': v,        # ATR% of price
        'q': int(q),   # squeeze percentile
        'g': int(g),   # regime code
        'h': int(h_idx),
        'l': int(l_idx),
    }


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
        tail_n = int(p.get('context_tail', 40))
        if csv_header and csv_data:
            tail_rows = parse_csv_tail(csv_header, csv_data, tail=tail_n)
        else:
            # Fallbacks when calling through minimal formatter
            if isinstance(ctx, dict) and isinstance(ctx.get('data'), list):
                tail_rows = ctx.get('data')[-tail_n:]  # type: ignore[index]
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
        compact = _compute_compact_trend(tail_rows)
        ctx_obj: Dict[str, Any] = {
            'symbol': symbol,
            'timeframe': tf,
            'last_snapshot': last,
            'sparkline_close': clos,
            'notes': 'Indicators included: EMA(20), EMA(50), RSI(14), MACD(12,26,9).',
        }
        if compact:
            ctx_obj['trend_compact'] = compact
        report['sections']['context'] = ctx_obj

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
            'timeframe': 'D1',
        }
        # Attach multi-timeframe context and pivots for MTF alignment (lightweight)
        try:
            attach_multi_timeframes(report, symbol, denoise, extra_timeframes=['M15','H1','H4','D1'], pivot_timeframes=['H4','D1'])
        except Exception:
            pass


    # Fallback: if MTF sections missing, build minimal ones inline
    try:
        secs = report.setdefault('sections', {})
        if 'contexts_multi' not in secs or 'pivot_multi' not in secs:
            from ..report_utils import context_for_tf
            tf_list = ['M15','H1','H4','D1']
            ctxs: Dict[str, Any] = {}
            for tf_i in tf_list:
                snap = context_for_tf(symbol, tf_i, denoise, limit=200, tail=30)
                if snap:
                    ctxs[tf_i] = snap
            if ctxs:
                secs['contexts_multi'] = ctxs
        if 'pivot_multi' not in secs:
            from ..pivot import pivot_compute_points as _compute_pivot_points
            pivs: Dict[str, Any] = {}
            pivot_sec = secs.get('pivot')
            base_tf = None
            if isinstance(pivot_sec, dict) and pivot_sec.get('timeframe'):
                base_tf = str(pivot_sec.get('timeframe')).upper()
            for tfp in ['H4','D1']:
                tfp_upper = str(tfp).upper()
                if base_tf and tfp_upper == base_tf:
                    continue
                res = _get_raw_result(_compute_pivot_points, symbol=symbol, timeframe=tfp)
                if isinstance(res, dict) and not res.get('error'):
                    pivs[tfp] = {'levels': res.get('levels'), 'methods': res.get('methods'), 'period': res.get('period'), 'timeframe': tfp}
            if pivs:
                if base_tf:
                    pivs['__base_timeframe__'] = base_tf
                secs['pivot_multi'] = pivs
    except Exception:
        pass

    # Volatility (EWMA)
    from ..forecast import forecast_volatility_estimate
    # Sensible default horizons: short/base/long without requiring params
    try:
        base_h = int(horizon)
    except Exception:
        base_h = 12
    short_h = max(1, int(round(base_h / 3)))
    long_h = max(base_h + 1, int(base_h * 2))
    # Clamp very large long horizon to avoid heavy calls
    long_h = min(long_h, base_h * 3)
    vol_horizons = []
    for hh in (short_h, base_h, long_h):
        if hh not in vol_horizons:
            vol_horizons.append(hh)

    # Build method x horizon matrix (Horizon σ); keep per-bar for potential future use
    methods = ['ewma', 'parkinson', 'gk', 'yang_zhang']
    matrix_rows: List[Dict[str, Any]] = []
    for hh in vol_horizons:
        row: Dict[str, Any] = {'horizon': int(hh)}
        vals: List[float] = []
        for m in methods:
            vres = _get_raw_result(
                forecast_volatility_estimate,
                symbol=symbol,
                timeframe=tf,
                horizon=int(hh),
                method=m,
                params={'lambda_': 0.94} if m == 'ewma' else None,
            )
            if 'error' in vres:
                row[m + '_err'] = vres.get('error')
                continue
            sh = vres.get('horizon_sigma_return') or vres.get('horizon_sigma_price')
            sb = vres.get('sigma_bar_return') or vres.get('sigma_bar_price')
            use_val = None
            try:
                fv = float(sh) if sh is not None else None
                if fv is not None and fv == fv and fv >= 0.0:  # finite check (fv==fv filters NaN)
                    use_val = fv
            except Exception:
                use_val = None
            if use_val is None:
                # fallback: rolling_std proxy
                proxy_res = _get_raw_result(
                    forecast_volatility_estimate,
                    symbol=symbol,
                    timeframe=tf,
                    horizon=int(hh),
                    method='rolling_std',
                )
                psh = proxy_res.get('horizon_sigma_return') or proxy_res.get('horizon_sigma_price')
                try:
                    pf = float(psh) if psh is not None else None
                    if pf is not None and pf == pf and pf >= 0.0:
                        row[m] = pf
                        row[m + '_note'] = 'std proxy'
                        vals.append(pf)
                    else:
                        row[m + '_err'] = proxy_res.get('error') or 'no value'
                except Exception:
                    row[m + '_err'] = proxy_res.get('error') or 'invalid proxy value'
            else:
                row[m] = use_val
                vals.append(use_val)
            # store bar sigma too in case renderer wants it later
            if sb is not None:
                try:
                    fb = float(sb)
                    if fb == fb and fb >= 0.0:
                        row[m + '_bar'] = fb
                    else:
                        row[m + '_bar_err'] = 'nan bar sigma'
                except Exception:
                    row[m + '_bar_err'] = 'invalid bar sigma value'
        if len(vals) > 0:
            try:
                row['avg'] = sum(vals) / len(vals)
            except Exception:
                pass
            matrix_rows.append(row)
    if matrix_rows:
        report['sections']['volatility'] = {
            'methods': methods,
            'matrix': matrix_rows,
        }
    else:
        report['sections']['volatility'] = {'error': 'Volatility estimation failed.'}

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
    # Dynamic defaults to keep levels realistic and adaptive
    p.setdefault('grid_style', 'volatility')
    p.setdefault('vol_window', 250)
    p.setdefault('vol_min_mult', 0.6)
    p.setdefault('vol_max_mult', 2.2)
    p.setdefault('vol_sl_multiplier', 1.7)
    p.setdefault('vol_sl_steps', 9)
    # Set floors to avoid too-tight levels depending on mode
    if str(p.get('mode', 'pct')) == 'pct':
        p.setdefault('vol_floor_pct', 0.2)
    else:
        p.setdefault('vol_floor_pips', 8.0)
    # Include trading costs to discourage too-tight levels in EV
    base_params = dict(p.get('params') or {})
    base_params.setdefault('spread_bps', 1.0)
    base_params.setdefault('slippage_bps', 0.5)
    # Reasonable risk/reward filter defaults per template
    rr_min_default = p.get('rr_min', 0.8)
    rr_max_default = p.get('rr_max', 2.0)
    base_params.setdefault('rr_min', rr_min_default)
    base_params.setdefault('rr_max', rr_max_default)
    p['params'] = base_params

    mode_val = str(p.get('mode', 'pct'))
    grid_long = _get_raw_result(forecast_barrier_optimize,
        symbol=symbol,
        timeframe=tf,
        horizon=int(horizon),
        method='hmm_mc',
        mode=mode_val,
        tp_min=float(p.get('tp_min', 0.25)),
        tp_max=float(p.get('tp_max', 1.5)),
        tp_steps=int(p.get('tp_steps', 7)),
        sl_min=float(p.get('sl_min', 0.25)),
        sl_max=float(p.get('sl_max', 2.5)),
        sl_steps=int(p.get('sl_steps', 9)),
        params=p.get('params'),
        objective=str(p.get('objective','ev_uncond')),
        return_grid=False,
        top_k=int(p.get('top_k', 5)),
        output='summary',
        grid_style=str(p.get('grid_style', 'fixed')),
        preset=p.get('grid_preset', p.get('preset')),
        refine=bool(p.get('refine', False)),
        refine_radius=float(p.get('refine_radius', 0.3)),
        refine_steps=int(p.get('refine_steps', 5)),
        direction='long',
    )
    grid_short = _get_raw_result(forecast_barrier_optimize,
        symbol=symbol,
        timeframe=tf,
        horizon=int(horizon),
        method='hmm_mc',
        mode=mode_val,
        tp_min=float(p.get('tp_min', 0.25)),
        tp_max=float(p.get('tp_max', 1.5)),
        tp_steps=int(p.get('tp_steps', 7)),
        sl_min=float(p.get('sl_min', 0.25)),
        sl_max=float(p.get('sl_max', 2.5)),
        sl_steps=int(p.get('sl_steps', 9)),
        params=p.get('params'),
        objective=str(p.get('objective','ev_uncond')),
        return_grid=False,
        top_k=int(p.get('top_k', 5)),
        output='summary',
        grid_style=str(p.get('grid_style', 'fixed')),
        preset=p.get('grid_preset', p.get('preset')),
        refine=bool(p.get('refine', False)),
        refine_radius=float(p.get('refine_radius', 0.3)),
        refine_steps=int(p.get('refine_steps', 5)),
        direction='short',
    )
    sec_bar: Dict[str, Any] = {}
    if 'error' in grid_long and 'error' in grid_short:
        sec_bar = {'error': grid_long.get('error') or grid_short.get('error') or 'Barrier optimization failed'}
    else:
        if not 'error' in grid_long:
            sec_bar['long'] = summarize_barrier_grid(grid_long, top_k=int(p.get('top_k', 5)))
        else:
            sec_bar['long'] = {'error': grid_long.get('error')}
        if not 'error' in grid_short:
            sec_bar['short'] = summarize_barrier_grid(grid_short, top_k=int(p.get('top_k', 5)))
        else:
            sec_bar['short'] = {'error': grid_short.get('error')}
    report['sections']['barriers'] = sec_bar

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
