from typing import Any, Dict, Optional, List

from math import isfinite
from ..schema import DenoiseSpec
from ..report_utils import now_utc_iso, parse_table_tail, pick_best_forecast_method, summarize_barrier_grid, attach_multi_timeframes
from ..tool_calling import call_tool_sync_raw
from ...utils.utils import _safe_float


_TREND_COMPACT_LEGEND: Dict[str, str] = {
    "s": "ATR-adjusted slope score (x100) for windows [5, 20, 60] bars.",
    "r": "Linear fit quality (R^2 percent) for windows [5, 20, 60] bars.",
    "v": "ATR as basis points of price (volatility proxy).",
    "q": "Bollinger bandwidth percentile (squeeze percentile).",
    "g": "Regime code: 0=neutral, 1=uptrend, 2=downtrend, 3=breakout_up, 4=breakout_down.",
    "h": "Bars since most recent swing high (within lookback window).",
    "l": "Bars since most recent swing low (within lookback window).",
}
_TREND_REGIME_LABELS = {
    0: "neutral",
    1: "uptrend",
    2: "downtrend",
    3: "breakout_up",
    4: "breakout_down",
}


def _get_raw_result(func: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Call function and return raw dict, handling both wrapped and unwrapped cases."""
    try:
        result = call_tool_sync_raw(func, *args, cli_raw=True, **kwargs)
        
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
        table_lines = []
        
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
                    # Look ahead for table data or nested structure
                    i += 1

                    if key == 'levels':
                        # Special handling for levels - expect delimited table rows
                        table_lines = []
                        while i < len(lines):
                            next_line = lines[i]
                            stripped_line = next_line.strip()
                            if not stripped_line:
                                i += 1
                                continue
                            # Look for delimited table rows (may be indented)
                            if ',' in stripped_line and (stripped_line.startswith('level,') or
                                all(c.isdigit() or c in '.,- ' or c.isalpha() for c in stripped_line[:20])):
                                table_lines.append(stripped_line)
                                i += 1
                            elif stripped_line and not next_line.startswith(' ') and ':' in stripped_line and ',' not in stripped_line:
                                # New section starting
                                break
                            else:
                                i += 1

                        if table_lines:
                            result[key] = _parse_table_data(table_lines)
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

            # Standalone delimited table data
            elif ',' in line and not line.startswith(' '):
                table_lines = [line]
                i += 1
                # Collect following delimited rows
                while i < len(lines) and ',' in lines[i] and not (lines[i].count(':') > lines[i].count(',')):
                    table_lines.append(lines[i])
                    i += 1

                if table_lines:
                    parsed_table = _parse_table_data(table_lines)
                    if current_section:
                        result[current_section] = parsed_table
                    else:
                        result['data'] = parsed_table
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


def _parse_table_data(table_lines: List[str]) -> Any:
    """Parse comma-delimited table lines into structured data."""
    if not table_lines:
        return None

    try:
        # First line should be headers
        headers = [h.strip() for h in table_lines[0].split(',')]

        if len(table_lines) == 1:
            return {'headers': headers}

        # Parse data rows
        rows = []
        for line in table_lines[1:]:
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
        return table_lines

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


def _explain_compact_trend(compact: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(compact, dict):
        return {}
    out: Dict[str, Any] = {}
    s = compact.get("s")
    r = compact.get("r")
    if isinstance(s, list):
        out["slope_atr_score_5_20_60"] = [int(v) for v in s[:3] if isinstance(v, (int, float))]
    if isinstance(r, list):
        out["fit_r2_pct_5_20_60"] = [int(v) for v in r[:3] if isinstance(v, (int, float))]
    for key in ("v", "q", "h", "l"):
        val = compact.get(key)
        if isinstance(val, (int, float)):
            out[key] = int(val)
    g_val = compact.get("g")
    if isinstance(g_val, (int, float)):
        g_i = int(g_val)
        out["g"] = g_i
        out["regime_label"] = _TREND_REGIME_LABELS.get(g_i, "neutral")
    return out


def _extract_forecast_values(payload: Dict[str, Any]) -> Optional[List[float]]:
    if not isinstance(payload, dict):
        return None
    for key in ('forecast_price', 'forecast_return', 'forecast_series', 'forecast'):
        vals = payload.get(key)
        if isinstance(vals, list) and vals:
            parsed: List[float] = []
            for value in vals:
                try:
                    parsed.append(float(value))
                except Exception:
                    parsed = []
                    break
            if parsed:
                return parsed
    return None


def _is_degenerate_forecast_payload(payload: Dict[str, Any]) -> bool:
    vals = _extract_forecast_values(payload)
    if not isinstance(vals, list) or len(vals) < 3:
        return False
    first = vals[0]
    span = max(vals) - min(vals)
    tol = max(1e-9, abs(first) * 1e-6)
    return span <= tol


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
        # Extract a tail window of candle rows
        tail_n = int(p.get('context_tail', 40))
        tail_rows = parse_table_tail(ctx, tail=tail_n)
        if not tail_rows:
            # Fallbacks when calling through minimal formatter
            if isinstance(ctx, dict) and isinstance(ctx.get('data'), list):     
                tail_rows = ctx.get('data')[-tail_n:]  # type: ignore[index]    
            elif isinstance(ctx, list):
                tail_rows = ctx
            else:
                tail_rows = []

        if not tail_rows:
            report['sections']['context'] = {'error': 'No candle data available for context section.'}
        else:
            last = tail_rows[-1] if tail_rows else {}
            compact = _compute_compact_trend(tail_rows)
            ctx_obj: Dict[str, Any] = {
                'symbol': symbol,
                'timeframe': tf,
                'last_snapshot': last,
                'notes': 'Indicators included: EMA(20), EMA(50), RSI(14), MACD(12,26,9).',
            }
            if compact:
                ctx_obj['trend_compact'] = compact
                ctx_obj['trend_compact_legend'] = dict(_TREND_COMPACT_LEGEND)
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
            'calculation_basis': piv.get('calculation_basis'),
            'timezone': piv.get('timezone'),
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
            from ..report_utils import context_for_tf, _extract_base_timeframe
            base_tf = _extract_base_timeframe(report)
            tf_list = ['M15','H1','H4','D1']
            ctxs: Dict[str, Any] = {}
            for tf_i in tf_list:
                if base_tf and tf_i.upper() == base_tf:
                    continue
                snap = context_for_tf(symbol, tf_i, denoise, limit=200, tail=30)
                if snap and any(v is not None for v in snap.values()):
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
                    pivs[tfp] = {
                        'levels': res.get('levels'),
                        'methods': res.get('methods'),
                        'period': res.get('period'),
                        'timeframe': tfp,
                        'calculation_basis': res.get('calculation_basis'),
                        'timezone': res.get('timezone'),
                    }
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
    min_dir_acc_raw = p.get('backtest_min_directional_accuracy', p.get('backtest_min_accuracy'))
    try:
        min_dir_acc = float(min_dir_acc_raw) if min_dir_acc_raw is not None else None
    except Exception:
        min_dir_acc = None
    if min_dir_acc is not None:
        if not isfinite(min_dir_acc):
            min_dir_acc = None
        else:
            min_dir_acc = max(0.0, min(1.0, float(min_dir_acc)))
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
        criteria_notes = 'Choose lowest RMSE; when methods are within tolerance of best RMSE, prefer higher directional accuracy.'
        if min_dir_acc is not None:
            criteria_notes += f' Require directional accuracy >= {min_dir_acc:.2f}.'
        sec_bt['selection_criteria'] = {
            'primary_metric': 'avg_rmse',
            'rmse_tolerance': float(rmse_tol),
            'rmse_tolerance_pct': float(rmse_tol * 100.0),
            'tie_breaker': 'avg_directional_accuracy',
            'secondary_tie_breaker': 'successful_tests',
            'notes': criteria_notes,
        }
        if min_dir_acc is not None:
            sec_bt['selection_criteria']['min_directional_accuracy'] = float(min_dir_acc)
            sec_bt['selection_criteria']['min_directional_accuracy_pct'] = float(min_dir_acc * 100.0)
        best = pick_best_forecast_method(
            bt,
            rmse_tolerance=rmse_tol,
            min_directional_accuracy=min_dir_acc,
        )
        if best is None and min_dir_acc is not None:
            sec_bt['selection_warning'] = (
                "No method met the minimum directional accuracy threshold."
            )
            sec_bt['selection_filtered_by_min_directional_accuracy'] = True
    report['sections']['backtest'] = sec_bt

    if best is not None:
        best_name, best_stats = best
        from ..forecast import forecast_generate
        bt_results = bt.get('results') if isinstance(bt, dict) else {}
        stats_by_method: Dict[str, Dict[str, Any]] = {}
        if isinstance(bt_results, dict):
            for method_name, method_stats in bt_results.items():
                if isinstance(method_stats, dict):
                    stats_by_method[str(method_name)] = method_stats

        ranked_methods: List[str] = []
        for row in ranking:
            if not isinstance(row, dict):
                continue
            method_name = str(row.get('method') or '').strip()
            if method_name and method_name not in ranked_methods:
                ranked_methods.append(method_name)

        candidate_methods: List[str] = [best_name]
        for method_name in ranked_methods:
            if method_name not in candidate_methods:
                candidate_methods.append(method_name)
        for method_name in stats_by_method.keys():
            if method_name not in candidate_methods:
                candidate_methods.append(method_name)

        selected_method = best_name
        selected_stats: Dict[str, Any] = dict(best_stats or {})
        selected_forecast: Optional[Dict[str, Any]] = None
        fallback_notes: List[str] = []
        first_error: Optional[str] = None
        first_degenerate: Optional[Dict[str, Any]] = None
        first_degenerate_method: Optional[str] = None

        for method_name in candidate_methods:
            fc = _get_raw_result(
                forecast_generate,
                symbol=symbol,
                timeframe=tf,
                method=method_name,
                horizon=int(horizon),
            )
            if 'error' in fc:
                if first_error is None:
                    first_error = str(fc.get('error') or '')
                fallback_notes.append(f"{method_name}: forecast error ({fc.get('error')})")
                continue
            if _is_degenerate_forecast_payload(fc):
                if first_degenerate is None:
                    first_degenerate = fc
                    first_degenerate_method = method_name
                fallback_notes.append(f"{method_name}: degenerate forecast")
                continue
            selected_method = method_name
            selected_stats = dict(stats_by_method.get(method_name) or best_stats or {})
            selected_forecast = fc
            break

        if selected_forecast is None and first_degenerate is not None:
            selected_method = first_degenerate_method or best_name
            selected_stats = dict(stats_by_method.get(selected_method) or best_stats or {})
            selected_forecast = first_degenerate

        if selected_forecast is None:
            report['sections']['forecast'] = {
                'error': first_error or 'No usable forecast generated.',
                'method': best_name,
            }
        else:
            report['sections']['forecast'] = {
                'method': selected_method,
                'forecast_price': selected_forecast.get('forecast_price'),
                'lower_price': selected_forecast.get('lower_price'),
                'upper_price': selected_forecast.get('upper_price'),
                'trend': selected_forecast.get('trend'),
                'ci_alpha': selected_forecast.get('ci_alpha'),
                'last_observation_epoch': selected_forecast.get('last_observation_epoch'),
                'forecast_start_epoch': selected_forecast.get('forecast_start_epoch'),
                'forecast_anchor': selected_forecast.get('forecast_anchor'),
                'forecast_start_gap_bars': selected_forecast.get('forecast_start_gap_bars'),
                'forecast_step_seconds': selected_forecast.get('forecast_step_seconds'),
            }
            if selected_method != best_name:
                report['sections']['forecast']['fallback_from'] = best_name
                report['sections']['forecast']['fallback_reason'] = 'initial best method produced a degenerate forecast'
                report['fallback_applied'] = True
                report['original_method'] = best_name
                report['fallback_method'] = selected_method
            if fallback_notes:
                report['sections']['forecast']['selection_warnings'] = fallback_notes

        best_method_payload: Dict[str, Any] = {
            'method': selected_method if selected_forecast is not None else best_name,
            'stats': {
                'avg_rmse': selected_stats.get('avg_rmse'),
                'avg_mae': selected_stats.get('avg_mae'),
                'avg_directional_accuracy': selected_stats.get('avg_directional_accuracy'),
                'successful_tests': selected_stats.get('successful_tests'),
            },
        }
        selection_basis: Dict[str, Any] = {
            'primary_metric': 'avg_rmse',
            'rmse_tolerance': float(rmse_tol),
            'rmse_tolerance_pct': float(rmse_tol * 100.0),
            'tie_breaker': 'avg_directional_accuracy',
            'secondary_tie_breaker': 'successful_tests',
            'initial_method': best_name,
            'selected_method': selected_method if selected_forecast is not None else best_name,
        }
        if min_dir_acc is not None:
            selection_basis['min_directional_accuracy'] = float(min_dir_acc)
            selection_basis['min_directional_accuracy_pct'] = float(min_dir_acc * 100.0)
        if ranking:
            try:
                best_rmse = float(ranking[0].get('avg_rmse'))
                if isfinite(best_rmse):
                    selection_basis['best_rmse'] = best_rmse
            except Exception:
                pass
        try:
            sel_rmse = float(selected_stats.get('avg_rmse'))
            if isfinite(sel_rmse):
                selection_basis['selected_rmse'] = sel_rmse
                if selection_basis.get('best_rmse') is not None:
                    tol_limit = float(selection_basis['best_rmse']) * (1.0 + float(rmse_tol))
                    selection_basis['rmse_tolerance_limit'] = tol_limit
                    selection_basis['within_rmse_tolerance'] = bool(sel_rmse <= tol_limit)
        except Exception:
            pass
        if selected_forecast is not None and selected_method != best_name:
            selection_basis['fallback_applied'] = True
            selection_basis['fallback_reason'] = 'initial best method produced a degenerate forecast'
        best_method_payload['selection_basis'] = selection_basis
        if selected_forecast is not None and selected_method != best_name:
            best_method_payload['initial_method'] = best_name
            best_method_payload['selection_warning'] = 'Initial best method forecast was degenerate; fallback applied.'
        if fallback_notes:
            best_method_payload['selection_warnings'] = fallback_notes
        report['sections']['backtest']['best_method'] = best_method_payload

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
    for barrier_key in (
        'vol_window',
        'vol_min_mult',
        'vol_max_mult',
        'vol_steps',
        'vol_sl_extra',
        'vol_sl_multiplier',
        'vol_sl_steps',
        'vol_floor_pct',
        'vol_floor_pips',
    ):
        if barrier_key in p:
            base_params.setdefault(barrier_key, p.get(barrier_key))
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
        objective=str(p.get('objective','ev')),
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
        objective=str(p.get('objective','ev')),
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
        if 'error' not in grid_long:
            sec_bar['long'] = summarize_barrier_grid(grid_long, top_k=int(p.get('top_k', 5)))
        else:
            sec_bar['long'] = {'error': grid_long.get('error')}
        if 'error' not in grid_short:
            sec_bar['short'] = summarize_barrier_grid(grid_short, top_k=int(p.get('top_k', 5)))
        else:
            sec_bar['short'] = {'error': grid_short.get('error')}
        conflict_directions: List[str] = []
        caution_parts: List[str] = []
        for direction in ('long', 'short'):
            sub = sec_bar.get(direction)
            if not isinstance(sub, dict):
                continue
            if bool(sub.get('ev_edge_conflict')):
                conflict_directions.append(direction)
            caution_text = sub.get('caution')
            if isinstance(caution_text, str) and caution_text.strip():
                caution_parts.append(f"{direction}: {caution_text.strip()}")
        if conflict_directions:
            sec_bar['ev_edge_conflict'] = True
            sec_bar['ev_edge_conflict_directions'] = conflict_directions
            sec_bar['ev_edge_conflict_reason'] = "ev and edge have opposite signs"
            if caution_parts:
                sec_bar['caution'] = "; ".join(caution_parts)
            else:
                sec_bar['caution'] = (
                    "EV and edge signs conflict in barrier recommendations; inspect win probability "
                    "and break-even thresholds before trading."
                )
    sec_bar['mode'] = mode_val
    sec_bar['note'] = (
        "Report barriers are produced by an independent optimization run; "
        "standalone forecast_barrier_optimize may yield different candidates. "
        "edge measures win-rate margin versus breakeven, while EV also weights reward/risk."
    )
    report['sections']['barriers'] = sec_bar

    # Patterns
    from ..patterns import patterns_detect
    pats = _get_raw_result(
        patterns_detect,
        symbol=symbol,
        timeframe=tf,
        mode='candlestick',
        detail='compact',
        limit=int(p.get('patterns_limit', 120)),
    )
    if 'error' in pats:
        report['sections']['patterns'] = {'error': pats['error']}
    else:
        recent_patterns = pats.get('recent_patterns') if isinstance(pats, dict) else None
        if isinstance(recent_patterns, list):
            detections = [row for row in recent_patterns[:5] if isinstance(row, dict)]
        else:
            rows = parse_table_tail(pats, tail=20)
            detections = rows[-5:] if rows else []
        report['sections']['patterns'] = {'recent': detections}

    return report
