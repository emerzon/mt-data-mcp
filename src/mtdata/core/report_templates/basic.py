from math import isfinite
from typing import Any, Dict, List, Optional

from ...shared.schema import DenoiseSpec
from ...utils.coercion import safe_float as _safe_float
from ..report.utils import (
    adapt_forecast_payload_for_report,
    attach_candle_freshness_diagnostics,
    attach_multi_timeframes,
    current_only_section_omission as _current_only_section_omission,
    extract_report_forecast_values,
    now_utc_iso,
    parse_table_tail,
    pick_best_forecast_method,
    report_section_enabled,
    is_bounded_report_window as _is_bounded_report_window,
    resolve_report_context_indicators,
    summarize_barrier_grid,
)
from ..tool_calling import call_tool_sync_structured

_TREND_COMPACT_LEGEND: Dict[str, str] = {
    "slope_atr_scores": "ATR-adjusted slope score (x100) for windows [5, 20, 60] bars.",
    "fit_r2_pcts": "Linear fit quality (R^2 percent) for windows [5, 20, 60] bars.",
    "volatility_bps": "ATR as basis points of price (volatility proxy).",
    "squeeze_percentile": "Bollinger bandwidth percentile (squeeze percentile).",
    "regime_code": "Regime code: 0=neutral, 1=uptrend, 2=downtrend, 3=breakout_up, 4=breakout_down.",
    "bars_since_swing_high": "Bars since most recent swing high (within lookback window).",
    "bars_since_swing_low": "Bars since most recent swing low (within lookback window).",
    "bars_analyzed": "Consecutive source-timeframe bars used by the calculations.",
    "input_resolution": "Input spacing used by bar-window calculations.",
    "data_quality": "Missing-input summary when close/high/low values were imputed for trend calculations.",
}

def _get_raw_result(
    func: Any,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Call a tool and require a structured payload."""
    try:
        result = call_tool_sync_structured(func, *args, raw_tool_output=True, **kwargs)
        
        # If it returns a dict, use it directly
        if isinstance(result, dict):
            return result
            
        if isinstance(result, str):
            return {
                'error': 'Expected structured tool result but received formatted text.',
                'raw_output': result[:200],
            }
        
        return {'error': f'Unexpected result type: {type(result)}'}
        
    except Exception as e:
        return {'error': f'Function call failed: {str(e)}'}


def _first_volatility_value(payload: Dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


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


def _wilder_rma(values: List[float], length: int) -> List[float]:
    """Return Wilder's moving average with an SMA seed."""
    if length <= 1 or not values:
        return list(values)
    out: List[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        numeric = float(value)
        if idx < length:
            running += numeric
            average = running / float(idx + 1)
        else:
            average = ((out[-1] * (length - 1)) + numeric) / float(length)
        out.append(average)
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


def _bars_since_latest_pivot(values: List[float], *, high: bool) -> int:
    """Return bars since the latest one-bar confirmed local extremum."""
    if len(values) < 3:
        return 0
    for index in range(len(values) - 2, 0, -1):
        value = values[index]
        if high and value >= values[index - 1] and value > values[index + 1]:
            return (len(values) - 1) - index
        if not high and value <= values[index - 1] and value < values[index + 1]:
            return (len(values) - 1) - index
    return 0


def _compute_compact_trend(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows or len(rows) < 5:
        return None
    closes: List[Optional[float]] = [_safe_float(r.get('close')) for r in rows]
    highs: List[Optional[float]] = [_safe_float(r.get('high')) for r in rows]
    lows: List[Optional[float]] = [_safe_float(r.get('low')) for r in rows]
    seed_close = next((float(c) for c in closes if c is not None and float(c) > 0.0), None)
    if seed_close is None:
        return None
    imputed_fields = {"close": 0, "high": 0, "low": 0}
    imputed_bars: set[int] = set()
    # Replace missing/invalid close values with the earliest known close, then forward-fill.
    clean_close: List[float] = []
    lastc = seed_close
    for idx, c in enumerate(closes):
        if c is None or float(c) <= 0.0:
            c = lastc if clean_close else seed_close
            imputed_fields["close"] += 1
            imputed_bars.add(idx)
        clean_close.append(float(c))
        lastc = float(c)
    clean_high: List[float] = []
    clean_low: List[float] = []
    for idx, (h, l, c) in enumerate(zip(highs, lows, clean_close)):
        high_val = h
        low_val = l
        if high_val is None or float(high_val) <= 0.0:
            high_val = c
            imputed_fields["high"] += 1
            imputed_bars.add(idx)
        if low_val is None or float(low_val) <= 0.0:
            low_val = c
            imputed_fields["low"] += 1
            imputed_bars.add(idx)
        clean_high.append(float(high_val))
        clean_low.append(float(low_val))

    # ATR(14) in price units
    tr = _compute_tr(clean_high, clean_low, clean_close)
    atr_series = _wilder_rma(tr, 14)
    atr = atr_series[-1] if atr_series else 0.0
    last_price = clean_close[-1] if clean_close else 0.0

    # Slope windows
    wins = [5, 20, 60]
    s_vals: List[Optional[int]] = []
    r_vals: List[Optional[int]] = []
    for w in wins:
        if len(clean_close) < w:
            s_vals.append(None)
            r_vals.append(None)
            continue
        seg = clean_close[-w:]
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
    s5 = s_vals[0] if s_vals[0] is not None else 0
    s20 = s_vals[1] if s_vals[1] is not None else 0
    r20 = r_vals[1] if r_vals[1] is not None else 0
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
        h_idx = _bars_since_latest_pivot(segment_h, high=True)
        l_idx = _bars_since_latest_pivot(segment_l, high=False)

    # ATR% of price as basis points (bps)
    v = int(round(((atr / last_price) * 10000.0) if last_price > 0 and atr > 0 else 0.0))

    out = {
        'slope_atr_scores': s_vals,   # slopes ATRu*100
        'fit_r2_pcts': r_vals,   # R2%
        'volatility_bps': v,        # ATR% of price
        'squeeze_percentile': int(q),   # squeeze percentile
        'regime_code': int(g),   # regime code
        'bars_since_swing_high': int(h_idx),
        'bars_since_swing_low': int(l_idx),
        'bars_analyzed': int(len(clean_close)),
        'input_resolution': 'consecutive_timeframe_bars',
    }
    if imputed_bars:
        out['data_quality'] = {
            'status': 'imputed',
            'imputed_bars': int(len(imputed_bars)),
            'imputed_pct': round((100.0 * len(imputed_bars)) / float(len(rows)), 1),
            'imputed_fields': {
                key: int(value) for key, value in imputed_fields.items() if int(value) > 0
            },
            'warning': (
                'Trend metrics include imputed close/high/low values; treat regime and slope scores as '
                'lower-confidence when gaps are present.'
            ),
        }
    return out


def _extract_forecast_values(payload: Dict[str, Any]) -> Optional[List[float]]:
    values = extract_report_forecast_values(payload)
    return values or None


def _is_degenerate_forecast_payload(payload: Dict[str, Any]) -> bool:
    vals = _extract_forecast_values(payload)
    if not vals:
        return True
    if len(vals) < 3:
        return False
    first = vals[0]
    span = max(vals) - min(vals)
    tol = max(1e-9, abs(first) * 1e-6)
    return span <= tol


def template_basic(  # noqa: C901
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
    *,
    include_default_timeframes: bool = True,
) -> Dict[str, Any]:
    p = dict(params or {})
    tf = str(p.get('timeframe', 'H1'))
    start = p.get('start')
    end = p.get('end')
    bounded_window = _is_bounded_report_window(start, end)
    
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

    # Request-scoped cache avoids re-fetching the same symbol/timeframe.
    _fetch_cache: Dict = {}

    # Context
    indicators = resolve_report_context_indicators(
        p,
        default="ema(20),ema(50),rsi(14),macd(12,26,9)",
    )
    from ..data import data_fetch_candles
    
    ctx = (
        _get_raw_result(data_fetch_candles,
            symbol=symbol,
            timeframe=tf,
            limit=int(p.get('context_limit', 300)),
            start=start,
            end=end,
            indicators=indicators,  # type: ignore[arg-type]
            denoise=denoise,
        )
        if report_section_enabled(p, 'context')
        else {'error': 'context section not requested'}
    )
    
    if 'error' in ctx:
        report['sections']['context'] = attach_candle_freshness_diagnostics({'error': ctx['error']}, ctx)
    else:
        # Metrics require consecutive source bars. Only the snapshot is
        # projected to the requested display tail.
        context_limit = int(p.get('context_limit', 300))
        context_rows = parse_table_tail(ctx, tail=context_limit)
        tail_n = int(p.get('context_tail', 40))
        tail_rows = context_rows[-tail_n:]
        if not tail_rows:
            # Fallbacks when calling through minimal formatter
            if isinstance(ctx, dict) and isinstance(ctx.get('data'), list):     
                context_rows = list(ctx.get('data'))  # type: ignore[arg-type]
                tail_rows = context_rows[-tail_n:]
            elif isinstance(ctx, list):
                context_rows = ctx
                tail_rows = context_rows[-tail_n:]
            else:
                tail_rows = []

        if not tail_rows:
            report['sections']['context'] = attach_candle_freshness_diagnostics(
                {'error': 'No candle data available for context section.'},
                ctx,
            )
        else:
            last = tail_rows[-1] if tail_rows else {}
            compact = _compute_compact_trend(context_rows)
            ctx_obj: Dict[str, Any] = {
                'symbol': symbol,
                'timeframe': tf,
                'last_snapshot': last,
                'notes': f'Indicators included: {indicators}.',
            }
            timezone_label = ctx.get('timezone') if isinstance(ctx, dict) else None
            if timezone_label not in (None, '', [], {}):
                ctx_obj['timezone'] = timezone_label
            if compact:
                ctx_obj['trend_compact'] = compact
                ctx_obj['trend_compact_legend'] = dict(_TREND_COMPACT_LEGEND)
            report['sections']['context'] = attach_candle_freshness_diagnostics(ctx_obj, ctx)

    pivot_enabled = report_section_enabled(p, 'pivot')
    contexts_multi_enabled = include_default_timeframes and report_section_enabled(
        p, 'contexts_multi'
    )
    pivot_multi_enabled = include_default_timeframes and report_section_enabled(
        p, 'pivot_multi'
    )

    # Pivots use the current completed source bar and cannot honor a report
    # window. Keep bounded reports temporally coherent by omitting them.
    if pivot_enabled and bounded_window:
        report['sections']['pivot'] = _current_only_section_omission(
            'pivot', start=start, end=end
        )
    elif pivot_enabled:
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

    if pivot_multi_enabled and bounded_window:
        report['sections']['pivot_multi'] = _current_only_section_omission(
            'pivot_multi', start=start, end=end
        )

    if contexts_multi_enabled or (pivot_multi_enabled and not bounded_window):
        attach_multi_timeframes(
            report,
            symbol,
            denoise,
            extra_timeframes=(
                ['M15','H1','H4','D1'] if contexts_multi_enabled else []
            ),
            pivot_timeframes=(
                ['H4','D1']
                if pivot_multi_enabled and not bounded_window
                else []
            ),
            context_indicators=indicators,
            start=start,
            end=end,
            _fetch_cache=_fetch_cache,
        )

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
    if report_section_enabled(p, 'volatility'):
        for hh in (short_h, base_h, long_h):
            if hh not in vol_horizons:
                vol_horizons.append(hh)

    # Build method x horizon matrix (Horizon σ); keep per-bar for potential future use
    methods = ['ewma', 'parkinson', 'gk', 'yang_zhang']
    matrix_rows: List[Dict[str, Any]] = []
    vol_errors: List[Dict[str, Any]] = []
    for hh in vol_horizons:
        row: Dict[str, Any] = {'horizon': int(hh)}
        contributors: List[Dict[str, Any]] = []
        for m in methods:
            vres = _get_raw_result(
                forecast_volatility_estimate,
                symbol=symbol,
                timeframe=tf,
                horizon=int(hh),
                method=m,
                params={'lambda_': 0.94} if m == 'ewma' else None,
                start=start,
                end=end,
                detail='full',
            )
            if 'error' in vres:
                error_text = str(vres.get('error') or 'volatility method failed')
                row[m + '_err'] = error_text
                vol_errors.append({
                    'horizon': int(hh),
                    'method': m,
                    'error': error_text,
                })
                continue
            sh = _first_volatility_value(
                vres,
                ('volatility_horizon', 'horizon_sigma_price'),
            )
            sb = _first_volatility_value(
                vres,
                ('volatility_per_bar', 'sigma_bar_price'),
            )
            use_val = None
            try:
                fv = float(sh) if sh is not None else None
                if fv is not None and fv == fv and fv >= 0.0:  # finite check (fv==fv filters NaN)
                    use_val = fv
            except Exception:
                use_val = None
            if use_val is None:
                error_text = 'requested estimator returned no usable horizon value'
                row[m + '_err'] = error_text
                vol_errors.append({
                    'horizon': int(hh),
                    'method': m,
                    'error': error_text,
                })
            else:
                row[m] = use_val
                contributors.append({'method': m, 'value': use_val, 'weight': 1.0})
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
        if not contributors:
            proxy_res = _get_raw_result(
                forecast_volatility_estimate,
                symbol=symbol,
                timeframe=tf,
                horizon=int(hh),
                method='rolling_std',
                start=start,
                end=end,
                detail='full',
            )
            proxy_value = _first_volatility_value(
                proxy_res,
                ('volatility_horizon', 'horizon_sigma_price'),
            )
            try:
                proxy_float = float(proxy_value) if proxy_value is not None else None
            except Exception:
                proxy_float = None
            if proxy_float is not None and isfinite(proxy_float) and proxy_float >= 0.0:
                row['rolling_std_proxy'] = proxy_float
                contributors.append({
                    'method': 'rolling_std_proxy',
                    'value': proxy_float,
                    'weight': 1.0,
                    'provenance': 'fallback_when_all_requested_estimators_unavailable',
                })
        if contributors:
            values = [float(item['value']) for item in contributors]
            row['avg'] = sum(values) / len(values)
            row['avg_method'] = (
                contributors[0]['method']
                if len(contributors) == 1
                else 'ensemble_mean'
            )
            equal_weight = 1.0 / len(contributors)
            for contributor in contributors:
                contributor['weight'] = equal_weight
            row['contributors'] = contributors
            matrix_rows.append(row)
    if matrix_rows:
        report['sections']['volatility'] = {
            'methods': methods,
            'aggregate_method': 'ensemble_mean',
            'matrix': matrix_rows,
        }
    else:
        report['sections']['volatility'] = {
            'error': 'Volatility estimation failed.',
            'errors': vol_errors[:8],
            'hint': 'Run forecast_volatility_estimate directly for full diagnostics.',
        }

    # Backtest select best
    steps = int(p.get('backtest_steps', 25))
    requested_spacing = int(p.get('backtest_spacing', 10))
    spacing = requested_spacing
    if steps > 1 and spacing <= int(horizon):
        spacing = int(horizon) + 1
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
    bt = (
        _get_raw_result(
            forecast_backtest_run,
            symbol=symbol,
            timeframe=tf,
            horizon=int(horizon),
            steps=steps,
            spacing=spacing,
            start=start,
            end=end,
            methods=methods,
            detail='full',
        )
        if report_section_enabled(p, 'backtest')
        else {'error': 'backtest section not requested'}
    )
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

        quality_candidates: List[tuple[str, float]] = []
        for method_name, method_stats in stats_by_method.items():
            if method_stats.get('success') is not True:
                continue
            try:
                method_rmse = float(method_stats.get('avg_rmse'))
            except Exception:
                continue
            if not isfinite(method_rmse):
                continue
            if min_dir_acc is not None:
                try:
                    method_da = float(method_stats.get('avg_directional_accuracy'))
                except Exception:
                    continue
                if not isfinite(method_da) or method_da < float(min_dir_acc):
                    continue
            quality_candidates.append((method_name, method_rmse))
        quality_best_rmse = min(
            (item[1] for item in quality_candidates),
            default=float('inf'),
        )
        quality_limit = quality_best_rmse * (1.0 + max(0.0, float(rmse_tol)))
        eligible_methods = {
            method_name
            for method_name, method_rmse in quality_candidates
            if method_rmse <= quality_limit
        }

        ranked_methods: List[str] = []
        for row in ranking:
            if not isinstance(row, dict):
                continue
            method_name = str(row.get('method') or '').strip()
            if method_name and method_name not in ranked_methods:
                ranked_methods.append(method_name)

        candidate_methods: List[str] = [best_name]
        for method_name in ranked_methods:
            if method_name in eligible_methods and method_name not in candidate_methods:
                candidate_methods.append(method_name)

        selected_method = best_name
        selected_stats: Dict[str, Any] = dict(best_stats or {})
        selected_forecast: Optional[Dict[str, Any]] = None
        fallback_notes: List[str] = []
        first_error: Optional[str] = None
        failure_causes: Dict[str, Dict[str, str]] = {}

        if not report_section_enabled(p, 'forecast'):
            candidate_methods = []

        for method_name in candidate_methods:
            fc = _get_raw_result(
                forecast_generate,
                symbol=symbol,
                timeframe=tf,
                method=method_name,
                horizon=int(horizon),
                start=start,
                end=end,
            )
            if 'error' in fc:
                if first_error is None:
                    first_error = str(fc.get('error') or '')
                failure_causes[method_name] = {
                    'code': 'forecast_error',
                    'message': str(fc.get('error') or 'forecast generation failed'),
                }
                fallback_notes.append(f"{method_name}: forecast error ({fc.get('error')})")
                continue
            if _is_degenerate_forecast_payload(fc):
                failure_causes[method_name] = {
                    'code': 'degenerate_forecast',
                    'message': 'forecast values were degenerate',
                }
                fallback_notes.append(f"{method_name}: degenerate forecast")
                continue
            selected_method = method_name
            selected_stats = dict(stats_by_method.get(method_name) or best_stats or {})
            selected_forecast = fc
            break

        if selected_forecast is None and report_section_enabled(p, 'forecast'):
            report['sections']['forecast'] = {
                'error': first_error or 'No quality-eligible method produced a usable forecast.',
                'method': best_name,
                'eligible_methods': sorted(eligible_methods),
            }
        elif selected_forecast is not None and report_section_enabled(p, 'forecast'):
            report['sections']['forecast'] = {
                'method': selected_method,
                **adapt_forecast_payload_for_report(selected_forecast),
            }
            if selected_method != best_name:
                initial_cause = failure_causes.get(best_name) or {
                    'code': 'forecast_fallback',
                    'message': 'initial method did not produce a usable forecast',
                }
                report['sections']['forecast']['fallback_from'] = best_name
                report['sections']['forecast']['fallback_reason_code'] = initial_cause['code']
                report['sections']['forecast']['fallback_reason'] = initial_cause['message']
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
            initial_cause = failure_causes.get(best_name) or {
                'code': 'forecast_fallback',
                'message': 'initial method did not produce a usable forecast',
            }
            selection_basis['fallback_applied'] = True
            selection_basis['fallback_reason_code'] = initial_cause['code']
            selection_basis['fallback_reason'] = initial_cause['message']
        best_method_payload['selection_basis'] = selection_basis
        if selected_forecast is not None and selected_method != best_name:
            best_method_payload['initial_method'] = best_name
            best_method_payload['selection_warning'] = (
                f"Initial best method failed ({initial_cause['code']}): "
                f"{initial_cause['message']}; fallback applied."
            )
        if fallback_notes:
            best_method_payload['selection_warnings'] = fallback_notes
        report['sections']['backtest']['best_method'] = best_method_payload

    if report_section_enabled(p, 'forecast') and 'forecast' not in report['sections']:
        report['sections']['forecast'] = {
            'error': 'No usable forecast method was selected by the backtest.',
        }

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
        p.setdefault('vol_floor_ticks', 8.0)
    # Include trading costs to discourage too-tight levels in EV
    base_params = dict(p.get('params') or {})
    base_params.setdefault('spread_bps', 1.0)
    base_params.setdefault('slippage_bps', 0.5)
    if 'fast_defaults' in p:
        base_params.setdefault('fast_defaults', bool(p.get('fast_defaults')))
    base_params.setdefault('tp_min', float(p.get('tp_min', 0.25)))
    base_params.setdefault('tp_max', float(p.get('tp_max', 1.5)))
    base_params.setdefault('tp_steps', int(p.get('tp_steps', 7)))
    base_params.setdefault('sl_min', float(p.get('sl_min', 0.25)))
    base_params.setdefault('sl_max', float(p.get('sl_max', 2.5)))
    base_params.setdefault('sl_steps', int(p.get('sl_steps', 9)))
    base_params.setdefault('refine', bool(p.get('refine', False)))
    base_params.setdefault('refine_radius', float(p.get('refine_radius', 0.3)))
    base_params.setdefault('refine_steps', int(p.get('refine_steps', 5)))
    # Reasonable risk/reward filter defaults per template
    rr_min_default = p.get('rr_min', 0.8)
    rr_max_default = p.get('rr_max', 2.0)
    base_params.setdefault('rr_min', rr_min_default)
    base_params.setdefault('rr_max', rr_max_default)
    for barrier_key in (
        'tp_min',
        'tp_max',
        'tp_steps',
        'sl_min',
        'sl_max',
        'sl_steps',
        'vol_window',
        'vol_min_mult',
        'vol_max_mult',
        'vol_steps',
        'vol_sl_multiplier',
        'vol_sl_steps',
        'vol_floor_pct',
        'vol_floor_ticks',
        'refine',
        'refine_radius',
        'refine_steps',
    ):
        if barrier_key in p:
            base_params.setdefault(barrier_key, p.get(barrier_key))
    p['params'] = base_params

    mode_val = str(p.get('mode', 'pct'))
    barrier_method = str(p.get('barrier_method', 'hmm_mc'))
    if bounded_window or not report_section_enabled(p, 'barriers'):
        grid_long = grid_short = None
    else:
        grid_long = _get_raw_result(forecast_barrier_optimize,
            symbol=symbol,
            timeframe=tf,
            horizon=int(horizon),
            method=barrier_method,
            mode=mode_val,
            params=p.get('params'),
            objective=str(p.get('objective','ev')),
            top_k=int(p.get('top_k', 5)),
            grid_style=str(p.get('grid_style', 'fixed')),
            preset=p.get('grid_preset', p.get('preset')),
            search_profile=str(p.get('search_profile', 'medium')),
            direction='long',
        )
        grid_short = _get_raw_result(forecast_barrier_optimize,
            symbol=symbol,
            timeframe=tf,
            horizon=int(horizon),
            method=barrier_method,
            mode=mode_val,
            params=p.get('params'),
            objective=str(p.get('objective','ev')),
            top_k=int(p.get('top_k', 5)),
            grid_style=str(p.get('grid_style', 'fixed')),
            preset=p.get('grid_preset', p.get('preset')),
            search_profile=str(p.get('search_profile', 'medium')),
            direction='short',
        )
    sec_bar: Dict[str, Any] = {}
    if bounded_window or not report_section_enabled(p, 'barriers'):
        sec_bar = _current_only_section_omission('barriers', start=start, end=end)
    elif isinstance(grid_long, dict) and isinstance(grid_short, dict) and 'error' in grid_long and 'error' in grid_short:
        sec_bar = {'error': grid_long.get('error') or grid_short.get('error') or 'Barrier optimization failed'}
    else:
        assert isinstance(grid_long, dict) and isinstance(grid_short, dict)
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
    sec_bar['method'] = barrier_method
    sec_bar['search_profile'] = str(p.get('search_profile', 'medium'))
    sec_bar['note'] = (
        "Report barriers are produced by an independent optimization run; "
        "standalone forecast_barrier_optimize may yield different candidates. "
        "edge measures win-rate margin versus breakeven, while EV also weights reward/risk."
    )
    report['sections']['barriers'] = sec_bar

    # Patterns
    from ..patterns import patterns_detect
    pats = (
        _get_raw_result(
            patterns_detect,
            symbol=symbol,
            timeframe=tf,
            mode='candlestick',
            detail='compact',
            limit=int(p.get('patterns_limit', 120)),
            start=start,
            end=end,
        )
        if report_section_enabled(p, 'patterns')
        else {'error': 'patterns section not requested'}
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

    for section_name in list(report['sections']):
        if not report_section_enabled(p, section_name):
            report['sections'].pop(section_name, None)

    return report
