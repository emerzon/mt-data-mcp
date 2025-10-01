from typing import Any, Dict, Optional, List, Tuple, Callable
from datetime import datetime
import math
import copy




def now_utc_iso() -> str:
    try:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def parse_csv_tail(csv_header: Optional[str], csv_data: Optional[str], tail: int = 1) -> List[Dict[str, Any]]:
    try:
        if not csv_header or not csv_data:
            return []
        cols = [c.strip() for c in csv_header.split(',')]
        lines = [ln for ln in csv_data.split('\n') if ln.strip()]
        if tail > 0:
            lines = lines[-int(tail):]
        out: List[Dict[str, Any]] = []
        for ln in lines:
            vals = [v.strip() for v in ln.split(',')]
            row: Dict[str, Any] = {}
            for i, c in enumerate(cols):
                v = vals[i] if i < len(vals) else ''
                try:
                    if v == '' or v is None:
                        row[c] = v
                    elif '.' in v or 'e' in v.lower():
                        row[c] = float(v)
                    elif v.lstrip('-').isdigit():
                        row[c] = int(v)
                    else:
                        row[c] = v
                except Exception:
                    row[c] = v
            out.append(row)
        return out
    except Exception:
        return []


def pick_best_forecast_method(bt: Dict[str, Any], rmse_tolerance: float = 0.05) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        results = bt.get('results') if isinstance(bt, dict) else None
        if not isinstance(results, dict) or not results:
            return None
        entries: List[Tuple[str, Dict[str, Any], float, Optional[float], int]] = []
        for m, res in results.items():
            if not isinstance(res, dict):
                continue
            if not res.get('success'):
                continue
            try:
                rmse = float(res.get('avg_rmse', float('inf')))
            except Exception:
                rmse = float('inf')
            try:
                da_val = res.get('avg_directional_accuracy')
                da = float(da_val) if da_val is not None else float('nan')
            except Exception:
                da = float('nan')
            ok = int(res.get('successful_tests', 0))
            if not (rmse == rmse):
                continue
            entries.append((m, res, rmse, da if da == da else None, ok))
        if not entries:
            return None
        entries.sort(key=lambda x: x[2])
        best_rmse = entries[0][2]
        tol = max(0.0, float(rmse_tolerance))
        limit = best_rmse * (1.0 + tol)
        candidates = [e for e in entries if e[2] <= limit]
        with_dir = [e for e in candidates if e[3] is not None]
        if with_dir:
            with_dir.sort(key=lambda x: (-(x[3] or 0.0), x[2], -x[4]))
            chosen = with_dir[0]
        else:
            entries.sort(key=lambda x: (x[2], -x[4]))
            chosen = entries[0]
        return (chosen[0], chosen[1])
    except Exception:
        return None


def summarize_barrier_grid(grid: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    try:
        best = grid.get('best') if isinstance(grid, dict) else None
        top = grid.get('top') if isinstance(grid, dict) else None
        if not top and isinstance(grid.get('results'), list):
            items = grid['results']
            try:
                items = sorted(items, key=lambda x: float(x.get('score', x.get('edge', -1e9))), reverse=True)
            except Exception:
                pass
            top = items[:top_k]
        out: Dict[str, Any] = {}
        direction = grid.get('direction') if isinstance(grid, dict) else None
        if isinstance(best, dict):
            out['best'] = {
                'tp': best.get('tp'),
                'sl': best.get('sl'),
                'objective': best.get('objective') or grid.get('objective'),
                'edge': best.get('edge'),
                'kelly': best.get('kelly'),
                'kelly_uncond': best.get('kelly_uncond'),
                'ev': best.get('ev'),
                'ev_uncond': best.get('ev_uncond'),
                'prob_tp_first': best.get('prob_tp_first'),
                'prob_sl_first': best.get('prob_sl_first'),
                'prob_no_hit': best.get('prob_no_hit'),
                'median_time_to_tp': best.get('median_time_to_tp'),
                'tp_price': best.get('tp_price'),
                'sl_price': best.get('sl_price'),
            }
            if direction:
                out['direction'] = direction
        if isinstance(top, list):
            trimmed = []
            for it in top[:top_k]:
                if not isinstance(it, dict):
                    continue
                trimmed.append({
                    'tp': it.get('tp'), 'sl': it.get('sl'),
                    'tp_price': it.get('tp_price'), 'sl_price': it.get('sl_price'),
                    'edge': it.get('edge'), 'kelly': it.get('kelly'), 'kelly_uncond': it.get('kelly_uncond'), 'ev': it.get('ev'), 'ev_uncond': it.get('ev_uncond'),
                    'prob_tp_first': it.get('prob_tp_first'), 'prob_sl_first': it.get('prob_sl_first'), 'prob_no_hit': it.get('prob_no_hit'),
                })
            if trimmed:
                out['top'] = trimmed
        return out or {"note": "no grid summary"}
    except Exception:
        return {"note": "no grid summary"}


def merge_params(base: Optional[Dict[str, Any]], extra: Dict[str, Any], override: bool = False) -> Dict[str, Any]:
    p = dict(base or {})
    for k, v in extra.items():
        if override or k not in p:
            p[k] = v
    return p


def market_snapshot(symbol: str, timezone: str = 'UTC') -> Dict[str, Any]:
    try:
        from .market_depth import market_depth_fetch as _fetch_market_depth
        import MetaTrader5 as _mt5
        dom = _fetch_market_depth(symbol=symbol, timezone=timezone)
        bid = None; ask = None; spread = None
        top_buy_vol = None; top_sell_vol = None; total_buy_vol = None; total_sell_vol = None
        if isinstance(dom, dict) and dom.get('success'):
            t = dom.get('type')
            data = dom.get('data') or {}
            if t == 'tick_data':
                bid = data.get('bid'); ask = data.get('ask')
                if bid is not None and ask is not None:
                    try:
                        spread = float(ask) - float(bid)
                    except Exception:
                        spread = None
            elif t == 'full_depth':
                buys = data.get('buy_orders') or []
                sells = data.get('sell_orders') or []
                if buys:
                    try:
                        top_buy_vol = float(buys[0].get('volume') or 0.0)
                    except Exception:
                        top_buy_vol = None
                if sells:
                    try:
                        top_sell_vol = float(sells[0].get('volume') or 0.0)
                    except Exception:
                        top_sell_vol = None
                try:
                    total_buy_vol = float(sum(float(b.get('volume') or 0.0) for b in buys)) if buys else None
                except Exception:
                    total_buy_vol = None
                try:
                    total_sell_vol = float(sum(float(s.get('volume') or 0.0) for s in sells)) if sells else None
                except Exception:
                    total_sell_vol = None
        pip = None
        try:
            info = _mt5.symbol_info(symbol)
            if info is not None:
                digits = int(getattr(info, 'digits', 0) or 0)
                point = float(getattr(info, 'point', 0.0) or 0.0)
                pip = float(point * (10.0 if digits in (3, 5) else 1.0)) if point > 0 else None
        except Exception:
            pip = None
        spread_pips = None
        if pip and spread is not None:
            try:
                spread_pips = float(spread) / float(pip) if pip > 0 else None
            except Exception:
                spread_pips = None
        return {
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'spread_pips': spread_pips,
            'dom_top_buy_vol': top_buy_vol,
            'dom_top_sell_vol': top_sell_vol,
            'dom_total_buy_vol': total_buy_vol,
            'dom_total_sell_vol': total_sell_vol,
        }
    except Exception as e:
        return {'error': str(e)}


def apply_market_gates(section: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    gate = {}
    try:
        max_pips = params.get('spread_max_pips')
        if max_pips is not None and isinstance(section, dict):
            sp = section.get('spread_pips')
            if sp is not None:
                gate['spread_ok'] = bool(float(sp) <= float(max_pips))
                gate['spread_pips'] = float(sp)
                gate['spread_max_pips'] = float(max_pips)
    except Exception:
        pass
    return gate


def context_for_tf(symbol: str, timeframe: str, denoise: Optional[Dict[str, Any]], limit: int = 200, tail: int = 30) -> Optional[Dict[str, Any]]:
    try:
        from .data import data_fetch_candles as _fetch_candles
        indicators = "ema(20),ema(50),ema(200),rsi(14),macd(12,26,9)"
        res = _fetch_candles(symbol=symbol, timeframe=timeframe, limit=int(limit), indicators=indicators, denoise=denoise)

        # Handle both dictionary and string response formats
        if isinstance(res, dict):
            if 'error' in res:
                return None
            rows = parse_csv_tail(res.get('csv_header'), res.get('csv_data'), tail=int(tail))
        elif isinstance(res, str):
            # Parse CSV string directly
            lines = res.strip().split('\n')
            if len(lines) < 2:
                return None
            csv_header = lines[0]
            # Filter out metadata lines and empty lines
            data_lines = []
            for line in lines[1:]:
                line = line.strip()
                if line and not line.startswith(('symbol:', 'timeframe:', 'candles:')):
                    data_lines.append(line)
            if not data_lines:
                return None
            csv_data = '\n'.join(data_lines)
            rows = parse_csv_tail(csv_header, csv_data, tail=int(tail))
        else:
            return None

        if not rows:
            return None
        last = rows[-1]
        out = {
            'close': last.get('close'),
            'EMA_20': last.get('EMA_20') or last.get('ema_20'),
            'EMA_50': last.get('EMA_50') or last.get('ema_50'),
            'RSI_14': last.get('RSI_14') or last.get('rsi_14'),
            'MACD': last.get('MACD_12_26_9') or last.get('macd_12_26_9'),
        }

        # Compute trend compact data for MTF matrix
        try:
            from .report_templates.basic import _compute_compact_trend
            compact = _compute_compact_trend(rows)
            if compact:
                out['trend_compact'] = compact
        except Exception:
            # If trend compact calculation fails, continue without it
            pass

        # Add individual indicator values for MTF matrix
        if rows:
            last_row = rows[-1]
            out['rsi'] = last_row.get('RSI_14') or last_row.get('rsi_14')
            out['macd'] = last_row.get('MACD_12_26_9') or last_row.get('macd_12_26_9')
            out['macd_signal'] = last_row.get('MACDs_12_26_9') or last_row.get('macds_12_26_9')
            out['ema20'] = last_row.get('EMA_20') or last_row.get('ema_20')
            out['ema50'] = last_row.get('EMA_50') or last_row.get('ema_50')
            out['ema200'] = last_row.get('EMA_200') or last_row.get('ema_200')
            out['price'] = last_row.get('close')

        return out
    except Exception:
        return None


def attach_multi_timeframes(report: Dict[str, Any], symbol: str, denoise: Optional[Dict[str, Any]], extra_timeframes: List[str], pivot_timeframes: Optional[List[str]] = None) -> None:
    contexts: Dict[str, Any] = {}
    trend_mtf: Dict[str, Any] = {}

    for tf in extra_timeframes or []:
        snap = context_for_tf(symbol, tf, denoise, limit=200, tail=30)
        if snap:
            contexts[str(tf)] = snap

            # Extract trend compact data for MTF matrix
            if isinstance(snap, dict):
                trend_compact = snap.get('trend_compact')
                if isinstance(trend_compact, dict):
                    # Add individual indicator values to trend_mtf structure
                    trend_mtf_data = trend_compact.copy()
                    trend_mtf_data.update({
                        'rsi': snap.get('rsi'),
                        'macd': snap.get('macd'),
                        'macd_signal': snap.get('macd_signal'),
                        'ema20': snap.get('ema20'),
                        'ema50': snap.get('ema50'),
                        'ema200': snap.get('ema200'),
                        'price': snap.get('price')
                    })
                    trend_mtf[str(tf)] = trend_mtf_data

    if contexts:
        report.setdefault('sections', {})['contexts_multi'] = contexts

    # Attach compact trend info for the main context section
    if trend_mtf or contexts:
        sections = report.setdefault('sections', {})
        if 'context' in sections and isinstance(sections['context'], dict):
            sections['context']['trend_mtf'] = trend_mtf
        else:
            sections['context'] = {'trend_mtf': trend_mtf}

    base_pivot_tf = None
    try:
        base_pivot = report.setdefault('sections', {}).get('pivot')
        if isinstance(base_pivot, dict) and base_pivot.get('timeframe'):
            base_pivot_tf = str(base_pivot.get('timeframe')).upper()
    except Exception:
        base_pivot_tf = None

    if pivot_timeframes:
        filtered_tfs: List[str] = []
        for tfp in pivot_timeframes:
            tfp_str = str(tfp).upper()
            if base_pivot_tf and tfp_str == base_pivot_tf:
                continue
            filtered_tfs.append(str(tfp))
        pivs: Dict[str, Any] = {}
        if filtered_tfs:
            try:
                from .pivot import pivot_compute_points as _compute_pivot_points
                for tfp in filtered_tfs:
                    res = _compute_pivot_points(symbol=symbol, timeframe=tfp)
                    if isinstance(res, dict) and not res.get('error'):
                        pivs[str(tfp)] = {
                            'levels': res.get('levels'),
                            'methods': res.get('methods'),
                            'period': res.get('period'),
                            'timeframe': tfp,
                        }
            except Exception:
                pivs = {}
        if pivs:
            if base_pivot_tf:
                pivs['__base_timeframe__'] = base_pivot_tf
            report.setdefault('sections', {})['pivot_multi'] = pivs


def format_number(value: Any) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return '1' if value else '0'
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(num):
        return str(value)
    precision = 6 if abs(num) >= 1 else 8
    text = f"{num:.{precision}f}".rstrip('0').rstrip('.')
    if text in ('', '-0'):
        text = '0'
    return text


def _needs_yaml_quotes(text: str) -> bool:
    if text == '':
        return True
    if text != text.strip():
        return True
    if text[0] in {'-', '?', ':', '#', '!', '*', '&', '%', '@', '`', '{', '}', '[', ']', ',', '|', '>'}:
        return True
    for ch in (':', ',', '#'):
        if ch in text:
            return True
    return any(c in text for c in ('\n', '\r', '\t'))


def _escape_yaml_string(text: str) -> str:
    escaped = text.replace('\\', '\\\\').replace('"', '\\"').replace('\r', ' ').replace('\n', ' ')
    return escaped


def _compact_scalar(value: Any) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format_number(value)
    text = str(value)
    if _needs_yaml_quotes(text):
        return f'"{_escape_yaml_string(text)}"'
    return text


def _compact_yaml(value: Any, indent: int = 0) -> str:
    prefix = '  ' * indent
    if isinstance(value, dict):
        if not value:
            return f"{prefix}{{}}"
        lines: List[str] = []
        for key, val in value.items():
            key_str = str(key)
            if isinstance(val, (dict, list)):
                lines.append(f"{prefix}{key_str}:")
                lines.append(_compact_yaml(val, indent + 1))
            else:
                lines.append(f"{prefix}{key_str}: {_compact_scalar(val)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        lines: List[str] = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_compact_yaml(item, indent + 1))
            else:
                lines.append(f"{prefix}- {_compact_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{_compact_scalar(value)}"


def _compact_csv_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        text = 'true' if value else 'false'
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        text = format_number(value)
    else:
        text = str(value)
    text = text.replace('\r', ' ').replace('\n', ' ')
    if any(ch in text for ch in (',', '"')):
        text = '"' + text.replace('"', '""') + '"'
    return text


def _register_table(rows: List[Dict[str, Any]], path: str, tables: Dict[str, str], counters: Dict[str, int]) -> str:
    keys: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if key not in keys:
                keys.append(str(key))
    if not keys:
        return ''
    csv_lines = [','.join(keys)]
    for row in rows:
        csv_lines.append(','.join(_compact_csv_value(row.get(k)) for k in keys))
    base = path.lower().replace('.', '_').replace('[', '_').replace(']', '')
    base = ''.join(ch for ch in base if ch.isalnum() or ch == '_') or 'table'
    idx = counters.get(base, 0)
    counters[base] = idx + 1
    name = base if idx == 0 else f"{base}_{idx}"
    tables[name] = '\n'.join(csv_lines)
    return f"{name}_csv"


def _extract_tables(value: Any, path: str, tables: Dict[str, str], counters: Dict[str, int]):
    if isinstance(value, dict):
        return {k: _extract_tables(v, f"{path}.{k}", tables, counters) for k, v in value.items()}
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            marker = _register_table(value, path, tables, counters)
            return marker or []
        return [_extract_tables(item, f"{path}[{idx}]", tables, counters) for idx, item in enumerate(value)]
    return value


def render_enhanced_report(report: Dict[str, Any]) -> str:
    """Render a markdown report with a consistent basic-style layout."""
    if not isinstance(report, dict):
        return 'error: invalid report payload\n'

    output_lines: List[str] = []

    sections = report.get('sections', {})
    if not isinstance(sections, dict):
        sections = {}

    rendered_keys: List[str] = []
    for key, formatter in _SECTION_RENDERERS:
        payload = sections.get(key)
        if payload is None:
            continue
        block = formatter(payload) if payload else []
        rendered_keys.append(key)
        if block:
            output_lines.extend(block)
            output_lines.append('')

    for key in sorted(sections.keys()):
        if key in rendered_keys:
            continue
        payload = sections[key]
        if isinstance(payload, dict) and set(payload.keys()) <= {'error'}:
            continue
        block = _render_generic_section(key, payload)
        if block:
            output_lines.extend(block)
            output_lines.append('')

    rendered = '\n'.join(line.rstrip() for line in output_lines).rstrip()
    return rendered + '\n' if rendered else ''


def _render_context_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines: List[str] = ['## Market Context']

    metrics: List[List[Optional[str]]] = []
    tf_ref = data.get('timeframe')
    if tf_ref:
        metrics.append(['Timeframe', str(tf_ref)])
    snap = data.get('last_snapshot') if isinstance(data.get('last_snapshot'), dict) else {}
    price = snap.get('close')
    if price is not None:
        metrics.append(['Close', _format_decimal(price, 5)])
    ema20 = snap.get('EMA_20') or snap.get('ema_20')
    ema50 = snap.get('EMA_50') or snap.get('ema_50')
    if ema20 is not None and ema50 is not None:
        trend_state = 'Above EMAs'
        p_val = _as_float(price)
        e20 = _as_float(ema20)
        e50 = _as_float(ema50)
        if not (p_val is not None and e20 is not None and e50 is not None and p_val > e20 > e50):
            trend_state = 'Mixed slope'
        metrics.append([
            'EMA trend',
            f"{trend_state} (EMA20 {_format_decimal(ema20, 5)}, EMA50 {_format_decimal(ema50, 5)})",
        ])
    rsi = snap.get('RSI_14') or snap.get('rsi_14')
    if rsi is not None:
        rsi_val = _as_float(rsi)
        if rsi_val is not None:
            if rsi_val >= 70:
                tag = 'overbought'
            elif rsi_val <= 30:
                tag = 'oversold'
            else:
                tag = 'neutral'
            metrics.append(['RSI(14)', f"{_format_decimal(rsi, 2)} ({tag})"])
        else:
            metrics.append(['RSI(14)', _format_decimal(rsi, 2)])
    trend = data.get('trend_compact') if isinstance(data.get('trend_compact'), dict) else None
    if trend:
        slopes = trend.get('s') or []
        if slopes:
            slope_vals = ', '.join(_format_signed(s / 100.0) for s in slopes[:3])
            metrics.append(['Slope (ATR adj 5/20/60)', slope_vals])
        r2_vals = trend.get('r') or []
        if r2_vals:
            r2_fmt = ', '.join(f"{max(0, min(100, int(r)))}%" for r in r2_vals[:3])
            metrics.append(['Fit quality (R^2 5/20/60)', r2_fmt])
        atr_bps = trend.get('v')
        if atr_bps is not None:
            metrics.append(['ATR as bps', str(int(atr_bps))])
        squeeze = trend.get('q')
        if squeeze is not None:
            metrics.append(['Squeeze percentile', f"{int(squeeze)}%"])
        regime_map = {0: 'neutral', 1: 'uptrend', 2: 'downtrend', 3: 'breakout up', 4: 'breakout down'}
        regime = trend.get('g')
        if regime in regime_map:
            metrics.append(['Regime signal', regime_map[int(regime)]])
        bars_high = trend.get('h')
        bars_low = trend.get('l')
        if bars_high is not None or bars_low is not None:
            metrics.append([
                'Bars since swing high/low',
                f"{bars_high if bars_high is not None else 'n/a'} / {bars_low if bars_low is not None else 'n/a'}",
            ])
    if metrics:
        lines.extend(_format_table(['Metric', 'Value'], metrics))
    note = str(data.get('notes', '')).strip()
    if note:
        lines.append(f'- Note: {note}')
    return lines


def _render_contexts_multi_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    rows: List[List[Optional[str]]] = []
    for tf in sorted(data.keys()):
        snap = data[tf]
        if not isinstance(snap, dict):
            continue
        trend = snap.get('trend_compact') if isinstance(snap.get('trend_compact'), dict) else None
        slope_val = None
        atr_bps = None
        if trend:
            slopes = trend.get('s') or []
            if slopes:
                slope_val = slopes[0] / 100.0 if slopes[0] is not None else None
            atr_bps = trend.get('v')
        close_val = snap.get('close')
        ema20_val = snap.get('EMA_20') or snap.get('ema20')
        ema50_val = snap.get('EMA_50') or snap.get('ema50')
        ema200_val = snap.get('EMA_200') or snap.get('ema200')
        rsi_val = snap.get('RSI_14') or snap.get('rsi')
        rows.append([
            str(tf),
            _format_decimal(close_val, 5),
            _format_decimal(ema20_val, 5),
            _format_decimal(ema50_val, 5),
            _format_decimal(ema200_val, 5),
            _format_decimal(rsi_val, 2),
            _format_signed(slope_val),
            str(int(atr_bps)) if atr_bps is not None else None,
        ])
    if not rows:
        return []
    lines = ['## Multi-Timeframe Context']
    lines.extend(_format_table(['TF', 'Close', 'EMA20', 'EMA50', 'EMA200', 'RSI', 'Slope(5)', 'ATR bps'], rows))
    return lines


def _render_pivot_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    levels = data.get('levels')
    if not isinstance(levels, list) or not levels:
        return []
    timeframe = str(data.get('timeframe') or '').upper()
    period = data.get('period')
    title_parts: List[str] = []
    if timeframe:
        title_parts.append(timeframe)
    if isinstance(period, dict):
        start = period.get('start')
        end = period.get('end')
        if start and end:
            title_parts.append(f"{start}->{end}")
    elif period:
        title_parts.append(str(period))
    title = '## Pivot Levels'
    if title_parts:
        title += ' (' + ', '.join(title_parts) + ')'
    lines = [title]
    methods = []
    method_meta = data.get('methods')
    if isinstance(method_meta, list):
        for item in method_meta:
            if isinstance(item, dict):
                name = item.get('method')
                if name and name not in methods:
                    methods.append(str(name))
    if not methods:
        for row in levels:
            if isinstance(row, dict):
                for key in row.keys():
                    if key != 'level' and key not in methods:
                        methods.append(key)
    headers = ['Level'] + [m.title() for m in methods]
    table_rows: List[List[str]] = []
    for row in levels:
        if not isinstance(row, dict):
            continue
        level_name = str(row.get('level') or '').upper()
        row_vals = [level_name or 'n/a']
        for method in methods:
            value = row.get(method)
            row_vals.append(format_number(value) if value is not None else None)
        table_rows.append(row_vals)
    lines.extend(_format_table(headers, table_rows))
    return lines


def _render_pivot_multi_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    base_tf = str(data.get('__base_timeframe__') or '').upper()
    lines = ['## Multi-Timeframe Pivots']
    for tf in sorted(data.keys()):
        if str(tf).startswith('__'):
            continue
        if base_tf and str(tf).upper() == base_tf:
            continue
        piv = data[tf]
        if not isinstance(piv, dict):
            continue
        levels = piv.get('levels')
        if not isinstance(levels, list) or not levels:
            continue
        rows: List[List[str]] = []
        for row in levels:
            if not isinstance(row, dict):
                continue
            level = str(row.get('level') or '').upper()
            piv_val = row.get('classic') or row.get('Classic') or None
            rows.append([level or 'n/a', format_number(piv_val) if piv_val is not None else None])
        filtered_rows = [r for r in rows if not all((c is None or str(c).lower() == 'n/a') for c in r[1:])]
        if filtered_rows:
            lines.append(f"### {tf}")
            lines.extend(_format_table(['Level', 'Classic'], filtered_rows))
            lines.append('')
    result = [line for line in lines if line != '']
    return result if len(result) > 1 else []
def _render_pivot_multi_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    base_tf = str(data.get('__base_timeframe__', '')).upper() if '__base_timeframe__' in data else ''
    lines = ['## Multi-Timeframe Pivots']
    for tf in sorted(data.keys()):
        if str(tf).startswith('__'):
            continue
        if base_tf and str(tf).upper() == base_tf:
            continue
        piv = data[tf]
        if not isinstance(piv, dict):
            continue
        levels = piv.get('levels')
        if not isinstance(levels, list) or not levels:
            continue
        rows: List[List[str]] = []
        for row in levels:
            if not isinstance(row, dict):
                continue
            level = str(row.get('level') or '').upper()
            piv_val = row.get('classic') or row.get('Classic') or None
            rows.append([level, format_number(piv_val)])
        if rows:
            lines.append(f"### {tf}")
            lines.extend(_format_table(['Level', 'Classic'], rows))
            lines.append('')
    return [line for line in lines if line != '']


def _render_volatility_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    matrix = data.get('matrix')
    methods = data.get('methods')
    if not isinstance(matrix, list) or not matrix:
        return []
    if not isinstance(methods, list) or not methods:
        methods = [key for key in matrix[0].keys() if key not in {'horizon', 'avg'} and not str(key).endswith('_note') and not str(key).endswith('_bar') and not str(key).endswith('_err')]
    headers = ['Horizon'] + [m.upper() for m in methods]
    if any('avg' in row for row in matrix):
        headers.append('AVG')
    rows: List[List[str]] = []
    for row in matrix:
        if not isinstance(row, dict):
            continue
        horizon = row.get('horizon')
        line = [str(int(horizon)) if horizon is not None else 'n/a']
        for method in methods:
            val = row.get(method)
            line.append(format_number(val) if val is not None else 'n/a')
        if 'avg' in row:
            line.append(format_number(row.get('avg')) if row.get('avg') is not None else 'n/a')
        rows.append(line)
    if not rows:
        return []
    lines = ['## Volatility Snapshot', '*values are horizon sigma (returns)*']
    lines.extend(_format_table(headers, rows))
    return lines


def _render_forecast_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines = ['## Forecast']
    method = data.get('method')
    if method:
        lines.append(f"- Method: {method}")
    if data.get('forecast_price') is not None:
        lines.append(f"- Forecast price: {format_number(data['forecast_price'])}")
    lower = data.get('lower_price')
    upper = data.get('upper_price')
    if lower is not None or upper is not None:
        lines.append(f"- Interval: {format_number(lower)} to {format_number(upper)}")
    if data.get('trend'):
        lines.append(f"- Trend: {data['trend']}")
    if data.get('ci_alpha') is not None:
        lines.append(f"- CI alpha: {format_number(data['ci_alpha'])}")
    return lines


def _render_barriers_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    # If nested long/short payloads present, render a single matrix
    if any(k in data for k in ('long','short')):
        lines: List[str] = ['## Barrier Analytics']
        headers = ['Direction', 'TP %', 'SL %', 'TP lvl', 'SL lvl', 'Edge', 'Kelly', 'EV', 'TP hit %', 'SL hit %', 'No-hit %']
        rows: List[List[str]] = []
        for dir_name in ('long','short'):
            sub = data.get(dir_name)
            if not isinstance(sub, dict):
                continue
            best = sub.get('best') if isinstance(sub.get('best'), dict) else None
            if not best:
                continue
            rows.append([
                dir_name,
                _format_decimal(best.get('tp'), 3),
                _format_decimal(best.get('sl'), 3),
                _format_decimal(best.get('tp_price'), 5),
                _format_decimal(best.get('sl_price'), 5),
                _format_decimal(best.get('edge'), 3),
                _format_decimal(best.get('kelly'), 3),
                _format_decimal(best.get('ev'), 3),
                _format_probability(best.get('prob_tp_first')),
                _format_probability(best.get('prob_sl_first')),
                _format_probability(best.get('prob_no_hit')),
            ])
        if rows:
            lines.extend(_format_table(headers, rows))
            return lines
        return []
    # Fallback: single-direction shape
    lines = ['## Barrier Analytics']
    # Direction context line
    if isinstance(data.get('direction'), str):
        lines.append(f"- Direction: {data.get('direction')}")
    best = data.get('best') if isinstance(data.get('best'), dict) else None
    if best:
        headers = ['TP %', 'SL %', 'TP lvl', 'SL lvl', 'Edge', 'Kelly', 'EV', 'TP hit %', 'SL hit %', 'No-hit %']
        row = [
            _format_decimal(best.get('tp'), 3),
            _format_decimal(best.get('sl'), 3),
            _format_decimal(best.get('tp_price'), 5),
            _format_decimal(best.get('sl_price'), 5),
            _format_decimal(best.get('edge'), 3),
            _format_decimal(best.get('kelly'), 3),
            _format_decimal(best.get('ev'), 3),
            _format_probability(best.get('prob_tp_first')),
            _format_probability(best.get('prob_sl_first')),
            _format_probability(best.get('prob_no_hit')),
        ]
        lines.extend(_format_table(headers, [row]))
    top = data.get('top')
    if isinstance(top, list) and top:
        headers = ['Rank', 'TP %', 'SL %', 'Edge', 'Kelly', 'EV']
        rows: List[List[str]] = []
        for idx, row_data in enumerate(top, start=1):
            if not isinstance(row_data, dict):
                continue
            rows.append([
                str(idx),
                _format_decimal(row_data.get('tp'), 3),
                _format_decimal(row_data.get('sl'), 3),
                _format_decimal(row_data.get('edge'), 3),
                _format_decimal(row_data.get('kelly'), 3),
                _format_decimal(row_data.get('ev'), 3),
            ])
        if rows:
            lines.append('')
            lines.extend(_format_table(headers, rows))
    return lines if len(lines) > 1 else []


def _render_market_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines = ['## Market Snapshot']
    entries = []
    for label, key in [('Bid', 'bid'), ('Ask', 'ask'), ('Spread', 'spread'), ('Spread (pips)', 'spread_pips')]:
        val = data.get(key)
        if val is not None:
            entries.append(f"- {label}: {format_number(val)}")
    depth = data.get('depth')
    if isinstance(depth, dict):
        buy = depth.get('total_buy')
        sell = depth.get('total_sell')
        if buy is not None or sell is not None:
            entries.append(f"- DOM volume (buy/sell): {format_number(buy)} / {format_number(sell)}")
    if not entries:
        return []
    lines.extend(entries)
    return lines


def _render_backtest_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    ranking = data.get('ranking')
    if not isinstance(ranking, list) or not ranking:
        return []
    rows: List[List[str]] = []
    for row in ranking:
        if not isinstance(row, dict):
            continue
        rows.append([
            str(row.get('method') or ''),
            format_number(row.get('avg_rmse')),
            format_number(row.get('avg_mae')),
            format_number(row.get('avg_directional_accuracy')),
            str(row.get('successful_tests') or '0'),
        ])
    lines = ['## Backtest Ranking']
    lines.extend(_format_table(['Method', 'RMSE', 'MAE', 'DirAcc', 'Wins'], rows))
    return lines


def _render_patterns_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    recent = data.get('recent')
    if not isinstance(recent, list) or not recent:
        return []
    lines = ['## Recent Patterns']
    for row in recent:
        if not isinstance(row, dict):
            continue
        pattern = row.get('pattern') or row.get('Pattern') or row.get('name')
        time_val = row.get('time') or row.get('Time')
        direction = row.get('direction') or row.get('Direction')
        desc_parts: List[str] = []
        if pattern:
            desc_parts.append(str(pattern))
        if direction:
            desc_parts.append(str(direction))
        if time_val:
            desc_parts.append(str(time_val))
        if desc_parts:
            lines.append('- ' + ' | '.join(desc_parts))
    return lines if len(lines) > 1 else []


def _render_regime_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines = ['## Regime Signals']
    bocpd = data.get('bocpd')
    if isinstance(bocpd, dict):
        summary = bocpd.get('summary')
        if summary:
            lines.append(f"- BOCPD: {summary}")
        elif bocpd.get('error'):
            lines.append(f"- BOCPD error: {bocpd['error']}")
    hmm = data.get('hmm')
    if isinstance(hmm, dict):
        summary = hmm.get('summary')
        if summary:
            lines.append(f"- HMM: {summary}")
        elif hmm.get('error'):
            lines.append(f"- HMM error: {hmm['error']}")
    return lines if len(lines) > 1 else []


def _render_execution_gates_section(data: Any) -> List[str]:
    if not isinstance(data, dict) or not data:
        return []
    lines = ['## Execution Gates']
    for key, value in data.items():
        label = key.replace('_', ' ').title()
        if isinstance(value, bool):
            lines.append(f"- {label}: {'yes' if value else 'no'}")
        else:
            lines.append(f"- {label}: {format_number(value)}")
    return lines


def _render_volatility_har_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines = ['## HAR-RV Volatility']
    if data.get('sigma_bar_return') is not None:
        lines.append(f"- Bar sigma: {format_number(data['sigma_bar_return'])}")
    if data.get('horizon_sigma_return') is not None:
        lines.append(f"- Horizon sigma: {format_number(data['horizon_sigma_return'])}")
    return lines


def _render_forecast_conformal_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines = ['## Conformal Intervals']
    if data.get('method'):
        lines.append(f"- Method: {data['method']}")
    if data.get('lower_price') is not None or data.get('upper_price') is not None:
        lines.append(
            f"- Interval: {format_number(data.get('lower_price'))} to {format_number(data.get('upper_price'))}"
        )
    per_step = data.get('per_step_q')
    if isinstance(per_step, list) and per_step:
        sliced = per_step[:min(5, len(per_step))]
        lines.append(f"- First step quantiles: {', '.join(format_number(x) for x in sliced)}")
    if data.get('alpha') is not None:
        lines.append(f"- Alpha: {format_number(data['alpha'])}")
    return lines


def _render_generic_section(name: str, payload: Any) -> List[str]:
    if not payload:
        return []
    title = name.replace('_', ' ').title()
    lines = [f"## {title}"]
    if isinstance(payload, dict):
        for key in sorted(payload.keys()):
            val = payload[key]
            if isinstance(val, (list, tuple)):
                preview = ', '.join(str(item) for item in list(val)[:5])
                lines.append(f"- {key}: {preview}{'...' if len(val) > 5 else ''}")
            elif isinstance(val, dict):
                nested = ', '.join(f"{k}={v}" for k, v in list(val.items())[:5])
                lines.append(f"- {key}: {nested}{'...' if len(val) > 5 else ''}")
            else:
                lines.append(f"- {key}: {format_number(val)}")
    elif isinstance(payload, (list, tuple)):
        for item in payload[:20]:
            lines.append(f"- {item}")
        if len(payload) > 20:
            lines.append('- ...')
    else:
        lines.append(str(payload))
    return lines


def _format_table(headers: List[str], rows: List[List[Optional[Any]]]) -> List[str]:
    if not headers or not rows:
        return []
    header_line = ' | '.join(headers)
    divider = ' | '.join(['-' * max(3, len(h)) for h in headers])
    table = [header_line, divider]
    for row in rows:
        formatted_row: List[str] = []
        for idx in range(len(headers)):
            val = row[idx] if idx < len(row) else None
            if val is None or val == '' or str(val).lower() == 'null':
                formatted_row.append('n/a')
            else:
                formatted_row.append(str(val))
        table.append(' | '.join(formatted_row))
    return table


def _format_signed(value: Optional[float]) -> str:
    if value is None:
        return 'n/a'
    try:
        return f"{value:+.1f}"
    except Exception:
        return str(value)


def _format_decimal(value: Any, decimals: int = 4) -> Optional[str]:
    """Delegate to utils._format_float to avoid duplication."""
    from ..utils.utils import _format_float
    val = _as_float(value)
    if val is None:
        return None
    return _format_float(val, decimals)


def _format_probability(value: Optional[Any]) -> str:
    prob = _as_float(value)
    if prob is None:
        return 'n/a'
    return f"{prob * 100:.1f}%"


def _as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result

_SECTION_RENDERERS: List[Tuple[str, Callable[[Any], List[str]]]] = [
    ('context', _render_context_section),
    ('contexts_multi', _render_contexts_multi_section),
    ('pivot', _render_pivot_section),
    ('pivot_multi', _render_pivot_multi_section),
    ('volatility', _render_volatility_section),
    ('forecast', _render_forecast_section),
    ('barriers', _render_barriers_section),
    ('market', _render_market_section),
    ('backtest', _render_backtest_section),
    ('patterns', _render_patterns_section),
    ('regime', _render_regime_section),
    ('execution_gates', _render_execution_gates_section),
    ('volatility_har_rv', _render_volatility_har_section),
    ('forecast_conformal', _render_forecast_conformal_section),
]

_SECTION_RENDERERS: List[Tuple[str, Callable[[Any], List[str]]]] = [
    ('context', _render_context_section),
    ('contexts_multi', _render_contexts_multi_section),
    ('pivot', _render_pivot_section),
    ('pivot_multi', _render_pivot_multi_section),
    ('volatility', _render_volatility_section),
    ('forecast', _render_forecast_section),
    ('barriers', _render_barriers_section),
    ('market', _render_market_section),
    ('backtest', _render_backtest_section),
    ('patterns', _render_patterns_section),
    ('regime', _render_regime_section),
    ('execution_gates', _render_execution_gates_section),
    ('volatility_har_rv', _render_volatility_har_section),
    ('forecast_conformal', _render_forecast_conformal_section),
]

