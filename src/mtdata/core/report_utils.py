from typing import Any, Dict, Optional, List, Tuple
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
        if isinstance(best, dict):
            out['best'] = {
                'tp': best.get('tp'),
                'sl': best.get('sl'),
                'objective': best.get('objective') or grid.get('objective'),
                'edge': best.get('edge'),
                'kelly': best.get('kelly'),
                'ev': best.get('ev'),
                'prob_tp_first': best.get('prob_tp_first'),
                'prob_sl_first': best.get('prob_sl_first'),
                'prob_no_hit': best.get('prob_no_hit'),
                'median_time_to_tp': best.get('median_time_to_tp'),
            }
        if isinstance(top, list):
            trimmed = []
            for it in top[:top_k]:
                if not isinstance(it, dict):
                    continue
                trimmed.append({
                    'tp': it.get('tp'), 'sl': it.get('sl'),
                    'edge': it.get('edge'), 'kelly': it.get('kelly'), 'ev': it.get('ev'),
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
        indicators = "ema(20),ema(50),rsi(14),macd(12,26,9)"
        res = _fetch_candles(symbol=symbol, timeframe=timeframe, limit=int(limit), indicators=indicators, denoise=denoise, timezone='UTC')
        if 'error' in res:
            return None
        rows = parse_csv_tail(res.get('csv_header'), res.get('csv_data'), tail=int(tail))
        if not rows:
            return None
        last = rows[-1]
        out = {
            'close': last.get('close'),
            'EMA_20': last.get('EMA_20') or last.get('ema_20'),
            'EMA_50': last.get('EMA_50') or last.get('ema_50'),
            'RSI_14': last.get('RSI_14') or last.get('rsi_14'),
            'MACD': last.get('MACD') or last.get('macd'),
        }
        return out
    except Exception:
        return None


def attach_multi_timeframes(report: Dict[str, Any], symbol: str, denoise: Optional[Dict[str, Any]], extra_timeframes: List[str], pivot_timeframes: Optional[List[str]] = None) -> None:
    contexts: Dict[str, Any] = {}
    for tf in extra_timeframes or []:
        snap = context_for_tf(symbol, tf, denoise, limit=200, tail=30)
        if snap:
            contexts[str(tf)] = snap
    if contexts:
        report.setdefault('sections', {})['contexts_multi'] = contexts
    if pivot_timeframes:
        pivs: Dict[str, Any] = {}
        try:
            from .pivot import pivot_compute_points as _compute_pivot_points
            for tfp in pivot_timeframes:
                res = _compute_pivot_points(symbol=symbol, timeframe=tfp)
                if isinstance(res, dict) and not res.get('error'):
                    pivs[str(tfp)] = {'levels': res.get('levels'), 'methods': res.get('methods'), 'period': res.get('period')}
        except Exception:
            pivs = {}
        if pivs:
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


def render_compact_report(report: Dict[str, Any]) -> str:
    if not isinstance(report, dict):
        return 'error: invalid report payload\n'
    tables: Dict[str, str] = {}
    counters: Dict[str, int] = {}
    sanitized = copy.deepcopy(report)
    summary_raw = sanitized.get('summary') if isinstance(sanitized.get('summary'), list) else []
    summary_lines = [str(item).strip() for item in summary_raw if str(item).strip()]
    sanitized['summary'] = summary_lines
    transformed = _extract_tables(sanitized, 'report', tables, counters)
    yaml_body = _compact_yaml(transformed)
    output_lines: List[str] = []
    if summary_lines:
        output_lines.append('summary')
        output_lines.extend(summary_lines)
    if tables:
        if output_lines:
            output_lines.append('')
        output_lines.append('tables')
        for name, csv_text in tables.items():
            output_lines.append(name)
            output_lines.extend(csv_text.splitlines())
    if yaml_body:
        if output_lines:
            output_lines.append('')
        output_lines.append('yaml')
        output_lines.extend(yaml_body.splitlines())
    if not output_lines:
        return 'yaml\n' + (yaml_body or '') + ('\n' if yaml_body and not yaml_body.endswith('\n') else '')
    text = '\n'.join(output_lines).rstrip()
    return text + '\n'
