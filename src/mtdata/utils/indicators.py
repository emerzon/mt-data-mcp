from typing import Any, Dict, List, Optional, Tuple

import inspect
import pydoc
import re
import pandas as pd
try:
    import pandas_ta as pta  # preferred module name
except ModuleNotFoundError:  # fallback for alternate distribution import
    try:
        import pandas_ta_classic as pta
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "pandas_ta not found. Install 'pandas-ta-classic' (or 'pandas-ta')."
        ) from e


def clean_help_text(text: str, func_name: Optional[str] = None) -> str:
    if not isinstance(text, str):
        return ''
    cleaned = re.sub(r'.\x08', '', text)
    lines = [ln.rstrip() for ln in cleaned.splitlines()]
    sig_re = re.compile(rf"^\s*{re.escape(func_name)}\s*\(.*\)") if func_name else re.compile(r"^\s*\w+\s*\(.*\)")
    start = 0
    for i, ln in enumerate(lines):
        if sig_re.match(ln):
            start = i
            break
    kept = lines[start:]
    if kept:
        kept[0] = re.sub(r"\s+method of.*", "", kept[0], flags=re.IGNORECASE)
        if len(kept) > 1 and re.search(r"method of", kept[1], re.IGNORECASE):
            kept.pop(1)
    return "\n".join(kept).strip()


def _try_number(s: str):
    try:
        if '.' in s:
            return float(s)
        return int(s)
    except Exception:
        return None


def infer_defaults_from_doc(func_name: str, doc_text: str, params: List[Dict[str, Any]]):
    if not doc_text:
        return
    text = re.sub(r'.\x08', '', doc_text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sig_line = None
    for ln in lines:
        if ln.startswith(func_name + '(') or re.match(rf"^\s*{re.escape(func_name)}\s*\(.*\)", ln):
            sig_line = ln
            break
    if sig_line:
        inside = sig_line[sig_line.find('(') + 1 : sig_line.rfind(')')] if '(' in sig_line and ')' in sig_line else ''
        for part in re.split(r'[\s,]+', inside):
            if '=' in part:
                k, v = part.split('=', 1)
                k = k.strip()
                v = v.strip().strip(',)')
                num = _try_number(v)
                if num is not None:
                    for p in params:
                        if p.get('name') == k and 'default' not in p:
                            p['default'] = num
    for p in params:
        if 'default' in p:
            continue
        k = p.get('name')
        if not k:
            continue
        m = re.search(rf"{re.escape(k)}[^\n]*?(?:Default|default)\s*:?[\s]*([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            p['default'] = _try_number(m.group(1))


def list_ta_indicators(*, detailed: bool = False) -> List[Dict[str, Any]]:
    """Return [{'name','params','description','category'}, ...] discovered from pandas_ta."""
    items: List[Dict[str, Any]] = []
    seen = set()

    categories = [
        'candles', 'momentum', 'overlap', 'performance', 'statistics',
        'trend', 'volatility', 'volume', 'cycles'
    ]

    for category in categories:
        try:
            cat_module = getattr(pta, category, None)
            if not cat_module or not hasattr(cat_module, '__file__'):
                continue

            for func_name in dir(cat_module):
                if func_name.startswith('_'):
                    continue
                func = getattr(cat_module, func_name, None)
                if not callable(func):
                    continue
                name = func_name.lower()
                if name in seen:
                    continue
                seen.add(name)
                try:
                    sig = inspect.signature(func)
                except (TypeError, ValueError):
                    continue
                params: List[Dict[str, Any]] = []
                for p in sig.parameters.values():
                    if p.name in {"open", "high", "low", "close", "volume"}:
                        continue
                    if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                        continue
                    entry: Dict[str, Any] = {"name": p.name}
                    if p.default is not inspect._empty and p.default is not None:
                        entry["default"] = p.default
                    params.append(entry)

                desc = ""
                if detailed:
                    raw = ""
                    try:
                        raw = pydoc.render_doc(func)
                        desc = clean_help_text(raw, func_name=name)
                    except Exception:
                        desc = inspect.getdoc(func) or ""
                    try:
                        doc_text = inspect.getdoc(func) or raw
                        infer_defaults_from_doc(name, doc_text, params)
                    except Exception:
                        pass
                else:
                    try:
                        desc = inspect.getdoc(func) or ""
                    except Exception:
                        desc = ""

                items.append({
                    "name": name,
                    "params": params,
                    "description": desc,
                    "category": category,
                })
        except Exception:
            continue
    items.sort(key=lambda x: x["name"])
    return items


def _list_ta_indicators() -> List[Dict[str, Any]]:
    """Dynamically list TA indicators available via pandas_ta."""
    return list_ta_indicators(detailed=False)


def _parse_ti_specs(spec: str) -> List[Tuple[str, List[float], Dict[str, float]]]:
    """Parse a compact indicator spec string into [(name, args, kwargs)].

    Splits top-level by comma, respecting parentheses so nested commas in
    argument lists don't split functions. Supports numeric args and k=v pairs.
    """
    text = str(spec).strip()
    if not text:
        return []

    # Split by commas at top level only
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in text:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == ',' and depth == 0:
            token = ''.join(buf).strip()
            if token:
                parts.append(token)
            buf = []
        else:
            buf.append(ch)
    last = ''.join(buf).strip()
    if last:
        parts.append(last)

    specs: List[Tuple[str, List[float], Dict[str, float]]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        name = part
        args: List[float] = []
        kwargs: Dict[str, float] = {}
        if '(' in part and part.endswith(')'):
            name = part[: part.index('(')].strip()
            inside = part[part.index('(') + 1 : -1]
            # Split inside by commas (no nested parens expected here)
            for tok in inside.split(','):
                tok = tok.strip().strip('\"\'')
                if not tok:
                    continue
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        kwargs[k] = float(v)
                    except Exception:
                        # Keep non-numeric as-is? For now ignore invalid
                        pass
                else:
                    try:
                        args.append(float(tok))
                    except Exception:
                        pass
        # Flex: detect trailing number in name (EMA21 -> length=21)
        import re
        m = re.search(r"(.*?)[_\-]?([0-9]{1,3})$", name)
        if m and not args and 'length' not in kwargs:
            try:
                kwargs['length'] = float(m.group(2))
                name = m.group(1)
            except Exception:
                pass
        specs.append((name.strip(), args, kwargs))
    return specs


def _apply_ta_indicators(df: pd.DataFrame, ti_spec: str) -> List[str]:
    """Apply indicators specified by ti_spec to df in-place, return list of added column names."""
    added_cols: List[str] = []
    if not ti_spec:
        return added_cols
    # Many TA funcs expect a DatetimeIndex
    original_index = df.index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df['time'], unit='s', utc=True)
        except Exception:
            try:
                df.index = pd.to_datetime(df['time'])
            except Exception:
                pass
    before = set(df.columns)
    specs = _parse_ti_specs(ti_spec)
    for name, args, kwargs in specs:
        lname = name.lower()
        func = getattr(pta, lname, None)
        if not callable(func):
            continue
        try:
            sig = inspect.signature(func)
            params = sig.parameters
            # Prepare positional and keyword arguments safely
            call_kwargs = dict(kwargs)
            call_args = []
            # Provide price/series inputs
            if 'close' in params and 'close' in df.columns:
                # Use positional for 'close' to prevent numeric args binding to it
                call_args.append(df['close'])
                call_kwargs.pop('close', None)
            # Additional series as keywords if accepted
            if 'open' in params and 'open' not in call_kwargs and 'open' in df.columns:
                call_kwargs['open'] = df['open']
            if 'high' in params and 'high' not in call_kwargs and 'high' in df.columns:
                call_kwargs['high'] = df['high']
            if 'low' in params and 'low' not in call_kwargs and 'low' in df.columns:
                call_kwargs['low'] = df['low']
            if 'volume' in params and 'volume' not in call_kwargs and 'volume' in df.columns:
                call_kwargs['volume'] = df['volume']

            # Generic mapping: map provided numeric args to function parameters in declared order
            # Skip series parameters and any already supplied in call_kwargs
            series_names = {'open', 'high', 'low', 'close', 'volume'}
            ordered_param_names = []
            for pname, p in params.items():
                if p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                    continue
                if pname in series_names:
                    continue
                ordered_param_names.append(pname)
            # Assign args to next available param name not already set
            ai = 0
            for pname in ordered_param_names:
                if ai >= len(args):
                    break
                if pname in call_kwargs:
                    continue
                # Use provided arg in order
                call_kwargs[pname] = args[ai]
                ai += 1

            # Call indicator function with constructed arguments, with fallbacks
            out = None
            try:
                out = func(*call_args, **call_kwargs)
            except Exception:
                try:
                    # Fallback: also pass numeric args positionally after series
                    out = func(*([*call_args, *args]), **call_kwargs)
                except Exception:
                    try:
                        # Fallback: keyword-only attempt including close
                        kw_only = dict(call_kwargs)
                        if 'close' in params and 'close' in df.columns:
                            kw_only['close'] = df['close']
                        out = func(**kw_only)
                    except Exception:
                        out = None
            if isinstance(out, pd.DataFrame):
                for c in out.columns:
                    df[c] = out[c]
            elif isinstance(out, pd.Series):
                df[out.name or lname] = out
        except Exception:
            continue
        new_cols = [c for c in df.columns if c not in before]
        added_cols.extend(new_cols)
        before = set(df.columns)

    if original_index is not None:
        df.index = original_index
    return added_cols

# Backwards-compat alias
_apply_ta_indicators_util = _apply_ta_indicators


def _estimate_warmup_bars(ti_spec: Optional[str]) -> int:
    if not ti_spec:
        return 0
    max_warmup = 0
    specs = _parse_ti_specs(ti_spec)
    for name, args, kwargs in specs:
        lname = name.lower()
        def geti(key, default):
            if key in kwargs:
                try:
                    return int(kwargs[key])
                except Exception:
                    return default
            if args:
                try:
                    return int(args[0])
                except Exception:
                    return default
            return default
        warm = 0
        if lname in ("sma", "ema", "rsi"):
            warm = geti("length", 14)
        elif lname == "macd":
            fast = kwargs.get("fast", args[0] if len(args) > 0 else 12)
            slow = kwargs.get("slow", args[1] if len(args) > 1 else 26)
            try:
                warm = int(max(int(fast), int(slow)))
            except Exception:
                warm = 26
        elif lname == "stoch":
            k = kwargs.get("k", args[0] if len(args) > 0 else 14)
            d = kwargs.get("d", args[1] if len(args) > 1 else 3)
            s = kwargs.get("smooth", args[2] if len(args) > 2 else 3)
            try:
                warm = int(k) + int(d) + int(s)
            except Exception:
                warm = 20
        elif lname in ("bbands", "bb"):
            length = kwargs.get("length", args[0] if len(args) > 0 else 20)
            try:
                warm = int(length)
            except Exception:
                warm = 20
        else:
            warm = 50
        if warm > max_warmup:
            max_warmup = warm
    scaled = max(int(max_warmup * 3), 50) if max_warmup > 0 else 0
    return scaled

# Backwards-compat alias
_estimate_warmup_bars_util = _estimate_warmup_bars
