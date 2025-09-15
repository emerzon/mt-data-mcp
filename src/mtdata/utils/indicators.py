from typing import Any, Dict, List, Optional, Tuple

import inspect
import pandas as pd
import pandas_ta as pta


def _list_ta_indicators() -> List[Dict[str, Any]]:
    """Dynamically list TA indicators available via pandas_ta."""
    items: List[Dict[str, Any]] = []
    
    # Use pandas_ta's own indicator categories to find real indicators
    categories = ['candles', 'momentum', 'overlap', 'performance', 'statistics', 'trend', 'volatility', 'volume', 'cycles']
    
    for category in categories:
        try:
            # Get the category module
            cat_module = getattr(pta, category, None)
            if not cat_module or not hasattr(cat_module, '__file__'):  # Check it's actually a module
                continue
                
            # Look for indicator functions in this category
            for name in dir(cat_module):
                if name.startswith('_'):
                    continue
                    
                obj = getattr(cat_module, name, None)
                if not callable(obj):
                    continue
                try:
                    sig = inspect.signature(obj)
                except Exception:
                    continue
                params = []
                for p in sig.parameters.values():
                    if p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                        continue
                    if p.name in ("close", "open", "high", "low", "volume", "length", "offset", "append", "talib", "mamode"):
                        default = None if p.default is inspect._empty else p.default
                        params.append({"name": p.name, "default": default})
                # Use the actual category
                items.append({"name": name, "params": params, "category": category, "description": obj.__doc__ or ""})
        except Exception:
            continue
    return items


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
