from __future__ import annotations

"""Options market-data service helpers."""

from typing import Any, Dict, List, Optional
import datetime as _dt

import requests


_YAHOO_OPTIONS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
_HTTP_TIMEOUT = 15.0


def _to_numeric(value: Any, numeric_type: type, default: Any) -> Any:
    try:
        return numeric_type(value)
    except Exception:
        return numeric_type(default)


def _extract_expiration_epochs(payload: Dict[str, Any]) -> List[int]:
    expiration_epochs = payload.get("expirationDates", [])
    if not isinstance(expiration_epochs, list):
        expiration_epochs = []
    return sorted(
        {
            _to_numeric(value, int, 0)
            for value in expiration_epochs
            if isinstance(value, (int, float))
        }
    )


def _epoch_to_ymd(epoch: int) -> str:
    dt = _dt.datetime.fromtimestamp(int(epoch), tz=_dt.timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _ymd_to_epoch(ymd: str) -> int:
    dt = _dt.datetime.strptime(str(ymd).strip(), "%Y-%m-%d")
    dt = dt.replace(tzinfo=_dt.timezone.utc)
    return int(dt.timestamp())


def _fetch_yahoo_options_payload(symbol: str, expiry_epoch: Optional[int] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if expiry_epoch is not None:
        params["date"] = int(expiry_epoch)
    headers = {"User-Agent": "Mozilla/5.0"}
    url = _YAHOO_OPTIONS_URL.format(symbol=str(symbol).upper().strip())
    response = requests.get(url, params=params, headers=headers, timeout=_HTTP_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    chain = data.get("optionChain", {})
    results = chain.get("result", [])
    if not isinstance(results, list) or not results:
        raise ValueError(f"No options data found for {symbol}")
    item = results[0]
    if not isinstance(item, dict):
        raise ValueError(f"Malformed options response for {symbol}")
    return item


def get_options_expirations(symbol: str) -> Dict[str, Any]:
    """Return available option expirations for a symbol."""
    try:
        payload = _fetch_yahoo_options_payload(symbol)
        expiration_epochs = _extract_expiration_epochs(payload)
        expirations = [_epoch_to_ymd(v) for v in expiration_epochs]
        quote = payload.get("quote", {}) if isinstance(payload.get("quote"), dict) else {}
        return {
            "success": True,
            "symbol": str(symbol).upper().strip(),
            "underlying_price": _to_numeric(quote.get("regularMarketPrice"), float, float("nan")),
            "currency": quote.get("currency"),
            "expirations": expirations,
            "expiration_count": int(len(expirations)),
        }
    except Exception as e:
        return {"error": f"Failed to fetch options expirations: {e}"}


def get_options_chain(
    symbol: str,
    expiration: Optional[str] = None,
    option_type: str = "both",
    min_open_interest: int = 0,
    min_volume: int = 0,
    limit: int = 200,
) -> Dict[str, Any]:
    """Fetch options chain (calls/puts) for a symbol and expiration."""
    try:
        symbol_norm = str(symbol).upper().strip()
        option_type_norm = str(option_type or "both").lower().strip()
        if option_type_norm not in {"call", "put", "both"}:
            return {"error": f"Invalid option_type: {option_type}. Use call|put|both."}

        base = _fetch_yahoo_options_payload(symbol_norm)
        expiration_epochs = _extract_expiration_epochs(base)
        if not expiration_epochs:
            return {"error": f"No option expirations found for {symbol_norm}"}

        available_map = {_epoch_to_ymd(ep): int(ep) for ep in expiration_epochs}
        chosen_expiry_ymd: str
        chosen_expiry_epoch: int
        if expiration is None:
            chosen_expiry_epoch = int(expiration_epochs[0])
            chosen_expiry_ymd = _epoch_to_ymd(chosen_expiry_epoch)
        else:
            chosen_expiry_ymd = str(expiration).strip()
            chosen_expiry_epoch = int(available_map.get(chosen_expiry_ymd, -1))
            if chosen_expiry_epoch < 0:
                return {
                    "error": f"Requested expiration {chosen_expiry_ymd} not available for {symbol_norm}",
                    "expirations": sorted(list(available_map.keys())),
                }

        payload = _fetch_yahoo_options_payload(symbol_norm, chosen_expiry_epoch)
        quote = payload.get("quote", {}) if isinstance(payload.get("quote"), dict) else {}
        options_arr = payload.get("options", [])
        if not isinstance(options_arr, list) or not options_arr:
            return {"error": f"No options chain returned for {symbol_norm} @ {chosen_expiry_ymd}"}
        chain = options_arr[0] if isinstance(options_arr[0], dict) else {}
        calls_raw = chain.get("calls", []) if isinstance(chain, dict) else []
        puts_raw = chain.get("puts", []) if isinstance(chain, dict) else []
        calls_raw = calls_raw if isinstance(calls_raw, list) else []
        puts_raw = puts_raw if isinstance(puts_raw, list) else []

        min_oi = max(0, _to_numeric(min_open_interest, int, 0))
        min_vol = max(0, _to_numeric(min_volume, int, 0))
        max_rows = max(1, _to_numeric(limit, int, 200))

        def _norm(rows: List[Dict[str, Any]], side: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                oi = max(0, _to_numeric(row.get("openInterest"), int, 0))
                vol = max(0, _to_numeric(row.get("volume"), int, 0))
                if oi < min_oi or vol < min_vol:
                    continue
                strike = _to_numeric(row.get("strike"), float, float("nan"))
                if not (strike == strike and strike > 0):
                    continue
                entry: Dict[str, Any] = {
                    "side": side,
                    "contract": row.get("contractSymbol"),
                    "strike": float(strike),
                    "last": _to_numeric(row.get("lastPrice"), float, float("nan")),
                    "bid": _to_numeric(row.get("bid"), float, float("nan")),
                    "ask": _to_numeric(row.get("ask"), float, float("nan")),
                    "change": _to_numeric(row.get("change"), float, float("nan")),
                    "percent_change": _to_numeric(row.get("percentChange"), float, float("nan")),
                    "volume": int(vol),
                    "open_interest": int(oi),
                    "implied_volatility": _to_numeric(row.get("impliedVolatility"), float, float("nan")),
                    "in_the_money": bool(row.get("inTheMoney", False)),
                    "last_trade_epoch": _to_numeric(row.get("lastTradeDate"), int, 0),
                    "currency": row.get("currency"),
                }
                out.append(entry)
            out.sort(key=lambda x: float(x.get("strike", 0.0)))
            return out

        calls = _norm(calls_raw, "call") if option_type_norm in {"call", "both"} else []
        puts = _norm(puts_raw, "put") if option_type_norm in {"put", "both"} else []
        combined = (calls + puts)[:max_rows]

        return {
            "success": True,
            "symbol": symbol_norm,
            "expiration": chosen_expiry_ymd,
            "underlying_price": _to_numeric(quote.get("regularMarketPrice"), float, float("nan")),
            "currency": quote.get("currency"),
            "contract_size": quote.get("contractSize"),
            "expirations": sorted(list(available_map.keys())),
            "option_type": option_type_norm,
            "min_open_interest": int(min_oi),
            "min_volume": int(min_vol),
            "count": int(len(combined)),
            "calls_count": int(len(calls)),
            "puts_count": int(len(puts)),
            "options": combined,
        }
    except Exception as e:
        return {"error": f"Failed to fetch options chain: {e}"}
