from __future__ import annotations

"""Options market-data service helpers."""

import datetime as _dt
import logging
import threading as _threading
import time as _time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_YAHOO_OPTIONS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
_HTTP_TIMEOUT = 15.0
_YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_YAHOO_RETRY_STATUS_CODES = {429, 503}
_YAHOO_MAX_ATTEMPTS = 3
_YAHOO_BACKOFF_SECONDS = 0.5
_YAHOO_MIN_REQUEST_INTERVAL_SECONDS = 1.0
_YAHOO_AUTH_REMEDIATION = (
    "Yahoo Finance options data is unavailable from the unauthenticated endpoint. "
    "mtdata has no Yahoo API-key setting to configure; use another options data "
    "provider."
)
_YAHOO_SESSION: Optional[requests.Session] = None
_YAHOO_SESSION_LOCK = _threading.Lock()
_YAHOO_RATE_LIMIT_LOCK = _threading.Lock()
_YAHOO_LAST_REQUEST_MONOTONIC = 0.0


def _to_numeric(
    value: Any,
    numeric_type: type,
    default: Any,
    *,
    field_name: Optional[str] = None,
) -> Any:
    try:
        return numeric_type(value)
    except Exception as exc:
        fallback = numeric_type(default)
        if value not in (None, ""):
            type_name = getattr(numeric_type, "__name__", str(numeric_type))
            field_label = f" '{field_name}'" if field_name else ""
            logger.warning(
                "Failed to coerce Yahoo options%s value %r to %s; using default %r: %s",
                field_label,
                value,
                type_name,
                fallback,
                exc,
            )
        return fallback


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


def _build_yahoo_session() -> requests.Session:
    """Create a configured Yahoo HTTP session."""
    return requests.Session()


def _reset_yahoo_session() -> None:
    """Close and discard the shared Yahoo session so the next request rebuilds it."""
    global _YAHOO_SESSION
    with _YAHOO_SESSION_LOCK:
        old = _YAHOO_SESSION
        _YAHOO_SESSION = None
    if old is not None:
        try:
            old.close()
        except Exception:
            pass


def _get_yahoo_session() -> requests.Session:
    global _YAHOO_SESSION
    with _YAHOO_SESSION_LOCK:
        if _YAHOO_SESSION is None:
            _YAHOO_SESSION = _build_yahoo_session()
        return _YAHOO_SESSION


def _options_error(message: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"success": False, "error": str(message)}
    message_text = str(message)
    if "Yahoo Finance options endpoint returned 401" in message_text:
        out["error_code"] = "options_provider_auth"
        out["remediation"] = _YAHOO_AUTH_REMEDIATION
    return out


def _throttle_yahoo_request() -> None:
    global _YAHOO_LAST_REQUEST_MONOTONIC
    with _YAHOO_RATE_LIMIT_LOCK:
        now = _time.monotonic()
        wait_seconds = _YAHOO_MIN_REQUEST_INTERVAL_SECONDS - (now - _YAHOO_LAST_REQUEST_MONOTONIC)
        if _YAHOO_LAST_REQUEST_MONOTONIC > 0.0 and wait_seconds > 0.0:
            _time.sleep(wait_seconds)
            now = _time.monotonic()
        _YAHOO_LAST_REQUEST_MONOTONIC = now


def _yahoo_http_get(url: str, *, params: Dict[str, Any], headers: Dict[str, str]) -> requests.Response:
    session = _get_yahoo_session()
    backoff_seconds = _YAHOO_BACKOFF_SECONDS
    response: Optional[requests.Response] = None
    for attempt in range(_YAHOO_MAX_ATTEMPTS):
        _throttle_yahoo_request()
        response = session.get(url, params=params, headers=headers, timeout=_HTTP_TIMEOUT)
        if response.status_code not in _YAHOO_RETRY_STATUS_CODES or attempt + 1 >= _YAHOO_MAX_ATTEMPTS:
            return response
        response.close()
        retry_after_raw = response.headers.get("Retry-After")
        try:
            retry_after = float(retry_after_raw) if retry_after_raw is not None else backoff_seconds
        except (TypeError, ValueError):
            retry_after = backoff_seconds
        _time.sleep(max(backoff_seconds, retry_after))
        backoff_seconds *= 2.0
    if response is None:
        raise RuntimeError("Yahoo options request did not return a response")
    return response


def _fetch_yahoo_options_payload(symbol: str, expiry_epoch: Optional[int] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if expiry_epoch is not None:
        params["date"] = int(expiry_epoch)
    url = _YAHOO_OPTIONS_URL.format(symbol=str(symbol).upper().strip())
    response = _yahoo_http_get(url, params=params, headers=dict(_YAHOO_HEADERS))
    try:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            # Sanitize 401 errors to avoid exposing API URLs to users
            if response.status_code == 401:
                raise ValueError(
                    "Authentication error: Yahoo Finance options endpoint returned "
                    "401 Unauthorized. No mtdata API-key setting is available for this "
                    "Yahoo endpoint."
                )
            # For other HTTP errors, re-raise as-is
            raise
        data = response.json()
    finally:
        response.close()
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
            "underlying_price": _to_numeric(
                quote.get("regularMarketPrice"),
                float,
                float("nan"),
                field_name="quote.regularMarketPrice",
            ),
            "currency": quote.get("currency"),
            "expirations": expirations,
            "expiration_count": int(len(expirations)),
        }
    except Exception as e:
        return _options_error(f"Failed to fetch options expirations: {e}")


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
                    "expirations": sorted(available_map),
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

        min_oi = max(0, _to_numeric(min_open_interest, int, 0, field_name="min_open_interest"))
        min_vol = max(0, _to_numeric(min_volume, int, 0, field_name="min_volume"))
        max_rows = max(1, _to_numeric(limit, int, 200, field_name="limit"))

        def _norm(rows: List[Dict[str, Any]], side: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                oi = max(0, _to_numeric(row.get("openInterest"), int, 0, field_name="openInterest"))
                vol = max(0, _to_numeric(row.get("volume"), int, 0, field_name="volume"))
                if oi < min_oi or vol < min_vol:
                    continue
                strike = _to_numeric(row.get("strike"), float, float("nan"), field_name="strike")
                if not (strike == strike and strike > 0):
                    continue
                entry: Dict[str, Any] = {
                    "side": side,
                    "contract": row.get("contractSymbol"),
                    "strike": float(strike),
                    "last": _to_numeric(row.get("lastPrice"), float, float("nan"), field_name="lastPrice"),
                    "bid": _to_numeric(row.get("bid"), float, float("nan"), field_name="bid"),
                    "ask": _to_numeric(row.get("ask"), float, float("nan"), field_name="ask"),
                    "change": _to_numeric(row.get("change"), float, float("nan"), field_name="change"),
                    "percent_change": _to_numeric(
                        row.get("percentChange"),
                        float,
                        float("nan"),
                        field_name="percentChange",
                    ),
                    "volume": int(vol),
                    "open_interest": int(oi),
                    "implied_volatility": _to_numeric(
                        row.get("impliedVolatility"),
                        float,
                        float("nan"),
                        field_name="impliedVolatility",
                    ),
                    "in_the_money": bool(row.get("inTheMoney", False)),
                    "last_trade_epoch": _to_numeric(row.get("lastTradeDate"), int, 0, field_name="lastTradeDate"),
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
            "underlying_price": _to_numeric(
                quote.get("regularMarketPrice"),
                float,
                float("nan"),
                field_name="quote.regularMarketPrice",
            ),
            "currency": quote.get("currency"),
            "contract_size": quote.get("contractSize"),
            "expirations": sorted(available_map),
            "option_type": option_type_norm,
            "min_open_interest": int(min_oi),
            "min_volume": int(min_vol),
            "count": int(len(combined)),
            "calls_count": int(len(calls)),
            "puts_count": int(len(puts)),
            "options": combined,
        }
    except Exception as e:
        return _options_error(f"Failed to fetch options chain: {e}")
