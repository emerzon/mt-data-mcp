from __future__ import annotations

"""Options market-data service helpers."""

import datetime as _dt
import email.utils as _email_utils
import logging
import re
import threading as _threading
import time as _time
from typing import Any, Callable, Dict, List, Optional

import requests

from ..bootstrap.settings import options_data_config

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
_TRADIER_DOCS_URL = "https://documentation.tradier.com/"
_YAHOO_AUTH_REMEDIATION = (
    "Run options_provider_status for configuration details. Yahoo options is "
    "unauthenticated and may reject requests; for reliable chains set "
    "MTDATA_OPTIONS_PROVIDER=tradier and MTDATA_OPTIONS_API_KEY."
)
_TRADIER_AUTH_REMEDIATION = (
    "Run options_provider_status for configuration details. Tradier options "
    "requires MTDATA_OPTIONS_API_KEY with MTDATA_OPTIONS_PROVIDER=tradier, or "
    "MTDATA_OPTIONS_PROVIDER=yahoo for the unauthenticated fallback."
)
_YAHOO_SESSION: Optional[requests.Session] = None
_YAHOO_SESSION_LOCK = _threading.Lock()
_YAHOO_RATE_LIMIT_LOCK = _threading.Lock()
_YAHOO_LAST_REQUEST_MONOTONIC = 0.0


class _OptionsRateLimitError(ValueError):
    def __init__(self, provider: str, retry_after_seconds: Optional[float]) -> None:
        self.provider = provider
        self.retry_after_seconds = retry_after_seconds
        retry_text = (
            f" retry_after_seconds={retry_after_seconds:g}"
            if retry_after_seconds is not None
            else ""
        )
        super().__init__(f"{provider} options provider rate limit exceeded.{retry_text}")


def _live_options_metadata(provider: str) -> Dict[str, Any]:
    return {
        "provider": provider,
        "cached": False,
        "data_age_seconds": 0,
    }


def _parse_retry_after_seconds(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        retry_at = _email_utils.parsedate_to_datetime(str(value))
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=_dt.timezone.utc)
        return max(
            0.0,
            (retry_at - _dt.datetime.now(tz=_dt.timezone.utc)).total_seconds(),
        )
    except Exception:
        return None


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


def _provider_remediation(provider: Any) -> str:
    provider_text = str(provider or "yahoo").strip().lower()
    if provider_text == "tradier":
        return _TRADIER_AUTH_REMEDIATION
    return _YAHOO_AUTH_REMEDIATION


def _attach_provider_remediation(out: Dict[str, Any], provider: Any) -> None:
    out["provider"] = str(provider or "yahoo").strip().lower() or "yahoo"
    out["next_tool"] = "options_provider_status"
    out["env_vars"] = ["MTDATA_OPTIONS_PROVIDER", "MTDATA_OPTIONS_API_KEY"]
    out["remediation"] = _provider_remediation(out["provider"])


def _retry_after_from_message(message: str) -> Optional[float]:
    match = re.search(r"retry_after_seconds=([0-9]+(?:\.[0-9]+)?)", message)
    if not match:
        return None
    try:
        return max(0.0, float(match.group(1)))
    except ValueError:
        return None


def _options_error(error: Any, *, prefix: Optional[str] = None) -> Dict[str, Any]:
    message_text = str(error)
    if prefix:
        message_text = f"{prefix}: {message_text}"
    out: Dict[str, Any] = {"success": False, "error": message_text}
    if isinstance(error, _OptionsRateLimitError):
        out["error_code"] = "options_provider_rate_limit"
        _attach_provider_remediation(out, error.provider)
        out["retry_after_seconds"] = error.retry_after_seconds
        return out
    if "Yahoo Finance options endpoint returned 401" in message_text:
        out["error_code"] = "options_provider_auth"
        _attach_provider_remediation(out, "yahoo")
    elif "Tradier options provider" in message_text or "Tradier options endpoint returned 401" in message_text:
        out["error_code"] = "options_provider_auth"
        _attach_provider_remediation(out, "tradier")
    elif "429" in message_text or "rate limit" in message_text.lower():
        out["error_code"] = "options_provider_rate_limit"
        _attach_provider_remediation(out, _configured_options_provider())
        out["retry_after_seconds"] = _retry_after_from_message(message_text)
    return out


def _configured_options_provider_mode() -> str:
    provider = str(getattr(options_data_config, "provider", "yahoo") or "yahoo").strip().lower()
    if provider not in {"auto", "yahoo", "tradier"}:
        return "yahoo"
    return provider


def _options_provider_attempt_order() -> List[str]:
    provider = _configured_options_provider_mode()
    if provider == "yahoo":
        return ["yahoo"]
    if provider == "auto":
        return ["tradier", "yahoo"] if getattr(options_data_config, "api_key", None) else ["yahoo"]
    return ["tradier", "yahoo"]


def _configured_options_provider() -> str:
    return _options_provider_attempt_order()[0]


def _provider_label(provider: str) -> str:
    return "Tradier" if provider == "tradier" else "Yahoo"


def _provider_attempt_metadata(
    provider: str,
    *,
    success: bool,
    error: Optional[BaseException] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"provider": provider, "success": bool(success)}
    if error is not None:
        out["error"] = str(error)
    return out


def _provider_error_from_payload(payload: Any) -> Optional[ValueError]:
    if not isinstance(payload, dict):
        return ValueError("Malformed options provider response")
    if payload.get("success") is True:
        return None
    message = payload.get("error")
    if message:
        return ValueError(str(message))
    return None


def _provider_failure_message(
    failures: List[tuple[str, BaseException]],
) -> str:
    parts: List[str] = []
    for index, (provider, error) in enumerate(failures):
        label = _provider_label(provider)
        prefix = (
            f"{label} options provider failed"
            if index == 0
            else f"{label} fallback also failed"
        )
        parts.append(f"{prefix}: {error}")
    return "; ".join(parts)


def _fallback_warning(
    failures: List[tuple[str, BaseException]],
    *,
    effective_provider: str,
) -> str:
    return (
        f"{_provider_label(effective_provider)} fallback returned data after "
        f"{_provider_failure_message(failures)}"
    )


def _annotate_fallback_payload(
    payload: Dict[str, Any],
    *,
    configured_provider: str,
    effective_provider: str,
    failures: List[tuple[str, BaseException]],
) -> Dict[str, Any]:
    out = dict(payload)
    out["configured_provider"] = configured_provider
    out["provider_effective"] = effective_provider
    out["warnings"] = [_fallback_warning(failures, effective_provider=effective_provider)]
    out["provider_attempts"] = [
        _provider_attempt_metadata(provider, success=False, error=error)
        for provider, error in failures
    ] + [_provider_attempt_metadata(effective_provider, success=True)]
    return out


def _provider_error_payload(
    error: Any,
    *,
    operation: str,
    provider: str,
    configured_provider: str,
    provider_attempts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload = _options_error(
        error,
        prefix=f"Failed to fetch {operation}",
    )
    payload["provider"] = provider
    payload["configured_provider"] = configured_provider
    if provider_attempts:
        payload["provider_attempts"] = provider_attempts
    return payload


def _run_options_provider_query(
    *,
    operation: str,
    yahoo_func: Callable[[], Dict[str, Any]],
    tradier_func: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    configured_provider = _configured_options_provider_mode()
    providers = _options_provider_attempt_order()
    failures: List[tuple[str, BaseException]] = []
    for index, provider in enumerate(providers):
        provider_func = tradier_func if provider == "tradier" else yahoo_func
        fallback_remaining = "yahoo" in providers[index + 1 :]
        try:
            payload = provider_func()
        except Exception as exc:
            failures.append((provider, exc))
            if fallback_remaining:
                logger.warning(
                    "%s options provider failed for %s; retrying Yahoo fallback: %s",
                    _provider_label(provider),
                    operation,
                    exc,
                )
                continue
            return _provider_error_payload(
                ValueError(_provider_failure_message(failures)),
                operation=operation,
                provider=provider,
                configured_provider=configured_provider,
                provider_attempts=[
                    _provider_attempt_metadata(item_provider, success=False, error=error)
                    for item_provider, error in failures
                ],
            )

        provider_error = _provider_error_from_payload(payload)
        if provider_error is None:
            if failures:
                return _annotate_fallback_payload(
                    payload,
                    configured_provider=configured_provider,
                    effective_provider=provider,
                    failures=failures,
                )
            return payload
        if fallback_remaining:
            failures.append((provider, provider_error))
            logger.warning(
                "%s options provider returned an error for %s; retrying Yahoo fallback: %s",
                _provider_label(provider),
                operation,
                provider_error,
            )
            continue
        if failures:
            failures.append((provider, provider_error))
            return _provider_error_payload(
                ValueError(_provider_failure_message(failures)),
                operation=operation,
                provider=provider,
                configured_provider=configured_provider,
                provider_attempts=[
                    _provider_attempt_metadata(item_provider, success=False, error=error)
                    for item_provider, error in failures
                ],
            )
        return payload
    return _provider_error_payload(
        RuntimeError("No supported options providers are available."),
        operation=operation,
        provider="yahoo",
        configured_provider=configured_provider,
    )


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
        retry_after = _parse_retry_after_seconds(response.headers.get("Retry-After"))
        if retry_after is None:
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
            if response.status_code == 429:
                raise _OptionsRateLimitError(
                    "yahoo",
                    _parse_retry_after_seconds(response.headers.get("Retry-After")),
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


def _tradier_http_get(path: str, *, params: Dict[str, Any]) -> Dict[str, Any]:
    api_key = getattr(options_data_config, "api_key", None)
    if not api_key:
        raise ValueError(
            "Authentication error: Tradier options provider requires "
            "MTDATA_OPTIONS_API_KEY."
        )
    base_url = str(getattr(options_data_config, "base_url", "") or "https://api.tradier.com/v1").rstrip("/")
    url = f"{base_url}/{str(path).lstrip('/')}"
    response = requests.get(
        url,
        params=params,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        timeout=_HTTP_TIMEOUT,
    )
    try:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            if response.status_code == 401:
                raise ValueError(
                    "Authentication error: Tradier options endpoint returned "
                    "401 Unauthorized."
                )
            if response.status_code == 429:
                raise _OptionsRateLimitError(
                    "tradier",
                    _parse_retry_after_seconds(response.headers.get("Retry-After")),
                )
            raise
        payload = response.json()
    finally:
        response.close()
    if not isinstance(payload, dict):
        raise ValueError("Malformed Tradier options response")
    return payload


def _fetch_tradier_expirations_payload(symbol: str) -> Dict[str, Any]:
    return _tradier_http_get(
        "/markets/options/expirations",
        params={
            "symbol": str(symbol).upper().strip(),
            "includeAllRoots": "true",
            "strikes": "false",
        },
    )


def _fetch_tradier_chain_payload(symbol: str, expiration: str) -> Dict[str, Any]:
    return _tradier_http_get(
        "/markets/options/chains",
        params={
            "symbol": str(symbol).upper().strip(),
            "expiration": str(expiration).strip(),
            "greeks": "true",
        },
    )


def _fetch_tradier_quote_payload(symbol: str) -> Dict[str, Any]:
    return _tradier_http_get(
        "/markets/quotes",
        params={"symbols": str(symbol).upper().strip()},
    )


def _extract_tradier_expiration_dates(payload: Dict[str, Any]) -> List[str]:
    expirations = payload.get("expirations")
    date_values: Any = None
    if isinstance(expirations, dict):
        date_values = expirations.get("date")
    elif isinstance(expirations, list):
        date_values = expirations
    if isinstance(date_values, str):
        values = [date_values]
    elif isinstance(date_values, list):
        values = [str(value).strip() for value in date_values]
    else:
        values = []
    return sorted(value for value in values if value)


def _extract_tradier_quote(payload: Dict[str, Any]) -> Dict[str, Any]:
    quotes = payload.get("quotes")
    quote = quotes.get("quote") if isinstance(quotes, dict) else None
    if isinstance(quote, list):
        quote = quote[0] if quote else {}
    return quote if isinstance(quote, dict) else {}


def _extract_tradier_option_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = payload.get("options")
    rows = options.get("option") if isinstance(options, dict) else None
    if isinstance(rows, dict):
        return [rows]
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _parse_tradier_epoch(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, (int, float)):
        return _to_numeric(value, int, 0, field_name="lastTradeDate")
    text = str(value).strip()
    try:
        parsed = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.timezone.utc)
        return int(parsed.timestamp())
    except Exception:
        return 0


def _tradier_option_side(row: Dict[str, Any]) -> str:
    raw = str(row.get("option_type") or row.get("type") or "").strip().lower()
    if raw in {"call", "put"}:
        return raw
    contract = str(row.get("symbol") or row.get("contractSymbol") or "").upper()
    if "C" in contract[-9:]:
        return "call"
    if "P" in contract[-9:]:
        return "put"
    return raw or "unknown"


def _normalize_tradier_options(
    rows: List[Dict[str, Any]],
    *,
    option_type: str,
    min_open_interest: int,
    min_volume: int,
    limit: int,
    underlying_price: Any,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        underlying = float(underlying_price)
    except Exception:
        underlying = float("nan")
    for row in rows:
        side = _tradier_option_side(row)
        if option_type != "both" and side != option_type:
            continue
        oi = max(0, _to_numeric(row.get("open_interest"), int, 0, field_name="open_interest"))
        vol = max(0, _to_numeric(row.get("volume"), int, 0, field_name="volume"))
        if oi < min_open_interest or vol < min_volume:
            continue
        strike = _to_numeric(row.get("strike"), float, float("nan"), field_name="strike")
        if not (strike == strike and strike > 0):
            continue
        greeks = row.get("greeks") if isinstance(row.get("greeks"), dict) else {}
        implied_volatility = row.get("implied_volatility")
        if implied_volatility in (None, ""):
            implied_volatility = greeks.get("mid_iv")
        in_the_money = False
        if underlying == underlying:
            if side == "call":
                in_the_money = float(strike) < underlying
            elif side == "put":
                in_the_money = float(strike) > underlying
        entry: Dict[str, Any] = {
            "side": side,
            "contract": row.get("symbol") or row.get("contractSymbol"),
            "strike": float(strike),
            "last": _to_numeric(row.get("last"), float, float("nan"), field_name="last"),
            "bid": _to_numeric(row.get("bid"), float, float("nan"), field_name="bid"),
            "ask": _to_numeric(row.get("ask"), float, float("nan"), field_name="ask"),
            "change": _to_numeric(row.get("change"), float, float("nan"), field_name="change"),
            "percent_change": _to_numeric(
                row.get("change_percentage"),
                float,
                float("nan"),
                field_name="change_percentage",
            ),
            "volume": int(vol),
            "open_interest": int(oi),
            "implied_volatility": _to_numeric(
                implied_volatility,
                float,
                float("nan"),
                field_name="implied_volatility",
            ),
            "in_the_money": bool(in_the_money),
            "last_trade_epoch": _parse_tradier_epoch(row.get("trade_date") or row.get("last_trade_date")),
            "currency": row.get("currency") or "USD",
        }
        out.append(entry)
    out.sort(key=lambda item: (item.get("side") != "call", float(item.get("strike", 0.0))))
    return out[:limit]


def _get_tradier_options_expirations(symbol: str) -> Dict[str, Any]:
    symbol_norm = str(symbol).upper().strip()
    payload = _fetch_tradier_expirations_payload(symbol_norm)
    expirations = _extract_tradier_expiration_dates(payload)
    quote: Dict[str, Any] = {}
    try:
        quote = _extract_tradier_quote(_fetch_tradier_quote_payload(symbol_norm))
    except Exception:
        quote = {}
    return {
        "success": True,
        **_live_options_metadata("tradier"),
        "symbol": symbol_norm,
        "underlying_price": _to_numeric(
            quote.get("last") or quote.get("close"),
            float,
            float("nan"),
            field_name="quote.last",
        ),
        "currency": quote.get("currency") or "USD",
        "expirations": expirations,
        "expiration_count": int(len(expirations)),
    }


def _get_tradier_options_chain(
    *,
    symbol: str,
    expiration: Optional[str],
    option_type: str,
    min_open_interest: int,
    min_volume: int,
    limit: int,
) -> Dict[str, Any]:
    symbol_norm = str(symbol).upper().strip()
    expirations = _extract_tradier_expiration_dates(
        _fetch_tradier_expirations_payload(symbol_norm)
    )
    if not expirations:
        return {"error": f"No option expirations found for {symbol_norm}"}
    chosen_expiry = str(expiration).strip() if expiration else expirations[0]
    if chosen_expiry not in expirations:
        return {
            "error": f"Requested expiration {chosen_expiry} not available for {symbol_norm}",
            "expirations": expirations,
        }
    quote: Dict[str, Any] = {}
    try:
        quote = _extract_tradier_quote(_fetch_tradier_quote_payload(symbol_norm))
    except Exception:
        quote = {}
    underlying_price = _to_numeric(
        quote.get("last") or quote.get("close"),
        float,
        float("nan"),
        field_name="quote.last",
    )
    rows = _extract_tradier_option_rows(
        _fetch_tradier_chain_payload(symbol_norm, chosen_expiry)
    )
    normalized = _normalize_tradier_options(
        rows,
        option_type=option_type,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
        limit=limit,
        underlying_price=underlying_price,
    )
    return {
        "success": True,
        **_live_options_metadata("tradier"),
        "symbol": symbol_norm,
        "expiration": chosen_expiry,
        "underlying_price": underlying_price,
        "currency": quote.get("currency") or "USD",
        "contract_size": "REGULAR",
        "expirations": expirations,
        "option_type": option_type,
        "min_open_interest": int(min_open_interest),
        "min_volume": int(min_volume),
        "count": int(len(normalized)),
        "calls_count": sum(1 for item in normalized if item.get("side") == "call"),
        "puts_count": sum(1 for item in normalized if item.get("side") == "put"),
        "options": normalized,
    }


def _get_yahoo_options_expirations(symbol: str) -> Dict[str, Any]:
    payload = _fetch_yahoo_options_payload(symbol)
    expiration_epochs = _extract_expiration_epochs(payload)
    expirations = [_epoch_to_ymd(v) for v in expiration_epochs]
    quote = payload.get("quote", {}) if isinstance(payload.get("quote"), dict) else {}
    return {
        "success": True,
        **_live_options_metadata("yahoo"),
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


def _get_yahoo_options_chain(
    *,
    symbol: str,
    expiration: Optional[str],
    option_type: str,
    min_open_interest: int,
    min_volume: int,
    limit: int,
) -> Dict[str, Any]:
    symbol_norm = str(symbol).upper().strip()
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

    def _norm(rows: List[Dict[str, Any]], side: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            oi = max(0, _to_numeric(row.get("openInterest"), int, 0, field_name="openInterest"))
            vol = max(0, _to_numeric(row.get("volume"), int, 0, field_name="volume"))
            if oi < min_open_interest or vol < min_volume:
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

    calls = _norm(calls_raw, "call") if option_type in {"call", "both"} else []
    puts = _norm(puts_raw, "put") if option_type in {"put", "both"} else []
    combined = (calls + puts)[:limit]

    return {
        "success": True,
        **_live_options_metadata("yahoo"),
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
        "option_type": option_type,
        "min_open_interest": int(min_open_interest),
        "min_volume": int(min_volume),
        "count": int(len(combined)),
        "calls_count": int(len(calls)),
        "puts_count": int(len(puts)),
        "options": combined,
    }


def get_options_expirations(symbol: str) -> Dict[str, Any]:
    """Return available option expirations for a symbol."""
    try:
        return _run_options_provider_query(
            operation="options expirations",
            yahoo_func=lambda: _get_yahoo_options_expirations(symbol),
            tradier_func=lambda: _get_tradier_options_expirations(symbol),
        )
    except Exception as e:
        return _options_error(e, prefix="Failed to fetch options expirations")


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
        min_oi = max(0, _to_numeric(min_open_interest, int, 0, field_name="min_open_interest"))
        min_vol = max(0, _to_numeric(min_volume, int, 0, field_name="min_volume"))
        max_rows = max(1, _to_numeric(limit, int, 200, field_name="limit"))

        return _run_options_provider_query(
            operation="options chain",
            yahoo_func=lambda: _get_yahoo_options_chain(
                symbol=symbol_norm,
                expiration=expiration,
                option_type=option_type_norm,
                min_open_interest=min_oi,
                min_volume=min_vol,
                limit=max_rows,
            ),
            tradier_func=lambda: _get_tradier_options_chain(
                symbol=symbol_norm,
                expiration=expiration,
                option_type=option_type_norm,
                min_open_interest=min_oi,
                min_volume=min_vol,
                limit=max_rows,
            ),
        )
    except Exception as e:
        return _options_error(e, prefix="Failed to fetch options chain")
