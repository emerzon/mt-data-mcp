"""Market status and trading hours MCP tool."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

import holidays

from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise
from ..utils.mt5_enums import decode_mt5_enum_label
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway
from .output_contract import resolve_output_detail
from .schema import CompactFullDetailLiteral

logger = logging.getLogger(__name__)


# Market definitions with trading hours (local time)
_MARKETS = {
    "NYSE": {
        "name": "New York Stock Exchange",
        "country": "US",
        "timezone": "America/New_York",
        "open": (9, 30),  # 9:30 AM
        "close": (16, 0),  # 4:00 PM
        "early_close": (13, 0),  # 1:00 PM on some holidays
        "early_close_holidays": [],
        "early_close_day_after": ["Thanksgiving"],
    },
    "NASDAQ": {
        "name": "NASDAQ",
        "country": "US",
        "timezone": "America/New_York",
        "open": (9, 30),
        "close": (16, 0),
        "early_close": (13, 0),
        "early_close_holidays": [],
        "early_close_day_after": ["Thanksgiving"],
    },
    "LSE": {
        "name": "London Stock Exchange",
        "country": "UK",
        "timezone": "Europe/London",
        "open": (8, 0),  # 8:00 AM
        "close": (16, 30),  # 4:30 PM
        "early_close": None,
        "early_close_holidays": [],
    },
    "XETRA": {
        "name": "Xetra (Frankfurt)",
        "country": "DE",
        "timezone": "Europe/Berlin",
        "open": (9, 0),  # 9:00 AM
        "close": (17, 30),  # 5:30 PM
        "early_close": None,
        "early_close_holidays": [],
    },
    "EURONEXT": {
        "name": "Euronext Paris",
        "country": "FR",
        "timezone": "Europe/Paris",
        "open": (9, 0),  # 9:00 AM
        "close": (17, 30),  # 5:30 PM
        "early_close": None,
        "early_close_holidays": [],
    },
    "TSE": {
        "name": "Tokyo Stock Exchange",
        "country": "JP",
        "timezone": "Asia/Tokyo",
        "open": (9, 0),  # 9:00 AM
        "close": (15, 0),  # 3:00 PM
        "lunch_start": (11, 30),
        "lunch_end": (12, 30),
        "early_close": None,
        "early_close_holidays": [],
    },
    "HKEX": {
        "name": "Hong Kong Stock Exchange",
        "country": "HK",
        "timezone": "Asia/Hong_Kong",
        "open": (9, 30),  # 9:30 AM
        "close": (16, 0),  # 4:00 PM
        "lunch_start": (12, 0),
        "lunch_end": (13, 0),
        "early_close": (12, 0),
        "early_close_holidays": [],
        "early_close_eves": ["Christmas Day", "New Year's Day"],
    },
    "SSE": {
        "name": "Shanghai Stock Exchange",
        "country": "CN",
        "timezone": "Asia/Shanghai",
        "open": (9, 30),  # 9:30 AM
        "close": (15, 0),  # 3:00 PM
        "lunch_start": (11, 30),
        "lunch_end": (13, 0),
        "early_close": None,
        "early_close_holidays": [],
    },
    "ASX": {
        "name": "Australian Securities Exchange",
        "country": "AU",
        "timezone": "Australia/Sydney",
        "open": (10, 0),  # 10:00 AM
        "close": (16, 0),  # 4:00 PM
        "early_close": (14, 0),  # 2:00 PM
        "early_close_holidays": [],
        "early_close_eves": ["Christmas Day"],
    },
}


@lru_cache(maxsize=64)
def _get_holidays(country: str, year: int) -> holidays.HolidayBase:
    """Get the holiday calendar for a country/year pair."""
    return holidays.country_holidays(country, years=[int(year)])


def _is_holiday(country: str, dt: datetime) -> Tuple[bool, Optional[str]]:
    """Check if date is a holiday and return holiday name if so."""
    h = _get_holidays(country, dt.year)
    date_key = dt.date()
    if date_key in h:
        return True, str(h[date_key])
    return False, None


def _get_local_time(tz_name: str) -> datetime:
    """Get current time in specified timezone."""
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_name)
    except ImportError:
        try:
            from dateutil import tz as dateutil_tz
            tz = dateutil_tz.gettz(tz_name)
        except Exception:
            tz = timezone.utc
    return datetime.now(tz)


def _normalize_time(dt: datetime) -> datetime:
    """Normalize datetime for comparison."""
    return dt.replace(second=0, microsecond=0)


def _format_duration(minutes: int) -> str:
    """Format minutes into human-readable duration."""
    if minutes < 60:
        return f"{minutes}min{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    mins = minutes % 60
    if mins == 0:
        return f"{hours}h{'ours' if hours > 1 else 'our'}"
    return f"{hours}h {mins}min{'s' if mins != 1 else ''}"


def _normalize_timezone_display(value: Optional[str]) -> Optional[str]:
    normalized = str(value or "local").strip().lower()
    if normalized == "auto":
        return "local"
    if normalized in {"local", "utc"}:
        return normalized
    return None


def _format_market_time(value: Any, display: str) -> Any:
    if display != "utc":
        return value
    if not isinstance(value, str) or not value:
        return value
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    if dt.tzinfo is None:
        return value
    return dt.astimezone(timezone.utc).isoformat()


def _apply_market_timezone_display(
    status: Dict[str, Any],
    *,
    now_local: datetime,
    display: str,
) -> Dict[str, Any]:
    if display != "utc":
        return status
    out = dict(status)
    out["local_time"] = now_local.astimezone(timezone.utc).strftime("%H:%M")
    for key in ("next_open", "next_close"):
        if key in out:
            out[key] = _format_market_time(out[key], display)
    return out


def _apply_global_weekend_reason(status: Dict[str, Any], *, now_utc: datetime) -> Dict[str, Any]:
    if now_utc.weekday() < 5:
        return status
    if status.get("status") != "closed" or status.get("reason") != "after_hours":
        return status
    out = dict(status)
    out["reason"] = "weekend"
    return out


def _is_early_close_session(
    market: Dict[str, Any],
    country: str,
    session_dt: datetime,
) -> bool:
    """Return whether *session_dt* should trade as an early-close session."""
    is_holiday_result, holiday_name = _is_holiday(country, session_dt)

    if is_holiday_result and holiday_name and market.get("early_close_holidays"):
        for h_name in market["early_close_holidays"]:
            if h_name.lower() in holiday_name.lower():
                return True

    if market.get("early_close_day_after"):
        yesterday = session_dt - timedelta(days=1)
        _, yesterday_holiday = _is_holiday(country, yesterday)
        if yesterday_holiday:
            for h_name in market["early_close_day_after"]:
                if h_name.lower() in yesterday_holiday.lower():
                    return True

    if market.get("early_close_eves"):
        tomorrow = session_dt + timedelta(days=1)
        _, tomorrow_holiday = _is_holiday(country, tomorrow)
        if tomorrow_holiday:
            for eve_name in market["early_close_eves"]:
                if eve_name.lower() in tomorrow_holiday.lower():
                    return True

    return False


def _next_market_open_datetime(
    market: Dict[str, Any],
    country: str,
    now_local: datetime,
) -> datetime:
    """Return the next tradable session open after *now_local*."""
    next_open = now_local + timedelta(days=1)
    while True:
        if next_open.weekday() >= 5:
            next_open += timedelta(days=1)
            continue
        is_holiday_result, _holiday_name = _is_holiday(country, next_open)
        if is_holiday_result and not _is_early_close_session(market, country, next_open):
            next_open += timedelta(days=1)
            continue
        return next_open.replace(
            hour=market["open"][0],
            minute=market["open"][1],
            second=0,
            microsecond=0,
        )


def _check_market_status(market_id: str, now_local: datetime) -> Dict[str, Any]:
    """Check status for a single market."""
    market = _MARKETS[market_id]
    country = market["country"]
    
    # Check weekend
    weekday = now_local.weekday()
    if weekday >= 5:  # Saturday or Sunday
        next_open = _next_market_open_datetime(market, country, now_local)
        minutes_until = int((next_open - _normalize_time(now_local)).total_seconds() // 60)
        return {
            "symbol": market_id,
            "name": market["name"],
            "status": "closed",
            "reason": "weekend",
            "local_time": now_local.strftime("%H:%M"),
            "message": f"{market_id}: Closed (opening in {_format_duration(minutes_until)})",
            "next_open": next_open.isoformat(),
            "minutes_until": minutes_until,
        }
    
    # Check holidays
    is_holiday_result, holiday_name = _is_holiday(country, now_local)

    # Determine early close BEFORE the holiday return so same-day
    # half-holidays are not treated as full closures.
    is_early_close = _is_early_close_session(market, country, now_local)

    # Full holiday (not a half-day session) → closed
    if is_holiday_result and not is_early_close:
        next_open = _next_market_open_datetime(market, country, now_local)
        minutes_until = int((next_open - _normalize_time(now_local)).total_seconds() // 60)
        return {
            "symbol": market_id,
            "name": market["name"],
            "status": "closed",
            "reason": "holiday",
            "holiday": holiday_name,
            "local_time": now_local.strftime("%H:%M"),
            "message": f"{market_id}: Closed - Holiday ({holiday_name}, opening in {_format_duration(minutes_until)})",
            "next_open": next_open.isoformat(),
            "minutes_until": minutes_until,
        }
    
    open_hour, open_minute = market["open"]
    close_hour, close_minute = market["close"]
    
    if is_early_close and market.get("early_close"):
        close_hour, close_minute = market["early_close"]
    
    open_time = now_local.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)
    close_time = now_local.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
    
    # Check pre-market (before open)
    now_norm = _normalize_time(now_local)
    if now_norm < open_time:
        minutes_until_open = int((open_time - now_norm).total_seconds() // 60)
        return {
            "symbol": market_id,
            "name": market["name"],
            "status": "pre_market",
            "local_time": now_local.strftime("%H:%M"),
            "message": f"{market_id}: Pre-market (opening in {_format_duration(minutes_until_open)})",
            "next_open": open_time.isoformat(),
            "minutes_until": minutes_until_open,
        }
    
    # Check if during lunch break
    if market.get("lunch_start") and market.get("lunch_end"):
        lunch_start = now_local.replace(hour=market["lunch_start"][0], minute=market["lunch_start"][1], second=0, microsecond=0)
        lunch_end = now_local.replace(hour=market["lunch_end"][0], minute=market["lunch_end"][1], second=0, microsecond=0)
        
        if lunch_start <= now_norm < lunch_end:
            minutes_until_resume = int((lunch_end - now_norm).total_seconds() // 60)
            return {
                "symbol": market_id,
                "name": market["name"],
                "status": "lunch_break",
                "local_time": now_local.strftime("%H:%M"),
                "message": f"{market_id}: Lunch break (resuming in {_format_duration(minutes_until_resume)})",
                "next_open": lunch_end.isoformat(),
                "minutes_until": minutes_until_resume,
            }
    
    # Check if market is open
    if now_norm < close_time:
        minutes_until_close = int((close_time - now_norm).total_seconds() // 60)
        return {
            "symbol": market_id,
            "name": market["name"],
            "status": "open",
            "local_time": now_local.strftime("%H:%M"),
            "message": f"{market_id}: Open (closing in {_format_duration(minutes_until_close)})",
            "next_close": close_time.isoformat(),
            "minutes_until": minutes_until_close,
        }
    
    # Market closed for the day
    next_open = _next_market_open_datetime(market, country, now_local)
    minutes_until = int((next_open - now_norm).total_seconds() // 60)
    
    return {
        "symbol": market_id,
        "name": market["name"],
        "status": "closed",
        "reason": "after_hours",
        "local_time": now_local.strftime("%H:%M"),
        "message": f"{market_id}: Closed (opening in {_format_duration(minutes_until)})",
        "next_open": next_open.isoformat(),
        "minutes_until": minutes_until,
    }


def _get_upcoming_holidays(market_ids: List[str], days_ahead: int = 14) -> List[Dict[str, Any]]:
    """Get upcoming holidays that will close markets within the next N days."""
    upcoming: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    seen: set = set()
    
    for market_id in market_ids:
        if market_id not in _MARKETS:
            continue
        
        market = _MARKETS[market_id]
        country = market["country"]
        
        try:
            # Check next N days
            for i in range(1, days_ahead + 1):
                check_date = now + timedelta(days=i)
                date_key = check_date.date()

                is_holiday_result, holiday_name = _is_holiday(country, check_date)
                if is_holiday_result and holiday_name is not None:
                    key = (country, date_key.isoformat())
                    
                    if key not in seen:
                        seen.add(key)
                        
                        # Determine impact: same-day half-holiday
                        is_early_close = False
                        early_close_time = None
                        if market.get("early_close_holidays"):
                            for h_name in market["early_close_holidays"]:
                                if h_name.lower() in holiday_name.lower():
                                    is_early_close = True
                                    break

                        if is_early_close and market.get("early_close"):
                            early_close_time = f"{market['early_close'][0]:02d}:{market['early_close'][1]:02d}"
                        
                        upcoming.append({
                            "date": date_key.isoformat(),
                            "holiday": holiday_name,
                            "country": country,
                            "markets_affected": [market_id],
                            "impact": "early_close" if is_early_close else "closed",
                            "early_close_time": early_close_time,
                            "days_away": i,
                        })

                        # Day-after: the day after this holiday may be an
                        # early close (e.g. Black Friday after Thanksgiving).
                        if market.get("early_close_day_after"):
                            for h_name in market["early_close_day_after"]:
                                if h_name.lower() in holiday_name.lower():
                                    after_date = check_date + timedelta(days=1)
                                    after_key = (country, after_date.date().isoformat())
                                    if after_key not in seen and after_date.date().weekday() < 5:
                                        seen.add(after_key)
                                        ect = None
                                        if market.get("early_close"):
                                            ect = f"{market['early_close'][0]:02d}:{market['early_close'][1]:02d}"
                                        upcoming.append({
                                            "date": after_date.date().isoformat(),
                                            "holiday": f"Day after {holiday_name}",
                                            "country": country,
                                            "markets_affected": [market_id],
                                            "impact": "early_close",
                                            "early_close_time": ect,
                                            "days_away": i + 1,
                                        })
                                    break

                        # Eve: the day before this holiday may be an early close.
                        if market.get("early_close_eves"):
                            for eve_name in market["early_close_eves"]:
                                if eve_name.lower() in holiday_name.lower():
                                    eve_date = check_date - timedelta(days=1)
                                    eve_key = (country, eve_date.date().isoformat())
                                    if eve_key not in seen and eve_date.date().weekday() < 5:
                                        seen.add(eve_key)
                                        ect = None
                                        if market.get("early_close"):
                                            ect = f"{market['early_close'][0]:02d}:{market['early_close'][1]:02d}"
                                        upcoming.append({
                                            "date": eve_date.date().isoformat(),
                                            "holiday": f"Eve of {holiday_name}",
                                            "country": country,
                                            "markets_affected": [market_id],
                                            "impact": "early_close",
                                            "early_close_time": ect,
                                            "days_away": max(0, i - 1),
                                        })
                                    break
                    else:
                        # Add market to existing holiday entry
                        for entry in upcoming:
                            if entry["date"] == date_key.isoformat() and entry["country"] == country:
                                if market_id not in entry["markets_affected"]:
                                    entry["markets_affected"].append(market_id)
                                break
        except Exception as exc:
            logger.warning(f"Failed to get holidays for {country}: {exc}")
    
    # Sort by date
    upcoming.sort(key=lambda x: (x["date"], x["country"]))
    return upcoming


def _summarize_upcoming_holiday(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry

    out: Dict[str, Any] = {}
    for key in ("date", "holiday", "impact", "days_away", "markets_affected"):
        if key in entry and entry.get(key) is not None:
            out[key] = entry.get(key)
    if entry.get("impact") == "early_close" and entry.get("early_close_time") is not None:
        out["early_close_time"] = entry.get("early_close_time")
    return out


def normalize_market_status_output(
    result: Dict[str, Any],
    *,
    detail: Any = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return dict(result)

    detail_mode = resolve_output_detail(detail=detail)
    out = dict(result)
    if detail_mode == "full":
        return out

    upcoming = out.pop("upcoming_holidays", None)
    if isinstance(upcoming, list) and upcoming:
        out["upcoming_holidays_count"] = len(upcoming)
        out["upcoming_holidays_summary"] = [
            _summarize_upcoming_holiday(entry) for entry in upcoming
        ]
    return out


def _symbol_trade_mode_status(gateway: Any, trade_mode: Any) -> Dict[str, Any]:
    label = decode_mt5_enum_label(
        gateway,
        trade_mode,
        prefix="SYMBOL_TRADE_MODE_",
    )
    label_text = str(label or "").strip()
    normalized = label_text.lower().replace("symbol_trade_mode_", "")
    if not normalized:
        normalized = str(trade_mode).strip().lower()

    full_values = {
        getattr(gateway, "SYMBOL_TRADE_MODE_FULL", object()),
    }
    disabled_values = {
        getattr(gateway, "SYMBOL_TRADE_MODE_DISABLED", object()),
    }
    close_only_values = {
        getattr(gateway, "SYMBOL_TRADE_MODE_CLOSEONLY", object()),
    }
    long_only_values = {
        getattr(gateway, "SYMBOL_TRADE_MODE_LONGONLY", object()),
    }
    short_only_values = {
        getattr(gateway, "SYMBOL_TRADE_MODE_SHORTONLY", object()),
    }

    if trade_mode in disabled_values or "disabled" in normalized:
        status = "disabled"
        can_open = False
    elif trade_mode in close_only_values or "close" in normalized:
        status = "close_only"
        can_open = False
    elif trade_mode in long_only_values or "long" in normalized:
        status = "long_only"
        can_open = True
    elif trade_mode in short_only_values or "short" in normalized:
        status = "short_only"
        can_open = True
    elif trade_mode in full_values or "full" in normalized:
        status = "tradable"
        can_open = True
    else:
        status = "unknown"
        can_open = None

    return {
        "trade_mode": trade_mode,
        "trade_mode_label": label_text or None,
        "status": status,
        "can_open_new_positions": can_open,
    }


def _symbol_tick_snapshot(tick: Any, *, now_utc: datetime) -> Dict[str, Any]:
    if tick is None:
        return {
            "tick_available": False,
            "tick_freshness": "missing",
        }

    out: Dict[str, Any] = {
        "tick_available": True,
    }
    tick_time = getattr(tick, "time", None)
    if tick_time is not None:
        try:
            tick_epoch = float(tick_time)
            age_seconds = max(0.0, now_utc.timestamp() - tick_epoch)
            out["last_tick_time"] = datetime.fromtimestamp(
                tick_epoch,
                tz=timezone.utc,
            ).isoformat()
            out["last_tick_age_seconds"] = round(age_seconds, 3)
            out["tick_freshness"] = "fresh" if age_seconds <= 300.0 else "stale"
        except (OSError, OverflowError, TypeError, ValueError):
            out["tick_freshness"] = "unknown"
    else:
        out["tick_freshness"] = "unknown"

    for field in ("bid", "ask", "last", "volume"):
        value = getattr(tick, field, None)
        if value is not None:
            out[field] = value
    return out


def _check_symbol_market_status(
    symbol: str,
    *,
    detail: str,
    gateway: Any = None,
) -> Dict[str, Any]:
    symbol_name = str(symbol or "").strip().upper()
    if not symbol_name:
        return {"error": "symbol cannot be empty."}

    mt5_gateway = gateway if gateway is not None else get_mt5_gateway(
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    try:
        mt5_gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}

    info = mt5_gateway.symbol_info(symbol_name)
    if info is None:
        return {"error": f"Symbol {symbol_name} not found"}

    now_utc = datetime.now(timezone.utc)
    trade_mode = getattr(info, "trade_mode", None)
    mode_status = _symbol_trade_mode_status(mt5_gateway, trade_mode)
    tick = mt5_gateway.symbol_info_tick(symbol_name)
    tick_status = _symbol_tick_snapshot(tick, now_utc=now_utc)

    trade_mode_can_open = mode_status["can_open_new_positions"]
    can_open = trade_mode_can_open
    tick_freshness = tick_status.get("tick_freshness")
    reason = None
    if can_open is True and now_utc.weekday() >= 5:
        open_state = "weekend_closed"
        can_open = False
        reason = "weekend"
    elif can_open is True and tick_freshness == "fresh":
        open_state = "probably_open"
    elif can_open is True:
        open_state = "trade_mode_allows_opening"
    elif can_open is False:
        open_state = mode_status["status"]
    else:
        open_state = "unknown"

    if reason == "weekend":
        message = (
            f"{symbol_name}: closed for UTC weekend even though MT5 trade_mode "
            "allows opening."
        )
    else:
        message = (
            f"{symbol_name}: {open_state.replace('_', ' ')} "
            "(heuristic from MT5 trade_mode and tick freshness)."
        )

    result: Dict[str, Any] = {
        "success": True,
        "mode": "symbol",
        "symbol": symbol_name,
        "status": open_state,
        "status_source": "trade_mode_and_tick_freshness",
        "status_confidence": "heuristic",
        "can_open_new_positions": can_open,
        "trade_mode_allows_opening": trade_mode_can_open,
        "trade_mode_label": mode_status.get("trade_mode_label"),
        "tick_freshness": tick_freshness,
        "message": message,
        "timestamp": now_utc.isoformat(),
    }
    if reason:
        result["reason"] = reason
    if detail == "full":
        result["trade_mode"] = trade_mode
        result["symbol_info"] = {
            key: getattr(info, key, None)
            for key in (
                "name",
                "description",
                "visible",
                "select",
                "session_deals",
                "session_buy_orders",
                "session_sell_orders",
                "start_time",
                "expiration_time",
            )
            if getattr(info, key, None) is not None
        }
        result["tick"] = tick_status
    else:
        for key in ("tick_available", "last_tick_time", "last_tick_age_seconds"):
            if key in tick_status:
                result[key] = tick_status[key]
    return result


@mcp.tool()
def market_status(
    symbol: Optional[str] = None,
    region: Optional[Literal["us", "europe", "asia", "all"]] = "all",
    timezone_display: Optional[Literal["local", "utc", "auto"]] = "local",
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Get trading status of major stock markets worldwide.

    Returns the current status (open/closed/pre-market/lunch break) for major
    markets including NYSE, NASDAQ, LSE, Xetra, Euronext, Tokyo, Hong Kong,
    Shanghai, and ASX. Handles weekends and holidays correctly.

    Parameters
    ----------
    symbol : str, optional
        Broker symbol to check via MT5 trade mode and tick freshness. When
        supplied, returns a heuristic symbol status instead of the exchange
        overview.
    region : str, optional
        Filter by region: "us", "europe", "asia", or "all" (default: "all")
    timezone_display : str, optional
        Time display format: "local" (market's local time), "utc", or "auto" (default: "local")
    detail : {"compact", "full"}, optional
        Response detail level. `compact` (default) summarizes upcoming holiday
        information, while `full` preserves the complete holiday list.

    Returns
    -------
    dict
        Response containing:
        - `timestamp`: Current UTC timestamp
        - `day_of_week`: Current day name (e.g., "Tuesday")
        - `summary`: Human-readable summary of market statuses (e.g., "1 market open: NYSE; 3 pre-market: LSE, XETRA, EURONEXT; 5 closed")
        - `markets_open`: Count of markets currently open
        - `markets_pre_market`: Count of markets in pre-market
        - `markets_lunch_break`: Count of markets in lunch break
        - `markets_closed`: Count of markets currently closed
        - `upcoming_holidays_summary`: Compact holiday summary rows when
          `detail='compact'`
        - `upcoming_holidays`: Full holiday rows when `detail='full'`
            - `date`: Holiday date (ISO format)
            - `holiday`: Holiday name
            - `markets_affected`: List of market codes that will be closed
            - `impact`: "closed" or "early_close"
            - `early_close_time`: If early close, the close time (HH:MM)
            - `days_away`: Days from now
        - `markets`: List of market status objects with:
            - `symbol`: Market code (e.g., "NYSE")
            - `name`: Full market name
            - `status`: "open", "closed", "pre_market", "lunch_break"
            - `reason`: Reason if closed ("weekend", "holiday", "after_hours")
            - `local_time`: Current time in market's timezone (HH:MM)
            - `message`: Human-readable status (e.g., "NYSE: Open (closing in 4h 30mins)")
            - `next_open` / `next_close`: ISO timestamp of next event
            - `minutes_until`: Minutes until next status change
    """

    detail_mode = resolve_output_detail(detail=detail)
    timezone_display_mode = _normalize_timezone_display(timezone_display)
    if timezone_display_mode is None:
        return {"error": "Invalid timezone_display. Use 'local', 'utc', or 'auto'."}

    def _run() -> Dict[str, Any]:
        if symbol not in (None, ""):
            return _check_symbol_market_status(str(symbol), detail=detail_mode)

        # Map regions to markets
        region_map = {
            "us": ["NYSE", "NASDAQ"],
            "europe": ["LSE", "XETRA", "EURONEXT"],
            "asia": ["TSE", "HKEX", "SSE", "ASX"],
        }
        
        if region == "all" or region is None:
            markets_to_check = list(_MARKETS.keys())
        else:
            markets_to_check = region_map.get(region, list(_MARKETS.keys()))
        
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        
        now_utc = datetime.now(timezone.utc)

        for market_id in markets_to_check:
            if market_id not in _MARKETS:
                continue
            
            market = _MARKETS[market_id]
            try:
                local_now = _get_local_time(market["timezone"])
                status = _check_market_status(market_id, local_now)
                status = _apply_global_weekend_reason(status, now_utc=now_utc)
                status = _apply_market_timezone_display(
                    status,
                    now_local=local_now,
                    display=timezone_display_mode,
                )
                results.append(status)
            except Exception as exc:
                logger.warning(f"Failed to check status for {market_id}: {exc}")
                errors.append({
                    "symbol": market_id,
                    "error": str(exc),
                })
        
        # Sort results: open first, then by region
        def _sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
            status_priority = {"open": 0, "lunch_break": 1, "pre_market": 2, "closed": 3}
            return (status_priority.get(item["status"], 4), item["symbol"])
        
        results.sort(key=_sort_key)
        
        # Build summary messages with status breakdown
        status_counts = {
            "open": sum(1 for m in results if m["status"] == "open"),
            "pre_market": sum(1 for m in results if m["status"] == "pre_market"),
            "lunch_break": sum(1 for m in results if m["status"] == "lunch_break"),
            "closed": sum(1 for m in results if m["status"] == "closed"),
        }
        
        summary_messages = []
        
        # Add open markets (always list them if any)
        if status_counts["open"] > 0:
            open_markets = [m["symbol"] for m in results if m["status"] == "open"]
            summary_messages.append(f"{status_counts['open']} market{'s' if status_counts['open'] != 1 else ''} open: {', '.join(open_markets)}")
        
        # Add pre-market markets (always list if any)
        if status_counts["pre_market"] > 0:
            pre_markets = [m["symbol"] for m in results if m["status"] == "pre_market"]
            summary_messages.append(f"{status_counts['pre_market']} pre-market: {', '.join(pre_markets)}")
        
        # Add lunch break markets (always list if any)
        if status_counts["lunch_break"] > 0:
            lunch_markets = [m["symbol"] for m in results if m["status"] == "lunch_break"]
            summary_messages.append(f"{status_counts['lunch_break']} lunch break: {', '.join(lunch_markets)}")
        
        # Add closed markets (list if <= 3, otherwise just count)
        if status_counts["closed"] > 0:
            closed_markets = [m["symbol"] for m in results if m["status"] == "closed"]
            if status_counts["closed"] <= 3:
                summary_messages.append(f"{status_counts['closed']} closed: {', '.join(closed_markets)}")
            else:
                summary_messages.append(f"{status_counts['closed']} closed")
        
        reason_counts: Dict[str, int] = {}
        for market in results:
            if market.get("status") == "closed" and market.get("reason"):
                reason = str(market.get("reason"))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        global_status = None
        if results and status_counts["closed"] == len(results) and reason_counts.get("weekend") == len(results):
            global_status = "weekend"

        # Get upcoming holidays impacting these markets
        upcoming_holidays = _get_upcoming_holidays(markets_to_check)

        payload = {
            "success": True,
            "timestamp": now_utc.isoformat(),
            "day_of_week": now_utc.strftime("%A"),
            "region": region or "all",
            "summary": "; ".join(summary_messages) if summary_messages else "No market data available",
            "markets_open": status_counts["open"],
            "markets_closed": status_counts["closed"],
            "markets_pre_market": status_counts["pre_market"],
            "markets_lunch_break": status_counts["lunch_break"],
            "markets": results,
            "upcoming_holidays": upcoming_holidays if upcoming_holidays else None,
            "errors": errors if errors else None,
        }
        if reason_counts:
            payload["closed_reason_counts"] = reason_counts
        if global_status:
            payload["global_status"] = global_status
        return normalize_market_status_output(payload, detail=detail_mode)

    return run_logged_operation(
        logger,
        operation="market_status",
            symbol=symbol,
            region=region,
            timezone_display=timezone_display_mode,
            detail=detail_mode,
            func=_run,
        )
