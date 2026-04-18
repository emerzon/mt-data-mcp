"""Market status and trading hours MCP tool."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

import holidays

from ._mcp_instance import mcp
from .execution_logging import run_logged_operation

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


@mcp.tool()
def market_status(
    region: Optional[Literal["us", "europe", "asia", "all"]] = "all",
    timezone_display: Optional[Literal["local", "utc", "auto"]] = "local",
) -> Dict[str, Any]:
    """Get trading status of major stock markets worldwide.

    Returns the current status (open/closed/pre-market/lunch break) for major
    markets including NYSE, NASDAQ, LSE, Xetra, Euronext, Tokyo, Hong Kong,
    Shanghai, and ASX. Handles weekends and holidays correctly.

    Parameters
    ----------
    region : str, optional
        Filter by region: "us", "europe", "asia", or "all" (default: "all")
    timezone_display : str, optional
        Time display format: "local" (market's local time), "utc", or "auto" (default: "local")

    Returns
    -------
    dict
        Response containing:
        - `timestamp`: Current UTC timestamp
        - `day_of_week`: Current day name (e.g., "Tuesday")
        - `upcoming_holidays`: List of holidays in next 14 days impacting markets:
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

    def _run() -> Dict[str, Any]:
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
        
        for market_id in markets_to_check:
            if market_id not in _MARKETS:
                continue
            
            market = _MARKETS[market_id]
            try:
                local_now = _get_local_time(market["timezone"])
                status = _check_market_status(market_id, local_now)
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
        
        # Build summary messages
        open_count = sum(1 for m in results if m["status"] == "open")
        closed_count = len(results) - open_count
        
        summary_messages = []
        if open_count > 0:
            open_markets = [m["symbol"] for m in results if m["status"] == "open"]
            summary_messages.append(f"{open_count} market{'s' if open_count != 1 else ''} open: {', '.join(open_markets)}")
        if closed_count > 0:
            closed_markets = [m["symbol"] for m in results if m["status"] == "closed"]
            if closed_count <= 3:
                summary_messages.append(f"{closed_count} closed: {', '.join(closed_markets)}")
        
        # Get upcoming holidays impacting these markets
        upcoming_holidays = _get_upcoming_holidays(markets_to_check)
        
        now_utc = datetime.now(timezone.utc)
        
        return {
            "success": True,
            "timestamp": now_utc.isoformat(),
            "day_of_week": now_utc.strftime("%A"),
            "region": region or "all",
            "summary": "; ".join(summary_messages) if summary_messages else "No markets available",
            "markets_open": open_count,
            "markets_closed": closed_count,
            "markets": results,
            "upcoming_holidays": upcoming_holidays if upcoming_holidays else None,
            "errors": errors if errors else None,
        }

    return run_logged_operation(
        logger,
        operation="market_status",
        region=region,
        func=_run,
    )
