"""Trading account and history views."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from ...bootstrap.settings import mt5_config
from ...utils.coercion import round_finite
from ...utils.mt5 import (
    MT5ConnectionError,
    ensure_mt5_connection_or_raise,
    mt5_adapter,
)
from ...utils.mt5_enums import decode_mt5_enum_label
from ...utils.time import (
    _format_time_minimal,
    _format_time_minimal_local,
    _use_client_tz,
)
from ...utils.utils import (
    _normalize_limit,
    _parse_start_datetime,
)
from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from ..output_contract import (
    ensure_common_meta,
    resolve_output_contract,
)
from . import comments, validation
from .gateway import create_trading_gateway
from .positions import normalize_trade_history_output
from .requests import TradeHistoryRequest, TradeJournalAnalyzeRequest
from .use_cases import _DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS, run_trade_history

logger = logging.getLogger(__name__)

_TRADE_ACCOUNT_COMPACT_KEYS = (
    "success",
    "source",
    "retrieved_at",
    "timezone",
    "broker_server_tz",
    "server_time",
    "server_time_source",
    "clock_skew_seconds",
    "login",
    "server",
    "company",
    "account_type",
    "is_demo",
    "is_live",
    "trade_mode",
    "balance",
    "equity",
    "profit",
    "margin",
    "margin_free",
    "margin_level",
    "margin_level_note",
    "currency",
    "leverage",
    "trade_allowed",
    "trade_expert",
)
_TRADE_JOURNAL_UNITS: Dict[str, str] = {
    "win_rate": "fraction",
    "win_rate_pct": "percentage_points",
}


def _utc_epoch_identity(value: Any) -> float:
    return float(value)


def _run_trade_history_request(request: TradeHistoryRequest) -> Any:
    result = run_trade_history(
        request,
        gateway=create_trading_gateway(
            include_trade_preflight=True,
            adapter=mt5_adapter,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        ),
        use_client_tz=_use_client_tz,
        format_time_minimal=_format_time_minimal,
        format_time_minimal_local=_format_time_minimal_local,
        mt5_epoch_to_utc=_utc_epoch_identity,
        parse_start_datetime=_parse_start_datetime,
        normalize_limit=_normalize_limit,
        comment_row_metadata=comments._comment_row_metadata,
        normalize_ticket_filter=validation._normalize_ticket_filter,
        normalize_minutes_back=validation._normalize_minutes_back,
        decode_mt5_enum_label=decode_mt5_enum_label,
        mt5_config=mt5_config,
    )
    return normalize_trade_history_output(result, request=request)


def _safe_trade_journal_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def _round_trade_journal_value(value: Any, *, digits: int) -> Optional[float]:
    return round_finite(value, digits, on_invalid="none")


def _is_exit_deal_row(row: Dict[str, Any]) -> bool:
    deal_effect = str(row.get("deal_effect") or "").strip().lower()
    if deal_effect in {"close", "close_by", "reverse"}:
        return True
    position_action = str(row.get("position_action") or "").strip().lower()
    if position_action.startswith(("close_", "close_by_", "reverse_")):
        return True
    entry_text = str(row.get("entry") or "").strip().lower()
    if entry_text and "out" in entry_text:
        return True
    return bool(str(row.get("exit_trigger") or "").strip())


def _trade_journal_net_pnl(row: Dict[str, Any]) -> Optional[float]:
    seen = False
    total = 0.0
    for key in ("profit", "commission", "swap", "fee"):
        value = _safe_trade_journal_float(row.get(key))
        if value is None:
            continue
        total += value
        seen = True
    return _round_trade_journal_value(total, digits=2) if seen else None


def _trade_journal_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnls = [
        float(row["net_pnl"])
        for row in rows
        if _safe_trade_journal_float(row.get("net_pnl")) is not None
    ]
    count = int(len(pnls))
    wins = int(sum(1 for pnl in pnls if pnl > 0.0))
    losses = int(sum(1 for pnl in pnls if pnl < 0.0))
    flats = int(sum(1 for pnl in pnls if pnl == 0.0))
    gross_profit = float(sum(pnl for pnl in pnls if pnl > 0.0))
    gross_loss = float(abs(sum(pnl for pnl in pnls if pnl < 0.0)))
    net_pnl = float(sum(pnls))
    avg_pnl = float(net_pnl / count) if count else None
    avg_win = float(gross_profit / wins) if wins else None
    avg_loss = float(gross_loss / losses) if losses else None
    profit_factor_note = None
    if wins == 0 and losses > 0:
        profit_factor = None
        profit_factor_note = "Undefined because there were no winning trades."
    elif losses == 0 and wins > 0:
        profit_factor = None
        profit_factor_note = "Undefined because there were no losing trades."
    else:
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else None
    win_rate = float(wins / count) if count else None
    win_rate_pct = float(win_rate * 100.0) if win_rate is not None else None
    metrics = {
        "closed_deals": count,
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "win_rate": _round_trade_journal_value(win_rate, digits=4),
        "win_rate_pct": _round_trade_journal_value(win_rate_pct, digits=2),
        "net_pnl": _round_trade_journal_value(net_pnl, digits=2),
        "gross_profit": _round_trade_journal_value(gross_profit, digits=2),
        "gross_loss": _round_trade_journal_value(gross_loss, digits=2),
        "profit_factor": _round_trade_journal_value(profit_factor, digits=4),
        "expectancy": _round_trade_journal_value(avg_pnl, digits=4),
        "avg_win": _round_trade_journal_value(avg_win, digits=2),
        "avg_loss": _round_trade_journal_value(avg_loss, digits=2),
        "best_trade": _round_trade_journal_value(max(pnls), digits=2) if pnls else None,
        "worst_trade": _round_trade_journal_value(min(pnls), digits=2) if pnls else None,
    }
    if profit_factor_note:
        metrics["profit_factor_note"] = profit_factor_note
    if 0 < count < 30:
        metrics["sample_notice"] = {
            "code": "low_sample_unreliable_metrics",
            "message": (
                f"Only {count} closed deals; win rate, profit factor and expectancy "
                "are statistically unreliable below ~30 trades."
            ),
        }
    return metrics


def _attach_trade_journal_units(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["units"] = dict(_TRADE_JOURNAL_UNITS)
    return payload


def _build_trade_journal_breakdown(
    rows: List[Dict[str, Any]],
    *,
    key_name: str,
    label_name: str,
    limit: int,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get(key_name) or "").strip() or "Unspecified"
        groups.setdefault(label, []).append(row)

    output: List[Dict[str, Any]] = []
    for label, items in groups.items():
        metrics = _trade_journal_metrics(items)
        metrics[label_name] = label
        output.append(metrics)

    output.sort(
        key=lambda item: (
            -abs(float(item.get("net_pnl") or 0.0)),
            -int(item.get("closed_deals") or 0),
            str(item.get(label_name) or ""),
        )
    )
    return output[: max(1, int(limit))]


def _compact_trade_journal_breakdown(
    items: List[Dict[str, Any]],
    *,
    label_name: str,
) -> List[Dict[str, Any]]:
    compact_keys = ("closed_deals", "win_rate", "win_rate_pct", "net_pnl", "expectancy")
    output: List[Dict[str, Any]] = []
    for item in items:
        compact_item = {label_name: item.get(label_name)}
        compact_item.update({key: item.get(key) for key in compact_keys})
        output.append(compact_item)
    return output


def _trade_journal_breakdowns(
    rows: List[Dict[str, Any]],
    *,
    limit: int,
    detail: str,
) -> Dict[str, List[Dict[str, Any]]]:
    if detail == "compact":
        return {}

    by_symbol = _build_trade_journal_breakdown(
        rows,
        key_name="symbol",
        label_name="symbol",
        limit=limit,
    )
    if detail in {"standard", "summary"}:
        breakdowns = {
            "by_symbol": _compact_trade_journal_breakdown(
                by_symbol,
                label_name="symbol",
            )
        }
        if detail == "summary":
            by_side = _build_trade_journal_breakdown(
                rows,
                key_name="side",
                label_name="side",
                limit=limit,
            )
            breakdowns["by_side"] = _compact_trade_journal_breakdown(
                by_side,
                label_name="side",
            )
        return breakdowns

    return {
        "by_symbol": by_symbol,
        "by_side": _build_trade_journal_breakdown(
            rows,
            key_name="side",
            label_name="side",
            limit=limit,
        ),
        "by_exit_trigger": _build_trade_journal_breakdown(
            rows,
            key_name="exit_trigger",
            label_name="exit_trigger",
            limit=limit,
        ),
    }


def _trade_journal_trade_snapshot(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "deal_ticket": row.get("deal_ticket", row.get("ticket")),
        "order_ticket": row.get("order_ticket"),
        "position_ticket": row.get("position_ticket"),
        "symbol": row.get("symbol"),
        "fill_time": row.get("fill_time", row.get("time")),
        "side": row.get("side"),
        "exit_trigger": row.get("exit_trigger"),
        "net_pnl": row.get("net_pnl"),
        "profit": row.get("profit"),
        "commission": row.get("commission"),
        "swap": row.get("swap"),
        "fee": row.get("fee"),
        "volume": row.get("volume"),
    }


def _trade_journal_period_context(
    request: TradeJournalAnalyzeRequest,
) -> Dict[str, Any]:
    from .common import resolve_trade_period_context

    return resolve_trade_period_context(
        start=request.start,
        end=request.end,
        minutes_back=request.minutes_back,
        default_lookback_days=_DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS,
        include_timezone_alias=True,
        default_lookback_style="note",
    )


def _run_trade_journal_request(request: TradeJournalAnalyzeRequest) -> Dict[str, Any]:
    period_context = _trade_journal_period_context(request)
    detail_mode = str(request.detail or "compact").strip().lower()
    minimum_sample = int(max(1, int(request.min_sample)))
    if detail_mode == "compact":
        period_context = {"timezone": "UTC"}
    history_result = _run_trade_history_request(
        TradeHistoryRequest(
            history_kind="deals",
            start=request.start,
            end=request.end,
            symbol=request.symbol,
            side=request.side,
            position_ticket=request.position_ticket,
            deal_ticket=request.deal_ticket,
            minutes_back=request.minutes_back,
            limit=request.limit,
        )
    )
    if not isinstance(history_result, dict):
        return {"error": "Unexpected trade_history response shape."}
    if history_result.get("error"):
        return history_result

    raw_rows = history_result.get("items")
    if not isinstance(raw_rows, list):
        return {"error": "Unexpected trade_history response shape."}

    message = history_result.get("message")
    if isinstance(message, str) and message.strip() and not raw_rows:
        breakdown_limit = int(max(1, int(request.breakdown_limit)))
        sample_quality = _trade_journal_sample_quality(0, minimum=minimum_sample)
        if request.check_only:
            return {
                "success": True,
                **period_context,
                "check_only": True,
                "check_result": sample_quality,
                "sample_size": 0,
                "sample_quality": sample_quality,
                "message": message,
                "meta": {
                    "history_rows": 0,
                    "exit_deals": 0,
                    "min_sample": minimum_sample,
                },
            }
        sample_warning = _trade_journal_sample_warning(0, minimum=minimum_sample)
        breakdowns = _trade_journal_breakdowns(
            [],
            limit=breakdown_limit,
            detail=detail_mode,
        )
        payload = {
            "success": True,
            **period_context,
            "summary": _trade_journal_metrics([]),
            "sample_size": 0,
            "sample_quality": sample_quality,
            "message": message,
            "meta": {
                "history_rows": 0,
                "exit_deals": 0,
                "breakdown_limit": breakdown_limit,
                "min_sample": minimum_sample,
            },
            "sample_warning": sample_warning,
        }
        if breakdowns:
            payload["breakdowns"] = breakdowns
        return _attach_trade_journal_units(payload)

    rows = [row for row in raw_rows if isinstance(row, dict)]
    analyzed_rows: List[Dict[str, Any]] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").strip()
        if not symbol or not _is_exit_deal_row(row):
            continue
        net_pnl = _trade_journal_net_pnl(row)
        if net_pnl is None:
            continue
        enriched = dict(row)
        enriched["symbol"] = symbol
        enriched["side"] = (
            str(row.get("position_side") or row.get("type") or "").strip() or "Unknown"
        )
        enriched["exit_trigger"] = (
            str(row.get("exit_trigger") or "").strip() or "Unspecified"
        )
        for money_key in ("profit", "commission", "swap", "fee"):
            if money_key in enriched:
                rounded_value = _round_trade_journal_value(
                    enriched.get(money_key),
                    digits=2,
                )
                if rounded_value is not None:
                    enriched[money_key] = rounded_value
        enriched["net_pnl"] = net_pnl
        analyzed_rows.append(enriched)

    breakdown_limit = int(max(1, int(request.breakdown_limit)))
    if not analyzed_rows:
        sample_quality = _trade_journal_sample_quality(0, minimum=minimum_sample)
        if request.check_only:
            return {
                "success": True,
                **period_context,
                "check_only": True,
                "check_result": sample_quality,
                "sample_size": 0,
                "sample_quality": sample_quality,
                "message": "No realized exit deals found in the requested trade history.",
                "meta": {
                    "history_rows": int(len(rows)),
                    "exit_deals": 0,
                    "min_sample": minimum_sample,
                },
            }
        sample_warning = _trade_journal_sample_warning(0, minimum=minimum_sample)
        breakdowns = _trade_journal_breakdowns(
            [],
            limit=breakdown_limit,
            detail=detail_mode,
        )
        payload = {
            "success": True,
            **period_context,
            "summary": _trade_journal_metrics([]),
            "sample_size": 0,
            "sample_quality": sample_quality,
            "message": "No realized exit deals found in the requested trade history.",
            "meta": {
                "history_rows": int(len(rows)),
                "exit_deals": 0,
                "breakdown_limit": breakdown_limit,
                "min_sample": minimum_sample,
            },
            **({"sample_warning": sample_warning} if sample_warning else {}),
        }
        if breakdowns:
            payload["breakdowns"] = breakdowns
        return _attach_trade_journal_units(payload)

    sample_quality = _trade_journal_sample_quality(len(analyzed_rows), minimum=minimum_sample)
    if request.check_only:
        return {
            "success": True,
            **period_context,
            "check_only": True,
            "check_result": sample_quality,
            "sample_size": int(len(analyzed_rows)),
            "sample_quality": sample_quality,
            "meta": {
                "history_rows": int(len(rows)),
                "exit_deals": int(len(analyzed_rows)),
                "min_sample": minimum_sample,
            },
        }

    # Filter trades by P&L sign: wins have positive P&L, losses have negative P&L
    wins = [row for row in analyzed_rows if float(row.get("net_pnl") or 0.0) > 0.0]
    losses = [row for row in analyzed_rows if float(row.get("net_pnl") or 0.0) < 0.0]
    
    # Sort wins descending (best wins first), losses ascending (worst losses first)
    ranked_best = sorted(
        wins,
        key=lambda row: float(row.get("net_pnl") or 0.0),
        reverse=True,
    )
    ranked_worst = sorted(
        losses,
        key=lambda row: float(row.get("net_pnl") or 0.0),
    )
    breakdowns = _trade_journal_breakdowns(
        analyzed_rows,
        limit=breakdown_limit,
        detail=detail_mode,
    )
    payload = {
        "success": True,
        **period_context,
        "summary": _trade_journal_metrics(analyzed_rows),
        "sample_size": int(len(analyzed_rows)),
        "sample_quality": sample_quality,
        "meta": {
            "history_rows": int(len(rows)),
            "exit_deals": int(len(analyzed_rows)),
            "breakdown_limit": breakdown_limit,
            "min_sample": minimum_sample,
        },
    }
    if breakdowns:
        payload["breakdowns"] = breakdowns
    sample_warning = _trade_journal_sample_warning(len(analyzed_rows), minimum=minimum_sample)
    if sample_warning:
        payload["sample_warning"] = sample_warning
    if detail_mode == "full":
        payload["items"] = [
            _trade_journal_trade_snapshot(row) for row in analyzed_rows
        ]
        payload["item_schema"] = "trade_journal_analyzed_exit.v2"
        payload["best_trades"] = [
            _trade_journal_trade_snapshot(row)
            for row in ranked_best[: min(5, len(ranked_best))]
        ]
        payload["worst_trades"] = [
            _trade_journal_trade_snapshot(row)
            for row in ranked_worst[: min(5, len(ranked_worst))]
        ]
    return _attach_trade_journal_units(payload)


def _trade_journal_sample_quality(exit_deals: int, *, minimum: int = 30) -> Dict[str, Any]:
    count = int(max(0, int(exit_deals)))
    recommended = int(max(1, int(minimum)))
    if count <= 0:
        status = "empty"
        confidence = "none"
    elif count < recommended:
        status = "insufficient"
        confidence = "low"
    elif count < 100:
        status = "usable"
        confidence = "basic"
    else:
        status = "robust"
        confidence = "higher"
    quality: Dict[str, Any] = {
        "status": status,
        "sample_size": count,
        "minimum_recommended": recommended,
        "robust_sample": 100,
        "confidence": confidence,
    }
    if count < recommended:
        quality["suggestions"] = [
            "Increase limit to fetch more raw deals.",
            "Increase minutes_back or provide a wider start/end range.",
            "Remove symbol or side filters if you want account-level statistics.",
        ]
    return quality


def _trade_journal_sample_warning(exit_deals: int, *, minimum: int = 30) -> Optional[str]:
    recommended = int(max(1, int(minimum)))
    if int(exit_deals) >= recommended:
        return None
    return (
        f"Only {int(exit_deals)} realized exit deal(s) were analyzed; "
        f"{recommended}+ is recommended for basic journal statistics. Increase limit to fetch more raw deals, "
        "increase minutes_back, or provide a wider start/end range."
    )


def lookup_trade_ticket_history(ticket: Any) -> Optional[Dict[str, Any]]:
    ticket_text = str(ticket).strip()
    if not ticket_text:
        return None

    lookback_minutes = 60 * 24 * 7

    def _latest_row(result: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(result, dict):
            return None
        items = result.get("items")
        if not isinstance(items, list) or not items:
            return None
        rows = [row for row in items if isinstance(row, dict)]
        return rows[-1] if rows else None

    def _time_label(row: Dict[str, Any], *keys: str) -> Optional[str]:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    deals = _run_trade_history_request(
        TradeHistoryRequest(
            history_kind="deals",
            position_ticket=ticket,
            minutes_back=lookback_minutes,
            limit=20,
        )
    )
    deal_row = _latest_row(deals)
    if deal_row is not None:
        symbol = str(deal_row.get("symbol") or "").strip()
        type_label = str(deal_row.get("type") or "position").strip()
        time_label = _time_label(deal_row, "time", "time_done", "time_setup")
        reason_label = str(deal_row.get("reason") or "").strip()
        message = f"Ticket {ticket_text} was a {type_label} position that has already been closed"
        if symbol:
            message += f" on {symbol}"
        if time_label:
            message += f" at {time_label}"
        if reason_label:
            message += f" ({reason_label})"
        message += ". No action taken."
        return {
            "message": message,
            "no_action": True,
            "checked_scopes": ["positions", "pending_orders", "history_deals"],
        }

    orders = _run_trade_history_request(
        TradeHistoryRequest(
            history_kind="orders",
            order_ticket=ticket,
            minutes_back=lookback_minutes,
            limit=20,
        )
    )
    order_row = _latest_row(orders)
    if order_row is not None:
        symbol = str(order_row.get("symbol") or "").strip()
        type_label = str(order_row.get("type") or "order").strip()
        state_label = str(order_row.get("state") or "completed").strip()
        time_label = _time_label(order_row, "time_done", "time_setup", "time")
        message = (
            f"Ticket {ticket_text} was a {type_label} order that was {state_label}"
        )
        if symbol:
            message += f" on {symbol}"
        if time_label:
            message += f" at {time_label}"
        message += ". No action taken."
        return {
            "message": message,
            "no_action": True,
            "checked_scopes": ["positions", "pending_orders", "history_orders"],
        }

    return None


def _trade_account_payload_for_mode(payload: Dict[str, Any], *, mode: str) -> Dict[str, Any]:
    if mode == "compact":
        keys = _TRADE_ACCOUNT_COMPACT_KEYS
    else:
        out = dict(payload)
        out["additional_fields"] = [
            key for key in payload if key not in _TRADE_ACCOUNT_COMPACT_KEYS
        ]
        return out
    return {key: payload.get(key) for key in keys if key in payload}


def _trade_account_iso_from_epoch(value: Any, *, milliseconds: bool = False) -> Optional[str]:
    try:
        epoch = float(value)
    except Exception:
        return None
    if not math.isfinite(epoch) or epoch <= 0:
        return None
    if milliseconds or epoch > 10_000_000_000:
        epoch /= 1000.0
    try:
        return (
            datetime.fromtimestamp(epoch, tz=timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except Exception:
        return None


def _trade_account_clock_fields(
    terminal_info: Any,
    *,
    retrieved_at_epoch: float,
) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    broker_tz = str(getattr(mt5_config, "server_tz_name", "") or "").strip()
    if broker_tz:
        fields["broker_server_tz"] = broker_tz

    for attr in ("server_time_msc", "time_msc", "server_time", "time"):
        raw_value = getattr(terminal_info, attr, None) if terminal_info is not None else None
        server_time = _trade_account_iso_from_epoch(
            raw_value,
            milliseconds=attr.endswith("_msc"),
        )
        if not server_time:
            continue
        fields["server_time"] = server_time
        fields["server_time_source"] = f"mt5_terminal_info.{attr}"
        try:
            raw_epoch = float(raw_value)
            if attr.endswith("_msc") or raw_epoch > 10_000_000_000:
                raw_epoch /= 1000.0
            fields["clock_skew_seconds"] = round(
                abs(float(retrieved_at_epoch) - raw_epoch),
                3,
            )
        except Exception:
            pass
        break
    return fields


def _trade_account_equity_balance_delta(info: Any) -> Optional[float]:
    try:
        balance = float(info.balance)
        equity = float(info.equity)
    except Exception:
        return None
    if not math.isfinite(balance) or not math.isfinite(equity):
        return None
    return round(equity - balance, 2)


@mcp.tool()
def trade_account_info(
    detail: Literal["compact", "full"] = "compact",
) -> dict:
    """Get account information with compact or full account output modes.

    Use `detail="compact"` (default) for routine balance and margin checks.
    Use `detail="full"` for broker/account identifiers and execution diagnostics.
    """

    def _run() -> dict:
        try:
            contract = resolve_output_contract(detail=detail, default_detail="compact")
        except ValueError:
            return {"error": "Invalid detail level. Use 'compact' or 'full'."}
        if contract.detail not in {"compact", "full"}:
            return {"error": "Invalid detail level. Use 'compact' or 'full'."}
        requested_mode = contract.detail

        mt5 = create_trading_gateway(
            include_trade_preflight=True,
            adapter=mt5_adapter,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )

        try:
            mt5.ensure_connection()
        except MT5ConnectionError as exc:
            return {"error": str(exc)}

        info = mt5.account_info()
        if info is None:
            return {"error": "Failed to get account info"}
        try:
            terminal_info = mt5.terminal_info()
        except Exception:
            terminal_info = None
        preflight = mt5.build_trade_preflight(
            account_info=info,
            terminal_info=terminal_info,
        )
        login = preflight.get("login")
        if login is None:
            login = getattr(info, "login", None)
        margin_level: Optional[float] = getattr(info, "margin_level", None)
        margin_level_note: Optional[str] = None
        try:
            margin_val = validation._safe_float_attr(info, "margin")
            ml_val = validation._safe_float_attr(info, "margin_level")
            if margin_val <= 0.0 and ml_val <= 0.0:
                margin_level = None
                margin_level_note = "N/A (no open margin/positions)"
            elif not math.isfinite(ml_val):
                margin_level = None
            else:
                margin_level = round(float(ml_val), 2)
        except Exception:
            pass

        retrieved_dt = datetime.now(timezone.utc).replace(microsecond=0)
        retrieved_at = (
            retrieved_dt
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        payload = {
            "success": True,
            "source": "mt5_account_snapshot",
            "retrieved_at": retrieved_at,
            "timezone": "UTC",
            **_trade_account_clock_fields(
                terminal_info,
                retrieved_at_epoch=retrieved_dt.timestamp(),
            ),
            "login": login,
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "floating_pnl": info.profit,
            "pnl_basis": "floating_open_positions",
            "equity_balance_delta": _trade_account_equity_balance_delta(info),
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": margin_level,
            "currency": info.currency,
            "leverage": info.leverage,
            "trade_allowed": info.trade_allowed,
            "trade_expert": info.trade_expert,
            "server": preflight.get("server"),
            "company": preflight.get("company"),
            "account_type": preflight.get("account_type"),
            "is_demo": preflight.get("is_demo"),
            "is_live": preflight.get("is_live"),
            "trade_mode": preflight.get("trade_mode"),
            "terminal_trade_allowed": preflight.get("terminal_trade_allowed"),
            "terminal_tradeapi_disabled": preflight.get("terminal_tradeapi_disabled"),
            "terminal_connected": preflight.get("terminal_connected"),
            "auto_trading_enabled": preflight.get("auto_trading_enabled"),
            "execution_ready": preflight.get("execution_ready"),
            "execution_ready_strict": preflight.get("execution_ready_strict"),
            "execution_hard_blockers": preflight.get("execution_hard_blockers"),
            "execution_soft_blockers": preflight.get("execution_soft_blockers"),
            "execution_blockers": preflight.get("execution_blockers"),
        }
        if login is None:
            payload.pop("login", None)
        if margin_level_note:
            payload["margin_level_note"] = margin_level_note
        payload = _trade_account_payload_for_mode(payload, mode=requested_mode)
        return ensure_common_meta(
            payload,
            tool_name="trade_account_info",
            mt5_config=mt5_config,
        )

    return run_logged_operation(
        logger,
        operation="trade_account_info",
        detail=detail,
        func=_run,
    )


@mcp.tool()
def trade_history(request: TradeHistoryRequest) -> Dict[str, Any]:
    """Get deal or order history as tabular data.

    `history_kind="deals"` returns executed fills with P&L and is the default
    for performance analysis. `history_kind="orders"` returns order lifecycle
    events for audit and reconciliation.
    Rows expose three distinct identifiers: `deal_ticket` (the unique executed
    fill), `order_ticket` (the order that produced the fill), and
    `position_ticket` (the position opened/closed; matches `trade_get_open.ticket`).
    Deals use `fill_time` and `fill_side`; orders use `placed_time` and
    `done_time` at every detail level. Full rows add native MT5 attributes
    under `raw` without renaming canonical fields. Every response includes the
    effective period context. Use `detail="full"` for request echo fields.
    """
    return run_logged_operation(
        logger,
        operation="trade_history",
        history_kind=request.history_kind,
        detail=request.detail,
        symbol=request.symbol,
        limit=request.limit,
        func=lambda: _run_trade_history_request(request),
    )


@mcp.tool()
def trade_journal_analyze(request: TradeJournalAnalyzeRequest) -> Dict[str, Any]:
    """Analyze realized exit deals from MT5 trade history."""
    return run_logged_operation(
        logger,
        operation="trade_journal_analyze",
        symbol=request.symbol,
        limit=request.limit,
        func=lambda: _run_trade_journal_request(request),
    )
