"""Trading account and history views."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Literal, Optional

from ...utils.mt5 import (
    MT5ConnectionError,
    ensure_mt5_connection_or_raise,
    mt5_adapter,
)
from ...utils.mt5_enums import decode_mt5_enum_label
from ...utils.utils import (
    _format_time_minimal,
    _format_time_minimal_local,
    _normalize_limit,
    _parse_start_datetime,
    _use_client_tz,
)
from .._mcp_instance import mcp
from ..config import mt5_config
from ..execution_logging import run_logged_operation
from ..output_contract import (
    ensure_common_meta,
    normalize_output_detail,
)
from . import comments, validation
from .gateway import create_trading_gateway
from .positions import normalize_trade_history_output
from .requests import TradeHistoryRequest, TradeJournalAnalyzeRequest
from .use_cases import run_trade_history

logger = logging.getLogger(__name__)

_TRADE_ACCOUNT_SUMMARY_KEYS = (
    "success",
    "balance",
    "equity",
    "margin_level",
    "margin_level_note",
    "currency",
)
_TRADE_ACCOUNT_COMPACT_KEYS = _TRADE_ACCOUNT_SUMMARY_KEYS + (
    "profit",
    "margin",
    "margin_free",
    "leverage",
)
_TRADE_ACCOUNT_BASIC_KEYS = _TRADE_ACCOUNT_COMPACT_KEYS + (
    "trade_allowed",
    "trade_expert",
    "server",
    "company",
    "trade_mode",
)


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


def _is_exit_deal_row(row: Dict[str, Any]) -> bool:
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
    return float(total) if seen else None


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
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else None
    win_rate = float(wins / count) if count else None
    return {
        "closed_deals": count,
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "expectancy": avg_pnl,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": max(pnls) if pnls else None,
        "worst_trade": min(pnls) if pnls else None,
    }


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


def _trade_journal_trade_snapshot(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ticket": row.get("ticket"),
        "symbol": row.get("symbol"),
        "time": row.get("time"),
        "side": row.get("side"),
        "exit_trigger": row.get("exit_trigger"),
        "net_pnl": row.get("net_pnl"),
        "profit": row.get("profit"),
        "commission": row.get("commission"),
        "swap": row.get("swap"),
        "fee": row.get("fee"),
        "volume": row.get("volume"),
    }


def _run_trade_journal_request(request: TradeJournalAnalyzeRequest) -> Dict[str, Any]:
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
        return {
            "success": True,
            "summary": _trade_journal_metrics([]),
            "breakdowns": {
                "by_symbol": [],
                "by_side": [],
                "by_exit_trigger": [],
            },
            "message": message,
            "meta": {
                "history_rows": 0,
                "exit_deals": 0,
                "breakdown_limit": int(max(1, int(request.breakdown_limit))),
            },
        }

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
        enriched["side"] = str(row.get("type") or "").strip() or "Unknown"
        enriched["exit_trigger"] = (
            str(row.get("exit_trigger") or "").strip() or "Unspecified"
        )
        enriched["net_pnl"] = net_pnl
        analyzed_rows.append(enriched)

    breakdown_limit = int(max(1, int(request.breakdown_limit)))
    if not analyzed_rows:
        return {
            "success": True,
            "summary": _trade_journal_metrics([]),
            "breakdowns": {
                "by_symbol": [],
                "by_side": [],
                "by_exit_trigger": [],
            },
            "message": "No realized exit deals found in the requested trade history.",
            "meta": {
                "history_rows": int(len(rows)),
                "exit_deals": 0,
                "breakdown_limit": breakdown_limit,
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
    return {
        "success": True,
        "summary": _trade_journal_metrics(analyzed_rows),
        "breakdowns": {
            "by_symbol": _build_trade_journal_breakdown(
                analyzed_rows,
                key_name="symbol",
                label_name="symbol",
                limit=breakdown_limit,
            ),
            "by_side": _build_trade_journal_breakdown(
                analyzed_rows,
                key_name="side",
                label_name="side",
                limit=breakdown_limit,
            ),
            "by_exit_trigger": _build_trade_journal_breakdown(
                analyzed_rows,
                key_name="exit_trigger",
                label_name="exit_trigger",
                limit=breakdown_limit,
            ),
        },
        "best_trades": [
            _trade_journal_trade_snapshot(row)
            for row in ranked_best[: min(5, len(ranked_best))]
        ],
        "worst_trades": [
            _trade_journal_trade_snapshot(row)
            for row in ranked_worst[: min(5, len(ranked_worst))]
        ],
        "meta": {
            "history_rows": int(len(rows)),
            "exit_deals": int(len(analyzed_rows)),
            "breakdown_limit": breakdown_limit,
        },
    }


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
    if mode == "summary":
        keys = _TRADE_ACCOUNT_SUMMARY_KEYS
    elif mode == "compact":
        keys = _TRADE_ACCOUNT_COMPACT_KEYS
    elif mode == "basic":
        keys = _TRADE_ACCOUNT_BASIC_KEYS
    else:
        return dict(payload)
    return {key: payload.get(key) for key in keys if key in payload}


@mcp.tool()
def trade_account_info(
    detail: Literal["summary", "compact", "basic", "full"] = "compact",  # type: ignore
) -> dict:
    """Get account information with summary, compact, basic, or full account output modes.

    Use `detail="summary"` for the smallest balance/equity snapshot,
    `detail="compact"` (default) for routine balance and margin checks,
    `detail="basic"` for account identity/configuration fields, and
    `detail="full"` for the existing execution-readiness diagnostics.
    """

    def _run() -> dict:
        requested_mode = normalize_output_detail(
            detail,
            default="full",
            aliases={"summary_only": "summary"},
        )
        if requested_mode not in {"summary", "compact", "basic", "full"}:
            return {
                "error": "Invalid detail level. Use 'summary', 'compact', 'basic', or 'full'."
            }

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
        preflight = mt5.build_trade_preflight(account_info=info)
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

        payload = {
            "success": True,
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": margin_level,
            "currency": info.currency,
            "leverage": info.leverage,
            "trade_allowed": info.trade_allowed,
            "trade_expert": info.trade_expert,
            "server": preflight.get("server"),
            "company": preflight.get("company"),
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

    Use `detail="compact"` (default) to suppress echoed request filters while
    keeping the standard trade-history envelope. Use `detail="full"` to retain
    the request echo fields.
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
