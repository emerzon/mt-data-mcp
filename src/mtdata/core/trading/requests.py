from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, Field, field_validator

from ...shared.schema import CompactFullDetailLiteral, TimeframeLiteral
from ...utils.barriers import normalize_trade_direction
from . import validation
from .time import ExpirationValue
from .validation import OrderTypeInput

MAGIC_NUMBER_DESCRIPTION = (
    "MT5 magic number: integer strategy/order identifier used to group EA or "
    "strategy trades. Use as a filter for one strategy; omit for all magic numbers."
)


def _normalize_trade_side_alias(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized, error = validation._normalize_trade_side_filter(value)
    if error is None and normalized is not None:
        return normalized
    return value


def _normalize_trade_direction_alias(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized, error = normalize_trade_direction(value)
    if error is None and normalized is not None:
        return normalized
    return value


class TradePlaceRequest(BaseModel):
    model_config = {"populate_by_name": True}

    symbol: Optional[str] = None
    volume: Optional[float] = Field(
        default=None,
        description="Order size in lots (e.g. 0.01), not traded/tick volume.",
    )
    order_type: Optional[OrderTypeInput] = Field(
        default=None,
        description=(
            "Order type: BUY/SELL for market orders, or "
            "BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP for pending orders."
        ),
    )
    price: Optional[Union[int, float]] = None
    stop_loss: Optional[Union[int, float]] = Field(
        default=None,
        validation_alias="sl",
    )
    take_profit: Optional[Union[int, float]] = Field(
        default=None,
        validation_alias="tp",
    )
    expiration: Optional[ExpirationValue] = None
    comment: Optional[str] = None
    magic: Optional[int] = Field(
        default=None,
        description=(
            "MT5 magic number: integer strategy/order identifier used to group EA or "
            "strategy trades. Defaults to configured order_magic when omitted."
        ),
    )
    deviation: int = Field(
        default=20,
        description="Maximum allowed execution slippage in points.",
    )
    dry_run: bool = Field(
        default=True,
        description=(
            "Preview the order without sending it to the broker. Defaults to true "
            "for safety; set dry_run=false only when intentionally placing a live "
            "order."
        ),
    )
    detail: CompactFullDetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Compact returns the lean dry-run preview; "
            "standard and summary add local validation context; full keeps all "
            "preview diagnostics."
        ),
    )
    require_sl_tp: bool = Field(
        default=True,
        description=(
            "Require both stop_loss and take_profit for market orders and fail "
            "if protection cannot be attached."
        ),
    )
    auto_close_on_sl_tp_fail: bool = Field(
        default=True,
        description=(
            "If a filled market order cannot attach TP/SL, immediately try to "
            "close the unprotected position."
        ),
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional in-process dedupe key with an in-memory ~5-minute TTL. "
            "Reusing the same key with the same payload replays the prior "
            "result instead of sending another order. It is not broker-side "
            "idempotency and does not survive restarts."
        ),
    )


class TradeModifyRequest(BaseModel):
    model_config = {"populate_by_name": True}

    ticket: Union[int, str]
    detail: CompactFullDetailLiteral = Field(
        default="compact",
        description="Response detail level for modify previews and result payloads.",
    )
    price: Optional[Union[int, float]] = None
    stop_loss: Optional[Union[int, float]] = Field(
        default=None,
        validation_alias="sl",
    )
    take_profit: Optional[Union[int, float]] = Field(
        default=None,
        validation_alias="tp",
    )
    expiration: Optional[ExpirationValue] = None
    comment: Optional[str] = None
    dry_run: bool = Field(
        default=False,
        description=(
            "Preview the modification without sending it to the broker. Defaults "
            "to false, so trade_modify changes the live order or position; set "
            "dry_run=true to preview only."
        ),
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional in-process dedupe key with an in-memory ~5-minute TTL. "
            "Reusing the same key with the same payload replays the prior "
            "result instead of sending another modify request. It is not "
            "broker-side idempotency and does not survive restarts."
        ),
    )


class TradeCloseRequest(BaseModel):
    ticket: Optional[Union[int, str]] = None
    detail: CompactFullDetailLiteral = Field(
        default="compact",
        description="Response detail level for close previews and result payloads.",
    )
    close_all: bool = Field(
        default=False,
        description="Close all matching open positions instead of a single ticket.",
    )
    symbol: Optional[str] = None
    magic: Optional[int] = Field(default=None, description=MAGIC_NUMBER_DESCRIPTION)
    volume: Optional[float] = Field(
        default=None,
        description="Partial close volume in lots. Requires ticket.",
    )
    dry_run: bool = Field(
        default=False,
        description=(
            "Preview the close request without sending it to the broker. Defaults "
            "to false, so trade_close closes the live position or order; set "
            "dry_run=true to preview only."
        ),
    )
    confirm_close_all: bool = Field(
        default=False,
        description=(
            "Required with close_all=true and dry_run=false to execute a live "
            "bulk close."
        ),
    )
    profit_only: bool = Field(
        default=False,
        description="Only close positions that are currently profitable.",
    )
    loss_only: bool = Field(
        default=False,
        description="Only close positions that are currently losing.",
    )
    close_priority: Optional[
        Literal["loss_first", "profit_first", "largest_first"]
    ] = Field(
        default=None,
        description=(
            "When multiple positions match, choose close order by loss_first, "
            "profit_first, or largest_first."
        ),
    )
    comment: Optional[str] = None
    deviation: int = 20


class TradeHistoryRequest(BaseModel):
    history_kind: Literal["deals", "orders"] = Field(
        default="deals",
        description=(
            "Trade history type. deals = executed fills with P&L for journals; "
            "orders = order lifecycle events for audit/reconciliation."
        ),
    )
    detail: CompactFullDetailLiteral = "compact"
    column_style: Literal["snake_case", "humanized"] = Field(
        default="snake_case",
        description=(
            "Primary history item key style. Defaults to snake_case to preserve "
            "raw MT5-style history keys; use humanized for display labels."
        ),
    )
    start: Optional[str] = None
    end: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = Field(
        default=None,
        description="Optional side filter. Accepts buy/sell or long/short.",
    )
    position_ticket: Optional[Union[int, str]] = None
    deal_ticket: Optional[Union[int, str]] = None
    order_ticket: Optional[Union[int, str]] = None
    minutes_back: Optional[int] = Field(
        default=None,
        description=(
            "History lookback in minutes. Defaults to 10080 minutes (7 days) "
            "when start, end, and minutes_back are omitted."
        ),
    )
    limit: Optional[int] = 100
    offset: int = 0
    page: Optional[int] = None

    @field_validator("side", mode="before")
    @classmethod
    def _normalize_side(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_side_alias(value)


class TradeJournalAnalyzeRequest(BaseModel):
    detail: CompactFullDetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Compact returns summary only; standard adds "
            "symbol aggregates; summary adds symbol and side aggregates; full "
            "includes expanded breakdowns and trade lists."
        ),
    )
    start: Optional[str] = None
    end: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = Field(
        default=None,
        description="Optional side filter. Accepts buy/sell or long/short.",
    )
    position_ticket: Optional[Union[int, str]] = None
    deal_ticket: Optional[Union[int, str]] = None
    minutes_back: Optional[int] = Field(
        default=None,
        description=(
            "Journal history lookback in minutes. Defaults to 10080 minutes "
            "(7 days) when start, end, and minutes_back are omitted."
        ),
    )
    limit: Optional[int] = Field(
        default=50,
        description="Maximum raw history rows to inspect. Default 50 keeps post-session review fast; raise for longer-term statistics.",
    )
    breakdown_limit: int = 10
    min_sample: int = Field(
        default=30,
        description=(
            "Recommended minimum realized exit deals for reliable journal "
            "statistics (default 30). Smaller samples still return metrics but "
            "are flagged via sample_quality/sample_warning rather than suppressed."
        ),
    )
    check_only: bool = Field(
        default=False,
        description="Return sample sufficiency metadata without computing journal statistics.",
    )

    @field_validator("side", mode="before")
    @classmethod
    def _normalize_side(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_side_alias(value)

    @field_validator("breakdown_limit", "min_sample")
    @classmethod
    def _validate_positive_count(cls, value: int) -> int:
        value_i = int(value)
        if value_i <= 0:
            raise ValueError("value must be greater than 0.")
        return value_i


class TradeRiskAnalyzeRequest(BaseModel):
    model_config = {"populate_by_name": True}

    symbol: Optional[str] = None
    detail: CompactFullDetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Compact keeps sizing/action fields; full "
            "includes broker volume diagnostics and incomplete-sizing context."
        ),
    )
    desired_risk_pct: Optional[float] = None
    sizing_method: Literal["fixed_fraction", "kelly"] = Field(
        default="fixed_fraction",
        description=(
            "Position sizing method. fixed_fraction uses desired_risk_pct; "
            "kelly uses win-rate and average win/loss inputs to derive risk. "
            "Use trade_journal_analyze to estimate those inputs from realized "
            "trade history."
        ),
    )
    kelly_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional metrics dict containing win_rate, avg_win_return, and "
            "avg_loss_return. The trade_journal_analyze summary provides "
            "compatible win_rate/avg_win/avg_loss inputs. Explicit kelly_* "
            "fields override this dict."
        ),
    )
    kelly_win_rate: Optional[float] = Field(
        default=None,
        description=(
            "Kelly win probability as a fraction in [0, 1]. Map from "
            "trade_journal_analyze summary.win_rate when available."
        ),
    )
    kelly_avg_win: Optional[float] = Field(
        default=None,
        description=(
            "Average winning return for Kelly sizing. A practical source is "
            "trade_journal_analyze summary.avg_win."
        ),
    )
    kelly_avg_loss: Optional[float] = Field(
        default=None,
        description=(
            "Average losing return magnitude for Kelly sizing. A practical "
            "source is trade_journal_analyze summary.avg_loss."
        ),
    )
    kelly_fraction_multiplier: float = Field(
        default=0.5,
        ge=0.0,
        description="Multiplier applied to the raw Kelly fraction; half-Kelly is 0.5.",
    )
    kelly_max_risk_pct: float = Field(
        default=2.0,
        gt=0.0,
        description="Maximum account risk percentage allowed for Kelly sizing.",
    )
    strict_risk: bool = Field(
        default=True,
        description=(
            "When true, return suggested_volume=0.0 if the broker minimum "
            "volume would exceed desired_risk_pct."
        ),
    )
    include_pending: bool = Field(
        default=True,
        description=(
            "Include contingent stop-loss risk from pending orders in portfolio "
            "risk totals when enough order price/SL metadata is available."
        ),
    )
    direction: Optional[str] = None
    entry: Optional[float] = Field(
        default=None,
        alias="entry",
        validation_alias=AliasChoices("entry", "proposed_entry"),
        description=(
            "Proposed entry price. When omitted with symbol and stop_loss, "
            "trade_risk_analyze resolves it from the live tick: ask for long, "
            "bid for short, or mid when direction is not specified."
        ),
    )
    stop_loss: Optional[float] = Field(
        default=None,
        alias="sl",
        validation_alias=AliasChoices("sl", "proposed_sl"),
    )
    take_profit: Optional[float] = Field(
        default=None,
        alias="tp",
        validation_alias=AliasChoices("tp", "proposed_tp"),
    )

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_direction_alias(value)


class TradeVarCvarRequest(BaseModel):
    symbol: Optional[str] = None
    timeframe: TimeframeLiteral = Field(
        default="H1",
        description="Return interval and one-bar VaR/CVaR holding period.",
    )
    lookback: int = 500
    confidence: float = Field(
        0.95,
        description=(
            "VaR/CVaR confidence level. Use a fraction such as 0.95 or 0.99, "
            "or a percentage such as 95. Values must resolve to 0 < confidence < 1."
        ),
    )
    method: str = "historical"
    transform: str = "log_return"
    min_observations: int = 50
    detail: CompactFullDetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Compact returns the risk summary; full also "
            "includes position, symbol-exposure, and worst-observation tables."
        ),
    )


class TradeStressTestRequest(BaseModel):
    shocks: Dict[str, float] = Field(
        ...,
        description=(
            "Per-symbol percentage price shocks, for example {'EURUSD': -2.0}. "
            "Use '*' as a fallback shock for symbols without an explicit entry."
        ),
    )
    include_unshocked: bool = False
    detail: CompactFullDetailLiteral = "compact"

    @field_validator("shocks")
    @classmethod
    def _validate_shocks(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("shocks must contain at least one symbol or '*' fallback.")
        normalized: Dict[str, float] = {}
        for raw_symbol, raw_shock in value.items():
            symbol = str(raw_symbol or "").strip().upper()
            if not symbol:
                raise ValueError("shock symbols must be non-empty strings.")
            shock = float(raw_shock)
            if not math.isfinite(shock) or shock <= -100.0:
                raise ValueError("shock percentages must be finite and greater than -100.")
            normalized[symbol] = shock
        return normalized


class TradeGetOpenRequest(BaseModel):
    symbol: Optional[str] = None
    ticket: Optional[Union[int, str]] = None
    side: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("side", "direction"),
        description="Optional direction filter. Accepts buy/sell or long/short.",
    )
    magic: Optional[int] = Field(default=None, description=MAGIC_NUMBER_DESCRIPTION)
    profit_only: bool = Field(
        default=False,
        description="Only return currently profitable open positions.",
    )
    loss_only: bool = Field(
        default=False,
        description="Only return currently losing open positions.",
    )
    close_priority: Optional[
        Literal["loss_first", "profit_first", "largest_first"]
    ] = Field(
        default=None,
        description=(
            "Order matching open positions as trade_close would process them: "
            "loss_first, profit_first, or largest_first."
        ),
    )
    limit: Optional[int] = 50
    detail: CompactFullDetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Use full to include echoed request metadata "
            "while preserving the standard read envelope."
        ),
    )

    @field_validator("side", mode="before")
    @classmethod
    def _normalize_side(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_side_alias(value)


class TradeGetPendingRequest(BaseModel):
    symbol: Optional[str] = None
    ticket: Optional[Union[int, str]] = None
    side: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("side", "direction"),
        description="Optional order direction filter. Accepts buy/sell or long/short.",
    )
    order_type: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("order_type", "type"),
        description=(
            "Optional pending order type filter: buy_limit, sell_limit, "
            "buy_stop, sell_stop, buy_stop_limit, or sell_stop_limit."
        ),
    )
    magic: Optional[int] = Field(default=None, description=MAGIC_NUMBER_DESCRIPTION)
    limit: Optional[int] = 50
    detail: CompactFullDetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Use full to include echoed request metadata "
            "while preserving the standard read envelope."
        ),
    )

    @field_validator("side", mode="before")
    @classmethod
    def _normalize_side(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_side_alias(value)

    @field_validator("order_type", mode="before")
    @classmethod
    def _normalize_order_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().upper()
        return text or None


class TradeSessionContextRequest(BaseModel):
    symbol: str
    detail: CompactFullDetailLiteral = "compact"
    include_account: bool = True
