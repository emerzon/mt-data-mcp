from __future__ import annotations

import math
from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...shared.schema import DetailLiteral, TimeframeLiteral
from ...utils.barriers import normalize_trade_direction_alias
from . import validation
from .time import ExpirationValue
from .validation import OrderTypeLiteral

MAGIC_NUMBER_DESCRIPTION = (
    "MT5 magic number: integer strategy/order identifier used to group EA or "
    "strategy trades. Use as a filter for one strategy; omit for all magic numbers."
)


class FixedFractionSizing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["fixed_fraction"] = "fixed_fraction"
    risk_pct: float = Field(gt=0.0, description="Target account risk percentage.")


class KellySizing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["kelly"] = "kelly"
    win_rate: float = Field(ge=0.0, le=1.0, description="Win probability as a fraction.")
    avg_win: float = Field(gt=0.0, description="Average stake-normalized winning return.")
    avg_loss: float = Field(gt=0.0, description="Average stake-normalized losing return magnitude.")
    fraction_multiplier: float = Field(0.5, ge=0.0, description="Multiplier applied to raw Kelly.")
    max_risk_pct: float = Field(2.0, gt=0.0, description="Maximum Kelly account risk percentage.")


RiskSizing = Annotated[Union[FixedFractionSizing, KellySizing], Field(discriminator="method")]


def _normalize_trade_side_alias(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized, error = validation._normalize_trade_side_filter(value)
    if error is None and normalized is not None:
        return normalized
    if error is not None:
        raise ValueError(error)
    return None


def _normalize_positive_ticket(value: Union[int, str]) -> int:
    if isinstance(value, bool):
        raise ValueError("ticket must be a positive integer")
    text = str(value).strip()
    if not text.isdigit():
        raise ValueError("ticket must be a positive integer")
    ticket = int(text)
    if ticket <= 0:
        raise ValueError("ticket must be a positive integer")
    return ticket


class _SideNormalizedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("side", mode="before", check_fields=False)
    @classmethod
    def _normalize_side(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_side_alias(value)


class TradePlaceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    symbol: str = Field(min_length=1)
    volume: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description="Order size in lots (e.g. 0.01), not traded/tick volume.",
    )
    order_type: OrderTypeLiteral = Field(
        description=(
            "Order type: BUY/SELL for market orders, or "
            "BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP for pending orders."
        ),
    )
    price: Optional[Union[int, float]] = None
    stop_loss: Optional[Union[int, float]] = None
    take_profit: Optional[Union[int, float]] = None
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
        ge=0,
        description="Maximum allowed execution slippage in points.",
    )
    dry_run: bool = Field(
        default=True,
        description=(
            "Preview the order without sending it to the broker. Defaults to "
            "true; set dry_run=false explicitly to place a live order."
        ),
    )
    detail: Literal["compact", "standard", "full"] = Field(
        default="compact",
        description=(
            "Response detail level. Compact returns the lean dry-run preview; "
            "standard adds local validation context; full keeps all "
            "preview diagnostics."
        ),
    )
    require_sl_tp: bool = Field(
        default=True,
        description=(
            "Require both stop_loss and take_profit for market orders and fail "
            "if protection cannot be attached. Market orders using this guarantee "
            "uses the internal unprotected-position recovery fail-safe."
        ),
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional durable dedupe key with a configurable 24-hour TTL. "
            "Reusing the same key with the same payload replays the prior "
            "result instead of sending another order. The SQLite store is shared "
            "across processes and restarts; this is not broker-side idempotency."
        ),
    )

    @field_validator("order_type", mode="before")
    @classmethod
    def _normalize_order_type(cls, value: Any) -> Any:
        return str(value).strip().upper() if isinstance(value, str) else value

    @property
    def auto_close_on_sl_tp_fail(self) -> bool:
        return True


class TradeModifyRequest(BaseModel):
    model_config = {"populate_by_name": True}

    ticket: Union[int, str]

    @field_validator("ticket", mode="before")
    @classmethod
    def _validate_ticket(cls, value: Union[int, str]) -> int:
        return _normalize_positive_ticket(value)
    detail: DetailLiteral = Field(
        default="compact",
        description="Response detail level for modify previews and result payloads.",
    )
    price: Optional[Union[int, float]] = None
    stop_loss: Optional[Union[int, float]] = None
    take_profit: Optional[Union[int, float]] = None
    expiration: Optional[ExpirationValue] = None
    comment: Optional[str] = None
    dry_run: bool = Field(
        default=True,
        description=(
            "Preview the modification without sending it to the broker. Defaults "
            "to true; set dry_run=false explicitly to modify a live order or "
            "position."
        ),
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional durable dedupe key with a configurable 24-hour TTL. "
            "Reusing the same key with the same payload replays the prior "
            "result instead of sending another modify request. The SQLite store "
            "is shared across processes and restarts; this is not broker-side idempotency."
        ),
    )


class TradeCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket: Optional[Union[int, str]] = None
    detail: DetailLiteral = Field(
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
        gt=0.0,
        description="Partial close volume in lots. Requires ticket.",
    )
    dry_run: bool = Field(
        default=True,
        description=(
            "Preview the close request without sending it to the broker. Defaults "
            "to true; set dry_run=false explicitly to close a live position or "
            "order."
        ),
    )
    confirm_close_all: bool = Field(
        default=False,
        description=(
            "Required with close_all=true and dry_run=false to execute a live "
            "bulk close."
        ),
    )
    pnl_filter: Literal["all", "profit", "loss"] = Field(
        default="all",
        description="Restrict matching positions by current profit-and-loss sign.",
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
    deviation: int = Field(default=20, ge=0)
    idempotency_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional durable dedupe key with a configurable 24-hour TTL. "
            "Reusing the same key with the same payload replays the prior "
            "close/cancel outcome instead of sending another broker request."
        ),
    )

    @property
    def profit_only(self) -> bool:
        return self.pnl_filter == "profit"

    @property
    def loss_only(self) -> bool:
        return self.pnl_filter == "loss"


class TradeHistoryRequest(_SideNormalizedRequest):
    history_kind: Literal["deals", "orders"] = Field(
        default="deals",
        description=(
            "Trade history type. deals = executed fills with P&L for journals; "
            "orders = order lifecycle events for audit/reconciliation."
        ),
    )
    detail: DetailLiteral = "compact"
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
    limit: int = Field(default=100, ge=1)
    offset: int = Field(default=0, ge=0)
    page: Optional[int] = Field(default=None, ge=1)
    order: Literal["desc", "asc"] = Field(
        default="desc",
        description="History time order. desc returns newest activity first.",
    )


class TradeJournalAnalyzeRequest(_SideNormalizedRequest):
    detail: DetailLiteral = Field(
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
    limit: int = Field(
        default=50,
        ge=1,
        description=(
            "Maximum realized exit deals to analyze. The command pages through "
            "raw deal history as needed. Default 50 keeps post-session review fast."
        ),
    )
    breakdown_limit: int = Field(default=10, ge=1)
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

    @field_validator("breakdown_limit", "min_sample")
    @classmethod
    def _validate_positive_count(cls, value: int) -> int:
        value_i = int(value)
        if value_i <= 0:
            raise ValueError("value must be greater than 0.")
        return value_i


class TradeRiskAnalyzeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    symbol: Optional[str] = None
    detail: DetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Compact keeps sizing/action fields; full "
            "includes broker volume diagnostics and incomplete-sizing context."
        ),
    )
    sizing: Optional[RiskSizing] = Field(
        default=None,
        description="Optional fixed-fraction or Kelly position-sizing inputs.",
    )
    strict_risk: bool = Field(
        default=True,
        description=(
            "When true, return suggested_volume=0.0 if the broker minimum "
            "volume would exceed the requested sizing risk."
        ),
    )
    include_pending: bool = Field(
        default=True,
        description=(
            "Include contingent stop-loss risk from pending orders in portfolio "
            "risk totals when enough order price/SL metadata is available."
        ),
    )
    direction: Optional[Literal["long", "short"]] = None
    entry: Optional[float] = Field(
        default=None,
        description=(
            "Proposed entry price. When omitted with symbol and stop_loss, "
            "trade_risk_analyze resolves it from the live tick: ask for long, "
            "bid for short, or mid when direction is not specified."
        ),
    )
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Optional[str]) -> Optional[str]:
        return normalize_trade_direction_alias(value)

    @property
    def desired_risk_pct(self) -> Optional[float]:
        return self.sizing.risk_pct if isinstance(self.sizing, FixedFractionSizing) else None

    @property
    def sizing_method(self) -> str:
        return self.sizing.method if self.sizing is not None else "fixed_fraction"

    @property
    def kelly_metrics(self) -> Optional[Dict[str, float]]:
        if not isinstance(self.sizing, KellySizing):
            return None
        return {
            "win_rate": self.sizing.win_rate,
            "avg_win_return": self.sizing.avg_win,
            "avg_loss_return": self.sizing.avg_loss,
        }

    @property
    def kelly_win_rate(self) -> Optional[float]:
        return self.sizing.win_rate if isinstance(self.sizing, KellySizing) else None

    @property
    def kelly_avg_win(self) -> Optional[float]:
        return self.sizing.avg_win if isinstance(self.sizing, KellySizing) else None

    @property
    def kelly_avg_loss(self) -> Optional[float]:
        return self.sizing.avg_loss if isinstance(self.sizing, KellySizing) else None

    @property
    def kelly_fraction_multiplier(self) -> float:
        return self.sizing.fraction_multiplier if isinstance(self.sizing, KellySizing) else 0.5

    @property
    def kelly_max_risk_pct(self) -> float:
        return self.sizing.max_risk_pct if isinstance(self.sizing, KellySizing) else 2.0


class TradeVarCvarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: Optional[str] = Field(
        default=None,
        description=(
            "Optional scope: calculate VaR/CVaR for currently open positions in this "
            "symbol. Omit it for the full open portfolio."
        ),
    )
    timeframe: TimeframeLiteral = Field(
        default="H1",
        description="Return interval and one-bar VaR/CVaR holding period.",
    )
    lookback: int = Field(500, ge=2)
    include_incomplete: bool = Field(
        default=False,
        description=(
            "Include the current forming candle in return history. Defaults to false "
            "so VaR/CVaR uses completed bars only."
        ),
    )
    confidence: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description=(
            "VaR/CVaR confidence level. Use a fraction such as 0.95 or 0.99, "
            "Values must satisfy 0 < confidence < 1."
        ),
    )
    method: Literal["historical", "parametric"] = Field(
        default="historical",
        description=(
            "Tail-risk method: historical or parametric."
        ),
    )
    transform: Literal["log_return", "pct"] = Field(
        default="log_return",
        description=(
            "Return transform: log_return or pct."
        ),
    )
    min_observations: int = Field(50, ge=2)
    detail: DetailLiteral = Field(
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
    detail: DetailLiteral = "compact"

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


class TradeGetOpenRequest(_SideNormalizedRequest):
    symbol: Optional[str] = None
    ticket: Optional[Union[int, str]] = None
    side: Optional[str] = Field(
        default=None,
        description="Optional direction filter. Accepts buy/sell or long/short.",
    )
    magic: Optional[int] = Field(default=None, description=MAGIC_NUMBER_DESCRIPTION)
    pnl_filter: Literal["all", "profit", "loss"] = Field(
        default="all",
        description="Restrict open positions by current profit-and-loss sign.",
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
    limit: int = Field(default=50, ge=1)
    detail: DetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Use full to include echoed request metadata "
            "while preserving the standard read envelope."
        ),
    )

    @property
    def profit_only(self) -> bool:
        return self.pnl_filter == "profit"

    @property
    def loss_only(self) -> bool:
        return self.pnl_filter == "loss"


class TradeGetPendingRequest(_SideNormalizedRequest):
    symbol: Optional[str] = None
    ticket: Optional[Union[int, str]] = None
    side: Optional[str] = Field(
        default=None,
        description="Optional order direction filter. Accepts buy/sell or long/short.",
    )
    order_type: Optional[str] = Field(
        default=None,
        description=(
            "Optional pending order type filter: buy_limit, sell_limit, "
            "buy_stop, sell_stop, buy_stop_limit, or sell_stop_limit."
        ),
    )
    magic: Optional[int] = Field(default=None, description=MAGIC_NUMBER_DESCRIPTION)
    limit: int = Field(default=50, ge=1)
    detail: DetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Use full to include echoed request metadata "
            "while preserving the standard read envelope."
        ),
    )

    @field_validator("order_type", mode="before")
    @classmethod
    def _normalize_order_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().upper()
        if not text:
            return None
        allowed = {
            "BUY_LIMIT",
            "SELL_LIMIT",
            "BUY_STOP",
            "SELL_STOP",
            "BUY_STOP_LIMIT",
            "SELL_STOP_LIMIT",
        }
        if text not in allowed:
            raise ValueError(
                "order_type must be one of: " + ", ".join(sorted(allowed))
            )
        return text


class TradeSessionContextRequest(BaseModel):
    symbol: str
    detail: DetailLiteral = "compact"
    include_account: bool = True
