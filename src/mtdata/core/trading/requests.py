from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, Field, field_validator

from ...shared.schema import CompactFullDetailLiteral, TimeframeLiteral
from .time import ExpirationValue
from . import validation
from .validation import OrderTypeInput


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
    text = str(value)
    normalized = text.strip().lower()
    if normalized in {"long", "short", "buy", "sell", "up", "down"}:
        return "long" if normalized in {"long", "buy", "up"} else "short"
    return value


class TradePlaceRequest(BaseModel):
    model_config = {"populate_by_name": True}

    symbol: Optional[str] = None
    volume: Optional[float] = None
    order_type: Optional[OrderTypeInput] = None
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
    deviation: int = 20
    dry_run: bool = False
    detail: Literal["preview", "basic", "full", "compact", "summary"] = Field(
        default="basic",
        description=(
            "Dry-run preview detail level. Use preview/basic/full; compact/summary "
            "aliases map to preview."
        ),
    )
    require_sl_tp: bool = True
    auto_close_on_sl_tp_fail: bool = False
    idempotency_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional in-process dedupe key. Reusing the same key with the same "
            "payload replays the prior result instead of sending another order."
        ),
    )


class TradeModifyRequest(BaseModel):
    model_config = {"populate_by_name": True}

    ticket: Union[int, str]
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
    idempotency_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional in-process dedupe key. Reusing the same key with the same "
            "payload replays the prior result instead of sending another modify request."
        ),
    )


class TradeCloseRequest(BaseModel):
    ticket: Optional[Union[int, str]] = None
    close_all: bool = False
    symbol: Optional[str] = None
    volume: Optional[float] = None
    profit_only: bool = False
    loss_only: bool = False
    close_priority: Optional[Literal["loss_first", "profit_first", "largest_first"]] = None
    comment: Optional[str] = None
    deviation: int = 20


class TradeHistoryRequest(BaseModel):
    history_kind: Literal["deals", "orders"] = "deals"
    detail: CompactFullDetailLiteral = "compact"
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
    minutes_back: Optional[int] = None
    limit: Optional[int] = 200

    @field_validator("side", mode="before")
    @classmethod
    def _normalize_side(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_side_alias(value)


class TradeJournalAnalyzeRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = Field(
        default=None,
        description="Optional side filter. Accepts buy/sell or long/short.",
    )
    position_ticket: Optional[Union[int, str]] = None
    deal_ticket: Optional[Union[int, str]] = None
    minutes_back: Optional[int] = None
    limit: Optional[int] = 200
    breakdown_limit: int = 10

    @field_validator("side", mode="before")
    @classmethod
    def _normalize_side(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_side_alias(value)


class TradeRiskAnalyzeRequest(BaseModel):
    model_config = {"populate_by_name": True}

    symbol: Optional[str] = None
    desired_risk_pct: Optional[float] = None
    direction: Optional[str] = None
    entry: Optional[float] = Field(
        default=None,
        validation_alias="proposed_entry",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("proposed_sl", "sl"),
    )
    take_profit: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("proposed_tp", "tp"),
    )

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_direction_alias(value)


class TradeVarCvarRequest(BaseModel):
    symbol: Optional[str] = None
    timeframe: TimeframeLiteral = "H1"
    lookback: int = 500
    confidence: float = 0.95
    method: str = "historical"
    transform: str = "log_return"
    min_observations: int = 50


class TradeGetOpenRequest(BaseModel):
    symbol: Optional[str] = None
    ticket: Optional[Union[int, str]] = None
    limit: Optional[int] = 200
    detail: CompactFullDetailLiteral = Field(
        default="full",
        description=(
            "Response detail level. Use compact to omit echoed request metadata "
            "while preserving the standard read envelope."
        ),
    )
    column_style: Literal["humanized", "snake_case"] = Field(
        default="humanized",
        description=(
            "Trading table column naming style. Use humanized for the legacy "
            "title-cased headers or snake_case for MT5-style field names."
        ),
    )


class TradeGetPendingRequest(BaseModel):
    symbol: Optional[str] = None
    ticket: Optional[Union[int, str]] = None
    limit: Optional[int] = 200
    detail: CompactFullDetailLiteral = Field(
        default="full",
        description=(
            "Response detail level. Use compact to omit echoed request metadata "
            "while preserving the standard read envelope."
        ),
    )
    column_style: Literal["humanized", "snake_case"] = Field(
        default="humanized",
        description=(
            "Trading table column naming style. Use humanized for the legacy "
            "title-cased headers or snake_case for MT5-style field names."
        ),
    )


class TradeSessionContextRequest(BaseModel):
    symbol: str
    detail: CompactFullDetailLiteral = "compact"
