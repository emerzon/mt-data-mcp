from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel

from .trading_time import ExpirationValue
from .trading_validation import OrderTypeInput


class TradePlaceRequest(BaseModel):
    symbol: Optional[str] = None
    volume: Optional[float] = None
    order_type: Optional[OrderTypeInput] = None
    price: Optional[Union[int, float]] = None
    stop_loss: Optional[Union[int, float]] = None
    take_profit: Optional[Union[int, float]] = None
    expiration: Optional[ExpirationValue] = None
    comment: Optional[str] = None
    deviation: int = 20
    require_sl_tp: bool = True
    auto_close_on_sl_tp_fail: bool = False


class TradeModifyRequest(BaseModel):
    ticket: Union[int, str]
    price: Optional[Union[int, float]] = None
    stop_loss: Optional[Union[int, float]] = None
    take_profit: Optional[Union[int, float]] = None
    expiration: Optional[ExpirationValue] = None
    comment: Optional[str] = None


class TradeCloseRequest(BaseModel):
    ticket: Optional[Union[int, str]] = None
    symbol: Optional[str] = None
    volume: Optional[float] = None
    profit_only: bool = False
    loss_only: bool = False
    comment: Optional[str] = None
    deviation: int = 20


class TradeHistoryRequest(BaseModel):
    history_kind: Literal["deals", "orders"] = "deals"
    start: Optional[str] = None
    end: Optional[str] = None
    symbol: Optional[str] = None
    position_ticket: Optional[Union[int, str]] = None
    deal_ticket: Optional[Union[int, str]] = None
    order_ticket: Optional[Union[int, str]] = None
    minutes_back: Optional[int] = None
    limit: Optional[int] = 200


class TradeRiskAnalyzeRequest(BaseModel):
    symbol: Optional[str] = None
    desired_risk_pct: Optional[float] = None
    direction: Optional[str] = None
    proposed_entry: Optional[float] = None
    proposed_sl: Optional[float] = None
    proposed_tp: Optional[float] = None


class TradeGetOpenRequest(BaseModel):
    symbol: Optional[str] = None
    ticket: Optional[Union[int, str]] = None
    limit: Optional[int] = 200


class TradeGetPendingRequest(BaseModel):
    symbol: Optional[str] = None
    ticket: Optional[Union[int, str]] = None
    limit: Optional[int] = 200
