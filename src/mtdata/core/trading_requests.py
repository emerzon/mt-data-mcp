from __future__ import annotations

from typing import Optional, Union

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
    profit_only: bool = False
    loss_only: bool = False
    comment: Optional[str] = None
    deviation: int = 20
