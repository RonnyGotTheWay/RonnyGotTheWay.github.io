from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Order:
    symbol: str
    signal_date: date
    side: str
    quantity: int


@dataclass(frozen=True, slots=True)
class MarketState:
    trade_date: date
    open_price: float
    volume: int
    trade_status: str = "trading"
    limit_state: str = "none"


def executable(
    order: Order, market: MarketState, acquired_on: date | None = None, max_participation: float = 0.05
) -> tuple[bool, str]:
    if market.trade_date <= order.signal_date:
        return False, "next_session_only"
    if market.trade_status != "trading":
        return False, "suspended"
    if order.side == "buy" and market.limit_state == "up":
        return False, "limit_up"
    if order.side == "sell" and market.limit_state == "down":
        return False, "limit_down"
    if order.side == "sell" and acquired_on == market.trade_date:
        return False, "t_plus_one"
    if order.quantity > int(market.volume * max_participation):
        return False, "capacity"
    return True, "ok"
