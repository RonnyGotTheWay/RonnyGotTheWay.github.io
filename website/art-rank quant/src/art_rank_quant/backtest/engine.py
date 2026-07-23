from __future__ import annotations

from dataclasses import dataclass, field

from art_rank_quant.backtest.costs import CostModel


@dataclass(slots=True)
class PortfolioState:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)

    def value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(quantity * prices.get(symbol, 0.0) for symbol, quantity in self.positions.items())


class BacktestEngine:
    def __init__(self, initial_cash: float = 1_000_000.0, costs: CostModel | None = None) -> None:
        self.state = PortfolioState(initial_cash)
        self.costs = costs or CostModel()

    def fill(self, symbol: str, quantity: int, price: float) -> None:
        notional = quantity * price
        side = "buy" if quantity > 0 else "sell"
        fee = self.costs.estimate(notional, side)
        self.state.cash -= notional + fee
        self.state.positions[symbol] = self.state.positions.get(symbol, 0) + quantity
