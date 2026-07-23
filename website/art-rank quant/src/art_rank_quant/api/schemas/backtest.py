from pydantic import BaseModel


class EquityPoint(BaseModel):
    date: str
    strategy: float
    benchmark: float


class BacktestResult(BaseModel):
    run_id: str
    total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    rank_ic: float
    equity: list[EquityPoint]
