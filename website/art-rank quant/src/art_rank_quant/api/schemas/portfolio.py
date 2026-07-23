from pydantic import BaseModel


class Position(BaseModel):
    symbol: str
    weight: float
    return_pct: float
    sector: str


class PortfolioSnapshot(BaseModel):
    nav: float
    daily_return: float
    max_drawdown: float
    turnover: float
    positions: list[Position]
    constraint_status: str
