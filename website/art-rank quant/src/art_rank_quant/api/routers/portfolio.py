from fastapi import APIRouter

from art_rank_quant.api.schemas.common import Envelope
from art_rank_quant.api.schemas.portfolio import PortfolioSnapshot, Position
from art_rank_quant.api.services.demo import rankings

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _snapshot() -> PortfolioSnapshot:
    rows = rankings()[:12]
    positions = [
        Position(
            symbol=str(row["symbol"]),
            weight=round(1 / len(rows), 4),
            return_pct=round((7 - i) * 0.0021, 4),
            sector=str(row["sector"]),
        )
        for i, row in enumerate(rows)
    ]
    return PortfolioSnapshot(
        nav=1.0874,
        daily_return=0.0046,
        max_drawdown=-0.063,
        turnover=0.217,
        positions=positions,
        constraint_status="全部通过",
    )


@router.get("/current", response_model=Envelope[PortfolioSnapshot])
def current() -> Envelope[PortfolioSnapshot]:
    return Envelope(data=_snapshot())


@router.get("/history", response_model=Envelope[list[PortfolioSnapshot]])
def history() -> Envelope[list[PortfolioSnapshot]]:
    return Envelope(data=[_snapshot()])
