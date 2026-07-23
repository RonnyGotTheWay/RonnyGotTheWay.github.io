from fastapi import APIRouter

from art_rank_quant.api.schemas.backtest import BacktestResult, EquityPoint
from art_rank_quant.api.schemas.common import Envelope
from art_rank_quant.api.services.demo import equity_curve

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.get("/{run_id}", response_model=Envelope[BacktestResult])
def backtest(run_id: str) -> Envelope[BacktestResult]:
    curve = [EquityPoint.model_validate(point) for point in equity_curve()]
    return Envelope(
        data=BacktestResult(
            run_id=run_id,
            total_return=0.184,
            annualized_return=0.142,
            sharpe=1.31,
            max_drawdown=-0.087,
            rank_ic=0.041,
            equity=curve,
        )
    )
