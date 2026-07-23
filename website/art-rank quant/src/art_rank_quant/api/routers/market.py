from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from art_rank_quant.api.schemas.common import Envelope
from art_rank_quant.api.schemas.market import (
    MarketOverview,
    MarketSnapshot,
    RealtimeQuoteFeed,
    RefreshAccepted,
    RegimeForecast,
)
from art_rank_quant.api.services.realtime import get_realtime_market_service
from art_rank_quant.api.services.research_runtime import get_research_runtime_service

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/overview", response_model=Envelope[MarketOverview])
def overview() -> Envelope[MarketOverview]:
    return Envelope(data=MarketOverview(regime="震荡偏强", turnover_billion=982.4, breadth=0.574, volatility=0.187))


@router.get("/regime", response_model=Envelope[RegimeForecast])
def regime() -> Envelope[RegimeForecast]:
    return Envelope(
        data=RegimeForecast(state="震荡偏强", probabilities={"牛市": 0.31, "熊市": 0.12, "震荡": 0.46, "高波动": 0.11})
    )


@router.get("/quotes", response_model=Envelope[RealtimeQuoteFeed])
def realtime_quotes(
    symbols: str | None = Query(default=None, description="Comma-separated TS codes"),
) -> Envelope[RealtimeQuoteFeed]:
    service = get_realtime_market_service()
    selected = symbols.split(",") if symbols else None
    try:
        feed = service.quotes(selected)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Envelope(meta=service.metadata(feed), data=feed)


@router.get("/snapshot/latest", response_model=Envelope[MarketSnapshot])
def latest_snapshot() -> Envelope[MarketSnapshot]:
    snapshot, metadata = get_research_runtime_service().latest_market()
    return Envelope(meta=metadata, data=snapshot)


@router.post("/refresh", response_model=RefreshAccepted, status_code=status.HTTP_202_ACCEPTED)
def force_refresh(background_tasks: BackgroundTasks) -> RefreshAccepted:
    service = get_research_runtime_service()
    accepted = service.start_refresh()
    if accepted.status == "queued":
        background_tasks.add_task(service.execute_refresh, accepted.run_id)
    return accepted
