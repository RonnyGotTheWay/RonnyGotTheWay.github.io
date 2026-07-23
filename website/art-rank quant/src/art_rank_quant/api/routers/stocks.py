from fastapi import APIRouter, HTTPException, Query

from art_rank_quant.api.schemas.common import Envelope
from art_rank_quant.api.schemas.stocks import CandidateFeed, PredictionFeed, StockDetail, StockScore
from art_rank_quant.api.services.demo import rankings
from art_rank_quant.api.services.realtime import get_realtime_market_service
from art_rank_quant.api.services.research_runtime import get_research_runtime_service
from art_rank_quant.api.services.short_term import get_short_term_runtime_service

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/rankings", response_model=Envelope[list[StockScore]])
def stock_rankings(
    index_code: str | None = None, sector: str | None = None, limit: int = Query(50, ge=1, le=100)
) -> Envelope[list[StockScore]]:
    rows = rankings()
    if index_code:
        rows = [row for row in rows if row["index_code"] == index_code]
    if sector:
        rows = [row for row in rows if row["sector"] == sector]
    return Envelope(data=[StockScore.model_validate(row) for row in rows[:limit]])


@router.get("/candidates/realtime", response_model=Envelope[CandidateFeed])
def realtime_candidates(
    symbols: str | None = Query(default=None, description="Comma-separated TS codes"),
    limit: int = Query(default=20, ge=1, le=50),
) -> Envelope[CandidateFeed]:
    service = get_realtime_market_service()
    selected = symbols.split(",") if symbols else None
    try:
        feed = service.candidates(selected, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Envelope(meta=service.metadata(feed), data=feed)


@router.get("/predictions/latest", response_model=Envelope[PredictionFeed])
def latest_predictions(
    limit: int = Query(default=20, ge=1, le=5000),
    include_all: bool = Query(default=False),
) -> Envelope[PredictionFeed]:
    feed, metadata = get_research_runtime_service().latest_prediction(limit=limit, include_all=include_all)
    return Envelope(meta=metadata, data=feed)


@router.get("/predictions/short-term/latest", response_model=Envelope[PredictionFeed])
def latest_short_term_predictions(
    limit: int = Query(default=20, ge=1, le=5000),
    include_all: bool = Query(default=False),
) -> Envelope[PredictionFeed]:
    feed, metadata = get_short_term_runtime_service().latest_prediction(limit=limit, include_all=include_all)
    return Envelope(meta=metadata, data=feed)


@router.get("/predictions/history", response_model=Envelope[list[dict[str, object]]])
def prediction_history() -> Envelope[list[dict[str, object]]]:
    service = get_research_runtime_service()
    rows = service.prediction_history()
    _, metadata = service.latest_prediction()
    return Envelope(meta=metadata, data=rows)


@router.get("/{symbol}", response_model=Envelope[StockDetail])
def stock_detail(symbol: str) -> Envelope[StockDetail]:
    row = next((item for item in rankings() if item["symbol"] == symbol), None)
    if row is None:
        raise HTTPException(404, "Unknown synthetic symbol")
    return Envelope(
        data=StockDetail.model_validate(
            row
            | {
                "price": 24.81,
                "momentum_20d": 0.064,
                "contributions": {"20日动量": 0.29, "质量": 0.18, "估值": -0.08, "波动率": -0.05},
            }
        )
    )
