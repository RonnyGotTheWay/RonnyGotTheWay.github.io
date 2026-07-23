from fastapi import APIRouter, Query

from art_rank_quant.api.schemas.common import Envelope
from art_rank_quant.api.schemas.market import StyleForecast

router = APIRouter(prefix="/style", tags=["style"])


@router.get("/forecast", response_model=Envelope[StyleForecast])
def forecast(horizon: int = Query(5, pattern="^(5|20)$")) -> Envelope[StyleForecast]:
    probability = {"SYN300": 0.27, "SYN500": 0.41, "SYN1000": 0.32}
    expected = {"SYN300": 0.003, "SYN500": 0.008, "SYN1000": 0.005}
    return Envelope(data=StyleForecast(horizon=horizon, probabilities=probability, expected_relative_returns=expected))
