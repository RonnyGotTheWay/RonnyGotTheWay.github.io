from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class StockScore(BaseModel):
    symbol: str
    name: str
    index_code: str
    sector: str
    score: float
    rank: int
    risk_flags: list[str] = []


class StockDetail(StockScore):
    price: float
    momentum_20d: float
    contributions: dict[str, float]


class ResearchCandidate(BaseModel):
    symbol: str
    name: str
    close: float
    pct_change: float
    amount: float
    observed_at: datetime
    observation_score: float
    rank: int
    factors: dict[str, float]
    risk_flags: list[str] = Field(default_factory=list)
    gate_status: Literal["research_only"] = "research_only"


class CandidateFeed(BaseModel):
    provider: str
    status: Literal["live", "degraded", "configuration_required", "upstream_error"]
    market_session: Literal["trading", "lunch_break", "closed"]
    message: str
    methodology: str
    refresh_seconds: int
    universe_count: int = 0
    rankable_count: int = 0
    fetched_at: datetime | None = None
    candidates: list[ResearchCandidate] = Field(default_factory=list)


class PredictionCandidate(BaseModel):
    symbol: str
    name: str
    industry: str
    exchange: str
    rank: int
    close: float
    art_score: float
    score_percentile: float
    stability_score: float
    momentum_5d: float | None = None
    momentum_1d: float | None = None
    momentum_3d: float | None = None
    momentum_10d: float | None = None
    momentum_20d: float | None = None
    momentum_60d: float | None = None
    volatility_20d: float | None = None
    liquidity_score: float | None = None
    volume_progress: float | None = None
    patchtst_contribution: float | None = None
    tft_contribution: float | None = None
    ranker_score: float
    contributions: dict[str, float] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)


class PredictionContext(BaseModel):
    hmm_state: str | None = None
    hmm_probabilities: dict[str, float] = Field(default_factory=dict)
    tft_style: str | None = None
    tft_probabilities: dict[str, float] = Field(default_factory=dict)


class PredictionFeed(BaseModel):
    status: Literal["published", "blocked", "stale", "failed"]
    message: str
    prediction_at: datetime | None = None
    target_trade_date: date | None = None
    eligible_count: int = 0
    context: PredictionContext = Field(default_factory=PredictionContext)
    candidates: list[PredictionCandidate] = Field(default_factory=list)
    prediction_mode: Literal["long_term", "short_term"] = "long_term"
    training_window_days: int | None = None
    validation_days: int = 0
    experimental: bool = False
    warnings: list[str] = Field(default_factory=list)
