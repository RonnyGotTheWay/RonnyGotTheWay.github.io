from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class MarketOverview(BaseModel):
    regime: str
    turnover_billion: float
    breadth: float
    volatility: float


class RegimeForecast(BaseModel):
    state: str
    probabilities: dict[str, float]


class StyleForecast(BaseModel):
    horizon: int
    probabilities: dict[str, float]
    expected_relative_returns: dict[str, float]


class RealtimeQuote(BaseModel):
    symbol: str
    name: str
    pre_close: float
    open: float
    high: float
    low: float
    close: float
    pct_change: float
    volume: float
    amount: float
    trade_count: int
    observed_at: datetime


class RealtimeQuoteFeed(BaseModel):
    provider: str
    status: Literal["live", "degraded", "configuration_required", "upstream_error"]
    market_session: Literal["trading", "lunch_break", "closed"]
    message: str
    refresh_seconds: int
    fetched_at: datetime | None = None
    quotes: list[RealtimeQuote] = Field(default_factory=list)


class MarketQuoteItem(BaseModel):
    symbol: str
    name: str
    industry: str
    exchange: Literal["SH", "SZ", "BJ"]
    close: float | None
    pct_change: float | None
    amount: float
    turnover_rate: float | None
    volume_ratio: float | None
    amplitude: float | None
    trade_status: str
    risk_flags: list[str] = Field(default_factory=list)


class MarketSnapshot(BaseModel):
    status: Literal["live", "stale", "not_available", "upstream_error"]
    market_session: Literal["trading", "lunch_break", "closed"]
    message: str
    trade_date: date | None = None
    fetched_at: datetime | None = None
    duration_seconds: float | None = None
    universe_count: int = 0
    trading_count: int = 0
    advancing_count: int = 0
    declining_count: int = 0
    total_amount: float = 0.0
    quotes: list[MarketQuoteItem] = Field(default_factory=list)


class RefreshAccepted(BaseModel):
    run_id: str
    status: Literal["queued", "running"]
    message: str


class RefreshRun(BaseModel):
    run_id: str
    status: Literal["queued", "running", "success", "failed"]
    step: Literal["queued", "ingestion", "quality", "features", "inference", "publish", "complete", "failed"]
    progress: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    data_version: str | None = None
    prediction_status: str | None = None
    error: str | None = None
