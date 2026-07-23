from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from art_rank_quant.api.schemas.stocks import PredictionContext


class WeeklyBreadthPoint(BaseModel):
    trade_date: date
    advancing: int
    declining: int
    unchanged: int
    total_amount: float
    mean_return_pct: float


class WeeklyReturnBucket(BaseModel):
    label: str
    lower: float | None = None
    upper: float | None = None
    count: int


class WeeklyIndustrySummary(BaseModel):
    industry: str
    median_return_pct: float
    advancing_ratio_pct: float
    stock_count: int
    median_return_5d_pct: float | None = None
    median_return_20d_pct: float | None = None
    median_return_60d_pct: float | None = None
    advancing_ratio_5d_pct: float | None = None
    advancing_ratio_20d_pct: float | None = None
    advancing_ratio_60d_pct: float | None = None
    covered_stock_count: int = 0
    trend: Literal["up", "down", "mixed", "insufficient"] = "insufficient"
    trend_label: str = "数据不足"
    composite_score: float = 0.0
    recommendation_tier: Literal["focus", "neutral", "avoid", "insufficient"] = "insufficient"
    recommendation_label: str = "数据不足"
    confidence_pct: float = 0.0
    recommendation_reasons: list[str] = Field(default_factory=list)


class WeeklyPriceGuidance(BaseModel):
    status: Literal["ready", "watch", "blocked"] = "blocked"
    horizon_days: int = 5
    valid_trade_date: date | None = None
    buy_price_low: float | None = None
    buy_price_high: float | None = None
    sell_price_low: float | None = None
    sell_price_high: float | None = None
    expected_gain_low_pct: float | None = None
    expected_gain_high_pct: float | None = None
    stop_loss_price: float | None = None
    positive_return_probability: float | None = None
    target_hit_probability: float | None = None
    risk_reward_ratio: float | None = None
    calibration_sample_size: int = 0
    methodology_version: str = "empirical-5d-v2"
    is_reference_value: bool = False
    reference_method: Literal[
        "industry_positive_quantiles", "global_positive_quantiles", "atr_volatility"
    ] | None = None
    blocked_reasons: list[str] = Field(default_factory=list)


class WeeklyMarketReport(BaseModel):
    universe_count: int
    weekly_median_return_pct: float
    advancing_count: int
    declining_count: int
    unchanged_count: int
    total_amount: float
    daily_return_volatility_pct: float
    breadth: list[WeeklyBreadthPoint]
    return_distribution: list[WeeklyReturnBucket]
    industries: list[WeeklyIndustrySummary]


class WeeklyRankingItem(BaseModel):
    symbol: str
    name: str
    industry: str
    exchange: str
    rank: int
    close: float
    art_score: float
    score_percentile: float
    strategy_score: float | None = None
    total_score: float | None = None
    stability_score: float
    weekly_return_pct: float | None = None
    contributions: dict[str, float] = Field(default_factory=dict)
    strategy_adjustments: dict[str, float] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)
    price_guidance: WeeklyPriceGuidance = Field(default_factory=WeeklyPriceGuidance)


class WeeklySectorRanking(BaseModel):
    industry: str
    trend: Literal["up", "down", "mixed", "insufficient"] = "insufficient"
    trend_label: str = "数据不足"
    stock_count: int = 0
    eligible_count: int = 0
    candidates: list[WeeklyRankingItem] = Field(default_factory=list, max_length=10)


class WeeklyRankingReport(BaseModel):
    status: Literal["published", "blocked"]
    message: str
    scoring_date: date
    target_trade_date: date | None = None
    model_version: str
    experimental: bool = False
    training_rows: int = 0
    validation_rank_ic: float | None = None
    art_weight: float = 0.65
    strategy_weight: float = 0.35
    strategy_version: str = "weekly-experience-v1"
    screened_out_count: int = 0
    context: PredictionContext = Field(default_factory=PredictionContext)
    blocked_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    candidates: list[WeeklyRankingItem] = Field(default_factory=list, max_length=10)
    sector_rankings: list[WeeklySectorRanking] = Field(default_factory=list)


class WeeklyDataSource(BaseModel):
    provider: Literal["akshare", "tushare", "baostock", "eastmoney", "local_history"]
    role: Literal["primary", "cross_check", "fallback"]
    status: Literal["verified", "configuration_required", "unavailable"]
    rows: int = 0
    message: str


class WeeklyReport(BaseModel):
    status: Literal["published", "ranking_blocked", "stale", "not_available"]
    message: str
    period_start: date | None = None
    period_end: date | None = None
    trading_dates: list[date] = Field(default_factory=list)
    generated_at: datetime
    market: WeeklyMarketReport | None = None
    ranking: WeeklyRankingReport | None = None
    data_sources: list[WeeklyDataSource] = Field(default_factory=list)
