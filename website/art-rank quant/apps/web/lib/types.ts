export type VersionMeta = {
  as_of_date: string;
  generated_at: string;
  data_version: string;
  feature_version: string;
  model_version: string;
  run_id: string;
  is_stale: boolean;
};

export type Envelope<T> = {meta: VersionMeta; data: T};
export type StockScore = {symbol: string; name: string; index_code: string; sector: string; score: number; rank: number; risk_flags: string[]};
export type Position = {symbol: string; weight: number; return_pct: number; sector: string};
export type EquityPoint = {date: string; strategy: number; benchmark: number};
export type RealtimeStatus = "live" | "degraded" | "configuration_required" | "upstream_error";
export type MarketSession = "trading" | "lunch_break" | "closed";
export type ResearchCandidate = {
  symbol: string;
  name: string;
  close: number;
  pct_change: number;
  amount: number;
  observed_at: string;
  observation_score: number;
  rank: number;
  factors: Record<string, number>;
  risk_flags: string[];
  gate_status: "research_only";
};
export type CandidateFeed = {
  provider: string;
  status: RealtimeStatus;
  market_session: MarketSession;
  message: string;
  methodology: string;
  refresh_seconds: number;
  universe_count: number;
  rankable_count: number;
  fetched_at: string | null;
  candidates: ResearchCandidate[];
};

export type RefreshRun = {
  run_id: string;
  status: "queued" | "running" | "success" | "failed";
  step: "queued" | "ingestion" | "quality" | "features" | "inference" | "publish" | "complete" | "failed";
  progress: number;
  started_at: string | null;
  finished_at: string | null;
  data_version: string | null;
  prediction_status: string | null;
  error: string | null;
};

export type MarketQuoteItem = {
  symbol: string;
  name: string;
  industry: string;
  exchange: "SH" | "SZ" | "BJ";
  close: number | null;
  pct_change: number | null;
  amount: number;
  turnover_rate: number | null;
  volume_ratio: number | null;
  amplitude: number | null;
  trade_status: string;
  risk_flags: string[];
};

export type MarketSnapshot = {
  status: "live" | "stale" | "not_available" | "upstream_error";
  market_session: MarketSession;
  message: string;
  trade_date: string | null;
  fetched_at: string | null;
  duration_seconds: number | null;
  universe_count: number;
  trading_count: number;
  advancing_count: number;
  declining_count: number;
  total_amount: number;
  quotes: MarketQuoteItem[];
};

export type PredictionCandidate = {
  symbol: string; name: string; industry: string; exchange: string; rank: number;
  close: number; art_score: number; score_percentile: number; stability_score: number;
  momentum_1d: number | null; momentum_3d: number | null; momentum_5d: number | null; momentum_10d: number | null;
  momentum_20d: number | null; momentum_60d: number | null;
  volatility_20d: number | null; liquidity_score: number | null; volume_progress: number | null;
  patchtst_contribution: number | null; tft_contribution: number | null; ranker_score: number;
  contributions: Record<string, number>; risk_flags: string[];
};

export type PredictionFeed = {
  status: "published" | "blocked" | "stale" | "failed";
  message: string;
  prediction_at: string | null;
  target_trade_date: string | null;
  eligible_count: number;
  context: {hmm_state: string | null; hmm_probabilities: Record<string, number>; tft_style: string | null; tft_probabilities: Record<string, number>};
  candidates: PredictionCandidate[];
  prediction_mode: "long_term" | "short_term";
  training_window_days: number | null;
  validation_days: number;
  experimental: boolean;
  warnings: string[];
};

export type TrainingRun = {
  run_id: string; status: string; pid?: number | null; stage: string | null; progress: number; current_symbol: string | null;
  stage_label?: string | null; stage_completed_items?: number; stage_total_items?: number;
  completed_symbols: number; total_symbols: number; daily_rows: number; intraday_rows: number; retries: number;
  message: string | null; started_at: string | null; finished_at: string | null;
  duration_seconds: number | null; error: string | null; mode: "long_term" | "short_term";
  heartbeat_at?: string | null; last_success_at?: string | null; provider?: string | null;
  failed_items?: number; memory_rss_mb?: number | null; elapsed_seconds?: number | null;
  eta_seconds?: number | null; cancel_requested?: boolean;
};

export type ComponentHealth = {
  component: "hmm" | "tft" | "patchtst" | "ranker";
  display_name: string;
  status: "not_started" | "running" | "healthy" | "warning" | "failed" | "blocked";
  version: string | null;
  progress: number;
  last_trained_at: string | null;
  metrics: Record<string, string | number | boolean | number[]>;
  checkpoint: string | null;
  artifact_sha256: string | null;
  error: string | null;
};

export type ModelMonitoring = {
  status: "idle" | "training" | "warning" | "failed" | "blocked";
  message: string;
  data_coverage_pct: number;
  drift_psi: number | null;
  latest_prediction_status: string;
  daily_history_days: number;
  intraday_history_days: number;
  intraday_coverage_pct: number;
  components: ComponentHealth[];
  ensemble: {
    status: string; champion_version: string | null; weight_sum: number | null;
    lagged_rank_ic: number | null; horizon_consistent: boolean;
    publish_gate_passed: boolean; blocked_reasons: string[];
  };
  source_checks: Array<{
    provider: string; datasets: string[]; status: "ready" | "partial" | "unavailable"; message: string;
  }>;
  training_runs: TrainingRun[];
  mode: "long_term" | "short_term";
  experimental: boolean;
};

export type WeeklyBreadthPoint = {
  trade_date: string; advancing: number; declining: number; unchanged: number;
  total_amount: number; mean_return_pct: number;
};
export type WeeklyReturnBucket = {label: string; lower: number | null; upper: number | null; count: number};
export type WeeklyIndustrySummary = {
  industry: string; median_return_pct: number; advancing_ratio_pct: number; stock_count: number;
  median_return_5d_pct: number | null; median_return_20d_pct: number | null; median_return_60d_pct: number | null;
  advancing_ratio_5d_pct: number | null; advancing_ratio_20d_pct: number | null; advancing_ratio_60d_pct: number | null;
  covered_stock_count: number; trend: "up" | "down" | "mixed" | "insufficient"; trend_label: string;
  composite_score: number; recommendation_tier: "focus" | "neutral" | "avoid" | "insufficient";
  recommendation_label: string; confidence_pct: number; recommendation_reasons: string[];
};
export type WeeklyMarketReport = {
  universe_count: number; weekly_median_return_pct: number; advancing_count: number;
  declining_count: number; unchanged_count: number; total_amount: number;
  daily_return_volatility_pct: number; breadth: WeeklyBreadthPoint[];
  return_distribution: WeeklyReturnBucket[]; industries: WeeklyIndustrySummary[];
};
export type WeeklyRankingItem = {
  symbol: string; name: string; industry: string; exchange: string; rank: number; close: number;
  art_score: number; score_percentile: number; strategy_score: number | null; total_score: number | null;
  stability_score: number; weekly_return_pct: number | null;
  contributions: Record<string, number>; strategy_adjustments: Record<string, number>; risk_flags: string[];
  price_guidance: {
    status: "ready" | "watch" | "blocked"; horizon_days: number; valid_trade_date: string | null;
    buy_price_low: number | null; buy_price_high: number | null; sell_price_low: number | null; sell_price_high: number | null;
    expected_gain_low_pct: number | null; expected_gain_high_pct: number | null; stop_loss_price: number | null;
    positive_return_probability: number | null; target_hit_probability: number | null; risk_reward_ratio: number | null;
    calibration_sample_size: number; methodology_version: string; is_reference_value?: boolean;
    reference_method?: "industry_positive_quantiles" | "global_positive_quantiles" | "atr_volatility" | null;
    blocked_reasons: string[];
  };
};
export type WeeklySectorRanking = {
  industry: string; trend: "up" | "down" | "mixed" | "insufficient"; trend_label: string;
  stock_count: number; eligible_count: number; candidates: WeeklyRankingItem[];
};
export type WeeklyRankingReport = {
  status: "published" | "blocked"; message: string; scoring_date: string; target_trade_date: string | null;
  model_version: string; experimental: boolean; training_rows: number; validation_rank_ic: number | null;
  art_weight: number; strategy_weight: number; strategy_version: string; screened_out_count: number;
  context: PredictionFeed["context"]; blocked_reasons: string[]; warnings: string[]; candidates: WeeklyRankingItem[];
  sector_rankings: WeeklySectorRanking[];
};
export type WeeklyReport = {
  status: "published" | "ranking_blocked" | "stale" | "not_available"; message: string;
  period_start: string | null; period_end: string | null; trading_dates: string[]; generated_at: string;
  market: WeeklyMarketReport | null; ranking: WeeklyRankingReport | null;
  data_sources: Array<{
    provider: "akshare" | "tushare" | "baostock" | "eastmoney" | "local_history";
    role: "primary" | "cross_check" | "fallback";
    status: "verified" | "configuration_required" | "unavailable";
    rows: number; message: string;
  }>;
};
