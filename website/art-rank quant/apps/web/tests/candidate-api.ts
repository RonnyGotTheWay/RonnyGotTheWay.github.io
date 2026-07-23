import type {Page} from "@playwright/test";

const meta = {
  as_of_date: "2026-07-15",
  generated_at: "2026-07-15T15:10:00+08:00",
  data_version: "eastmoney-contract-test",
  feature_version: "market-snapshot-v2",
  model_version: "observation-only",
  run_id: "refresh-contract",
  is_stale: false,
};

const weeklyTradingDates = (() => {
  const values: string[] = [];
  const cursor = new Date("2026-07-15T00:00:00Z");
  while (values.length < 60) {
    const day = cursor.getUTCDay();
    if (day !== 0 && day !== 6) values.push(cursor.toISOString().slice(0, 10));
    cursor.setUTCDate(cursor.getUTCDate() - 1);
  }
  return values.reverse();
})();

const weeklyCandidates = Array.from({length: 10}, (_, index) => ({symbol: `60000${index}.SH`, name: `测试股票${index + 1}`,
  industry: index < 5 ? "银行" : "电子", exchange: "SH", rank: index + 1, close: 10 + index, art_score: 99 - index,
  score_percentile: 99 - index, strategy_score: 92 - index, total_score: 96.55 - index,
  stability_score: 90 - index, weekly_return_pct: 12.5 - index * 0.8,
  contributions: {LambdaRank: 0.8, PatchTST: 0.4},
  strategy_adjustments: index === 0 ? {"流动性一般": -5} : {}, risk_flags: [],
  price_guidance: {status: index === 9 ? "watch" : "ready", horizon_days: 5, valid_trade_date: "2026-07-16",
    buy_price_low: 9.8 + index, buy_price_high: 10.2 + index, sell_price_low: 10.7 + index,
    sell_price_high: 11.2 + index, expected_gain_low_pct: 6.0, expected_gain_high_pct: 10.9,
    stop_loss_price: 9.3 + index, positive_return_probability: 0.62, target_hit_probability: 0.58,
    risk_reward_ratio: 1.8, calibration_sample_size: 1200, methodology_version: "empirical-5d-v1",
    is_reference_value: index === 9, reference_method: index === 9 ? "industry_positive_quantiles" : null,
    blocked_reasons: index === 9 ? ["参考盈亏比低于 1.5。"] : []},
}));

export async function mockThreePageApi(page: Page) {
  await page.route("**/api/weekly", route => route.fulfill({
    contentType: "application/json",
    json: {meta: {...meta, feature_version: "weekly-market-v2", model_version: "art-rank-weekly-test"}, data: {
      status: "published", message: "两月报告已发布。", period_start: weeklyTradingDates[0], period_end: "2026-07-15",
      trading_dates: weeklyTradingDates,
      generated_at: "2026-07-15T16:00:00+08:00",
      data_sources: [
        {provider: "local_history", role: "primary", status: "verified", rows: 25000, message: "本地历史"},
        {provider: "akshare", role: "cross_check", status: "verified", rows: 5, message: "已校验"},
        {provider: "tushare", role: "cross_check", status: "configuration_required", rows: 0, message: "待配置"},
        {provider: "baostock", role: "cross_check", status: "verified", rows: 5, message: "已校验"},
      ],
      market: {
        universe_count: 5534, weekly_median_return_pct: 1.25, advancing_count: 3210,
        declining_count: 2200, unchanged_count: 124, total_amount: 5234000000000,
        daily_return_volatility_pct: 0.82,
        breadth: weeklyTradingDates.map((trade_date, index) => ({trade_date,
          advancing: 2500 + index * 100, declining: 2800 - index * 100, unchanged: 100,
          total_amount: 900000000000 + index * 30000000000, mean_return_pct: index * 0.1})),
        return_distribution: ["≤ -10%", "-10% ~ -5%", "-5% ~ -2%", "-2% ~ 0%", "0% ~ 2%", "2% ~ 5%", "5% ~ 10%", "> 10%"].map((label, index) => ({label, lower: null, upper: null, count: 100 + index * 30})),
        industries: [{industry: "银行", median_return_pct: 8.8, advancing_ratio_pct: 70, stock_count: 42, median_return_5d_pct: 1.2, median_return_20d_pct: 3.4, median_return_60d_pct: 8.8, advancing_ratio_5d_pct: 62, advancing_ratio_20d_pct: 66, advancing_ratio_60d_pct: 70, covered_stock_count: 42, trend: "up", trend_label: "上行", composite_score: 82, recommendation_tier: "focus", recommendation_label: "趋势重点关注", confidence_pct: 40, recommendation_reasons: ["趋势上行"]},
          {industry: "电子", median_return_pct: -2.4, advancing_ratio_pct: 48, stock_count: 320, median_return_5d_pct: 0.8, median_return_20d_pct: -1.2, median_return_60d_pct: -2.4, advancing_ratio_5d_pct: 55, advancing_ratio_20d_pct: 46, advancing_ratio_60d_pct: 48, covered_stock_count: 315, trend: "down", trend_label: "下行（短期反弹）", composite_score: 32, recommendation_tier: "avoid", recommendation_label: "暂时回避", confidence_pct: 39, recommendation_reasons: ["趋势下行"]}],
      },
      ranking: {status: "published", message: "已发布", scoring_date: "2026-07-15",
        target_trade_date: "2026-07-16", model_version: "art-rank-weekly-test", experimental: true,
        training_rows: 12000, validation_rank_ic: 0.08, art_weight: 0.65, strategy_weight: 0.35,
        strategy_version: "weekly-experience-v1", screened_out_count: 17,
        blocked_reasons: [], warnings: ["使用最近60个完整交易日。"],
        context: {hmm_state: "状态1", hmm_probabilities: {状态1: 0.7}, tft_style: "成长", tft_probabilities: {}},
        candidates: weeklyCandidates,
        sector_rankings: [
          {industry: "银行", trend: "up", trend_label: "上行", stock_count: 42, eligible_count: 5, candidates: weeklyCandidates.slice(0, 5)},
          {industry: "电子", trend: "down", trend_label: "下行（短期反弹）", stock_count: 320, eligible_count: 5, candidates: weeklyCandidates.slice(5).map((item, index) => ({...item, rank: index + 1}))},
        ],
      },
    }},
  }));
  await page.route("**/api/v1/market/snapshot/latest", route => route.fulfill({
    contentType: "application/json",
    json: {meta, data: {
      status: "live", market_session: "closed", message: "全市场快照已通过完整性检查。",
      trade_date: "2026-07-15", fetched_at: "2026-07-15T15:10:00+08:00", duration_seconds: 18.2,
      universe_count: 5534, trading_count: 5523, advancing_count: 2980, declining_count: 2410,
      total_amount: 1234000000000,
      quotes: [{symbol: "600000.SH", name: "浦发银行", industry: "银行", exchange: "SH", close: 10.3,
        pct_change: 1.2, amount: 80000000, turnover_rate: 0.7, volume_ratio: 1.1, amplitude: 2.3,
        trade_status: "trading", risk_flags: []}],
    }},
  }));
  await page.route("**/api/v1/market/refresh", route => route.fulfill({
    status: 202, contentType: "application/json",
    json: {run_id: "refresh-contract", status: "queued", message: "已提交"},
  }));
  await page.route("**/api/v1/system/runs/*", route => route.fulfill({
    contentType: "application/json",
    json: {run_id: "refresh-contract", status: "success", step: "complete", progress: 100,
      started_at: "2026-07-15T15:09:00+08:00", finished_at: "2026-07-15T15:10:00+08:00",
      data_version: "eastmoney-contract-test", prediction_status: "blocked", error: null},
  }));
  await page.route("**/api/v1/stocks/predictions/latest?*", route => route.fulfill({
    contentType: "application/json",
    json: {meta: {...meta, model_version: "art-rank-blocked", is_stale: true}, data: {
      status: "blocked", message: "正式预测尚未发布。四组件尚未全部达到 healthy。",
      prediction_at: null, target_trade_date: null, eligible_count: 0,
      context: {hmm_state: null, hmm_probabilities: {}, tft_style: null, tft_probabilities: {}}, candidates: [],
    }},
  }));
  await page.route("**/api/v1/models/health", route => route.fulfill({
    contentType: "application/json",
    json: {meta: {...meta, model_version: "art-rank-blocked", is_stale: true}, data: {
      status: "blocked", message: "四组件尚未全部达到 healthy。", data_coverage_pct: 100,
      daily_history_days: 1, intraday_history_days: 0, intraday_coverage_pct: 0,
      drift_psi: null, latest_prediction_status: "blocked",
      components: [
        ["hmm", "HMM 市场状态"], ["tft", "TFT 风格预测"],
        ["patchtst", "PatchTST 时序编码"], ["ranker", "LambdaRank 排序"],
      ].map(([component, display_name]) => ({component, display_name, status: "not_started", version: null,
        progress: 0, last_trained_at: null, metrics: {availability: "no_validated_artifact"},
        checkpoint: null, artifact_sha256: null, error: null})),
      ensemble: {status: "blocked", champion_version: null, weight_sum: null, lagged_rank_ic: null,
        horizon_consistent: false, publish_gate_passed: false,
        blocked_reasons: ["HMM、TFT、PatchTST、LambdaRank 尚未全部达到 healthy。"]}, training_runs: [],
    }},
  }));
  await page.route("**/api/v1/models/bootstrap", route => route.fulfill({
    status: 202, contentType: "application/json",
    json: {run_id: "bootstrap-contract", status: "queued", message: "已启动"},
  }));
  await page.route("**/api/v1/stocks/predictions/short-term/latest?*", route => route.fulfill({
    contentType: "application/json",
    json: {meta: {...meta, feature_version: "short-pit-15d-not-ready", model_version: "art-rank-short-blocked", is_stale: true}, data: {
      status: "blocked", message: "短期实验预测尚未发布。", prediction_at: null, target_trade_date: null,
      eligible_count: 0, context: {hmm_state: null, hmm_probabilities: {}, tft_style: null, tft_probabilities: {}},
      candidates: [], prediction_mode: "short_term", training_window_days: 15, validation_days: 0,
      experimental: true, warnings: ["仅使用15个交易日，统计显著性弱于长期模型。"],
    }},
  }));
  await page.route("**/api/v1/models/short-term/health", route => route.fulfill({
    contentType: "application/json",
    json: {meta: {...meta, model_version: "art-rank-short-blocked", is_stale: true}, data: {
      status: "blocked", message: "短期四组件尚未完成。", data_coverage_pct: 0, drift_psi: null,
      latest_prediction_status: "blocked", daily_history_days: 0, intraday_history_days: 0,
      intraday_coverage_pct: 0, mode: "short_term", experimental: true,
      components: [
        ["hmm", "短期HMM市场状态"], ["tft", "短期TFT收益分位"],
        ["patchtst", "短期PatchTST编码"], ["ranker", "短期LambdaRank排序"],
      ].map(([component, display_name]) => ({component, display_name, status: "not_started", version: null,
        progress: 0, last_trained_at: null, metrics: {availability: "not_ready"}, checkpoint: null,
        artifact_sha256: null, error: null})),
      ensemble: {status: "blocked", champion_version: null, weight_sum: null, lagged_rank_ic: null,
        horizon_consistent: false, publish_gate_passed: false, blocked_reasons: ["共同交易日仅0日，要求15日。"]},
      training_runs: [],
    }},
  }));
  await page.route("**/api/v1/models/short-term/bootstrap", route => route.fulfill({
    status: 202, contentType: "application/json",
    json: {run_id: "short-term-contract", status: "queued", message: "15日实验训练任务已启动。"},
  }));
  await page.route("**/api/v1/models/short-term/runs/*", route => route.fulfill({
    contentType: "application/json",
    json: {run_id: "short-term-contract", status: "running", stage: "daily", progress: 18,
      current_symbol: "600000.SH", completed_symbols: 100, total_symbols: 5534, daily_rows: 1500,
      intraday_rows: 0, retries: 0, message: "并发回填日线", started_at: "2026-07-15T15:00:00+08:00",
      finished_at: null, duration_seconds: null, error: null, mode: "short_term"},
  }));
}
