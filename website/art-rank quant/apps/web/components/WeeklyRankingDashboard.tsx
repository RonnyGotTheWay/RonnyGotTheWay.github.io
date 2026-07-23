"use client";

import {useEffect, useMemo, useState} from "react";

import {PageHeader} from "@/components/PageHeader";
import {apiFetch} from "@/lib/api";
import type {Envelope, VersionMeta, WeeklyRankingItem, WeeklyReport} from "@/lib/types";

const emptyMeta: VersionMeta = {as_of_date: "—", generated_at: "—", data_version: "weekly-not-available", feature_version: "weekly-market-v2", model_version: "art-rank-blocked", run_id: "none", is_stale: true};

function range(low: number | null | undefined, high: number | null | undefined, suffix = "", isReference = false) {
  if (low == null || high == null) return "—";
  return `${low.toFixed(2)}–${high.toFixed(2)}${suffix}${isReference ? "（参考值）" : ""}`;
}

function referenceMethod(method: WeeklyRankingItem["price_guidance"]["reference_method"]) {
  if (method === "industry_positive_quantiles") return "同行业正收益样本分位数";
  if (method === "global_positive_quantiles") return "全市场正收益样本分位数";
  if (method === "atr_volatility") return "个股 ATR 波动区间";
  return "模拟回退";
}

function RankingTable({rows}: {rows: WeeklyRankingItem[]}) {
  return <div className="table-scroll"><table><thead><tr><th>排名 / 股票</th><th>行业</th><th className="num">收盘价</th><th className="num">总评分</th><th className="num">60日涨跌</th><th className="num">研究参考买入</th><th className="num">目标价格</th><th className="num">预期涨幅</th><th>价格校准</th><th>风险</th></tr></thead><tbody>
    {rows.map(item => {
      const guide = item.price_guidance;
      return <tr key={item.symbol}><td><div className="candidate-name"><span className="rank">#{item.rank}</span><div><strong>{item.name}</strong><small>{item.symbol}</small></div></div></td><td>{item.industry}</td><td className="num">{item.close.toFixed(2)}</td><td className="num"><strong className="score">{(item.total_score ?? item.art_score).toFixed(2)}</strong></td><td className={`num ${(item.weekly_return_pct ?? 0) >= 0 ? "price-up" : "price-down"}`}>{item.weekly_return_pct == null ? "—" : `${item.weekly_return_pct.toFixed(2)}%`}</td><td className="num">{range(guide?.buy_price_low, guide?.buy_price_high)}</td><td className="num">{range(guide?.sell_price_low, guide?.sell_price_high, "", guide?.is_reference_value)}</td><td className="num">{range(guide?.expected_gain_low_pct, guide?.expected_gain_high_pct, "%", guide?.is_reference_value)}</td><td><details className="price-detail"><summary className={guide?.status === "ready" ? "gate-pass" : "risk-flag"}>{guide?.status === "ready" ? "可关注入场" : guide?.status === "watch" ? "暂不建议入场" : "校准不足"}</summary><div><span>有效日 {guide?.valid_trade_date ?? "—"}</span><span>止损/失效 {guide?.stop_loss_price?.toFixed(2) ?? "—"}</span><span>正收益概率 {guide?.positive_return_probability == null ? "—" : `${(guide.positive_return_probability * 100).toFixed(1)}%`}</span><span>目标达到概率 {guide?.target_hit_probability == null ? "—" : `${(guide.target_hit_probability * 100).toFixed(1)}%`}</span><span>盈亏比 {guide?.risk_reward_ratio?.toFixed(2) ?? "—"}</span><span>样本 {guide?.calibration_sample_size?.toLocaleString("zh-CN") ?? 0}</span>{guide?.is_reference_value && <><span>参考方法 {referenceMethod(guide.reference_method)}</span><span className="risk-flag">低置信度情景参考，不参与入场门禁。</span></>}{guide?.blocked_reasons.map(reason => <span className="risk-flag" key={reason}>{reason}</span>)}</div></details></td><td>{item.risk_flags.length ? item.risk_flags.map(flag => <span className="risk-flag" key={flag}>{flag}</span>) : <span className="gate-pass">通过</span>}</td></tr>;
    })}
  </tbody></table></div>;
}

export function WeeklyRankingDashboard() {
  const [payload, setPayload] = useState<Envelope<WeeklyReport> | null>(null);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  useEffect(() => {
    let active = true;
    void apiFetch("/api/weekly", {cache: "no-store"})
      .then(response => {if (!response.ok) throw new Error(String(response.status)); return response.json();})
      .then(body => {if (active) setPayload(body as Envelope<WeeklyReport>);})
      .catch(() => {if (active) setError("无法读取本机两月报告 API。");});
    return () => {active = false;};
  }, []);
  const report = payload?.data;
  const ranking = report?.ranking;
  const sectors = useMemo(() => (ranking?.sector_rankings ?? []).filter(item => item.industry.includes(query)), [ranking, query]);
  return <>
    <PageHeader eyebrow="ART-Rank sixty-session selection" title="Top 10" description="全市场与各行业板块的下一交易日研究候选，并提供未来5个交易日的历史校准价格区间。" meta={payload?.meta ?? emptyMeta} disclaimer="研究价格区间 · 非交易指令" staleLabel="尚无已发布排名"/>
    {error && <div className="note candidate-note">{error}</div>}
    {ranking?.status === "published" ? <>
      <div className="ranking-summary card"><div><span>评分基准日</span><strong>{ranking.scoring_date}</strong></div><div><span>目标交易日</span><strong>{ranking.target_trade_date ?? "—"}</strong></div><div><span>模型版本</span><strong>{ranking.model_version}</strong></div><div><span>总评分口径</span><strong>ART {(ranking.art_weight * 100).toFixed(0)}% · 策略 {(ranking.strategy_weight * 100).toFixed(0)}%</strong></div></div>
      {ranking.warnings.map(warning => <div className="note candidate-note ranking-warning" key={warning}>{warning}</div>)}
      <section className="card section"><div className="section-head"><div><h2>全市场研究候选 Top 10</h2><p className="methodology">买卖价格来自历史五日路径分位数校准；不足时输出带“（参考值）”的低置信度模拟区间。仅对所示目标交易日有效，开盘越界、停牌或触及涨跌停即失效。</p></div></div><RankingTable rows={ranking.candidates.slice(0, 10)}/></section>
      <section className="section"><div className="section-head"><div><h2>板块 Top 10</h2><p className="methodology">所有板块使用同一总评分，在板块内重新排序；没有合格候选时保留趋势但不展示股票。</p></div><label className="sector-search">板块搜索<input value={query} onChange={event => setQuery(event.target.value)} placeholder="输入行业名称"/></label></div>
        <div className="sector-ranking-list">{sectors.map(sector => <details className="card sector-ranking" key={sector.industry}><summary><strong>{sector.industry}</strong><span className={`status-badge ${sector.trend === "up" ? "healthy" : sector.trend === "down" ? "failed" : "warning"}`}>{sector.trend_label}</span><span>{sector.eligible_count} 只合格候选</span></summary>{sector.candidates.length ? <RankingTable rows={sector.candidates}/> : <div className="empty-state">当前没有通过门禁的板块候选。</div>}</details>)}</div>
      </section>
    </> : <section className="card section blocked-panel"><h2>模型排名门禁开启</h2><p>{ranking?.message || report?.message || "正在读取两月报告…"}</p>{ranking?.blocked_reasons.map(reason => <div className="gate-reason" key={reason}>{reason}</div>)}</section>}
  </>;
}
