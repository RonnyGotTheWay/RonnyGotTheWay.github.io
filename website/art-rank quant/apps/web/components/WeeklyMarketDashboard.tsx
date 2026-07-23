"use client";

import {useEffect, useMemo, useState} from "react";
import ReactECharts from "echarts-for-react";

import {PageHeader} from "@/components/PageHeader";
import {apiFetch} from "@/lib/api";
import type {Envelope, VersionMeta, WeeklyReport} from "@/lib/types";

const emptyMeta: VersionMeta = {as_of_date: "—", generated_at: "—", data_version: "weekly-not-available", feature_version: "weekly-market-v2", model_version: "art-rank-blocked", run_id: "none", is_stale: true};
const axis = {axisLabel: {color: "#71837f"}, axisLine: {lineStyle: {color: "#263b37"}}};
const chartGrid = {left: 48, right: 22, top: 45, bottom: 42};

function money(value: number) {
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} 万亿`;
  if (value >= 1e8) return `${(value / 1e8).toFixed(1)} 亿`;
  return value.toLocaleString("zh-CN");
}

export function WeeklyMarketDashboard() {
  const [payload, setPayload] = useState<Envelope<WeeklyReport> | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    void apiFetch("/api/weekly", {cache: "no-store"})
      .then(response => {if (!response.ok) throw new Error(String(response.status)); return response.json();})
      .then(body => {if (active) setPayload(body as Envelope<WeeklyReport>);})
      .catch(() => {if (active) setError("无法读取本机周报 API。");});
    return () => {active = false;};
  }, []);
  const report = payload?.data;
  const market = report?.market;
  const tierCounts = useMemo(() => ({
    focus: market?.industries.filter(item => item.recommendation_tier === "focus").length ?? 0,
    neutral: market?.industries.filter(item => item.recommendation_tier === "neutral").length ?? 0,
    avoid: market?.industries.filter(item => item.recommendation_tier === "avoid").length ?? 0,
  }), [market]);
  const options = useMemo(() => {
    const dates = market?.breadth.map(point => point.trade_date.slice(5)) ?? [];
    const common = {backgroundColor: "transparent", tooltip: {trigger: "axis"}, grid: chartGrid};
    return {
      breadth: {...common, dataZoom: [{type: "inside"}, {type: "slider", bottom: 4, height: 14}], legend: {textStyle: {color: "#90a4a0"}, data: ["上涨", "下跌", "平盘"]}, xAxis: {type: "category", data: dates, ...axis, axisLabel: {color: "#71837f", interval: 6}}, yAxis: {type: "value", ...axis, splitLine: {lineStyle: {color: "#1d2d29"}}}, series: [
        {name: "上涨", type: "line", data: market?.breadth.map(point => point.advancing), lineStyle: {color: "#ff817a"}},
        {name: "下跌", type: "line", data: market?.breadth.map(point => point.declining), lineStyle: {color: "#65d6a9"}},
        {name: "平盘", type: "line", data: market?.breadth.map(point => point.unchanged), lineStyle: {color: "#90a4a0"}},
      ]},
      turnover: {...common, dataZoom: [{type: "inside"}, {type: "slider", bottom: 4, height: 14}], xAxis: {type: "category", data: dates, ...axis, axisLabel: {color: "#71837f", interval: 6}}, yAxis: {type: "value", ...axis, splitLine: {lineStyle: {color: "#1d2d29"}}, axisLabel: {color: "#71837f", formatter: (value: number) => `${(value / 1e8).toFixed(0)}亿`}}, series: [{type: "bar", data: market?.breadth.map(point => point.total_amount), itemStyle: {color: "#58e0b6", borderRadius: [5, 5, 0, 0]}}]},
      distribution: {...common, xAxis: {type: "category", data: market?.return_distribution.map(item => item.label), ...axis, axisLabel: {color: "#71837f", rotate: 28}}, yAxis: {type: "value", ...axis, splitLine: {lineStyle: {color: "#1d2d29"}}}, series: [{type: "bar", data: market?.return_distribution.map(item => item.count), itemStyle: {color: "#f2c66d", borderRadius: [5, 5, 0, 0]}}]},
      industries: {backgroundColor: "transparent", tooltip: {formatter: (params: {name: string; value: number[]}) => `${params.name}<br/>60日收益中位数 ${params.value[2].toFixed(2)}%`}, grid: {left: 35, right: 28, top: 15, bottom: 90}, xAxis: {type: "category", data: market?.industries.map(item => item.industry), ...axis, axisLabel: {color: "#71837f", rotate: 45}}, yAxis: {type: "category", data: ["行业"], show: false}, dataZoom: [{type: "inside"}, {type: "slider", bottom: 12, height: 16}], visualMap: {min: -15, max: 15, calculable: false, orient: "horizontal", left: "center", top: 8, textStyle: {color: "#90a4a0"}, inRange: {color: ["#327e68", "#172422", "#9a473f"]}}, series: [{type: "heatmap", data: market?.industries.map((item, index) => ({name: item.industry, value: [index, 0, item.median_return_pct]}))}]},
    };
  }, [market]);

  return <>
    <PageHeader eyebrow="Sixty-session market review" title="市场两月报告" description="最近60个完整交易日的全市场静态复盘与板块趋势。" meta={payload?.meta ?? emptyMeta} disclaimer="两月研究 · 非实时行情" staleLabel="尚无已发布两月报告"/>
    {(error || !market) && <div className="note candidate-note">{error || report?.message || "正在读取周报…"}</div>}
    {market && <>
      <div className="weekly-period">报告区间 <strong>{report?.period_start} — {report?.period_end}</strong> · {report?.trading_dates.length} 个交易日</div>
      <div className="source-strip" aria-label="周报数据源">{report?.data_sources?.map(source => <div key={source.provider} className={`source-chip ${source.status}`} title={source.message}><strong>{source.provider === "local_history" ? "本地历史" : source.provider.toUpperCase()}</strong><span>{source.status === "verified" ? `已校验${source.rows ? ` · ${source.rows} 行` : ""}` : source.status === "configuration_required" ? "待配置" : "不可用"}</span></div>)}</div>
      <div className="grid weekly-kpis">
        <div className="card"><span>覆盖股票</span><strong>{market.universe_count.toLocaleString("zh-CN")}</strong></div>
        <div className="card"><span>60日收益中位数</span><strong className={market.weekly_median_return_pct >= 0 ? "price-up" : "price-down"}>{market.weekly_median_return_pct.toFixed(2)}%</strong></div>
        <div className="card"><span>上涨 / 下跌</span><strong>{market.advancing_count} / {market.declining_count}</strong></div>
        <div className="card"><span>60日成交额</span><strong>{money(market.total_amount)}</strong></div>
        <div className="card"><span>市场日波动</span><strong>{market.daily_return_volatility_pct.toFixed(2)}%</strong></div>
      </div>
      <div className="weekly-chart-grid section">
        <section className="card"><h2>市场宽度</h2><ReactECharts className="chart" option={options.breadth}/></section>
        <section className="card"><h2>全市场成交额</h2><ReactECharts className="chart" option={options.turnover}/></section>
        <section className="card"><h2>个股60日收益分布</h2><ReactECharts className="chart" option={options.distribution}/></section>
        <section className="card"><h2>行业60日涨跌热力图</h2><ReactECharts className="chart" option={options.industries}/></section>
      </div>
      <section className="card section"><div className="section-head"><div><h2>板块整体趋势</h2><p className="methodology">综合5日、20日与60日收益中位数及60日上涨覆盖率；数据不足时不做方向判断。</p></div></div>
        <div className="tier-summary"><span className="gate-pass">趋势重点关注 {tierCounts.focus}</span><span>中性观察 {tierCounts.neutral}</span><span className="risk-flag">暂时回避 {tierCounts.avoid}</span></div>
        <div className="note candidate-note">板块分级当前只使用已验证价格与市场宽度，置信度最高40%；AKShare资金流和经交叉验证的资讯未齐备时，不展示为完整入场推荐。</div>
        <div className="table-scroll"><table><thead><tr><th>板块</th><th>趋势</th><th>研究分级</th><th className="num">综合分</th><th className="num">置信度</th><th className="num">5日</th><th className="num">20日</th><th className="num">60日</th><th className="num">60日上涨占比</th><th className="num">覆盖</th></tr></thead><tbody>
          {market.industries.map(item => <tr key={item.industry}><td><strong title={item.recommendation_reasons.join("\n")}>{item.industry}</strong></td><td><span className={`status-badge ${item.trend === "up" ? "healthy" : item.trend === "down" ? "failed" : "warning"}`}>{item.trend_label}</span></td><td><span className={`status-badge ${item.recommendation_tier === "focus" ? "healthy" : item.recommendation_tier === "avoid" ? "failed" : "warning"}`}>{item.recommendation_label}</span></td><td className="num">{item.composite_score.toFixed(1)}</td><td className="num">{item.confidence_pct.toFixed(0)}%</td><td className="num">{item.median_return_5d_pct?.toFixed(2) ?? "—"}%</td><td className="num">{item.median_return_20d_pct?.toFixed(2) ?? "—"}%</td><td className="num">{item.median_return_60d_pct?.toFixed(2) ?? "—"}%</td><td className="num">{item.advancing_ratio_60d_pct?.toFixed(1) ?? "—"}%</td><td className="num">{item.covered_stock_count}/{item.stock_count}</td></tr>)}
        </tbody></table></div>
      </section>
    </>}
  </>;
}
