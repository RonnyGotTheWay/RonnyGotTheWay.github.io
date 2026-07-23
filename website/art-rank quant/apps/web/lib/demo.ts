import type {Envelope, EquityPoint, Position, StockScore, VersionMeta} from "./types";

export const meta: VersionMeta = {
  as_of_date: "2026-07-13",
  generated_at: "2026-07-13T18:20:00+08:00",
  data_version: "synthetic-v1",
  feature_version: "features-v1",
  model_version: "baseline-v1",
  run_id: "demo-20260713",
  is_stale: false,
};

const sectors = ["科技", "消费", "工业", "医药", "金融", "材料"];
const indices = ["SYN300", "SYN500", "SYN1000"];
export const demoRankings: Envelope<StockScore[]> = {
  meta,
  data: Array.from({length: 40}, (_, index) => ({
    symbol: `SYN${String(index + 1).padStart(4, "0")}`,
    name: `模拟证券 ${String(index + 1).padStart(3, "0")}`,
    index_code: indices[index % 3], sector: sectors[index % 6],
    score: Number((96.8 - index * 1.17).toFixed(2)), rank: index + 1,
    risk_flags: index % 11 === 0 ? ["流动性观察"] : [],
  })),
};

export const demoPositions: Position[] = demoRankings.data.slice(0, 10).map((stock, index) => ({
  symbol: stock.symbol, weight: 0.1, return_pct: Number(((6 - index) * 0.0031).toFixed(4)), sector: stock.sector,
}));

export const demoEquity: EquityPoint[] = Array.from({length: 120}, (_, index) => ({
  date: `D${index + 1}`,
  strategy: Number((1 + index * 0.0013 + Math.sin(index / 9) * 0.014).toFixed(4)),
  benchmark: Number((1 + index * 0.0008 + Math.sin(index / 10) * 0.011).toFixed(4)),
}));

