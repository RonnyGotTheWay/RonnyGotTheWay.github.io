from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from art_rank_quant.api.schemas.common import VersionMetadata
from art_rank_quant.api.schemas.stocks import PredictionContext, PredictionFeed
from art_rank_quant.api.schemas.weekly import (
    WeeklyBreadthPoint,
    WeeklyDataSource,
    WeeklyIndustrySummary,
    WeeklyMarketReport,
    WeeklyPriceGuidance,
    WeeklyRankingItem,
    WeeklyRankingReport,
    WeeklyReport,
    WeeklyReturnBucket,
    WeeklySectorRanking,
)
from art_rank_quant.common.config import Settings, get_settings


class WeeklyReportService:
    """Build and serve immutable sixty-trading-day research reports."""

    REPORT_WINDOW_DAYS = 60
    STRATEGY_WINDOW_DAYS = 5
    ART_WEIGHT = 0.65
    STRATEGY_WEIGHT = 0.35
    STRATEGY_VERSION = "weekly-experience-v1"
    ESTIMATED_TRANSACTION_COST = 0.0015
    MIN_COMPARABLE_SAMPLES = 30
    MIN_GLOBAL_CALIBRATION_SAMPLES = 1_000

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def report_root(self) -> Path:
        return self.settings.artifact_root / "published" / "weekly_reports"

    def generate(
        self, as_of: date | None = None, *, validate_sources: bool = False
    ) -> tuple[WeeklyReport, VersionMetadata, Path]:
        effective_as_of = as_of or self._last_complete_date()
        daily = self._load_daily(effective_as_of)
        dates = daily.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
        if len(dates) < self.REPORT_WINDOW_DAYS:
            raise ValueError(
                f"市场两月报告至少需要 {self.REPORT_WINDOW_DAYS} 个交易日，当前只有 {len(dates)} 个。"
            )
        trading_dates = dates[-self.REPORT_WINDOW_DAYS:]
        frame = daily.filter(pl.col("trade_date").is_in(trading_dates))
        identity = self._identity_map()
        returns = self._weekly_returns(frame, identity)
        market = self._market_report(frame, returns, identity)
        ranking = self._ranking_report(trading_dates[-1], frame, returns, identity, market)
        if ranking.status == "blocked":
            ranking = self._weekly_experimental_ranking(frame, returns, identity, trading_dates[-1], market)
        data_sources = (
            self._validate_sources(trading_dates[0], trading_dates[-1], frame.height)
            if validate_sources
            else [
                WeeklyDataSource(
                    provider="local_history", role="fallback", status="verified", rows=frame.height,
                    message="本地不可变日线缓存；运行发布命令时执行 AKShare/BaoStock 来源检查。",
                )
            ]
        )
        if validate_sources:
            required_providers: tuple[Literal["akshare", "baostock"], ...] = ("akshare", "baostock")
            missing = [
                provider
                for provider in required_providers
                if not any(source.provider == provider and source.status == "verified" for source in data_sources)
            ]
            if missing:
                raise RuntimeError(
                    "市场两月报告双源门禁未通过：" + "、".join(missing) + " 不可用；稳定快照未覆盖。"
                )
        status: Literal["published", "ranking_blocked"] = (
            "published" if ranking.status == "published" else "ranking_blocked"
        )
        generated_at = datetime.now(UTC)
        run_id = f"weekly-{trading_dates[-1]:%Y%m%d}-{uuid4().hex[:8]}"
        data_version = f"weekly-{trading_dates[0]:%Y%m%d}-{trading_dates[-1]:%Y%m%d}"
        model_version = ranking.model_version
        report = WeeklyReport(
            status=status,
            message=(
                "市场两月报告与 ART-Rank Top 10 已发布。"
                if status == "published"
                else "市场两月报告已发布；模型排名被门禁阻塞。"
            ),
            period_start=trading_dates[0],
            period_end=trading_dates[-1],
            trading_dates=trading_dates,
            generated_at=generated_at,
            market=market,
            ranking=ranking,
            data_sources=data_sources,
        )
        metadata = VersionMetadata(
            as_of_date=trading_dates[-1],
            generated_at=generated_at,
            data_version=data_version,
            feature_version="weekly-market-v2",
            model_version=model_version,
            run_id=run_id,
            is_stale=False,
        )
        body = {"meta": metadata.model_dump(mode="json"), "data": report.model_dump(mode="json")}
        self.report_root.mkdir(parents=True, exist_ok=True)
        target = self.report_root / f"snapshot-{run_id}.json"
        self._atomic_json(target, body)
        self._atomic_json(self.report_root / "stable.json", body)
        return report, metadata, target

    @staticmethod
    def _last_complete_date() -> date:
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        if now.weekday() < 5 and (now.hour, now.minute) < (15, 10):
            return now.date() - timedelta(days=1)
        return now.date()

    def _validate_sources(self, start: date, end: date, primary_rows: int) -> list[WeeklyDataSource]:
        from art_rank_quant.data.clients.akshare_sina_history import AkshareSinaHistoryClient
        from art_rank_quant.data.clients.baostock_history import BaoStockHistoryClient
        from art_rank_quant.data.clients.tushare import TushareDailyClient

        checks = [
            WeeklyDataSource(
                provider="local_history", role="fallback", status="verified", rows=primary_rows,
                message="本地不可变日线缓存用于报告聚合；缓存需由双源回填任务生成。",
            )
        ]
        symbol = "600000.SH"
        try:
            frame = AkshareSinaHistoryClient(delay_seconds=0.1).fetch_daily_check(symbol, start, end)
            checks.append(WeeklyDataSource(
                provider="akshare", role="primary", status="verified", rows=frame.height,
                message=f"AKShare/新浪主源完成 {symbol} 周期收盘价校验。",
            ))
        except Exception as exc:
            checks.append(WeeklyDataSource(
                provider="akshare", role="primary", status="unavailable",
                message=f"AKShare 校验失败：{str(exc)[:160]}",
            ))
        token = self.settings.tushare_token
        if token is None or not token.get_secret_value():
            checks.append(WeeklyDataSource(
                provider="tushare", role="cross_check", status="configuration_required",
                message="未配置 TUSHARE_TOKEN，未伪造 Tushare 校验结果。",
            ))
        else:
            try:
                frame = TushareDailyClient(
                    token.get_secret_value(), self.settings.tushare_api_url
                ).fetch_daily_check(symbol, start, end)
                checks.append(WeeklyDataSource(
                    provider="tushare", role="cross_check", status="verified", rows=frame.height,
                    message=f"Tushare Pro 完成 {symbol} 周期收盘价校验。",
                ))
            except Exception as exc:
                checks.append(WeeklyDataSource(
                    provider="tushare", role="cross_check", status="unavailable",
                    message=f"Tushare 校验失败：{str(exc)[:160]}",
                ))
        try:
            with BaoStockHistoryClient() as client:
                frame = client.fetch_daily_check(symbol, start, end)
            checks.append(WeeklyDataSource(
                provider="baostock", role="cross_check", status="verified", rows=frame.height,
                message=f"BaoStock 双源校验完成 {symbol} 周期收盘价检查。",
            ))
        except Exception as exc:
            checks.append(WeeklyDataSource(
                provider="baostock", role="cross_check", status="unavailable",
                message=f"BaoStock 校验失败：{str(exc)[:160]}",
            ))
        return checks

    def latest(self) -> tuple[WeeklyReport, VersionMetadata]:
        path = self.report_root / "stable.json"
        if not path.exists():
            now = datetime.now(UTC)
            return (
                WeeklyReport(
                    status="not_available",
                    message="尚未生成周报，请运行 art-rank weekly-report。",
                    generated_at=now,
                ),
                VersionMetadata(
                    generated_at=now,
                    data_version="weekly-not-available",
                    feature_version="weekly-market-v2",
                    model_version="art-rank-blocked",
                    run_id="none",
                    is_stale=True,
                ),
            )
        body = json.loads(path.read_text(encoding="utf-8"))
        report = WeeklyReport.model_validate(body["data"])
        metadata = VersionMetadata.model_validate(body["meta"])
        latest_source_date = self._latest_source_date()
        stale = bool(
            metadata.is_stale
            or (
                report.period_end is not None
                and latest_source_date is not None
                and report.period_end < latest_source_date
            )
        )
        if stale:
            report = report.model_copy(
                update={"status": "stale", "message": "本地已有更新的日线数据，请重新生成周报。"}
            )
            metadata = metadata.model_copy(update={"is_stale": True})
        return report, metadata

    def _latest_source_date(self) -> date | None:
        dates: list[date] = []
        for path in self.settings.data_root.glob("raw/eastmoney_stock_daily/data_version=*/manifest.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get("max_date")
                if value:
                    dates.append(date.fromisoformat(str(value)))
            except (OSError, ValueError, TypeError):
                continue
        for root in self.settings.data_root.glob("staging/history-*/daily"):
            try:
                dates.append(date.fromisoformat(root.parent.name.rsplit("-", 1)[-1]))
            except ValueError:
                continue
        complete_cutoff = self._last_complete_date()
        complete_dates = [value for value in dates if value <= complete_cutoff]
        return max(complete_dates) if complete_dates else None

    def _load_daily(self, as_of: date | None) -> pl.DataFrame:
        frames: list[pl.LazyFrame] = []
        history = sorted(self.settings.data_root.glob("staging/history-*/daily/*.parquet"))
        if history:
            history_scan = pl.scan_parquet(
                [str(path) for path in history], extra_columns="ignore", missing_columns="insert"
            )
            history_columns = set(history_scan.collect_schema().names())
            frames.append(
                history_scan.select(
                    "symbol", "trade_date", "close", "pct_change", "amount",
                    self._optional_column(history_columns, "open", pl.col("close")).alias("open"),
                    self._optional_column(history_columns, "high", pl.col("close")).alias("high"),
                    self._optional_column(history_columns, "low", pl.col("close")).alias("low"),
                    self._optional_column(history_columns, "pre_close", pl.lit(None, dtype=pl.Float64)).alias(
                        "pre_close"
                    ),
                    self._optional_column(history_columns, "limit_up", pl.lit(None, dtype=pl.Float64)).alias(
                        "limit_up"
                    ),
                    self._optional_column(history_columns, "limit_down", pl.lit(None, dtype=pl.Float64)).alias(
                        "limit_down"
                    ),
                    self._optional_column(history_columns, "trade_status", pl.lit(None, dtype=pl.String)).alias(
                        "trade_status"
                    ),
                    pl.coalesce(
                        [
                            self._optional_column(history_columns, "close_forward_adjusted", pl.col("close")),
                            pl.col("close"),
                        ]
                    ).alias("adjusted_close"),
                )
                .with_columns(pl.lit(1).alias("source_priority"))
            )
        raw = sorted(self.settings.data_root.glob("raw/eastmoney_stock_daily/data_version=*/part-*.parquet"))
        if raw:
            raw_scan = pl.scan_parquet(
                [str(path) for path in raw], extra_columns="ignore", missing_columns="insert"
            )
            raw_columns = set(raw_scan.collect_schema().names())
            frames.append(
                raw_scan.select(
                    "symbol", "trade_date", "close", "pct_change", "amount",
                    self._optional_column(raw_columns, "open", pl.col("close")).alias("open"),
                    self._optional_column(raw_columns, "high", pl.col("close")).alias("high"),
                    self._optional_column(raw_columns, "low", pl.col("close")).alias("low"),
                    self._optional_column(raw_columns, "pre_close", pl.lit(None, dtype=pl.Float64)).alias(
                        "pre_close"
                    ),
                    self._optional_column(raw_columns, "limit_up", pl.lit(None, dtype=pl.Float64)).alias(
                        "limit_up"
                    ),
                    self._optional_column(raw_columns, "limit_down", pl.lit(None, dtype=pl.Float64)).alias(
                        "limit_down"
                    ),
                    self._optional_column(raw_columns, "trade_status", pl.lit(None, dtype=pl.String)).alias(
                        "trade_status"
                    ),
                    pl.col("close").alias("adjusted_close"),
                )
                .with_columns(pl.lit(2).alias("source_priority"))
            )
        if not frames:
            raise FileNotFoundError("没有可用于周报的历史日线或已校验日线快照。")
        query = pl.concat(frames, how="vertical_relaxed")
        if as_of is not None:
            query = query.filter(pl.col("trade_date") <= as_of)
        return (
            query.filter(pl.col("close").is_not_null() & (pl.col("close") > 0))
            .sort(["symbol", "trade_date", "source_priority"])
            .unique(["symbol", "trade_date"], keep="last")
            .sort(["symbol", "trade_date"])
            .with_columns(
                pl.col("open").fill_null(pl.col("close")),
                pl.col("high").fill_null(pl.max_horizontal("open", "close")),
                pl.col("low").fill_null(pl.min_horizontal("open", "close")),
                pl.col("pre_close").fill_null(pl.col("close").shift(1).over("symbol")),
                pl.when(pl.col("trade_status").is_null())
                .then(pl.when(pl.col("amount") > 0).then(pl.lit("trading")).otherwise(pl.lit("suspended")))
                .otherwise(pl.col("trade_status"))
                .alias("trade_status"),
            )
            .collect()
        )

    @staticmethod
    def _optional_column(columns: set[str], name: str, fallback: pl.Expr) -> pl.Expr:
        return pl.col(name) if name in columns else fallback

    def _identity_map(self) -> dict[str, tuple[str, str]]:
        path = self.settings.artifact_root / "published" / "market" / "stable.json"
        if not path.exists():
            return {}
        body = json.loads(path.read_text(encoding="utf-8"))
        return {
            str(row["symbol"]): (str(row.get("name") or row["symbol"]), str(row.get("industry") or "未分类"))
            for row in body.get("data", {}).get("quotes", [])
        }

    @staticmethod
    def _weekly_returns(frame: pl.DataFrame, identity: dict[str, tuple[str, str]]) -> pl.DataFrame:
        result = (
            frame.sort(["symbol", "trade_date"])
            .group_by("symbol")
            .agg(
                pl.first("adjusted_close").alias("first_close"),
                pl.last("adjusted_close").alias("last_close"),
                pl.len().alias("observations"),
            )
            .filter((pl.col("observations") >= 2) & (pl.col("first_close") > 0))
            .with_columns(((pl.col("last_close") / pl.col("first_close") - 1) * 100).alias("weekly_return_pct"))
        )
        names = pl.DataFrame(
            [{"symbol": symbol, "name": value[0], "industry": value[1]} for symbol, value in identity.items()],
            schema={"symbol": pl.String, "name": pl.String, "industry": pl.String},
        )
        return result.join(names, on="symbol", how="left").with_columns(
            pl.col("name").fill_null(pl.col("symbol")), pl.col("industry").fill_null("未分类")
        )

    def _market_report(
        self,
        frame: pl.DataFrame,
        returns: pl.DataFrame,
        identity: dict[str, tuple[str, str]],
    ) -> WeeklyMarketReport:
        breadth_frame = (
            frame.group_by("trade_date")
            .agg(
                (pl.col("pct_change") > 0).sum().alias("advancing"),
                (pl.col("pct_change") < 0).sum().alias("declining"),
                (pl.col("pct_change") == 0).sum().alias("unchanged"),
                pl.col("amount").sum().alias("total_amount"),
                pl.col("pct_change").mean().alias("mean_return_pct"),
            )
            .sort("trade_date")
        )
        breadth = [WeeklyBreadthPoint.model_validate(row) for row in breadth_frame.to_dicts()]
        values = returns["weekly_return_pct"]
        horizon_returns: dict[int, pl.DataFrame] = {}
        trading_dates = sorted(cast(list[date], frame["trade_date"].unique().to_list()))
        for horizon in (5, 20, 60):
            selected = trading_dates[-horizon:]
            horizon_returns[horizon] = self._weekly_returns(
                frame.filter(pl.col("trade_date").is_in(selected)), identity
            ).rename({"weekly_return_pct": f"return_{horizon}d_pct"})
        industries = horizon_returns[60].select(
            "symbol", "industry", "return_60d_pct", pl.col("observations").alias("observations_60d")
        )
        for horizon in (5, 20):
            industries = industries.join(
                horizon_returns[horizon].select("symbol", f"return_{horizon}d_pct"),
                on="symbol",
                how="left",
            )
        industries = (
            industries.group_by("industry")
            .agg(
                pl.col("return_60d_pct").median().alias("median_return_pct"),
                ((pl.col("return_60d_pct") > 0).mean() * 100).alias("advancing_ratio_pct"),
                pl.col("return_5d_pct").median().alias("median_return_5d_pct"),
                pl.col("return_20d_pct").median().alias("median_return_20d_pct"),
                pl.col("return_60d_pct").median().alias("median_return_60d_pct"),
                ((pl.col("return_5d_pct") > 0).mean() * 100).alias("advancing_ratio_5d_pct"),
                ((pl.col("return_20d_pct") > 0).mean() * 100).alias("advancing_ratio_20d_pct"),
                ((pl.col("return_60d_pct") > 0).mean() * 100).alias("advancing_ratio_60d_pct"),
                pl.len().alias("stock_count"),
                (pl.col("observations_60d") == self.REPORT_WINDOW_DAYS).sum().alias("covered_stock_count"),
            )
            .sort("stock_count", descending=True)
        )
        industry_rows = []
        for row in industries.to_dicts():
            medium = float(row.get("median_return_20d_pct") or 0.0)
            long = float(row.get("median_return_60d_pct") or 0.0)
            short = float(row.get("median_return_5d_pct") or 0.0)
            breadth_60 = float(row.get("advancing_ratio_60d_pct") or 0.0)
            if int(row.get("covered_stock_count") or 0) == 0:
                trend, label = "insufficient", "数据不足"
            elif medium > 0 and long > 0 and breadth_60 >= 50:
                trend = "up"
                label = "上行（短期回调）" if short < 0 else "上行"
            elif medium < 0 and long < 0 and breadth_60 < 50:
                trend = "down"
                label = "下行（短期反弹）" if short > 0 else "下行"
            else:
                trend, label = "mixed", "震荡分化"
            industry_rows.append({**row, "trend": trend, "trend_label": label})
        if industry_rows:
            def metric_values(name: str) -> np.ndarray:
                return np.asarray([float(row.get(name) or 0) for row in industry_rows])

            return_strength = (
                0.20 * self._percentiles(metric_values("median_return_5d_pct"))
                + 0.30 * self._percentiles(metric_values("median_return_20d_pct"))
                + 0.50 * self._percentiles(metric_values("median_return_60d_pct"))
            )
            breadth_strength = (
                0.20 * self._percentiles(metric_values("advancing_ratio_5d_pct"))
                + 0.30 * self._percentiles(metric_values("advancing_ratio_20d_pct"))
                + 0.50 * self._percentiles(metric_values("advancing_ratio_60d_pct"))
            )
            partial_scores = 0.625 * return_strength + 0.375 * breadth_strength
            for index, row in enumerate(industry_rows):
                coverage = int(row.get("covered_stock_count") or 0) / max(int(row.get("stock_count") or 0), 1)
                score = float(partial_scores[index])
                trend = str(row["trend"])
                if coverage < 0.70:
                    tier, recommendation = "insufficient", "数据不足"
                elif score >= 70 and trend == "up":
                    tier, recommendation = "focus", "趋势重点关注"
                elif score < 40 or trend == "down":
                    tier, recommendation = "avoid", "暂时回避"
                else:
                    tier, recommendation = "neutral", "中性观察"
                reasons = [
                    f"5/20/60日收益中位数为 {float(row.get('median_return_5d_pct') or 0):.2f}% / "
                    f"{float(row.get('median_return_20d_pct') or 0):.2f}% / "
                    f"{float(row.get('median_return_60d_pct') or 0):.2f}%",
                    f"60日上涨占比 {float(row.get('advancing_ratio_60d_pct') or 0):.1f}%",
                    "当前分级仅使用价格与市场宽度；资金和资讯证据缺失时置信度上限为40%。",
                ]
                row.update(
                    composite_score=round(score, 2),
                    recommendation_tier=tier,
                    recommendation_label=recommendation,
                    confidence_pct=round(min(40.0, coverage * 40), 1),
                    recommendation_reasons=reasons,
                )
        return WeeklyMarketReport(
            universe_count=returns.height,
            weekly_median_return_pct=float(cast(float, values.median() or 0.0)),
            advancing_count=int((values > 0).sum()),
            declining_count=int((values < 0).sum()),
            unchanged_count=int((values == 0).sum()),
            total_amount=float(frame["amount"].sum() or 0),
            daily_return_volatility_pct=float(cast(float, breadth_frame["mean_return_pct"].std() or 0.0)),
            breadth=breadth,
            return_distribution=self._distribution(values.to_list()),
            industries=[WeeklyIndustrySummary.model_validate(row) for row in industry_rows],
        )

    @staticmethod
    def _distribution(values: list[float]) -> list[WeeklyReturnBucket]:
        bounds: list[tuple[str, float | None, float | None]] = [
            ("≤ -10%", None, -10), ("-10% ~ -5%", -10, -5), ("-5% ~ -2%", -5, -2),
            ("-2% ~ 0%", -2, 0), ("0% ~ 2%", 0, 2), ("2% ~ 5%", 2, 5),
            ("5% ~ 10%", 5, 10), ("> 10%", 10, None),
        ]
        buckets: list[WeeklyReturnBucket] = []
        for label, lower, upper in bounds:
            count = sum(
                (lower is None or value > lower) and (upper is None or value <= upper)
                for value in values
            )
            buckets.append(WeeklyReturnBucket(label=label, lower=lower, upper=upper, count=count))
        return buckets

    def _ranking_report(
        self,
        scoring_date: date,
        frame: pl.DataFrame,
        returns: pl.DataFrame,
        identity: dict[str, tuple[str, str]],
        market: WeeklyMarketReport,
    ) -> WeeklyRankingReport:
        path = self.settings.artifact_root / "published" / "predictions" / "stable.json"
        if not path.exists():
            return WeeklyRankingReport(
                status="blocked", message="尚无通过样本外门禁的 ART-Rank 正式预测。", scoring_date=scoring_date,
                model_version="art-rank-blocked", blocked_reasons=["缺少已发布的 ART-Rank champion 预测。"],
            )
        body = json.loads(path.read_text(encoding="utf-8"))
        metadata = VersionMetadata.model_validate(body["meta"])
        feed = PredictionFeed.model_validate(body["data"])
        if feed.status != "published" or metadata.as_of_date != scoring_date:
            return WeeklyRankingReport(
                status="blocked", message="现有模型评分与本周基准日不一致，拒绝沿用旧排名。", scoring_date=scoring_date,
                model_version=metadata.model_version,
                blocked_reasons=[f"最新正式评分基准日为 {metadata.as_of_date}，本周基准日为 {scoring_date}。"],
            )
        return_map = dict(returns.select("symbol", "weekly_return_pct").iter_rows())
        grouped = frame.sort(["symbol", "trade_date"]).partition_by("symbol", as_dict=True)
        stock_frames = {
            str(key[0] if isinstance(key, tuple) else key): stock for key, stock in grouped.items()
        }
        scored: list[WeeklyRankingItem] = []
        screened_out_count = 0
        for item in feed.candidates:
            stock = stock_frames.get(item.symbol)
            if stock is None or stock.height != self.REPORT_WINDOW_DAYS:
                screened_out_count += 1
                continue
            name, industry = identity.get(item.symbol, (item.name, item.industry))
            strategy_score, strategy_adjustments, hard_flags = self._experience_score(
                stock.tail(self.STRATEGY_WINDOW_DAYS), item.symbol, name
            )
            hard_flags = list(dict.fromkeys([*item.risk_flags, *hard_flags]))
            if hard_flags:
                screened_out_count += 1
                continue
            total_score = self._total_score(item.art_score, strategy_score)
            scored.append(
                WeeklyRankingItem(
                    symbol=item.symbol, name=name, industry=industry, exchange=item.exchange, rank=0,
                    close=item.close, art_score=item.art_score, score_percentile=item.score_percentile,
                    strategy_score=strategy_score, total_score=total_score,
                    stability_score=item.stability_score, weekly_return_pct=return_map.get(item.symbol),
                    contributions=item.contributions, strategy_adjustments=strategy_adjustments,
                )
            )
        scored = self._attach_price_guidance(scored, frame, identity, scoring_date)
        scored.sort(key=lambda item: (-(item.total_score or 0), -item.art_score, item.symbol))
        candidates = [item.model_copy(update={"rank": rank}) for rank, item in enumerate(scored[:10], start=1)]
        if len(candidates) < 10:
            return WeeklyRankingReport(
                status="blocked", message="通过经验策略硬门禁的正式模型候选不足 10 只。",
                scoring_date=scoring_date, target_trade_date=feed.target_trade_date,
                model_version=metadata.model_version, screened_out_count=screened_out_count,
                blocked_reasons=[f"合格候选仅 {len(candidates)} 只。"],
            )
        return WeeklyRankingReport(
            status="published", message="ART-Rank 与交易经验策略总评分已发布。", scoring_date=scoring_date,
            target_trade_date=feed.target_trade_date, model_version=metadata.model_version,
            art_weight=self.ART_WEIGHT, strategy_weight=self.STRATEGY_WEIGHT,
            strategy_version=self.STRATEGY_VERSION, screened_out_count=screened_out_count,
            context=feed.context,
            warnings=[
                "总评分由 65% ART 分数与 35% 经验策略分构成。",
                "价格区间来自历史五日路径校准，仅对下一交易日有效，不构成交易指令。",
                "下一交易日仍需检查停牌、集合竞价封板和实际卖盘；静态报告不替代下单前校验。",
            ],
            candidates=candidates,
            sector_rankings=self._sector_rankings(scored, market),
        )

    def _weekly_experimental_ranking(
        self,
        frame: pl.DataFrame,
        returns: pl.DataFrame,
        identity: dict[str, tuple[str, str]],
        scoring_date: date,
        market: WeeklyMarketReport,
    ) -> WeeklyRankingReport:
        """Fit a leakage-safe, cross-sectional sixty-session scorer and rank the latest close."""
        feature_names = ["当日收益", "窗口动量", "窗口波动", "成交额", "成交额相对值"]
        train_features: list[list[float]] = []
        train_targets: list[float] = []
        validation_features: list[list[float]] = []
        validation_targets: list[float] = []
        latest_rows: list[dict[str, Any]] = []
        grouped = frame.sort(["symbol", "trade_date"]).partition_by("symbol", as_dict=True)
        for key, stock in grouped.items():
            if stock.height != self.REPORT_WINDOW_DAYS:
                continue
            symbol = str(key[0] if isinstance(key, tuple) else key)
            closes = stock["adjusted_close"].to_numpy().astype(float)
            raw_closes = stock["close"].to_numpy().astype(float)
            daily_returns = stock["pct_change"].fill_null(0).to_numpy().astype(float)
            amounts = stock["amount"].fill_null(0).to_numpy().astype(float)
            if not np.isfinite(closes).all() or np.any(closes <= 0) or not np.isfinite(amounts).all():
                continue
            samples = [
                self._weekly_feature_vector(closes, daily_returns, amounts, index)
                for index in range(1, self.REPORT_WINDOW_DAYS)
            ]
            for index in range(1, self.REPORT_WINDOW_DAYS - 2):
                train_features.append(samples[index - 1])
                train_targets.append(float(daily_returns[index + 1]))
            validation_features.append(samples[-2])
            validation_targets.append(float(daily_returns[-1]))
            name, industry = identity.get(symbol, (symbol, "未分类"))
            strategy_score, strategy_adjustments, risk_flags = self._experience_score(
                stock.tail(self.STRATEGY_WINDOW_DAYS), symbol, name
            )
            latest_rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "industry": industry,
                    "close": float(raw_closes[-1]),
                    "features": samples[-1],
                    "risk_flags": risk_flags,
                    "strategy_score": strategy_score,
                    "strategy_adjustments": strategy_adjustments,
                    "volatility": float(np.std(daily_returns, ddof=1)),
                }
            )
        if len(train_features) < 1_000 or len(validation_features) < 500:
            return WeeklyRankingReport(
                status="blocked", message="一周实验评分所需的完整横截面样本不足。", scoring_date=scoring_date,
                target_trade_date=self._next_trade_date(scoring_date), model_version="weekly-ridge-blocked",
                experimental=True, training_rows=len(train_features),
                blocked_reasons=[f"训练样本 {len(train_features)}，验证样本 {len(validation_features)}。"],
            )
        scaler = StandardScaler()
        train_matrix = scaler.fit_transform(np.asarray(train_features, dtype=float))
        model = Ridge(alpha=20.0)
        model.fit(train_matrix, np.asarray(train_targets, dtype=float))
        validation_predictions = model.predict(scaler.transform(np.asarray(validation_features, dtype=float)))
        validation_ic = self._rank_correlation(validation_predictions, np.asarray(validation_targets, dtype=float))

        all_features = train_features + validation_features
        all_targets = train_targets + validation_targets
        final_scaler = StandardScaler()
        final_matrix = final_scaler.fit_transform(np.asarray(all_features, dtype=float))
        final_model = Ridge(alpha=20.0)
        final_model.fit(final_matrix, np.asarray(all_targets, dtype=float))
        eligible = [row for row in latest_rows if not row["risk_flags"]]
        if len(eligible) < 10:
            return WeeklyRankingReport(
                status="blocked", message="通过周度风险与流动性过滤的股票不足 10 只。", scoring_date=scoring_date,
                target_trade_date=self._next_trade_date(scoring_date), model_version="weekly-ridge-blocked",
                experimental=True, training_rows=len(all_features), validation_rank_ic=validation_ic,
                blocked_reasons=[f"合格股票仅 {len(eligible)} 只。"],
            )
        latest_matrix = np.asarray([row["features"] for row in eligible], dtype=float)
        latest_scaled = final_scaler.transform(latest_matrix)
        predictions = final_model.predict(latest_scaled)
        percentiles = self._percentiles(predictions)
        strategy_scores = np.asarray([float(row["strategy_score"]) for row in eligible])
        total_scores = self.ART_WEIGHT * percentiles + self.STRATEGY_WEIGHT * strategy_scores
        order = np.argsort(-total_scores)
        volatilities = np.asarray([float(row["volatility"]) for row in eligible])
        stability = 100.0 - self._percentiles(volatilities)
        return_map = dict(returns.select("symbol", "weekly_return_pct").iter_rows())
        candidates: list[WeeklyRankingItem] = []
        for rank, index in enumerate(order[:10], start=1):
            row = eligible[int(index)]
            contributions = {
                name: float(final_model.coef_[position] * latest_scaled[int(index), position])
                for position, name in enumerate(feature_names)
            }
            symbol = str(row["symbol"])
            candidates.append(
                WeeklyRankingItem(
                    symbol=symbol, name=str(row["name"]), industry=str(row["industry"]),
                    exchange=symbol.rsplit(".", 1)[-1], rank=rank, close=float(row["close"]),
                    art_score=float(percentiles[int(index)]), score_percentile=float(percentiles[int(index)]),
                    strategy_score=float(strategy_scores[int(index)]),
                    total_score=float(total_scores[int(index)]),
                    stability_score=float(stability[int(index)]), weekly_return_pct=return_map.get(symbol),
                    contributions=contributions,
                    strategy_adjustments=dict(row["strategy_adjustments"]),
                )
            )
        all_scored: list[WeeklyRankingItem] = []
        for index in order:
            row = eligible[int(index)]
            symbol = str(row["symbol"])
            contributions = {
                name: float(final_model.coef_[position] * latest_scaled[int(index), position])
                for position, name in enumerate(feature_names)
            }
            all_scored.append(
                WeeklyRankingItem(
                    symbol=symbol,
                    name=str(row["name"]),
                    industry=str(row["industry"]),
                    exchange=symbol.rsplit(".", 1)[-1],
                    rank=0,
                    close=float(row["close"]),
                    art_score=float(percentiles[int(index)]),
                    score_percentile=float(percentiles[int(index)]),
                    strategy_score=float(strategy_scores[int(index)]),
                    total_score=float(total_scores[int(index)]),
                    stability_score=float(stability[int(index)]),
                    weekly_return_pct=return_map.get(symbol),
                    contributions=contributions,
                    strategy_adjustments=dict(row["strategy_adjustments"]),
                )
            )
        all_scored = self._attach_price_guidance(all_scored, frame, identity, scoring_date)
        candidates = [item.model_copy(update={"rank": rank}) for rank, item in enumerate(all_scored[:10], start=1)]
        warnings = [
            "总评分由 65% ART 分数与 35% 经验策略分构成。",
            "已硬性排除触及涨跌停、ST、停牌、低流动性和明显五日下降趋势股票。",
            "使用最近 60 个完整交易日提取短期信息；经验风控仍使用最近 5 日。",
            "实验评分的价格区间仅作历史路径参考，正式 ART 门禁未通过时不视为入场指令。",
            "下一交易日仍需检查停牌、集合竞价封板和实际卖盘；排名不构成必须买入或收益承诺。",
        ]
        if validation_ic <= 0:
            warnings.append("周内留出验证 Rank IC 非正，模型方向性较弱，需谨慎使用。")
        return WeeklyRankingReport(
            status="published", message="两月实验评分已生成，仅供下一交易日研究。", scoring_date=scoring_date,
            target_trade_date=self._next_trade_date(scoring_date), model_version=f"weekly-ridge-{scoring_date:%Y%m%d}",
            experimental=True, training_rows=len(all_features), validation_rank_ic=validation_ic,
            art_weight=self.ART_WEIGHT, strategy_weight=self.STRATEGY_WEIGHT,
            strategy_version=self.STRATEGY_VERSION,
            screened_out_count=len(latest_rows) - len(eligible),
            context=PredictionContext(tft_style="60交易日横截面 Ridge"),
            warnings=warnings,
            candidates=candidates,
            sector_rankings=self._sector_rankings(all_scored, market),
        )

    def _attach_price_guidance(
        self,
        items: list[WeeklyRankingItem],
        frame: pl.DataFrame,
        identity: dict[str, tuple[str, str]],
        scoring_date: date,
    ) -> list[WeeklyRankingItem]:
        """Attach auditable five-session empirical price ranges to ranked stocks."""
        samples_by_industry: dict[str, list[tuple[float, float, float]]] = {}
        global_samples: list[tuple[float, float, float]] = []
        grouped = frame.sort(["symbol", "trade_date"]).partition_by("symbol", as_dict=True)
        stocks: dict[str, pl.DataFrame] = {}
        for key, stock in grouped.items():
            symbol = str(key[0] if isinstance(key, tuple) else key)
            stocks[symbol] = stock
            if stock.height < 21:
                continue
            closes = stock["adjusted_close"].to_numpy().astype(float)
            highs = stock["high"].fill_null(stock["close"]).to_numpy().astype(float)
            lows = stock["low"].fill_null(stock["close"]).to_numpy().astype(float)
            industry = identity.get(symbol, (symbol, "未分类"))[1]
            bucket = samples_by_industry.setdefault(industry, [])
            for index in range(14, stock.height - 5):
                entry = closes[index]
                if not np.isfinite(entry) or entry <= 0:
                    continue
                final_return = float(closes[index + 5] / entry - 1)
                max_gain = float(np.max(highs[index + 1 : index + 6]) / entry - 1)
                max_drawdown = float(np.min(lows[index + 1 : index + 6]) / entry - 1)
                sample = (final_return, max_gain, max_drawdown)
                if np.isfinite(sample).all():
                    bucket.append(sample)
                    global_samples.append(sample)

        result: list[WeeklyRankingItem] = []
        valid_trade_date = self._next_trade_date(scoring_date)
        for item in items:
            current_stock = stocks.get(item.symbol)
            industry_samples = samples_by_industry.get(item.industry, [])
            samples = industry_samples
            if len(samples) < self.MIN_COMPARABLE_SAMPLES:
                samples = global_samples
            has_history = current_stock is not None and current_stock.height > 0
            if has_history:
                assert current_stock is not None
                closes = current_stock["close"].to_numpy().astype(float)
                highs = current_stock["high"].fill_null(current_stock["close"]).to_numpy().astype(float)
                lows = current_stock["low"].fill_null(current_stock["close"]).to_numpy().astype(float)
                latest = current_stock.row(-1, named=True)
            else:
                closes = np.asarray([item.close], dtype=float)
                highs = closes.copy()
                lows = closes.copy()
                latest = {"pre_close": item.close}
            previous = np.concatenate(([closes[0]], closes[:-1]))
            true_ranges = np.maximum(highs - lows, np.maximum(np.abs(highs - previous), np.abs(lows - previous)))
            finite_ranges = true_ranges[np.isfinite(true_ranges)]
            observed_atr = float(np.mean(finite_ranges[-14:])) if finite_ranges.size else 0.0
            close = float(closes[-1])
            atr = max(observed_atr, close * 0.005, 0.01)
            support = float(np.min(lows[-20:])) if lows.size else close
            buy_low = max(support, close - 0.5 * atr)
            buy_high = close + 0.25 * atr
            pre_close = float(latest.get("pre_close") or close)
            limit_ratio = self._price_limit_ratio(item.symbol)
            lower_limit = self._limit_price(pre_close, limit_ratio, direction=-1)
            upper_limit = self._limit_price(pre_close, limit_ratio, direction=1)
            buy_low = round(max(lower_limit, min(buy_low, upper_limit)), 2)
            buy_high = round(max(buy_low, min(buy_high, upper_limit)), 2)
            entry = (buy_low + buy_high) / 2

            calibration_ready = (
                current_stock is not None
                and current_stock.height >= 20
                and len(global_samples) >= self.MIN_GLOBAL_CALIBRATION_SAMPLES
                and len(samples) >= self.MIN_COMPARABLE_SAMPLES
            )
            if not calibration_ready:
                reference_low, reference_high, blocked_reference_method = self._reference_return_range(
                    industry_samples, global_samples, atr, entry
                )
                sell_low, sell_high, gain_low, gain_high = self._target_prices(
                    entry, reference_low, reference_high
                )
                guidance = WeeklyPriceGuidance(
                    status="blocked",
                    valid_trade_date=valid_trade_date,
                    buy_price_low=buy_low,
                    buy_price_high=buy_high,
                    sell_price_low=sell_low,
                    sell_price_high=sell_high,
                    expected_gain_low_pct=gain_low,
                    expected_gain_high_pct=gain_high,
                    calibration_sample_size=len(samples),
                    is_reference_value=True,
                    reference_method=blocked_reference_method,
                    blocked_reasons=[
                        "历史五日校准样本不足；已生成模拟参考值，不参与入场门禁。"
                    ],
                )
                result.append(item.model_copy(update={"price_guidance": guidance}))
                continue

            matrix = np.asarray(samples, dtype=float)
            q50, q75 = np.quantile(matrix[:, 0], [0.50, 0.75])
            positive_probability = float(
                np.mean(matrix[:, 0] > self.ESTIMATED_TRANSACTION_COST)
            )
            reasons: list[str] = []
            is_reference_value = bool(q50 <= self.ESTIMATED_TRANSACTION_COST)
            reference_method: Literal[
                "industry_positive_quantiles", "global_positive_quantiles", "atr_volatility"
            ] | None = None
            if is_reference_value:
                reasons.append("五日收益中位数未覆盖估算交易成本。")
            gate_sell_low = entry * (1 + max(float(q50), 0.0))
            gate_sell_high = entry * (1 + max(float(q75), float(q50), 0.0))
            target_hit_probability = float(np.mean(matrix[:, 1] >= max(float(q50), 0.0)))
            statistical_stop = entry * (1 + float(np.quantile(matrix[:, 2], 0.20)))
            technical_stop = support - 0.25 * atr
            stop = max(lower_limit, min(max(statistical_stop, technical_stop), buy_low - 0.01))
            target_mid = (gate_sell_low + gate_sell_high) / 2
            downside = max(entry - stop, 0.01)
            risk_reward = max(target_mid - entry, 0.0) / downside
            if positive_probability < 0.55:
                reasons.append("扣除成本后的历史正收益概率低于 55%。")
            if target_hit_probability < 0.50:
                reasons.append("历史目标达到概率低于 50%。")
            if risk_reward < 1.5:
                reasons.append("参考盈亏比低于 1.5。")
            status: Literal["ready", "watch", "blocked"] = "ready" if not reasons else "watch"
            display_low = float(q50)
            display_high = float(q75)
            if is_reference_value:
                display_low, display_high, reference_method = self._reference_return_range(
                    industry_samples, global_samples, atr, entry
                )
                reasons.append("目标价格与预期涨幅为模拟参考值，不参与入场门禁。")
            sell_low, sell_high, gain_low, gain_high = self._target_prices(
                entry, display_low, display_high
            )
            guidance = WeeklyPriceGuidance(
                status=status,
                valid_trade_date=valid_trade_date,
                buy_price_low=buy_low,
                buy_price_high=buy_high,
                sell_price_low=sell_low,
                sell_price_high=sell_high,
                expected_gain_low_pct=gain_low,
                expected_gain_high_pct=gain_high,
                stop_loss_price=round(stop, 2),
                positive_return_probability=round(positive_probability, 4),
                target_hit_probability=round(target_hit_probability, 4),
                risk_reward_ratio=round(risk_reward, 3),
                calibration_sample_size=len(samples),
                is_reference_value=is_reference_value,
                reference_method=reference_method,
                blocked_reasons=reasons,
            )
            result.append(item.model_copy(update={"price_guidance": guidance}))
        return result

    def _reference_return_range(
        self,
        industry_samples: list[tuple[float, float, float]],
        global_samples: list[tuple[float, float, float]],
        atr: float,
        entry: float,
    ) -> tuple[
        float,
        float,
        Literal["industry_positive_quantiles", "global_positive_quantiles", "atr_volatility"],
    ]:
        """Return a positive scenario range without changing formal calibration gates."""
        for samples, method in (
            (industry_samples, "industry_positive_quantiles"),
            (global_samples, "global_positive_quantiles"),
        ):
            positive_returns = [
                sample[0]
                for sample in samples
                if sample[0] > self.ESTIMATED_TRANSACTION_COST
            ]
            if len(positive_returns) >= self.MIN_COMPARABLE_SAMPLES:
                q50, q75 = np.quantile(np.asarray(positive_returns, dtype=float), [0.50, 0.75])
                return float(q50), float(q75), cast(
                    Literal["industry_positive_quantiles", "global_positive_quantiles"], method
                )
        atr_ratio = max(atr / max(entry, 0.01), self.ESTIMATED_TRANSACTION_COST)
        return 0.5 * atr_ratio, atr_ratio, "atr_volatility"

    @staticmethod
    def _target_prices(entry: float, low_return: float, high_return: float) -> tuple[float, float, float, float]:
        """Round target prices first, then derive matching percentage gains."""
        sell_low = round(max(entry + 0.01, entry * (1 + max(low_return, 0.0))), 2)
        sell_high = round(max(sell_low, entry * (1 + max(high_return, low_return, 0.0))), 2)
        gain_low = round((sell_low / entry - 1) * 100, 2)
        gain_high = round((sell_high / entry - 1) * 100, 2)
        return sell_low, sell_high, gain_low, gain_high

    @staticmethod
    def _sector_rankings(
        scored: list[WeeklyRankingItem], market: WeeklyMarketReport
    ) -> list[WeeklySectorRanking]:
        trend_map = {item.industry: item for item in market.industries}
        grouped: dict[str, list[WeeklyRankingItem]] = {}
        for item in scored:
            grouped.setdefault(item.industry, []).append(item)
        rankings: list[WeeklySectorRanking] = []
        for industry, summary in sorted(trend_map.items()):
            candidates = sorted(
                grouped.get(industry, []), key=lambda item: (-(item.total_score or 0), -item.art_score, item.symbol)
            )[:10]
            candidates = [item.model_copy(update={"rank": rank}) for rank, item in enumerate(candidates, start=1)]
            rankings.append(
                WeeklySectorRanking(
                    industry=industry,
                    trend=summary.trend,
                    trend_label=summary.trend_label,
                    stock_count=summary.stock_count,
                    eligible_count=len(grouped.get(industry, [])),
                    candidates=candidates,
                )
            )
        return rankings

    def _experience_score(
        self, stock: pl.DataFrame, symbol: str, name: str
    ) -> tuple[float, dict[str, float], list[str]]:
        """Return an explainable 0-100 trading-experience score and hard no-buy flags."""
        stock = stock.sort("trade_date")
        latest = stock.row(-1, named=True)
        closes = stock["adjusted_close"].to_numpy().astype(float)
        daily_returns = stock["pct_change"].fill_null(0).to_numpy().astype(float)
        amounts = stock["amount"].fill_null(0).to_numpy().astype(float)
        hard_flags: list[str] = []

        if "ST" in name.upper():
            hard_flags.append("ST风险")
        if str(latest.get("trade_status") or "unknown") != "trading" or float(amounts[-1]) <= 0:
            hard_flags.append("最新交易日停牌或无成交")
        median_amount = float(np.median(amounts))
        if median_amount < self.settings.candidate_min_amount_cny:
            hard_flags.append("周内流动性不足")

        limit_ratio = self._price_limit_ratio(symbol)
        latest_return = float(daily_returns[-1])
        pre_close = float(latest.get("pre_close") or 0)
        limit_up = float(latest.get("limit_up") or 0)
        limit_down = float(latest.get("limit_down") or 0)
        if max(abs(latest_return), 0) > limit_ratio * 100 + 0.5 and limit_up <= 0:
            hard_flags.append("无涨跌幅限制阶段")
        elif pre_close > 0:
            if limit_up <= 0:
                limit_up = self._limit_price(pre_close, limit_ratio, direction=1)
            if limit_down <= 0:
                limit_down = self._limit_price(pre_close, limit_ratio, direction=-1)
            high = float(latest.get("high") or latest.get("close") or 0)
            low = float(latest.get("low") or latest.get("close") or 0)
            if high >= limit_up - 0.005:
                hard_flags.append("最新交易日触及涨停")
            if low <= limit_down + 0.005:
                hard_flags.append("最新交易日触及跌停")

        weekly_return = round(float((closes[-1] / closes[0] - 1) * 100), 6)
        log_slope_pct = float(np.polyfit(np.arange(len(closes)), np.log(closes), 1)[0] * 100)
        down_streak = self._longest_streak(daily_returns < 0)
        if weekly_return <= -8 and log_slope_pct <= -1 and down_streak >= 3:
            hard_flags.append("明显五日下降趋势")

        adjustments: dict[str, float] = {}

        def penalize(label: str, points: float) -> None:
            adjustments[label] = -float(points)

        if weekly_return >= 30:
            penalize("五日涨幅≥30%", 35)
        elif weekly_return >= 20:
            penalize("五日涨幅≥20%", 25)
        elif weekly_return >= 12:
            penalize("五日涨幅≥12%", 15)
        elif weekly_return >= 8:
            penalize("五日涨幅≥8%", 8)

        if latest_return >= 7:
            penalize("最新日涨幅≥7%", 12)
        elif latest_return >= 5:
            penalize("最新日涨幅≥5%", 6)

        big_up_streak = self._longest_streak(daily_returns >= 3)
        if big_up_streak >= 3:
            penalize("连续3日以上大涨", 15)
        elif big_up_streak == 2:
            penalize("连续2日大涨", 8)
        if int((daily_returns >= 5).sum()) >= 2:
            penalize("五日内多次大涨", 10)

        below_average = bool(closes[-1] < float(np.mean(closes)))
        if weekly_return <= -5 and log_slope_pct < 0 and below_average:
            penalize("五日下降趋势", 20)
        elif weekly_return < 0 and log_slope_pct < 0 and below_average:
            penalize("短期趋势偏弱", 8)
        if down_streak >= 3:
            penalize("连续3日以上下跌", 12)

        volatility = float(np.std(daily_returns, ddof=1))
        if volatility >= 5:
            penalize("五日波动过高", 10)
        elif volatility >= 3:
            penalize("五日波动偏高", 5)

        minimum_amount = self.settings.candidate_min_amount_cny
        if median_amount < minimum_amount * 2:
            penalize("流动性接近门槛", 10)
        elif median_amount < minimum_amount * 4:
            penalize("流动性一般", 5)

        score = float(np.clip(100 + sum(adjustments.values()), 0, 100))
        return score, adjustments, list(dict.fromkeys(hard_flags))

    @classmethod
    def _total_score(cls, art_score: float, strategy_score: float) -> float:
        art = float(np.clip(art_score, 0, 100))
        strategy = float(np.clip(strategy_score, 0, 100))
        return cls.ART_WEIGHT * art + cls.STRATEGY_WEIGHT * strategy

    @staticmethod
    def _longest_streak(mask: np.ndarray) -> int:
        longest = 0
        current = 0
        for value in mask:
            current = current + 1 if bool(value) else 0
            longest = max(longest, current)
        return longest

    @staticmethod
    def _price_limit_ratio(symbol: str) -> float:
        code, _, exchange = symbol.partition(".")
        if exchange == "BJ" or code.startswith(("4", "8", "92")):
            return 0.30
        if code.startswith(("300", "301", "688", "689")):
            return 0.20
        return 0.10

    @staticmethod
    def _limit_price(pre_close: float, ratio: float, *, direction: int) -> float:
        multiplier = Decimal("1") + Decimal(str(ratio)) * Decimal(direction)
        return float(
            (Decimal(str(pre_close)) * multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

    @staticmethod
    def _weekly_feature_vector(
        closes: np.ndarray, daily_returns: np.ndarray, amounts: np.ndarray, index: int
    ) -> list[float]:
        history_amount = amounts[: index + 1]
        median_amount = max(float(np.median(history_amount)), 1.0)
        return [
            float(daily_returns[index]),
            float((closes[index] / closes[0] - 1) * 100),
            float(np.std(daily_returns[: index + 1], ddof=1)),
            float(np.log1p(max(amounts[index], 0))),
            float(amounts[index] / median_amount),
        ]

    @staticmethod
    def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
        if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
            return 0.0
        left_rank = np.argsort(np.argsort(left)).astype(float)
        right_rank = np.argsort(np.argsort(right)).astype(float)
        return float(np.corrcoef(left_rank, right_rank)[0, 1])

    @staticmethod
    def _percentiles(values: np.ndarray) -> np.ndarray:
        if values.size <= 1:
            return np.full(values.shape, 50.0)
        ranks = np.argsort(np.argsort(values)).astype(float)
        return ranks / (values.size - 1) * 100.0

    @staticmethod
    def _next_trade_date(value: date) -> date:
        target = value + timedelta(days=1)
        while target.weekday() >= 5:
            target += timedelta(days=1)
        return target

    @staticmethod
    def _atomic_json(path: Path, body: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


@lru_cache
def get_weekly_report_service() -> WeeklyReportService:
    return WeeklyReportService(get_settings())
