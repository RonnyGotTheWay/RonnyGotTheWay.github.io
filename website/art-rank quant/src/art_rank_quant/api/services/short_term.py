from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl

from art_rank_quant.api.schemas.common import VersionMetadata
from art_rank_quant.api.schemas.models import (
    ComponentHealth,
    EnsembleHealth,
    ModelMonitoring,
    SourceCheck,
    TrainingRunSummary,
)
from art_rank_quant.api.schemas.stocks import PredictionFeed
from art_rank_quant.api.services.model_bootstrap import load_training_run, training_run_root
from art_rank_quant.common.config import Settings, get_settings


class ShortTermRuntimeService:
    """Reads only the isolated 15-day experimental artifacts."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def root(self) -> Path:
        return self._settings.artifact_root / "short_term"

    @property
    def prediction_root(self) -> Path:
        return self._settings.artifact_root / "published" / "short_term_predictions"

    def latest_prediction(self, limit: int = 20, include_all: bool = False) -> tuple[PredictionFeed, VersionMetadata]:
        path = self.prediction_root / "stable.json"
        monitoring = self.monitoring()
        if path.exists() and monitoring.ensemble.publish_gate_passed:
            body = json.loads(path.read_text(encoding="utf-8"))
            feed = PredictionFeed.model_validate(body["data"])
            if not include_all:
                feed = feed.model_copy(update={"candidates": feed.candidates[:limit]})
            return feed, VersionMetadata.model_validate(body["meta"])
        reasons = monitoring.ensemble.blocked_reasons
        message = "短期实验预测尚未发布。" + (f" {'；'.join(reasons)}" if reasons else "")
        coverage = self._coverage()
        return (
            PredictionFeed(
                status="blocked",
                message=message,
                prediction_mode="short_term",
                training_window_days=15,
                experimental=True,
                warnings=["仅使用15个交易日，统计显著性弱于长期模型。"],
            ),
            VersionMetadata(
                data_version=str(coverage.get("version", "short-history-not-ready")),
                feature_version="short-pit-15d-not-ready",
                model_version="art-rank-short-blocked",
                run_id="none",
                is_stale=True,
            ),
        )

    def monitoring(self) -> ModelMonitoring:
        components = [self._component(name) for name in ("hmm", "tft", "patchtst", "ranker")]
        coverage = self._coverage()
        runs = self.training_runs()
        latest_run = runs[0] if runs else None
        daily_days = int(coverage.get("distinct_trade_dates", 0))
        daily_coverage = float(coverage.get("daily_coverage", 0.0))
        intraday_coverage = float(coverage.get("intraday_coverage", 0.0))
        daily_check_coverage = float(coverage.get("daily_check_coverage", 0.0))
        reasons: list[str] = []
        if latest_run and latest_run.status == "failed":
            reasons.append(f"最近短期任务失败：{latest_run.error or latest_run.message or '未知错误'}")
        if any(component.status != "healthy" for component in components):
            reasons.append("短期HMM、TFT、PatchTST、LambdaRank尚未全部达到healthy。")
        if daily_days < 15:
            reasons.append(f"共同交易日仅{daily_days}日，要求15日。")
        if daily_coverage < self._settings.short_term_daily_coverage:
            reasons.append(
                f"15日日线覆盖率{daily_coverage:.2%}，低于{self._settings.short_term_daily_coverage:.0%}。"
            )
        if intraday_coverage < self._settings.short_term_intraday_coverage:
            reasons.append(
                f"120根30分钟K线覆盖率{intraday_coverage:.2%}，低于{self._settings.short_term_intraday_coverage:.0%}。"
            )
        if daily_check_coverage < self._settings.short_term_daily_coverage:
            reasons.append(
                f"跨源日线校验覆盖率{daily_check_coverage:.2%}，"
                f"低于{self._settings.short_term_daily_coverage:.0%}。"
            )
        ensemble_manifest = self._read_json(self.root / "models" / "ensemble" / "champion.json") or self._read_json(
            self.root / "models" / "ensemble" / "latest.json"
        )
        if ensemble_manifest:
            reasons.extend(
                str(reason) for reason in ensemble_manifest.get("blocked_reasons", []) if str(reason) not in reasons
            )
        gate = bool(ensemble_manifest and ensemble_manifest.get("publish_gate_passed") and not reasons)
        ensemble_status: Literal["healthy", "blocked", "not_started"] = (
            "healthy" if gate else "blocked" if ensemble_manifest else "not_started"
        )
        ensemble = EnsembleHealth(
            status=ensemble_status,
            champion_version=ensemble_manifest.get("version") if ensemble_manifest else None,
            weight_sum=ensemble_manifest.get("weight_sum") if ensemble_manifest else None,
            lagged_rank_ic=ensemble_manifest.get("lagged_rank_ic") if ensemble_manifest else None,
            horizon_consistent=bool(ensemble_manifest and ensemble_manifest.get("horizon_consistent")),
            publish_gate_passed=gate,
            blocked_reasons=reasons,
        )
        active = next((run for run in runs if run.status in {"queued", "running"}), None)
        prediction_exists = (self.prediction_root / "stable.json").exists()
        if active:
            message = "15日实验模型训练中。"
        elif latest_run and latest_run.status == "failed":
            message = f"最近短期任务失败：{latest_run.error or latest_run.message or '未知错误'}"
        elif ensemble_manifest:
            message = "；".join(reasons) if reasons else "短期实验门禁正常。"
        else:
            message = "短期模型训练尚未完成，实验发布门禁尚未执行。" + (
                f" 当前前置条件：{'；'.join(reasons)}" if reasons else ""
            )
        monitoring_status: Literal["training", "failed", "blocked", "idle"] = (
            "training"
            if active
            else "failed"
            if latest_run and latest_run.status == "failed"
            else "blocked"
            if reasons
            else "idle"
        )
        return ModelMonitoring(
            status=monitoring_status,
            message=message,
            data_coverage_pct=daily_coverage * 100,
            drift_psi=None,
            latest_prediction_status="published" if prediction_exists and gate else "blocked",
            daily_history_days=daily_days,
            intraday_history_days=daily_days if intraday_coverage else 0,
            intraday_coverage_pct=intraday_coverage * 100,
            components=components,
            ensemble=ensemble,
            source_checks=[
                SourceCheck.model_validate(item)
                for item in coverage.get("source_checks", [])
                if isinstance(item, dict)
            ],
            training_runs=runs,
            mode="short_term",
            experimental=True,
        )

    def training_runs(self) -> list[TrainingRunSummary]:
        runs: list[TrainingRunSummary] = []
        modern = list(training_run_root(self._settings, "short_term").glob("*.json"))
        legacy = list((self._settings.artifact_root / "training_runs").glob("short-term-*.json"))
        for path in sorted(modern + legacy, reverse=True)[:20]:
            if path.name.endswith(".heartbeat.json"):
                continue
            try:
                runs.append(
                    load_training_run(
                        path,
                        self._settings.model_run_stale_minutes,
                        self._settings.training_unresponsive_seconds,
                    )
                )
            except (ValueError, OSError):
                continue
        return runs

    def _component(self, name: str) -> ComponentHealth:
        display = {
            "hmm": "短期HMM市场状态",
            "tft": "短期TFT收益分位",
            "patchtst": "短期PatchTST编码",
            "ranker": "短期LambdaRank排序",
        }[name]
        component = cast(Literal["hmm", "tft", "patchtst", "ranker"], name)
        manifest = self._read_json(self.root / "models" / name / "champion.json") or self._read_json(
            self.root / "models" / name / "latest.json"
        )
        if manifest is None:
            active = next((run for run in self.training_runs() if run.status == "running" and run.stage == name), None)
            return ComponentHealth(
                component=component,
                display_name=display,
                status="running" if active else "not_started",
                progress=active.progress if active else 0,
                metrics=(
                    {"current_symbol": active.current_symbol or "training"}
                    if active
                    else {"availability": "not_ready"}
                ),
            )
        return ComponentHealth(
            component=component,
            display_name=display,
            status=manifest.get("status", "warning"),
            version=manifest.get("version"),
            progress=100,
            last_trained_at=manifest.get("last_trained_at"),
            metrics=manifest.get("metrics", {}),
            checkpoint=manifest.get("checkpoint"),
            artifact_sha256=manifest.get("artifact_sha256"),
            error=manifest.get("error"),
        )

    def _coverage(self) -> dict[str, Any]:
        return self._read_json(self.root / "data_coverage.json") or self._derived_cache_coverage()

    def _derived_cache_coverage(self) -> dict[str, Any]:
        daily_root = self._settings.data_root / "short_term_cache" / "daily"
        paths = list(daily_root.glob("*.parquet"))
        if not paths:
            return {}
        try:
            daily = pl.scan_parquet(str(daily_root / "*.parquet")).select("symbol", "trade_date").collect()
        except (OSError, pl.exceptions.PolarsError):
            return {}
        universe_size = self._market_universe_size() or daily["symbol"].n_unique()
        threshold = math.ceil(universe_size * self._settings.short_term_daily_coverage)
        dates = (
            daily.group_by("trade_date")
            .agg(pl.col("symbol").n_unique().alias("symbols"))
            .filter(pl.col("symbols") >= threshold)
            .sort("trade_date")["trade_date"]
            .tail(15)
            .to_list()
        )
        if not dates:
            return {}
        complete = (
            daily.filter(pl.col("trade_date").is_in(dates))
            .group_by("symbol")
            .agg(pl.col("trade_date").n_unique().alias("days"))
            .filter(pl.col("days") == len(dates))
            .height
        )
        coverage = complete / max(1, universe_size)
        return {
            "version": f"short-cache-{max(dates):%Y%m%d}",
            "status": "partial",
            "distinct_trade_dates": len(dates),
            "daily_coverage": coverage,
            "intraday_coverage": 0.0,
            "daily_check_coverage": 0.0,
            "source_checks": [
                {
                    "provider": "eastmoney",
                    "datasets": ["15日日线"],
                    "status": "ready" if len(dates) == 15 else "partial",
                    "message": f"从缓存识别{len(dates)}个共同交易日，完整覆盖率{coverage:.2%}。",
                },
                {
                    "provider": self._settings.short_term_history_provider,
                    "datasets": ["30分钟K线", "跨源日线校验"],
                    "status": "partial",
                    "message": "等待新数据源预检和逐股回填。",
                },
            ],
        }

    def _market_universe_size(self) -> int:
        market = self._read_json(self._settings.artifact_root / "published" / "market" / "stable.json")
        quotes = market.get("data", {}).get("quotes", []) if market else []
        return len(quotes) if isinstance(quotes, list) else 0

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None


@lru_cache
def get_short_term_runtime_service() -> ShortTermRuntimeService:
    return ShortTermRuntimeService(get_settings())
