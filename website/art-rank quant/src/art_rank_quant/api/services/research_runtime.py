from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import polars as pl

from art_rank_quant.api.schemas.common import VersionMetadata
from art_rank_quant.api.schemas.market import MarketQuoteItem, MarketSnapshot, RefreshAccepted, RefreshRun
from art_rank_quant.api.schemas.models import (
    ComponentHealth,
    EnsembleHealth,
    ModelMonitoring,
    TrainingRunSummary,
)
from art_rank_quant.api.schemas.stocks import PredictionFeed
from art_rank_quant.api.services.model_bootstrap import load_training_run, training_run_root
from art_rank_quant.api.services.realtime import market_session
from art_rank_quant.common.config import Settings, get_settings
from art_rank_quant.data.clients.eastmoney_daily import EastmoneyDailyCrawler


class ResearchRuntimeService:
    """Local, append-only runtime for market refreshes and gated model publication."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._active_run_id: str | None = None

    @property
    def run_root(self) -> Path:
        return self._settings.artifact_root / "refresh_runs"

    @property
    def market_root(self) -> Path:
        return self._settings.artifact_root / "published" / "market"

    @property
    def prediction_root(self) -> Path:
        return self._settings.artifact_root / "published" / "predictions"

    def start_refresh(self) -> RefreshAccepted:
        with self._lock:
            active = self._persistent_active_refresh()
            if active is not None:
                self._active_run_id = active.run_id
                return RefreshAccepted(
                    run_id=active.run_id,
                    status="running",
                    message="已有全市场更新任务正在运行。",
                )
            run_id = f"refresh-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:6]}"
            run = RefreshRun(run_id=run_id, status="queued", step="queued", progress=0)
            self._write_run(run)
            self._write_refresh_lock(run_id)
            self._active_run_id = run_id
            return RefreshAccepted(run_id=run_id, status="queued", message="全市场更新任务已提交。")

    def execute_refresh(self, run_id: str) -> None:
        started = datetime.now(UTC)
        run = RefreshRun(
            run_id=run_id,
            status="running",
            step="ingestion",
            progress=5,
            started_at=started,
        )
        self._write_run(run)
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._refresh_heartbeat,
            args=(run_id, heartbeat_stop),
            name=f"refresh-heartbeat-{run_id}",
            daemon=True,
        )
        heartbeat.start()
        started_clock = time.monotonic()
        try:
            crawler = EastmoneyDailyCrawler(
                permission_confirmed=self._settings.eastmoney_permission_confirmed,
                endpoint=self._settings.eastmoney_endpoint,
                page_size=self._settings.eastmoney_page_size,
                timeout_seconds=self._settings.eastmoney_request_timeout_seconds,
                request_delay_seconds=self._settings.eastmoney_request_delay_seconds,
            )
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    result = crawler.crawl_to_storage(self._settings.data_root)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(1.0 + attempt * 2.0)
            else:
                assert last_error is not None
                raise last_error
            run.data_version = result.raw_manifest.version
            self._advance(run, "quality", 55)
            frame = pl.read_parquet(result.raw_path)
            self._validate_market_frame(frame, result.raw_manifest.rows)
            self._advance(run, "features", 70)
            self._advance(run, "inference", 82)
            monitoring = self.monitoring()
            if not monitoring.ensemble.publish_gate_passed:
                run.prediction_status = "blocked"
            else:
                run.prediction_status = "published" if (self.prediction_root / "stable.json").exists() else "blocked"
            self._advance(run, "publish", 92)
            duration = time.monotonic() - started_clock
            self._publish_market(frame, result.raw_manifest.version, run_id, duration)
            run.status = "success"
            run.step = "complete"
            run.progress = 100
            run.finished_at = datetime.now(UTC)
            self._write_run(run)
        except Exception as exc:
            run.status = "failed"
            run.step = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            run.finished_at = datetime.now(UTC)
            self._write_run(run)
            self._mark_market_stale(run.error)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2)
            self._release_refresh_lock(run_id)
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None

    def get_run(self, run_id: str) -> RefreshRun:
        path = self.run_root / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(run_id)
        return RefreshRun.model_validate_json(path.read_text(encoding="utf-8"))

    def latest_market(self) -> tuple[MarketSnapshot, VersionMetadata]:
        path = self.market_root / "stable.json"
        if not path.exists():
            stored = self._latest_stored_market()
            if stored is not None:
                return stored
            return self._missing_market()
        body = json.loads(path.read_text(encoding="utf-8"))
        snapshot = MarketSnapshot.model_validate(body["data"])
        metadata = VersionMetadata.model_validate(body["meta"])
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        current_session = market_session(now)
        stale = metadata.is_stale or snapshot.trade_date is None or snapshot.trade_date < now.date()
        if snapshot.fetched_at is not None and current_session == "trading":
            fetched_at = snapshot.fetched_at.astimezone(ZoneInfo("Asia/Shanghai"))
            stale = stale or (now - fetched_at).total_seconds() > self._settings.realtime_cache_seconds
        snapshot = snapshot.model_copy(
            update={
                "market_session": current_session,
                "status": "stale" if stale else snapshot.status,
                "message": "当前显示最近一次已校验快照，数据已过实时有效期。" if stale else snapshot.message,
            }
        )
        metadata = metadata.model_copy(update={"is_stale": stale})
        return snapshot, metadata

    def latest_prediction(self, limit: int = 20, include_all: bool = False) -> tuple[PredictionFeed, VersionMetadata]:
        path = self.prediction_root / "stable.json"
        if path.exists():
            body = json.loads(path.read_text(encoding="utf-8"))
            feed = PredictionFeed.model_validate(body["data"])
            if not include_all:
                feed = feed.model_copy(update={"candidates": feed.candidates[:limit]})
            return feed, VersionMetadata.model_validate(body["meta"])
        monitoring = self.monitoring()
        reasons = monitoring.ensemble.blocked_reasons
        message = "正式预测尚未发布。" + (f" {'；'.join(reasons)}" if reasons else "")
        return (
            PredictionFeed(status="blocked", message=message),
            VersionMetadata(
                data_version=self._latest_data_version(),
                feature_version="intraday-features-not-ready",
                model_version="art-rank-blocked",
                run_id="none",
                is_stale=True,
            ),
        )

    def prediction_history(self) -> list[dict[str, Any]]:
        if not self.prediction_root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.prediction_root.glob("snapshot-*.json"), reverse=True)[:100]:
            body = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"path": path.name, "meta": body.get("meta", {}), "status": body.get("data", {}).get("status")})
        return rows

    def monitoring(self) -> ModelMonitoring:
        components = [self._component_health(name) for name in ("hmm", "tft", "patchtst", "ranker")]
        reasons: list[str] = []
        training_runs = self._training_runs()
        latest_run = training_runs[0] if training_runs else None
        if latest_run and latest_run.status == "failed":
            reasons.append(f"最近长期任务失败：{latest_run.error or latest_run.message or '未知错误'}。")
        if any(component.status != "healthy" for component in components):
            reasons.append("HMM、TFT、PatchTST、LambdaRank 尚未全部达到 healthy。")
        coverage = self._coverage_manifest()
        daily_days = int(coverage.get("distinct_trade_dates", self._real_history_days()))
        intraday_days = int(coverage.get("intraday_trade_dates", 0))
        intraday_coverage = float(coverage.get("intraday_coverage", 0.0))
        if daily_days < self._settings.model_min_daily_days:
            reasons.append(
                f"真实历史仅覆盖 {daily_days} 个交易日，低于 {self._settings.model_min_daily_days} 日发布门槛。"
            )
        if intraday_coverage < self._settings.model_min_intraday_coverage:
            reasons.append(
                f"30分钟历史覆盖率 {intraday_coverage:.2%}，低于 {self._settings.model_min_intraday_coverage:.0%}。"
            )
        ensemble_dir = self._settings.artifact_root / "models" / "ensemble"
        ensemble_manifest = self._read_json(ensemble_dir / "champion.json") or self._read_json(
            ensemble_dir / "latest.json"
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
        active_run = self._active_training_run()
        if active_run:
            message = "模型训练中。"
        elif ensemble_manifest:
            message = "；".join(reasons) if reasons else "四组件与发布门禁正常。"
        else:
            message = "模型训练尚未完成，统计发布门禁尚未执行。" + (
                f" 当前前置条件：{'；'.join(reasons)}" if reasons else ""
            )
        return ModelMonitoring(
            status="training" if active_run else "blocked" if reasons else "idle",
            message=message,
            data_coverage_pct=self._market_coverage_pct(),
            drift_psi=None,
            latest_prediction_status="published" if (self.prediction_root / "stable.json").exists() else "blocked",
            daily_history_days=daily_days,
            intraday_history_days=intraday_days,
            intraday_coverage_pct=intraday_coverage * 100,
            components=components,
            ensemble=ensemble,
            training_runs=training_runs,
        )

    def _component_health(self, name: str) -> ComponentHealth:
        display = {
            "hmm": "HMM 市场状态",
            "tft": "TFT 风格预测",
            "patchtst": "PatchTST 时序编码",
            "ranker": "LambdaRank 排序",
        }[name]
        component = cast(Literal["hmm", "tft", "patchtst", "ranker"], name)
        manifest = self._read_json(self._settings.artifact_root / "models" / name / "champion.json")
        if manifest is None:
            active = next((run for run in self._training_runs() if run.status == "running"), None)
            if active and active.stage == name:
                return ComponentHealth(
                    component=component,
                    display_name=display,
                    status="running",
                    progress=active.progress,
                    metrics={"current_symbol": active.current_symbol or "training"},
                )
            return ComponentHealth(
                component=component,
                display_name=display,
                status="not_started",
                metrics={"availability": "no_validated_artifact"},
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

    def _validate_market_frame(self, frame: pl.DataFrame, expected_rows: int) -> None:
        required = {"symbol", "name", "trade_date", "close", "amount", "trade_status"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Market snapshot missing columns: {sorted(missing)}")
        if frame.height != expected_rows or frame["symbol"].n_unique() != expected_rows:
            raise ValueError("Market snapshot coverage or symbol uniqueness failed")
        if frame["name"].null_count() or frame.filter(pl.col("name").str.len_chars() == 0).height:
            raise ValueError("Market snapshot contains missing stock names")

    def _publish_market(self, frame: pl.DataFrame, version: str, run_id: str, duration: float) -> None:
        market = self._snapshot_from_frame(frame, duration)
        trade_date = market.trade_date
        if trade_date is None:
            raise ValueError("Market snapshot has no trade date")
        meta = VersionMetadata(
            as_of_date=trade_date,
            data_version=version,
            feature_version="market-snapshot-v2",
            model_version="observation-only",
            run_id=run_id,
            is_stale=False,
        )
        body = {"meta": meta.model_dump(mode="json"), "data": market.model_dump(mode="json")}
        self.market_root.mkdir(parents=True, exist_ok=True)
        target = self.market_root / f"snapshot-{version}.json"
        self._atomic_json(target, body)
        self._atomic_json(self.market_root / "stable.json", body)

    def _snapshot_from_frame(self, frame: pl.DataFrame, duration: float | None) -> MarketSnapshot:
        quotes: list[MarketQuoteItem] = []
        for row in frame.iter_rows(named=True):
            symbol = str(row["symbol"])
            close = row.get("close")
            pct_change = row.get("pct_change")
            flags: list[str] = []
            name = str(row["name"])
            if "ST" in name.upper():
                flags.append("ST风险")
            if row.get("trade_status") != "trading":
                flags.append("停牌")
            if close is not None and row.get("limit_up") is not None and float(close) >= float(row["limit_up"]):
                flags.append("涨停不可买入")
            if float(row.get("amount") or 0) < self._settings.candidate_min_amount_cny:
                flags.append("流动性偏低")
            quotes.append(
                MarketQuoteItem(
                    symbol=symbol,
                    name=name,
                    industry=str(row.get("industry") or "未分类"),
                    exchange=cast(Literal["SH", "SZ", "BJ"], symbol.rsplit(".", 1)[-1]),
                    close=close,
                    pct_change=pct_change,
                    amount=float(row.get("amount") or 0),
                    turnover_rate=row.get("turnover_rate"),
                    volume_ratio=row.get("volume_ratio"),
                    amplitude=row.get("amplitude"),
                    trade_status=str(row.get("trade_status") or "unknown"),
                    risk_flags=flags,
                )
            )
        fetched_at = max(frame["ingested_at"].to_list())
        trade_date = max(frame["trade_date"].to_list())
        return MarketSnapshot(
            status="live",
            market_session=market_session(fetched_at),
            message="东方财富全市场快照已通过完整性检查。",
            trade_date=trade_date,
            fetched_at=fetched_at,
            duration_seconds=round(duration, 2) if duration is not None else None,
            universe_count=len(quotes),
            trading_count=sum(quote.trade_status == "trading" for quote in quotes),
            advancing_count=sum((quote.pct_change or 0) > 0 for quote in quotes),
            declining_count=sum((quote.pct_change or 0) < 0 for quote in quotes),
            total_amount=sum(quote.amount for quote in quotes),
            quotes=quotes,
        )

    def _latest_stored_market(self) -> tuple[MarketSnapshot, VersionMetadata] | None:
        paths = self._stored_market_paths()
        if not paths:
            return None
        path = paths[-1]
        frame = pl.read_parquet(path)
        market = self._snapshot_from_frame(frame, None).model_copy(
            update={
                "status": "stale",
                "message": "最新抓取失败，当前显示最近一次已校验的东方财富真实行情快照。",
            }
        )
        if market.trade_date is None:
            return None
        version = path.parent.name.removeprefix("data_version=")
        metadata = VersionMetadata(
            as_of_date=market.trade_date,
            data_version=version,
            feature_version="market-snapshot-v2",
            model_version="observation-only",
            run_id="stored-fallback",
            is_stale=True,
        )
        return market, metadata

    def _missing_market(self) -> tuple[MarketSnapshot, VersionMetadata]:
        now = datetime.now(UTC)
        return (
            MarketSnapshot(
                status="not_available",
                market_session=market_session(now),
                message="尚无已发布的东方财富全市场快照，请点击强制更新。",
            ),
            VersionMetadata(
                data_version="eastmoney-not-available",
                feature_version="market-snapshot-v2",
                model_version="none",
                run_id="none",
                is_stale=True,
            ),
        )

    def _mark_market_stale(self, error: str) -> None:
        path = self.market_root / "stable.json"
        if not path.exists():
            return
        body = json.loads(path.read_text(encoding="utf-8"))
        body["meta"]["is_stale"] = True
        body["data"]["status"] = "stale"
        body["data"]["message"] = f"最新抓取失败，保留上一稳定快照。{error}"
        self._atomic_json(path, body)

    def _advance(self, run: RefreshRun, step: str, progress: int) -> None:
        run.step = cast(
            Literal["queued", "ingestion", "quality", "features", "inference", "publish", "complete", "failed"],
            step,
        )
        run.progress = progress
        self._write_run(run)

    def _write_run(self, run: RefreshRun) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._atomic_json(self.run_root / f"{run.run_id}.json", run.model_dump(mode="json"))
        lock = self.run_root / "active-refresh.json"
        if lock.exists():
            try:
                body = json.loads(lock.read_text(encoding="utf-8"))
                if body.get("run_id") == run.run_id:
                    body["heartbeat_at"] = datetime.now(UTC).isoformat()
                    self._atomic_json(lock, body)
            except (ValueError, OSError):
                pass

    def _persistent_active_refresh(self) -> RefreshRun | None:
        lock = self.run_root / "active-refresh.json"
        if not lock.exists():
            return None
        try:
            body = json.loads(lock.read_text(encoding="utf-8"))
            run = self.get_run(str(body["run_id"]))
            heartbeat = datetime.fromisoformat(str(body["heartbeat_at"]))
            pid = int(body.get("pid", 0))
        except (KeyError, ValueError, OSError, FileNotFoundError):
            lock.unlink(missing_ok=True)
            return None
        alive = pid > 0 and self._process_is_alive(pid)
        if run.status in {"queued", "running"} and alive and datetime.now(UTC) - heartbeat < timedelta(minutes=10):
            return run
        if run.status in {"queued", "running"}:
            run.status = "failed"
            run.step = "failed"
            run.error = "stale_refresh_run"
            run.finished_at = datetime.now(UTC)
            self._atomic_json(self.run_root / f"{run.run_id}.json", run.model_dump(mode="json"))
        lock.unlink(missing_ok=True)
        return None

    def _write_refresh_lock(self, run_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        self._atomic_json(
            self.run_root / "active-refresh.json",
            {"run_id": run_id, "pid": os.getpid(), "created_at": now, "heartbeat_at": now},
        )

    def _refresh_heartbeat(self, run_id: str, stop: threading.Event) -> None:
        interval = min(30, self._settings.training_heartbeat_seconds)
        while not stop.wait(interval):
            lock = self.run_root / "active-refresh.json"
            try:
                body = json.loads(lock.read_text(encoding="utf-8"))
                if body.get("run_id") != run_id:
                    return
                body["heartbeat_at"] = datetime.now(UTC).isoformat()
                self._atomic_json(lock, body)
            except (ValueError, OSError):
                return

    def _release_refresh_lock(self, run_id: str) -> None:
        lock = self.run_root / "active-refresh.json"
        if not lock.exists():
            return
        try:
            if json.loads(lock.read_text(encoding="utf-8")).get("run_id") == run_id:
                lock.unlink(missing_ok=True)
        except (ValueError, OSError):
            lock.unlink(missing_ok=True)

    @staticmethod
    def _process_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _latest_data_version(self) -> str:
        path = self.market_root / "stable.json"
        if not path.exists():
            stored = self._stored_market_paths()
            return stored[-1].parent.name.removeprefix("data_version=") if stored else "eastmoney-not-available"
        return str(json.loads(path.read_text(encoding="utf-8"))["meta"]["data_version"])

    def _market_coverage_pct(self) -> float:
        path = self.market_root / "stable.json"
        if path.exists():
            body = json.loads(path.read_text(encoding="utf-8"))
            count = int(body["data"].get("universe_count", 0))
        else:
            stored = self._stored_market_paths()
            count = pl.read_parquet(stored[-1], columns=["symbol"]).height if stored else 0
        return 100.0 if count >= 5000 else round(count / 5000 * 100, 2)

    def _stored_market_paths(self) -> list[Path]:
        return sorted(self._settings.data_root.glob("raw/eastmoney_stock_daily/data_version=*/part-*.parquet"))

    def _real_history_days(self) -> int:
        manifests = self._settings.data_root.glob("raw/eastmoney_stock_daily/data_version=*/manifest.json")
        dates: set[str] = set()
        for path in manifests:
            body = self._read_json(path)
            if body and int(body.get("rows", 0)) >= 5000:
                dates.add(str(body.get("min_date")))
        return len(dates)

    def _coverage_manifest(self) -> dict[str, Any]:
        return self._read_json(self._settings.artifact_root / "data_coverage.json") or {}

    def _active_training_run(self) -> bool:
        return any(run.status in {"queued", "running"} for run in self._training_runs())

    def _training_runs(self) -> list[TrainingRunSummary]:
        runs: list[TrainingRunSummary] = []
        modern = list(training_run_root(self._settings, "long_term").glob("*.json"))
        legacy = list((self._settings.artifact_root / "training_runs").glob("bootstrap-*.json"))
        for path in sorted(modern + legacy, reverse=True)[:20]:
            if path.name.endswith(".heartbeat.json"):
                continue
            try:
                run = load_training_run(
                    path,
                    self._settings.model_run_stale_minutes,
                    self._settings.training_unresponsive_seconds,
                )
                if run.mode == "long_term":
                    runs.append(run)
            except (ValueError, OSError):
                continue
        return runs

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None

    @staticmethod
    def _atomic_json(path: Path, body: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


@lru_cache
def get_research_runtime_service() -> ResearchRuntimeService:
    return ResearchRuntimeService(get_settings())
