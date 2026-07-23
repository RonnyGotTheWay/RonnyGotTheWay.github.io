from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from art_rank_quant.api.schemas.common import VersionMetadata
from art_rank_quant.api.schemas.models import TrainingRunSummary
from art_rank_quant.api.schemas.stocks import PredictionCandidate, PredictionContext, PredictionFeed
from art_rank_quant.api.services.model_bootstrap import (
    TrainingCancelled,
    TrainingHeartbeat,
    TrainingMemoryLimitExceeded,
    check_run_control,
    release_training_lock,
    resolve_training_run_path,
    training_run_root,
)
from art_rank_quant.common.config import Settings, get_settings
from art_rank_quant.data.clients.akshare_sina_history import (
    AkshareSinaHistoryClient,
    AkshareSinaHistoryError,
)
from art_rank_quant.data.clients.baostock_history import BaoStockHistoryClient, BaoStockHistoryError
from art_rank_quant.data.clients.eastmoney_history import EastmoneyHistoryClient, EastmoneyHistoryError
from art_rank_quant.models.ensemble.calibration import percentile_calibrate
from art_rank_quant.models.hmm.model import RegimeHMM
from art_rank_quant.models.patchtst.model import PatchTSTEncoder
from art_rank_quant.models.ranker.model import ArtRanker
from art_rank_quant.models.ranker.train import rank_ic
from art_rank_quant.models.tft.model import TftStyleForecaster
from art_rank_quant.models.tft.train import quantile_loss

WINDOW_DAYS = 15
INTRADAY_BARS = 120
SHORT_BASE_FEATURES = [
    "momentum_1d",
    "momentum_3d",
    "momentum_5d",
    "momentum_10d",
    "volatility_5d",
    "volatility_10d",
    "log_amount",
    "turnover_proxy_5d",
    "intraday_return",
    "intraday_amplitude",
    "intraday_log_amount",
]
STAGES = {
    "universe": 2,
    "source_check": 3,
    "daily": 5,
    "intraday": 30,
    "quality": 55,
    "features": 62,
    "hmm": 68,
    "tft": 73,
    "patchtst": 80,
    "ranker": 87,
    "validation": 93,
    "ensemble": 97,
    "publish": 99,
}
STAGE_LABELS = {
    "universe": "股票池",
    "source_check": "数据源预检",
    "daily": "15日日线",
    "intraday": "30分钟K线与跨源校验",
    "quality": "数据质量门禁",
    "features": "短期特征",
    "hmm": "HMM",
    "tft": "TFT",
    "patchtst": "PatchTST",
    "ranker": "LambdaRank",
    "validation": "滚动验证",
    "ensemble": "实验集成",
    "publish": "发布排名",
    "complete": "完成",
    "failed": "失败",
}


class ShortTermGateBlocked(RuntimeError):
    pass


class ShortTermPipeline:
    """15-trading-day experimental pipeline isolated from the long-term champions."""

    def __init__(self, run_id: str, settings: Settings) -> None:
        self.run_id = run_id
        self.settings = settings
        self.run_root = training_run_root(settings, "short_term")
        self.run_path = resolve_training_run_path(settings, run_id, "short_term")
        self.started = datetime.now(UTC)
        self.end = self._latest_trade_date()
        self.fetch_start = self.end - timedelta(days=40)
        self.version = f"short-history-{self.end:%Y%m%d}"
        self.cache_root = settings.data_root / "short_term_cache"
        self.daily_root = self.cache_root / "daily"
        self.check_root = self.cache_root / "daily_check"
        self.intraday_root = self.cache_root / "intraday"
        self.universe_root = self.cache_root / "universe"
        self.artifact_root = settings.artifact_root / "short_term"
        self.prediction_root = settings.artifact_root / "published" / "short_term_predictions"
        self.secondary_provider = settings.short_term_history_provider
        self.failed_providers: set[str] = set()
        for path in (
            self.daily_root,
            self.check_root,
            self.intraday_root,
            self.universe_root,
            self.artifact_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.run = self._read_run()

    def execute(self) -> None:
        with TrainingHeartbeat(self.settings, self.run_path, self.run_id, "short_term"):
            self._execute()

    def _execute(self) -> None:
        self.run.status = "running"
        self.run.started_at = self.run.started_at or self.started
        self.run.mode = "short_term"
        self._update("universe", STAGES["universe"], "读取全市场股票池并准备15日缓存。")
        try:
            universe = self._universe()
            symbols = sorted(universe)
            self.run.total_symbols = len(symbols)
            self._source_preflight()
            self._backfill_daily(symbols)
            common_dates = select_common_trading_dates(self.daily_root, len(symbols), WINDOW_DAYS)
            self._trim_daily_cache(common_dates)
            self._write_partial_coverage(common_dates, symbols)
            self._backfill_secondary(universe, common_dates)
            coverage = self._quality(common_dates, symbols)
            if not coverage["passed"]:
                raise ShortTermGateBlocked("；".join(cast(list[str], coverage["blocked_reasons"])))
            self._train_and_publish(common_dates, universe, coverage)
            self.run = self._read_run()
            self.run.status = "success"
            self.run.stage = "complete"
            self.run.stage_label = STAGE_LABELS["complete"]
            self.run.progress = 100
            self.run.message = "15日数据、四组件实验训练与短期排名发布已完成。"
            self.run.finished_at = datetime.now(UTC)
            self.run.duration_seconds = self._duration(self.run.finished_at)
            self._write_run()
        except ShortTermGateBlocked as exc:
            self.run = self._read_run()
            self.run.status = "blocked"
            self.run.stage = "validation" if self.run.progress >= STAGES["features"] else "quality"
            self.run.message = str(exc)
            self.run.finished_at = datetime.now(UTC)
            self.run.duration_seconds = self._duration(self.run.finished_at)
            self._write_run()
        except (TrainingCancelled, TrainingMemoryLimitExceeded) as exc:
            self.run = self._read_run()
            if isinstance(exc, TrainingMemoryLimitExceeded):
                self._write_checkpoint("memory_limit")
            self.run.status = "cancelled" if isinstance(exc, TrainingCancelled) else "failed"
            self.run.stage = self.run.status
            self.run.stage_label = "已取消" if isinstance(exc, TrainingCancelled) else STAGE_LABELS["failed"]
            self.run.error = f"{type(exc).__name__}: {exc}"
            self.run.message = str(exc)
            self.run.finished_at = datetime.now(UTC)
            self.run.duration_seconds = self._duration(self.run.finished_at)
            self._write_run()
        except Exception as exc:
            self.run = self._read_run()
            self.run.status = "failed"
            self.run.stage = "failed"
            self.run.stage_label = STAGE_LABELS["failed"]
            self.run.error = f"{type(exc).__name__}: {exc}"
            self.run.message = "短期任务失败；已完成的逐股缓存会在下次运行复用。"
            self.run.finished_at = datetime.now(UTC)
            self.run.duration_seconds = self._duration(self.run.finished_at)
            self._write_run()
            raise
        finally:
            release_training_lock(self.settings, self.run_id)

    def _universe(self) -> dict[str, dict[str, str]]:
        path = self.settings.artifact_root / "published" / "market" / "stable.json"
        if not path.exists():
            raise RuntimeError("请先在实时候选页完成一次东方财富全市场刷新")
        body = json.loads(path.read_text(encoding="utf-8"))
        rows = body.get("data", {}).get("quotes", [])
        universe = {
            str(row["symbol"]): {
                "name": str(row.get("name") or row["symbol"]),
                "industry": str(row.get("industry") or "未分类"),
            }
            for row in rows
        }
        if len(universe) < 5000:
            raise RuntimeError(f"当前全市场列表仅 {len(universe)} 只，拒绝启动15日训练")
        return universe

    def _source_preflight(self) -> None:
        self._set_stage("source_check", STAGES["source_check"], "预检短期历史数据源。", 2)
        providers = [self.settings.short_term_history_provider, *self.settings.history_provider_fallback_list]
        failures: list[str] = []
        for provider in dict.fromkeys(providers):
            if provider == "baostock" and not self.settings.baostock_enabled:
                failures.append("baostock: 已禁用")
                continue
            try:
                self._preflight_provider(provider)
                self.secondary_provider = cast(Literal["eastmoney", "akshare_sina", "baostock"], provider)
                self.run.provider = provider
                break
            except (EastmoneyHistoryError, AkshareSinaHistoryError, BaoStockHistoryError) as exc:
                self.failed_providers.add(provider)
                failures.append(f"{provider}: {exc}")
        else:
            message = "；".join(failures) or "没有可用的30分钟历史数据源"
            self._write_source_failure(message)
            raise RuntimeError(f"短期数据源预检失败：{message}")
        self.run.stage_completed_items = 2
        self._write_run()

    def _preflight_provider(self, provider: str) -> None:
        if provider == "eastmoney":
            eastmoney_client = EastmoneyHistoryClient(
                permission_confirmed=self.settings.eastmoney_permission_confirmed,
                timeout_seconds=self.settings.eastmoney_request_timeout_seconds,
                request_delay_seconds=self.settings.eastmoney_request_delay_seconds,
            )
            try:
                eastmoney_client.preflight(self.fetch_start, self.end)
            finally:
                eastmoney_client.close()
            return
        if provider == "akshare_sina":
            AkshareSinaHistoryClient(
                delay_seconds=self.settings.short_term_secondary_delay_seconds,
                timeout_seconds=self.settings.eastmoney_request_timeout_seconds,
            ).preflight(self.fetch_start, self.end)
            return
        if provider == "baostock":
            with BaoStockHistoryClient():
                return
        raise RuntimeError(f"不支持的短期历史数据源: {provider}")

    def _backfill_daily(self, symbols: list[str]) -> None:
        self._set_stage("daily", STAGES["daily"], "并发回填东方财富最近15个交易日日线。", len(symbols))
        pending = [symbol for symbol in symbols if not self._daily_cache_is_current(symbol)]
        completed = len(symbols) - len(pending)
        self.run.completed_symbols = completed
        self.run.stage_completed_items = completed
        self._write_run()
        workers = self._adaptive_workers(self.settings.short_term_history_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._fetch_eastmoney_symbol, symbol): symbol for symbol in pending}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    future.result()
                    self.run.last_success_at = datetime.now(UTC)
                except (EastmoneyHistoryError, ValueError, OSError) as exc:
                    self.run.retries += 1
                    self.run.failed_items += 1
                    self._quarantine("daily", symbol, str(exc))
                completed += 1
                self.run.current_symbol = symbol
                self.run.completed_symbols = completed
                self.run.stage_completed_items = completed
                self.run.progress = 5 + int(completed / max(1, len(symbols)) * 24)
                self._update_eta(completed, len(symbols))
                if completed % 10 == 0 or completed == len(symbols):
                    self._write_run()

    def _fetch_eastmoney_symbol(self, symbol: str) -> None:
        target = self.daily_root / f"{safe_symbol(symbol)}.parquet"
        existing: pl.DataFrame | None = None
        request_start = self.fetch_start
        if target.exists():
            try:
                existing = pl.read_parquet(target).filter(pl.col("trade_date") <= self.end)
            except (OSError, pl.exceptions.PolarsError):
                existing = None
            if existing is not None and existing["trade_date"].n_unique() >= WINDOW_DAYS:
                latest = cast(date, existing["trade_date"].max())
                request_start = latest + timedelta(days=1)
        client = EastmoneyHistoryClient(
            permission_confirmed=self.settings.eastmoney_permission_confirmed,
            timeout_seconds=self.settings.eastmoney_request_timeout_seconds,
            request_delay_seconds=self.settings.eastmoney_request_delay_seconds,
        )
        try:
            raw = client.fetch_daily(symbol, request_start, self.end, "raw")
            time.sleep(self.settings.eastmoney_request_delay_seconds)
            adjusted = client.fetch_daily(symbol, request_start, self.end, "forward")
            if raw.is_empty() or adjusted.is_empty():
                raise EastmoneyHistoryError(f"empty short-term daily history for {symbol}")
            prices = adjusted.select(
                "trade_date",
                pl.col("open").alias("open_forward_adjusted"),
                pl.col("high").alias("high_forward_adjusted"),
                pl.col("low").alias("low_forward_adjusted"),
                pl.col("close").alias("close_forward_adjusted"),
            )
            joined = raw.join(prices, on="trade_date", how="left")
            if existing is not None and not existing.is_empty():
                joined = pl.concat([existing, joined], how="diagonal_relaxed")
            prepared = joined.unique(subset=["symbol", "trade_date"], keep="last").sort("trade_date").tail(
                WINDOW_DAYS
            )
            write_cache_atomic(prepared, target)
        finally:
            client.close()

    def _daily_cache_is_current(self, symbol: str) -> bool:
        target = self.daily_root / f"{safe_symbol(symbol)}.parquet"
        if not target.exists():
            self._seed_from_long_cache(symbol, "daily", target)
        if not target.exists():
            return False
        try:
            frame = pl.read_parquet(target, columns=["trade_date"])
        except (OSError, pl.exceptions.PolarsError):
            return False
        return frame["trade_date"].n_unique() >= WINDOW_DAYS and frame["trade_date"].max() == self.end

    def _seed_from_long_cache(self, symbol: str, dataset: str, target: Path) -> bool:
        required_columns = {
            "daily": {"symbol", "trade_date", "close", "close_forward_adjusted"},
            "intraday": {"symbol", "trade_date", "event_time", "open", "high", "low", "close"},
            "daily_check": {"symbol", "trade_date", "close_baostock", "trade_status_baostock", "is_st"},
        }[dataset]
        candidates = sorted(
            self.settings.data_root.glob(f"staging/history-*/{dataset}/{safe_symbol(symbol)}.parquet"),
            reverse=True,
        )
        for source in candidates:
            manifest_path = source.with_suffix(".manifest.json")
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest.get("sha256") != file_sha256(source):
                        continue
                except (OSError, ValueError):
                    continue
            try:
                frame = pl.read_parquet(source).filter(pl.col("trade_date") <= self.end).sort("trade_date")
            except (OSError, pl.exceptions.PolarsError):
                continue
            if not required_columns.issubset(frame.columns):
                continue
            if frame.is_empty() or frame["symbol"].n_unique() != 1 or str(frame["symbol"][0]) != symbol:
                continue
            dates = frame["trade_date"].unique().sort()
            if len(dates) < WINDOW_DAYS:
                continue
            selected = dates[-WINDOW_DAYS:].to_list()
            write_cache_atomic(frame.filter(pl.col("trade_date").is_in(selected)), target)
            return True
        return False

    def _trim_daily_cache(self, common_dates: list[date]) -> None:
        for target in self.daily_root.glob("*.parquet"):
            frame = pl.read_parquet(target)
            trimmed = frame.filter(pl.col("trade_date").is_in(common_dates)).sort("trade_date")
            write_cache_atomic(trimmed, target)

    def _backfill_secondary(
        self,
        universe: dict[str, dict[str, str]],
        common_dates: list[date],
    ) -> None:
        supported = sorted(symbol for symbol in universe if symbol.endswith((".SH", ".SZ")))
        provider = self.secondary_provider
        self._set_stage(
            "intraday",
            STAGES["intraday"],
            f"回填{provider}的15日30分钟K线与跨源日线校验。",
            len(supported),
        )
        self.run.completed_symbols = 0
        consecutive_failures = 0
        failure_window: deque[bool] = deque(maxlen=100)
        completed = 0
        workers = self._adaptive_workers(self.settings.short_term_secondary_workers)
        batch_size = max(1, workers * 2)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for offset in range(0, len(supported), batch_size):
                batch = supported[offset : offset + batch_size]
                futures = {
                    executor.submit(
                        self._fetch_secondary_symbol,
                        symbol,
                        common_dates,
                        universe[symbol]["name"],
                    ): symbol
                    for symbol in batch
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        future.result()
                        consecutive_failures = 0
                        failure_window.append(False)
                        self.run.last_success_at = datetime.now(UTC)
                    except (
                        AkshareSinaHistoryError,
                        BaoStockHistoryError,
                        EastmoneyHistoryError,
                        OSError,
                        pl.exceptions.PolarsError,
                    ) as exc:
                        consecutive_failures += 1
                        failure_window.append(True)
                        self.run.retries += 1
                        self.run.failed_items += 1
                        self._quarantine("intraday", symbol, str(exc))
                    completed += 1
                    self.run.current_symbol = symbol
                    self.run.completed_symbols = completed
                    self.run.stage_completed_items = completed
                    self.run.progress = 30 + int(completed / max(1, len(supported)) * 24)
                    self._update_eta(completed, len(supported))
                    if completed % 10 == 0 or completed == len(supported):
                        self._write_run()
                window_open = len(failure_window) == 100 and sum(failure_window) >= 20
                if failure_circuit_open(consecutive_failures, self.settings.short_term_failure_circuit) or window_open:
                    message = (
                        f"{provider}失败率过高（连续{consecutive_failures}只，最近100只失败{sum(failure_window)}只），"
                        "已熔断以避免继续触发数据源封禁。"
                    )
                    self._write_source_failure(message)
                    self.failed_providers.add(provider)
                    if self._activate_fallback_provider():
                        self._backfill_secondary(universe, common_dates)
                        return
                    raise RuntimeError(message)

    def _activate_fallback_provider(self) -> bool:
        providers = [self.settings.short_term_history_provider, *self.settings.history_provider_fallback_list]
        for provider in dict.fromkeys(providers):
            if provider in self.failed_providers:
                continue
            if provider == "baostock" and not self.settings.baostock_enabled:
                continue
            try:
                self._preflight_provider(provider)
            except (EastmoneyHistoryError, AkshareSinaHistoryError, BaoStockHistoryError):
                self.failed_providers.add(provider)
                continue
            self.secondary_provider = cast(Literal["eastmoney", "akshare_sina", "baostock"], provider)
            self.run.provider = provider
            self._write_run()
            return True
        return False

    def _fetch_secondary_symbol(
        self,
        symbol: str,
        common_dates: list[date],
        name: str,
    ) -> None:
        intraday_target = self.intraday_root / f"{safe_symbol(symbol)}.parquet"
        check_target = self.check_root / f"{safe_symbol(symbol)}.parquet"
        if not intraday_target.exists():
            self._seed_from_long_cache(symbol, "intraday", intraday_target)
        if not check_target.exists():
            self._seed_from_long_cache(symbol, "daily_check", check_target)
        intraday_missing = self._dates_to_fetch(intraday_target, common_dates, INTRADAY_BARS)
        check_missing = self._dates_to_fetch(check_target, common_dates, WINDOW_DAYS)
        if not intraday_missing and not check_missing:
            return
        provider = self.secondary_provider
        if provider == "eastmoney":
            eastmoney_client = EastmoneyHistoryClient(
                permission_confirmed=self.settings.eastmoney_permission_confirmed,
                timeout_seconds=self.settings.eastmoney_request_timeout_seconds,
                request_delay_seconds=self.settings.eastmoney_request_delay_seconds,
            )
            try:
                if intraday_missing:
                    frame = eastmoney_client.fetch_intraday(symbol, min(intraday_missing), max(intraday_missing))
                    self._merge_cache(intraday_target, frame, common_dates, ["symbol", "event_time"])
                if check_missing:
                    frame = eastmoney_client.fetch_daily_check(symbol, min(check_missing), max(check_missing))
                    if "ST" in name.upper():
                        frame = frame.with_columns(pl.lit(True).alias("is_st"))
                    self._merge_cache(check_target, frame, common_dates, ["symbol", "trade_date"])
            finally:
                eastmoney_client.close()
            return
        if provider == "akshare_sina":
            sina_client = AkshareSinaHistoryClient(
                delay_seconds=self.settings.short_term_secondary_delay_seconds,
                timeout_seconds=self.settings.eastmoney_request_timeout_seconds,
            )
            if intraday_missing:
                frame = sina_client.fetch_intraday(symbol, min(intraday_missing), max(intraday_missing))
                self._merge_cache(intraday_target, frame, common_dates, ["symbol", "event_time"])
            if check_missing:
                frame = sina_client.fetch_daily_check(symbol, min(check_missing), max(check_missing))
                if "ST" in name.upper():
                    frame = frame.with_columns(pl.lit(True).alias("is_st"))
                self._merge_cache(check_target, frame, common_dates, ["symbol", "trade_date"])
                self._fill_checks_from_intraday(
                    check_target,
                    intraday_target,
                    common_dates,
                    is_st="ST" in name.upper(),
                )
            return
        if provider == "baostock":
            with BaoStockHistoryClient() as bao_client:
                if intraday_missing:
                    frame = bao_client.fetch_intraday(symbol, min(intraday_missing), max(intraday_missing))
                    self._merge_cache(intraday_target, frame, common_dates, ["symbol", "event_time"])
                if check_missing:
                    frame = bao_client.fetch_daily_check(symbol, min(check_missing), max(check_missing))
                    self._merge_cache(check_target, frame, common_dates, ["symbol", "trade_date"])
            return
        raise RuntimeError(f"不支持的短期历史数据源: {provider}")

    def _fill_checks_from_intraday(
        self,
        check_target: Path,
        intraday_target: Path,
        common_dates: list[date],
        *,
        is_st: bool,
    ) -> None:
        if not intraday_target.exists():
            return
        present: set[date] = set()
        if check_target.exists():
            with suppress(OSError, pl.exceptions.PolarsError):
                present = set(pl.read_parquet(check_target, columns=["trade_date"])["trade_date"].to_list())
        missing = [trade_date for trade_date in common_dates if trade_date not in present]
        if not missing:
            return
        intraday = pl.read_parquet(intraday_target).filter(pl.col("trade_date").is_in(missing))
        derived = daily_checks_from_intraday(intraday, is_st=is_st)
        if not derived.is_empty():
            self._merge_cache(check_target, derived, common_dates, ["symbol", "trade_date"])

    @staticmethod
    def _cache_covers(path: Path, dates: list[date], minimum_rows: int) -> bool:
        if not path.exists():
            return False
        try:
            frame = pl.read_parquet(path, columns=["trade_date"])
        except (OSError, pl.exceptions.PolarsError):
            return False
        present = set(frame["trade_date"].unique().to_list())
        return len(frame) >= minimum_rows and set(dates).issubset(present)

    @staticmethod
    def _dates_to_fetch(path: Path, dates: list[date], minimum_rows: int) -> list[date]:
        if not path.exists():
            return dates
        try:
            frame = pl.read_parquet(path, columns=["trade_date"])
        except (OSError, pl.exceptions.PolarsError):
            return dates
        present = set(frame["trade_date"].unique().to_list())
        missing = [trade_date for trade_date in dates if trade_date not in present]
        if missing:
            return missing
        return dates if frame.height < minimum_rows else []

    @staticmethod
    def _merge_cache(
        target: Path,
        fresh: pl.DataFrame,
        dates: list[date],
        unique_columns: list[str],
    ) -> None:
        frames = [fresh]
        if target.exists():
            with suppress(OSError, pl.exceptions.PolarsError):
                frames.insert(0, pl.read_parquet(target))
        merged = pl.concat(frames, how="diagonal_relaxed").filter(pl.col("trade_date").is_in(dates)).unique(
            subset=unique_columns, keep="last"
        ).sort(unique_columns)
        write_cache_atomic(merged, target)

    def _write_partial_coverage(self, common_dates: list[date], symbols: list[str]) -> None:
        daily = pl.scan_parquet(str(self.daily_root / "*.parquet")).collect()
        complete_daily = (
            daily.group_by("symbol")
            .agg(pl.col("trade_date").n_unique().alias("days"))
            .filter(pl.col("days") == WINDOW_DAYS)
        )
        daily_coverage = complete_daily.height / max(1, len(symbols))
        manifest: dict[str, Any] = {
            "version": self.version,
            "mode": "short_term",
            "status": "partial",
            "start": common_dates[0].isoformat(),
            "end": common_dates[-1].isoformat(),
            "distinct_trade_dates": len(common_dates),
            "daily_rows": daily.height,
            "intraday_rows": 0,
            "daily_coverage": daily_coverage,
            "intraday_coverage": 0.0,
            "universe_symbols": len(symbols),
            "passed": False,
            "blocked_reasons": ["30分钟K线与跨源日线校验尚未完成。"],
            "source_checks": [
                {
                    "provider": "eastmoney",
                    "datasets": ["15日日线"],
                    "status": "ready",
                    "message": f"已找到{len(common_dates)}个共同交易日，完整覆盖率{daily_coverage:.2%}。",
                },
                {
                    "provider": self.secondary_provider,
                    "datasets": ["30分钟K线", "跨源日线校验"],
                    "status": "partial",
                    "message": "数据源预检已通过，逐股缓存尚未完成。",
                },
            ],
            "created_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(self.artifact_root / "data_coverage.json", manifest)
        self.run.daily_rows = daily.height
        self._write_run()

    def _write_source_failure(self, message: str) -> None:
        path = self.artifact_root / "data_coverage.json"
        manifest: dict[str, Any] = {}
        if path.exists():
            with suppress(OSError, ValueError):
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
        manifest.update(
            {
                "version": self.version,
                "mode": "short_term",
                "status": "partial",
                "passed": False,
                "source_checks": [
                    {
                        "provider": "eastmoney",
                        "datasets": ["15日日线"],
                        "status": "partial",
                        "message": "保留已有短期日线缓存。",
                    },
                    {
                        "provider": self.secondary_provider,
                        "datasets": ["30分钟K线", "跨源日线校验"],
                        "status": "unavailable",
                        "message": message,
                    },
                ],
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        atomic_json(path, manifest)

    def _quality(self, common_dates: list[date], symbols: list[str]) -> dict[str, Any]:
        self._update("quality", STAGES["quality"], "检查15日共同交易日、逐股覆盖与跨源一致性。")
        daily = pl.scan_parquet(str(self.daily_root / "*.parquet")).collect()
        supported = [symbol for symbol in symbols if symbol.endswith((".SH", ".SZ"))]
        complete_daily = (
            daily.group_by("symbol").agg(pl.col("trade_date").n_unique().alias("days")).filter(pl.col("days") == 15)
        )
        daily_coverage = complete_daily.height / max(1, len(symbols))
        intraday_paths = list(self.intraday_root.glob("*.parquet"))
        intraday_rows = 0
        complete_intraday = 0
        for path in intraday_paths:
            frame = pl.read_parquet(path, columns=["trade_date"])
            intraday_rows += frame.height
            if frame.height >= INTRADAY_BARS and len(set(common_dates) & set(frame["trade_date"].unique())) == 15:
                complete_intraday += 1
        intraday_coverage = complete_intraday / max(1, len(supported))
        check_paths = list(self.check_root.glob("*.parquet"))
        complete_checks = 0
        check_rows = 0
        if check_paths:
            checks = pl.scan_parquet(str(self.check_root / "*.parquet")).collect()
            check_rows = checks.height
            complete_checks = (
                checks.group_by("symbol")
                .agg(pl.col("trade_date").n_unique().alias("days"))
                .filter(pl.col("days") == WINDOW_DAYS)
                .height
            )
        daily_check_coverage = complete_checks / max(1, len(supported))
        reasons: list[str] = []
        if len(common_dates) != WINDOW_DAYS:
            reasons.append(f"共同交易日仅{len(common_dates)}日，要求15日。")
        if daily_coverage < self.settings.short_term_daily_coverage:
            reasons.append(
                f"15日日线覆盖率{daily_coverage:.2%}，低于{self.settings.short_term_daily_coverage:.0%}。"
            )
        if intraday_coverage < self.settings.short_term_intraday_coverage:
            reasons.append(
                f"120根30分钟K线覆盖率{intraday_coverage:.2%}，低于{self.settings.short_term_intraday_coverage:.0%}。"
            )
        if daily_check_coverage < self.settings.short_term_daily_coverage:
            reasons.append(
                f"跨源日线校验覆盖率{daily_check_coverage:.2%}，"
                f"低于{self.settings.short_term_daily_coverage:.0%}。"
            )
        manifest: dict[str, Any] = {
            "version": self.version,
            "mode": "short_term",
            "status": "ready" if not reasons else "partial",
            "start": common_dates[0].isoformat(),
            "end": common_dates[-1].isoformat(),
            "distinct_trade_dates": len(common_dates),
            "daily_rows": daily.height,
            "intraday_rows": intraday_rows,
            "daily_check_rows": check_rows,
            "daily_coverage": daily_coverage,
            "intraday_coverage": intraday_coverage,
            "daily_check_coverage": daily_check_coverage,
            "universe_symbols": len(symbols),
            "passed": not reasons,
            "blocked_reasons": reasons,
            "source_checks": [
                {
                    "provider": "eastmoney",
                    "datasets": ["15日日线"],
                    "status": "ready",
                    "message": f"完整覆盖率{daily_coverage:.2%}。",
                },
                {
                    "provider": self.secondary_provider,
                    "datasets": ["30分钟K线", "跨源日线校验"],
                    "status": (
                        "ready"
                        if intraday_coverage >= self.settings.short_term_intraday_coverage
                        and daily_check_coverage >= self.settings.short_term_daily_coverage
                        else "partial"
                    ),
                    "message": (
                        f"120根K线覆盖率{intraday_coverage:.2%}；"
                        f"跨源日线覆盖率{daily_check_coverage:.2%}。"
                    ),
                },
            ],
            "created_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(self.artifact_root / "data_coverage.json", manifest)
        self.run.daily_rows = daily.height
        self.run.intraday_rows = intraday_rows
        self._write_run()
        return manifest

    def _train_and_publish(
        self,
        common_dates: list[date],
        universe: dict[str, dict[str, str]],
        coverage: dict[str, Any],
    ) -> None:
        self._update("features", STAGES["features"], "构建15日PIT特征和下一交易日标签。")
        daily = build_short_features(self.daily_root, self.check_root, self.intraday_root, common_dates)

        self._update("hmm", STAGES["hmm"], "训练15日两状态HMM市场环境模型。")
        market, hmm_columns = short_market_features(daily)
        hmm = RegimeHMM(n_components=2, random_state=self.settings.random_seed).fit(
            market.select(hmm_columns).to_numpy()
        )
        hmm_probabilities = hmm.predict(market.select(hmm_columns).to_numpy())
        hmm_frame = market.select("trade_date").with_columns(
            [pl.Series(f"hmm_p{index}", hmm_probabilities[:, index]) for index in range(2)]
        )
        hmm_path = model_dir(self.artifact_root, "hmm") / f"{self.run_id}.pkl"
        hmm.save(hmm_path)
        write_component(
            self.artifact_root,
            "hmm",
            self.run_id,
            hmm_path,
            {
                "states": 2,
                "observations": market.height,
                "converged": bool(hmm.model.monitor_.converged),
                "probability_sum_max_error": float(np.abs(hmm_probabilities.sum(axis=1) - 1).max()),
            },
        )
        daily = daily.join(hmm_frame, on="trade_date", how="left")

        self._update("tft", STAGES["tft"], "训练5日编码、1日目标的短周期TFT。")
        tft_frame, tft_metrics, tft_path = train_short_tft(
            daily,
            self.artifact_root,
            self.run_id,
            lambda completed, total: self._update(
                "tft",
                73 + min(6, int(6 * completed / max(1, total))),
                f"短周期TFT epoch {completed}/{total}。",
            ),
        )
        write_component(self.artifact_root, "tft", self.run_id, tft_path, tft_metrics)
        daily = daily.join(tft_frame, on=["symbol", "trade_date"], how="left")

        self._update("patchtst", STAGES["patchtst"], "训练120根30分钟K线PatchTST并生成embedding。")
        patch_frame, patch_metrics, patch_path = train_short_patchtst(
            self.intraday_root,
            self.artifact_root,
            self.run_id,
            lambda phase, completed, total: self._update(
                "patchtst",
                80 + min(6, int(6 * completed / max(1, total))),
                f"PatchTST {phase} {completed}/{total}。",
            ),
        )
        write_component(self.artifact_root, "patchtst", self.run_id, patch_path, patch_metrics)
        daily = daily.join(patch_frame, on=["symbol", "trade_date"], how="left")

        self._update("ranker", STAGES["ranker"], "训练15日LambdaRank并执行3个隔离滚动窗口。")
        patch_features = [f"patch_{index}" for index in range(64)]
        context_features = ["tft_q50", "hmm_p0", "hmm_p1"]
        features = SHORT_BASE_FEATURES + context_features + patch_features
        prepared = prepare_short_ranker_frame(daily, features)
        result = walk_forward_short_rankers(
            prepared,
            features,
            lambda completed, total: self._update(
                "ranker",
                87 + min(5, int(5 * completed / max(1, total))),
                f"LambdaRank滚动窗口 {completed}/{total}。",
            ),
        )
        ranker_path = model_dir(self.artifact_root, "ranker") / f"{self.run_id}.txt"
        cast(ArtRanker, result["model"]).save(ranker_path)
        write_component(
            self.artifact_root,
            "ranker",
            self.run_id,
            ranker_path,
            {
                "rank_ic": result["rank_ic"],
                "ndcg_at_20": result["ndcg_at_20"],
                "validation_days": result["validation_days"],
                "walk_forward_windows": 3,
                "label_distribution": result["label_distribution"],
            },
        )

        self._update("validation", STAGES["validation"], "执行短期实验门禁、打乱标签和扣费检查。")
        gate = short_publication_gate(prepared, result, coverage)
        self._update("ensemble", STAGES["ensemble"], "冻结短期实验集成权重并记录弱样本警告。")
        ensemble = {
            "version": f"short-ensemble-{self.run_id}",
            "status": "healthy" if gate["passed"] else "blocked",
            "publish_gate_passed": gate["passed"],
            "weight_sum": 1.0,
            "weights": {"ranker": 0.7, "tft": 0.3},
            "lagged_rank_ic": result["rank_ic"],
            "horizon_consistent": True,
            "metrics": gate,
            "blocked_reasons": gate["blocked_reasons"],
            "warnings": gate["warnings"],
            "last_trained_at": datetime.now(UTC).isoformat(),
        }
        ensemble_root = model_dir(self.artifact_root, "ensemble")
        atomic_json(ensemble_root / "latest.json", ensemble)
        if not gate["passed"]:
            raise ShortTermGateBlocked("；".join(cast(list[str], gate["blocked_reasons"])))
        atomic_json(ensemble_root / "champion.json", ensemble)

        self._update("publish", STAGES["publish"], "生成下一交易日短期实验排名。")
        publish_short_predictions(
            prepared,
            result,
            hmm_probabilities[-1],
            universe,
            self.version,
            self.run_id,
            self.prediction_root,
            features,
            cast(list[str], gate["warnings"]),
        )

    def _latest_trade_date(self) -> date:
        path = self.settings.artifact_root / "published" / "market" / "stable.json"
        if path.exists():
            body = json.loads(path.read_text(encoding="utf-8"))
            value = body.get("data", {}).get("trade_date")
            if value:
                return date.fromisoformat(value)
        return date.today()

    def _update(self, stage: str, progress: int, message: str) -> None:
        self.run.stage = stage
        self.run.stage_label = STAGE_LABELS.get(stage, stage)
        self.run.progress = progress
        self.run.message = message
        check_run_control(self.settings, self.run_path, self.run)
        self._write_run()

    def _set_stage(self, stage: str, progress: int, message: str, total_items: int) -> None:
        self.run.stage_completed_items = 0
        self.run.stage_total_items = total_items
        self._update(stage, progress, message)

    def _quarantine(self, dataset: str, symbol: str, reason: str) -> None:
        root = self.settings.data_root / "quarantine" / self.version
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{dataset}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"symbol": symbol, "reason": reason}, ensure_ascii=False) + "\n")

    def _read_run(self) -> TrainingRunSummary:
        return TrainingRunSummary.model_validate_json(self.run_path.read_text(encoding="utf-8"))

    def _write_run(self) -> None:
        atomic_json(self.run_path, self.run.model_dump(mode="json"))

    def _write_checkpoint(self, reason: str) -> None:
        atomic_json(
            self.run_path.with_suffix(".checkpoint"),
            {
                "run_id": self.run_id,
                "mode": "short_term",
                "reason": reason,
                "stage": self.run.stage,
                "progress": self.run.progress,
                "data_version": self.version,
                "daily_cache": str(self.daily_root),
                "intraday_cache": str(self.intraday_root),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

    def _duration(self, finished_at: datetime) -> float:
        return (finished_at - (self.run.started_at or self.started)).total_seconds()

    def _update_eta(self, completed: int, total: int) -> None:
        elapsed = (datetime.now(UTC) - (self.run.started_at or self.started)).total_seconds()
        self.run.elapsed_seconds = elapsed
        self.run.eta_seconds = elapsed / completed * max(0, total - completed) if completed else None

    def _adaptive_workers(self, configured: int) -> int:
        used = self.run.memory_rss_mb or 0.0
        soft_limit = self.settings.model_max_memory_gb * 1024 * 0.8
        if used >= soft_limit:
            reduced = max(1, configured // 2)
            self.run.message = f"内存达到80%软限制，并发已从{configured}降至{reduced}。"
            self._write_run()
            return reduced
        return configured


def select_common_trading_dates(root: Path, universe_size: int, days: int = WINDOW_DAYS) -> list[date]:
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        raise ShortTermGateBlocked("尚无短期日线缓存。")
    counts = (
        pl.scan_parquet(str(root / "*.parquet"))
        .group_by("trade_date")
        .agg(pl.col("symbol").n_unique().alias("symbols"))
        .collect()
        .filter(pl.col("symbols") >= math.ceil(universe_size * 0.95))
        .sort("trade_date")
    )
    selected = counts["trade_date"].tail(days).to_list()
    if len(selected) != days:
        raise ShortTermGateBlocked(f"仅找到{len(selected)}个覆盖率达到95%的共同交易日。")
    return cast(list[date], selected)


def failure_circuit_open(consecutive_failures: int, threshold: int) -> bool:
    return consecutive_failures >= threshold


def daily_checks_from_intraday(frame: pl.DataFrame, *, is_st: bool) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "trade_date": pl.Date,
                "close_baostock": pl.Float64,
                "trade_status_baostock": pl.String,
                "is_st": pl.Boolean,
                "check_source": pl.String,
            }
        )
    return (
        frame.sort(["symbol", "event_time"])
        .group_by(["symbol", "trade_date"], maintain_order=True)
        .agg(
            pl.col("close").last().alias("close_baostock"),
            pl.col("volume").sum().alias("_daily_volume"),
        )
        .with_columns(
            pl.when(pl.col("_daily_volume") > 0)
            .then(pl.lit("trading"))
            .otherwise(pl.lit("suspended"))
            .alias("trade_status_baostock"),
            pl.lit(is_st).alias("is_st"),
            pl.lit("akshare_sina_intraday_close").alias("check_source"),
        )
        .drop("_daily_volume")
    )


def build_short_features(
    daily_root: Path,
    check_root: Path,
    intraday_root: Path,
    common_dates: list[date],
) -> pl.DataFrame:
    daily = (
        pl.scan_parquet(str(daily_root / "*.parquet"))
        .filter(pl.col("trade_date").is_in(common_dates))
        .collect()
    )
    checks = list(check_root.glob("*.parquet"))
    if checks:
        check = pl.scan_parquet(str(check_root / "*.parquet")).filter(
            pl.col("trade_date").is_in(common_dates)
        ).collect()
        daily = daily.join(check, on=["symbol", "trade_date"], how="left")
    else:
        daily = daily.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("close_baostock"),
            pl.lit("unknown").alias("trade_status_baostock"),
            pl.lit(True).alias("is_st"),
        )
    intraday = short_intraday_daily_features(intraday_root, common_dates)
    ordered = daily.sort(["symbol", "trade_date"]).with_columns(
        pl.col("close_forward_adjusted").pct_change().over("symbol").alias("daily_return"),
        *[
            (pl.col("close_forward_adjusted") / pl.col("close_forward_adjusted").shift(window).over("symbol") - 1)
            .alias(f"momentum_{window}d")
            for window in (1, 3, 5, 10)
        ],
        pl.col("amount").log1p().alias("log_amount"),
        pl.col("amount").rolling_median(5).over("symbol").alias("median_amount_5d"),
        pl.col("symbol").count().over("symbol").alias("available_days"),
    ).with_columns(
        pl.col("daily_return").rolling_std(5).over("symbol").alias("volatility_5d"),
        pl.col("daily_return").rolling_std(10).over("symbol").alias("volatility_10d"),
        (pl.col("amount") / pl.col("amount").rolling_mean(5).over("symbol")).alias("turnover_proxy_5d"),
        (pl.col("close_forward_adjusted").shift(-1).over("symbol") / pl.col("close_forward_adjusted") - 1).alias(
            "forward_return"
        ),
    ).with_columns(
        (pl.col("forward_return") - pl.col("forward_return").mean().over("trade_date")).alias(
            "forward_excess_return"
        )
    ).with_columns(
        (
            ((pl.col("forward_excess_return").rank("ordinal").over("trade_date") - 1) * 5 / pl.len().over("trade_date"))
            .floor()
            .clip(0, 4)
            .cast(pl.Int8)
        ).alias("relevance"),
        (
            (pl.col("available_days") == WINDOW_DAYS)
            & (pl.col("median_amount_5d") >= 50_000_000)
            & (pl.col("volume") > 0)
            & ~pl.col("is_st").fill_null(True)
            & (pl.col("trade_status_baostock").fill_null("unknown") == "trading")
            & ((pl.col("close") / pl.col("close_baostock") - 1).abs().fill_null(1.0) <= 0.005)
        ).alias("eligible"),
    )
    return ordered.join(intraday, on=["symbol", "trade_date"], how="left")


def short_intraday_daily_features(root: Path, dates: list[date]) -> pl.DataFrame:
    if not list(root.glob("*.parquet")):
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "trade_date": pl.Date,
                "intraday_return": pl.Float64,
                "intraday_amplitude": pl.Float64,
                "intraday_log_amount": pl.Float64,
            }
        )
    return (
        pl.scan_parquet(str(root / "*.parquet"))
        .filter(pl.col("trade_date").is_in(dates))
        .sort(["symbol", "event_time"])
        .group_by(["symbol", "trade_date"])
        .agg(
            (pl.col("close").last() / pl.col("open").first() - 1).alias("intraday_return"),
            (pl.col("high").max() / pl.col("low").min() - 1).alias("intraday_amplitude"),
            pl.col("amount").sum().log1p().alias("intraday_log_amount"),
        )
        .collect()
    )


def short_market_features(daily: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    market = (
        daily.group_by("trade_date")
        .agg(
            pl.col("daily_return").mean().fill_null(0.0).alias("market_return"),
            (pl.col("daily_return").fill_null(0.0) > 0).mean().alias("breadth"),
            pl.col("daily_return").std().fill_null(0.0).alias("cross_section_volatility"),
            pl.col("amount").sum().log1p().alias("market_log_turnover"),
        )
        .sort("trade_date")
    )
    columns = ["market_return", "breadth", "cross_section_volatility", "market_log_turnover"]
    normalized = market.select(
        [
            ((pl.col(column) - pl.col(column).mean()) / pl.col(column).std())
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias(column)
            for column in columns
        ]
    )
    return pl.concat([market.select("trade_date"), normalized], how="horizontal_extend"), columns


def train_short_tft(
    daily: pl.DataFrame,
    artifact_root: Path,
    run_id: str,
    update: Callable[[int, int], None] | None = None,
) -> tuple[pl.DataFrame, dict[str, float | int | str | bool | list[int]], Path]:
    columns = ["daily_return", "momentum_1d", "momentum_3d", "momentum_5d", "log_amount", "turnover_proxy_5d"]
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    target_dates: list[date] = []
    inference: list[np.ndarray] = []
    keys: list[tuple[str, date]] = []
    for group in daily.sort(["symbol", "trade_date"]).partition_by("symbol"):
        values = group.select(columns).fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32)
        target_values = group["forward_excess_return"].to_numpy()
        dates = group["trade_date"].to_list()
        symbol = str(group["symbol"][0])
        for index in range(4, len(values)):
            sequence = values[index - 4 : index + 1]
            inference.append(sequence)
            keys.append((symbol, dates[index]))
            target = target_values[index]
            if target is not None and np.isfinite(target):
                sequences.append(sequence)
                targets.append(float(target))
                target_dates.append(dates[index])
    if len(sequences) < 1_000:
        raise RuntimeError(f"Short TFT only has {len(sequences)} labelled sequences")
    train_x = torch.tensor(np.stack(sequences), dtype=torch.float32)
    train_y = torch.tensor(np.asarray(targets)[:, None], dtype=torch.float32)
    model = TftStyleForecaster(input_size=len(columns), horizons=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    quantiles = torch.tensor([0.1, 0.5, 0.9])
    unique_target_dates = sorted(set(target_dates))
    if len(unique_target_dates) < 3:
        raise RuntimeError("Short TFT requires at least three labelled dates")
    split_date = unique_target_dates[-2]
    train_indices = [index for index, trade_date in enumerate(target_dates) if trade_date < split_date]
    validation_indices = [index for index, trade_date in enumerate(target_dates) if trade_date >= split_date]
    loader = DataLoader(
        TensorDataset(train_x[train_indices], train_y[train_indices]), batch_size=128, shuffle=True
    )
    best = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    for epoch in range(12):
        model.train()
        for inputs, target in loader:
            optimizer.zero_grad()
            loss = quantile_loss(model(inputs), target, quantiles)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation = float(
                quantile_loss(model(train_x[validation_indices]), train_y[validation_indices], quantiles)
            )
        if validation < best:
            best = validation
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if update is not None:
                update(epoch + 1, 12)
            if patience >= 3:
                break
        if update is not None and patience == 0:
            update(epoch + 1, 12)
    if best_state is None or not math.isfinite(best):
        raise RuntimeError("Short TFT validation loss is not finite")
    model.load_state_dict(best_state)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(inference), 1024):
            inputs = torch.tensor(np.stack(inference[start : start + 1024]), dtype=torch.float32)
            batches.append(model(inputs)[:, 0, 1].numpy())
    predictions = np.concatenate(batches)
    frame = pl.DataFrame(
        {
            "symbol": [key[0] for key in keys],
            "trade_date": [key[1] for key in keys],
            "tft_q50": predictions,
        }
    )
    path = model_dir(artifact_root, "tft") / f"{run_id}.pt"
    torch.save({"state_dict": model.state_dict(), "input_size": len(columns), "encoder": 5, "horizon": 1}, path)
    return frame, {
        "validation_quantile_loss": best,
        "encoder_days": 5,
        "horizon_days": 1,
        "validation_dates": len(set(target_dates[index] for index in validation_indices)),
        "time_ordered_split": True,
    }, path


def train_short_patchtst(
    intraday_root: Path,
    artifact_root: Path,
    run_id: str,
    update: Callable[[str, int, int], None] | None = None,
) -> tuple[pl.DataFrame, dict[str, float | int | str | bool | list[int]], Path]:
    sequences: list[np.ndarray] = []
    for path in sorted(intraday_root.glob("*.parquet")):
        frame = pl.read_parquet(path).sort("event_time")
        if frame.height < INTRADAY_BARS:
            continue
        matrix = intraday_matrix(frame)
        sequences.append(matrix[-INTRADAY_BARS:])
    if len(sequences) < 1_000:
        raise RuntimeError(f"Short PatchTST only has {len(sequences)} valid 120-bar sequences")
    values = torch.tensor(np.stack(sequences), dtype=torch.float32)
    model = PatchTSTEncoder(channels=5)
    head = nn.Linear(64, 5)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-3)
    loader = DataLoader(TensorDataset(values), batch_size=128, shuffle=True)
    best = math.inf
    for epoch in range(6):
        model.train()
        losses: list[float] = []
        for (inputs,) in loader:
            mask = torch.rand_like(inputs) < 0.15
            optimizer.zero_grad()
            prediction = head(model(inputs.masked_fill(mask, 0)))
            loss = nn.functional.mse_loss(prediction, inputs[:, -1])
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(loss.detach()))
        best = min(best, float(np.mean(losses)))
        if update is not None:
            update("epoch", epoch + 1, 6)
    model.eval()
    rows: list[dict[str, Any]] = []
    batch_windows: list[np.ndarray] = []
    batch_keys: list[tuple[str, date]] = []

    def flush_embeddings() -> None:
        if not batch_windows:
            return
        inputs = torch.tensor(np.stack(batch_windows), dtype=torch.float32)
        with torch.inference_mode():
            embeddings = model(inputs).numpy()
        for (symbol, trade_date), embedding in zip(batch_keys, embeddings, strict=True):
            row: dict[str, Any] = {"symbol": symbol, "trade_date": trade_date}
            row.update({f"patch_{index}": float(value) for index, value in enumerate(embedding)})
            rows.append(row)
        batch_windows.clear()
        batch_keys.clear()

    embedding_paths = sorted(intraday_root.glob("*.parquet"))
    for file_index, parquet_path in enumerate(embedding_paths, start=1):
        frame = pl.read_parquet(parquet_path).sort("event_time")
        if frame.is_empty():
            continue
        matrix = intraday_matrix(frame)
        symbol = str(frame["symbol"][0])
        last_indices: dict[date, int] = {}
        for index, trade_date in enumerate(frame["trade_date"].to_list()):
            last_indices[trade_date] = index
        for trade_date, index in last_indices.items():
            batch_windows.append(padded_intraday_window(matrix, index))
            batch_keys.append((symbol, trade_date))
            if len(batch_windows) >= 512:
                flush_embeddings()
        if update is not None:
            update("embedding文件", file_index, len(embedding_paths))
    flush_embeddings()
    path = model_dir(artifact_root, "patchtst") / f"{run_id}.pt"
    torch.save({"state_dict": model.state_dict(), "channels": 5, "embedding_dim": 64, "sequence": 120}, path)
    return pl.DataFrame(rows), {
        "masked_mse": best,
        "embedding_dim": 64,
        "sequence_bars": 120,
        "complete_sequences": len(sequences),
        "materialized_sequences": len(rows),
        "left_padding_for_early_dates": True,
    }, path


def intraday_matrix(frame: pl.DataFrame) -> np.ndarray:
    values = frame.select("open", "high", "low", "close", "volume").to_numpy().astype(np.float32)
    prices = np.clip(values[:, :4], 1e-6, None)
    returns = np.vstack([np.zeros((1, 4), dtype=np.float32), np.diff(np.log(prices), axis=0)])
    volume = np.log1p(np.clip(values[:, 4:5], 0, None))
    volume = (volume - volume.mean()) / max(float(volume.std()), 1e-6)
    return np.concatenate([returns, volume], axis=1)


def padded_intraday_window(matrix: np.ndarray, end: int, length: int = INTRADAY_BARS) -> np.ndarray:
    start = max(0, end - length + 1)
    window = matrix[start : end + 1]
    if len(window) == length:
        return window
    padding = np.zeros((length - len(window), matrix.shape[1]), dtype=np.float32)
    return np.concatenate([padding, window], axis=0)


def prepare_short_ranker_frame(frame: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    missing = [column for column in features if column not in frame.columns]
    prepared = frame.with_columns([pl.lit(0.0).alias(column) for column in missing])
    return prepared.with_columns(
        [
            pl.when(pl.col(column).cast(pl.Float64).is_finite())
            .then(pl.col(column).cast(pl.Float64))
            .otherwise(0.0)
            .alias(column)
            for column in features
        ]
    ).filter(pl.col("eligible"))


def short_walk_forward_splits(dates: list[date]) -> list[tuple[list[date], list[date], list[date]]]:
    if len(dates) < 14:
        raise RuntimeError(f"Only {len(dates)} labelled dates; short validation requires 14")
    # Each fold uses an expanding train window, a one-day embargo, and two untouched test days.
    return [
        (dates[:6], dates[6:7], dates[7:9]),
        (dates[:8], dates[8:9], dates[9:11]),
        (dates[:10], dates[10:11], dates[11:13]),
    ]


def walk_forward_short_rankers(
    frame: pl.DataFrame,
    features: list[str],
    update: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    labelled = frame.filter(pl.col("relevance").is_not_null())
    dates = sorted(cast(list[date], labelled["trade_date"].unique().to_list()))
    splits = short_walk_forward_splits(dates)
    daily_ics: list[float] = []
    ndcgs: list[float] = []
    latest_scores: list[np.ndarray] = []
    test_frames: list[pl.DataFrame] = []
    latest_date = cast(date, frame["trade_date"].max())
    latest = frame.filter(pl.col("trade_date") == latest_date).sort("symbol")
    for window_index, (train_dates, _embargo, test_dates) in enumerate(splits, start=1):
        train = labelled.filter(pl.col("trade_date").is_in(train_dates))
        test = labelled.filter(pl.col("trade_date").is_in(test_dates))
        model = fit_short_ranker(train, features)
        scores = np.asarray(model.predict(test.select(features).to_numpy()), dtype=float)
        scored = test.select("trade_date", "forward_excess_return").with_columns(pl.Series("score", scores))
        daily_ics.extend(daily_rank_ics(scored))
        ndcgs.append(ndcg_at_20(scored))
        test_frames.append(scored)
        latest_scores.append(np.asarray(model.predict(latest.select(features).to_numpy()), dtype=float))
        if update is not None:
            update(window_index, len(splits))
    final_model = fit_short_ranker(labelled, features)
    labels, counts = np.unique(labelled["relevance"].to_numpy(), return_counts=True)
    oos = pl.concat(test_frames)
    cost_adjusted_returns: list[float] = []
    for group in oos.partition_by("trade_date"):
        top_return = group.sort("score", descending=True).head(20)["forward_excess_return"].mean()
        if top_return is not None:
            cost_adjusted_returns.append(float(np.asarray(top_return, dtype=float).item()) - 0.0015)
    all_latest_scores = np.concatenate(latest_scores)
    return {
        "model": final_model,
        "rank_ic": float(np.mean(daily_ics)) if daily_ics else 0.0,
        "daily_ics": daily_ics,
        "ndcg_at_20": float(np.mean(ndcgs)),
        "validation_days": len(daily_ics),
        "label_distribution": [
            int(counts[labels.tolist().index(index)]) if index in labels else 0 for index in range(5)
        ],
        "latest_scores": latest_scores,
        "oos_scores": oos["score"].to_numpy(),
        "oos_returns": oos["forward_excess_return"].to_numpy(),
        "cost_adjusted_top20_return": float(np.mean(cost_adjusted_returns)) if cost_adjusted_returns else 0.0,
        "finite_outputs": bool(
            np.isfinite(oos["score"].to_numpy()).all()
            and np.isfinite(oos["forward_excess_return"].to_numpy()).all()
            and np.isfinite(all_latest_scores).all()
        ),
        "no_future_data": all(
            max(train_dates) < embargo_dates[0] < min(test_dates)
            for train_dates, embargo_dates, test_dates in splits
        ),
    }


def fit_short_ranker(frame: pl.DataFrame, features: list[str]) -> ArtRanker:
    ordered = frame.sort(["trade_date", "symbol"])
    groups = ordered.group_by("trade_date", maintain_order=True).len()["len"].to_list()
    if not groups or min(groups) < 20:
        raise RuntimeError("Short LambdaRank needs at least 20 eligible stocks per training day")
    model = ArtRanker(
        n_estimators=160,
        learning_rate=0.04,
        num_leaves=23,
        max_bin=63,
        n_jobs=8,
        verbosity=-1,
        random_state=474,
    )
    model.fit(ordered.select(features).to_numpy(), ordered["relevance"].to_numpy(), group=groups)
    return model


def daily_rank_ics(scored: pl.DataFrame) -> list[float]:
    values: list[float] = []
    for group in scored.partition_by("trade_date"):
        if group.height >= 20:
            values.append(rank_ic(group["score"].to_numpy(), group["forward_excess_return"].to_numpy()))
    return values


def ndcg_at_20(scored: pl.DataFrame) -> float:
    values: list[float] = []
    for group in scored.partition_by("trade_date"):
        ordered = group.sort("score", descending=True).head(20)
        gains = np.maximum(0, ordered["forward_excess_return"].to_numpy())
        discounts = 1 / np.log2(np.arange(2, len(gains) + 2))
        ideal = np.sort(np.maximum(0, group["forward_excess_return"].to_numpy()))[::-1][:20]
        denominator = float((ideal * discounts[: len(ideal)]).sum())
        values.append(float((gains * discounts).sum() / denominator) if denominator else 0.0)
    return float(np.mean(values)) if values else 0.0


def short_publication_gate(frame: pl.DataFrame, result: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    ics = np.asarray(result["daily_ics"], dtype=float)
    rng = np.random.default_rng(474)
    bootstrap = (
        [float(rng.choice(ics, size=len(ics), replace=True).mean()) for _ in range(1_000)]
        if len(ics)
        else [0.0]
    )
    shuffled_returns = np.asarray(result["oos_returns"], dtype=float).copy()
    rng.shuffle(shuffled_returns)
    shuffled_ic = rank_ic(np.asarray(result["oos_scores"], dtype=float), shuffled_returns)
    checks = {
        "fifteen_common_days": int(coverage["distinct_trade_dates"]) == 15,
        "daily_coverage": float(coverage["daily_coverage"]) >= 0.95,
        "intraday_coverage": float(coverage["intraday_coverage"]) >= 0.85,
        "three_validation_days": int(result["validation_days"]) >= 3,
        "finite_outputs": bool(np.isfinite(ics).all() and result["finite_outputs"]),
        "positive_rank_ic": float(result["rank_ic"]) > 0,
        "shuffled_near_random": abs(float(shuffled_ic)) < 0.05,
        "no_future_data": bool(result["no_future_data"]),
    }
    reason_labels = {
        "fifteen_common_days": "共同交易日不足15日。",
        "daily_coverage": "日线股票覆盖率低于95%。",
        "intraday_coverage": "沪深30分钟K线覆盖率低于85%。",
        "three_validation_days": "有效样本外验证日少于3日。",
        "finite_outputs": "模型输出含非有限数值。",
        "positive_rank_ic": "平均样本外Rank IC未大于0。",
        "shuffled_near_random": "打乱标签Rank IC绝对值未低于0.05。",
        "no_future_data": "滚动验证检测到未来数据泄漏。",
    }
    reasons = [reason_labels[name] for name, passed in checks.items() if not passed]
    bootstrap_lower = float(np.quantile(bootstrap, 0.025))
    bootstrap_upper = float(np.quantile(bootstrap, 0.975))
    cost_adjusted = float(result["cost_adjusted_top20_return"])
    warnings = [
        "仅使用15个交易日，统计显著性弱于长期模型。",
        f"Bootstrap 95% Rank IC区间 [{bootstrap_lower:.4f}, {bootstrap_upper:.4f}]，仅作警告。",
        f"扣费后Top20平均超额收益 {cost_adjusted:.4%}，仅作警告。",
    ]
    return {
        "passed": not reasons,
        "blocked_reasons": reasons,
        "warnings": warnings,
        "rank_ic": result["rank_ic"],
        "bootstrap_95_lower": bootstrap_lower,
        "bootstrap_95_upper": bootstrap_upper,
        "cost_adjusted_top20_return": cost_adjusted,
        "shuffled_rank_ic": float(shuffled_ic),
        "checks": checks,
    }


def publish_short_predictions(
    frame: pl.DataFrame,
    result: dict[str, Any],
    hmm_probabilities: np.ndarray,
    universe: dict[str, dict[str, str]],
    data_version: str,
    run_id: str,
    prediction_root: Path,
    features: list[str],
    warnings: list[str],
) -> None:
    latest_date = cast(date, frame["trade_date"].max())
    latest = frame.filter(pl.col("trade_date") == latest_date).sort("symbol")
    ranker_scores = np.asarray(
        cast(ArtRanker, result["model"]).predict(latest.select(features).to_numpy()), dtype=float
    )
    tft_scores = latest["tft_q50"].to_numpy().astype(float)
    scores = 0.7 * percentile_calibrate(ranker_scores) + 0.3 * percentile_calibrate(tft_scores)
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    history = np.stack(cast(list[np.ndarray], result["latest_scores"]))
    rank_history = np.argsort(np.argsort(-history, axis=1), axis=1)
    stability = 100 * (1 - rank_history.std(axis=0) / max(1, len(latest)))
    calibrated = percentile_calibrate(scores)
    candidates: list[PredictionCandidate] = []
    for index, row in enumerate(latest.iter_rows(named=True)):
        symbol = str(row["symbol"])
        info = universe.get(symbol, {"name": symbol, "industry": "未分类"})
        patch_contribution = float(sum(abs(float(row.get(f"patch_{item}") or 0)) for item in range(64)))
        contributions = {
            "LambdaRank": float(ranker_scores[index]),
            "TFT 1D": float(tft_scores[index]),
            "PatchTST": patch_contribution,
            "1日动量": float(row["momentum_1d"] or 0),
            "5日动量": float(row["momentum_5d"] or 0),
        }
        candidates.append(
            PredictionCandidate(
                symbol=symbol,
                name=info["name"],
                industry=info["industry"],
                exchange=symbol.rsplit(".", 1)[-1],
                rank=int(ranks[index]),
                close=float(row["close"]),
                art_score=float(scores[index] * 100),
                score_percentile=float(calibrated[index] * 100),
                stability_score=float(np.clip(stability[index], 0, 100)),
                momentum_1d=float(row["momentum_1d"] or 0),
                momentum_3d=float(row["momentum_3d"] or 0),
                momentum_5d=float(row["momentum_5d"] or 0),
                momentum_10d=float(row["momentum_10d"] or 0),
                liquidity_score=float(row["turnover_proxy_5d"] or 0),
                volume_progress=1.0,
                patchtst_contribution=patch_contribution,
                tft_contribution=float(tft_scores[index]),
                ranker_score=float(ranker_scores[index]),
                contributions=contributions,
                risk_flags=["15日弱样本"],
            )
        )
    candidates.sort(key=lambda item: item.rank)
    target_date = latest_date + timedelta(days=1)
    while target_date.weekday() >= 5:
        target_date += timedelta(days=1)
    feed = PredictionFeed(
        status="published",
        message="15日短期实验门禁已通过；排名仅供研究，不属于长期正式预测。",
        prediction_at=datetime.now(UTC),
        target_trade_date=target_date,
        eligible_count=len(candidates),
        context=PredictionContext(
            hmm_state=f"短期状态{int(np.argmax(hmm_probabilities)) + 1}",
            hmm_probabilities={f"状态{index + 1}": float(value) for index, value in enumerate(hmm_probabilities)},
            tft_style="下一交易日收益分位数",
            tft_probabilities={},
        ),
        candidates=candidates,
        prediction_mode="short_term",
        training_window_days=15,
        validation_days=int(result["validation_days"]),
        experimental=True,
        warnings=warnings,
    )
    metadata = VersionMetadata(
        as_of_date=latest_date,
        data_version=data_version,
        feature_version="short-pit-15d-v1",
        model_version=f"art-rank-short-{run_id}",
        run_id=run_id,
        is_stale=False,
    )
    body = {"meta": metadata.model_dump(mode="json"), "data": feed.model_dump(mode="json")}
    prediction_root.mkdir(parents=True, exist_ok=True)
    atomic_json(prediction_root / f"snapshot-{run_id}.json", body)
    atomic_json(prediction_root / "stable.json", body)


def write_component(
    root: Path,
    component: str,
    run_id: str,
    checkpoint: Path,
    metrics: dict[str, float | int | str | bool | list[int]],
) -> None:
    body = {
        "version": run_id,
        "status": "healthy",
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "artifact_sha256": file_sha256(checkpoint),
        "last_trained_at": datetime.now(UTC).isoformat(),
    }
    directory = model_dir(root, component)
    atomic_json(directory / "latest.json", body)
    atomic_json(directory / "champion.json", body)


def model_dir(root: Path, component: str) -> Path:
    path = root / "models" / component
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_symbol(symbol: str) -> str:
    return symbol.replace(".", "_")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_cache_atomic(frame: pl.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(".tmp.parquet")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(target)
    trade_dates = frame["trade_date"] if "trade_date" in frame.columns and not frame.is_empty() else None
    atomic_json(
        target.with_suffix(".manifest.json"),
        {
            "rows": frame.height,
            "min_date": str(trade_dates.min()) if trade_dates is not None else None,
            "max_date": str(trade_dates.max()) if trade_dates is not None else None,
            "sha256": file_sha256(target),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


def atomic_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    ShortTermPipeline(arguments.run_id, get_settings()).execute()


if __name__ == "__main__":
    main()
