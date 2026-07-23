from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl

from art_rank_quant.api.schemas.models import TrainingRunSummary
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
from art_rank_quant.data.clients.akshare_sina_history import AkshareSinaHistoryClient, AkshareSinaHistoryError
from art_rank_quant.data.clients.baostock_history import BaoStockHistoryClient, BaoStockHistoryError
from art_rank_quant.data.clients.eastmoney_history import EastmoneyHistoryClient, EastmoneyHistoryError
from art_rank_quant.models.training_orchestrator import ModelGateBlocked, train_and_validate

STAGES = {
    "universe": 2,
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


class BootstrapPipeline:
    def __init__(self, run_id: str, settings: Settings) -> None:
        self.run_id = run_id
        self.settings = settings
        self.run_root = training_run_root(settings, "long_term")
        self.run_path = resolve_training_run_path(settings, run_id, "long_term")
        self.started = datetime.now(UTC)
        self.end = self._latest_trade_date()
        self.start = _years_before(self.end, settings.model_history_years)
        self.version = f"history-{self.start:%Y%m%d}-{self.end:%Y%m%d}"
        self.staging = settings.data_root / "staging" / self.version
        self.daily_root = self.staging / "daily"
        self.intraday_root = self.staging / "intraday"
        self.check_root = self.staging / "daily_check"
        self.universe_root = self.staging / "universe"
        self.intraday_provider = settings.long_term_intraday_provider
        self.failed_intraday_providers: set[str] = set()
        for path in (self.daily_root, self.intraday_root, self.check_root, self.universe_root):
            path.mkdir(parents=True, exist_ok=True)
        self.run = self._read_run()

    def execute(self) -> None:
        with TrainingHeartbeat(self.settings, self.run_path, self.run_id, "long_term"):
            self._execute()

    def _execute(self) -> None:
        self.run.status = "running"
        self.run.started_at = self.run.started_at or self.started
        self._update("universe", STAGES["universe"], "读取东方财富当前全市场证券列表。")
        try:
            symbols = self._symbols()
            self.run.total_symbols = len(symbols)
            self._source_preflight()
            self._backfill_daily(symbols)
            trading_dates = self._daily_trade_dates()
            self._backfill_intraday(symbols, trading_dates)
            coverage = self._quality(trading_dates, symbols)
            if not coverage["passed"]:
                self.run.status = "blocked"
                self.run.stage = "quality"
                self.run.message = "；".join(coverage["blocked_reasons"])
                self.run.finished_at = datetime.now(UTC)
                self._write_run()
                return
            try:
                train_and_validate(
                    daily_glob=str(self.daily_root / "*.parquet"),
                    daily_check_glob=str(self.check_root / "*.parquet"),
                    intraday_glob=str(self.intraday_root / "*.parquet"),
                    artifact_root=self.settings.artifact_root,
                    prediction_root=self.settings.artifact_root / "published" / "predictions",
                    data_version=self.version,
                    run_id=self.run_id,
                    update=self._training_update,
                )
            except ModelGateBlocked as exc:
                self.run = self._read_run()
                self.run.status = "blocked"
                self.run.stage = "validation"
                self.run.message = str(exc)
                self.run.finished_at = datetime.now(UTC)
                self._write_run()
                return
            self.run = self._read_run()
            self.run.status = "success"
            self.run.stage = "complete"
            self.run.progress = 100
            self.run.message = "两年回填、四组件训练和正式发布门禁已执行完成。"
            self.run.finished_at = datetime.now(UTC)
            self._write_run()
        except (TrainingCancelled, TrainingMemoryLimitExceeded) as exc:
            self.run = self._read_run()
            if isinstance(exc, TrainingMemoryLimitExceeded):
                self._write_checkpoint("memory_limit")
            self.run.status = "cancelled" if isinstance(exc, TrainingCancelled) else "failed"
            self.run.stage = self.run.status
            self.run.error = f"{type(exc).__name__}: {exc}"
            self.run.message = str(exc)
            self.run.finished_at = datetime.now(UTC)
            self.run.duration_seconds = (self.run.finished_at - self.started).total_seconds()
            self._write_run()
        except Exception as exc:
            self.run = self._read_run()
            self.run.status = "failed"
            self.run.stage = "failed"
            self.run.error = f"{type(exc).__name__}: {exc}"
            self.run.message = "任务失败，可使用相同数据范围重新提交并从已完成证券继续。"
            self.run.finished_at = datetime.now(UTC)
            self._write_run()
            raise
        finally:
            release_training_lock(self.settings, self.run_id)

    def _backfill_daily(self, symbols: list[str]) -> None:
        self._update("daily", STAGES["daily"], "回填东方财富两年日线。")
        client = EastmoneyHistoryClient(
            permission_confirmed=self.settings.eastmoney_permission_confirmed,
            timeout_seconds=self.settings.eastmoney_request_timeout_seconds,
            request_delay_seconds=self.settings.eastmoney_request_delay_seconds,
        )
        consecutive_failures = 0
        failure_window: deque[bool] = deque(maxlen=100)
        try:
            for index, symbol in enumerate(symbols, 1):
                target = self.daily_root / f"{_safe_symbol(symbol)}.parquet"
                if not target.exists():
                    for attempt in range(5):
                        try:
                            raw = client.fetch_daily(symbol, self.start, self.end, "raw")
                            time.sleep(max(0.5, self.settings.eastmoney_request_delay_seconds))
                            adjusted = client.fetch_daily(symbol, self.start, self.end, "forward")
                            if raw.is_empty():
                                raise EastmoneyHistoryError("empty Eastmoney daily history")
                            adjusted_prices = adjusted.select(
                                "trade_date",
                                pl.col("open").alias("open_forward_adjusted"),
                                pl.col("high").alias("high_forward_adjusted"),
                                pl.col("low").alias("low_forward_adjusted"),
                                pl.col("close").alias("close_forward_adjusted"),
                            )
                            _write_parquet_atomic(raw.join(adjusted_prices, on="trade_date", how="left"), target)
                            consecutive_failures = 0
                            failure_window.append(False)
                            self.run.last_success_at = datetime.now(UTC)
                            break
                        except EastmoneyHistoryError as exc:
                            self.run.retries += 1
                            if attempt == 4:
                                self._quarantine("daily", symbol, str(exc))
                                consecutive_failures += 1
                                failure_window.append(True)
                                self.run.failed_items += 1
                            else:
                                jitter = 0.8 + time.monotonic() % 0.4
                                time.sleep((2 ** (attempt + 1)) * jitter)
                    time.sleep(self.settings.eastmoney_request_delay_seconds)
                self.run.current_symbol = symbol
                self.run.completed_symbols = index
                self.run.progress = 5 + int(index / max(1, len(symbols)) * 24)
                self._update_eta(index, len(symbols))
                if index % 10 == 0 or index == len(symbols):
                    check_run_control(self.settings, self.run_path, self.run)
                    self._write_run()
                if consecutive_failures >= 20 or (len(failure_window) == 100 and sum(failure_window) >= 20):
                    raise RuntimeError("东方财富日线连续失败20只证券，已熔断；可稍后从缓存续跑。")
        finally:
            client.close()

    def _source_preflight(self) -> None:
        providers = [self.settings.long_term_intraday_provider, *self.settings.history_provider_fallback_list]
        failures: list[str] = []
        for provider in dict.fromkeys(providers):
            if provider == "baostock" and not self.settings.baostock_enabled:
                failures.append("baostock: 已禁用")
                continue
            try:
                self._preflight_intraday_provider(provider)
                self.intraday_provider = cast(Literal["eastmoney", "akshare_sina", "baostock"], provider)
                self.run.provider = provider
                self._write_run()
                return
            except (EastmoneyHistoryError, AkshareSinaHistoryError, BaoStockHistoryError) as exc:
                self.failed_intraday_providers.add(provider)
                failures.append(f"{provider}: {exc}")
        raise RuntimeError(f"长期30分钟数据源预检失败：{'；'.join(failures)}")

    def _preflight_intraday_provider(self, provider: str) -> None:
        if provider == "eastmoney":
            client = EastmoneyHistoryClient(
                permission_confirmed=self.settings.eastmoney_permission_confirmed,
                timeout_seconds=self.settings.eastmoney_request_timeout_seconds,
                request_delay_seconds=self.settings.eastmoney_request_delay_seconds,
            )
            try:
                client.preflight(max(self.start, self.end - timedelta(days=40)), self.end)
            finally:
                client.close()
            return
        if provider == "akshare_sina":
            AkshareSinaHistoryClient(
                delay_seconds=self.settings.short_term_secondary_delay_seconds,
                timeout_seconds=self.settings.eastmoney_request_timeout_seconds,
            ).preflight(max(self.start, self.end - timedelta(days=40)), self.end)
            return
        if provider == "baostock":
            with BaoStockHistoryClient():
                return
        raise RuntimeError(f"不支持的长期30分钟数据源: {provider}")

    def _backfill_intraday(self, symbols: list[str], trading_dates: list[date]) -> None:
        provider = self.intraday_provider
        self._update("intraday", STAGES["intraday"], f"回填 {provider} 两年30分钟K线和日线校验。")
        if provider == "baostock":
            self._backfill_with_baostock(symbols, trading_dates)
            return
        supported = [symbol for symbol in symbols if symbol.endswith((".SH", ".SZ"))]
        consecutive_failures = 0
        failure_window: deque[bool] = deque(maxlen=100)
        for index, symbol in enumerate(supported, 1):
            intraday_target = self.intraday_root / f"{_safe_symbol(symbol)}.parquet"
            check_target = self.check_root / f"{_safe_symbol(symbol)}.parquet"
            try:
                if provider == "eastmoney":
                    eastmoney_client = EastmoneyHistoryClient(
                        permission_confirmed=self.settings.eastmoney_permission_confirmed,
                        timeout_seconds=self.settings.eastmoney_request_timeout_seconds,
                        request_delay_seconds=self.settings.eastmoney_request_delay_seconds,
                    )
                    try:
                        if not intraday_target.exists():
                            _write_parquet_atomic(
                                eastmoney_client.fetch_intraday(symbol, self.start, self.end), intraday_target
                            )
                        if not check_target.exists():
                            _write_parquet_atomic(
                                eastmoney_client.fetch_daily_check(symbol, self.start, self.end), check_target
                            )
                    finally:
                        eastmoney_client.close()
                elif provider == "akshare_sina":
                    sina_client = AkshareSinaHistoryClient(
                        delay_seconds=self.settings.short_term_secondary_delay_seconds,
                        timeout_seconds=self.settings.eastmoney_request_timeout_seconds,
                    )
                    if not intraday_target.exists():
                        _write_parquet_atomic(
                            sina_client.fetch_intraday(symbol, self.start, self.end), intraday_target
                        )
                    if not check_target.exists():
                        _write_parquet_atomic(
                            sina_client.fetch_daily_check(symbol, self.start, self.end), check_target
                        )
                consecutive_failures = 0
                failure_window.append(False)
                self.run.last_success_at = datetime.now(UTC)
            except (EastmoneyHistoryError, AkshareSinaHistoryError, OSError) as exc:
                consecutive_failures += 1
                failure_window.append(True)
                self.run.retries += 1
                self.run.failed_items += 1
                self._quarantine("intraday", symbol, str(exc))
            self.run.current_symbol = symbol
            self.run.completed_symbols = index
            self.run.progress = 30 + int(index / max(1, len(supported)) * 24)
            self._update_eta(index, len(supported))
            if index % 10 == 0 or index == len(supported):
                check_run_control(self.settings, self.run_path, self.run)
                self._write_run()
            if consecutive_failures >= 20 or (len(failure_window) == 100 and sum(failure_window) >= 20):
                self.failed_intraday_providers.add(provider)
                if self._activate_long_fallback():
                    self._backfill_intraday(symbols, trading_dates)
                    return
                raise RuntimeError(f"{provider}失败率过高且备用源不可用；30分钟数据不足，拒绝进入拟合。")

    def _activate_long_fallback(self) -> bool:
        providers = [self.settings.long_term_intraday_provider, *self.settings.history_provider_fallback_list]
        for provider in dict.fromkeys(providers):
            if provider in self.failed_intraday_providers:
                continue
            if provider == "baostock" and not self.settings.baostock_enabled:
                continue
            try:
                self._preflight_intraday_provider(provider)
            except (EastmoneyHistoryError, AkshareSinaHistoryError, BaoStockHistoryError):
                self.failed_intraday_providers.add(provider)
                continue
            self.intraday_provider = cast(Literal["eastmoney", "akshare_sina", "baostock"], provider)
            self.run.provider = provider
            self._write_run()
            return True
        return False

    def _backfill_with_baostock(self, symbols: list[str], trading_dates: list[date]) -> None:
        with BaoStockHistoryClient() as client:
            for trade_date in trading_dates:
                target = self.universe_root / f"{trade_date.isoformat()}.parquet"
                if not target.exists():
                    _write_parquet_atomic(client.fetch_universe(trade_date), target)
            supported = [symbol for symbol in symbols if symbol.endswith((".SH", ".SZ"))]
            for index, symbol in enumerate(supported, 1):
                intraday_target = self.intraday_root / f"{_safe_symbol(symbol)}.parquet"
                check_target = self.check_root / f"{_safe_symbol(symbol)}.parquet"
                try:
                    if not intraday_target.exists():
                        _write_parquet_atomic(client.fetch_intraday(symbol, self.start, self.end), intraday_target)
                    if not check_target.exists():
                        _write_parquet_atomic(client.fetch_daily_check(symbol, self.start, self.end), check_target)
                except BaoStockHistoryError as exc:
                    self.run.retries += 1
                    self._quarantine("intraday", symbol, str(exc))
                self.run.current_symbol = symbol
                self.run.completed_symbols = index
                self.run.progress = 30 + int(index / max(1, len(supported)) * 24)
                if index % 10 == 0 or index == len(supported):
                    check_run_control(self.settings, self.run_path, self.run)
                    self._write_run()

    def _quality(self, trading_dates: list[date], symbols: list[str]) -> dict[str, Any]:
        self._update("quality", STAGES["quality"], "执行交易日、K线、覆盖率和跨源一致性检查。")
        daily = pl.scan_parquet(str(self.daily_root / "*.parquet"))
        daily_stats = daily.select(
            pl.col("trade_date").n_unique().alias("days"),
            pl.len().alias("rows"),
        ).collect()
        intraday = pl.scan_parquet(str(self.intraday_root / "*.parquet"))
        intraday_stats = intraday.select(
            pl.col("trade_date").n_unique().alias("days"),
            pl.len().alias("rows"),
        ).collect()
        check = pl.scan_parquet(str(self.check_root / "*.parquet"))
        comparison = (
            daily.select("symbol", "trade_date", "close")
            .join(check.select("symbol", "trade_date", "close_baostock"), on=["symbol", "trade_date"], how="inner")
            .with_columns(
                ((pl.col("close") / pl.col("close_baostock") - 1).abs()).alias("close_difference_pct")
            )
        )
        mismatches = comparison.filter(pl.col("close_difference_pct") > 0.005).collect()
        if not mismatches.is_empty():
            mismatches.write_parquet(self.staging / "quarantine_price_mismatch.parquet", compression="zstd")
        expected = (
            daily.filter((pl.col("volume") > 0) & pl.col("symbol").str.ends_with_any([".SH", ".SZ"]))
            .select((pl.len() * 8).alias("expected"))
            .collect()
            .item()
        )
        actual = int(intraday_stats["rows"][0])
        coverage = min(1.0, actual / max(1, int(expected)))
        bar_violations = (
            intraday.group_by(["symbol", "trade_date"])
            .len()
            .filter(pl.col("len") > 8)
            .select(pl.len())
            .collect()
            .item()
        )
        daily_days = int(daily_stats["days"][0])
        intraday_days = int(intraday_stats["days"][0])
        reasons = []
        if daily_days < self.settings.model_min_daily_days:
            reasons.append(f"日线仅覆盖 {daily_days} 个交易日，低于 {self.settings.model_min_daily_days} 日。")
        if coverage < self.settings.model_min_intraday_coverage:
            reasons.append(
                f"30分钟加权覆盖率 {coverage:.2%}，低于 {self.settings.model_min_intraday_coverage:.0%}。"
            )
        if bar_violations:
            reasons.append(f"发现 {bar_violations} 个证券交易日超过8根30分钟K线。")
        manifest = {
            "version": self.version,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "daily_rows": int(daily_stats["rows"][0]),
            "intraday_rows": actual,
            "distinct_trade_dates": daily_days,
            "intraday_trade_dates": intraday_days,
            "intraday_coverage": coverage,
            "universe_symbols": len(symbols),
            "bar_count_violations": int(bar_violations),
            "cross_source_price_mismatches": mismatches.height,
            "passed": not reasons,
            "blocked_reasons": reasons,
            "created_at": datetime.now(UTC).isoformat(),
        }
        target = self.settings.artifact_root / "data_coverage.json"
        _atomic_json(target, manifest)
        self.run.daily_rows = manifest["daily_rows"]
        self.run.intraday_rows = manifest["intraday_rows"]
        self._write_run()
        return manifest

    def _symbols(self) -> list[str]:
        path = self.settings.artifact_root / "published" / "market" / "stable.json"
        if not path.exists():
            raise RuntimeError("请先在实时候选页完成一次东方财富全市场刷新")
        body = json.loads(path.read_text(encoding="utf-8"))
        symbols = sorted({str(row["symbol"]) for row in body["data"]["quotes"]})
        if len(symbols) < 5000:
            raise RuntimeError(f"当前全市场列表仅 {len(symbols)} 只，拒绝启动历史回填")
        return symbols

    def _daily_trade_dates(self) -> list[date]:
        return (
            pl.scan_parquet(str(self.daily_root / "*.parquet"))
            .select("trade_date")
            .unique()
            .sort("trade_date")
            .collect()["trade_date"]
            .to_list()
        )

    def _latest_trade_date(self) -> date:
        path = self.settings.artifact_root / "published" / "market" / "stable.json"
        if path.exists():
            body = json.loads(path.read_text(encoding="utf-8"))
            value = body.get("data", {}).get("trade_date")
            if value:
                return date.fromisoformat(value)
        return date.today()

    def _training_update(self, stage: str, progress: int, message: str) -> None:
        self.run = self._read_run()
        self._update(stage, progress, message)

    def _update(self, stage: str, progress: int, message: str) -> None:
        self.run.stage = stage
        self.run.progress = progress
        self.run.message = message
        check_run_control(self.settings, self.run_path, self.run)
        self._write_run()

    def _quarantine(self, dataset: str, symbol: str, reason: str) -> None:
        root = self.settings.data_root / "quarantine" / self.version
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{dataset}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"symbol": symbol, "reason": reason}, ensure_ascii=False) + "\n")

    def _read_run(self) -> TrainingRunSummary:
        return TrainingRunSummary.model_validate_json(self.run_path.read_text(encoding="utf-8"))

    def _write_run(self) -> None:
        _atomic_json(self.run_path, self.run.model_dump(mode="json"))

    def _write_checkpoint(self, reason: str) -> None:
        _atomic_json(
            self.run_path.with_suffix(".checkpoint"),
            {
                "run_id": self.run_id,
                "mode": "long_term",
                "reason": reason,
                "stage": self.run.stage,
                "progress": self.run.progress,
                "data_version": self.version,
                "daily_cache": str(self.daily_root),
                "intraday_cache": str(self.intraday_root),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

    def _update_eta(self, completed: int, total: int) -> None:
        elapsed = (datetime.now(UTC) - (self.run.started_at or self.started)).total_seconds()
        self.run.elapsed_seconds = elapsed
        self.run.eta_seconds = elapsed / completed * max(0, total - completed) if completed else None


def _years_before(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(".", "_")


def _atomic_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_parquet_atomic(frame: pl.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(".tmp.parquet")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(target)
    trade_dates = frame["trade_date"] if "trade_date" in frame.columns and not frame.is_empty() else None
    _atomic_json(
        target.with_suffix(".manifest.json"),
        {
            "rows": frame.height,
            "min_date": str(trade_dates.min()) if trade_dates is not None else None,
            "max_date": str(trade_dates.max()) if trade_dates is not None else None,
            "sha256": fingerprint(target),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    BootstrapPipeline(arguments.run_id, get_settings()).execute()


if __name__ == "__main__":
    main()
