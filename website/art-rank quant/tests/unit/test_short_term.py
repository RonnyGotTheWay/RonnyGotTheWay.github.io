import json
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from art_rank_quant.api.schemas.models import TrainingRunSummary
from art_rank_quant.common.config import Settings
from art_rank_quant.data.clients.akshare_sina_history import AkshareSinaHistoryError
from art_rank_quant.models.short_term import (
    experimental_gate,
    latest_trading_dates,
    padded_patch_window,
    rolling_splits,
    trim_to_dates,
)
from art_rank_quant.pipelines.short_term import (
    ShortTermPipeline,
    daily_checks_from_intraday,
    failure_circuit_open,
    file_sha256,
    padded_intraday_window,
    short_market_features,
)


def test_short_window_keeps_exactly_latest_fifteen_trading_dates() -> None:
    start = date(2026, 6, 1)
    dates = [start + timedelta(days=index) for index in range(20)]
    frame = pl.DataFrame({"symbol": ["600000.SH"] * 20, "trade_date": dates, "close": range(20)})

    selected = latest_trading_dates(frame)
    trimmed = trim_to_dates(frame, selected)

    assert selected == dates[-15:]
    assert trimmed.height == 15
    assert trimmed["trade_date"].min() == dates[5]


def test_patch_window_is_120_bars_without_future_values() -> None:
    matrix = np.arange(150 * 5, dtype=np.float32).reshape(150, 5)

    complete = padded_patch_window(matrix, 149)
    early = padded_patch_window(matrix, 19)

    assert complete.shape == (120, 5)
    np.testing.assert_array_equal(complete, matrix[30:150])
    assert early.shape == (120, 5)
    np.testing.assert_array_equal(early[-20:], matrix[:20])
    assert np.count_nonzero(early[:-20]) == 0

    pipeline_early = padded_intraday_window(matrix, 19)
    np.testing.assert_array_equal(pipeline_early, early)


def test_short_hmm_market_frame_retains_all_fifteen_dates() -> None:
    dates = [date(2026, 6, 1) + timedelta(days=index) for index in range(15)]
    frame = pl.DataFrame(
        {
            "trade_date": dates,
            "daily_return": [None, *([0.01] * 14)],
            "amount": [100_000_000.0] * 15,
        }
    )

    market, _ = short_market_features(frame)

    assert market.height == 15
    assert market["trade_date"].to_list() == dates


def test_short_cache_only_requests_missing_dates(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dates = [date(2026, 6, 1) + timedelta(days=index) for index in range(15)]
    target = tmp_path / "cache.parquet"
    pl.DataFrame({"trade_date": dates[:-1]}).write_parquet(target)

    assert ShortTermPipeline._dates_to_fetch(target, dates, 15) == [dates[-1]]

    pl.DataFrame({"trade_date": dates}).write_parquet(target)
    assert ShortTermPipeline._dates_to_fetch(target, dates, 15) == []


def test_three_rolling_folds_have_one_day_embargo_and_no_overlap() -> None:
    dates = [date(2026, 6, 1) + timedelta(days=index) for index in range(14)]

    splits = rolling_splits(dates)

    assert len(splits) == 3
    for train, embargo, test in splits:
        assert len(test) == 2
        assert max(train) < embargo < min(test)
        assert not set(train) & set(test)


def test_experimental_gate_reports_each_minimum_contract() -> None:
    passed = experimental_gate(
        daily_days=15,
        daily_coverage=0.95,
        intraday_coverage=0.85,
        validation_days=6,
        rank_ic_value=0.01,
        shuffled_rank_ic=0.0,
        finite_outputs=True,
    )
    blocked = experimental_gate(
        daily_days=14,
        daily_coverage=0.94,
        intraday_coverage=0.84,
        validation_days=2,
        rank_ic_value=-0.01,
        shuffled_rank_ic=0.2,
        finite_outputs=False,
    )

    assert passed["passed"] is True
    assert blocked["passed"] is False
    assert len(blocked["blocked_reasons"]) == 7


def test_short_window_rejects_insufficient_dates() -> None:
    frame = pl.DataFrame({"trade_date": [date(2026, 7, 1)] * 14})
    with pytest.raises(ValueError, match="15 required"):
        latest_trading_dates(frame)


def test_secondary_source_failure_circuit_uses_configured_threshold() -> None:
    assert failure_circuit_open(19, 20) is False
    assert failure_circuit_open(20, 20) is True


def test_intraday_close_fills_lagging_sina_daily_check() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "trade_date": [date(2026, 7, 20), date(2026, 7, 20)],
            "event_time": [
                "2026-07-20T14:30:00+08:00",
                "2026-07-20T15:00:00+08:00",
            ],
            "close": [10.1, 10.2],
            "volume": [1000.0, 2000.0],
        }
    ).with_columns(pl.col("event_time").str.to_datetime(time_zone="Asia/Shanghai"))

    check = daily_checks_from_intraday(frame, is_st=False)

    assert check.height == 1
    assert check["close_baostock"][0] == 10.2
    assert check["trade_status_baostock"][0] == "trading"
    assert check["check_source"][0] == "akshare_sina_intraday_close"


def test_daily_stage_writes_partial_coverage_before_intraday(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    run_id = "short-term-partial"
    run_path = settings.artifact_root / "training_runs" / f"{run_id}.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        TrainingRunSummary(run_id=run_id, status="running", mode="short_term").model_dump_json(),
        encoding="utf-8",
    )
    pipeline = ShortTermPipeline(run_id, settings)
    dates = [date(2026, 7, 1) + timedelta(days=index) for index in range(15)]
    for symbol in ("600000.SH", "000001.SZ"):
        pl.DataFrame({"symbol": [symbol] * 15, "trade_date": dates}).write_parquet(
            pipeline.daily_root / f"{symbol.replace('.', '_')}.parquet"
        )

    pipeline._write_partial_coverage(dates, ["600000.SH", "000001.SZ"])
    coverage = json.loads((settings.artifact_root / "short_term" / "data_coverage.json").read_text())

    assert coverage["distinct_trade_dates"] == 15
    assert coverage["daily_coverage"] == 1.0
    assert coverage["intraday_coverage"] == 0.0
    assert coverage["source_checks"][0]["status"] == "ready"


def test_source_preflight_fails_before_symbol_backfill(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    run_id = "short-term-preflight"
    run_path = settings.artifact_root / "training_runs" / f"{run_id}.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        TrainingRunSummary(run_id=run_id, status="running", mode="short_term").model_dump_json(),
        encoding="utf-8",
    )
    pipeline = ShortTermPipeline(run_id, settings)

    def fail_preflight(self, start, end):  # type: ignore[no-untyped-def]
        raise AkshareSinaHistoryError("source unavailable")

    monkeypatch.setattr(
        "art_rank_quant.pipelines.short_term.AkshareSinaHistoryClient.preflight",
        fail_preflight,
    )

    with pytest.raises(RuntimeError, match="数据源预检失败"):
        pipeline._source_preflight()

    stored = TrainingRunSummary.model_validate_json(run_path.read_text())
    coverage = json.loads((settings.artifact_root / "short_term" / "data_coverage.json").read_text())
    assert stored.stage == "source_check"
    assert stored.progress == 3
    assert coverage["source_checks"][1]["status"] == "unavailable"


def test_short_cache_copy_rejects_corrupt_manifest_and_keeps_long_cache_read_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    run_id = "short-term-cache-copy"
    run_path = settings.artifact_root / "training_runs" / "short_term" / f"{run_id}.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        TrainingRunSummary(run_id=run_id, status="running", mode="short_term").model_dump_json(),
        encoding="utf-8",
    )
    pipeline = ShortTermPipeline(run_id, settings)
    source = settings.data_root / "staging" / "history-20260701-20260721" / "daily" / "600000_SH.parquet"
    source.parent.mkdir(parents=True)
    dates = [date(2026, 7, 1) + timedelta(days=index) for index in range(15)]
    pl.DataFrame(
        {
            "symbol": ["600000.SH"] * 15,
            "trade_date": dates,
            "close": [10.0] * 15,
            "close_forward_adjusted": [10.0] * 15,
        }
    ).write_parquet(source)
    original = source.read_bytes()
    manifest = source.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"sha256": "corrupt"}), encoding="utf-8")
    target = pipeline.daily_root / "600000_SH.parquet"

    assert pipeline._seed_from_long_cache("600000.SH", "daily", target) is False
    assert not target.exists()
    manifest.write_text(json.dumps({"sha256": file_sha256(source)}), encoding="utf-8")
    assert pipeline._seed_from_long_cache("600000.SH", "daily", target) is True
    assert source.read_bytes() == original
    assert target.exists()
