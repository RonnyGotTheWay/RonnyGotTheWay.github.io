import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from art_rank_quant.api.schemas.common import VersionMetadata
from art_rank_quant.api.schemas.stocks import PredictionFeed
from art_rank_quant.api.services.short_term import ShortTermRuntimeService
from art_rank_quant.common.config import Settings


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def published_body() -> dict[str, object]:
    return {
        "meta": VersionMetadata(
            data_version="short-test",
            feature_version="short-pit-15d-v1",
            model_version="art-rank-short-test",
            run_id="short-term-test",
        ).model_dump(mode="json"),
        "data": PredictionFeed(
            status="published",
            message="published",
            prediction_mode="short_term",
            training_window_days=15,
            validation_days=6,
            experimental=True,
        ).model_dump(mode="json"),
    }


def test_short_runtime_never_reads_long_term_prediction_directory(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    write_json(settings.artifact_root / "published" / "predictions" / "stable.json", published_body())

    feed, _ = ShortTermRuntimeService(settings).latest_prediction()

    assert feed.status == "blocked"
    assert feed.candidates == []


def test_short_runtime_hides_previous_ranking_when_latest_gate_is_blocked(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    write_json(settings.artifact_root / "published" / "short_term_predictions" / "stable.json", published_body())
    write_json(
        settings.artifact_root / "short_term" / "models" / "ensemble" / "latest.json",
        {"status": "blocked", "publish_gate_passed": False, "blocked_reasons": ["Rank IC未通过。"]},
    )

    feed, _ = ShortTermRuntimeService(settings).latest_prediction()

    assert feed.status == "blocked"
    assert feed.candidates == []
    assert "Rank IC未通过" in feed.message


def test_short_runtime_returns_published_feed_only_after_its_own_gate_passes(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    short_root = settings.artifact_root / "short_term"
    write_json(settings.artifact_root / "published" / "short_term_predictions" / "stable.json", published_body())
    write_json(
        short_root / "data_coverage.json",
        {
            "distinct_trade_dates": 15,
            "daily_coverage": 0.95,
            "intraday_coverage": 0.85,
            "daily_check_coverage": 0.95,
        },
    )
    for component in ("hmm", "tft", "patchtst", "ranker"):
        write_json(
            short_root / "models" / component / "champion.json",
            {"status": "healthy", "version": f"short-{component}", "metrics": {}},
        )
    write_json(
        short_root / "models" / "ensemble" / "champion.json",
        {
            "status": "healthy",
            "version": "short-ensemble",
            "publish_gate_passed": True,
            "blocked_reasons": [],
            "weight_sum": 1.0,
            "lagged_rank_ic": 0.02,
            "horizon_consistent": True,
        },
    )

    feed, metadata = ShortTermRuntimeService(settings).latest_prediction()

    assert feed.status == "published"
    assert feed.prediction_mode == "short_term"
    assert metadata.run_id == "short-term-test"


def test_short_runtime_exposes_partial_source_coverage(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    write_json(
        settings.artifact_root / "short_term" / "data_coverage.json",
        {
            "distinct_trade_dates": 15,
            "daily_coverage": 0.9901,
            "intraday_coverage": 0.0,
            "source_checks": [
                {
                    "provider": "eastmoney",
                    "datasets": ["15日日线"],
                    "status": "ready",
                    "message": "日线缓存已完成。",
                },
                {
                    "provider": "akshare_sina",
                    "datasets": ["30分钟K线", "跨源日线校验"],
                    "status": "partial",
                    "message": "正在逐股回填。",
                },
            ],
        },
    )

    monitoring = ShortTermRuntimeService(settings).monitoring()

    assert monitoring.daily_history_days == 15
    assert monitoring.data_coverage_pct == pytest.approx(99.01)
    assert [source.status for source in monitoring.source_checks] == ["ready", "partial"]


def test_short_runtime_derives_daily_coverage_from_cache_without_manifest(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    write_json(
        settings.artifact_root / "published" / "market" / "stable.json",
        {"data": {"quotes": [{"symbol": "600000.SH"}, {"symbol": "000001.SZ"}]}},
    )
    dates = [date(2026, 7, 1) + timedelta(days=index) for index in range(15)]
    daily_root = settings.data_root / "short_term_cache" / "daily"
    daily_root.mkdir(parents=True)
    for symbol in ("600000.SH", "000001.SZ"):
        pl.DataFrame({"symbol": [symbol] * 15, "trade_date": dates}).write_parquet(
            daily_root / f"{symbol.replace('.', '_')}.parquet"
        )

    monitoring = ShortTermRuntimeService(settings).monitoring()

    assert monitoring.daily_history_days == 15
    assert monitoring.data_coverage_pct == 100
    assert monitoring.source_checks[0].status == "ready"
