import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from art_rank_quant.api.schemas.weekly import WeeklyDataSource, WeeklyRankingItem
from art_rank_quant.api.services.weekly_report import WeeklyReportService
from art_rank_quant.common.config import Settings


def _service(tmp_path: Path) -> WeeklyReportService:
    return WeeklyReportService(Settings(data_root=tmp_path / "data", artifact_root=tmp_path / "artifacts"))


def _trading_dates(end: date, count: int) -> list[date]:
    values = []
    current = end
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current)
        current -= timedelta(days=1)
    return sorted(values)


def _write_history(tmp_path: Path) -> None:
    root = tmp_path / "data" / "staging" / "history-test" / "daily"
    root.mkdir(parents=True)
    dates = _trading_dates(date(2026, 7, 17), 60)
    for index, symbol in enumerate(("600000.SH", "000001.SZ")):
        rows = []
        for offset, trade_date in enumerate(dates):
            close = 10 + index + offset
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "close": float(close),
                    "close_forward_adjusted": float(close),
                    "pct_change": float(offset - 2),
                    "amount": float(1000 * (index + 1)),
                }
            )
        pl.DataFrame(rows).write_parquet(root / f"{symbol.replace('.', '_')}.parquet")


def test_two_month_report_uses_latest_sixty_sessions_and_blocks_missing_model(tmp_path: Path) -> None:
    _write_history(tmp_path)
    service = _service(tmp_path)

    report, metadata, target = service.generate(date(2026, 7, 17))

    assert report.trading_dates == _trading_dates(date(2026, 7, 17), 60)
    assert report.market is not None
    assert report.market.universe_count == 2
    assert report.market.total_amount == 180_000
    assert sum(bucket.count for bucket in report.market.return_distribution) == 2
    assert report.ranking is not None
    assert report.ranking.status == "blocked"
    assert report.ranking.candidates == []
    assert report.status == "ranking_blocked"
    assert metadata.as_of_date == date(2026, 7, 17)
    assert target.exists()
    assert (service.report_root / "stable.json").exists()


def test_failed_generation_keeps_previous_stable_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_history(tmp_path)
    service = _service(tmp_path)
    service.generate(date(2026, 7, 17))
    stable = service.report_root / "stable.json"
    original = json.loads(stable.read_text(encoding="utf-8"))["meta"]["run_id"]
    monkeypatch.setattr(service, "_market_report", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        service.generate(date(2026, 7, 17))

    assert json.loads(stable.read_text(encoding="utf-8"))["meta"]["run_id"] == original


def test_dual_source_gate_keeps_previous_stable_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_history(tmp_path)
    service = _service(tmp_path)
    service.generate(date(2026, 7, 17))
    stable = service.report_root / "stable.json"
    original = json.loads(stable.read_text(encoding="utf-8"))["meta"]["run_id"]
    monkeypatch.setattr(service, "_validate_sources", lambda *_: [
        WeeklyDataSource(provider="akshare", role="primary", status="unavailable", message="offline"),
        WeeklyDataSource(provider="baostock", role="cross_check", status="verified", rows=60, message="ok"),
    ])

    with pytest.raises(RuntimeError, match="akshare 不可用"):
        service.generate(date(2026, 7, 17), validate_sources=True)

    assert json.loads(stable.read_text(encoding="utf-8"))["meta"]["run_id"] == original


def test_newer_local_data_marks_published_report_stale(tmp_path: Path) -> None:
    _write_history(tmp_path)
    service = _service(tmp_path)
    service.generate(date(2026, 7, 17))
    manifest = tmp_path / "data" / "raw" / "eastmoney_stock_daily" / "data_version=newer" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"max_date": "2026-07-20"}), encoding="utf-8")

    report, metadata = service.latest()

    assert report.status == "stale"
    assert metadata.is_stale is True


def test_incomplete_intraday_partition_does_not_mark_report_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_history(tmp_path)
    service = _service(tmp_path)
    service.generate(date(2026, 7, 17))
    manifest = tmp_path / "data" / "raw" / "eastmoney_stock_daily" / "data_version=intraday" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"max_date": "2026-07-20"}), encoding="utf-8")
    monkeypatch.setattr(service, "_last_complete_date", lambda: date(2026, 7, 17))

    report, metadata = service.latest()

    assert report.status == "ranking_blocked"
    assert metadata.is_stale is False


def test_weekly_experimental_model_publishes_exactly_ten_candidates(tmp_path: Path) -> None:
    root = tmp_path / "data" / "staging" / "history-large" / "daily"
    root.mkdir(parents=True)
    rows = []
    dates = _trading_dates(date(2026, 7, 17), 60)
    for stock_index in range(600):
        symbol = f"{stock_index:06d}.SZ"
        base = 10 + stock_index / 100
        for day_index, trade_date in enumerate(dates):
            daily_return = ((stock_index % 31) - 15) / 20 + day_index * ((stock_index % 7) - 3) / 30
            close = base * (1 + daily_return / 100) ** (day_index + 1)
            rows.append(
                {
                    "symbol": symbol, "trade_date": trade_date, "close": close,
                    "close_forward_adjusted": close, "pct_change": daily_return,
                    "amount": 100_000_000 + stock_index * 10_000,
                }
            )
    pl.DataFrame(rows).write_parquet(root / "weekly.parquet")

    report, _, _ = _service(tmp_path).generate(date(2026, 7, 17))

    assert report.status == "published"
    assert report.ranking is not None
    assert report.ranking.experimental is True
    assert report.ranking.training_rows == 34_800
    assert len(report.ranking.candidates) == 10
    assert report.ranking.target_trade_date == date(2026, 7, 20)
    assert report.ranking.art_weight == 0.65
    assert report.ranking.strategy_weight == 0.35
    assert all(item.total_score is not None for item in report.ranking.candidates)
    assert all(item.strategy_score is not None for item in report.ranking.candidates)
    assert report.ranking.sector_rankings
    assert all(item.price_guidance.calibration_sample_size >= 1_000 for item in report.ranking.candidates)
    top = report.ranking.candidates[0]
    sector_copy = next(
        item
        for sector in report.ranking.sector_rankings
        for item in sector.candidates
        if item.symbol == top.symbol
    )
    assert sector_copy.price_guidance == top.price_guidance
    assert [item.total_score for item in report.ranking.candidates] == sorted(
        [item.total_score for item in report.ranking.candidates if item.total_score is not None], reverse=True
    )


def _guidance_rows(
    symbol: str, dates: list[date], daily_return: float, *, base: float = 10.0
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    close = base
    for trade_date in dates:
        pre_close = close
        close = pre_close * (1 + daily_return)
        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "close": close,
                "adjusted_close": close,
                "pre_close": pre_close,
                "high": max(pre_close, close) * 1.002,
                "low": min(pre_close, close) * 0.998,
            }
        )
    return rows


def _guidance_item(symbol: str, industry: str, close: float) -> WeeklyRankingItem:
    return WeeklyRankingItem(
        symbol=symbol,
        name="测试股票",
        industry=industry,
        exchange="SH",
        rank=1,
        close=close,
        art_score=90,
        score_percentile=90,
        stability_score=80,
    )


def test_low_median_return_outputs_industry_reference_without_passing_gate(tmp_path: Path) -> None:
    dates = _trading_dates(date(2026, 7, 17), 60)
    rows: list[dict[str, object]] = []
    identity: dict[str, tuple[str, str]] = {}
    for index in range(60):
        symbol = f"60{index:04d}.SH"
        daily_return = 0.004 if index < 25 else -0.003
        rows.extend(_guidance_rows(symbol, dates, daily_return, base=10 + index / 10))
        identity[symbol] = (f"股票{index}", "测试行业")
    frame = pl.DataFrame(rows)
    symbol = "600059.SH"
    item = _guidance_item(symbol, "测试行业", float(frame.filter(pl.col("symbol") == symbol)["close"][-1]))

    result = _service(tmp_path)._attach_price_guidance([item], frame, identity, dates[-1])[0]
    guide = result.price_guidance

    assert guide.status == "watch"
    assert guide.is_reference_value is True
    assert guide.reference_method == "industry_positive_quantiles"
    assert guide.sell_price_low is not None
    assert guide.sell_price_high is not None
    assert guide.expected_gain_low_pct == round(
        (guide.sell_price_low / ((guide.buy_price_low + guide.buy_price_high) / 2) - 1) * 100, 2
    )
    assert "五日收益中位数未覆盖估算交易成本。" in guide.blocked_reasons


def test_reference_uses_global_positive_samples_when_industry_is_insufficient(tmp_path: Path) -> None:
    dates = _trading_dates(date(2026, 7, 17), 60)
    rows = _guidance_rows("600000.SH", dates, -0.003)
    identity = {"600000.SH": ("目标股票", "稀疏行业")}
    for index in range(40):
        symbol = f"00{index:04d}.SZ"
        rows.extend(_guidance_rows(symbol, dates, 0.004, base=12 + index / 10))
        identity[symbol] = (f"正收益{index}", "其他行业")
    frame = pl.DataFrame(rows)
    target_close = float(frame.filter(pl.col("symbol") == "600000.SH")["close"][-1])

    result = _service(tmp_path)._attach_price_guidance(
        [_guidance_item("600000.SH", "稀疏行业", target_close)], frame, identity, dates[-1]
    )[0]

    assert result.price_guidance.is_reference_value is True
    assert result.price_guidance.reference_method == "global_positive_quantiles"
    assert result.price_guidance.sell_price_low is not None


def test_insufficient_global_samples_output_atr_reference_and_remain_blocked(tmp_path: Path) -> None:
    dates = _trading_dates(date(2026, 7, 17), 20)
    rows = _guidance_rows("600000.SH", dates, -0.002)
    frame = pl.DataFrame(rows)
    close = float(frame["close"][-1])

    result = _service(tmp_path)._attach_price_guidance(
        [_guidance_item("600000.SH", "稀疏行业", close)],
        frame,
        {"600000.SH": ("目标股票", "稀疏行业")},
        dates[-1],
    )[0]
    guide = result.price_guidance

    assert guide.status == "blocked"
    assert guide.is_reference_value is True
    assert guide.reference_method == "atr_volatility"
    assert guide.sell_price_low is not None
    assert guide.expected_gain_low_pct is not None
    assert guide.positive_return_probability is None
    assert guide.target_hit_probability is None
    assert guide.risk_reward_ratio is None


def _strategy_frame(
    closes: list[float], returns: list[float], *, final_high: float | None = None
) -> pl.DataFrame:
    rows = []
    for index, (close, daily_return) in enumerate(zip(closes, returns, strict=True)):
        pre_close = closes[index - 1] if index else close / (1 + daily_return / 100)
        rows.append(
            {
                "symbol": "600000.SH", "trade_date": date(2026, 7, 13) + timedelta(days=index),
                "open": pre_close, "high": final_high if index == 4 and final_high is not None else close,
                "low": min(pre_close, close), "close": close, "pre_close": pre_close,
                "limit_up": None, "limit_down": None, "adjusted_close": close,
                "pct_change": daily_return, "amount": 300_000_000.0, "trade_status": "trading",
            }
        )
    return pl.DataFrame(rows)


def test_experience_gate_rejects_a_stock_that_touched_limit_up(tmp_path: Path) -> None:
    service = _service(tmp_path)
    stock = _strategy_frame([10.0, 10.2, 10.4, 10.5, 10.7], [0.0, 2.0, 1.96, 0.96, 1.9], final_high=11.55)

    _, _, hard_flags = service._experience_score(stock, "600000.SH", "浦发银行")

    assert "最新交易日触及涨停" in hard_flags


def test_experience_score_penalizes_chasing_repeated_large_gains(tmp_path: Path) -> None:
    service = _service(tmp_path)
    stock = _strategy_frame([10.0, 10.5, 11.0, 11.5, 12.0], [0.0, 5.0, 4.76, 4.55, 4.35])

    score, adjustments, hard_flags = service._experience_score(stock, "600000.SH", "浦发银行")

    assert hard_flags == []
    assert score < 70
    assert "五日涨幅≥20%" in adjustments
    assert "连续3日以上大涨" in adjustments


def test_experience_gate_rejects_an_obvious_five_day_downtrend(tmp_path: Path) -> None:
    service = _service(tmp_path)
    stock = _strategy_frame([10.0, 9.7, 9.4, 9.1, 8.8], [0.0, -3.0, -3.09, -3.19, -3.3])

    _, _, hard_flags = service._experience_score(stock, "600000.SH", "浦发银行")

    assert "明显五日下降趋势" in hard_flags


def test_total_score_combines_art_and_experience_weights(tmp_path: Path) -> None:
    service = _service(tmp_path)

    assert service._total_score(90, 70) == pytest.approx(83.0)
