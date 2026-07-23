from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import polars as pl
import pytest

from art_rank_quant.data.clients.eastmoney_daily import (
    EastmoneyCrawlerError,
    EastmoneyDailyCrawler,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _row(code: str, market: int, name: str, close: float) -> dict[str, object]:
    return {
        "f12": code,
        "f13": market,
        "f14": name,
        "f2": close,
        "f3": 2.5,
        "f4": 0.25,
        "f5": 1_000_000,
        "f6": 12_000_000,
        "f7": 4.2,
        "f8": 1.8,
        "f9": 15.0,
        "f10": 1.1,
        "f15": close + 0.2,
        "f16": close - 0.3,
        "f17": close - 0.1,
        "f18": close - 0.25,
        "f20": 10_000_000_000,
        "f21": 8_000_000_000,
        "f23": 1.7,
    }


def _client() -> tuple[httpx.Client, list[int]]:
    pages: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["pn"])
        pages.append(page)
        assert request.url.params["fs"] == "test:a-shares"
        rows = (
            [_row("600000", 1, "浦发银行", 10.3), _row("000001", 0, "平安银行", 12.4)]
            if page == 1
            else [_row("920001", 0, "北证样本", 8.2)]
        )
        return httpx.Response(200, json={"rc": 0, "data": {"total": 3, "diff": rows}})

    return httpx.Client(transport=httpx.MockTransport(handler)), pages


def test_eastmoney_crawler_paginates_and_maps_all_a_share_exchanges() -> None:
    client, pages = _client()
    crawler = EastmoneyDailyCrawler(
        permission_confirmed=True,
        page_size=2,
        market_partitions=(("测试分区", "test:a-shares"),),
        http_client=client,
        clock=lambda: datetime(2026, 7, 14, 16, 5, tzinfo=SHANGHAI),
        sleeper=lambda _: None,
    )

    records = crawler.fetch_daily(date(2026, 7, 14))
    assert pages == [1, 2]
    assert [record.symbol for record in records] == ["000001.SZ", "600000.SH", "920001.BJ"]
    assert [record.name for record in records] == ["平安银行", "浦发银行", "北证样本"]
    assert all(record.source == "eastmoney_quote_center" for record in records)


def test_eastmoney_crawler_writes_immutable_raw_and_training_manifests(tmp_path: Path) -> None:
    client, _ = _client()
    crawler = EastmoneyDailyCrawler(
        permission_confirmed=True,
        page_size=2,
        market_partitions=(("测试分区", "test:a-shares"),),
        http_client=client,
        clock=lambda: datetime(2026, 7, 14, 16, 5, tzinfo=SHANGHAI),
        sleeper=lambda _: None,
    )

    result = crawler.crawl_to_storage(tmp_path, date(2026, 7, 14))
    assert result.raw_manifest.rows == 3
    assert result.training_manifest.rows == 3
    assert result.raw_path.exists()
    assert result.training_path.exists()
    assert pl.read_parquet(result.training_path).get_column("name").null_count() == 0
    manifest = json.loads(result.raw_path.with_name("manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == result.raw_manifest.sha256


def test_eastmoney_crawler_requires_permission_confirmation() -> None:
    with pytest.raises(EastmoneyCrawlerError, match="PERMISSION_CONFIRMED"):
        EastmoneyDailyCrawler(permission_confirmed=False)


def test_eastmoney_crawler_merges_both_code_sort_directions_past_window() -> None:
    calls: list[tuple[int, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["pn"])
        order = request.url.params["po"]
        calls.append((page, order))
        if order == "0":
            start = (page - 1) * 100
            codes = range(start, min(start + 100, 600))
        else:
            codes = range(600, 500, -1)
        rows = [_row(f"{code:06d}", 0, f"股票{code:06d}", 10.0) for code in codes]
        return httpx.Response(200, json={"rc": 0, "data": {"total": 601, "diff": rows}})

    crawler = EastmoneyDailyCrawler(
        permission_confirmed=True,
        page_size=100,
        market_partitions=(("大分区", "test:large"),),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: datetime(2026, 7, 14, 16, 5, tzinfo=SHANGHAI),
        sleeper=lambda _: None,
    )

    records = crawler.fetch_daily(date(2026, 7, 14))
    assert len(records) == 601
    assert calls == [
        (1, "0"),
        (2, "0"),
        (3, "0"),
        (4, "0"),
        (5, "0"),
        (6, "0"),
        (1, "1"),
    ]
