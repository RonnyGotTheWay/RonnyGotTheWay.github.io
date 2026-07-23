from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import SecretStr

from art_rank_quant.api.services.realtime import RealtimeMarketService
from art_rank_quant.common.config import Settings
from art_rank_quant.data.clients.tushare import MarketDataError, RealtimeQuoteRecord

SHANGHAI = ZoneInfo("Asia/Shanghai")


class _SequencedClient:
    def __init__(self, records: list[RealtimeQuoteRecord]) -> None:
        self.records = records
        self.calls = 0
        self.should_fail = False

    def fetch_quotes(self, symbols: object) -> list[RealtimeQuoteRecord]:
        self.calls += 1
        if self.should_fail:
            raise MarketDataError("测试上游故障")
        return self.records


def _record(symbol: str, close: float, amount: float) -> RealtimeQuoteRecord:
    return RealtimeQuoteRecord(
        symbol=symbol,
        name=f"测试{symbol[:2]}",
        pre_close=10.0,
        open=10.0,
        high=max(close, 10.2),
        low=min(close, 9.9),
        close=close,
        volume=1_000_000,
        amount=amount,
        trade_count=3_000,
        observed_at=datetime(2026, 7, 14, 10, 0, tzinfo=SHANGHAI),
    )


def test_realtime_service_caches_and_falls_back_to_last_stable_snapshot() -> None:
    timer = [100.0]
    client = _SequencedClient([_record("600000.SH", 10.2, 80_000_000)])
    settings = Settings(
        market_data_provider="tushare",
        tushare_token=SecretStr("token"),
        realtime_symbols="600000.SH",
        realtime_cache_seconds=15,
    )
    service = RealtimeMarketService(
        settings,
        client=client,
        clock=lambda: datetime(2026, 7, 14, 10, 1, tzinfo=SHANGHAI),
        monotonic=lambda: timer[0],
    )

    assert service.quotes().status == "live"
    assert service.quotes().status == "live"
    assert client.calls == 1

    timer[0] += 16
    client.should_fail = True
    degraded = service.quotes()
    assert degraded.status == "degraded"
    assert degraded.quotes[0].symbol == "600000.SH"
    assert "上一稳定快照" in degraded.message


def test_candidates_are_ranked_and_remain_research_only() -> None:
    client = _SequencedClient(
        [
            _record("600000.SH", 10.5, 180_000_000),
            _record("000001.SZ", 10.1, 90_000_000),
            _record("600519.SH", 9.8, 240_000_000),
        ]
    )
    settings = Settings(
        market_data_provider="tushare",
        tushare_token=SecretStr("token"),
        realtime_symbols="600000.SH,000001.SZ,600519.SH",
        candidate_min_amount_cny=50_000_000,
    )
    service = RealtimeMarketService(settings, client=client)

    feed = service.candidates(limit=2)
    assert len(feed.candidates) == 2
    assert feed.candidates[0].rank == 1
    assert feed.candidates[0].gate_status == "research_only"
    assert feed.universe_count == 3
    assert feed.rankable_count == 3
    assert "不是收益预测" in feed.methodology


def test_missing_token_never_substitutes_synthetic_quotes() -> None:
    service = RealtimeMarketService(
        Settings(market_data_provider="tushare", tushare_token=None, realtime_symbols="600000.SH")
    )
    feed = service.quotes()
    assert feed.status == "configuration_required"
    assert feed.quotes == []


def test_eastmoney_provider_needs_no_token_after_permission_confirmation() -> None:
    client = _SequencedClient([_record("600000.SH", 10.2, 80_000_000)])
    service = RealtimeMarketService(
        Settings(
            market_data_provider="eastmoney",
            eastmoney_permission_confirmed=True,
            tushare_token=None,
        ),
        client=client,
    )
    feed = service.quotes()
    assert feed.status == "live"
    assert feed.provider == "eastmoney"
    assert feed.quotes[0].name.startswith("测试")
