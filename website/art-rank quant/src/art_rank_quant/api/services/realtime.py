from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

from art_rank_quant.api.schemas.common import VersionMetadata
from art_rank_quant.api.schemas.market import RealtimeQuote, RealtimeQuoteFeed
from art_rank_quant.api.schemas.stocks import CandidateFeed, ResearchCandidate
from art_rank_quant.common.config import Settings, get_settings
from art_rank_quant.data.clients.eastmoney_daily import EastmoneyDailyCrawler
from art_rank_quant.data.clients.tushare import (
    MarketDataError,
    RealtimeQuoteRecord,
    TushareRealtimeClient,
    normalize_symbols,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
METHODOLOGY = (
    "观察分 = 横截面实时涨跌幅 45% + 日内价格位置 30% + 成交额 25%；"
    "它不是收益预测，不使用个人风险信息，也不会触发交易。"
)
FeedStatus = Literal["live", "degraded", "configuration_required", "upstream_error"]
MarketSession = Literal["trading", "lunch_break", "closed"]


class QuoteClient(Protocol):
    def fetch_quotes(self, symbols: Sequence[str]) -> list[RealtimeQuoteRecord]: ...


@dataclass(slots=True)
class _CachedQuotes:
    records: list[RealtimeQuoteRecord]
    fetched_at: datetime
    cached_at: float


class RealtimeMarketService:
    def __init__(
        self,
        settings: Settings,
        client: QuoteClient | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._clock = clock or (lambda: datetime.now(SHANGHAI))
        self._monotonic = monotonic or time.monotonic
        self._cache: dict[tuple[str, ...], _CachedQuotes] = {}

    def quotes(self, symbols: Sequence[str] | None = None, *, force: bool = False) -> RealtimeQuoteFeed:
        now = self._localized_now()
        provider = self._settings.market_data_provider.lower()
        if provider == "tushare":
            selected = tuple(symbols or self._settings.realtime_symbol_list)
            normalized = normalize_symbols(selected) if selected else None
            if self._settings.tushare_token is None or not self._settings.tushare_token.get_secret_value():
                return self._empty_feed(now, "configuration_required", "请在本机 .env 配置 TUSHARE_TOKEN。")
            if not selected:
                return self._empty_feed(now, "configuration_required", "请在本机 .env 配置 REALTIME_SYMBOLS。")
            assert normalized is not None
            request_symbols = normalized
            cache_key = normalized
        elif provider == "eastmoney":
            if not self._settings.eastmoney_permission_confirmed:
                return self._empty_feed(
                    now,
                    "configuration_required",
                    "请先确认并记录东方财富个人研究数据许可范围。",
                )
            request_symbols = normalize_symbols(symbols) if symbols else ()
            cache_key = request_symbols or ("__eastmoney_all_a_shares__",)
        else:
            return self._empty_feed(
                now,
                "configuration_required",
                f"暂不支持行情提供方 {provider!r}。",
            )
        cached = self._cache.get(cache_key)
        if cached and not force and self._monotonic() - cached.cached_at < self._settings.realtime_cache_seconds:
            return self._feed_from_cache(cached, now, "live", "实时行情已更新（服务端短缓存）。")

        try:
            records = self._quote_client().fetch_quotes(request_symbols)
            if not records:
                raise MarketDataError("行情接口未返回所选股票，请检查代码与接口权限")
            snapshot = _CachedQuotes(records=records, fetched_at=now, cached_at=self._monotonic())
            self._cache[cache_key] = snapshot
            provider_name = "东方财富行情中心" if provider == "eastmoney" else "Tushare"
            return self._feed_from_cache(snapshot, now, "live", f"{provider_name}行情已更新。")
        except MarketDataError as exc:
            if cached is not None:
                return self._feed_from_cache(
                    cached,
                    now,
                    "degraded",
                    f"上游更新失败，展示上一稳定快照：{exc}",
                )
            return self._empty_feed(now, "upstream_error", str(exc))

    def candidates(self, symbols: Sequence[str] | None = None, *, limit: int = 20) -> CandidateFeed:
        quote_feed = self.quotes(symbols)
        if not quote_feed.quotes:
            return CandidateFeed(
                provider=quote_feed.provider,
                status=quote_feed.status,
                market_session=quote_feed.market_session,
                message=quote_feed.message,
                methodology=METHODOLOGY,
                refresh_seconds=quote_feed.refresh_seconds,
                universe_count=0,
                rankable_count=0,
                fetched_at=quote_feed.fetched_at,
            )

        valid = [quote for quote in quote_feed.quotes if quote.name and quote.close > 0 and quote.pre_close > 0]
        momentum = [quote.pct_change for quote in valid]
        positions = [_intraday_position(quote) for quote in valid]
        liquidity = [math.log1p(max(quote.amount, 0.0)) for quote in valid]
        momentum_scores = _percentile_scores(momentum)
        position_scores = _percentile_scores(positions)
        liquidity_scores = _percentile_scores(liquidity)
        candidates: list[ResearchCandidate] = []
        for quote, momentum_score, position_score, liquidity_score in zip(
            valid,
            momentum_scores,
            position_scores,
            liquidity_scores,
            strict=True,
        ):
            score = 100.0 * (0.45 * momentum_score + 0.30 * position_score + 0.25 * liquidity_score)
            risk_flags: list[str] = []
            if quote.amount < self._settings.candidate_min_amount_cny:
                risk_flags.append("成交额低于流动性门槛")
            if abs(quote.pct_change) >= 9.0:
                risk_flags.append("接近涨跌停，不宜追价")
            candidates.append(
                ResearchCandidate(
                    symbol=quote.symbol,
                    name=quote.name,
                    close=quote.close,
                    pct_change=quote.pct_change,
                    amount=quote.amount,
                    observed_at=quote.observed_at,
                    observation_score=round(score, 2),
                    rank=0,
                    factors={
                        "实时动量": round(momentum_score * 100.0, 2),
                        "日内位置": round(position_score * 100.0, 2),
                        "成交活跃": round(liquidity_score * 100.0, 2),
                    },
                    risk_flags=risk_flags,
                )
            )

        candidates.sort(
            key=lambda candidate: (bool(candidate.risk_flags), -candidate.observation_score, candidate.symbol)
        )
        ranked = [
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(candidates[:limit], start=1)
        ]
        return CandidateFeed(
            provider=quote_feed.provider,
            status=quote_feed.status,
            market_session=quote_feed.market_session,
            message=quote_feed.message,
            methodology=METHODOLOGY,
            refresh_seconds=quote_feed.refresh_seconds,
            universe_count=len(quote_feed.quotes),
            rankable_count=len(valid),
            fetched_at=quote_feed.fetched_at,
            candidates=ranked,
        )

    def metadata(self, feed: RealtimeQuoteFeed | CandidateFeed) -> VersionMetadata:
        generated_at = datetime.now(UTC)
        source_time = feed.fetched_at or self._localized_now()
        identity = f"{feed.provider}:{source_time.isoformat()}:{feed.status}"
        run_id = f"rt-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
        source_version = "eastmoney-clist" if feed.provider == "eastmoney" else "tushare-rt-k"
        return VersionMetadata(
            as_of_date=source_time.astimezone(SHANGHAI).date(),
            generated_at=generated_at,
            data_version=f"{source_version}-{source_time:%Y%m%dT%H%M%S}",
            feature_version="realtime-observation-v1",
            model_version="research-candidate-v1",
            run_id=run_id,
            is_stale=feed.status != "live",
        )

    def _quote_client(self) -> QuoteClient:
        if self._client is not None:
            return self._client
        if self._settings.market_data_provider.lower() == "eastmoney":
            self._client = EastmoneyDailyCrawler(
                permission_confirmed=self._settings.eastmoney_permission_confirmed,
                endpoint=self._settings.eastmoney_endpoint,
                page_size=self._settings.eastmoney_page_size,
                timeout_seconds=self._settings.eastmoney_request_timeout_seconds,
                request_delay_seconds=self._settings.eastmoney_request_delay_seconds,
            )
            return self._client
        token = self._settings.tushare_token
        if token is None:
            raise MarketDataError("Tushare Token 未配置")
        self._client = TushareRealtimeClient(
            token=token.get_secret_value(),
            api_url=self._settings.tushare_api_url,
            timeout_seconds=self._settings.realtime_request_timeout_seconds,
        )
        return self._client

    def _localized_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            return value.replace(tzinfo=SHANGHAI)
        return value.astimezone(SHANGHAI)

    def _empty_feed(
        self,
        now: datetime,
        status: FeedStatus,
        message: str,
    ) -> RealtimeQuoteFeed:
        return RealtimeQuoteFeed(
            provider=self._settings.market_data_provider.lower(),
            status=status,
            market_session=market_session(now),
            message=message,
            refresh_seconds=self._settings.realtime_cache_seconds,
        )

    def _feed_from_cache(
        self,
        cached: _CachedQuotes,
        now: datetime,
        status: FeedStatus,
        message: str,
    ) -> RealtimeQuoteFeed:
        return RealtimeQuoteFeed(
            provider=self._settings.market_data_provider.lower(),
            status=status,
            market_session=market_session(now),
            message=message,
            refresh_seconds=self._settings.realtime_cache_seconds,
            fetched_at=cached.fetched_at,
            quotes=[_to_schema(record) for record in cached.records],
        )


def _to_schema(record: RealtimeQuoteRecord) -> RealtimeQuote:
    return RealtimeQuote(
        symbol=record.symbol,
        name=record.name,
        pre_close=record.pre_close,
        open=record.open,
        high=record.high,
        low=record.low,
        close=record.close,
        pct_change=round(record.pct_change, 4),
        volume=record.volume,
        amount=record.amount,
        trade_count=record.trade_count,
        observed_at=record.observed_at,
    )


def _intraday_position(quote: RealtimeQuote) -> float:
    spread = quote.high - quote.low
    if spread <= 0:
        return 0.5
    return min(1.0, max(0.0, (quote.close - quote.low) / spread))


def _percentile_scores(values: Sequence[float]) -> list[float]:
    """Return tie-aware cross-sectional percentiles in O(n log n)."""
    if len(values) <= 1:
        return [0.5] * len(values)
    result = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        score = (start + (end - start - 1) / 2) / (len(values) - 1)
        for ordered_index in range(start, end):
            result[ordered[ordered_index][0]] = score
        start = end
    return result


def market_session(now: datetime) -> MarketSession:
    local = now.astimezone(SHANGHAI)
    if local.weekday() >= 5:
        return "closed"
    minute = local.hour * 60 + local.minute
    if 9 * 60 + 30 <= minute < 11 * 60 + 30 or 13 * 60 <= minute < 15 * 60:
        return "trading"
    if 11 * 60 + 30 <= minute < 13 * 60:
        return "lunch_break"
    return "closed"


@lru_cache
def get_realtime_market_service() -> RealtimeMarketService:
    return RealtimeMarketService(get_settings())
