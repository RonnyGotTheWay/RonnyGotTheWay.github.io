from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from art_rank_quant.common.config import get_settings
from art_rank_quant.data.clients.eastmoney_daily import EastmoneyDailyCrawler
from art_rank_quant.pipelines.backfill import backfill_synthetic


def daily_ingestion(as_of: date, root: Path) -> str:
    settings = get_settings()
    if settings.market_data_provider.lower() == "eastmoney":
        crawler = EastmoneyDailyCrawler(
            permission_confirmed=settings.eastmoney_permission_confirmed,
            endpoint=settings.eastmoney_endpoint,
            page_size=settings.eastmoney_page_size,
            timeout_seconds=settings.eastmoney_request_timeout_seconds,
            request_delay_seconds=settings.eastmoney_request_delay_seconds,
        )
        return crawler.crawl_to_storage(root, as_of).raw_manifest.version
    return backfill_synthetic(as_of - timedelta(days=90), as_of, root, symbols=60)
