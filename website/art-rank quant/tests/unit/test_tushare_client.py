from __future__ import annotations

import httpx
import pytest

from art_rank_quant.data.clients.tushare import TushareRealtimeClient, normalize_symbols


def test_tushare_rt_k_contract_is_parsed_without_exposing_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        assert '"api_name":"rt_k"' in body
        assert '"ts_code":"600000.SH,000001.SZ"' in body
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {
                    "fields": [
                        "ts_code", "name", "pre_close", "open", "high", "low", "close", "vol", "amount",
                        "num", "trade_time",
                    ],
                    "items": [
                        [
                            "600000.SH", "浦发银行", 10.0, 10.1, 10.4, 10.0, 10.3, 123_000, 1_260_000, 812,
                            "2026-07-14 10:31:00",
                        ],
                    ],
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = TushareRealtimeClient(token="secret-token", http_client=http_client)
    quotes = client.fetch_quotes(["600000.SH", "000001.SZ"])

    assert len(quotes) == 1
    assert quotes[0].symbol == "600000.SH"
    assert quotes[0].name == "浦发银行"
    assert quotes[0].pct_change == pytest.approx(3.0)
    assert quotes[0].observed_at.isoformat() == "2026-07-14T10:31:00+08:00"


def test_symbol_validation_rejects_web_scraping_style_urls() -> None:
    with pytest.raises(ValueError, match="Invalid A-share symbol"):
        normalize_symbols(["https://example.com/quote/600000"])


def test_official_full_market_wildcards_are_allowed() -> None:
    assert normalize_symbols(["3*.SZ", "6*.SH", "0*.SZ", "9*.BJ"]) == (
        "3*.SZ",
        "6*.SH",
        "0*.SZ",
        "9*.BJ",
    )
