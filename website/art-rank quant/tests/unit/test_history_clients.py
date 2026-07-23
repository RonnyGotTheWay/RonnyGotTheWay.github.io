from datetime import date

import httpx
import pandas as pd

from art_rank_quant.data.clients.akshare_sina_history import AkshareSinaHistoryClient
from art_rank_quant.data.clients.baostock_history import BaoStockHistoryClient
from art_rank_quant.data.clients.eastmoney_history import EastmoneyHistoryClient
from art_rank_quant.data.clients.tushare import TushareDailyClient


class BaoResult:
    error_code = "0"
    error_msg = "success"

    def __init__(self, rows: list[list[str]]) -> None:
        self.rows = iter(rows)
        self.current: list[str] = []

    def next(self) -> bool:
        try:
            self.current = next(self.rows)
            return True
        except StopIteration:
            return False

    def get_row_data(self) -> list[str]:
        return self.current


def test_eastmoney_history_maps_raw_daily_rows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["klt"] == "101"
        assert request.url.params["fqt"] == "0"
        return httpx.Response(
            200,
            json={"data": {"klines": ["2026-07-14,10,10.2,10.3,9.9,1000,10000,4,2,0.2,1.5"]}},
        )

    client = EastmoneyHistoryClient(
        permission_confirmed=True,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    frame = client.fetch_daily("600000.SH", date(2026, 7, 1), date(2026, 7, 14))
    assert frame.height == 1
    assert frame["symbol"][0] == "600000.SH"
    assert frame["close"][0] == 10.2


def test_eastmoney_history_maps_30_minute_rows_and_daily_check() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["klt"] == "30":
            return httpx.Response(
                200,
                json={"data": {"klines": ["2026-07-14 10:00,10,10.2,10.3,9.9,1000,10000,4,2,0.2,1.5"]}},
            )
        return httpx.Response(
            200,
            json={"data": {"klines": ["2026-07-14,10,10.2,10.3,9.9,1000,10000,4,2,0.2,1.5"]}},
        )

    client = EastmoneyHistoryClient(
        permission_confirmed=True,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    intraday = client.fetch_intraday("600000.SH", date(2026, 7, 1), date(2026, 7, 14))
    check = client.fetch_daily_check("600000.SH", date(2026, 7, 1), date(2026, 7, 14))

    assert intraday.height == 1
    assert intraday["event_time"][0].hour == 10
    assert intraday["source"][0] == "eastmoney"
    assert check["close_baostock"][0] == 10.2
    assert check["check_source"][0] == "eastmoney"


def test_baostock_intraday_uses_anonymous_session(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    success = BaoResult([])
    monkeypatch.setattr("art_rank_quant.data.clients.baostock_history.bs.login", lambda: success)
    monkeypatch.setattr("art_rank_quant.data.clients.baostock_history.bs.logout", lambda: success)
    monkeypatch.setattr(
        "art_rank_quant.data.clients.baostock_history.bs.query_history_k_data_plus",
        lambda *args, **kwargs: BaoResult(
            [["2026-07-14", "20260714100000000", "sh.600000", "10", "10.2", "9.9", "10.1", "1000", "10000", "3"]]
        ),
    )
    with BaoStockHistoryClient() as client:
        frame = client.fetch_intraday("600000.SH", date(2026, 7, 1), date(2026, 7, 14))
    assert frame.height == 1
    assert frame["event_time"][0].hour == 10
    assert frame["source"][0] == "baostock"


def test_baostock_daily_check_maps_status_and_st_flag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    success = BaoResult([])
    monkeypatch.setattr("art_rank_quant.data.clients.baostock_history.bs.login", lambda: success)
    monkeypatch.setattr("art_rank_quant.data.clients.baostock_history.bs.logout", lambda: success)
    monkeypatch.setattr(
        "art_rank_quant.data.clients.baostock_history.bs.query_history_k_data_plus",
        lambda *args, **kwargs: BaoResult(
            [["2026-07-14", "sh.600000", "10", "10.3", "9.9", "10.2", "10", "1000", "10000", "0.2", "1", "2", "0"]]
        ),
    )
    with BaoStockHistoryClient() as client:
        frame = client.fetch_daily_check("600000.SH", date(2026, 7, 1), date(2026, 7, 14))
    assert frame["trade_status_baostock"][0] == "trading"
    assert frame["is_st"][0] is False


def test_baostock_normalized_daily_includes_adjusted_close(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    success = BaoResult([])
    monkeypatch.setattr("art_rank_quant.data.clients.baostock_history.bs.login", lambda: success)
    monkeypatch.setattr("art_rank_quant.data.clients.baostock_history.bs.logout", lambda: success)

    def fetch(*args, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs["adjustflag"] == "2":
            return BaoResult([["2026-07-14", "sh.600000", "10.1"]])
        return BaoResult(
            [["2026-07-14", "sh.600000", "10", "10.3", "9.9", "10.2", "10", "1000", "10000", "0.2", "1", "2", "0"]]
        )

    monkeypatch.setattr("art_rank_quant.data.clients.baostock_history.bs.query_history_k_data_plus", fetch)
    with BaoStockHistoryClient() as client:
        frame = client.fetch_daily("600000.SH", date(2026, 7, 1), date(2026, 7, 14))
    assert frame["close"][0] == 10.2
    assert frame["adjusted_close"][0] == 10.1
    assert frame["trade_status"][0] == "trading"


def test_tushare_daily_check_maps_rows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        assert '"api_name":"daily"' in body
        return httpx.Response(
            200,
            json={"code": 0, "data": {"fields": ["ts_code", "trade_date", "close", "vol"],
                "items": [["600000.SH", "20260714", 10.2, 1000]]}},
        )

    client = TushareDailyClient("token", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    frame = client.fetch_daily_check("600000.SH", date(2026, 7, 1), date(2026, 7, 14))
    assert frame.height == 1
    assert frame["close_baostock"][0] == 10.2
    assert frame["check_source"][0] == "tushare"


def test_akshare_sina_maps_intraday_and_daily_check() -> None:
    def minute_fetcher(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {"symbol": "sh600000", "period": "30", "adjust": ""}
        return pd.DataFrame(
            [
                {
                    "day": "2026-07-14 10:00:00",
                    "open": 10,
                    "high": 10.3,
                    "low": 9.9,
                    "close": 10.2,
                    "volume": 1000,
                    "amount": 10000,
                }
            ]
        )

    def daily_fetcher(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["symbol"] == "sh600000"
        assert kwargs["end_date"] == "20260715"
        return pd.DataFrame([{"date": "2026-07-14", "close": 10.2, "volume": 1000}])

    client = AkshareSinaHistoryClient(
        delay_seconds=0.0,
        minute_fetcher=minute_fetcher,
        daily_fetcher=daily_fetcher,
    )
    intraday = client.fetch_intraday("600000.SH", date(2026, 7, 1), date(2026, 7, 14))
    daily = client.fetch_daily_check("600000.SH", date(2026, 7, 1), date(2026, 7, 14))

    assert intraday.height == 1
    assert intraday["event_time"][0].hour == 10
    assert intraday["source"][0] == "akshare_sina"
    assert daily["close_baostock"][0] == 10.2
    assert daily["trade_status_baostock"][0] == "trading"
    assert daily["check_source"][0] == "akshare_sina"


def test_akshare_sina_normalized_daily_includes_adjusted_close() -> None:
    def daily_fetcher(**kwargs):  # type: ignore[no-untyped-def]
        close = 10.1 if kwargs["adjust"] == "qfq" else 10.2
        return pd.DataFrame([{
            "date": "2026-07-14", "open": 10, "high": 10.3, "low": 9.9,
            "close": close, "volume": 1000, "amount": 10000,
        }])

    client = AkshareSinaHistoryClient(delay_seconds=0.0, daily_fetcher=daily_fetcher)
    frame = client.fetch_daily("600000.SH", date(2026, 7, 1), date(2026, 7, 14))
    assert frame["close"][0] == 10.2
    assert frame["adjusted_close"][0] == 10.1
    assert frame["source"][0] == "akshare_sina"


def test_akshare_sina_retries_transient_errors() -> None:
    attempts = 0

    def minute_fetcher(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary")
        return pd.DataFrame(
            [
                {
                    "day": "2026-07-14 10:00:00",
                    "open": 10,
                    "high": 10.3,
                    "low": 9.9,
                    "close": 10.2,
                    "volume": 1000,
                    "amount": 10000,
                }
            ]
        )

    client = AkshareSinaHistoryClient(delay_seconds=0.0, minute_fetcher=minute_fetcher)
    frame = client.fetch_intraday("600000.SH", date(2026, 7, 1), date(2026, 7, 14))

    assert frame.height == 1
    assert attempts == 3
