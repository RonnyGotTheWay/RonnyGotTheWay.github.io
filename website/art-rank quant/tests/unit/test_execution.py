from datetime import date

from art_rank_quant.backtest.execution import MarketState, Order, executable


def test_same_day_execution_is_rejected() -> None:
    today = date(2024, 1, 2)
    ok, reason = executable(Order("A", today, "buy", 100), MarketState(today, 10, 10000))
    assert not ok and reason == "next_session_only"
