from datetime import date

from art_rank_quant.data.clients.synthetic import SyntheticAshareClient


def test_synthetic_data_is_deterministic() -> None:
    client = SyntheticAshareClient(seed=7, symbol_count=10)
    first = client.fetch(date(2023, 1, 1), date(2023, 6, 30))["stock_daily"].frame
    second = client.fetch(date(2023, 1, 1), date(2023, 6, 30))["stock_daily"].frame
    assert first.equals(second)
    assert first.height > 500
