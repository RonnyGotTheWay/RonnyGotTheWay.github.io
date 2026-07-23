from datetime import date

from art_rank_quant.data.clients.synthetic import SyntheticAshareClient
from art_rank_quant.data.validation.quality_checks import validate_stock_daily


def test_generated_market_passes_quality_gate() -> None:
    frame = SyntheticAshareClient(symbol_count=8).fetch(date(2024, 1, 1), date(2024, 6, 30))["stock_daily"].frame
    assert validate_stock_daily(frame).passed
