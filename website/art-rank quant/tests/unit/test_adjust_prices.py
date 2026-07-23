import polars as pl

from art_rank_quant.data.processing.adjust_prices import add_adjusted_prices


def test_adjusted_prices_retain_raw_values() -> None:
    result = add_adjusted_prices(
        pl.DataFrame({"open": [10.0], "high": [12.0], "low": [9.0], "close": [11.0], "adjust_factor": [2.0]})
    )
    assert result["close"][0] == 11.0
    assert result["close_forward_adjusted"][0] == 5.5
