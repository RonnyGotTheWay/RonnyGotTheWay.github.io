from datetime import date

import polars as pl

from art_rank_quant.data.validation.quality_checks import assert_pit


def test_future_available_record_is_rejected() -> None:
    frame = pl.DataFrame({"available_time": [date(2024, 5, 1)], "as_of_date": [date(2024, 4, 30)]})
    try:
        assert_pit(frame)
    except ValueError as error:
        assert "PIT violation" in str(error)
    else:
        raise AssertionError("future record was accepted")
