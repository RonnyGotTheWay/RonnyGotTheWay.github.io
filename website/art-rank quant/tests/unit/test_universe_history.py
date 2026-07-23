from datetime import date

import polars as pl

from art_rank_quant.data.processing.universe_history import universe_on


def test_historical_membership_respects_effective_dates() -> None:
    members = pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "index_code": ["I", "I"],
            "weight": [0.5, 0.5],
            "effective_from": [date(2020, 1, 1), date(2021, 1, 1)],
            "effective_to": [date(2020, 12, 31), date(2022, 1, 1)],
        }
    )
    assert universe_on(members, date(2020, 6, 1))["symbol"].to_list() == ["A"]
