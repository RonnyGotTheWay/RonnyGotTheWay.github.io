from datetime import date, timedelta

import polars as pl

from art_rank_quant.features.labels import forward_excess_labels


def test_labels_are_integer_relevance_grades() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    rows = [
        {"trade_date": d, "symbol": s, "index_code": "I", "close_forward_adjusted": 10 + i * (1 + j * 0.1)}
        for j, s in enumerate(["A", "B", "C", "D", "E"])
        for i, d in enumerate(dates)
    ]
    result = forward_excess_labels(pl.DataFrame(rows), horizon=2).drop_nulls("relevance")
    assert result["relevance"].dtype == pl.Int8
    assert set(result["relevance"].unique().to_list()) <= set(range(5))
