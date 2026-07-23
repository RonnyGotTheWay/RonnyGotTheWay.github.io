from datetime import date

import polars as pl

from art_rank_quant.models.ensemble.dynamic_weights import lagged_dynamic_weights


def test_dynamic_weights_ignore_unrealized_future() -> None:
    history = pl.DataFrame(
        {
            "model_name": ["a", "b", "a"],
            "realized_at": [date(2024, 1, 1), date(2024, 1, 1), date(2025, 1, 1)],
            "rank_ic": [0.1, 0.05, 99.0],
        }
    )
    weights = lagged_dynamic_weights(history, date(2024, 6, 1), ["a", "b"])
    assert abs(sum(weights.values()) - 1) < 1e-9
    assert weights["a"] > weights["b"]
