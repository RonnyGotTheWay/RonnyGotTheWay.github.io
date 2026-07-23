from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl


def lagged_dynamic_weights(
    history: pl.DataFrame,
    as_of_date: date,
    model_names: list[str],
    window: int = 60,
    minimum: float = 0.1,
    maximum: float = 0.7,
) -> dict[str, float]:
    eligible = history.filter((pl.col("realized_at") < as_of_date) & pl.col("model_name").is_in(model_names))
    means = (
        eligible.sort("realized_at")
        .group_by("model_name")
        .tail(window)
        .group_by("model_name")
        .agg(pl.col("rank_ic").mean())
    )
    raw = {name: max(0.0, float(value)) for name, value in means.iter_rows()}
    values = np.array([raw.get(name, 0.0) for name in model_names], dtype=float)
    values = np.ones(len(model_names)) if values.sum() == 0 else values
    values /= values.sum()
    values = np.clip(values, minimum, maximum)
    values /= values.sum()
    return dict(zip(model_names, values.tolist(), strict=True))
