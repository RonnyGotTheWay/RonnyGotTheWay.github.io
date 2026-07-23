from __future__ import annotations

import numpy as np
import polars as pl


def regime_matrix(market: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return market.sort("trade_date").select(columns).drop_nulls().to_numpy().astype(np.float64)
