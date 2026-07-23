from __future__ import annotations

import polars as pl


def liquidity_features(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns((pl.col("daily_return").abs() / pl.col("amount").clip(1.0, None) * 1e8).alias("amihud"))
