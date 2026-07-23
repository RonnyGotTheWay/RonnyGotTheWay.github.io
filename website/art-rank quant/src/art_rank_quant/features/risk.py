from __future__ import annotations

import polars as pl


def risk_features(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("daily_return").rolling_std(60).over("symbol").alias("volatility_60d"),
        pl.col("close").log().alias("size_proxy"),
    )
