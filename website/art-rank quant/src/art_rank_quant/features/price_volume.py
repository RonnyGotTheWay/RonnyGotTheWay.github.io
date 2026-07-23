from __future__ import annotations

import polars as pl


def price_volume_features(frame: pl.DataFrame) -> pl.DataFrame:
    ordered = frame.sort(["symbol", "trade_date"])
    return ordered.with_columns(
        pl.col("close_forward_adjusted").pct_change().over("symbol").alias("daily_return"),
        (pl.col("close_forward_adjusted") / pl.col("close_forward_adjusted").shift(5).over("symbol") - 1).alias(
            "momentum_5d"
        ),
        (pl.col("close_forward_adjusted") / pl.col("close_forward_adjusted").shift(20).over("symbol") - 1).alias(
            "momentum_20d"
        ),
        pl.col("amount").log1p().alias("log_amount"),
    ).with_columns(
        pl.col("daily_return").rolling_std(20).over("symbol").alias("volatility_20d"),
        (pl.col("amount") / pl.col("amount").rolling_mean(20).over("symbol")).alias("turnover_proxy"),
    )
