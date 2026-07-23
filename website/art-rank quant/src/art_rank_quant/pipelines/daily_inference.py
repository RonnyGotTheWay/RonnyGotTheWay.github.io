from __future__ import annotations

import polars as pl


def baseline_inference(features: pl.DataFrame) -> pl.DataFrame:
    latest = features.filter(pl.col("trade_date") == pl.col("trade_date").max())
    return latest.with_columns(
        (
            pl.col("momentum_20d").fill_null(0) * 0.45
            - pl.col("volatility_20d").fill_null(0) * 0.25
            + pl.col("turnover_proxy").fill_null(1) * 0.10
        ).alias("art_score")
    ).with_columns(pl.col("art_score").rank("ordinal", descending=True).alias("rank"))
