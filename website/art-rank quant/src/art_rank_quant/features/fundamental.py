from __future__ import annotations

import polars as pl


def fundamental_features(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("roe") * (1 - pl.col("leverage"))).alias("quality_score"),
        pl.col("revenue_growth").clip(-1, 2).alias("growth_score"),
    )
