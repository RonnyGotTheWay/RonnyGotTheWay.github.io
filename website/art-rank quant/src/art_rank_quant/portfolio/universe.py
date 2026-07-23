from __future__ import annotations

import polars as pl


def eligible_universe(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("is_investable")).with_columns(
        pl.when(pl.col("exclusion_reason").is_null())
        .then(pl.lit("eligible"))
        .otherwise(pl.col("exclusion_reason"))
        .alias("decision")
    )
