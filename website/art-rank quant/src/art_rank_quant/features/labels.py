from __future__ import annotations

import polars as pl


def forward_excess_labels(frame: pl.DataFrame, horizon: int = 5, bins: int = 5) -> pl.DataFrame:
    ordered = frame.sort(["symbol", "trade_date"])
    labeled = ordered.with_columns(
        (pl.col("close_forward_adjusted").shift(-horizon).over("symbol") / pl.col("close_forward_adjusted") - 1).alias(
            "forward_return"
        )
    ).with_columns(
        (pl.col("forward_return") - pl.col("forward_return").mean().over(["trade_date", "index_code"])).alias(
            "forward_excess_return"
        )
    )
    return labeled.with_columns(
        (
            (
                (pl.col("forward_excess_return").rank("ordinal").over("trade_date") - 1)
                * bins
                / pl.len().over("trade_date")
            )
            .floor()
            .clip(0, bins - 1)
            .cast(pl.Int8)
        ).alias("relevance")
    )
