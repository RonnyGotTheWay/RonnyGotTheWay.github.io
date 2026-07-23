from __future__ import annotations

import polars as pl


def cross_sectional_zscore(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for name in columns:
        median = pl.col(name).median().over("trade_date")
        mad = (pl.col(name) - median).abs().median().over("trade_date")
        clipped = pl.col(name).clip(median - 5 * mad, median + 5 * mad)
        expressions.append(
            ((clipped - clipped.mean().over("trade_date")) / clipped.std().over("trade_date").clip(1e-9, None)).alias(
                f"{name}_z"
            )
        )
    return frame.with_columns(expressions)
