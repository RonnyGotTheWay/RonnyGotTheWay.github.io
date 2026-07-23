from __future__ import annotations

import polars as pl


def valuation_features(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"market_cap", "earnings"}
    if not required.issubset(frame.columns):
        return frame
    return frame.with_columns((pl.col("market_cap") / pl.col("earnings").clip(1e-6, None)).alias("pe_proxy"))
