from __future__ import annotations

import polars as pl


def equal_weight_top_ranked(scores: pl.DataFrame, holdings: int = 20) -> pl.DataFrame:
    selected = scores.sort("rank").head(holdings)
    return selected.select("trade_date", "symbol", "sector", "rank", "art_score").with_columns(
        pl.lit(1 / max(1, selected.height)).alias("target_weight")
    )
