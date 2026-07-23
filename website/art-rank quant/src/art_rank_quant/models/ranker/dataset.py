from __future__ import annotations

import numpy as np
import polars as pl


def ranker_dataset(frame: pl.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    ordered = frame.sort(["trade_date", "symbol"]).drop_nulls([*features, "relevance"])
    groups = ordered.group_by("trade_date", maintain_order=True).len()["len"].to_list()
    return ordered.select(features).to_numpy(), ordered["relevance"].to_numpy(), groups
