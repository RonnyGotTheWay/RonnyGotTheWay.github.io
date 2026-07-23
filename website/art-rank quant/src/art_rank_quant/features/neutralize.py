from __future__ import annotations

import numpy as np
import polars as pl


def neutralize_cross_section(frame: pl.DataFrame, feature: str, controls: list[str]) -> pl.DataFrame:
    def residuals(group: pl.DataFrame) -> pl.DataFrame:
        usable = group.drop_nulls([feature, *controls])
        if usable.height <= len(controls) + 1:
            return group.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{feature}_neutral"))
        x = usable.select(controls).to_numpy()
        x = np.column_stack([np.ones(x.shape[0]), x])
        y = usable[feature].to_numpy()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        values = y - x @ beta
        mapping = usable.select("symbol").with_columns(pl.Series(f"{feature}_neutral", values))
        return group.join(mapping, on="symbol", how="left")

    return frame.group_by("trade_date", maintain_order=True).map_groups(residuals)
