from __future__ import annotations

import polars as pl


def completeness(frame: pl.DataFrame, required: list[str]) -> dict[str, float]:
    return {column: 1 - frame[column].null_count() / max(1, frame.height) for column in required}
