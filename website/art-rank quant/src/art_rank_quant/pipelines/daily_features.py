from __future__ import annotations

from pathlib import Path

import polars as pl

from art_rank_quant.features.materialize import materialize_daily_features


def daily_features(market_path: Path, output_path: Path) -> Path:
    frame = pl.read_parquet(market_path)
    features = materialize_daily_features(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(output_path)
    return output_path
