from __future__ import annotations

import polars as pl

from art_rank_quant.features.labels import forward_excess_labels
from art_rank_quant.features.liquidity import liquidity_features
from art_rank_quant.features.price_volume import price_volume_features
from art_rank_quant.features.risk import risk_features


def materialize_daily_features(frame: pl.DataFrame, include_labels: bool = False) -> pl.DataFrame:
    result = risk_features(liquidity_features(price_volume_features(frame)))
    return forward_excess_labels(result) if include_labels else result
