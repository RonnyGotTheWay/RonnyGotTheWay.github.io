from __future__ import annotations

import numpy as np

from art_rank_quant.models.ranker.model import ArtRanker


def rank_stocks(model: ArtRanker, features: np.ndarray, symbols: list[str]) -> list[dict[str, object]]:
    scores = model.predict(features)
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return [
        {"symbol": symbol, "score": float(score), "rank": int(rank)}
        for symbol, score, rank in zip(symbols, scores, ranks, strict=True)
    ]
