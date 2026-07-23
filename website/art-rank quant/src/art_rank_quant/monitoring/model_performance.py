from __future__ import annotations

import numpy as np

from art_rank_quant.models.ranker.train import rank_ic


def rolling_rank_ic(scores: np.ndarray, realized: np.ndarray) -> float:
    return rank_ic(scores, realized)
