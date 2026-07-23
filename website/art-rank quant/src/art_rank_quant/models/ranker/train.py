from __future__ import annotations

import numpy as np


def rank_ic(scores: np.ndarray, realized: np.ndarray) -> float:
    score_rank = np.argsort(np.argsort(scores))
    realized_rank = np.argsort(np.argsort(realized))
    return float(np.corrcoef(score_rank, realized_rank)[0, 1])
