from __future__ import annotations

import numpy as np

from art_rank_quant.models.hmm.model import RegimeHMM


def select_hmm(features: np.ndarray, candidates: list[int], seed: int = 474) -> tuple[RegimeHMM, dict[int, float]]:
    scores: dict[int, float] = {}
    models: dict[int, RegimeHMM] = {}
    n, d = features.shape
    for states in candidates:
        model = RegimeHMM(states, seed).fit(features)
        parameters = states * states + 2 * states * d
        scores[states] = -2 * model.model.score(features) + parameters * np.log(n)
        models[states] = model
    best = min(scores, key=lambda candidate: scores[candidate])
    return models[best], scores
