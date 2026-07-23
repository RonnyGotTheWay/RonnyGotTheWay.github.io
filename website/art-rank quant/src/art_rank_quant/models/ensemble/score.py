from __future__ import annotations

import numpy as np

from art_rank_quant.models.ensemble.calibration import percentile_calibrate


def ensemble_scores(scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    if set(scores) != set(weights):
        raise ValueError("Scores and weights must contain the same comparable models")
    lengths = {len(value) for value in scores.values()}
    if len(lengths) != 1:
        raise ValueError("All model scores must cover the same universe")
    result = np.zeros(next(iter(lengths)), dtype=float)
    for name, values in scores.items():
        result += weights[name] * percentile_calibrate(values)
    return result
