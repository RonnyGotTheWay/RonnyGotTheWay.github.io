from __future__ import annotations

import numpy as np


def top_contributions(
    feature_names: list[str], contributions: np.ndarray, limit: int = 5
) -> list[dict[str, float | str]]:
    indices = np.argsort(-np.abs(contributions))[:limit]
    return [{"feature": feature_names[index], "contribution": float(contributions[index])} for index in indices]
