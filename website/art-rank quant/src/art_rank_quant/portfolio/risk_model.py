from __future__ import annotations

import numpy as np


def shrink_covariance(returns: np.ndarray, shrinkage: float = 0.2) -> np.ndarray:
    sample = np.cov(returns, rowvar=False)
    diagonal = np.diag(np.diag(sample))
    covariance = (1 - shrinkage) * sample + shrinkage * diagonal
    return covariance + np.eye(covariance.shape[0]) * 1e-8
