from __future__ import annotations

import numpy as np

from art_rank_quant.portfolio.constraints import PortfolioConstraints


def check_constraints(
    weights: np.ndarray, sectors: np.ndarray, beta: np.ndarray, limits: PortfolioConstraints
) -> list[str]:
    errors: list[str] = []
    if abs(weights.sum() - 1) > 1e-5:
        errors.append("weights_not_fully_invested")
    if weights.max(initial=0) > limits.max_stock_weight + 1e-6:
        errors.append("stock_limit")
    if limits.long_only and weights.min(initial=0) < -1e-8:
        errors.append("short_position")
    if float(beta @ weights) > limits.max_beta + 1e-6:
        errors.append("beta_limit")
    for sector in np.unique(sectors):
        if weights[sectors == sector].sum() > limits.max_sector_weight + 1e-6:
            errors.append(f"sector_limit:{sector}")
    return errors
