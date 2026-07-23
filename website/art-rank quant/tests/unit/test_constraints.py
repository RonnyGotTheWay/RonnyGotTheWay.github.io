import numpy as np

from art_rank_quant.portfolio.constraints import PortfolioConstraints
from art_rank_quant.portfolio.risk_checks import check_constraints


def test_valid_equal_weight_portfolio_passes() -> None:
    weights = np.repeat(0.05, 20)
    sectors = np.array([f"s{i % 5}" for i in range(20)])
    beta = np.ones(20)
    assert check_constraints(weights, sectors, beta, PortfolioConstraints()) == []
