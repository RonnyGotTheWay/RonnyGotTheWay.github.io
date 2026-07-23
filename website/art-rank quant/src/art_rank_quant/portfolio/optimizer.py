# mypy: disable-error-code="attr-defined,no-untyped-call,list-item"
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from art_rank_quant.portfolio.constraints import PortfolioConstraints


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    weights: np.ndarray
    status: str
    objective: float


def optimize_portfolio(
    alpha: np.ndarray,
    covariance: np.ndarray,
    previous: np.ndarray,
    sectors: np.ndarray,
    beta: np.ndarray,
    constraints: PortfolioConstraints | None = None,
    risk_aversion: float = 3.0,
    turnover_penalty: float = 0.5,
) -> OptimizationResult:
    constraints = constraints or PortfolioConstraints()
    count = len(alpha)
    if covariance.shape != (count, count) or len(previous) != count:
        raise ValueError("Portfolio inputs have inconsistent dimensions")
    weights = cp.Variable(count)
    objective = cp.Maximize(
        alpha @ weights
        - risk_aversion * cp.quad_form(weights, cp.psd_wrap(covariance))
        - turnover_penalty * cp.norm1(weights - previous)
    )
    rules: list[cp.Constraint] = [cp.sum(weights) == 1, weights <= constraints.max_stock_weight]
    if constraints.long_only:
        rules.append(weights >= 0)
    rules.extend([cp.norm1(weights - previous) <= constraints.max_turnover, beta @ weights <= constraints.max_beta])
    for sector in np.unique(sectors):
        rules.append(cp.sum(weights[sectors == sector]) <= constraints.max_sector_weight)
    problem = cp.Problem(objective, rules)
    value = problem.solve(solver=cp.CLARABEL)
    if weights.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise ValueError(f"Portfolio infeasible: {problem.status}")
    return OptimizationResult(np.asarray(weights.value), str(problem.status), float(value))
