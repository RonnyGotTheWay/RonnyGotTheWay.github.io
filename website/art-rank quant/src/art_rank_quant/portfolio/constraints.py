from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    max_stock_weight: float = 0.05
    max_sector_weight: float = 0.25
    max_turnover: float = 0.40
    max_beta: float = 1.10
    long_only: bool = True
