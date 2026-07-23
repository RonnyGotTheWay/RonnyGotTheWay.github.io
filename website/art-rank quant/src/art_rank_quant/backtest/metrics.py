from __future__ import annotations

import numpy as np


def performance_metrics(equity: np.ndarray, periods_per_year: int = 252) -> dict[str, float]:
    if equity.size < 2:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    returns = equity[1:] / equity[:-1] - 1
    std = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1
    return {
        "total_return": float(equity[-1] / equity[0] - 1),
        "sharpe": float(returns.mean() / std * np.sqrt(periods_per_year)) if std else 0.0,
        "max_drawdown": float(drawdown.min()),
    }
