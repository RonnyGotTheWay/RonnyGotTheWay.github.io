import numpy as np

from art_rank_quant.backtest.metrics import performance_metrics


def test_backtest_metric_snapshot() -> None:
    metrics = performance_metrics(np.array([1.0, 1.01, 1.005, 1.02]))
    assert round(metrics["total_return"], 4) == 0.02
    assert round(metrics["max_drawdown"], 4) == -0.005
