from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from math import sin


@lru_cache
def rankings() -> list[dict[str, object]]:
    sectors = ["科技", "消费", "工业", "医药", "金融", "材料"]
    indices = ["SYN300", "SYN500", "SYN1000"]
    return [
        {
            "symbol": f"SYN{i:04d}",
            "name": f"模拟证券 {i:03d}",
            "index_code": indices[i % 3],
            "sector": sectors[i % 6],
            "score": round(100 - i * 0.73, 2),
            "rank": i,
            "risk_flags": ["流动性观察"] if i % 11 == 0 else [],
        }
        for i in range(1, 81)
    ]


def equity_curve() -> list[dict[str, object]]:
    start = date.today() - timedelta(days=119)
    strategy = benchmark = 1.0
    points: list[dict[str, object]] = []
    for i in range(120):
        benchmark *= 1 + 0.00035 + sin(i / 8) * 0.0015
        strategy *= 1 + 0.00055 + sin(i / 8) * 0.0019 + sin(i / 19) * 0.0006
        points.append(
            {"date": str(start + timedelta(days=i)), "strategy": round(strategy, 5), "benchmark": round(benchmark, 5)}
        )
    return points
