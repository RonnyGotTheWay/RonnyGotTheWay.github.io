from __future__ import annotations


def additive_attribution(total_return: float, contributions: dict[str, float]) -> dict[str, float]:
    residual = total_return - sum(contributions.values())
    return contributions | {"residual": residual}
