from __future__ import annotations


def promotion_decision(
    champion_metric: float, challenger_metric: float, minimum_improvement: float = 0.005
) -> dict[str, object]:
    eligible = challenger_metric >= champion_metric + minimum_improvement
    return {"eligible": eligible, "requires_manual_approval": True, "automatic_promotion": False}
