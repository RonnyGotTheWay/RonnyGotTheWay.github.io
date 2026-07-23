from __future__ import annotations


def validation_gate(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if metrics.get("rank_ic", 0) <= 0:
        failures.append("non_positive_rank_ic")
    if metrics.get("constraint_violations", 1) != 0:
        failures.append("constraint_violations")
    if metrics.get("pit_failures", 1) != 0:
        failures.append("pit_failures")
    return not failures, failures
