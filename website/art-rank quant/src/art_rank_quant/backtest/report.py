from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BacktestReport:
    run_id: str
    metrics: dict[str, float]
    data_version: str
    model_version: str
    out_of_sample_only: bool = True
